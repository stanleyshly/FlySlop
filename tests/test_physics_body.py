import unittest

import numpy as np


def _need(test, *mods):
    for mod in mods:
        try:
            __import__(mod)
        except ImportError:
            test.skipTest(f"{mod} not installed")


class PhysicsKeyboardTests(unittest.TestCase):
    def test_mujoco_matches_vendored_forward_kinematics(self):
        _need(self, "mujoco")
        from backend.embodiment import MM, Planner
        from backend.flybody import euler_zyx
        from backend.physics import LEGS, FlySim
        sim = FlySim()
        body = Planner().body.copy()
        body[2] += 1.0   # clear of the keyboard
        q = np.array([sim.fly.rest_q[leg] for leg in LEGS]) + np.random.default_rng(0).normal(0, 0.2, (6, 7))
        sim.reset(body, q)
        rot = euler_zyx(*body[3:])
        fk = np.array([sim.fly.tip(leg, q[i]) for i, leg in enumerate(LEGS)]) * MM @ rot.T + body[:3]
        self.assertLess(np.abs(fk - sim.tips()).max(), 1e-5)

    def test_standing_does_not_type(self):
        _need(self, "mujoco")
        from backend.embodiment import Planner
        from backend.physics import FlySim, physical_targets
        sim = FlySim()
        planner = Planner()
        goals = physical_targets(planner.feet, sim.params.tip_radius)
        q, _ = sim.solve_targets(planner.body, goals)
        sim.reset(planner.body, q)
        events = [e for _ in range(60) for e in sim.step(q)]
        self.assertEqual([e for e in events if e["type"] == "key"], [])

    def test_scripted_physics_episode_types_and_replays(self):
        _need(self, "mujoco")
        from backend.embodiment import replay_text
        from backend.physics import physics_episode
        result = physics_episode("t", "aA ;")
        self.assertEqual(result["final_text"], "aA ;")
        self.assertEqual(replay_text(result["events"]), "aA ;")
        keys = [e for e in result["events"] if e["type"] == "key"]
        self.assertTrue(all(e["foot"] in {"LF", "RF"} for e in keys))

    def test_deck_has_holes_under_keys(self):
        _need(self, "mujoco")
        from backend.keyboard import LAYOUT
        from backend.physics import deck_rectangles
        for x0, y0, x1, y1 in deck_rectangles():
            for key in LAYOUT:
                overlap_x = min(x1, key.x + key.width) - max(x0, key.x)
                overlap_y = min(y1, key.y + key.height) - max(y0, key.y)
                self.assertFalse(overlap_x > 1e-6 and overlap_y > 1e-6, key.id)


class FlyEnvTests(unittest.TestCase):
    def test_env_checker_and_expert(self):
        _need(self, "mujoco", "gymnasium")
        from gymnasium.utils.env_checker import check_env
        from backend.embodiment import replay_text
        from backend.fly_env import FlyTypingEnv
        env = FlyTypingEnv(targets=("fly",))
        check_env(env, skip_render_check=True)
        env.reset(seed=1, options={"target": "fly"})
        done = False
        while not done:
            _o, _r, terminated, truncated, _i = env.step(env.expert_action())
            done = terminated or truncated
        self.assertEqual(env.typed, "fly")
        self.assertEqual(replay_text(env.events), "fly")

    def test_hovering_policy_types_nothing(self):
        _need(self, "mujoco", "gymnasium")
        from backend.fly_env import FlyTypingEnv
        env = FlyTypingEnv(targets=("a",))
        env.reset(seed=0, options={"target": "a"})
        for _ in range(60):
            env.step(np.array([0, 0, 0, 0, 1.0, 0, 0, 1.0], dtype=np.float32))
        self.assertEqual(env.typed, "")


class ConnectomeTests(unittest.TestCase):
    def test_circuit_and_variants(self):
        from backend.connectome import group_indices, load_circuit, signed_matrix, summary, variant_matrix
        circuit = load_circuit()
        info = summary(circuit)
        self.assertEqual(info["neurons"]["MN"], 40)
        self.assertEqual(circuit["source"]["license"], "CC-BY")
        w = signed_matrix(circuit)
        idx = group_indices(circuit)
        shuffled = variant_matrix(circuit, "shuffled", 0)
        self.assertEqual(np.count_nonzero(shuffled), np.count_nonzero(w))
        self.assertAlmostEqual(float(np.abs(shuffled).sum()), float(np.abs(w).sum()), places=2)
        self.assertFalse(np.array_equal(shuffled != 0, w != 0))
        sparse = variant_matrix(circuit, "random_sparse", 0)
        self.assertEqual(np.count_nonzero(sparse), np.count_nonzero(w))

    def test_connectome_actor_respects_mask(self):
        _need(self, "torch", "mujoco", "gymnasium", "stable_baselines3")
        import torch
        from backend.fly_env import FlyTypingEnv
        from training.connectome_policy import PROPRIO, ConnectomeActor
        obs_dim = len(FlyTypingEnv.OBS_LABELS)
        actor = ConnectomeActor(obs_dim, mode="connectome")
        weight = actor.effective_weight()
        self.assertTrue(torch.all(weight[actor.mask == 0] == 0))
        self.assertTrue(torch.all(torch.sign(weight[actor.mask == 1]) == actor.sign[actor.mask == 1]))
        silenced = ConnectomeActor(obs_dim, silence=("proprio",))
        self.assertEqual(int((silenced.keep == 0).sum()), len(PROPRIO))


class FlyPipelineSmokeTests(unittest.TestCase):
    def test_imitation_then_ppo_smoke(self):
        _need(self, "torch", "mujoco", "gymnasium", "stable_baselines3")
        import json, subprocess, sys, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            bc = Path(tmp) / "bc"
            subprocess.run([sys.executable, "-m", "training.imitate", "--config", "training/ppo_fly.json",
                            "--episodes", "8", "--epochs", "2", "--eval-episodes", "2", "--workers", "2",
                            "--out", str(bc)], check=True, capture_output=True)
            self.assertTrue((bc / "bc_model.zip").exists())
            subprocess.run([sys.executable, "-m", "training.train_ppo", "--config", "training/ppo_fly.json",
                            "--smoke", "--timesteps", "1024", "--init-from", str(bc / "bc_model.zip"),
                            "--name", "p", "--out", tmp], check=True, capture_output=True)
            manifest = json.loads((Path(tmp) / "p" / "manifest.json").read_text())
            self.assertEqual(manifest["init_from"], str(bc / "bc_model.zip"))
            self.assertTrue((Path(tmp) / "p" / "final_model.zip").exists())


if __name__ == "__main__":
    unittest.main()
