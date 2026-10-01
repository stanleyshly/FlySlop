"""P3: MN-driven legs env (action_mode="mn"), the mn_antagonist readout, and legacy-mode parity."""

import hashlib
import json
import unittest

import numpy as np

try:
    import torch  # noqa: F401
    import stable_baselines3  # noqa: F401
    HAVE_TRAINING = True
except ImportError:  # optional extra
    HAVE_TRAINING = False

from backend.fly_env import JOINT_NAMES, JOINT_SCALE, FlyTypingEnv, obs_labels

# Fingerprints recorded from the pre-P3 env (claw mode, terminate_on_error False, noisy expert): observations,
# rewards, key events and final joint targets must stay identical.
GOLDEN = {(3, "hi"): ("2e3b670ba1d45e41fc96213da4bfcd76c5b4af2c026d0cd4602be147b3de2987", 25, "hi", 5),
          (5, "Ab"): ("699c6478a3f3de787b6b0db41b4e08754ae0bc8d4c3ce1605256a080103ec897", 50, "Ab", 8)}


def fingerprint(seed, target, steps=400):
    env = FlyTypingEnv(targets=(target,), terminate_on_error=False)
    obs, _ = env.reset(seed=seed, options={"target": target})
    h = hashlib.sha256()
    h.update(obs.tobytes())
    rng = np.random.default_rng(seed)
    n = 0
    for i in range(steps):
        a = env.expert_action()
        if i % 7 == 0:
            a = np.clip(a + rng.normal(0, .3, a.shape), -1, 1).astype(np.float32)
        obs, r, te, tr, _ = env.step(a)
        h.update(obs.tobytes())
        h.update(np.float64(r).tobytes())
        n += 1
        if te or tr:
            break
    events = [(e["tick"], e["type"], e["key_id"], e.get("char")) for e in env.events]
    h.update(json.dumps(events).encode())
    h.update(env.sim.joint_q().tobytes())
    return h.hexdigest(), n, env.typed, len(events)


class LegacyParity(unittest.TestCase):
    def test_claw_mode_is_unchanged(self):
        for (seed, target), expected in GOLDEN.items():
            self.assertEqual(fingerprint(seed, target), expected)

    def test_claw_defaults(self):
        env = FlyTypingEnv(targets=("a",))
        self.assertEqual(env.action_mode, "claw")
        self.assertEqual(env.action_space.shape, (8,))
        self.assertEqual(env.observation_space.shape, (len(FlyTypingEnv.OBS_LABELS),))

    def test_key_events_carry_editor_fields(self):
        env = FlyTypingEnv(targets=("a",))
        env.reset(seed=1, options={"target": "a"})
        done = False
        while not done:
            _o, _r, te, tr, _i = env.step(env.expert_action())
            done = te or tr
        keys = [e for e in env.events if e["type"] == "key"]
        self.assertTrue(keys)
        self.assertEqual(keys[-1]["op"], "insert")
        self.assertEqual(keys[-1]["buffer_hash"], keys[-1]["text_hash"])


class MNEnv(unittest.TestCase):
    def test_spaces_and_labels(self):
        env = FlyTypingEnv(targets=("a",), action_mode="mn")
        self.assertEqual(env.action_space.shape, (16,))
        obs, _ = env.reset(seed=0, options={"target": "a"})
        self.assertEqual(obs.shape, (len(obs_labels(True, False)),))
        self.assertEqual(int(obs[len(FlyTypingEnv.OBS_LABELS):].sum()), 1)   # one-hot target key
        self.assertEqual(len(JOINT_NAMES), 14)
        env2 = FlyTypingEnv(targets=("a",), action_mode="mn", editor_obs=True, state_assist=False)
        obs2, _ = env2.reset(seed=0, options={"target": "a"})
        self.assertEqual(obs2.shape, (len(obs_labels(True, True)),))
        for name in ("anchor_dx", "tip_dx", "key_side"):
            self.assertEqual(obs2[FlyTypingEnv.OBS_LABELS.index(name)], 0.0)

    def test_expert_joint_labels_type_characters(self):
        env = FlyTypingEnv(targets=("a",), action_mode="mn")
        for seed, target in enumerate(["fly", "Hi"]):
            env.reset(seed=seed, options={"target": target})
            done = False
            while not done:
                action = env.expert_action()
                self.assertEqual(action.shape, (16,))
                self.assertLessEqual(float(np.abs(action).max()), 1.0)
                _o, _r, te, tr, info = env.step(action)
                done = te or tr
            self.assertEqual(info["text"], target)

    def test_joint_targets_are_rest_plus_scaled_action(self):
        env = FlyTypingEnv(targets=("a",), action_mode="mn", max_joint_rate=10.0)
        env.reset(seed=0, options={"target": "a"})
        action = np.zeros(16)
        action[2:] = np.linspace(-0.5, 0.5, 14)
        env.step(action)
        from backend.fly_env import TYPER_INDEX
        expected = env.joint_rest + action[2:].reshape(2, 7) * JOINT_SCALE
        np.testing.assert_allclose(env.q[TYPER_INDEX], expected, atol=1e-9)


@unittest.skipUnless(HAVE_TRAINING, "needs the training extra")
class MNReadout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from stable_baselines3.common.vec_env import DummyVecEnv
        from training.train_ppo import make_model
        cls.torch = torch
        config = {"seed": 0, "ppo": {"n_steps": 32, "batch_size": 16, "policy_kwargs": {}},
                  "policy": {"type": "connectome", "connectome": {"readout": "mn_antagonist", "device": "cpu"}}}
        vec = DummyVecEnv([lambda: FlyTypingEnv(targets=("a",), action_mode="mn")])
        cls.model = make_model(config, vec, seed=0, device="cpu")

    def test_no_free_linear_action_net(self):
        policy = self.model.policy
        self.assertIsInstance(policy.action_net, self.torch.nn.Identity)
        readout = policy.actor.readout
        self.assertEqual(readout.described()["learned_joint_params"], 28)
        self.assertEqual(readout.one_sided, ["joint_RFFemur_roll", "joint_LFFemur_roll"])
        self.assertEqual(readout.fallback_joints, [])
        self.assertIn("not MN driven", readout.body_velocity_source)
        opt_params = {id(p) for g in policy.optimizer.param_groups for p in g["params"]}
        self.assertIn(id(readout.gain), opt_params)

    def test_action_equals_readout_output(self):
        torch = self.torch
        policy = self.model.policy
        obs = torch.rand(3, policy.observation_space.shape[0]) * 2 - 1
        with torch.no_grad():
            mean = policy.get_distribution(obs).distribution.mean
            rates_out = policy.mlp_extractor.forward_actor(policy.extract_features(obs))
        self.assertTrue(torch.allclose(mean, rates_out))
        self.assertEqual(mean.shape, (3, 16))

    def test_joint_drive_uses_side_matched_mns_only(self):
        torch = self.torch
        policy = self.model.policy
        actor, readout = policy.actor, policy.actor.readout
        rates = torch.zeros(1, actor.runtime.n)
        left_mn = actor.runtime.idx_MN_L if hasattr(actor.runtime, "idx_MN_L") else None
        del left_mn
        arrays = actor.runtime.circuit_arrays
        # Drive every right-side Ta depressor MN: only the RF Tarsus1 joint sees positive drive.
        is_r = (arrays["mn_type"] == "Ta depressor MN") & (arrays["mn_side"] == "R")
        rates[0, torch.as_tensor(np.flatnonzero(is_r))] = 1.0
        drive = readout.mn_drive(actor.runtime, rates)[0]
        rf, lf = JOINT_NAMES.index("joint_RFTarsus1"), JOINT_NAMES.index("joint_LFTarsus1")
        self.assertEqual(float(drive[rf]), float(is_r.sum()))
        self.assertEqual(float(drive[lf]), 0.0)
        self.assertEqual(int((drive != 0).sum()), 1)


if __name__ == "__main__":
    unittest.main()
