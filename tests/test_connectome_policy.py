import tempfile
import unittest
from pathlib import Path


def _need(test):
    try:
        import stable_baselines3, torch, mujoco  # noqa: F401
    except ImportError:
        test.skipTest("install the training extra")


def _config(policy):
    from training.common import load_config
    cfg = load_config("training/ppo_fly.json")
    cfg["ppo"].update(n_steps=32, batch_size=32, n_epochs=1)
    cfg["policy"] = policy
    return cfg


class ConnectomePolicyTests(unittest.TestCase):
    def _model(self, policy, n_envs=1):
        from stable_baselines3.common.vec_env import DummyVecEnv
        from training.common import make_env
        from training.train_ppo import make_model
        cfg = _config(policy)
        vec = DummyVecEnv([lambda: make_env(cfg, ["ab"]) for _ in range(n_envs)])
        return make_model(cfg, vec, seed=0), vec

    def test_make_model_selects_policy(self):
        _need(self)
        from training.connectome_policy import ConnectomeActorCriticPolicy
        m, vec = self._model({"type": "mlp"})
        self.assertNotIsInstance(m.policy, ConnectomeActorCriticPolicy)
        c, vec2 = self._model({"type": "connectome", "connectome": {"device": "cpu"}})
        self.assertIsInstance(c.policy, ConnectomeActorCriticPolicy)
        self.assertEqual(c.policy.actor.runtime.n, 3173)
        with self.assertRaises(ValueError):
            self._model({"type": "bogus"})

    def test_variants_are_matched_and_actions_depend_on_obs(self):
        _need(self)
        import numpy as np
        sig = {}
        for variant in ("real", "shuffled", "random_sparse", "frozen"):
            m, _ = self._model({"type": "connectome", "connectome": {"device": "cpu", "variant": variant}})
            sig[variant] = m.policy.actor.runtime.variant_signature()
        self.assertEqual({s["n_edges"] for s in sig.values()}, {sig["real"]["n_edges"]})
        self.assertEqual(sig["frozen"]["n_trainable"], 0)
        m, _ = self._model({"type": "connectome", "connectome": {"device": "cpu"}})
        rng = np.random.default_rng(0)
        o = rng.uniform(-1, 1, (2, 33)).astype("float32")
        import torch
        with torch.no_grad():
            lat = m.policy.mlp_extractor.forward_actor(torch.as_tensor(o))
        self.assertEqual(tuple(lat.shape), (2, 473))
        self.assertGreater(float((lat[0] - lat[1]).abs().mean()), 1e-4)   # signal propagates to MN/DN

    def test_checkpoint_reload_identical_actions(self):
        _need(self)
        import numpy as np
        from training.train_ppo import load_ppo
        m, _ = self._model({"type": "connectome", "connectome": {"device": "cpu", "k_steps": 3, "silence": ["proprio"]}})
        o = np.random.default_rng(1).uniform(-1, 1, (5, 33)).astype("float32")
        before = m.predict(o, deterministic=True)[0]
        with tempfile.TemporaryDirectory() as tmp:
            m.save(Path(tmp) / "m.zip")
            m2 = load_ppo(Path(tmp) / "m.zip", device="cpu")
        self.assertTrue(np.array_equal(before, m2.predict(o, deterministic=True)[0]))
        self.assertEqual(m2.policy.connectome_kwargs["k_steps"], 3)

    def test_legacy_circuit_adapter_dense_backend(self):
        _need(self)
        import torch
        from training.connectome_policy import RuntimeActor
        a = RuntimeActor(33, circuit="data/connectome/rf_leg_circuit.json", device="cpu")
        self.assertEqual((a.runtime.backend, a.runtime.n), ("dense", 360))
        self.assertEqual(a(torch.zeros(1, 33)).shape[1], a.latent_dim)

    def test_pluggable_readout(self):
        _need(self)
        import torch
        from training.connectome_policy import PopulationReadout, RuntimeActor, register_readout
        register_readout("mn_only_test", lambda rt: PopulationReadout(rt, ("MN",)))
        a = RuntimeActor(33, device="cpu", readout="mn_only_test")
        self.assertEqual(a(torch.zeros(1, 33)).shape[1], 173)

    def test_code_hash_covers_connectome_files(self):
        from training import common
        self.assertIn("backend/editor.py", common.CODE_FILES)
        self.assertTrue(common.CODE_GLOBS)
        self.assertEqual(common.code_hash(), common.code_hash())

    def test_replay_activity_summary_and_mlp_tolerance(self):
        _need(self)
        import numpy as np
        from backend.policy_replay import _actor_of, activity_summary
        s = activity_summary(np.array([[0.1, -0.9, 0.5], [0.0, 0.2, -0.3]]), ["DN", "MN", "IN"], k=2)
        self.assertEqual(s["top_ids"], [[1, 2], [2, 1]])
        self.assertEqual(set(s["population_mean_abs_rate"]), {"DN", "MN", "IN"})
        m, _ = self._model({"type": "mlp"})
        self.assertIsNone(_actor_of(m))


if __name__ == "__main__":
    unittest.main()
