import json
import tempfile
import unittest
from pathlib import Path


def _skip_without_training(test):
    try:
        import stable_baselines3  # noqa: F401
        from backend.env import KeyboardTypingEnv  # noqa: F401
    except ImportError:
        test.skipTest("install the training extra to exercise PPO")


class PPOPipelineTests(unittest.TestCase):
    def test_env_passes_gymnasium_checker(self):
        _skip_without_training(self)
        from gymnasium.utils.env_checker import check_env
        from backend.env import KeyboardTypingEnv
        check_env(KeyboardTypingEnv(targets=("aB;",)), skip_render_check=True)

    def test_scripted_controller_types_heldout_tokens_with_replayable_contacts(self):
        _skip_without_training(self)
        from training.common import load_config, make_env, run_episode, scripted_action
        from training.tokens import token_splits
        config = load_config()
        splits = token_splits(**config["tokens"])
        env = make_env(config, splits["heldout"])
        for i, token in enumerate(splits["heldout"][:10]):
            result = run_episode(env, token, i, lambda _o, e: scripted_action(e))
            self.assertTrue(result["exact"], result)

    def test_hovering_never_types_and_wrong_key_terminates(self):
        _skip_without_training(self)
        from backend.env import KeyboardTypingEnv, key_center
        env = KeyboardTypingEnv(targets=("a",), random_start=False)
        env.reset(seed=0, options={"target": "a"})
        for _ in range(20):
            env.step([0.0, 0.0, 0.0])
        self.assertEqual(env.typed, "")
        # Press 's' instead of 'a'.
        cx, cy = key_center("s")
        env.foot[:] = [cx, cy, 0.6]
        terminated = False
        for dz in (-1, -1, -1, -1):
            _o, reward, terminated, _t, info = env.step([0.0, 0.0, dz])
            if terminated:
                break
        self.assertTrue(terminated and info["error"])
        self.assertEqual(env.typed, "s")

    def test_token_splits_are_disjoint(self):
        _skip_without_training(self)
        from training.tokens import token_splits
        splits = token_splits()
        self.assertTrue(splits["train"] and splits["heldout"])
        self.assertFalse(set(splits["train"]) & set(splits["heldout"]))

    def test_smoke_training_writes_manifest_and_model(self):
        _skip_without_training(self)
        from training.common import load_config
        from training.train_ppo import apply_smoke
        import subprocess, sys
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, "-m", "training.train_ppo", "--smoke", "--timesteps", "2048",
                            "--name", "t", "--out", tmp], check=True, capture_output=True)
            run = Path(tmp) / "t"
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertTrue(manifest["smoke"])
            for key in ("config_sha256", "code_sha256", "dataset_sha256", "seed"):
                self.assertIn(key, manifest)
            self.assertTrue((run / "final_model.zip").exists())
            self.assertTrue((run / "eval.jsonl").read_text().strip())
        self.assertEqual(apply_smoke(load_config())["n_envs"], 4)


if __name__ == "__main__":
    unittest.main()
