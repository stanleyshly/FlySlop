"""Persistent progress/error logging contracts for curriculum training."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from contextlib import redirect_stdout
from io import StringIO

from training import curriculum as cur
from training.stages import stage5_oracle_data
from training.stages import physical_common as pcm


class TrainingLoggingTest(unittest.TestCase):
    def test_curriculum_writes_stage_events_and_full_traceback(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            stage_cfg = {"id": 1, "name": "broken", "module": "does.not.exist", "budget": {}}
            entry = {"id": 1, "attempts": 1}
            state = {"config": {"data_files": []}}
            with (mock.patch.object(cur.importlib, "import_module", side_effect=ValueError("bad training config")),
                  mock.patch.object(cur, "code_hash", return_value="hash"),
                  mock.patch.object(cur, "data_hash", return_value=None),
                  mock.patch.object(cur, "circuit_hash", return_value=None),
                  mock.patch.object(cur.memory_budget, "peak_rss", return_value=0)):
                cur.run_stage(state, entry, stage_cfg, run_dir, False, 1.0, 7)

            self.assertEqual(entry["status"], "error")
            self.assertIn("ValueError: bad training config", entry["traceback"])
            events = [json.loads(line) for line in (run_dir / "training.jsonl").read_text().splitlines()]
            self.assertEqual([e["event"] for e in events], ["stage_start", "stage_end"])
            self.assertIn("Traceback", events[-1]["traceback"])

    def test_oracle_child_is_unbuffered_and_progress_enabled(self):
        cmd = stage5_oracle_data.build_cmd(stage5_oracle_data.oracle_cfg({}, False), 30, 1.0)
        self.assertEqual(cmd[1:4], ["-u", "-m", "training.oracle_gen"])
        self.assertNotIn("--quiet", cmd)

    def test_real_ppo_logging_across_phases(self):
        from stable_baselines3 import PPO
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            ctx = cur.StageContext(run_dir, run_dir / "stage1", 0, {}, False, 2)
            model = PPO("MlpPolicy", "CartPole-v1", n_steps=16, batch_size=16,
                        n_epochs=1, seed=0, device="cpu")
            try:
                with redirect_stdout(StringIO()):
                    pcm.ppo_learn(ctx, model, 32)
                    pcm.ppo_learn(ctx, model, 32)
                events = [json.loads(line) for line in (run_dir / "training.jsonl").read_text().splitlines()]
                endings = [e for e in events if e["event"] == "ppo_end"]
                self.assertEqual([e["timesteps"] for e in endings], [32, 32])
                self.assertEqual(ctx.steps, 64)
                for row in endings:
                    self.assertGreater(row["steps_per_s"], 0)
                    self.assertIn("mean_step_reward", row)
                    self.assertIn("train/loss", row)
                self.assertTrue(any(e["event"] == "ppo_optimizer" for e in events))
            finally:
                model.get_env().close()


if __name__ == "__main__":
    unittest.main()
