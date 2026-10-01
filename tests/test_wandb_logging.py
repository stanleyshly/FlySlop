"""W&B tracking remains opt-in and sends only scalar progress, never training text."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from training import wandb_tracking as wt
from training import curriculum as cur
from training.stages import physical_common as pcm


class FakeRun:
    def __init__(self, fail=False):
        self.logged = []
        self.finished = []
        self.fail = fail

    def log(self, payload, **kwargs):
        if self.fail:
            raise OSError("offline")
        self.logged.append((payload, kwargs))

    def finish(self, **kwargs):
        self.finished.append(kwargs)


def fake_wandb(run, init_error=None):
    calls = []

    def init(**kwargs):
        calls.append(kwargs)
        if init_error:
            raise init_error
        return run

    return SimpleNamespace(Settings=lambda **kw: kw, init=init), calls


class WandbTrackingTest(unittest.TestCase):
    def test_no_project_disables_import_and_init(self):
        self.assertIsNone(wt.WandbTracker.start(Path("unused"), None, None, "online", {}))

    def test_offline_run_logs_scalar_progress_without_source_or_text(self):
        run = FakeRun()
        fake, calls = fake_wandb(run)
        with tempfile.TemporaryDirectory() as td, mock.patch.dict("sys.modules", {"wandb": fake}):
            tracker = wt.WandbTracker.start(Path(td), "flyslop", "team", "offline",
                                             {"run_type": "curriculum"})
            tracker.log_event({"event": "ppo_progress", "stage": "stage1", "timesteps": 12,
                               "steps_per_s": 50.5, "code_exact": 0.75,
                               "prompt": "private prompt", "text": "private code", "traceback": "secret"})
            tracker.log_values("curriculum/stage1/bc", {"loss": 0.125, "phase": "keys8"})
            tracker.finish(0)

            self.assertEqual(calls[0]["mode"], "offline")
            self.assertFalse(calls[0]["save_code"])
            settings = calls[0]["settings"]
            self.assertEqual(settings, {"console": "off", "disable_code": True, "disable_git": True,
                                        "init_timeout": 30})
            self.assertEqual(len(run.logged), 2)
            first = run.logged[0][0]
            self.assertEqual(first["curriculum/stage1/ppo_progress/steps_per_s"], 50.5)
            self.assertEqual(first["curriculum/stage1/ppo_progress/code_exact"], 0.75)
            self.assertFalse(any("prompt" in key or "text" in key or "traceback" in key for key in first))
            self.assertEqual(run.finished, [{"exit_code": 0}])

    def test_online_resume_reuses_stable_id_only_for_same_project_and_entity(self):
        run = FakeRun()
        fake, calls = fake_wandb(run)
        with tempfile.TemporaryDirectory() as td, mock.patch.dict("sys.modules", {"wandb": fake}):
            meta = Path(td) / "wandb_run.json"
            meta.write_text(json.dumps({"id": "stable123", "mode": "online", "project": "p", "entity": "e"}))
            tracker = wt.WandbTracker.start(Path(td), "p", "e", "online", {}, resume=True)
            self.assertEqual(calls[0]["id"], "stable123")
            self.assertEqual(calls[0]["resume"], "must")
            tracker.finish()
        with tempfile.TemporaryDirectory() as td, mock.patch.dict("sys.modules", {"wandb": fake}):
            meta = Path(td) / "wandb_run.json"
            meta.write_text(json.dumps({"id": "stable123", "mode": "online", "project": "p", "entity": "e"}))
            wt.WandbTracker.start(Path(td), "p", "other", "online", {}, resume=True)
            self.assertNotEqual(calls[1]["id"], "stable123")
            self.assertEqual(calls[1]["resume"], "never")

    def test_logging_failure_disables_tracker_without_raising(self):
        tracker = wt.WandbTracker(FakeRun(fail=True), "id", "offline")
        tracker.log_values("ppo", {"loss": 0.5})
        self.assertTrue(tracker.disabled)
        tracker.log_values("ppo", {"loss": 0.25})

    def test_environment_opt_in(self):
        with mock.patch.dict(os.environ, {"WANDB_PROJECT": "p", "WANDB_ENTITY": "e", "WANDB_MODE": "offline"}):
            self.assertEqual(wt.options(), {"project": "p", "entity": "e", "mode": "offline"})
            self.assertEqual(wt.options("explicit"), {"project": "explicit", "entity": "e", "mode": "offline"})

    def test_curriculum_physical_event_keeps_local_log_and_tracks_scalars(self):
        run = FakeRun()
        tracker = wt.WandbTracker(run, "id", "offline")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx = cur.StageContext(root, root / "stage2", 1, {}, False, 1.0, tracker=tracker)
            with mock.patch("builtins.print"):
                pcm._event(ctx, "ppo_progress", stage="stage2", timesteps=100, steps_per_s=20.0,
                           mean_step_reward=0.25)
            local = json.loads((root / "training.jsonl").read_text().splitlines()[0])
            remote = run.logged[0][0]
            self.assertEqual(local["event"], "ppo_progress")
            self.assertEqual(remote["curriculum/stage2/ppo_progress/steps_per_s"], 20.0)
            self.assertEqual(remote["curriculum/stage2/ppo_progress/mean_step_reward"], 0.25)


if __name__ == "__main__":
    unittest.main()
