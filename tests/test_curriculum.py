"""Curriculum orchestrator: transitions, resume after kill, budgets, manifests."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from training import curriculum as cur

ROOT = Path(__file__).resolve().parents[1]
PY = [sys.executable, "-m", "training.curriculum"]


def make_cfg(over=None, n=3):
    over = over or {}
    stages = [{"id": i, "name": f"s{i}", "module": "tests.curriculum_stub_stage", "exclusive": False,
               "gate": {}, "stub": True, "budget": {"wall_s": 600, "steps": 100}} for i in range(1, n + 1)]
    for i, o in over.items():
        stages[i - 1].update(o)
    return {"seed": 7, "ram_gb": 1, "stages": stages, "data_files": ["training/curriculum.json"], "smoke": {"budget": {"steps": 5}}}


def go(tmp, cfg, **kw):
    run = Path(tmp) / "run"
    state = cur.new_state(cfg, False, {})
    return run, cur.drive(run, state, False, kw.pop("advance", False), **kw)


class CurriculumTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def status(self, state):
        return [s["status"] for s in state["stages"]]

    def test_all_pass(self):
        run, st = go(self.tmp, make_cfg())
        self.assertEqual(st["status"], "complete")
        self.assertEqual(self.status(st), ["passed"] * 3)
        self.assertEqual(json.loads((run / "state.json").read_text())["status"], "complete")
        self.assertEqual(st["stages"][0]["metrics"]["status"], "not implemented")
        self.assertFalse(list(run.glob("state.json.tmp*")))

    def test_gate_failure_stops(self):
        run, st = go(self.tmp, make_cfg({2: {"stub": False}}))
        self.assertEqual(st["status"], "stopped")
        self.assertEqual(self.status(st), ["passed", "gate_failed", "pending"])

    def test_step_budget_stop_and_advance(self):
        cfg = make_cfg({2: {"stub_steps": 50, "budget": {"steps": 10}}})
        run, st = go(self.tmp, cfg)
        self.assertEqual(self.status(st), ["passed", "budget_stopped", "pending"])
        self.assertIn("steps", st["stages"][1]["reason"])
        _, st = go(self.tmp + "/b", cfg, advance=True)
        self.assertEqual(self.status(st), ["passed", "advanced", "passed"])
        self.assertEqual(st["status"], "complete")

    def test_wall_budget(self):
        cfg = make_cfg({1: {"stub_steps": 20, "stub_sleep_s": 0.02, "budget": {"wall_s": 0.1}}})
        _, st = go(self.tmp, cfg)
        self.assertEqual(st["stages"][0]["status"], "budget_stopped")
        self.assertIn("wall_s", st["stages"][0]["reason"])

    def test_error_then_resume(self):
        cfg = make_cfg({2: {"stub_raise": True}})
        run, st = go(self.tmp, cfg)
        self.assertEqual(self.status(st), ["passed", "error", "pending"])
        st["config"]["stages"][1]["stub_raise"] = False  # fixed between runs
        st = cur.drive(run, st, False, False)
        self.assertEqual(self.status(st), ["passed"] * 3)
        self.assertEqual(st["stages"][1]["attempts"], 2)
        self.assertEqual(st["stages"][0]["attempts"], 1)

    def test_from_stage(self):
        _, st = go(self.tmp, make_cfg(), from_stage=3)
        self.assertEqual(self.status(st), ["skipped", "skipped", "passed"])

    def test_manifest_fields(self):
        run, st = go(self.tmp, make_cfg({1: {"exclusive": True}}, n=1))
        m = json.loads((run / "stage1" / "manifest.json").read_text())
        for k in ("code_hash", "data_hash", "circuit_hash", "config", "seed", "time_s", "peak_rss_bytes", "exclusive"):
            self.assertIn(k, m)
        self.assertEqual(len(m["code_hash"]), 64)
        self.assertEqual(len(m["data_hash"]), 64)
        self.assertEqual(m["circuit_hash"] is None, not cur.CIRCUIT_IO.exists())
        self.assertEqual(m["seed"], 8)
        self.assertTrue(m["exclusive"] and m["peak_rss_bytes"] > 0)
        ctx = cur.StageContext(run, run, 0, {}, True, 4)
        self.assertEqual(ctx.max_workers, 0)

    def test_kill_and_resume_subprocess(self):
        cfg = make_cfg({2: {"stub_steps": 100, "stub_sleep_s": 0.05}})
        cfgp = Path(self.tmp) / "c.json"
        cfgp.write_text(json.dumps(cfg))
        runs = Path(self.tmp) / "runs"
        env = {**os.environ, "FLYSLOP_MAX_RAM_GB": "0.5"}
        base = PY + ["--config", str(cfgp), "--runs-dir", str(runs), "--ram-gb", "0.5"]
        p = subprocess.Popen(base, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 60
        state_p = None
        while time.time() < deadline:
            found = list(runs.glob("*/state.json"))
            if found:
                s = json.loads(found[0].read_text())["stages"]
                if s[1]["status"] == "running":
                    state_p = found[0]
                    break
            time.sleep(0.05)
        p.send_signal(signal.SIGKILL)
        p.wait()
        self.assertIsNotNone(state_p, "never reached stage 2")
        self.assertEqual([s["status"] for s in json.loads(state_p.read_text())["stages"]], ["passed", "running", "pending"])
        # resume with a fast stage 2 (edit the persisted config, as an operator fixing budgets would)
        st = json.loads(state_p.read_text())
        st["config"]["stages"][1]["stub_steps"] = 1
        state_p.write_text(json.dumps(st))
        r = subprocess.run(PY + ["--resume", state_p.parent.name, "--runs-dir", str(runs), "--ram-gb", "0.5"],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        st = json.loads(state_p.read_text())
        self.assertEqual([s["status"] for s in st["stages"]], ["passed"] * 3)
        self.assertEqual([s["attempts"] for s in st["stages"]], [1, 2, 1])

    def test_real_config_smoke_has_nine_stages(self):
        cfg = cur.effective_config(json.loads((ROOT / "training" / "curriculum.json").read_text()), True)
        self.assertEqual(len(cfg["stages"]), 9)
        self.assertEqual(cfg["ram_gb"], 4)
        self.assertTrue(all(s["budget"]["steps"] <= 200000 and s["budget"]["wall_s"] <= 240 for s in cfg["stages"]))
        self.assertTrue(all(s["smoke_gates"] == "relaxed" for s in cfg["stages"]))
        self.assertTrue(cfg["stages"][4]["exclusive"])


    def test_ram_precedence(self):
        cfg = {"ram_gb": 4}
        self.assertEqual(cur.resolve_ram_gb(0.7, cfg, {"FLYSLOP_MAX_RAM_GB": "2"}), 0.7)   # --ram-gb wins
        self.assertEqual(cur.resolve_ram_gb(None, cfg, {"FLYSLOP_MAX_RAM_GB": "2"}), 2.0)  # env beats config
        self.assertEqual(cur.resolve_ram_gb(None, cfg, {}), 4.0)                            # config
        self.assertEqual(cur.resolve_ram_gb(None, {}, {}), 4.0)                             # default

    def test_env_ram_reaches_stage_context(self):
        r = subprocess.run([sys.executable, "-c", "import os,sys;from training import curriculum as c;"
                            "cfg={'ram_gb':4};print(c.resolve_ram_gb(None,cfg))"], cwd=ROOT, capture_output=True, text=True,
                           env={**os.environ, "FLYSLOP_MAX_RAM_GB": "1.5"})
        self.assertEqual(r.stdout.strip(), "1.5", r.stderr)


if __name__ == "__main__":
    unittest.main()
