"""Stage runners 1-5: budget/argument plumbing, warm-start artifact passing and gate logic with fakes (no physics)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from training import curriculum as cur
from training.stages import StageResult
from training.stages import physical_common as pcm
from training.stages import stage1_keys, stage2_transcription, stage3_editing, stage4_recovery, stage5_oracle_data

ROOT = Path(__file__).resolve().parents[1]


def mkctx(tmp, budget=None, sid=1):
    run = Path(tmp) / "run"
    sd = run / f"stage{sid}"
    sd.mkdir(parents=True, exist_ok=True)
    return cur.StageContext(run, sd, 0, budget or {"wall_s": 100, "steps": 1000}, False, 0.6)


class GateLogic(unittest.TestCase):
    def test_key_metrics_and_gate(self):
        eps = [{"target": "a", "text": "a"}] * 19 + [{"target": "a", "text": "s"}]
        m = pcm.key_metrics(eps)
        self.assertAlmostEqual(m["key_accuracy"], 0.95)
        self.assertAlmostEqual(m["unintended_rate"], 0.05)
        ok, d = pcm.gate_check(m, {"key_accuracy": 0.95, "unintended_rate_max": 0.01}, stage1_keys.RULES)
        self.assertFalse(ok)
        self.assertTrue(d["key_accuracy"]["ok"] and not d["unintended_rate_max"]["ok"])
        ok, _ = pcm.gate_check({"key_accuracy": 0.97, "unintended_rate": 0.0}, {"key_accuracy": 0.95, "unintended_rate_max": 0.01},
                               stage1_keys.RULES)
        self.assertTrue(ok)

    def test_unintended_counts_extra_chars(self):
        m = pcm.key_metrics([{"target": "a", "text": "aa"}, {"target": "b", "text": ""}])
        self.assertEqual(m["unintended_rate"], 0.5)

    def test_missing_metric_fails(self):
        ok, _ = pcm.gate_check({"token_exact": 0.95}, {"token_exact": 0.9, "code_exact": 0.7}, stage2_transcription.RULES)
        self.assertFalse(ok)

    def test_relaxed_smoke_is_labelled(self):
        m = {}
        self.assertTrue(pcm.finalize_gate(m, False, {}, {}, True))
        self.assertIn("NOT a real gate", m["gate_mode"])
        self.assertFalse(m["real_gate_passed"])
        m = {}
        self.assertFalse(pcm.finalize_gate(m, False, {}, {"smoke_gates": "strict"}, True))
        self.assertFalse(pcm.finalize_gate({}, False, {}, {}, False))   # never relaxed outside smoke
        self.assertTrue(pcm.finalize_gate({}, True, {}, {}, False))

    def test_phys_cfg_smoke_merge(self):
        sc = {"physical": {"bc_episodes": 400}, "smoke_physical": {"bc_episodes": 3}}
        self.assertEqual(pcm.phys_cfg(sc, False)["bc_episodes"], 400)
        self.assertEqual(pcm.phys_cfg(sc, True)["bc_episodes"], 3)
        self.assertEqual(pcm.phys_cfg(sc, True)["readout"], "mn_antagonist")

    def test_real_config_wires_stages(self):
        cfg = json.loads((ROOT / "training" / "curriculum.json").read_text())
        for s in cfg["stages"][:5]:
            self.assertNotIn("stub", s)
        self.assertEqual(cfg["stages"][0]["gate"], {"key_accuracy": 0.95, "unintended_rate_max": 0.01})
        self.assertEqual(cfg["stages"][4]["oracle"]["pairs_min"], 10000)
        self.assertEqual(cfg["stages"][3]["physical"]["eval_episodes"], 50)


class WarmStart(unittest.TestCase):
    def test_prev_checkpoint_from_state_then_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = mkctx(tmp, sid=2)
            self.assertIsNone(pcm.prev_checkpoint(ctx, 2))
            ck = ctx.run_dir / "stage1" / "best_model.zip"
            ck.parent.mkdir(parents=True)
            ck.write_bytes(b"x")
            self.assertEqual(pcm.prev_checkpoint(ctx, 2), str(ck))          # on-disk fallback
            other = ctx.run_dir / "elsewhere" / "best_model.zip"
            other.parent.mkdir()
            other.write_bytes(b"y")
            (ctx.run_dir / "state.json").write_text(json.dumps({"stages": [{"id": 1, "artifacts": [str(other)]}]}))
            self.assertEqual(pcm.prev_checkpoint(ctx, 2), str(other))       # state.json artifacts win

    def test_pick_best_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.zip", Path(tmp) / "b.zip"
            a.write_bytes(b"A")
            b.write_bytes(b"B")
            best, score = pcm.pick_best([(str(a), 0.1), (str(b), 0.9), ("/nope", 5.0)], Path(tmp) / "best_model.zip")
            self.assertEqual((Path(best).read_bytes(), score), (b"B", 0.9))

    def _fake_train(self, log_eval, calls):
        def fake(ctx, config, pc, phases, warm, seed, out_dir, on_phase=None):
            calls.append({"warm": warm, "phases": [p["name"] for p in phases], "pc": pc})
            out_dir.mkdir(parents=True, exist_ok=True)
            p = out_dir / "phase0.zip"
            p.write_bytes(b"ck")
            return [{"phase": "p", "ckpt": str(p), "eval": log_eval}], [(str(p), 1.0)], str(p)
        return fake

    def test_stage1_passes_warm_start_and_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = mkctx(tmp, sid=1)
            calls = []
            ev = {"key_accuracy": 0.99, "unintended_rate": 0.0, "episodes": 5}
            cfg = {"gate": {"key_accuracy": 0.95, "unintended_rate_max": 0.01}, "init_from": "/x/init.zip",
                   "physical": {"config": "training/ppo_fly_mn.json"}}
            with mock.patch.object(pcm, "train_phases", self._fake_train(ev, calls)), \
                    mock.patch.object(pcm, "build_config", lambda pc: {"tokens": {"min_len": 3, "max_len": 8}}):
                res = stage1_keys.run(ctx, cfg, False)
            self.assertEqual(calls[0]["warm"], "/x/init.zip")
            self.assertEqual(calls[0]["phases"], ["keys8", "keys26", "keys_full"])
            self.assertTrue(res.gate_passed)
            self.assertTrue(res.artifacts[0].endswith("stage1/best_model.zip"))
            self.assertTrue(Path(res.artifacts[0]).exists())
            # stage 2 finds stage 1's artifact through state.json
            ctx2 = mkctx(tmp, sid=2)
            (ctx2.run_dir / "state.json").write_text(json.dumps({"stages": [{"id": 1, "artifacts": res.artifacts}]}))
            self.assertEqual(pcm.prev_checkpoint(ctx2, 2), res.artifacts[0])

    def test_stage1_gate_fail_and_smoke_relaxed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = mkctx(tmp)
            ev = {"key_accuracy": 0.5, "unintended_rate": 0.3, "episodes": 5}
            cfg = {"gate": {"key_accuracy": 0.95, "unintended_rate_max": 0.01}}
            with mock.patch.object(pcm, "train_phases", self._fake_train(ev, [])), \
                    mock.patch.object(pcm, "build_config", lambda pc: {"tokens": {"min_len": 3, "max_len": 8}}):
                self.assertFalse(stage1_keys.run(ctx, cfg, False).gate_passed)
                res = stage1_keys.run(ctx, {**cfg, "smoke_gates": "relaxed"}, True)
            self.assertTrue(res.gate_passed)
            self.assertFalse(res.metrics["real_gate_passed"])
            self.assertIn("NOT a real gate", res.metrics["status"])


class SymbolicWiring(unittest.TestCase):
    def _run(self, stage_mod, sym, ev, smoke=False, gate=None):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = mkctx(tmp, sid=stage_mod.STAGE)
            ph = {"phases": [], "warm_start": None, "best_checkpoint": None, "eval": ev}
            with mock.patch.object(stage_mod, "physical", lambda *a, **k: ph), \
                    mock.patch.object(pcm, "run_symbolic", lambda *a, **k: sym):
                return stage_mod.run(ctx, {"gate": gate or {}, "physical": {}}, smoke)

    def test_stage3_needs_both(self):
        g = {"edit_key_accuracy": 0.95, "symbolic_within_1.5x_oracle": 0.95}
        ok = {"status": "ok", "metrics": {"first_round_ok": 0.97}, "gate_passed": True, "artifacts": ["s.pt"]}
        r = self._run(stage3_editing, ok, {"edit_key_accuracy": 0.96}, gate=g)
        self.assertTrue(r.gate_passed)
        self.assertIn("s.pt", r.artifacts)
        bad = {**ok, "metrics": {"first_round_ok": 0.5}}
        self.assertFalse(self._run(stage3_editing, bad, {"edit_key_accuracy": 0.96}, gate=g).gate_passed)
        self.assertFalse(self._run(stage3_editing, ok, {"edit_key_accuracy": 0.9}, gate=g).gate_passed)

    def test_symbolic_pending_reported(self):
        pend = {"status": "symbolic part pending", "gate_passed": False}
        r = self._run(stage4_recovery, pend, {"recovery_exact": 0.95}, gate={"recovery_exact": 0.9})
        self.assertFalse(r.gate_passed)
        self.assertEqual(r.metrics["symbolic"]["status"], "symbolic part pending")
        r = self._run(stage4_recovery, pend, {"recovery_exact": 0.95}, smoke=True, gate={"recovery_exact": 0.9})
        self.assertTrue(r.gate_passed)                                   # relaxed smoke, labelled
        self.assertIn("NOT a real gate", r.metrics["gate_mode"])

    def test_run_symbolic_missing_module(self):
        import training
        # Hide the package attribute too: once another test imported edit_trainer,
        # `from training import edit_trainer` would find it there despite sys.modules.
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(sys.modules, {"training.edit_trainer": None}), \
                mock.patch.dict(training.__dict__):
            training.__dict__.pop("edit_trainer", None)
            out = pcm.run_symbolic(SimpleNamespace(budget={}, elapsed=0, stage_dir=tmp), {}, True, 3)
        self.assertEqual(out["status"], "symbolic part pending")

    def test_run_symbolic_delegates_with_subctx(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = mkctx(tmp, {"wall_s": 100, "steps": 10}, sid=3)
            seen = {}

            def fake(c, sc, smoke, stage):
                seen.update(stage_dir=c.stage_dir, budget=c.budget, ov=sc["overrides"], stage=stage)
                return StageResult({"first_round_ok": 1.0}, True, ["ck.pt"])
            fake_mod = SimpleNamespace(run_stage=fake)
            with mock.patch.dict(sys.modules, {"training.edit_trainer": fake_mod}), \
                    mock.patch("training.edit_trainer", fake_mod, create=True):
                out = pcm.run_symbolic(ctx, {}, True, 3)
            self.assertEqual(out["status"], "ok")
            self.assertEqual(seen["stage_dir"].name, "symbolic")
            self.assertEqual(seen["budget"], ctx.budget)
            self.assertLess(seen["ov"]["train.max_seconds"], 100)


class Samplers(unittest.TestCase):
    def test_edit_sampler_compiles_and_slip_sequences_recover(self):
        import random
        from training.physical_eval import symbolic
        s = pcm.EditSampler(["hello", "world"], include_fn=False)
        cmds, keys, _ = s.compiled(random.Random(1))
        self.assertTrue(keys)
        s = pcm.EditSampler(["hello", "world", "abcde"], include_fn=False, slip_p=0.6)
        for i in range(10):
            cmds, want = s.commands(random.Random(i))
            self.assertEqual(symbolic(cmds)["text"], want)              # slip then Backspace + retype ends clean

    def test_fn_filtered_when_unsupported(self):
        import random
        from training.physical_eval import Unsupported, compile_commands
        s = pcm.EditSampler(["hello"], include_fn=True)
        for i in range(20):
            try:
                cmds, keys, _ = s.compiled(random.Random(i))
            except RuntimeError:
                continue
            compile_commands(cmds)                                       # never hands out an uncompilable sequence


class TrainPhasesPlumbing(unittest.TestCase):
    def test_ticks_and_interrupted_checkpoint_on_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = mkctx(tmp, {"steps": 5})
            model = mock.MagicMock()
            model.save.side_effect = lambda p: Path(p).write_bytes(b"m")

            def boom(c, *a, **k):
                c.tick(10)
            with mock.patch.object(pcm, "new_model", lambda *a, **k: model), \
                    mock.patch.object(pcm, "bc_collect_pool", lambda *a, **k: (None, None, {})), \
                    mock.patch.object(pcm, "bc_fit", lambda c, *a, **k: boom(c)):
                with self.assertRaises(cur.BudgetExceeded):
                    pcm.train_phases(ctx, {}, {"bc_workers": 4}, [{"name": "p", "pool": ["a"], "bc_episodes": 2,
                                                                    "bc_epochs": 1}], None, 0, Path(tmp) / "o")
            self.assertTrue((Path(tmp) / "o" / "interrupted.zip").exists())

    def test_bc_workers_capped_by_ctx(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = mkctx(tmp)
            ctx.max_workers = 2
            seen = {}
            model = mock.MagicMock()
            with mock.patch.object(pcm, "new_model", lambda *a, **k: model), \
                    mock.patch.object(pcm, "bc_collect_pool", lambda c, cfg, pool, n, s, w: seen.setdefault("w", w) and (None, None, {})), \
                    mock.patch.object(pcm, "bc_fit", lambda *a, **k: [{"total": 1.0}]), \
                    mock.patch.object(pcm, "save_model", lambda m, p: str(p)):
                pcm.train_phases(ctx, {}, {"bc_workers": 8}, [{"name": "p", "pool": ["a"], "bc_episodes": 2, "bc_epochs": 1}],
                                 "warm.zip", 0, Path(tmp) / "o")
            self.assertEqual(seen["w"], 2)


class Stage5(unittest.TestCase):
    def test_cfg_defaults_and_smoke(self):
        o = stage5_oracle_data.oracle_cfg({}, False)
        self.assertEqual((o["pairs_min"], o["spec_tokens"], o["code_tokens"], o["neardup_test"], o["neardup_accepted"]),
                         (10000, 160, 384, 0.7, 0.9))
        s = stage5_oracle_data.oracle_cfg({"oracle": {"run": "r"}}, True)
        self.assertTrue(s["standin"])
        self.assertEqual(s["limit"], 12)

    def test_build_cmd(self):
        o = stage5_oracle_data.oracle_cfg({"oracle": {"sources": ["mbpp", "rtllm"], "holdout_sources": "lab", "run": "r",
                                                      "time_budget_s": 99}, "smoke_oracle": {"run": "r"}}, True)
        c = stage5_oracle_data.build_cmd(o, 40.0, 0.6)
        for tok in ("--standin", "--limit", "12", "--sources", "mbpp,rtllm", "--holdout-sources", "lab", "--time-budget-s", "40",
                    "--run", "r", "--max-ram-gb", "0.6"):
            self.assertIn(tok, c)

    def test_count_pairs_dedupes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "r"
            d.mkdir()
            rows = [{"id": "a", "passed": True}, {"id": "b", "passed": False}, {"id": "a", "passed": True}]
            (d / "pairs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json")
            self.assertEqual(stage5_oracle_data.count_pairs(Path(tmp), {"run": "r", "pairs_glob": None}), (1, 2))
            self.assertEqual(stage5_oracle_data.count_pairs(Path(tmp), {"run": "x", "pairs_glob": str(d / "*.jsonl")}), (1, 2))

    def test_poll_kills_child_on_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = mkctx(tmp, {"steps": 3})
            proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            with self.assertRaises(cur.BudgetExceeded):
                stage5_oracle_data.poll(ctx, proc, 0.05)
            self.assertIsNotNone(proc.poll())

    def test_run_gate_with_fake_generator(self):
        from training import datasets as ds
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "oracle"
            (root / "r").mkdir(parents=True)
            rows = [{"id": str(i), "passed": i % 2 == 0} for i in range(10)]
            (root / "r" / "pairs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            (root / "r" / "manifest.json").write_text(json.dumps({"stopped": "limit", "standin": True}))
            ctx = mkctx(tmp, sid=5)
            fake_proc = SimpleNamespace(poll=lambda: 0, returncode=0)
            cfg = {"gate": {"filtered_pairs": 5}, "oracle": {"run": "r", "pairs_min": 5, "poll_s": 0.01}}
            with mock.patch.object(ds, "PRIVATE", Path(tmp)), \
                    mock.patch.object(stage5_oracle_data.subprocess, "Popen") as popen:
                (Path(tmp) / "oracle").mkdir(exist_ok=True)
                res = stage5_oracle_data.run(ctx, cfg, False)          # 5 >= pairs_min -> no generation
                popen.assert_not_called()
                self.assertTrue(res.gate_passed)
                self.assertEqual(res.metrics["filtered_pairs"], 5)
                self.assertAlmostEqual(res.metrics["pass_rate"], 0.5)
                cfg["gate"]["filtered_pairs"] = 6
                self.assertFalse(stage5_oracle_data.run(ctx, cfg, False).gate_passed)
                popen.return_value = fake_proc      # smoke always runs the generator (mocked, exits 0)
                res = stage5_oracle_data.run(ctx, {**cfg, "smoke_oracle": {"run": "r"}}, True)
                popen.assert_called_once()
                self.assertTrue(res.gate_passed)
                self.assertIn("NOT a real gate", res.metrics["gate_mode"])
                self.assertFalse(res.metrics["real_gate_passed"])


if __name__ == "__main__":
    unittest.main()
