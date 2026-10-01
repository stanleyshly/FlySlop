"""Stages 6-9 (P8c): fakes and monkeypatches only (no long training, no Verilator)."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from training import selfcorrect_train as sct
from training.stages import StageResult, stage6_distill, stage7_microprograms, stage8_longer, stage9_selfcorrect
from training.tokenizer import EOS, RUN, BPETokenizer
from training import thinker_train_common as C

# Stages call C.guard_ram(ctx.ram_gb), which starts a daemon watchdog that outlives the test and
# would abort the whole unittest process (and its subprocess tree) once it grows past 1 GB.
_guard_patch = mock.patch.object(C, "guard_ram", lambda ram_gb: None)


def setUpModule():
    _guard_patch.start()


def tearDownModule():
    _guard_patch.stop()


def ctx_for(tmp, budget=None):
    return SimpleNamespace(run_dir=Path(tmp), stage_dir=Path(tmp) / "stage", seed=0, ram_gb=1.0, budget=budget or {},
                           max_workers=1, elapsed=0.0, tick=lambda n=1: None)


class Stage6(unittest.TestCase):
    def test_beats_controls(self):
        ok, _ = stage6_distill.beats_controls({"shuffled": {"real_minus_token_acc_ci95": [0.01, 0.1]},
                                               "random_sparse": {"real_minus_token_acc_ci95": [0.02, 0.1]}})
        self.assertTrue(ok)
        bad, d = stage6_distill.beats_controls({"shuffled": {"real_minus_token_acc_ci95": [0.01, 0.1]},
                                                "random_sparse": {"real_minus_token_acc_ci95": [-0.02, 0.1]}})
        self.assertFalse(bad)
        self.assertFalse(d["random_sparse"]["ok"])
        self.assertFalse(stage6_distill.beats_controls({})[0])

    def test_smoke_relaxed_and_real_gate_recorded(self):
        fake = StageResult({"token_acc": 0.05, "kind": "gru"}, False, ["x/ckpt.pt"])
        with tempfile.TemporaryDirectory() as d, mock.patch.object(stage6_distill.distill, "run_stage", return_value=fake) as rs:
            r = stage6_distill.run(ctx_for(d), {"kind": "real", "smoke_stage": {"kind": "gru"}, "gate": {}}, True)
        self.assertEqual(rs.call_args[0][1]["kind"], "gru")
        self.assertTrue(r.gate_passed)
        self.assertFalse(r.metrics["real_gate_passed"])
        self.assertIn("NOT a real gate", r.metrics["gate_mode"])

    def test_full_controls_gate_and_artifact(self):
        summ = {"real": {"token_acc": {"mean": 0.7}}, "shuffled": {"real_minus_token_acc_ci95": [0.1, 0.2]},
                "random_sparse": {"real_minus_token_acc_ci95": [0.1, 0.2]}}
        fake = StageResult({"controls": summ}, True, ["c/controls.md"])
        with tempfile.TemporaryDirectory() as d:
            ck = Path(d) / "stage" / "real_s0" / "ckpt.pt"
            ck.parent.mkdir(parents=True)
            ck.write_bytes(b"x")
            with mock.patch.object(stage6_distill.distill, "run_stage", return_value=fake) as rs:
                r = stage6_distill.run(ctx_for(d, {"wall_s": 1000}), {"controls": True, "seeds": [0, 1], "kinds": ["real", "shuffled", "random_sparse"]}, False)
        self.assertTrue(r.gate_passed and r.metrics["real_gate_passed"])
        self.assertEqual(r.artifacts[0], str(ck))
        self.assertIn("train.max_seconds", rs.call_args[0][1]["overrides"])


class Stage7(unittest.TestCase):
    def test_physical_subset_uses_generated_text(self):
        pol = SimpleNamespace(write=lambda t: [ord(c) for c in "ab"] if t["id"] != "e" else [])
        tok = SimpleNamespace(decode=lambda ids: "".join(chr(i) for i in ids) + ("zzzzzz" if ids else ""))
        tasks = [{"id": "e", "lang": "py", "difficulty": 1, "lines": 1}, {"id": "a", "lang": "py", "difficulty": 1, "lines": 2},
                 {"id": "b", "lang": "sv", "difficulty": 1, "lines": 2}]
        calls = []

        def typer(text, **kw):
            calls.append(text)
            return {"exact": text == "abzz", "text_match": True, "contact_verified": True, "ticks": 3, "wall_s": 0.1}
        r = stage7_microprograms.physical_subset(pol, tok, tasks, 3, 4, "expert", "mn", 0, typer=typer)
        self.assertEqual(calls, ["abzz"] * 2)
        self.assertEqual(r["n"], 2)
        self.assertEqual(r["physical_exact_rate"], 1.0)
        self.assertTrue(r["rows"][0]["truncated"])

    def test_select_tasks_caps(self):
        ts = stage7_microprograms.select_tasks(3, 1)
        self.assertEqual(sorted(t["lang"] for t in ts), ["py", "py", "py", "sv"])


class Stage8(unittest.TestCase):
    def test_features_and_levels(self):
        f = stage8_longer.features
        self.assertEqual(f("def a(x):\n    if x:\n        return 1\n    return 2\n", "py"), {"branch"})
        self.assertEqual(f("def a():\n    for i in x:\n        pass\ndef b():\n    pass\n", "py"), {"loop", "multi_fn"})
        self.assertIn("branch", f("always @(*) if (a) y = 1; else y = 0;", "sv"))
        lv = {"max_lines": 2}
        pairs = [{"code": "a\nb", "lang": "py"}, {"code": "a\nb\nc", "lang": "py"}]
        self.assertEqual(len(stage8_longer.level_train_pairs(lv, pairs)), 1)
        fl = {"feature": "loop", "max_lines": 12}
        loop = {"code": "for i in x:\n    pass\n" + "\n".join("x" for _ in range(8)), "lang": "py"}
        self.assertEqual(len(stage8_longer.level_train_pairs(fl, pairs + [loop])), 3)   # loop pair + <=5-line replay

    def _run(self, rates, floor):
        it = iter(rates)
        tasks = [{"id": f"t{i}", "lang": "py", "lines": i + 1, "code": "x\n" * i + "x", "prompt": "p", "difficulty": 1} for i in range(3)]
        data = {"train": [{"code": "a", "lang": "py", "prompt": "p"}], "val": [], "label": "fake"}

        def ev(policy, ts, *a, **k):
            r = next(it)
            return {"pass_at_1": r, "n": len(ts), "rows": [{"id": t["id"], "lines": t["lines"], "ok": r > 0} for t in ts]}
        tr = lambda cfg, kind, seed, out, d, steps, resume=None, **k: (Path(out).mkdir(parents=True, exist_ok=True), (Path(out) / "ckpt.pt").write_bytes(b"c"),
                                                                       {"steps": steps, "final": {"token_acc": 0.1}})[2]
        with tempfile.TemporaryDirectory() as d:
            ck = Path(d) / "init.pt"
            torch.save({"kind": "gru", "step": 5}, ck)
            sc = {"init_from": str(ck), "steps_per_level": 4, "max_py": None, "max_sv": None,
                  "levels": [{"max_lines": 1}, {"max_lines": 2}, {"max_lines": 3}], "smoke_gates": "strict",
                  "gate": {"pass_rate_floor": floor}}
            with mock.patch("training.distill.load_pairs", return_value=data), mock.patch("training.distill._tok"), \
                    mock.patch("training.distill.train_kind", side_effect=tr), \
                    mock.patch("training.selfcorrect_train.load_thinker", return_value=(object(), {})), \
                    mock.patch("training.eval_microsuite.evaluate", side_effect=ev), \
                    mock.patch("training.stages.stage7_microprograms.select_tasks", return_value=tasks):
                return stage8_longer.run(ctx_for(d), sc, True)

    def test_advance_only_above_floor(self):
        ok = self._run([0.9, 0.8, 0.7], 0.5)
        self.assertEqual(ok.metrics["levels_reached"], 3)
        self.assertTrue(ok.gate_passed)
        stop = self._run([0.9, 0.2, 0.9], 0.5)
        self.assertEqual(stop.metrics["levels_reached"], 1)
        self.assertFalse(stop.gate_passed)
        self.assertIn("pass_rate_by_lines", stop.metrics)
        self.assertIsNotNone(stop.metrics["stopped"])


class SelfCorrect(unittest.TestCase):
    def test_baselines(self):
        self.assertEqual(sct.advantages([1, 0], ["a", "b"], "batch"), [0.5, -0.5])
        self.assertEqual(sct.advantages([1, 0, 1], ["a", "a", "b"], "loo"), [1.0, -1.0, 1 - 2 / 3])
        st = {}
        self.assertEqual(sct.advantages([1.0], ["a"], "ema", st, 0.5), [0.0])
        self.assertAlmostEqual(st["b"], 1.0)

    def test_example_format_and_rft_filter(self):
        ids, m = sct.make_example([1, 40, 41], [50, 51], 100)
        self.assertEqual(ids, [1, 40, 41, sct.SEP, 50, 51, EOS])
        self.assertEqual(m, [False] * 4 + [True] * 3)
        self.assertIsNone(sct.make_example([1] * 50, [2] * 60, 100))
        ro = [{"skipped": False, "reward": 1.0, "trace": [([1, 2], [5]), ([1, 3], [6])]},
              {"skipped": False, "reward": 0.0, "trace": [([1, 4], [7])]}, {"skipped": True, "reward": 0.0, "trace": []},
              {"skipped": False, "reward": 1.0, "trace": [([1, 2], [5])]}]
        self.assertEqual(len(sct.rft_examples(ro, 50, "last")), 2)   # failures dropped, duplicate ([1,3],[6]) vs ([1,2],[5]) kept once each
        self.assertEqual(len(sct.rft_examples(ro, 50, "all")), 2)

    def test_train_tasks_are_never_heldout(self):
        from training.microsuite import load_tasks
        held = load_tasks(max_difficulty=99)[0]
        recs = [{"id": "r1", "lang": "py", "prompt": "add", "code": "x=1", "tests": "assert 1", "source": "mbpp"},
                {"id": "r2", "lang": "py", "prompt": "sub", "code": "x=1", "tests": None}]
        self.assertEqual([t["id"] for t in sct.build_train_tasks(recs, "py", 5)], ["r1"])
        with self.assertRaises(AssertionError):
            sct.assert_not_heldout([{"id": held["id"], "prompt": "x"}])
        with self.assertRaises(AssertionError):
            sct.assert_not_heldout([{"id": "z", "prompt": held["prompt"]}])

    def test_starts_deterministic_and_corrupted(self):
        t = {"id": "a", "code": "def f(x):\n    return x + 1\n"}
        a = sct.make_start(t, 0, 3, [2], 0.0)
        self.assertEqual(a, sct.make_start(t, 0, 3, [2], 0.0))
        self.assertIsNone(sct.make_start(t, 0, 3, [1], 1.0))

    def test_checkpoint_adapter_reads_distill_layout(self):
        from backend.connectome.thinker import ThinkerConfig, build_model
        m = build_model("gru", ThinkerConfig(embed_dim=8, readout_rank=8, k_steps=1, gru_hidden=16))
        with tempfile.TemporaryDirectory() as d:
            p = sct.save_thinker(m, Path(d) / "c.pt")
            m2, blob = sct.load_thinker(p)
            self.assertEqual(blob["kind"], "gru")
            for a, b in zip(m.parameters(), m2.parameters()):
                self.assertTrue(torch.equal(a, b))
            torch.save({"kind": "gru", "cfg": m.cfg.__dict__, "state_dict": m.state_dict()}, Path(d) / "s.pt")
            sct.load_thinker(Path(d) / "s.pt")        # the state_dict spelling of eval_microsuite.load_policy

    def test_sft_and_reinforce_move_weights(self):
        from backend.connectome.thinker import ThinkerConfig, build_model
        from training import thinker_train_common as C
        m = build_model("gru", ThinkerConfig(embed_dim=8, readout_rank=8, k_steps=1, gru_hidden=16, tbptt=8))
        tcfg = {"lr": 0.05, "conn_lr": 0.05, "clip": 1.0, "batch": 4, "max_len": 40}
        items = [([1, 40, 41, sct.SEP, 50, EOS], [False] * 4 + [True] * 2)] * 3
        w0 = [p.detach().clone() for p in m.parameters()]
        opt = C.make_optimizer(m, 0.05, 0.05)
        r = sct.sft(m, opt, items, 5, 4, tcfg)
        self.assertEqual(r["steps"], 5)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(w0, m.parameters())))
        w1 = [p.detach().clone() for p in m.parameters()]
        sct.pg_step(m, opt, items[:2], [1.0, -1.0], tcfg)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(w1, m.parameters())))
        w2 = [p.detach().clone() for p in m.parameters()]
        sct.pg_step(m, C.make_optimizer(m, 0.05, 0.05), items[:2], [0.0, 0.0], tcfg)   # zero advantage: no gradient signal
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(w2, m.parameters())))

    def test_policy_rounds_and_trace(self):
        tok = BPETokenizer()
        calls = []

        class M:
            def generate(self, prompts, max_new, temperature=0.0):
                calls.append(len(prompts[0]))
                return torch.tensor([[40, 41, EOS, 0]])
        env = SimpleNamespace(rounds=0, editor=SimpleNamespace(text=""), task={"prompt": "p"}, _trace=[])
        pol = sct.SelfCorrectPolicy(M(), M(), tok)
        self.assertEqual(pol.act_env(env, [1, 2]), [40, 41, RUN])
        self.assertEqual(env._trace, [])
        env.rounds, env.editor.text = 1, "code"
        self.assertEqual(pol.act_env(env, [1, 2, 3]), [40, 41, RUN])
        self.assertEqual(env._trace, [([1, 2, 3], [40, 41])])
        self.assertEqual(sct.SelfCorrectPolicy(M(), None, tok).act_env(env, [1]), [EOS])

    def test_run_selfcorrect_rft_with_fake_rollouts(self):
        tasks = [{"id": f"t{i}", "lang": "py", "prompt": "p", "code": "x=1\n", "tests": ""} for i in range(2)]
        sc = {**sct.DEFAULT_SC, "iters": 2, "tasks_per_iter": 2, "samples": 1, "warmup_examples": 0, "eval_modes": ["scratch"],
              "ft_steps": 2, "batch": 2}
        fake_ro = [{"task": "t0", "reward": 1.0, "skipped": False, "trace": [([1, 40], [50])], "first_ok": False, "rounds": 1,
                    "from_scratch": False}]
        gain = {"mode": "scratch", "n": 2, "no_repair_pass_at_1": 0.0, "repair_pass_at_1": 0.5, "improvement": 0.5, "max_rounds": 3,
                "mean_repair_rounds": 1.0, "no_repair_by_lang": {}, "repair_by_lang": {}}
        from backend.connectome.thinker import ThinkerConfig, build_model
        m = build_model("gru", ThinkerConfig(embed_dim=8, readout_rank=8, k_steps=1, gru_hidden=16, tbptt=8))
        tcfg = {"lr": 0.01, "conn_lr": 0.01, "clip": 1.0, "max_len": 64}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(sct, "collect_rollouts", return_value=fake_ro), \
                mock.patch.object(sct, "eval_repair", return_value=gain):
            res = sct.run_selfcorrect(m, m, BPETokenizer(), tcfg, sc, d, tasks, [], log=lambda *_: None)
            self.assertTrue(Path(res["checkpoint"]).exists())
        self.assertEqual([h["examples"] for h in res["train"]["history"]], [1, 1])
        self.assertEqual(res["after"]["scratch"]["improvement"], 0.5)


class Stage9(unittest.TestCase):
    def test_gain_gate(self):
        self.assertTrue(stage9_selfcorrect.gain_ok(0.1, 0.0))
        self.assertFalse(stage9_selfcorrect.gain_ok(0.0, 0.0))
        self.assertFalse(stage9_selfcorrect.gain_ok(0.05, 0.1))
        self.assertFalse(stage9_selfcorrect.gain_ok(None, 0.0))

    def test_config_stages_6_9_load(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / "training" / "curriculum.json").read_text())
        for s in cfg["stages"]:
            if s["id"] >= 6:
                self.assertNotIn("stub", s)
                self.assertIn("smoke_stage", s)
        self.assertEqual(cfg["stages"][5]["smoke_stage"]["kind"], "gru")


if __name__ == "__main__":
    unittest.main()
