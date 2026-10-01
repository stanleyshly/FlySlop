"""P6b: distillation + edit-token trainer (GRU kind only; connectome runs are too slow for unit tests)."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import keyplan  # noqa: E402
from backend.connectome.thinker import BOS, EOS, PAD, SEP  # noqa: E402
from backend.editor import Editor  # noqa: E402
from training import distill, edit_trainer as E  # noqa: E402
from training import thinker_train_common as C  # noqa: E402

TOK = ROOT / "data/corpus/tokenizer/bpe.json"
PAIRS = [{"id": f"p{i}", "source": "t", "lang": "py", "prompt": f"add {i} to x", "code": f"def f(x):\n    return x + {i}\n",
          "tests": None, "origin": "dataset"} for i in range(12)]
DATA = {"train": PAIRS[:10], "val": PAIRS[10:], "test": [], "label": "synthetic"}


def tiny_cfg():
    cfg = C.load_cfg(None, {"model.k_steps": 2, "train.batch": 4, "train.max_len": 64, "train.eval_pairs": 2,
                            "train.gen_pairs": 0, "train.eval_every": 0, "train.ckpt_every": 0,
                            "train.log_every": 100, "train.warmup": 1})
    return cfg


@unittest.skipUnless(TOK.exists(), "tokenizer missing")
class DistillTests(unittest.TestCase):
    def test_schedule_phases(self):
        s = {"a_frac": 0.4, "b_frac": 0.4, "b_levels": [0.25, 0.5, 0.75, 0.95]}
        seen = [C.schedule_p(i, 100, s) for i in range(100)]
        self.assertEqual({p for ph, p in seen if ph == "A"}, {0.0})
        self.assertEqual(sorted({p for ph, p in seen if ph == "B"}), [0.25, 0.5, 0.75, 0.95])
        self.assertEqual({p for ph, p in seen if ph == "C"}, {1.0})
        self.assertEqual([ph for ph, _ in seen], sorted(ph for ph, _ in seen))

    def test_gated_schedule_thresholds_caps_and_state(self):
        s = {"mode": "gated", "a_frac": 0.5, "b_frac": 0.4, "b_levels": [0.25, 0.5], "window": 3, "min_steps": 3,
             "a_thresh": 0.9, "b_floor": 0.8}
        g = C.make_schedule(s, 100)
        self.assertIsInstance(g, C.GatedSchedule)
        evs = [e for i in range(1, 4) for e in g.observe(i, 0.95)]
        self.assertEqual([(e["from"], e["to"], e["reason"]) for e in evs], [("A", "B", "threshold")])
        self.assertEqual(g.current(3), ("B", 0.25))
        h = C.make_schedule(s, 100)
        h.load(json.loads(json.dumps(g.dump())))
        self.assertEqual(h.current(3), ("B", 0.25))
        ev = [e for i in range(4, 44) for e in g.observe(i, 0.1)]      # never competent: level cap (20 steps)
        self.assertEqual([e["reason"] for e in ev], ["step_cap", "step_cap"])
        self.assertEqual(g.current(44), ("C", 1.0))
        self.assertEqual(g.observe(45, 0.1), [])
        pin = C.make_schedule({"a_frac": 1.0, "b_frac": 0.0, "b_levels": [0.5]}, 10)
        self.assertEqual([e for i in range(1, 10) for e in pin.observe(i, 1.0)], [])
        self.assertIsInstance(C.make_schedule({**s, "mode": "fixed"}, 100), C.FixedSchedule)

    def test_parallel_mix_and_lambda_step(self):
        cfg = tiny_cfg()
        model, _ = C.build("gru", cfg, 0)
        seq = torch.randint(20, 100, (3, 12))
        mask = torch.zeros(3, 12, dtype=torch.bool)
        mask[:, 6:] = True
        g = torch.Generator().manual_seed(0)
        x = seq[:, :-1]
        preds = torch.randint(20, 100, x.shape)
        same, f0 = C.mix_parallel(x, mask[:, 1:], preds, 0.0, g)
        self.assertTrue(torch.equal(same, x))
        full, f1 = C.mix_parallel(x, mask[:, 1:], preds, 1.0, g)
        self.assertEqual(f1, 1.0)
        self.assertTrue(torch.equal(full[:, :6], x[:, :6]))
        self.assertTrue(torch.equal(full[:, 6:], preds[:, 5:-1]))
        opt = C.make_optimizer(model, 0.01, 0.01)
        for mode in ("parallel", "sequential"):
            loss, acc, frac, tf = C.train_step(model, opt, seq, mask, 0.5, g, 8, 1.0, 1.0, mode=mode, tf_lambda=0.5)
            self.assertTrue(loss == loss and tf is not None and 0 <= frac <= 1)

    def test_pair_encoding_masks_only_code(self):
        tok = distill._tok(tiny_cfg())
        ids, m = distill.encode_pair(tok, "hello", "x = 1\n", 64)
        self.assertEqual(ids[0], BOS)
        self.assertEqual(ids[-1], EOS)
        k = ids.index(SEP)
        self.assertFalse(any(m[:k + 1]))
        self.assertTrue(all(m[k + 1:]))
        self.assertIsNone(distill.encode_pair(tok, "hello", "x = 1\n" * 50, 64))

    def test_pad_left_and_mix(self):
        seq, mask = C.pad_batch([([1, 5, 6, 2], [False, False, True, True]), ([1, 2], [False, True])])
        self.assertEqual(seq[1].tolist(), [PAD, PAD, 1, 2])
        cfg = tiny_cfg()
        model, _ = C.build("gru", cfg, 0)
        x = torch.randint(20, 100, (3, 10))
        m = torch.zeros(3, 10, dtype=torch.bool)
        m[:, 5:] = True
        g = torch.Generator().manual_seed(0)
        same, f0 = C.mix_inputs(model, x, m, 0.0, g)
        self.assertTrue(torch.equal(same, x))
        mixed, f1 = C.mix_inputs(model, x, m, 1.0, g)
        self.assertEqual(f1, 1.0)
        self.assertTrue(torch.equal(mixed[:, :5], x[:, :5]))

    def test_controls_share_one_config_and_resume(self):
        cfg = tiny_cfg()
        with tempfile.TemporaryDirectory() as d:
            res = distill.run_controls(cfg, ["gru", "gru"], [0, 1], d, DATA, 4)
            self.assertEqual(len({r["cfg_hash"] for r in res["rows"]}), 1)
            self.assertEqual(res["cfg_hash"], distill.cfg_fingerprint(cfg))
            self.assertTrue((Path(d) / "controls.md").exists())
            # resume: 6 straight steps == 3 + resume to 6 (batches and mixing depend only on seed, step)
            a = distill.train_kind(cfg, "gru", 0, Path(d) / "a", DATA, 6, log=lambda *_: None)

            class Stop:                       # interrupt the 6-step run after 3 steps (checkpoint is saved on exit)
                n = 0
                def tick(self, k=1):
                    self.n += k
                    if self.n >= 3:
                        raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                distill.train_kind(cfg, "gru", 0, Path(d) / "b", DATA, 6, ctx=Stop(), log=lambda *_: None)
            b = distill.train_kind(cfg, "gru", 0, Path(d) / "b", DATA, 6, resume=Path(d) / "b" / "ckpt.pt",
                                   log=lambda *_: None)
            rows = lambda n: [r for r in map(json.loads, (Path(d) / n / "schedule.jsonl").read_text().splitlines())
                              if "event" not in r]
            la, lb = rows("a"), rows("b")
            self.assertEqual(a["steps"], b["steps"])
            self.assertAlmostEqual(la[-1]["loss"], lb[-1]["loss"], places=4)
            self.assertEqual(len(la), 6)
            self.assertEqual([r["phase"] for r in la][0], "A")
            self.assertTrue(all("reason" in json.loads(l) for l in (Path(d) / "a" / "schedule.jsonl").read_text()
                                .splitlines() if '"event"' in l))

    def test_wall_budget_stops(self):
        cfg = tiny_cfg()
        cfg["train"]["max_seconds"] = 1e-9
        with tempfile.TemporaryDirectory() as d:
            r = distill.train_kind(cfg, "gru", 0, d, DATA, 50, log=lambda *_: None)
            self.assertEqual(r["stopped"], "wall")
            self.assertLess(r["steps"], 5)


@unittest.skipUnless(TOK.exists(), "tokenizer missing")
class EditTests(unittest.TestCase):
    def test_oracle_examples_round_trip_and_markers(self):
        cfg = tiny_cfg()
        tok = distill._tok(cfg)
        ecfg = cfg["edit"]
        code = "def add(a, b):\n    return a + b\n"
        n = 0
        for stage in (3, 4):
            for i in range(30):
                ex = E.make_example(tok, code, i, stage, ecfg, 256)
                if ex is None:
                    continue
                n += 1
                ed = Editor(ex["buffer"], ex["cursor"], ex["anchor"])
                keyplan.apply_tokens(ed, ex["act"], tok)
                self.assertEqual(ed.text, ex["target"])
                self.assertEqual(ex["ids"][len(ex["state"])], SEP)
                self.assertEqual(ex["ids"].count(E.CUR), 1)
        self.assertGreater(n, 30)

    def test_edit_metrics_with_perfect_oracle_model(self):
        cfg = tiny_cfg()
        tok = distill._tok(cfg)
        exs = [e for e in (E.make_example(tok, "x = foo(1, 2)\ny = 3\n", i, 3, cfg["edit"], 256) for i in range(10)) if e]

        class Oracle:
            def eval(self): return self
        orig = E.C.generate_lists
        table = {tuple(e["state"]): e["act"] for e in exs}
        E.C.generate_lists = lambda m, prompts, mx, batch=16: [table.get(tuple(p), []) for p in prompts]
        try:
            m = E.edit_metrics(Oracle(), tok, exs, cfg["edit"])
        finally:
            E.C.generate_lists = orig
        self.assertEqual(m["first_round_ok"], 1.0)

    def test_edit_trainer_smoke(self):
        cfg = tiny_cfg()
        cfg["edit"].update({"train_examples": 24, "val_examples": 6})
        with tempfile.TemporaryDirectory() as d:
            r = E.train_edit(cfg, "gru", 0, d, 4, stage=3, log=lambda *_: None)
            self.assertIn("first_round_ok", r["final"])
            self.assertEqual(r["steps"], 4)


if __name__ == "__main__":
    unittest.main()
