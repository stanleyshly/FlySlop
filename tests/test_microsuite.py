import os
import unittest

from backend.error_inject import corrupt_buffer
from training.eval_microsuite import evaluate, repair_gain
from training.microsuite import dup_check, load_tasks, split_hash
from training.selfcorrect_env import NullPolicy, OracleRepairPolicy, SelfCorrectEnv
from training.tokenizer import BPETokenizer, EOS

TOK = BPETokenizer()


class TypeText:
    """Duck-typed policy: types a fixed program then <RUN> on every round."""

    def __init__(self, text):
        self.ids = TOK.encode(text) + [3]

    def act(self, ctx):
        return list(self.ids)


class SuiteTests(unittest.TestCase):
    def test_counts_and_metadata(self):
        ts = load_tasks()
        self.assertGreaterEqual(len([t for t in ts if t["lang"] == "py"]), 50)
        self.assertGreaterEqual(len([t for t in ts if t["lang"] == "sv"]), 30)
        self.assertEqual(len({t["id"] for t in ts}), len(ts))
        for t in ts:
            self.assertIn(t["difficulty"], (1, 2, 3))
            self.assertGreater(t["lines"], 0)
            self.assertIn("MIT", t["licence"])
        self.assertEqual(split_hash(), split_hash())
        self.assertEqual(len(load_tasks(max_difficulty=1)), len([t for t in ts if t["difficulty"] == 1]))

    def test_no_literal_overlap_with_training_records(self):
        d = dup_check()
        self.assertEqual(d["n_literal"], 0, d["literal_overlaps"][:5])

    def test_python_oracle_and_null(self):
        py = load_tasks("py")
        r = evaluate(OracleRepairPolicy(), py)
        self.assertEqual(r["pass_at_1"], 1.0, [x["id"] for x in r["rows"] if not x["ok"]])
        n = evaluate(NullPolicy(), py)
        self.assertEqual(n["pass_at_1"], 0.0)


class LoopTests(unittest.TestCase):
    def test_python_repair_from_corrupted(self):
        py = load_tasks("py")
        r = evaluate(OracleRepairPolicy(), py, max_rounds=3, mode="corrupt", n_errors=2)
        self.assertEqual(r["pass_at_1"], 1.0, [x["id"] for x in r["rows"] if not x["ok"]])
        self.assertGreater(r["repair_gain"], 0.0)
        self.assertGreaterEqual(r["first_run_pass"], 0.0)

    def test_repair_gain_metric(self):
        g = repair_gain(OracleRepairPolicy(), load_tasks("py")[:10], mode="corrupt", n_errors=2)
        self.assertEqual(g["repair_pass_at_1"], 1.0)
        self.assertGreater(g["improvement"], 0.0)

    def test_python_infinite_loop_terminates(self):
        task = dict(load_tasks("py")[0])
        env = SelfCorrectEnv(TOK, max_rounds=2, run_timeout=2.0)
        info = env.run_episode(TypeText("def five():\n    while True:\n        pass\n"), task)
        self.assertFalse(info["ok"])
        self.assertEqual(info["reason"], "max_rounds")
        self.assertLess(info["elapsed"], 30)
        self.assertEqual(info["n_runs"], 3)

    def test_token_and_buffer_caps(self):
        class Spam:
            def act(self, ctx):
                return [ord("a") + 0 - 0 + 0] * 0 + TOK.encode("x" * 100)
        env = SelfCorrectEnv(TOK, max_rounds=5, max_tokens_per_round=50, max_buffer_chars=120)
        info = env.run_episode(Spam(), dict(load_tasks("py")[0]))
        self.assertLessEqual(len(info["buffer"]), 120)
        self.assertEqual(info["rounds"], 6)

    def test_episode_budget(self):
        env = SelfCorrectEnv(TOK, max_rounds=50, episode_timeout=0.0)
        info = env.run_episode(TypeText("pass\n"), dict(load_tasks("py")[0]))
        self.assertEqual(info["reason"], "budget")

    def test_context_contains_prompt_buffer_output(self):
        env = SelfCorrectEnv(TOK, max_rounds=1)
        t = load_tasks("py")[0]
        ctx = env.reset(t, start_buffer="def five():\n    return 6\n")
        text = TOK.decode(ctx)
        self.assertIn(t["prompt"][:20], text)
        self.assertIn("return 6", text)
        self.assertIn("FAIL", text)

    @unittest.skipUnless(os.environ.get("FLYSLOP_SLOW_TESTS") or True, "slow")
    def test_verilator_hang_terminates(self):
        t = load_tasks("sv")[0]
        hang_tb = t["tests"].replace("$finish;", "forever #1;")   # never finishes: sim must be killed
        task = dict(t, tests=hang_tb)
        env = SelfCorrectEnv(TOK, max_rounds=0, run_timeout=25.0)
        info = env.run_episode(OracleRepairPolicy(), task)
        self.assertFalse(info["ok"])
        self.assertLess(info["elapsed"], 60)

    def test_sv_oracle_repair_sample(self):
        sv = load_tasks("sv")[::12][:3]
        r = evaluate(OracleRepairPolicy(), sv, max_rounds=3, mode="corrupt", n_errors=2, run_timeout=30.0)
        self.assertEqual(r["pass_at_1"], 1.0, [x["id"] for x in r["rows"] if not x["ok"]])


if __name__ == "__main__":
    unittest.main()
