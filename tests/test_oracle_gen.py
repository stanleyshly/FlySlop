import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training import datasets as ds
from training import oracle_gen as og

GOOD_PY = "def add(a, b):\n    return a + b\n"
GOOD_SV = ("module m(input logic clk, input logic d, output logic q);\n"
           "  always_ff @(posedge clk) q <= d;\nendmodule\n")


class FakeTeacher:
    model_id = "fake/teacher"

    def __init__(self, replies=None):
        self.replies, self.max_tokens, self.calls = list(replies or []), 64, []
        self.last_stats = {"new_tokens": 10, "seconds": 0.1, "tokens_per_s": 100.0}

    def generate(self, prompt):
        self.calls.append(prompt)
        return self.replies.pop(0) if self.replies else "```python\n" + GOOD_PY + "```"


def ok_judge(lang, code, tests="", **kw):
    if lang == "py":
        try:
            compile(code, "x", "exec")
        except SyntaxError as e:
            return {"ok": False, "stage": "parse", "error": str(e)}
        if "FAIL" in tests:
            return {"ok": False, "stage": "test", "error": "assert"}
        return {"ok": True, "stage": "test"}
    return {"ok": "endmodule" in code, "stage": "elab", "error": ""}


class ExtractTests(unittest.TestCase):
    def test_fenced_tagged_preferred(self):
        t = "text\n```bash\nls\n```\nmore\n```python\ndef f():\n    return 1\n```\n"
        code, info = og.extract_code(t, "py")
        self.assertEqual(code, "def f():\n    return 1")
        self.assertFalse(info["truncated"])

    def test_unterminated_fence_flagged(self):
        code, info = og.extract_code("```verilog\nmodule a;\nendmodu", "sv")
        self.assertTrue(info["truncated"])
        self.assertIn("module a;", code)

    def test_raw_code_and_none(self):
        self.assertEqual(og.extract_code("def g(x):\n    return x", "py")[0], "def g(x):\n    return x")
        self.assertIsNone(og.extract_code("I cannot help with that.", "py")[0])

    def test_think_removed_and_longest_block(self):
        t = "<think>```python\nx=1\n```</think>```python\na=1\n```\n```python\ndef long():\n    pass\n```"
        self.assertIn("def long", og.extract_code(t, "py")[0])

    def test_spec_rejects_code(self):
        self.assertIsNone(og.clean_spec("Here:\n```python\nx\n```"))
        self.assertIsNone(og.clean_spec("short"))
        self.assertTrue(og.clean_spec("Specification: Write add(a, b) that returns their sum.").startswith("Write"))


class UnitTests(unittest.TestCase):
    def test_py_and_sv_units(self):
        src = "import os\n\ndef helper_function(a, b):\n    total = a + b\n    return total * 2 + len(str(total)) + 1\n"
        u = og.py_units(src)
        self.assertEqual([x["name"] for x in u], ["helper_function"])
        sv = og.sv_units("module m(input logic clk, input logic d, output logic q);\n  always_ff @(posedge clk) q <= d;\nendmodule\n")
        self.assertEqual(sv[0]["name"], "m")

    def test_template_hash_stable(self):
        self.assertEqual(len(og.TEMPLATE_HASH), 16)


class FilterTests(unittest.TestCase):
    def test_chain(self):
        f = og.Filters(judge=ok_judge)
        task = {"lang": "py", "tests": ""}
        r = f.run("```python\n" + GOOD_PY + "```", task)
        self.assertTrue(r["passed"])
        f.register(r["code"], "py")
        self.assertEqual(f.run("```python\n" + GOOD_PY + "```", task)["fail_stage"], "dup")
        self.assertEqual(f.run("no code here", task)["fail_stage"], "extract")
        self.assertEqual(f.run("```python\ndef broken_function(a, b:\n    return a\n```", task)["fail_stage"], "syntax")
        self.assertEqual(f.run("```python\ndef ok_long_enough_name():\n    return 1\n```", {"lang": "py", "tests": "FAIL"})["fail_stage"], "test")
        self.assertEqual(f.run("```python\ndef a():\n    pass\n" + "#" * 4000 + "\n```", task)["fail_stage"], "length")

    def test_ascii_normalisation(self):
        f = og.Filters(judge=ok_judge)
        r = f.run("```python\ndef f():\n\treturn 'é'  # café\n```", {"lang": "py", "tests": ""})
        self.assertTrue(r["passed"])
        self.assertTrue(all(c == "\n" or 32 <= ord(c) < 127 for c in r["code"]))

    def test_neardup_vs_test_split(self):
        tr = [{"lang": "py", "code": GOOD_PY}]
        f = og.Filters(tr, judge=ok_judge)
        self.assertEqual(f.run("```python\n" + GOOD_PY + "```", {"lang": "py"})["fail_stage"], "neardup")
        self.assertTrue(f.run("```python\n" + GOOD_PY + "```", {"lang": "py"}, check_test_neardup=False)["passed"])

    def test_real_sv_judge(self):
        f = og.Filters()
        self.assertTrue(f.run("```systemverilog\n" + GOOD_SV + "```", {"lang": "sv", "tests": ""})["passed"])
        bad = f.run("```systemverilog\nmodule m(input a, output b;\n  assign b = a;\nendmodule\n```", {"lang": "sv", "tests": ""})
        self.assertEqual(bad["fail_stage"], "syntax")


def _pool():
    return [
        ds.record("t1", "mbpp", "py", "Write a function add(a, b) returning the sum of two numbers.", GOOD_PY, "assert add(1,2)==3"),
        ds.record("t2", "mbpp", "py", "Write a function that multiplies two numbers together.", "def mul(a, b):\n    return a * b\n", "assert mul(2,3)==6"),
        ds.record("t3", "mbpp", "py", "Write a function that negates a number and returns it.", "def neg(a):\n    return -a\n", "assert neg(1)==-1"),
    ]


class RunTests(unittest.TestCase):
    def _args(self, d, **kw):
        base = dict(run="t", out_dir=str(d), sources="mbpp", limit=None, time_budget_s=None, model="fake", standin=False,
                    temperature=0.0, seed=3, holdout_sources="", spec_tokens=32, code_tokens=32, judge_timeout=5.0,
                    max_ram_gb=None, quiet=True)
        base.update(kw)
        return argparse.Namespace(**base)

    def _run(self, d, teacher, **kw):
        with mock.patch.object(ds, "load_all", return_value=_pool()):
            return og.run(self._args(d, **kw), teacher=teacher, judge=ok_judge)

    def test_resume_manifest_and_split(self):
        with tempfile.TemporaryDirectory(dir=ds.PRIVATE) as d:
            d = Path(d)
            t1 = FakeTeacher(["```python\ndef add(a, b):\n    return a + b\n```",
                              "```python\ndef mul(a, b):\n    return a * b\n```"])
            m1 = self._run(d, t1, limit=2)
            self.assertEqual(m1["counts"]["attempts"], 2)
            ids1 = [json.loads(l)["id"] for l in (d / "pairs.jsonl").read_text().splitlines()]
            t2 = FakeTeacher()
            m2 = self._run(d, t2)                       # resume: only the third id is generated
            self.assertEqual(len(t2.calls), 1)
            ids2 = [json.loads(l)["id"] for l in (d / "pairs.jsonl").read_text().splitlines()]
            self.assertEqual(ids2[:2], ids1)
            self.assertEqual(len(set(ids2)), 3)
            man = json.loads((d / "manifest.json").read_text())
            for k in ("teacher_model", "template_hash", "seed", "counts", "stage_table", "split_hash"):
                self.assertIn(k, man)
            self.assertEqual(man["seed"], 3)
            self.assertEqual(man["template_hash"], og.TEMPLATE_HASH)
            row = json.loads((d / "pairs.jsonl").read_text().splitlines()[0])
            for k in ("teacher", "licence", "provenance", "filters", "split", "private", "passed"):
                self.assertIn(k, row)
            self.assertEqual(row["teacher"], "fake/teacher")
            self.assertEqual(sum(m2["split_counts"].values()), m2["counts"]["passed"])

    def test_dedup_in_run(self):
        with tempfile.TemporaryDirectory(dir=ds.PRIVATE) as d:
            same = "```python\ndef add(a, b):\n    return a + b\n```"
            m = self._run(Path(d), FakeTeacher([same, same, same]), holdout_sources="")
            self.assertLessEqual(m["counts"]["passed"], 1)
            self.assertGreaterEqual(m["counts"]["fail_stage"].get("dup", 0), 1)

    def test_time_budget_and_private_guard(self):
        with tempfile.TemporaryDirectory(dir=ds.PRIVATE) as d:
            m = self._run(Path(d), FakeTeacher(), time_budget_s=-1)
            self.assertEqual(m["stopped"], "time_budget")
            self.assertEqual(m["counts"]["attempts"], 0)
        with self.assertRaises(SystemExit):
            self._run(Path(tempfile.gettempdir()) / "oracle_x", FakeTeacher())

    def test_estimate(self):
        e = og.estimate(26.0, 300.0, pass_rate=0.5)
        self.assertEqual(e["200_pairs"]["attempts"], 400)
        self.assertGreater(e["10000_pairs"]["hours"], e["200_pairs"]["hours"])


if __name__ == "__main__":
    unittest.main()
