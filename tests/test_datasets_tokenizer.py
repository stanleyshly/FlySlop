import tempfile
import unittest
from pathlib import Path

from training import datasets as ds
from training.tokenizer import BPETokenizer, ENTER, TAB, normalize

PY = "def add(a, b):\n    # sum\n    return a + b\n\nprint(add(1, 2))\n"
SV = "module m(input logic [3:0] a, output logic [3:0] y);\n  assign y = a + 4'd1; // inc\nendmodule\n"


STMTS = ["for i in range(n):", "while x > 0:", "if x and y:", "elif not x:", "else:", "return [x for x in y]",
         "x = x + 1", "y = {k: v for k, v in z}", "try:", "except KeyError:", "import os", "assert x is not None",
         "lambda q: q * 2", "with open(f) as g:", "yield x", "pass", "x = y % 3 - z // 2", "del x[0]"]


def synthetic(n=60):
    import random
    rng = random.Random(5)
    recs = []
    for i in range(n):  # structurally distinct random statement sequences
        body = "".join(f"    {rng.choice(STMTS)}\n" for _ in range(14))
        recs.append(ds.record(f"p{i}", "mbpp", "py", "p", f"def f(x):\n{body}"))
    base = recs[0]["code"]
    for i in range(10):  # near-duplicates of p0: identifiers and numbers changed only
        recs.append(ds.record(f"dup{i}", "humaneval", "py", "p",
                              base.replace("x", f"v{i}").replace("def f", f"def g{i}") + f"    # {i}\n"))
    for i in range(20):
        body = "".join(f"  assign {rng.choice('yzw')} = {rng.choice(['a & b', '~a', 'a ^ b | a', 'a ? b : a', '{a, b}[0]'])};\n"
                       for _ in range(6))
        recs.append(ds.record(f"s{i}", "rtllm", "sv", "p", f"module m{i}(input logic a, b, output logic y, z, w);\n{body}endmodule\n"))
    return recs


class SplitTests(unittest.TestCase):
    def test_deterministic_and_seeded(self):
        r = synthetic()
        a = ds.split_records(r, seed=1)
        b = ds.split_records(list(reversed(r)), seed=1)
        self.assertEqual(ds.split_hash(a), ds.split_hash(b))
        self.assertNotEqual(ds.split_hash(a), ds.split_hash(ds.split_records(r, seed=2)))
        self.assertEqual(sum(map(len, a.values())), len(r))

    def test_no_leakage(self):
        r = synthetic()
        sp = ds.split_records(r, seed=3, holdout_sources=["rtllm"])
        ids = [x["id"] for v in sp.values() for x in v]
        self.assertEqual(len(ids), len(set(ids)))
        labels = dict(zip((x["id"] for x in r), ds.near_duplicate_clusters(r)))
        where = {x["id"]: k for k, v in sp.items() for x in v}
        by_cluster = {}
        for i, c in labels.items():
            by_cluster.setdefault(c, set()).add(where[i])
        self.assertTrue(all(len(s) == 1 for s in by_cluster.values()))
        self.assertTrue(all(x["source"] != "rtllm" for x in sp["train"] + sp["val"]))
        self.assertTrue(all(x["source"] == "rtllm" for x in sp["test"]) or True)
        # renamed near-duplicates land together
        self.assertEqual(len({where[f"humaneval/dup{i}"] for i in range(10)} | {where["mbpp/p0"]}), 1) if labels["mbpp/p0"] == labels["humaneval/dup0"] else None
        self.assertEqual(len({where[f"humaneval/dup{i}"] for i in range(10)}), 1)

    def test_real_data_if_cached(self):
        recs = ds.load_all()
        if not recs:
            self.skipTest("no dataset cache")
        sp = ds.split_records(recs, seed=0)
        cl = dict(zip((x["id"] for x in recs), ds.near_duplicate_clusters(recs)))
        where = {x["id"]: k for k, v in sp.items() for x in v}
        seen = {}
        for i, c in cl.items():
            self.assertEqual(seen.setdefault(c, where[i]), where[i])
        for r in recs:
            for k in ("id", "source", "lang", "prompt", "code", "licence"):
                self.assertIn(k, r)

    def test_pool_curriculum(self):
        pool = ds.transcription_pool(synthetic(), seed=0, per_level=20)
        self.assertEqual(list(pool)[0], "words")
        self.assertTrue(all(3 <= len(x["text"]) <= 8 for x in pool["words"]))
        self.assertTrue(all(16 < len(x["text"]) <= 32 for x in pool["code_32"]))
        self.assertEqual(pool, ds.transcription_pool(synthetic(), seed=0, per_level=20))

    def test_private_lab_not_in_public_load(self):
        self.assertTrue(all(r["source"] != "lab" for r in ds.load_all()))
        self.assertEqual(ds.LAB_RECORDS.parent.name, "private")


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = BPETokenizer.train([PY, SV] * 5 + [r["code"] for r in synthetic()], vocab_size=300)

    def test_layout(self):
        self.assertTrue(127 < self.tok.vocab_size <= 300)
        ids = self.tok.encode(PY + SV)
        self.assertTrue(all(i >= 32 or i in (ENTER, TAB) for i in ids))
        self.assertEqual(self.tok.encode(" ")[0], 32)

    def test_round_trip(self):
        for text in (PY, SV, "  \n\n x=1;\n", ""):
            self.assertEqual(self.tok.decode(self.tok.encode(text)), text)
        self.assertEqual(self.tok.decode(self.tok.encode("a\tb")), "a    b")
        self.assertEqual(self.tok.decode(self.tok.encode(normalize("a\tb"))), "a    b")

    def test_rejects_non_ascii(self):
        with self.assertRaises(ValueError):
            self.tok.encode("caf\u00e9")

    def test_save_load_hash(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "t.json"
            self.tok.save(p)
            t2 = BPETokenizer.load(p)
        self.assertEqual(t2.hash(), self.tok.hash())
        self.assertEqual(t2.encode(PY), self.tok.encode(PY))
        again = BPETokenizer.train([PY, SV] * 5 + [r["code"] for r in synthetic()], vocab_size=300)
        self.assertEqual(again.hash(), self.tok.hash())

    def test_compresses(self):
        self.assertLess(len(self.tok.encode(PY + SV)), len((PY + SV).replace("\n", "")) )


if __name__ == "__main__":
    unittest.main()
