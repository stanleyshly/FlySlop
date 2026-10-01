import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.connectome import circuit_io  # noqa: E402

spec = importlib.util.spec_from_file_location("extract_thinker_circuit", ROOT / "scripts" / "extract_thinker_circuit.py")
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)

DATA = ROOT / "data" / "connectome"


def synthetic():
    """VP 1..6 -> CB 10..29 (chain of layers) -> DN 100..109, plus decoys."""
    rows = [{"bodyId": i, "superclass": "visual_projection", "class": None, "type": "LC", "somaSide": "R", "somaNeuromere": None, "status": "Traced"} for i in range(1, 7)]
    rows += [{"bodyId": i, "superclass": "cb_intrinsic", "class": "CX", "type": "X", "somaSide": "L", "somaNeuromere": None, "status": "Traced"} for i in range(10, 30)]
    rows += [{"bodyId": i, "superclass": "descending_neuron", "class": None, "type": "DN", "somaSide": "R", "somaNeuromere": None, "status": "Traced"} for i in range(100, 110)]
    ann = pd.DataFrame(rows)
    nt = pd.DataFrame({"body": [1, 10, 11, 12, 100], "consensus_nt": ["acetylcholine", "gaba", "glutamate", "histamine", "unclear"]})
    pre, post, w = [], [], []
    def e(a, b, x): pre.append(a); post.append(b); w.append(x)
    for v in range(1, 7):
        for c in range(10, 15): e(v, c, 5)          # VP -> layer1 (10..14)
    for a in range(10, 15):
        for b in range(15, 20): e(a, b, 4)          # layer1 -> layer2
    for a in range(15, 20):
        for d in range(100, 108): e(a, d, 6)        # layer2 -> DN 100..107
    for c in range(21, 30): e(c, 20, 9)             # decoys feed only cb 20 (no path to DN)
    e(20, 21, 1)
    return ann, nt, (np.array(pre, np.int64), np.array(post, np.int64), np.array(w, np.int64))


CFG = {"n_vp": 4, "n_cb": 10, "n_dn": 6, "min_weight": 2, "topk_in": 50, "seed": 3}


class SyntheticTests(unittest.TestCase):
    def test_deterministic(self):
        ann, nt, edges = synthetic()
        a1, r1 = ext.build_circuit(ann, nt, edges, CFG)
        a2, r2 = ext.build_circuit(ann.sample(frac=1, random_state=1), nt, edges, CFG)
        self.assertEqual(circuit_io.content_hash(a1), circuit_io.content_hash(a2))
        self.assertEqual(r1, r2)

    def test_seed_changes_sample(self):
        ann, nt, edges = synthetic()
        a1, _ = ext.build_circuit(ann, nt, edges, CFG)
        a2, _ = ext.build_circuit(ann, nt, edges, {**CFG, "seed": 4})
        self.assertNotEqual(a1["body_id"][a1["idx_VP"]].tolist(), a2["body_id"][a2["idx_VP"]].tolist())

    def test_counts_disjoint_and_reach(self):
        ann, nt, edges = synthetic()
        a, r = ext.build_circuit(ann, nt, edges, CFG)
        self.assertEqual([r["populations"][k]["neurons"] for k in ext.ROLES], [4, 10, 6])
        idx = [set(a[f"idx_{k}"].tolist()) for k in ext.ROLES]
        self.assertFalse(idx[0] & idx[1] or idx[0] & idx[2] or idx[1] & idx[2])
        self.assertEqual(len(set(a["body_id"].tolist())), len(a["body_id"]))
        self.assertEqual(sum(map(len, idx)), len(a["body_id"]))
        # decoy cb neurons (no DN path) are not chosen ahead of layer1/2
        chosen = set(a["body_id"][a["idx_CB"]].tolist())
        self.assertTrue(set(range(10, 20)) <= chosen)
        self.assertGreaterEqual(r["dn_reachable_frac_by_k"]["3"], 1.0)
        self.assertLess(r["dn_reachable_frac_by_k"]["1"], 1.0)

    def test_csr_and_signs(self):
        ann, nt, edges = synthetic()
        a, _ = ext.build_circuit(ann, nt, edges, {**CFG, "topk_in": 3})
        n = len(a["body_id"])
        self.assertEqual(len(a["indptr"]), n + 1)
        self.assertEqual(a["indptr"][-1], len(a["pre"]))
        self.assertTrue((np.diff(a["indptr"]) <= 3).all())
        self.assertTrue((np.diff(a["post"]) >= 0).all())
        self.assertTrue((a["weight"] >= CFG["min_weight"]).all())
        by = dict(zip(a["body_id"].tolist(), zip(a["sign"].tolist(), a["sign_code"].tolist())))
        self.assertEqual(by[1], (1, 0)); self.assertEqual(by[10], (-1, 0)); self.assertEqual(by[11], (-1, 0))
        self.assertEqual(by[12], (1, 1)); self.assertEqual(by[100], (1, 2))


class ArtifactTests(unittest.TestCase):
    RANGES = {"small": (4500, 5500, 500_000, 2_000_000), "medium": (15_000, 20_000, 500_000, 2_000_000)}

    def test_artifacts(self):
        for name, (nlo, nhi, elo, ehi) in self.RANGES.items():
            path = DATA / f"thinker_circuit_{name}.npz"
            if not path.exists():
                self.skipTest(f"{path.name} not extracted")
            a, meta, digest = circuit_io.load_circuit(path)
            summary = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(summary["content_hash"], digest)
            n = len(a["body_id"])
            self.assertTrue(nlo <= n <= nhi, (name, n))
            self.assertTrue(elo <= len(a["pre"]) <= ehi, (name, len(a["pre"])))
            self.assertLess(path.stat().st_size, 20 * 2**20)
            idx = [set(a[f"idx_{k}"].tolist()) for k in ext.ROLES]
            self.assertEqual(sum(map(len, idx)), n)
            self.assertEqual(len(idx[0] | idx[1] | idx[2]), n)
            self.assertEqual(len(np.unique(a["body_id"])), n)
            self.assertEqual(a["indptr"][-1], len(a["pre"]))
            self.assertGreaterEqual(ext.reachability(a)["4"]["DN"], 0.9)
            self.assertGreaterEqual(summary["dn_reachable_k4"], 0.9)


if __name__ == "__main__":
    unittest.main()
