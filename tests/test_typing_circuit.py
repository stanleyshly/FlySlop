import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.connectome import circuit_io  # noqa: E402

spec = importlib.util.spec_from_file_location("extract_typing_circuit", ROOT / "scripts" / "extract_typing_circuit.py")
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)

NPZ = ROOT / "data" / "connectome" / "typing_circuit.npz"
JSON = NPZ.with_suffix(".json")
MAP = ROOT / "data" / "connectome" / "mn_joint_map.json"
MODEL = ROOT / "data" / "neuromechfly" / "model.json"


def synthetic():
    rows = []
    def add(i, sc, **kw):
        rows.append({"bodyId": i, "superclass": sc, "class": kw.get("cls"), "subclass": kw.get("sub"), "type": kw.get("type"),
                     "somaSide": kw.get("side", "R"), "somaNeuromere": "T1", "entryNerve": kw.get("entry"),
                     "exitNerve": kw.get("exit"), "status": "Traced"})
    add(1, "vnc_motor", type="Ti flexor MN", exit="ProLN", sub="fl")
    add(2, "vnc_motor", type="Ti extensor MN", exit="ProLN", sub="fl", side="L")
    for i in range(10, 16):
        add(i, "vnc_intrinsic")
    add(20, "descending_neuron"); add(21, "descending_neuron")
    add(30, "vnc_sensory", cls="mechanosensory_tactile", entry="ProLN")
    ann = pd.DataFrame(rows)
    nt = pd.DataFrame({"body": [1, 2, 10, 11, 12, 20, 30, 13],
                       "consensus_nt": ["acetylcholine", "acetylcholine", "gaba", "glutamate", "unclear", "acetylcholine", "histamine", "acetylcholine"]})
    pre = np.array([10, 11, 12, 13, 14, 20, 21, 30, 1, 15], np.int64)
    post = np.array([1, 1, 2, 1, 10, 10, 11, 12, 10, 13], np.int64)
    w = np.array([5, 6, 7, 8, 9, 10, 11, 12, 4, 1], np.int64)
    return ann, nt, (pre, post, w)


class SyntheticCircuit(unittest.TestCase):
    cfg = {"interneurons": 4, "descending": 5, "sensory": 5, "min_weight": 2}

    def test_deterministic_hash_and_signs(self):
        ann, nt, edges = synthetic()
        a1, r1 = ext.build_circuit(ann, nt, edges, self.cfg)
        a2, _ = ext.build_circuit(ann, nt, edges, self.cfg)
        self.assertEqual(circuit_io.content_hash(a1), circuit_io.content_hash(a2))
        self.assertEqual(r1["populations"]["MN"]["neurons"], 2)
        self.assertLessEqual(r1["populations"]["IN"]["neurons"], 4)
        bid = list(a1["body_id"])
        self.assertEqual(a1["sign"][bid.index(10)], -1)          # gaba
        self.assertEqual(a1["sign"][bid.index(11)], -1)          # glutamate
        self.assertEqual((a1["sign"][bid.index(12)], a1["sign_code"][bid.index(12)]), (1, 2))   # unclear
        self.assertEqual((a1["sign"][bid.index(30)], a1["sign_code"][bid.index(30)]), (1, 1))   # other NT
        self.assertTrue(np.array_equal(a1["edge_sign"], a1["sign"][a1["pre"]]))
        # CSR (rows = post) consistent
        self.assertEqual(a1["indptr"][-1], len(a1["pre"]))
        self.assertTrue(np.array_equal(np.repeat(np.arange(len(bid)), np.diff(a1["indptr"])), a1["post"]))
        self.assertTrue((a1["weight"] >= 2).all())

    def test_save_load_roundtrip(self):
        ann, nt, edges = synthetic()
        arrays, _ = ext.build_circuit(ann, nt, edges, self.cfg)
        with tempfile.TemporaryDirectory() as d:
            h = circuit_io.save_circuit(Path(d) / "c.npz", arrays, {"x": 1})
            back, meta, stored = circuit_io.load_circuit(Path(d) / "c.npz")
        self.assertEqual((h, stored, meta), (circuit_io.content_hash(arrays), h, {"x": 1}))
        self.assertEqual(circuit_io.content_hash(back), h)

    def test_scan_matches_bruteforce(self):
        ann, nt, (pre, post, w) = synthetic()
        cand = np.array([1, 2, 10, 11, 12], np.int64)
        batches = [(pre[:4], post[:4], w[:4]), (pre[4:], post[4:], w[4:])]
        sp, so, sw = ext.scan_weights(iter(batches), cand, 2)
        m = np.isin(pre, cand) & np.isin(post, cand) & (w >= 2)
        self.assertEqual(list(sp), list(pre[m]))
        self.assertEqual(list(so), list(post[m])); self.assertEqual(list(sw), list(w[m]))


@unittest.skipUnless(NPZ.exists() and JSON.exists() and MAP.exists(), "run scripts/extract_typing_circuit.py first")
class GeneratedCircuit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a, cls.meta, cls.hash = circuit_io.load_circuit(NPZ)   # verifies stored hash
        cls.summary = json.loads(JSON.read_text())
        cls.jm = json.loads(MAP.read_text())

    def test_hash_recorded(self):
        self.assertEqual(self.hash, self.summary["content_hash"])
        self.assertEqual(circuit_io.content_hash(self.a), self.hash)

    def test_population_counts(self):
        s = self.summary["populations"]
        self.assertEqual(s["MN"]["neurons"], 173)
        self.assertGreaterEqual(int((self.a["mn_group"] == "proln_foreleg").sum()), 81)
        self.assertEqual(int((self.a["mn_group"] == "proln_foreleg").sum()), 81)
        self.assertTrue(2000 <= s["IN"]["neurons"] <= 2600)
        self.assertTrue(100 <= s["DN"]["neurons"] <= 400)
        self.assertTrue(50 <= s["SN"]["neurons"] <= 324)
        self.assertTrue(3000 <= self.summary["neurons"] <= 4000)
        self.assertTrue(70_000 <= self.summary["edges"] <= 150_000)
        for r in ("DN", "SN", "IN", "MN"):
            self.assertEqual(len(self.a[f"idx_{r}"]), s[r]["neurons"])
        n = len(self.a["body_id"])
        self.assertEqual(len(np.unique(self.a["body_id"])), n)
        self.assertEqual(len(self.a["idx_MN_L"]) + len(self.a["idx_MN_R"]), 173)
        self.assertTrue(set(np.unique(self.a["sign"])) <= {-1, 1})
        self.assertTrue(self.a["pre"].max() < n and self.a["post"].max() < n)

    def test_map_references_existing_types_and_joints(self):
        model = json.loads(MODEL.read_text())
        expected = {f"joint_{leg}{dof}" for leg in ("LF", "RF") for dof in model["leg_dofs"]}
        self.assertEqual(len(expected), 14)
        self.assertEqual(set(self.jm["joints"]), expected)
        types = set(self.a["mn_type"][self.a["role"] == "MN"].tolist())
        for name, j in self.jm["joints"].items():
            self.assertIn(j["status"], {"mn", "fallback"})
            if j["status"] == "fallback":
                self.assertTrue(j.get("reason"))
                continue
            for t in j["agonist"] + j["antagonist"]:
                self.assertIn(t, types, f"{name}: unknown MN type {t}")
            self.assertTrue(j["agonist"], name)
            self.assertFalse(set(j["agonist"]) & set(j["antagonist"]), name)

    def test_joint_coverage_report(self):
        cov = ext.joint_coverage(self.a, self.jm)
        self.assertEqual(len(cov), 14)
        for name, row in cov.items():
            if row["status"] == "mn":
                self.assertGreater(row["agonist"]["n_mn"], 0, name)
                self.assertGreater(row["agonist"]["input_synapses"], 0, name)
        # every ProLN MN type is used by some joint (distal joints fully covered)
        used = {t for j in self.jm["joints"].values() for k in ("agonist", "antagonist") for t in j[k]}
        proln = set(self.a["mn_type"][self.a["mn_group"] == "proln_foreleg"].tolist()) - {""}
        self.assertEqual(proln - used, set())


if __name__ == "__main__":
    unittest.main()
