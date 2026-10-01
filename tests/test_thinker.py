"""P6a: thinker connectome + matched controls (uses the 5k-neuron small thinker circuit)."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.connectome.thinker import (EOS, PAD, KINDS, MATCHED, ThinkerConfig, build_model,  # noqa: E402
                                        param_report, quick_overfit)

CIRCUIT = ROOT / "data/connectome/thinker_circuit_small.npz"
CFG = ThinkerConfig()


@unittest.skipUnless(CIRCUIT.exists(), "thinker_circuit_small.npz missing")
class ThinkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.models = {k: build_model(k, CFG) for k in KINDS}

    def test_param_counts_matched(self):
        rep = {k: param_report(m) for k, m in self.models.items()}
        ref = rep["real"]["trainable"]
        for k in MATCHED:
            self.assertLess(abs(rep[k]["trainable"] - ref) / ref, 0.02, (k, rep[k]["trainable"], ref))
        print("\nparam counts (trainable / frozen):",
              {k: (r["trainable"], r["frozen"]) for k, r in rep.items()})

    def test_frozen_reported_separately(self):
        rep = param_report(self.models["frozen"])
        real = param_report(self.models["real"])
        n_edges = self.models["real"].core.n_edges
        self.assertEqual(rep["frozen"], n_edges)
        self.assertEqual(rep["trainable"], real["trainable"] - n_edges)
        self.assertFalse(self.models["frozen"].core.log_mag.requires_grad)

    def test_frozen_random_magnitudes_deterministic(self):
        a = self.models["frozen"].core.log_mag
        b = build_model("frozen", CFG).core.log_mag
        c = build_model("frozen", replace(CFG, seed=1)).core.log_mag
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.allclose(a, c))
        self.assertFalse(torch.allclose(a, self.models["real"].core.log_mag))

    def test_controls_share_readout_init_and_topology_signature(self):
        real, shuf, rnd = (self.models[k].core.variant_signature() for k in ("real", "shuffled", "random_sparse"))
        self.assertEqual(real["n_edges"], shuf["n_edges"])
        self.assertEqual(real["n_edges"], rnd["n_edges"])
        self.assertEqual(real["indeg_hash"], shuf["indeg_hash"])
        self.assertTrue(torch.equal(self.models["real"].embed.weight, self.models["gru"].embed.weight))

    def test_step_matches_forward(self):
        m = self.models["real"]
        m.cfg = replace(m.cfg, tbptt=0)
        seq = torch.randint(16, 1024, (3, 6))
        with torch.no_grad():
            full = m(seq)
            st, outs = None, []
            for t in range(6):
                lg, st = m.step(seq[:, t], st)
                outs.append(lg)
        m.cfg = CFG
        self.assertTrue(torch.allclose(full, torch.stack(outs, 1), atol=1e-4))

    def test_pad_keeps_state_and_generate(self):
        m = self.models["real"]
        with torch.no_grad():
            _, s1 = m.step(torch.tensor([40, 41]))
            _, s2 = m.step(torch.tensor([PAD, PAD]), s1)
            self.assertTrue(torch.equal(s1, s2))
            out = m.generate([[40, 41, 42], [50]], max_new=5, eos=EOS)
        self.assertEqual(out.shape[0], 2)
        self.assertLessEqual(out.shape[1], 5)
        g = build_model("gru", CFG)
        self.assertEqual(g.generate(torch.tensor([[40, 41]]), max_new=4, temperature=1.0).shape[0], 1)

    def test_tbptt_chunks(self):
        m = self.models["gru"]
        seq = torch.randint(16, 1024, (2, 10))
        chunks = list(m.iter_chunks(seq, chunk=4))
        self.assertEqual([c[0] for c in chunks], [0, 4, 8])
        self.assertEqual(sum(c[1].shape[1] for c in chunks), 10)

    def test_tiny_overfit_connectome(self):
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(1)
        seq = torch.randint(32, 64, (20, 12), generator=g)     # small alphabet: needs context, not bigrams
        seq[:, 0] = torch.arange(20) + 100                    # distinct first token so every target is predictable
        m = build_model("real", CFG)
        curve = quick_overfit(m, seq, steps=100, max_seconds=50, target_acc=0.9)
        print("\noverfit curve (step, loss, acc, s):", [(s, round(l, 2), round(a, 2), round(t, 1)) for s, l, a, t in curve])
        self.assertGreaterEqual(curve[-1][2], 0.9)
        self.assertLess(curve[-1][3], 60)


if __name__ == "__main__":
    unittest.main()
