import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.connectome.circuit_io import load_circuit  # noqa: E402
from backend.connectome import runtime as rt  # noqa: E402
from backend.connectome.runtime import SparseRecurrent  # noqa: E402

REAL = ROOT / "data" / "connectome" / "typing_circuit.npz"


def toy(n=60, per=5, seed=0):
    rng = np.random.default_rng(seed)
    keys = np.unique(np.concatenate([rng.choice(n * n, per * n, replace=False)]))
    post, pre = keys // n, keys % n
    indptr = np.zeros(n + 1, np.int64); indptr[1:] = np.cumsum(np.bincount(post, minlength=n))
    return {"indptr": indptr, "pre": pre.astype(np.int32),
            "weight": rng.integers(1, 30, keys.size).astype(np.int32),
            "edge_sign": rng.choice([-1, 1], keys.size, p=[.3, .7]).astype(np.int8),
            "idx_DN": np.arange(0, 5), "idx_SN": np.arange(5, 15),
            "idx_IN": np.arange(15, 50), "idx_MN": np.arange(50, 60)}


def degs(m):
    return (np.bincount(m.post.numpy(), minlength=m.n), np.bincount(m.pre.numpy(), minlength=m.n))


class RuntimeTests(unittest.TestCase):
    def test_backend_agreement_and_grads(self):
        c = toy(); x = torch.randn(3, 10)
        outs = {}
        for b in ("dense", "csr", "index_add"):
            m = SparseRecurrent(c, backend=b, seed=1, init_scale=2.0)
            r, _ = m(x)
            (r ** 2).sum().backward()
            outs[b] = (r.detach(), m.log_mag.grad.clone())
        for b in ("csr", "index_add"):
            torch.testing.assert_close(outs[b][0], outs["dense"][0], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(outs[b][1], outs["dense"][1], atol=1e-5, rtol=1e-3)

    def test_variants_preserve_counts(self):
        c = toy(); real = SparseRecurrent(c, backend="dense")
        for v in ("shuffled", "random_sparse", "frozen"):
            m = SparseRecurrent(c, backend="dense", variant=v, seed=3)
            self.assertEqual(m.n_edges, real.n_edges)
            self.assertEqual(m.n_params(), real.n_params())
            self.assertEqual(int((m.sign < 0).sum()), int((real.sign < 0).sum()))
            key = m.post * m.n + m.pre
            self.assertEqual(key.unique().numel(), key.numel())
            if v == "shuffled":
                for a, b in zip(degs(m), degs(real)):
                    np.testing.assert_array_equal(a, b)
                self.assertFalse(torch.equal(m.pre, real.pre))
        self.assertEqual(SparseRecurrent(c, variant="frozen").n_params(True), 0)
        self.assertEqual(SparseRecurrent(c).variant_signature()["n_trainable"], real.n_edges)

    def test_frozen_no_grad(self):
        m = SparseRecurrent(toy(), variant="frozen", backend="csr")
        self.assertFalse(m.log_mag.requires_grad)
        r, _ = m(torch.randn(2, 10, requires_grad=True))
        r.sum().backward()
        self.assertIsNone(m.log_mag.grad)

    def test_determinism(self):
        c = toy(); x = torch.randn(2, 10)
        for v in ("real", "shuffled", "random_sparse"):
            a = SparseRecurrent(c, variant=v, seed=5); b = SparseRecurrent(c, variant=v, seed=5)
            self.assertTrue(torch.equal(a(x)[0], b(x)[0]))
        s1, s2 = (SparseRecurrent(c, variant="shuffled", seed=s) for s in (1, 2))
        self.assertFalse(torch.equal(s1.pre, s2.pre))

    def test_bounded_and_spectral(self):
        c = toy(); x = 50 * torch.randn(4, 10)
        m = SparseRecurrent(c, init_scale=100.0, tau=2.0)
        r = None
        for _ in range(20):
            r, _ = m(x, r)
        self.assertTrue(bool(r.abs().max() <= 1.0) and bool(torch.isfinite(r).all()))
        m = SparseRecurrent(c, spectral_radius=0.9)
        w = m.dense_weight().double()
        rho = float(torch.linalg.eigvals(w).abs().max())
        self.assertAlmostEqual(rho, 0.9, delta=0.15)

    def test_state_and_populations(self):
        m = SparseRecurrent(toy())
        r, rates = m(torch.randn(2, 10))
        self.assertEqual(m.rates("MN", rates).shape, (2, 10))
        self.assertEqual(set(m.population_rates(rates)), {"DN", "SN", "IN", "MN"})
        r2, _ = m(torch.zeros(2, 10), r)
        self.assertEqual(r2.shape, (2, 60))

    @unittest.skipUnless(REAL.exists(), "typing_circuit.npz missing")
    def test_real_circuit(self):
        c = load_circuit(REAL)[0]
        x = torch.randn(2, 200)
        res = {}
        for b in ("csr", "index_add"):
            m = SparseRecurrent(c, backend=b)
            r, _ = m(x); r.sum().backward()
            res[b] = (r.detach(), m.log_mag.grad)
            self.assertEqual(m.n, 3173); self.assertEqual(m.n_edges, 98551)
            self.assertEqual(m.mn_rates(r).shape, (2, 173))
        torch.testing.assert_close(res["csr"][0], res["index_add"][0], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(res["csr"][1], res["index_add"][1], atol=1e-5, rtol=1e-3)
        sh = SparseRecurrent(c, variant="shuffled", seed=0)
        real = SparseRecurrent(c)
        for a, b in zip(degs(sh), degs(real)):
            np.testing.assert_array_equal(a, b)
        rs = SparseRecurrent(c, variant="random_sparse", seed=0)
        self.assertEqual(rs.n_params(), real.n_params())


class FastPathTests(unittest.TestCase):
    """mv loop (small B), threaded row blocks, sampled_addmm value grads and the vals^T cache."""

    def _grads(self, backend, B, seed=0):
        c = toy(n=200, per=8, seed=3)
        m = SparseRecurrent(c, backend=backend, seed=1, init_scale=2.0)
        x = torch.randn(B, 10, generator=torch.Generator().manual_seed(seed), requires_grad=True)
        r, _ = m(x)
        (r ** 2).sum().backward()
        return r.detach(), m.log_mag.grad.clone(), x.grad.clone()

    def test_paths_match_dense(self):
        old = (rt._MV_MAX_B, rt._PAR_MIN_WORK)
        try:
            for B in (1, 3, 20):
                ref = self._grads("dense", B)
                for mv, par in ((12, 10 ** 9), (0, 10 ** 9), (0, 0)):
                    rt._MV_MAX_B, rt._PAR_MIN_WORK = mv, par
                    for a, b in zip(self._grads("csr", B), ref):
                        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-3)
        finally:
            rt._MV_MAX_B, rt._PAR_MIN_WORK = old

    def test_threaded_deterministic(self):
        old = rt._PAR_MIN_WORK
        rt._PAR_MIN_WORK = 0
        try:
            a, b = self._grads("csr", 20), self._grads("csr", 20)
            for u, v in zip(a, b):
                self.assertTrue(torch.equal(u, v))
        finally:
            rt._PAR_MIN_WORK = old

    def test_transpose_cache_follows_weight_updates(self):
        m = SparseRecurrent(toy(n=200, per=8, seed=3), backend="csr", seed=1)
        x = torch.randn(4, 10)
        for step in range(2):
            m.zero_grad(); r, _ = m(x); (r ** 2).sum().backward()
            g = m.log_mag.grad.clone()
            d = SparseRecurrent(toy(n=200, per=8, seed=3), backend="dense", seed=1)
            d.load_state_dict(m.state_dict()); r2, _ = d(x); (r2 ** 2).sum().backward()
            torch.testing.assert_close(g, d.log_mag.grad, atol=1e-5, rtol=1e-3)
            with torch.no_grad():
                m.log_mag += 0.1 * torch.randn_like(m.log_mag)     # in-place update, same storage

    def test_hoisted_weights_accumulate_over_steps(self):
        c = toy(n=200, per=8, seed=3)
        outs = []
        for b in ("dense", "csr"):
            m = SparseRecurrent(c, backend=b, seed=1, init_scale=2.0)
            w = m.edge_weights(); dense = m.dense_weight() if b == "dense" else None
            r = torch.zeros(5, m.n); x = torch.zeros(5, m.n); x[:, :10] = 1.0
            for _ in range(6):
                r = torch.tanh(m._matvec(w, r, dense) + x)
            (r ** 2).sum().backward()
            outs.append(m.log_mag.grad)
        torch.testing.assert_close(outs[1], outs[0], atol=1e-5, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
