"""Sparse recurrent connectome runtime: ``SparseRecurrent`` (P2a).

Dynamics per micro-step (K per forward):  r <- (1 - a) r + a * f(W r + x),  a = 1 / tau (default 1),
f = tanh.  ``x`` is zero except on the input population.  With r0 in [-1, 1] the state stays in
[-1, 1] for a in (0, 1].  Effective weight per edge = sign * exp(log_mag) * mask, where ``log_mag``
is the only learned tensor (one scalar per edge), initialised from log synapse counts normalised by
in-degree (each row of |W| sums to ``init_scale`` before optional spectral normalisation).

Backends (all agree numerically): ``dense`` (small circuits), ``csr`` (alias ``csr_fn``: CSR matmul
with a hand-written autograd.Function, since stock CSR backward fails), ``index_add`` (portable,
works on MPS).  ``csr_fn`` lives here (canonical copy; ``scripts/bench_device.py`` has its own).

Variants (equal edge count / trainable parameter count except ``frozen``, which has zero trainable):
``real``; ``shuffled`` (pre indices permuted across edges: in/out degrees, weights, signs multiset
preserved); ``random_sparse`` (Erdos-Renyi, same edge count and sign fraction, weight multiset
preserved); ``frozen`` (real topology, ``log_mag.requires_grad = False``).
"""
from __future__ import annotations

import hashlib
import warnings
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .circuit_io import load_circuit
from .device import resolve_device

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
warnings.filterwarnings("ignore", message="Sparse invariant checks")
BACKENDS = ("auto", "dense", "csr", "csr_fn", "index_add")
VARIANTS = ("real", "shuffled", "random_sparse", "frozen")
POPULATIONS = ("DN", "SN", "IN", "MN")


_POOL = None
_MAX_BLOCKS = 4                 # torch.sparse.mm(CSR) is single-threaded on CPU; row blocks scale to ~4 threads
_PAR_MIN_WORK = 2_000_000       # E * B below which threading the matmul does not pay
_MV_MAX_B = 12                  # CSR mm costs ~8 ns/edge for any small B, CSR mv ~0.8 ns/edge: loop mv for B <= this


def _pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(_MAX_BLOCKS)
    return _POOL


class CsrPlan:
    """Static CSR structure of W (rows = post) and of W^T, plus edge-balanced row blocks.

    Only the values change between steps, so the sparse tensors are re-wrapped around new values
    (no index copies) and ``vals[perm]`` for the transpose is cached while ``vals`` is unchanged."""

    def __init__(self, crow, col, row_of, crowT, colT, perm, n):
        self.crow, self.col, self.row_of = crow, col, row_of
        self.crowT, self.colT, self.perm, self.n = crowT, colT, perm, n
        self._blocks = {}
        self._vt = None                      # (vals, version, vals[perm])
        self._ones = None

    def blocks(self, which, P):
        key = (which, P)
        if key not in self._blocks:
            crow = self.crow if which == "N" else self.crowT
            col = self.col if which == "N" else self.colT
            E = int(crow[-1])
            cuts = torch.searchsorted(crow, torch.tensor([E * i // P for i in range(1, P)], dtype=crow.dtype))
            bounds = [0] + cuts.tolist() + [self.n]
            bl = []
            for a, b in zip(bounds[:-1], bounds[1:]):
                e0, e1 = int(crow[a]), int(crow[b])
                if b > a:
                    bl.append((b - a, e0, e1, crow[a:b + 1] - e0, col[e0:e1]))
            self._blocks[key] = bl
        return self._blocks[key]

    def vals_T(self, vals):
        c = self._vt
        if c is None or c[0].data_ptr() != vals.data_ptr() or c[1] != vals._version:
            c = self._vt = (vals, vals._version, vals[self.perm])
        return c[2]

    def spmm(self, which, vals, R):
        """(W or W^T) @ R for R [N, B] contiguous; ``vals`` are in that matrix's CSR order."""
        n, work = self.n, vals.numel() * R.shape[1]
        P = min(torch.get_num_threads(), _MAX_BLOCKS) if (work >= _PAR_MIN_WORK and R.device.type == "cpu") else 1
        if P <= 1:
            crow = self.crow if which == "N" else self.crowT
            col = self.col if which == "N" else self.colT
            S = torch.sparse_csr_tensor(crow, col, vals, size=(n, n))
            B = R.shape[1]
            if B <= _MV_MAX_B and R.device.type == "cpu":
                if B == 1:
                    return torch.mv(S, R[:, 0]).unsqueeze(1)
                return torch.stack([torch.mv(S, c) for c in R.T.contiguous()], 1)
            return torch.sparse.mm(S, R)
        bl = self.blocks(which, P)
        def run(b):
            rows, e0, e1, cr, cl = b
            return torch.sparse.mm(torch.sparse_csr_tensor(cr, cl, vals[e0:e1], size=(rows, n)), R)
        return torch.cat(list(_pool().map(run, bl)))

    def edge_dots(self, G, R):
        """d[e] = sum_b G[row_e, b] * R[col_e, b] for G, R [N, B]  (no E x B temporary)."""
        if G.shape[1] == 1:
            return G[self.row_of, 0] * R[self.col, 0]
        if self._ones is None or self._ones.dtype != G.dtype or self._ones.device != G.device:
            self._ones = torch.sparse_csr_tensor(self.crow, self.col, torch.ones(self.col.numel(), dtype=G.dtype, device=G.device),
                                                 size=(self.n, self.n))
        return torch.sparse.sampled_addmm(self._ones, G, R.T, beta=0.0).values()


class CsrMM(torch.autograd.Function):
    """out = W @ R with W in CSR (values differentiable); backward written by hand."""

    @staticmethod
    def forward(ctx, vals, R, plan):
        ctx.save_for_backward(vals, R)
        ctx.plan = plan
        return plan.spmm("N", vals, R)

    @staticmethod
    def backward(ctx, G):
        vals, R = ctx.saved_tensors
        plan = ctx.plan
        G = G.contiguous()
        gvals = plan.edge_dots(G, R) if ctx.needs_input_grad[0] else None
        gR = plan.spmm("T", plan.vals_T(vals), G) if ctx.needs_input_grad[1] else None
        return gvals, gR, None


def csr_fn(vals, R, crow, col, row_of, crowT, colT, perm, n, plan=None):
    """Legacy signature; pass ``plan`` (a CsrPlan) to reuse cached structure."""
    return CsrMM.apply(vals, R, plan or CsrPlan(crow, col, row_of, crowT, colT, perm, n))


# ---------------------------------------------------------------- topology helpers

def _sorted_unique(post: np.ndarray, pre: np.ndarray, n: int) -> bool:
    key = post.astype(np.int64) * n + pre
    return np.unique(key).size == key.size


def shuffle_edges(post, pre, n, seed):
    """Permute ``pre`` across edges (rows fixed): exact in/out degree preservation, no duplicates."""
    rng = np.random.default_rng(seed)
    pre = pre[rng.permutation(pre.size)].copy()
    for _ in range(1000):
        key = post.astype(np.int64) * n + pre
        _, first, counts = np.unique(key, return_index=True, return_counts=True)
        if first.size == key.size:
            return pre
        dup = np.ones(key.size, bool)
        dup[first] = False                       # later copies of a duplicated pair
        bad = np.nonzero(dup)[0]
        other = rng.integers(0, pre.size, bad.size)
        for i, j in zip(bad.tolist(), other.tolist()):   # sequential swaps keep the multiset exact
            pre[i], pre[j] = pre[j], pre[i]
    raise RuntimeError("could not remove duplicate edges while shuffling")


def random_sparse_edges(n, n_edges, seed):
    rng = np.random.default_rng(seed)
    keys = np.empty(0, np.int64)
    while keys.size < n_edges:
        draw = rng.integers(0, n * n, int((n_edges - keys.size) * 1.3) + 16, dtype=np.int64)
        keys = np.unique(np.concatenate([keys, draw]))
    keys = rng.permutation(keys)[:n_edges]
    keys.sort()
    return keys // n, keys % n


def _spectral_radius(mat_mv, n, iters=60, seed=0):
    """Geometric-mean growth of ||W^k v|| (robust to complex leading pairs)."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, 1, generator=g, dtype=torch.float64)
    v /= v.norm()
    logs = []
    for _ in range(iters):
        v = mat_mv(v)
        nrm = float(v.norm())
        if nrm == 0:
            return 0.0
        logs.append(math.log(nrm))
        v /= nrm
    return math.exp(sum(logs[iters // 2:]) / (iters - iters // 2))


class SparseRecurrent(nn.Module):
    def __init__(self, circuit, backend="auto", k_steps=4, variant="real", seed=0, init_scale=1.0,
                 tau=1.0, nonlinearity=torch.tanh, input_idx=None, spectral_radius=None,
                 device="cpu"):
        super().__init__()
        if isinstance(circuit, (str, Path)):
            circuit = load_circuit(circuit)[0]
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        if tau < 1.0:
            raise ValueError("tau must be >= 1 (leak = 1/tau in (0, 1])")
        n = int(circuit["indptr"].shape[0] - 1)
        post0 = np.repeat(np.arange(n, dtype=np.int64), np.diff(circuit["indptr"]))
        pre0 = circuit["pre"].astype(np.int64)
        cnt = circuit["weight"].astype(np.float64)
        sgn = circuit["edge_sign"].astype(np.float64)
        self.variant, self.seed, self.k_steps, self.n = variant, int(seed), int(k_steps), n
        self.alpha, self.nonlinearity = 1.0 / float(tau), nonlinearity
        self.init_scale, self.n_edges = float(init_scale), int(post0.size)

        if variant == "shuffled":
            post, pre = post0, shuffle_edges(post0, pre0, n, seed)
        elif variant == "random_sparse":
            post, pre = random_sparse_edges(n, post0.size, seed)
            rng = np.random.default_rng(seed + 1)
            cnt = cnt[rng.permutation(cnt.size)]                 # same weight multiset
            sgn = np.where(np.arange(sgn.size) < (sgn < 0).sum(), -1.0, 1.0)
            sgn = sgn[rng.permutation(sgn.size)]                 # same sign fraction
        else:
            post, pre = post0, pre0
        order = np.lexsort((pre, post))
        post, pre, cnt, sgn = post[order], pre[order], cnt[order], sgn[order]

        # log_mag init: log(count) - log(in-degree-sum of counts) + log(init_scale)
        rowsum = np.bincount(post, weights=cnt, minlength=n)
        log_mag = np.log(cnt) - np.log(rowsum[post]) + math.log(self.init_scale)

        self.backend = self._pick_backend(backend, n, device)
        self.device_ = torch.device(device)
        t = lambda a, dt=torch.int64: torch.as_tensor(np.ascontiguousarray(a), dtype=dt)
        self.register_buffer("post", t(post)); self.register_buffer("pre", t(pre))
        self.register_buffer("sign", t(sgn, torch.float32))
        perm = np.argsort(pre * n + post, kind="stable")
        self.register_buffer("perm", t(perm))
        self.register_buffer("post_T", t(post[perm]))
        crow = np.zeros(n + 1, np.int64); crow[1:] = np.cumsum(np.bincount(post, minlength=n))
        crowT = np.zeros(n + 1, np.int64); crowT[1:] = np.cumsum(np.bincount(pre, minlength=n))
        self.register_buffer("crow", t(crow)); self.register_buffer("crowT", t(crowT))
        self.log_mag = nn.Parameter(t(log_mag, torch.float32))
        if variant == "frozen":
            self.log_mag.requires_grad_(False)

        if spectral_radius is not None:       # uniform log-shift so that rho(W) = spectral_radius
            with torch.no_grad():
                w = (self.sign * self.log_mag.exp()).double()
                pi, po = self.pre, self.post
                mv = lambda v: torch.zeros(n, 1, dtype=torch.float64).index_add_(0, po, w[:, None] * v[pi])
                rho = _spectral_radius(mv, n, seed=seed)
                if rho > 0:
                    self.log_mag += math.log(spectral_radius / rho)

        idx = {}
        for p in POPULATIONS:
            key = f"idx_{p}"
            idx[p] = circuit[key].astype(np.int64) if key in circuit else np.empty(0, np.int64)
            self.register_buffer(f"idx_{p}", t(idx[p]))
        if input_idx is None:
            input_idx = idx["SN"]
        elif isinstance(input_idx, str):
            input_idx = idx[input_idx]
        self.register_buffer("input_idx", t(np.asarray(input_idx)))
        self.to(self.device_)

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _pick_backend(backend, n, device):
        if backend == "csr_fn":
            backend = "csr"
        if backend != "auto":
            return backend
        if str(device).startswith("mps"):
            return "index_add"
        return "dense" if n <= 1000 else "csr"

    @property
    def n_in(self) -> int:
        return int(self.input_idx.numel())

    def edge_weights(self) -> torch.Tensor:
        """Signed effective weight per (post, pre)-sorted edge."""
        return self.sign * self.log_mag.exp()

    def dense_weight(self) -> torch.Tensor:
        return torch.zeros(self.n, self.n, device=self.log_mag.device).index_put(
            (self.post, self.pre), self.edge_weights())

    # ------------------------------------------------------------------ dynamics
    def _plan(self):
        p = self.__dict__.get("_csr_plan")
        if p is None or p.crow is not self.crow:          # rebuilt after .to(device)/dtype moves
            p = self.__dict__["_csr_plan"] = CsrPlan(self.crow, self.pre, self.post, self.crowT, self.post_T,
                                                     self.perm, self.n)
        return p

    def _matvec(self, w, r, dense):
        """(W r) for r [B, N] -> [B, N]."""
        if self.backend == "dense":
            return r @ dense.T
        if self.backend == "csr":
            return CsrMM.apply(w, r.T.contiguous(), self._plan()).T
        out = torch.zeros_like(r)
        return out.index_add_(1, self.post, r[:, self.pre] * w)

    def forward(self, input_currents, state=None):
        B = input_currents.shape[0]
        x = input_currents.new_zeros(B, self.n)
        x[:, self.input_idx] = input_currents
        r = x.new_zeros(B, self.n) if state is None else state
        w = self.edge_weights()
        dense = self.dense_weight() if self.backend == "dense" else None
        a = self.alpha
        for _ in range(self.k_steps):
            act = self.nonlinearity(self._matvec(w, r, dense) + x)
            r = act if a == 1.0 else (1.0 - a) * r + a * act
        return r, r

    def rates(self, name: str, rates: torch.Tensor) -> torch.Tensor:
        return rates[:, getattr(self, f"idx_{name}")]

    def population_rates(self, rates: torch.Tensor) -> dict[str, torch.Tensor]:
        return {p: self.rates(p, rates) for p in POPULATIONS}

    def dn_rates(self, r): return self.rates("DN", r)
    def sn_rates(self, r): return self.rates("SN", r)
    def in_rates(self, r): return self.rates("IN", r)
    def mn_rates(self, r): return self.rates("MN", r)

    # ------------------------------------------------------------------ controls
    def n_params(self, trainable_only=False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def variant_signature(self) -> dict:
        """Counts/degree hashes for asserting that controls are matched to the real circuit."""
        indeg = np.bincount(self.post.cpu().numpy(), minlength=self.n)
        outdeg = np.bincount(self.pre.cpu().numpy(), minlength=self.n)
        sign = self.sign.cpu().numpy()
        h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]
        return {"variant": self.variant, "n_neurons": self.n, "n_edges": self.n_edges,
                "n_params": self.n_params(), "n_trainable": self.n_params(True),
                "n_neg": int((sign < 0).sum()), "indeg_hash": h(indeg), "outdeg_hash": h(outdeg)}
