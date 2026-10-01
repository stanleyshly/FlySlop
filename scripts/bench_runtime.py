"""Benchmark SparseRecurrent (P2c): forward / backward on the typing and thinker circuits.

  uv run --extra training python scripts/bench_runtime.py [--threads 4] [--backend csr] [--json out.json]

Cases: typing B=1 (forward = one policy step), typing B=256 (PPO minibatch fwd+bwd), thinker B=32 with
T=128 tokens x K=4 micro-steps (weights hoisted once per sequence, like thinker.py does).
Reports ms (median of --reps) and peak RSS (MB, whole process so far).
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.connectome.runtime import SparseRecurrent  # noqa: E402

TYPING = ROOT / "data/connectome/typing_circuit.npz"
THINKER = ROOT / "data/connectome/thinker_circuit_small.npz"


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def med(f, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2] * 1e3


def run_modules(m, B, reps):
    x = torch.randn(B, m.n_in, generator=torch.Generator().manual_seed(0))

    def fwd():
        with torch.no_grad():
            m(x)

    def fb():
        m.zero_grad(); r, _ = m(x); (r ** 2).sum().backward()
    fwd(); fb()
    return {"fwd_ms": med(fwd, reps), "fwd_bwd_ms": med(fb, reps)}


def run_seq(m, B, T, reps):
    g = torch.Generator().manual_seed(0)
    xs = [torch.randn(B, m.n_in, generator=g) for _ in range(T)]
    idx = m.input_idx

    def seq():
        w = m.edge_weights()
        dense = m.dense_weight() if m.backend == "dense" else None
        r = torch.zeros(B, m.n)
        for t in range(T):
            x = torch.zeros(B, m.n); x[:, idx] = xs[t]
            for _ in range(m.k_steps):
                r = torch.tanh(m._matvec(w, r, dense) + x)
        return r

    def fwd():
        with torch.no_grad():
            seq()

    def fb():
        m.zero_grad(); (seq() ** 2).mean().backward()
    fwd(); fb()
    return {"fwd_ms": med(fwd, reps), "fwd_bwd_ms": med(fb, reps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--backend", default="csr")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--only", default="", help="comma list of: t1,t256,th")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    only = set(a.only.split(",")) - {""}
    res = {"threads": a.threads, "backend": a.backend}
    if not only or only & {"t1", "t256"}:
        m = SparseRecurrent(TYPING, backend=a.backend, k_steps=4)
        for tag, B, reps in (("t1", 1, 20), ("t256", 256, a.reps)):
            if not only or tag in only:
                res[f"typing_B{B}"] = run_modules(m, B, reps); res[f"typing_B{B}"]["rss_mb"] = rss_mb()
    if not only or "th" in only:
        m = SparseRecurrent(THINKER, backend=a.backend, k_steps=4, spectral_radius=4.0)
        res[f"thinker_B32_T{a.T}"] = run_seq(m, 32, a.T, max(1, a.reps - 1))
        res[f"thinker_B32_T{a.T}"]["rss_mb"] = rss_mb()
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
