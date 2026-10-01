"""Benchmark recurrent-step backends (dense, CSR, index_add_ edge scatter) on CPU and MPS.

Usage: uv run --extra training python scripts/bench_device.py [--quick] [--out PATH]
Writes data/connectome/device_bench.json (read by backend.connectome.device) and prints a
markdown table. One iteration = K recurrent micro-steps  r <- tanh(W r + x).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gc
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import memory_budget as mb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "connectome" / "device_bench.json"
SIZES = {"360": (360, 1, None), "4k": (4000, 64, 25), "20k": (20000, 64, 25)}  # n, batch, edges/neuron
K = 4


def make_edges(n: int, per: int | None, gen: torch.Generator):
    if per is None:  # dense-ish: ~half of all pairs
        m = torch.rand(n, n, generator=gen) < 0.5
        dst, src = m.nonzero(as_tuple=True)
    else:
        dst = torch.arange(n).repeat_interleave(per)
        src = torch.randint(0, n, (n * per,), generator=gen)
    return dst, src, torch.randn(dst.numel(), generator=gen) / (per or n / 2) ** 0.5


class CsrMM(torch.autograd.Function):
    """out = W @ R with W in CSR (values differentiable). Stock CSR autograd fails on CPU
    ("invalid gradient ... SparseCompressedTensorBackward0"), so backward is written by hand."""

    @staticmethod
    def forward(ctx, vals, R, crow, col, row_of, crowT, colT, perm, n):
        ctx.save_for_backward(vals, R, crowT, colT, perm, row_of, col)
        ctx.n = n
        return torch.sparse.mm(torch.sparse_csr_tensor(crow, col, vals, size=(n, n)), R)

    @staticmethod
    def backward(ctx, G):
        vals, R, crowT, colT, perm, row_of, col = ctx.saved_tensors
        n = ctx.n
        gvals = (G[row_of] * R[col]).sum(1)
        gR = torch.sparse.mm(torch.sparse_csr_tensor(crowT, colT, vals[perm], size=(n, n)), G)
        return gvals, gR, None, None, None, None, None, None, None


def build(backend: str, n: int, dst, src, val, device):
    dst, src = dst.to(device), src.to(device)
    w = torch.nn.Parameter(val.clone().to(device))
    if backend == "dense":
        if n <= 4000:
            wd = torch.nn.Parameter(torch.zeros(n, n, device=device).index_put((dst, src), val.to(device)))
        else:  # 20k dense: random values, no edge list (memory)
            wd = torch.nn.Parameter(torch.randn(n, n, device=device) * 0.01)
        def step(r, x, wd=wd):
            return torch.tanh(r @ wd.T + x)
        return step, [wd]
    if backend == "csr":
        order = torch.argsort(dst * n + src)
        d, s = dst[order], src[order]
        crow = torch.zeros(n + 1, dtype=torch.int64, device=device)
        crow[1:] = torch.cumsum(torch.bincount(d, minlength=n), 0)
        wo = torch.nn.Parameter(w.detach()[order].clone())
        def step(r, x, wo=wo):
            mat = torch.sparse_csr_tensor(crow, s, wo, size=(n, n))
            return torch.tanh(torch.sparse.mm(mat, r.T).T + x)
        return step, [wo]
    if backend == "csr_fn":
        order = torch.argsort(dst * n + src)
        d, s_ = dst[order], src[order]
        crow = torch.zeros(n + 1, dtype=torch.int64, device=device)
        crow[1:] = torch.cumsum(torch.bincount(d, minlength=n), 0)
        perm = torch.argsort(s_ * n + d)   # edges sorted by (src, dst) = CSR of W^T
        crowT = torch.zeros(n + 1, dtype=torch.int64, device=device)
        crowT[1:] = torch.cumsum(torch.bincount(s_, minlength=n), 0)
        wo = torch.nn.Parameter(w.detach()[order].clone())
        def step(r, x, wo=wo):
            return torch.tanh(CsrMM.apply(wo, r.T.contiguous(), crow, s_, d, crowT, d[perm], perm, n).T + x)
        return step, [wo]
    if backend == "index_add":
        def step(r, x, w=w):
            out = torch.zeros_like(r)
            out.index_add_(1, dst, r[:, src] * w)
            return torch.tanh(out + x)
        return step, [w]
    raise ValueError(backend)


def estimate_bytes(backend, n, batch, per, backward):
    """Conservative peak-memory estimate (bytes) used to skip cases that break the RAM cap."""
    edges = n * n // 2 if per is None else n * per
    if backend == "dense":
        return int(n * n * 4 * (3 if backward else 1.5) + K * batch * n * 4 * 4 + (edges * 16 if n <= 4000 else 0))
    base = edges * 64 + K * batch * n * 4 * 6
    if backend == "index_add":
        base += K * batch * edges * 4 * (3 if backward else 1)
    if backend in ("csr", "csr_fn"):
        base += batch * edges * 4 * 3
    return int(base)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def time_case(backend, n, batch, per, device, reps, backward):
    gen = torch.Generator().manual_seed(0)
    if backend == "dense" and n > 4000:
        dst = src = torch.zeros(1, dtype=torch.int64); val = torch.zeros(1)
    else:
        dst, src, val = make_edges(n, per, gen)
    step, params = build(backend, n, dst, src, val, device)
    x = torch.randn(batch, n, device=device) * 0.1
    r0 = torch.zeros(batch, n, device=device)

    def run():
        r = r0
        for _ in range(K):
            r = step(r, x)
        if backward:
            r.square().sum().backward()
            for p in params:
                p.grad = None
        sync(device)

    run(); run()  # warmup
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); run(); t.append(time.perf_counter() - t0)
    t.sort()
    return 1000 * t[len(t) // 2], int(dst.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer reps and skip 20k backward")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    mb.start_watchdog()
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    results, rows = [], []
    for size, (n, batch, per) in SIZES.items():
        for dev in devices:
            for backend in ("dense", "csr", "csr_fn", "index_add"):
                for backward in (False, True):
                    if size == "20k" and backend == "dense" and backward:
                        continue  # 1.6 GB weight + grad: not viable on 8 GB, forward-only is reported
                    if a.quick and size == "20k" and (backend == "dense" or backward):
                        continue
                    need = estimate_bytes(backend, n, batch, per, backward)
                    reps = 3 if a.quick else (5 if n >= 20000 else 15)
                    rec = dict(size=size, n=n, batch=batch, device=dev, backend=backend,
                               mode="fwd+bwd" if backward else "fwd", ms=None, error=None)
                    rec["est_gb"] = round(need / mb.GB, 2)
                    if not mb.check_fits(need, f"{size}/{dev}/{backend}/{rec['mode']}"):
                        rec["error"] = "skipped: over RAM budget"
                        rec["skipped"] = True
                        results.append(rec)
                        print(f"{size} {dev} {backend} {rec['mode']}: {rec['error']}", flush=True)
                        continue
                    try:
                        rec["ms"], rec["edges"] = time_case(backend, n, batch, per, torch.device(dev), reps, backward)
                    except Exception as e:  # unsupported op on this device etc.
                        rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:100]}"
                    rec["rss_gb_after"] = round(mb.current_rss() / mb.GB, 2)
                    results.append(rec)
                    gc.collect()
                    print(f"{size} {dev} {backend} {rec['mode']}: {rec['ms'] or rec['error']} (rss {rec['rss_gb_after']} GB)", flush=True)
    best = {}
    for size in SIZES:
        cand = [r for r in results if r["size"] == size and r["mode"] == "fwd+bwd" and r["ms"] is not None
                and (size == "360" or r["backend"] != "dense")]  # dense at >=4k ignores sparsity (16M+ params)
        if cand:
            b = min(cand, key=lambda r: r["ms"])
            best[size] = {"device": b["device"], "backend": b["backend"], "ms": b["ms"]}
    print(f"\npeak RSS {mb.peak_rss() / mb.GB:.2f} GB (cap {mb.max_ram_bytes() / mb.GB:.1f} GB)")
    args_out = {"peak_rss_gb": round(mb.peak_rss() / mb.GB, 2), "ram_cap_gb": mb.max_ram_bytes() / mb.GB, "torch": torch.__version__, "K": K, "quick": a.quick, "results": results, "best_train": best,
                "recommended_device": best.get("4k", {}).get("device", "cpu")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(args_out, indent=1))
    print("\n| size | batch | device | backend | fwd ms | fwd+bwd ms |\n|---|---|---|---|---|---|")
    keys = sorted({(r["size"], r["device"], r["backend"]) for r in results}, key=lambda k: (list(SIZES).index(k[0]), k[1], k[2]))
    for size, dev, be in keys:
        def cell(mode):
            r = next((r for r in results if (r["size"], r["device"], r["backend"], r["mode"]) == (size, dev, be, mode)), None)
            if r is None:
                return "skipped"
            return f"{r['ms']:.2f}" if r["ms"] is not None else ("over RAM budget" if r.get("skipped") else "unsupported")
        print(f"| {size} | {SIZES[size][1]} | {dev} | {be} | {cell('fwd')} | {cell('fwd+bwd')} |")
    print("\nbest (fwd+bwd):", best)


if __name__ == "__main__":
    main()
