"""Device selection for Torch models: CUDA, then MPS, then CPU when set to auto."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

BENCH_PATH = Path(__file__).resolve().parents[2] / "data" / "connectome" / "device_bench.json"
DEVICES = ("auto", "cpu", "cuda", "mps")


@lru_cache(maxsize=4)
def load_bench(path: str | Path = BENCH_PATH) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _size_key(n_neurons: int | None) -> str:
    if n_neurons is None or n_neurons <= 1000:
        return "360" if n_neurons is not None else "4k"
    return "4k" if n_neurons <= 8000 else "20k"


def mps_available() -> bool:
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def recommended(n_neurons: int | None = None, path: str | Path = BENCH_PATH) -> dict:
    """{'device','backend'} recorded as fastest (fwd+bwd) for the nearest benchmarked size."""
    bench = load_bench(path)
    rec = (bench or {}).get("best_train", {}).get(_size_key(n_neurons))
    return dict(rec) if rec else {"device": "cpu", "backend": "csr_fn" if (n_neurons or 0) > 1000 else "dense"}


def resolve_device(pref: str = "auto", n_neurons: int | None = None, path: str | Path = BENCH_PATH) -> str:
    """Resolve a device preference; auto prefers CUDA, then MPS, then CPU.

    ``n_neurons`` and ``path`` remain accepted for compatibility with older callers;
    the benchmark cache no longer overrides the live hardware choice.
    """
    pref = (pref or "auto").lower()
    if pref not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {pref!r}")
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if cuda_available() else "cpu"
    if pref == "mps":
        return "mps" if mps_available() else "cpu"
    if cuda_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"
