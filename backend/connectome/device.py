"""Device selection for connectome runtimes: ``resolve_device("auto"|"cpu"|"mps")``.

``auto`` reads the cached benchmark written by ``scripts/bench_device.py``
(``data/connectome/device_bench.json``). Without the cache it returns ``cpu``
(the measured default: CSR sparse ops are not implemented on MPS).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

BENCH_PATH = Path(__file__).resolve().parents[2] / "data" / "connectome" / "device_bench.json"
DEVICES = ("auto", "cpu", "mps")


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


def recommended(n_neurons: int | None = None, path: str | Path = BENCH_PATH) -> dict:
    """{'device','backend'} recorded as fastest (fwd+bwd) for the nearest benchmarked size."""
    bench = load_bench(path)
    rec = (bench or {}).get("best_train", {}).get(_size_key(n_neurons))
    return dict(rec) if rec else {"device": "cpu", "backend": "csr_fn" if (n_neurons or 0) > 1000 else "dense"}


def resolve_device(pref: str = "auto", n_neurons: int | None = None, path: str | Path = BENCH_PATH) -> str:
    """Return ``"cpu"`` or ``"mps"``. Explicit ``mps`` falls back to ``cpu`` if unavailable."""
    pref = (pref or "auto").lower()
    if pref not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {pref!r}")
    if pref == "cpu":
        return "cpu"
    if pref == "mps":
        return "mps" if mps_available() else "cpu"
    choice = recommended(n_neurons, path)["device"]
    return "mps" if choice == "mps" and mps_available() else "cpu"
