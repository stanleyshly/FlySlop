"""Cached ``.npz`` load/save for extracted connectome circuits, with a deterministic content hash.

The hash covers only the numeric/label arrays (sorted by name, dtype and shape included), never
timestamps or file paths, so re-running an extractor on the same inputs gives the same hash.
Metadata (config, source hashes, timings) lives in a JSON string stored beside the arrays and is
NOT hashed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HASH_KEY = "content_hash"
META_KEY = "meta_json"
_RESERVED = {HASH_KEY, META_KEY}


def content_hash(arrays: dict[str, np.ndarray]) -> str:
    """sha256 over arrays in sorted-name order (name, dtype, shape, raw bytes)."""
    digest = hashlib.sha256()
    for name in sorted(k for k in arrays if k not in _RESERVED):
        arr = np.asarray(arrays[name])
        if arr.dtype.kind == "O":
            raise TypeError(f"array {name!r} has object dtype; cast to a fixed dtype")
        if arr.dtype.kind in "US":     # width-independent: length-prefixed utf-8 per element
            digest.update(f"{name}|str|{arr.shape}|".encode())
            for item in arr.ravel().tolist():
                raw = item if isinstance(item, bytes) else str(item).encode("utf-8")
                digest.update(len(raw).to_bytes(4, "little") + raw)
        else:
            arr = np.ascontiguousarray(arr)
            digest.update(f"{name}|{arr.dtype.str}|{arr.shape}|".encode())
            digest.update(arr.tobytes())
    return digest.hexdigest()


def save_circuit(path: str | Path, arrays: dict[str, np.ndarray], meta: dict | None = None) -> str:
    """Write arrays + meta to a compressed npz; returns the content hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = content_hash(arrays)
    payload = {k: np.asarray(v) for k, v in arrays.items() if k not in _RESERVED}
    payload[HASH_KEY] = np.array(digest)
    payload[META_KEY] = np.array(json.dumps(meta or {}, sort_keys=True))
    np.savez_compressed(path, **payload)
    return digest


def load_circuit(path: str | Path, verify: bool = True) -> tuple[dict[str, np.ndarray], dict, str]:
    """Load (arrays, meta, stored_hash). ``verify`` recomputes the hash and raises on mismatch."""
    with np.load(Path(path), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k not in _RESERVED}
        stored = str(z[HASH_KEY]) if HASH_KEY in z.files else ""
        meta = json.loads(str(z[META_KEY])) if META_KEY in z.files else {}
    if verify and stored and content_hash(arrays) != stored:
        raise ValueError(f"{path}: content hash mismatch (file corrupted or edited)")
    return arrays, meta, stored
