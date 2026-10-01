"""MaleCNS v1.0 right-front-leg circuit: loading, masks, and ablation variants.

``data/connectome/rf_leg_circuit.json`` is a small derived subset of the
Janelia MaleCNS v1.0 connectome (CC-BY), written by
``scripts/extract_connectome_circuit.py``. It lists the selected neurons
(descending, right-foreleg sensory, T1 interneurons, right-foreleg motor) and
the synaptic edges among them with a predicted-neurotransmitter sign.

The wiring is measured topology. Everything dynamic built on top of it (rates,
weights magnitudes, readouts) is a model and is labelled as such.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CIRCUIT_PATH = ROOT / "data" / "connectome" / "rf_leg_circuit.json"
GROUPS = ("DN", "SN", "IN", "MN")
MASK_MODES = ("connectome", "shuffled", "random_sparse", "dense")


@lru_cache(maxsize=2)
def load_circuit(path: str | Path = CIRCUIT_PATH) -> dict:
    path = Path(path)
    raw = path.read_bytes()
    circuit = json.loads(raw)
    circuit["file_sha256"] = hashlib.sha256(raw).hexdigest()
    return circuit


def group_indices(circuit: dict) -> dict[str, np.ndarray]:
    groups = [n["group"] for n in circuit["neurons"]]
    return {g: np.array([i for i, x in enumerate(groups) if x == g]) for g in GROUPS}


def signed_matrix(circuit: dict) -> np.ndarray:
    """(N, N) signed synapse counts, ``W[post, pre]``."""
    n = len(circuit["neurons"])
    w = np.zeros((n, n), dtype=np.float32)
    for pre, post, count, sign in circuit["edges"]:
        w[post, pre] += sign * count
    return w


def variant_matrix(circuit: dict, mode: str, seed: int = 0) -> np.ndarray:
    """Signed connectivity for an ablation variant, same neuron groups throughout.

    ``connectome``: measured edges. ``shuffled``: within each (pre group, post
    group) block, the measured edge weights are moved to random positions (edge
    count, weight distribution and signs per block preserved; topology
    destroyed). ``random_sparse``: same overall density, random positions,
    weights resampled from the measured edge weights. ``dense``: every allowed
    block fully connected with small random signed weights (no wiring prior).
    """
    if mode not in MASK_MODES:
        raise ValueError(f"mask mode must be one of {MASK_MODES}")
    w = signed_matrix(circuit)
    if mode == "connectome":
        return w
    rng = np.random.default_rng(seed)
    idx = group_indices(circuit)
    out = np.zeros_like(w)
    blocks = [(pre, post) for pre in GROUPS for post in GROUPS]
    if mode == "random_sparse":
        values = w[w != 0]
        allowed = np.zeros_like(w, dtype=bool)
        for pre, post in blocks:
            if np.any(w[np.ix_(idx[post], idx[pre])]):
                allowed[np.ix_(idx[post], idx[pre])] = True
        cells = np.flatnonzero(allowed)
        chosen = rng.choice(cells, size=len(values), replace=False)
        out.ravel()[chosen] = rng.permutation(values)
        return out
    for pre, post in blocks:
        block = w[np.ix_(idx[post], idx[pre])]
        if not np.any(block):
            continue
        if mode == "shuffled":
            flat = block.ravel().copy()
            out[np.ix_(idx[post], idx[pre])] = rng.permutation(flat).reshape(block.shape)
        else:  # dense
            scale = float(np.abs(block[block != 0]).mean())
            out[np.ix_(idx[post], idx[pre])] = rng.normal(0, scale / 4, block.shape)
    return out


def summary(circuit: dict) -> dict:
    idx = group_indices(circuit)
    w = signed_matrix(circuit)
    return {"neurons": {g: int(len(v)) for g, v in idx.items()}, "edges": int(np.count_nonzero(w)),
            "synapses": int(np.abs(w).sum()), "excitatory_fraction": float((w > 0).sum() / max(1, (w != 0).sum()))}
