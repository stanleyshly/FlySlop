"""Micro-suite loader, metadata and the near-duplicate check against training records.

Licence: every task, reference solution and testbench in this package is original FlySlop work (MIT)."""
from __future__ import annotations

import hashlib
import json
import re

from .tasks_py import TASKS as _PY
from .tasks_sv import TASKS as _SV

LICENCE = "MIT (FlySlop-authored, original)"
HELD_OUT = True     # fixed split: never add these to training data


def load_tasks(lang: str | None = None, max_difficulty: int = 3, max_lines: int | None = None) -> list[dict]:
    """Tasks with ``lines`` (reference length), sorted by (difficulty, lines, id) for the length curriculum."""
    out = []
    for t in _PY + _SV:
        if lang and t["lang"] != lang:
            continue
        t = dict(t, lines=len(t["code"].strip().splitlines()), licence=LICENCE, split="heldout")
        if t["difficulty"] <= max_difficulty and (max_lines is None or t["lines"] <= max_lines):
            out.append(t)
    return sorted(out, key=lambda t: (t["difficulty"], t["lines"], t["id"]))


def split_hash() -> str:
    blob = json.dumps([[t["id"], t["prompt"], t["code"], t["tests"]] for t in sorted(_PY + _SV, key=lambda t: t["id"])])
    return hashlib.sha256(blob.encode()).hexdigest()


def _literal(code: str) -> list[str]:
    return re.findall(r"\w+|\S", code)


def _jac(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def dup_check(records: list[dict] | None = None, threshold: float = 0.8) -> dict:
    """Exact 3-shingle Jaccard of each task's reference against every same-language record.

    ``literal`` keeps identifiers (real copies); ``structural`` collapses them (renamed copies). Tiny programs
    collide structurally by nature, so ``literal`` is the gate. Returns overlaps >= threshold."""
    from training import datasets as ds
    if records is None:
        records = ds.load_all()
    prep = []
    for r in records:
        prep.append((r, ds.shingles(_literal(r["code"])), ds.shingles(ds.normalized_tokens(r["code"], r["lang"]))))
    lit, stru = [], []
    for t in load_tasks():
        tl = ds.shingles(_literal(t["code"]))
        ts = ds.shingles(ds.normalized_tokens(t["code"], t["lang"]))
        for r, rl, rs in prep:
            if r["lang"] != t["lang"]:
                continue
            a, b = _jac(tl, rl), _jac(ts, rs)
            if a >= threshold:
                lit.append({"task": t["id"], "record": r["id"], "jaccard": round(a, 3)})
            if b >= threshold:
                stru.append({"task": t["id"], "record": r["id"], "jaccard": round(b, 3)})
    return {"records_checked": len(records), "threshold": threshold, "literal_overlaps": lit,
            "structural_overlaps": stru, "n_literal": len(lit), "n_structural": len(stru)}
