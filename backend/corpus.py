"""Read the curated transcription corpus and its provenance manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "corpus" / "manifest.jsonl"


def _records() -> list[dict[str, Any]]:
    if not MANIFEST.exists():
        return []
    records = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def list_targets(split: str | None = None) -> list[dict[str, Any]]:
    """Return target metadata (without source text), optionally filtered by split."""
    items = _records()
    if split is not None:
        items = [item for item in items if item.get("split") == split]
    return items


def get_target(target_id: str) -> dict[str, Any]:
    """Return manifest metadata and UTF-8 source text for one target."""
    item = next((x for x in _records() if x["id"] == target_id), None)
    if item is None:
        raise KeyError(f"Unknown corpus target: {target_id}")
    path = (ROOT / item["snippet_path"]).resolve()
    if ROOT not in path.parents:
        raise ValueError(f"Corpus path escapes project root: {item['snippet_path']}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if item.get("sha256") and digest != item["sha256"]:
        raise ValueError(f"Digest mismatch for corpus target {target_id}")
    return {**item, "text": raw.decode("utf-8")}
