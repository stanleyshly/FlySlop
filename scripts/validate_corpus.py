"""Validate corpus provenance, split isolation, syntax, and elaboration."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.corpus import get_target, list_targets
from backend.judge import validate_sv


def validate() -> None:
    targets = list_targets()
    if not targets:
        raise SystemExit("Corpus is empty")

    group_splits: dict[str, set[str]] = defaultdict(set)
    authored = 0
    for item in targets:
        if not item.get("split") or not item.get("similarity_group"):
            raise SystemExit(f"Missing split metadata: {item['id']}")
        group_splits[item["similarity_group"]].add(item["split"])
        source = get_target(item["id"])
        text = source["text"]
        actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual_hash != item["sha256"]:
            raise SystemExit(f"Digest mismatch: {item['id']}")
        result = validate_sv(text)
        if not result["syntax"]["passed"]:
            raise SystemExit(f"Syntax failed: {item['id']}: {result['syntax']['diagnostics']}")
        if item["validation_level"] == "standalone_module" and not result["elaboration"]["passed"]:
            raise SystemExit(f"Elaboration failed: {item['id']}: {result['elaboration']['diagnostics']}")
        if item.get("source_repo") == "FlySlop authored corpus":
            authored += 1
        errors = sum(d["severity"].endswith("Error") for d in result["elaboration"]["diagnostics"])
        print(f"OK {item['id']}: syntax=pass elaboration={'pass' if not errors else 'fail'} sha256={actual_hash[:12]}")

    leaked = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaked:
        raise SystemExit(f"Similarity groups cross train/test splits: {leaked}")
    if authored < 20:
        raise SystemExit(f"Expected at least 20 authored targets, found {authored}")
    print(f"Validated {len(targets)} targets ({authored} authored), split groups={len(group_splits)}")


if __name__ == "__main__":
    validate()
