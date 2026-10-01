"""Copy a small, pinned set of standalone SV examples from the local lab checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "lab-group40-fa25"
DESTINATION = PROJECT_ROOT / "data" / "corpus"

# These modules have no external includes or module dependencies.
EXAMPLES = (
    ("byte_enable_decoder", "sim/lab3_mem/wben_decoder.v", "cache byte enable decoder"),
    ("processor_immediate_generator", "sim/lab2_proc/ProcDpathImmGen.v", "TinyRV2 immediate generator"),
    ("processor_drop_unit", "sim/lab2_proc/DropUnit.v", "pipeline drop unit"),
)


def source_commit(source: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()


def extract(source: Path) -> None:
    if not (source / ".git").is_dir():
        raise SystemExit(f"Expected a Git checkout at {source}")

    commit = source_commit(source)
    snippets = DESTINATION / "snippets"
    snippets.mkdir(parents=True, exist_ok=True)
    records = []

    for example_id, relative_path, description in EXAMPLES:
        path = source / relative_path
        original = path.read_bytes()
        if not original.strip():
            raise SystemExit(f"Empty source file: {path}")
        if b"`include" in original:
            raise SystemExit(f"Source has an external include: {path}")
        destination = snippets / f"{example_id}.sv"
        destination.write_bytes(original)
        records.append(
            {
                "id": example_id,
                "description": description,
                "kind": "standalone_module",
                "language": "SystemVerilog",
                "source_repo": "cornell-ece4750/lab-group40-fa25",
                "source_commit": commit,
                "source_path": relative_path,
                "snippet_path": str(destination.relative_to(PROJECT_ROOT)),
                "sha256": hashlib.sha256(original).hexdigest(),
                "rights": "user-provided student lab checkout; no license file found",
            }
        )

    manifest = DESTINATION / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"Extracted {len(records)} examples from {commit} to {snippets}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    extract(args.source.resolve())
