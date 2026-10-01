"""Train/held-out token sets drawn from the corpus manifest splits.

Tokens are whitespace-separated pieces of each snippet with a length in
``[min_len, max_len]`` using only keyboard-supported characters. Held-out
tokens come from ``test`` split files and exclude any token also seen in
``train`` files, so evaluation strings are never training strings.
"""

from __future__ import annotations

import hashlib
import json

from backend.corpus import get_target, list_targets
from backend.keyboard import CHAR_TO_KEY


def _tokens(split: str, min_len: int, max_len: int) -> set[str]:
    found: set[str] = set()
    for item in list_targets(split):
        for token in get_target(item["id"])["text"].split():
            if min_len <= len(token) <= max_len and all(c in CHAR_TO_KEY for c in token):
                found.add(token)
    return found


def token_splits(min_len: int = 3, max_len: int = 8) -> dict[str, list[str]]:
    train = _tokens("train", min_len, max_len)
    heldout = _tokens("test", min_len, max_len) - train
    return {"train": sorted(train), "heldout": sorted(heldout)}


def single_characters(tokens: list[str]) -> list[str]:
    """Curriculum stage 0: every character that appears in the training tokens."""
    return sorted({c for token in tokens for c in token})


def splits_hash(splits: dict[str, list[str]]) -> str:
    return hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
    s = token_splits()
    print(json.dumps({"train": len(s["train"]), "heldout": len(s["heldout"]),
                      "sample_train": s["train"][:10], "sample_heldout": s["heldout"][:10],
                      "sha256": splits_hash(s)}, indent=2))
