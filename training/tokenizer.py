"""Small pure-Python BPE over printable ASCII (contract C3).

Id layout: 0-15 fixed action/edit/special tokens, 16-31 reserved, 32-126 the 95 printable ASCII
characters (id = 32 + offset, i.e. id 32 is the space), 127+ learned merges.  Newline and tab are
never inside a BPE token: ``encode`` emits <ENTER> (4) for ``\\n`` and <TAB> (5) for ``\\t``
(decoded as four spaces).  Merges never cross a pre-token boundary (word / punctuation run /
space run), so every BPE token expands to plain keystrokes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

SPECIALS = ["<PAD>", "<BOS>", "<EOS>", "<RUN>", "<ENTER>", "<TAB>", "<BS>", "<DEL>",
            "<LEFT>", "<RIGHT>", "<UP>", "<DOWN>", "<HOME>", "<END>", "<SEL_START>", "<SEL_END>"]
PAD, BOS, EOS, RUN, ENTER, TAB = 0, 1, 2, 3, 4, 5
BASE = 32
CHARS = "".join(chr(c) for c in range(32, 127))
_PRETOKEN = re.compile(r" ?[A-Za-z0-9_]+| ?[^A-Za-z0-9_\s]+| +")
_TAB_WIDTH = 4


def normalize(text: str) -> str:
    """CRLF -> LF; tab -> four spaces; characters outside printable ASCII/newline become '?'."""
    text = text.replace("\r\n", "\n").replace("\t", " " * _TAB_WIDTH)
    return "".join(c if c == "\n" or 32 <= ord(c) < 127 else "?" for c in text)


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges: list[tuple[int, int]] = [tuple(m) for m in (merges or [])]  # type: ignore[misc]
        self._rebuild()

    def _rebuild(self) -> None:
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.id_to_str: dict[int, str] = {BASE + i: c for i, c in enumerate(CHARS)}
        for i, (a, b) in enumerate(self.merges):
            self.id_to_str[BASE + len(CHARS) + i] = self.id_to_str[a] + self.id_to_str[b]
        self.str_to_id = {s: i for i, s in self.id_to_str.items()}
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return BASE + len(CHARS) + len(self.merges)

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 1024, min_count: int = 2) -> "BPETokenizer":
        if vocab_size < BASE + len(CHARS):
            raise ValueError("vocab_size below base alphabet")
        words: Counter[str] = Counter()
        for t in texts:
            for line in normalize(t).split("\n"):
                # characters outside printable ASCII are dropped from training text
                words.update(_PRETOKEN.findall("".join(c for c in line if c in CHARS)))
        seqs = {w: [BASE + CHARS.index(c) for c in w] for w in words}
        merges: list[tuple[int, int]] = []
        while BASE + len(CHARS) + len(merges) < vocab_size:
            pairs: Counter[tuple[int, int]] = Counter()
            for w, s in seqs.items():
                for p in zip(s, s[1:]):
                    pairs[p] += words[w]
            if not pairs:
                break
            # deterministic tie-break: highest count, then smallest pair
            best, n = min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))
            if n < min_count:
                break
            new = BASE + len(CHARS) + len(merges)
            merges.append(best)
            for w, s in seqs.items():
                if len(s) < 2:
                    continue
                out, i = [], 0
                while i < len(s):
                    if i + 1 < len(s) and (s[i], s[i + 1]) == best:
                        out.append(new)
                        i += 2
                    else:
                        out.append(s[i])
                        i += 1
                seqs[w] = out
        return cls(merges)

    # ------------------------------------------------------------------ encode/decode
    def _encode_word(self, w: str) -> list[int]:
        hit = self._cache.get(w)
        if hit is not None:
            return hit
        s = [BASE + CHARS.index(c) for c in w]
        while len(s) > 1:
            best = min(((self.rank.get(p, 1 << 30), i) for i, p in enumerate(zip(s, s[1:]))))
            if best[0] == 1 << 30:
                break
            r, i = best
            s[i:i + 2] = [BASE + len(CHARS) + r]
        self._cache[w] = s
        return s

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        """Encode text. Raises ValueError on characters outside printable ASCII/newline/tab."""
        ids = [BOS] if bos else []
        for li, line in enumerate(text.replace("\r\n", "\n").split("\n")):
            if li:
                ids.append(ENTER)
            for part_i, part in enumerate(line.split("\t")):
                if part_i:
                    ids.append(TAB)
                bad = next((c for c in part if not 32 <= ord(c) < 127), None)
                if bad is not None:
                    raise ValueError(f"unsupported character {bad!r}")
                for w in _PRETOKEN.findall(part):
                    ids += self._encode_word(w)
        if eos:
            ids.append(EOS)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Inverse of encode. Non-text control tokens (PAD/BOS/EOS/RUN/cursor edits) are skipped."""
        out = []
        for i in ids:
            if i == ENTER:
                out.append("\n")
            elif i == TAB:
                out.append(" " * _TAB_WIDTH)
            elif i in self.id_to_str:
                out.append(self.id_to_str[i])
        return "".join(out)

    # ------------------------------------------------------------------ persistence
    def to_json(self) -> dict:
        return {"type": "flyslop-bpe", "version": 1, "base": BASE, "chars": CHARS,
                "specials": SPECIALS, "merges": [list(m) for m in self.merges]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=0) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("chars") != CHARS or d.get("base") != BASE:
            raise ValueError("tokenizer file incompatible with contract C3 layout")
        return cls([tuple(m) for m in d["merges"]])

    def hash(self) -> str:
        blob = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


def compression_ratio(tok: BPETokenizer, texts: Iterable[str]) -> float:
    """Characters (newline and tab counted as one) per token."""
    chars = toks = 0
    for t in texts:
        t = normalize(t)
        chars += len(t)
        toks += len(tok.encode(t))
    return chars / max(toks, 1)


DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus" / "tokenizer" / "bpe.json"

if __name__ == "__main__":
    from training import datasets

    ap = argparse.ArgumentParser(description="Train the BPE on the train split of the public+authored data.")
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--private", action="store_true", help="also train on data/private lab records")
    a = ap.parse_args()
    recs = datasets.load_all(include_private=a.private)
    sp = datasets.split_records(recs, seed=a.seed)
    texts = lambda rs: [normalize(r["prompt"] + "\n" + r["code"]) for r in rs]  # noqa: E731
    tok = BPETokenizer.train(texts(sp["train"]), vocab_size=a.vocab)
    tok.save(a.out)
    print(f"vocab={tok.vocab_size} hash={tok.hash()[:16]}")
    for k in ("train", "val", "test"):
        print(k, f"{compression_ratio(tok, texts(sp[k])):.3f} chars/token")
