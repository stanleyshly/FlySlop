"""Seeded error injection for the symbolic stage 4 (buffer errors) and physical slips (key errors).

Everything is deterministic given the seed: buffer injectors take a ``random.Random`` (or build one from
an int seed) and key corruption builds its own generator from ``seed``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .keyboard import CHAR_TO_KEY, EDIT_KEYS, LAYOUT, MODIFIER_KEYS
from .keyplan import KeyCommand

PRINTABLE = "".join(chr(c) for c in range(32, 127))
BUFFER_KINDS = ("wrong", "missing", "duplicate", "indent", "cursor")

_CENTER = {k.id: (k.x + k.width / 2, k.y + k.height / 2) for k in LAYOUT}


def _rng(seed) -> random.Random:
    return seed if isinstance(seed, random.Random) else random.Random(seed)


def _typable(key_id: str) -> bool:
    key = next(k for k in LAYOUT if k.id == key_id)
    return bool(key.output) or key_id in EDIT_KEYS or key_id == "Enter"


def neighbours(key_id: str, radius: float = 1.35) -> list[str]:
    """Meaningful keys whose centres lie within ``radius`` key pitches (sorted by distance, then id)."""
    cx, cy = _CENTER[key_id]
    found = []
    for k in LAYOUT:
        if k.id == key_id or k.id in MODIFIER_KEYS or not _typable(k.id):
            continue
        x, y = _CENTER[k.id]
        d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        if d <= radius:
            found.append((round(d, 6), k.id))
    return [k for _, k in sorted(found)]


NEIGHBOURS = {k.id: neighbours(k.id) for k in LAYOUT if _typable(k.id)}


def neighbour_char(ch: str, rng: random.Random) -> str:
    """A character on a key next to ``ch``'s key (same shift state), else a random printable one."""
    if ch in CHAR_TO_KEY:
        key_id, shifted = CHAR_TO_KEY[ch]
        options = [k for k in NEIGHBOURS.get(key_id, []) if k not in EDIT_KEYS and k != "Enter"]
        if options:
            pick = rng.choice(options)
            other = next(k for k in LAYOUT if k.id == pick)
            out = other.shift_output if shifted else other.output
            if out and out != ch:
                return out
    return rng.choice([c for c in PRINTABLE if c != ch])


# ------------------------------------------------------------------ buffer injectors
# Each takes (text, cursor, rng) and returns (text, cursor, description) or None if inapplicable.

def wrong_char(text, cursor, rng):
    slots = [i for i, c in enumerate(text) if c != "\n"]
    if not slots:
        return None
    i = rng.choice(slots)
    new = neighbour_char(text[i], rng) if rng.random() < 0.7 else rng.choice([c for c in PRINTABLE if c != text[i]])
    return text[:i] + new + text[i + 1:], cursor, f"wrong@{i}:{text[i]!r}->{new!r}"


def missing_char(text, cursor, rng):
    if not text:
        return None
    i = rng.randrange(len(text))
    return text[:i] + text[i + 1:], _shift_cursor(cursor, i, -1), f"missing@{i}:{text[i]!r}"


def duplicate_char(text, cursor, rng):
    if not text:
        return None
    i = rng.randrange(len(text))
    return text[:i] + text[i] + text[i:], _shift_cursor(cursor, i, +1), f"duplicate@{i}:{text[i]!r}"


def bad_indent(text, cursor, rng):
    """Add or remove 1 to 3 leading spaces on a random line."""
    lines = text.split("\n")
    n = rng.randrange(len(lines))
    start = sum(len(l) + 1 for l in lines[:n])
    lead = len(lines[n]) - len(lines[n].lstrip(" "))
    delta = rng.choice([1, 2, 3])
    if lead >= delta and rng.random() < 0.5:
        lines[n] = lines[n][delta:]
        return "\n".join(lines), _shift_cursor(cursor, start, -delta), f"indent@{n}:-{delta}"
    lines[n] = " " * delta + lines[n]
    return "\n".join(lines), _shift_cursor(cursor, start, +delta), f"indent@{n}:+{delta}"


def stray_cursor(text, cursor, rng):
    if not text:
        return None
    new = rng.randrange(len(text) + 1)
    if new == cursor:
        new = (cursor + 1) % (len(text) + 1)
    return text, new, f"cursor:{cursor}->{new}"


def _shift_cursor(cursor, at, delta):
    return max(at, cursor + delta) if cursor > at else cursor


_INJECTORS = {"wrong": wrong_char, "missing": missing_char, "duplicate": duplicate_char,
              "indent": bad_indent, "cursor": stray_cursor}


@dataclass
class Corrupted:
    text: str
    cursor: int
    errors: list = field(default_factory=list)


def corrupt_buffer(text: str, cursor: int | None = None, seed=0, n: int = 1,
                   kinds: Sequence[str] = BUFFER_KINDS) -> Corrupted:
    """Apply ``n`` seeded errors of the given kinds; inapplicable draws are retried a few times."""
    rng = _rng(seed)
    cursor = len(text) if cursor is None else cursor
    out = Corrupted(text, cursor)
    for _ in range(n):
        for _try in range(8):
            res = _INJECTORS[rng.choice(list(kinds))](out.text, out.cursor, rng)
            if res is not None:
                out.text, out.cursor, desc = res
                out.errors.append(desc)
                break
    return out


# ------------------------------------------------------------------ physical slips
def corrupt_keys(commands: Iterable[KeyCommand], p: float, seed=0) -> tuple[list[KeyCommand], list[int]]:
    """Replace each intended key with a layout neighbour with probability ``p``.

    Modifiers and hold/release commands are never touched; the mods of a slipped command are kept.
    Returns ``(new commands, indices that slipped)``; identical for identical ``(commands, p, seed)``.
    """
    rng = random.Random(seed)
    out, slipped = [], []
    for idx, cmd in enumerate(commands):
        draw = rng.random()
        options = NEIGHBOURS.get(cmd.key, []) if not (cmd.hold or cmd.release) and cmd.key not in MODIFIER_KEYS else []
        if options and draw < p:
            pick = options[rng.randrange(len(options))]
            out.append(KeyCommand(pick, cmd.mods))
            slipped.append(idx)
        else:
            out.append(cmd)
    return out, slipped
