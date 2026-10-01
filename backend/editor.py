"""Text editor state machine behind the laptop keys (contract C2 effects, C4 events).

The buffer is a string; ``cursor`` (the selection head) and ``anchor`` are character
offsets ``0..len(text)``. A selection exists when ``anchor != cursor``. The editor knows
nothing about physics: it maps ``(key id, held modifiers)`` to an edit.

Conventions the contract leaves open (fixed here, tested):

* A plain move with a selection moves the head by one step from where it is, then
  collapses the selection (anchor := head). Left/Right do not jump to the selection edge.
* Shift+move keeps the anchor where the selection began and moves the head.
* Up on the first line and Down on the last line are no-ops. Otherwise the head goes to
  ``min(goal column, line length)`` on the target line; the goal column is remembered
  across consecutive Up/Down moves and reset by every other action.
* ``op`` is ``none`` whenever text, cursor and anchor are all unchanged.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Iterable

from .keyboard import BY_ID, MODIFIER_KEYS, SHIFT_KEYS, TAB_TEXT, key_char

OPS = ("insert", "backspace", "delete", "move", "select", "none")
MOVE_KEYS = {"ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"}


def buffer_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EditorEvent:
    """The editor half of a C4 ``key`` event."""
    key: str
    modifiers: list
    char: str
    op: str
    cursor_before: int
    cursor_after: int
    selection_before: dict | None
    selection_after: dict | None
    text: str
    text_hash: str
    buffer_hash: str

    def as_dict(self) -> dict:
        return asdict(self)


class Editor:
    def __init__(self, text: str = "", cursor: int | None = None, anchor: int | None = None):
        self.text = text
        self.cursor = len(text) if cursor is None else cursor
        self.anchor = self.cursor if anchor is None else anchor
        if not (0 <= self.cursor <= len(text) and 0 <= self.anchor <= len(text)):
            raise ValueError("cursor/anchor out of range")
        self.goal_col: int | None = None

    @property
    def selection(self) -> dict | None:
        if self.anchor == self.cursor:
            return None
        return {"anchor": self.anchor, "head": self.cursor}

    @property
    def selected_text(self) -> str:
        lo, hi = sorted((self.anchor, self.cursor))
        return self.text[lo:hi]

    def _line_start(self, pos: int) -> int:
        return self.text.rfind("\n", 0, pos) + 1

    def _line_end(self, pos: int) -> int:
        end = self.text.find("\n", pos)
        return len(self.text) if end < 0 else end

    def _replace_selection(self, replacement: str) -> None:
        lo, hi = sorted((self.anchor, self.cursor))
        self.text = self.text[:lo] + replacement + self.text[hi:]
        self.cursor = self.anchor = lo + len(replacement)

    def _move_head(self, key: str, fn: bool) -> None:
        text, pos = self.text, self.cursor
        vertical = False
        if key == "ArrowLeft":
            new = self._line_start(pos) if fn else max(0, pos - 1)
        elif key == "ArrowRight":
            new = self._line_end(pos) if fn else min(len(text), pos + 1)
        else:
            vertical = True
            start = self._line_start(pos)
            goal = pos - start if self.goal_col is None else self.goal_col
            if key == "ArrowUp":
                new = pos
                if start > 0:
                    prev_start = self._line_start(start - 1)
                    new = prev_start + min(goal, start - 1 - prev_start)
            else:
                end = self._line_end(pos)
                new = pos
                if end < len(text):
                    nxt_end = self._line_end(end + 1)
                    new = end + 1 + min(goal, nxt_end - (end + 1))
            self.goal_col = goal
        if not vertical:
            self.goal_col = None
        self.cursor = new

    def apply(self, key: str, mods: Iterable[str] = (), char: str | None = None) -> EditorEvent:
        """Apply one key press. ``char`` overrides the inserted text (legacy replays / latched Shift)."""
        if key not in BY_ID:
            raise ValueError(f"unknown key {key!r}")
        mods = list(mods)
        shift = any(m in SHIFT_KEYS for m in mods)
        fn = "Fn" in mods
        before = (self.text, self.cursor, self.anchor)
        cursor_before, selection_before = self.cursor, self.selection
        inserted = ""
        op = "none"
        if key in MOVE_KEYS:
            self._move_head(key, fn)
            if not shift:
                self.anchor = self.cursor
            op = "select" if shift else "move"
        elif key == "Backspace":
            self.goal_col = None
            if self.anchor != self.cursor:
                self._replace_selection("")
            elif fn:
                if self.cursor < len(self.text):
                    self.text = self.text[:self.cursor] + self.text[self.cursor + 1:]
            elif self.cursor > 0:
                self.text = self.text[:self.cursor - 1] + self.text[self.cursor:]
                self.cursor = self.anchor = self.cursor - 1
            op = "delete" if fn else "backspace"
        elif key in MODIFIER_KEYS:
            pass
        else:
            inserted = key_char(key, mods) if char is None else char
            if inserted:
                self.goal_col = None
                self._replace_selection(inserted)
                op = "insert"
        if (self.text, self.cursor, self.anchor) == before:
            op = "none"
        return EditorEvent(key=key, modifiers=mods, char=inserted, op=op, cursor_before=cursor_before,
                           cursor_after=self.cursor, selection_before=selection_before,
                           selection_after=self.selection, text=self.text, text_hash=buffer_hash(self.text),
                           buffer_hash=buffer_hash(self.text))


__all__ = ["Editor", "EditorEvent", "OPS", "TAB_TEXT", "buffer_hash"]
