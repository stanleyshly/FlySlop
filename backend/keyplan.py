"""Action tokens to key commands (contract C3 expansion, C2 schema) and a diff-to-keystrokes oracle.

``expand_tokens`` maps thinker tokens to :class:`KeyCommand` s. ``oracle_edits`` plans a
minimal-ish token sequence that turns ``buffer`` into ``target`` through the real
:class:`~backend.editor.Editor`, by simulating candidate plans and keeping the cheapest
valid one. Cost is the number of key commands (a Shift hold or release counts as one).
"""

from __future__ import annotations

import copy
import difflib
from dataclasses import dataclass
from typing import Iterable, Sequence

from .editor import Editor, EditorEvent
from .keyboard import CHAR_TO_KEY, TAB_TEXT

SHIFT = "ShiftLeft"
SPECIALS = ("<PAD>", "<BOS>", "<EOS>", "<RUN>", "<ENTER>", "<TAB>", "<BS>", "<DEL>", "<LEFT>", "<RIGHT>",
            "<UP>", "<DOWN>", "<HOME>", "<END>", "<SEL_START>", "<SEL_END>")   # ids 0..15 (C3)
NO_KEYS = {"<PAD>", "<BOS>", "<EOS>", "<RUN>"}


@dataclass(frozen=True)
class KeyCommand:
    key: str
    mods: tuple = ()
    hold: bool = False
    release: bool = False

    def as_dict(self) -> dict:
        if self.release:
            return {"key": self.key, "release": True}
        return {"key": self.key, "mods": list(self.mods), "hold": self.hold}


_EDIT = {
    "<ENTER>": KeyCommand("Enter"), "<TAB>": KeyCommand("Tab"), "<BS>": KeyCommand("Backspace"),
    "<DEL>": KeyCommand("Backspace", ("Fn",)), "<LEFT>": KeyCommand("ArrowLeft"),
    "<RIGHT>": KeyCommand("ArrowRight"), "<UP>": KeyCommand("ArrowUp"), "<DOWN>": KeyCommand("ArrowDown"),
    "<HOME>": KeyCommand("ArrowLeft", ("Fn",)), "<END>": KeyCommand("ArrowRight", ("Fn",)),
    "<SEL_START>": KeyCommand(SHIFT, hold=True), "<SEL_END>": KeyCommand(SHIFT, release=True),
}


def char_command(ch: str) -> KeyCommand:
    if ch == "\n":
        return _EDIT["<ENTER>"]
    if ch not in CHAR_TO_KEY:
        raise ValueError(f"character {ch!r} has no key")
    key, shifted = CHAR_TO_KEY[ch]
    return KeyCommand(key, (SHIFT,) if shifted else ())


def text_to_keys(text: str) -> list[KeyCommand]:
    """One self-contained key command per character (shifted chars are Shift chords)."""
    return [char_command(ch) for ch in text]


def _token_string(tok, tokenizer) -> str:
    if isinstance(tok, str):
        return tok
    tok = int(tok)
    if 0 <= tok < len(SPECIALS):
        return SPECIALS[tok]
    if tokenizer is None:
        raise ValueError("a tokenizer is required to expand BPE token ids")
    if hasattr(tokenizer, "decode_token"):
        return tokenizer.decode_token(tok)
    return tokenizer.decode([tok])


def expand_token(tok, tokenizer=None) -> list[KeyCommand]:
    s = _token_string(tok, tokenizer)
    if s in NO_KEYS:
        return []
    if s in _EDIT:
        return [_EDIT[s]]
    return text_to_keys(s)


def expand_tokens(tokens: Iterable, tokenizer=None) -> list[KeyCommand]:
    """Tokens (ids or strings) to key commands. Ids below 16 are the C3 control tokens; ids >= 32 need
    ``tokenizer.decode_token(id)`` (or ``decode([id])``). Strings naming a control token are control tokens,
    any other string is literal text."""
    out: list[KeyCommand] = []
    for tok in tokens:
        out.extend(expand_token(tok, tokenizer))
    return out


def keystroke_cost(tokens: Iterable, tokenizer=None) -> int:
    return len(expand_tokens(tokens, tokenizer))


def apply_commands(editor: Editor, commands: Iterable[KeyCommand], held: bool = False) -> list[EditorEvent]:
    """Drive an Editor with key commands; a Shift hold stays down until its release command."""
    events = []
    for cmd in commands:
        if cmd.key == SHIFT and cmd.hold:
            held = True
        elif cmd.release:
            held = False
        else:
            mods = list(cmd.mods)
            if held and SHIFT not in mods:
                mods.append(SHIFT)
            events.append(editor.apply(cmd.key, mods))
    return events


def apply_tokens(editor: Editor, tokens: Iterable, tokenizer=None) -> Editor:
    apply_commands(editor, expand_tokens(tokens, tokenizer))
    return editor


# --------------------------------------------------------------------------- oracle

def _type_tokens(text: str) -> list[str]:
    out: list[str] = []
    run = ""

    def flush():
        nonlocal run
        if run:
            out.append(run)
            run = ""

    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\n":
            flush(); out.append("<ENTER>")
        elif text.startswith(TAB_TEXT, i):
            flush(); out.append("<TAB>"); i += len(TAB_TEXT); continue
        elif ch == "<":
            flush(); out.append("<")
        elif ch in CHAR_TO_KEY:
            run += ch
        else:
            raise ValueError(f"character {ch!r} has no key")
        i += 1
    flush()
    return out


def _sim(ed: Editor, tokens: Sequence[str]) -> Editor:
    return apply_tokens(ed, tokens)


def _fresh(text: str, pos: int) -> Editor:
    return Editor(text, pos, pos)


def _nav_tokens(text: str, src: int, dst: int, simple: bool = False) -> list[str]:
    """Cheapest verified plain-move sequence taking the cursor from src to dst (collapsed editor)."""
    if src == dst:
        return []
    best = ["<RIGHT>"] * (dst - src) if dst > src else ["<LEFT>"] * (src - dst)
    if simple:
        return best
    ls = text.rfind("\n", 0, dst) + 1
    le = text.find("\n", dst)
    le = len(text) if le < 0 else le
    src_line = text.count("\n", 0, src)
    dst_line = text.count("\n", 0, dst)
    for pre in ([], ["<HOME>"], ["<END>"]):
        ed = _sim(_fresh(text, src), pre)
        line = text.count("\n", 0, ed.cursor)
        vert = ["<DOWN>" if dst_line > line else "<UP>"] * abs(dst_line - line)
        if not pre and not vert:
            continue
        ed = _sim(ed, vert)
        if text.count("\n", 0, ed.cursor) != dst_line:
            continue
        head = pre + vert
        p = ed.cursor
        opts = [(["<RIGHT>"] * (dst - p)) if dst >= p else (["<LEFT>"] * (p - dst)),
                ["<HOME>"] + ["<RIGHT>"] * (dst - ls), ["<END>"] + ["<LEFT>"] * (le - dst)]
        tail = min(opts, key=len)
        cand = head + tail
        if len(cand) < len(best) and _sim(_fresh(text, src), cand).cursor == dst:
            best = cand
    return best


def _reach(ed: Editor, dst: int, simple: bool) -> list[str]:
    """Tokens leaving the cursor at dst with the selection collapsed."""
    toks = _nav_tokens(ed.text, ed.cursor, dst, simple)
    if toks or ed.anchor == ed.cursor:
        return toks
    wig = ["<RIGHT>", "<LEFT>"] if ed.cursor < len(ed.text) else ["<LEFT>", "<RIGHT>"]
    return wig


def _selected_move(ed: Editor, src: int, dst: int, simple: bool) -> list[str]:
    return _nav_tokens(ed.text, src, dst, simple)


def _candidates(ed: Editor, i: int, j: int, new: str):
    """Yield token plans replacing text[i:j] by ``new`` from the editor's current state."""
    typed = _type_tokens(new)
    length = j - i
    for simple in (False, True):
        if ed.anchor != ed.cursor and (min(ed.anchor, ed.cursor), max(ed.anchor, ed.cursor)) == (i, j):
            yield (typed or ["<BS>"])
        if length == 0:
            yield _plan(ed, [("reach", i, simple)]) + typed
            continue
        yield _plan(ed, [("reach", j, simple)]) + ["<BS>"] * length + typed
        yield _plan(ed, [("reach", i, simple)]) + ["<DEL>"] * length + typed
        for a, b in ((i, j), (j, i)):
            w = _fresh_copy(ed)
            first = _reach(w, a, simple)
            _sim(w, first)
            mid = _selected_move(w, a, b, simple)
            yield first + ["<SEL_START>"] + mid + ["<SEL_END>"] + (typed or ["<BS>"])


def _fresh_copy(ed: Editor) -> Editor:
    return copy.copy(ed)


def _plan(ed: Editor, steps) -> list[str]:
    return [t for _kind, dst, simple in steps for t in _reach(ed, dst, simple)]


def _apply_ops(buffer, cursor, anchor, ops):
    """Greedy per-op planning; ops are (i, j, new, mode) with i, j in original buffer coordinates.
    Returns (tokens, final editor)."""
    ed = Editor(buffer, cursor, anchor)
    out: list[str] = []
    shift = 0
    for i, j, new, mode in ops:
        if mode == "forward":
            i2, j2 = i + shift, j + shift
        else:
            i2, j2 = i, j
        expected = ed.text[:i2] + new + ed.text[j2:]
        best = None
        for cand in _candidates(ed, i2, j2, new):
            w = copy.copy(ed)
            try:
                _sim(w, cand)
            except ValueError:
                continue
            if w.text != expected or w.anchor != w.cursor:
                continue
            if best is None or len(expand_tokens(cand)) < len(expand_tokens(best)):
                best = cand
        if best is None:                      # cannot happen: the simple reach + BS plan is always valid
            raise RuntimeError("no valid plan for edit")
        _sim(ed, best)
        out += best
        shift += len(new) - (j - i)
    return out, ed


def _norm_selection(cursor, selection):
    if selection is None:
        return cursor, cursor
    if isinstance(selection, dict):
        return selection["head"], selection["anchor"]
    anchor, head = selection
    return head, anchor


def oracle_edits(buffer: str, cursor: int | None, selection, target: str) -> list[str]:
    """Action tokens that turn ``buffer`` into ``target`` starting from ``cursor`` / ``selection``
    (``{"anchor","head"}``, ``(anchor, head)`` or None; the head overrides ``cursor``)."""
    return oracle_plan(buffer, cursor, selection, target)[0]


def oracle_plan(buffer: str, cursor: int | None, selection, target: str) -> tuple[list[str], int]:
    """Return ``(tokens, keystroke_cost)``."""
    if cursor is None:
        cursor = len(buffer)
    head, anchor = _norm_selection(cursor, selection)
    if buffer == target and head == anchor:
        return [], 0
    sm = difflib.SequenceMatcher(None, buffer, target, autojunk=False)
    ops = [(i1, i2, target[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
    variants = []
    if ops:
        variants.append([(i, j, n, "forward") for i, j, n in ops])
        variants.append([(i, j, n, "reverse") for i, j, n in reversed(ops)])
    variants.append([(0, len(buffer), target, "forward")] if buffer != target else [])
    best = None
    for v in variants:
        toks, ed = _apply_ops(buffer, head, anchor, v)
        if not toks and head != anchor:      # only a stray selection differed
            toks = _reach(ed, head, True)
        if ed.text == target and (best is None or len(expand_tokens(toks)) < len(expand_tokens(best))):
            best = toks
    return best, len(expand_tokens(best))


def naive_cost(buffer: str, target: str) -> int:
    """Baseline: delete every character with Backspace from the end, then retype the target."""
    return len(buffer) + len(text_to_keys(target))


def diff_size(buffer: str, target: str) -> int:
    """Characters inserted plus deleted according to the difflib opcodes."""
    sm = difflib.SequenceMatcher(None, buffer, target, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
