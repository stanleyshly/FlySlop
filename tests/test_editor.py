"""Editor (contract C4), held modifiers in the keyboard layer (C2), and editor-based replay."""

import copy
import random
import unittest

from backend.editor import Editor, buffer_hash
from backend.embodiment import annotate_key_event, apply_key, replay_text, text_hash
from backend.keyboard import BY_ID, CHAR_TO_KEY, ContactKeyboard, key_char


class RefModel:
    """Independent (row, col) model over a list of lines."""

    def __init__(self):
        self.lines = [""]
        self.cur = (0, 0)
        self.anc = (0, 0)
        self.goal = None

    def text(self):
        return "\n".join(self.lines)

    def off(self, rc):
        return sum(len(l) + 1 for l in self.lines[:rc[0]]) + rc[1]

    def rc(self, off):
        for r, l in enumerate(self.lines):
            if off <= len(l):
                return (r, off)
            off -= len(l) + 1

    def replace(self, s):
        lo, hi = sorted((self.off(self.anc), self.off(self.cur)))
        t = self.text()
        self.lines = (t[:lo] + s + t[hi:]).split("\n")
        self.cur = self.anc = self.rc(lo + len(s))

    def step(self, key, shift, fn):
        r, c = self.cur
        if key in ("ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"):
            if key == "ArrowLeft":
                new = (r, 0) if fn else (self.rc(max(0, self.off(self.cur) - 1)))
                self.goal = None
            elif key == "ArrowRight":
                new = (r, len(self.lines[r])) if fn else self.rc(min(len(self.text()), self.off(self.cur) + 1))
                self.goal = None
            else:
                goal = c if self.goal is None else self.goal
                self.goal = goal
                nr = r + (-1 if key == "ArrowUp" else 1)
                new = (nr, min(goal, len(self.lines[nr]))) if 0 <= nr < len(self.lines) else self.cur
            self.cur = new
            if not shift:
                self.anc = new
            return
        self.goal = None
        if key == "Backspace":
            if self.anc == self.cur:
                o = self.off(self.cur)
                if fn:
                    if o < len(self.text()):
                        self.anc = self.cur
                        self.cur = self.rc(o + 1)
                        self.replace("")
                        self.cur = self.anc = self.rc(o)
                elif o > 0:
                    self.anc = self.rc(o - 1)
                    self.replace("")
            else:
                self.replace("")
        else:
            self.replace(key_char(key, ["ShiftLeft"] if shift else []))


PLAIN_KEYS = ["a", "b", "1", ";", "Space", "Enter", "Tab", "Backspace", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]


class EditorTest(unittest.TestCase):
    def test_insert_replaces_selection(self):
        e = Editor("hello")
        for _ in range(3):
            e.apply("ArrowLeft", ["ShiftLeft"])
        self.assertEqual(e.selection, {"anchor": 5, "head": 2})
        ev = e.apply("x")
        self.assertEqual((e.text, e.cursor, e.selection, ev.op), ("hex", 3, None, "insert"))

    def test_backspace_and_forward_delete(self):
        e = Editor("abc", cursor=1)
        self.assertEqual(e.apply("Backspace").op, "backspace")
        self.assertEqual((e.text, e.cursor), ("bc", 0))
        self.assertEqual(e.apply("Backspace").op, "none")
        ev = e.apply("Backspace", ["Fn"])
        self.assertEqual((e.text, e.cursor, ev.op), ("c", 0, "delete"))
        e.apply("Backspace", ["Fn"])
        self.assertEqual(e.apply("Backspace", ["Fn"]).op, "none")
        s = Editor("abcd", cursor=1, anchor=3)
        s.apply("Backspace", ["Fn"])
        self.assertEqual((s.text, s.cursor, s.selection), ("ad", 1, None))

    def test_shift_extends_and_plain_move_collapses(self):
        e = Editor("abcdef", cursor=3)
        e.apply("ArrowRight", ["ShiftRight"])
        ev = e.apply("ArrowRight", ["ShiftRight"])
        self.assertEqual(ev.op, "select")
        self.assertEqual(ev.selection_before, {"anchor": 3, "head": 4})
        self.assertEqual(ev.selection_after, {"anchor": 3, "head": 5})
        e.apply("ArrowLeft", ["ShiftLeft"])
        e.apply("ArrowLeft", ["ShiftLeft"])
        e.apply("ArrowLeft", ["ShiftLeft"])
        self.assertEqual(e.selection, {"anchor": 3, "head": 2})
        ev = e.apply("ArrowLeft")
        self.assertEqual((ev.op, e.cursor, e.selection), ("move", 1, None))

    def test_up_down_goal_column(self):
        e = Editor("abcdef\nab\nabcdef", cursor=5)
        e.apply("ArrowDown")
        self.assertEqual(e.cursor, 9)                    # clamped to the short line
        e.apply("ArrowDown")
        self.assertEqual(e.cursor - 10, 5)               # goal column restored
        e.apply("ArrowUp")
        e.apply("ArrowLeft")
        e.apply("ArrowDown")
        self.assertEqual(e.cursor - 10, 1)               # horizontal move reset the goal
        self.assertEqual(Editor("ab", 1).apply("ArrowUp").op, "none")
        self.assertEqual(Editor("ab", 1).apply("ArrowDown").op, "none")

    def test_home_end_enter_tab(self):
        e = Editor("ab\ncd", cursor=4)
        e.apply("ArrowLeft", ["Fn"])
        self.assertEqual(e.cursor, 3)
        e.apply("ArrowRight", ["Fn", "ShiftLeft"])
        self.assertEqual((e.cursor, e.selection), (5, {"anchor": 3, "head": 5}))
        e.apply("Enter")
        self.assertEqual(e.text, "ab\n\n")
        ev = e.apply("Tab")
        self.assertEqual((ev.char, e.text), ("    ", "ab\n\n    "))
        self.assertEqual(ev.buffer_hash, buffer_hash(e.text))
        self.assertEqual(ev.text_hash, text_hash(e.text))

    def test_noop_keys_and_bad_key(self):
        e = Editor("ab")
        for key in ("ShiftLeft", "Fn", "F1", "Escape", "CapsLock", "ControlLeft"):
            self.assertEqual(e.apply(key).op, "none")
        with self.assertRaises(ValueError):
            e.apply("NotAKey")

    def test_legacy_apply_key(self):
        self.assertEqual(apply_key("ab", "c", "c"), "abc")
        self.assertEqual(apply_key("ab", "a", "A"), "abA")
        self.assertEqual(apply_key("ab", "Backspace", ""), "a")
        self.assertEqual(apply_key("ab", "ArrowLeft", ""), "ab")

    def test_randomized_against_reference_model(self):
        rng = random.Random(1234)
        for trial in range(300):
            e, ref = Editor(), RefModel()
            for _ in range(rng.randint(1, 60)):
                key = rng.choice(PLAIN_KEYS)
                fn = key in ("Backspace", "ArrowLeft", "ArrowRight") and rng.random() < 0.3
                shift = key.startswith("Arrow") and rng.random() < 0.5 or (key in "ab1;" and rng.random() < 0.3)
                mods = (["ShiftLeft"] if shift else []) + (["Fn"] if fn else [])
                before = e.text
                ev = e.apply(key, mods)
                ref.step(key, shift, fn)
                self.assertEqual(e.text, ref.text(), (trial, key, mods))
                self.assertEqual((e.cursor, e.anchor), (ref.off(ref.cur), ref.off(ref.anc)), (trial, key, mods))
                self.assertTrue(0 <= e.cursor <= len(e.text) and 0 <= e.anchor <= len(e.text))
                self.assertEqual(ev.cursor_after, e.cursor)
                self.assertEqual(ev.text_hash, text_hash(e.text))
                if ev.op == "none":
                    self.assertEqual(before, e.text)
                if ev.op in ("insert", "backspace", "delete"):
                    self.assertTrue(e.selection is None)


def press(kb, tick, key_id, foot, events, hold=False):
    """Drive a foot onto the key centre; returns after `release` unless hold."""
    k = BY_ID[key_id]
    x, y = k.x + k.width / 2, k.y + k.height / 2
    events += kb.update(tick, x, y, 0.6, foot=foot)
    events += kb.update(tick + 1, x, y, -0.1, foot=foot)
    if not hold:
        lift(kb, tick + 2, key_id, foot, events)


def lift(kb, tick, key_id, foot, events):
    k = BY_ID[key_id]
    events += kb.update(tick, k.x + k.width / 2, k.y + k.height / 2, 0.6, foot=foot)


def run(events):
    """Annotate every key event with editor fields (in place), as the episode runners do."""
    editor = Editor()
    for e in events:
        if e["type"] == "key":
            annotate_key_event(editor, e)
    return editor


class KeyboardLayerTest(unittest.TestCase):
    def type_text(self, kb, events, text, start=1):
        tick = start
        for ch in text:
            key, shifted = CHAR_TO_KEY[ch]
            assert not shifted
            press(kb, tick, key, "RF", events)
            tick += 4
        return tick

    def test_held_shift_selects_and_replays(self):
        kb, events = ContactKeyboard(shift_mode="held"), []
        tick = self.type_text(kb, events, "abcd")
        press(kb, tick, "ShiftLeft", "LF", events, hold=True)
        self.assertEqual(kb.held_modifiers(), ["ShiftLeft"])
        for i in range(2):
            press(kb, tick + 4 + 4 * i, "ArrowLeft", "RF", events)
        lift(kb, tick + 12, "ShiftLeft", "LF", events)
        self.assertEqual(kb.held_modifiers(), [])
        states = [e["state"] for e in events if e["type"] == "modifier"]
        self.assertEqual(states, ["shift_down", "shift_up"])
        press(kb, tick + 16, "x", "RF", events)      # replaces the selection, plain x
        editor = run(events)
        self.assertEqual(editor.text, "abx")
        keys = [e for e in events if e["type"] == "key"]
        self.assertEqual([e["op"] for e in keys], ["insert"] * 4 + ["select", "select", "insert"])
        self.assertEqual(keys[5]["selection_after"], {"anchor": 4, "head": 2})
        self.assertEqual(keys[4]["modifiers"], ["ShiftLeft"])
        self.assertEqual(replay_text(events), "abx")

    def test_shift_tap_is_not_latched_in_held_mode_but_is_in_latch_mode(self):
        for mode, expected in (("held", "a"), ("latch", "A")):
            kb, events = ContactKeyboard(shift_mode=mode), []
            press(kb, 1, "ShiftLeft", "LF", events)
            press(kb, 6, "a", "RF", events)
            run(events)
            self.assertEqual(replay_text(events), expected, mode)

    def test_shifted_chord_and_explicit_release(self):
        kb, events = ContactKeyboard(shift_mode="held"), []
        press(kb, 1, "ShiftRight", "RF", events, hold=True)
        press(kb, 5, "a", "LF", events)
        self.assertEqual(kb.release(9, "ShiftRight")[0]["state"], "shift_up")
        press(kb, 12, "a", "LF", events)
        run(events)
        self.assertEqual(replay_text(events), "Aa")
        self.assertEqual(kb.release(13, "ShiftRight"), [])

    def test_fn_chords_delete_home_end_and_tab_arrows_emit(self):
        kb, events = ContactKeyboard(shift_mode="held"), []
        tick = self.type_text(kb, events, "abc")
        press(kb, tick, "Fn", "LF", events, hold=True)
        press(kb, tick + 4, "ArrowLeft", "RF", events)          # Home
        press(kb, tick + 8, "Backspace", "RF", events)          # forward delete
        lift(kb, tick + 12, "Fn", "LF", events)
        press(kb, tick + 16, "Tab", "RF", events)
        self.assertEqual(run(events).text, "    bc")
        ops = [e["op"] for e in events if e["type"] == "key"]
        self.assertEqual(ops[3:], ["move", "delete", "insert"])
        self.assertIn("fn_down", [e.get("state") for e in events])
        self.assertEqual(replay_text(events), "    bc")

    def test_contact_ids_unique_and_repeated_on_key(self):
        kb, events = ContactKeyboard(), []
        self.type_text(kb, events, "abc")
        onsets = [e["contact_id"] for e in events if e["type"] == "contact_onset"]
        self.assertEqual(onsets, sorted(set(onsets)))
        self.assertEqual([e["contact_id"] for e in events if e["type"] == "key"], onsets)
        self.assertEqual([e["contact_id"] for e in events if e["type"] == "contact_offset"], onsets)

    def test_replay_rejections(self):
        kb, events = ContactKeyboard(shift_mode="held"), []
        press(kb, 1, "ShiftLeft", "LF", events, hold=True)
        press(kb, 5, "ArrowLeft", "RF", events)
        press(kb, 9, "b", "RF", events)
        lift(kb, 13, "ShiftLeft", "LF", events)
        run(events)
        self.assertEqual(replay_text(events), "B")
        # key without its contact
        with self.assertRaises(ValueError):
            replay_text([e for e in events if e["type"] != "contact_onset"])
        # chord claiming a modifier that is not held
        no_shift = [e for e in events if not (e.get("key_id") == "ShiftLeft")]
        with self.assertRaises(ValueError):
            replay_text(no_shift)
        # wrong contact id
        bad = copy.deepcopy(events)
        next(e for e in bad if e["type"] == "key")["contact_id"] += 100
        with self.assertRaises(ValueError):
            replay_text(bad)
        # tampered editor field
        bad = copy.deepcopy(events)
        next(e for e in bad if e["type"] == "key")["cursor_after"] = 99
        with self.assertRaises(ValueError):
            replay_text(bad)
        # forged shifted character with no modifier
        bad = copy.deepcopy(events)
        key = [e for e in bad if e["type"] == "key"][-1]
        key["modifiers"] = []
        with self.assertRaises(ValueError):
            replay_text(bad)


if __name__ == "__main__":
    unittest.main()
