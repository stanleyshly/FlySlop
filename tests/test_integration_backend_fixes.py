import unittest
from types import SimpleNamespace

import numpy as np

from backend.embodiment import replay_text, annotate_key_event, text_hash
from backend.editor import Editor
from backend.physics import PhysicalKeyboard, PhysicsParams
from backend.server import make_replay


def _tap_fn_then(key_id: str, fn_mode: str) -> list[dict]:
    ids = ["Fn", key_id]
    sim = SimpleNamespace(params=PhysicsParams(fn_mode=fn_mode), key_ids=ids)
    kb = PhysicalKeyboard(sim)
    p = sim.params
    deep, shallow = p.press_depth + 1.0, p.release_depth - 1.0
    out = []
    out += kb.update(0, np.array([deep, shallow]), {0: "L"})
    out += kb.update(1, np.array([shallow, shallow]), {})
    out += kb.update(2, np.array([shallow, deep]), {1: "R"})
    return out


class FnLatchedFlag(unittest.TestCase):
    def test_physics_emits_fn_latched_not_shift_latched(self):
        keys = [e for e in _tap_fn_then("ArrowLeft", "latch") if e["type"] == "key"]
        self.assertEqual(len(keys), 1)
        self.assertTrue(keys[0].get("fn_latched"))
        self.assertNotIn("shift_latched", keys[0])
        self.assertIn("Fn", keys[0]["modifiers"])

    def _events(self, extra):
        ed = Editor()
        ev = {"tick": 0, "type": "contact_onset", "key_id": "a", "foot": "L", "contact_id": 0}
        key = {"tick": 1, "type": "key", "key_id": "a", "foot": "L", "modifiers": ["Fn"],
               "char": "a", "contact_id": 0}
        annotate_key_event(ed, key)
        key["text"], key["text_hash"] = ed.text, text_hash(ed.text)
        key.update(extra)
        return [ev, key], ed.text

    def test_replay_accepts_fn_latched_and_legacy_flag(self):
        for flag in ({"fn_latched": True}, {"shift_latched": True}):
            events, text = self._events(flag)
            self.assertEqual(replay_text(events), text)

    def test_replay_still_checks_without_flag(self):
        events, _ = self._events({})
        with self.assertRaises(ValueError):
            replay_text(events)


class KeyEventCursorFields(unittest.TestCase):
    def test_key_events_carry_cursor_and_selection(self):
        for source in ("kinematic", "physics"):
            keys = [e for e in make_replay("fly_demo", source)["events"] if e["type"] == "key"]
            self.assertTrue(keys)
            for e in keys:
                self.assertIsInstance(e["cursor_after"], int)
                sel = e["selection_after"]
                self.assertTrue(sel is None or set(sel) == {"anchor", "head"})
            self.assertEqual(keys[-1]["cursor_after"], len(make_replay("fly_demo", source)["final_text"]))


if __name__ == "__main__":
    unittest.main()
