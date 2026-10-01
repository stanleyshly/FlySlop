"""P7b: physical-eval harness (scripted expert through MuJoCo, both action modes)."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from training.physical_eval import Unsupported, compile_commands, type_tokens, verify_events, write_replay
from backend.keyplan import expand_tokens


class PhysicalEval(unittest.TestCase):
    def test_text_with_shift_and_newline_both_modes(self):
        for mode in ("mn", "claw"):
            r = type_tokens("Hi\nx", "expert", mode, seed=0)
            self.assertEqual(r["text"], "Hi\nx", mode)
            self.assertTrue(r["exact"] and r["contact_verified"] and r["complete"], mode)
            self.assertEqual(r["success_rate"], 1.0)
            self.assertEqual(r["slips"], [])
            self.assertGreater(r["ticks"], 0)

    def test_edit_tokens_arrows_backspace_tab(self):
        toks = ["ab", "<ENTER>", "cd", "<LEFT>", "<LEFT>", "<BS>", "<UP>", "<TAB>", "x"]
        r = type_tokens(toks, "expert", "mn")
        self.assertTrue(r["exact"], r["text"])
        self.assertTrue(r["contact_verified"])
        self.assertEqual(r["text"], r["expected"]["text"])

    def test_held_shift_selection_is_relatched(self):
        r = type_tokens(["hello", "<SEL_START>", "<LEFT>", "<LEFT>", "<SEL_END>", "X"], "expert", "mn")
        self.assertEqual(r["text"], "helX")
        self.assertTrue(r["exact"])

    def test_fn_chord_unsupported_is_reported(self):
        # Legacy compile (chords=False) and the held-only mode still report Fn chords instead of typing them.
        with self.assertRaises(Unsupported):
            compile_commands(expand_tokens(["ab", "<HOME>"]))
        r = type_tokens(["ab", "<DEL>"], "expert", "mn", recover=False, chords=False)
        self.assertIsNotNone(r["unsupported"])
        self.assertFalse(r["exact"])
        r = type_tokens(["ab", "<DEL>"], "expert", "mn", recover=False, fn_mode="held")
        self.assertIsNotNone(r["unreachable"])
        self.assertFalse(r["exact"])

    def test_slip_recovery(self):
        r = type_tokens("Hi there\nyo", "expert", "mn", seed=0, noise=0.05)
        self.assertGreaterEqual(len(r["slips"]), 1)
        self.assertGreaterEqual(r["replans"], 1)
        self.assertLess(r["success_rate"], 1.0)
        self.assertTrue(r["text_match"] and r["contact_verified"])

    def test_replay_verification_and_tamper(self):
        r = type_tokens("Ab", "expert", "mn")
        events = r["events"]
        self.assertEqual(verify_events(events, "Ab"), (True, None))
        drop = [e for e in events if not (e["type"] == "contact_onset" and e["key_id"] == "a")]
        ok, err = verify_events(drop, "Ab")
        self.assertFalse(ok)
        self.assertIn("contact", err.lower())
        forged = copy.deepcopy(events)
        next(e for e in forged if e["type"] == "key")["char"] = "z"
        self.assertFalse(verify_events(forged, "Ab")[0])
        self.assertFalse(verify_events(events, "AB")[0])

    def test_write_replay_json(self):
        r = type_tokens("ab", "expert", "mn", record=True)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.json"
            replay = write_replay(r, path)
            loaded = json.loads(path.read_text())
        self.assertEqual(loaded["final_text"], "ab")
        self.assertEqual(loaded["target"], "ab")
        for key in ("metadata", "frames", "events", "layout", "validation", "final_text_hash"):
            self.assertIn(key, loaded)
        self.assertGreater(len(replay["frames"]["rows"]), 10)


if __name__ == "__main__":
    unittest.main()
