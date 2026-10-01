"""P4c: held-modifier chords on the physical fly (two-foreleg chord expert, both action modes)."""

import unittest

from backend.fly_env import FlyTypingEnv, UnreachableChord, normalize_key_items
from backend.keyplan import KeyCommand
from backend.physics import PhysicsParams
from training.physical_eval import type_tokens

FN_TOKENS = {"<DEL>": ["abc", "<LEFT>", "<DEL>"], "<HOME>": ["abc", "<ENTER>", "de", "<HOME>", "X"],
             "<END>": ["ab", "<ENTER>", "cd", "<UP>", "<HOME>", "<END>", "Y"]}


def check(test, r, text=None):
    test.assertTrue(r["exact"], (r["text"], r["expected"]))
    test.assertTrue(r["contact_verified"], r["contact_error"])
    test.assertEqual(r["slips"], [])
    if text is not None:
        test.assertEqual(r["text"], text)


class HeldChords(unittest.TestCase):
    def test_defaults_are_legacy(self):
        p = PhysicsParams()
        self.assertEqual((p.shift_mode, p.fn_mode), ("latch", "held"))

    def test_queue_item_formats(self):
        items = normalize_key_items([("a", True), ("b", False), ("Backspace", ["Fn"]), {"key": "c", "mods": ["ShiftRight"]},
                                     KeyCommand("ShiftLeft", hold=True), KeyCommand("d"), KeyCommand("ShiftLeft", release=True),
                                     KeyCommand("e")])
        self.assertEqual(items, [("a", True, ()), ("b", False, ()), ("Backspace", False, ("Fn",)),
                                 ("c", True, ("ShiftRight",)), ("d", True, ("ShiftLeft",)), ("e", False, ())])
        with self.assertRaises(ValueError):
            normalize_key_items([("a", ["ControlLeft"])])

    def test_shifted_char_in_held_mode_both_action_modes(self):
        for mode in ("mn", "claw"):
            r = type_tokens("aBc(", "expert", mode, seed=1, shift_mode="held")
            check(self, r, "aBc(")
            self.assertEqual(r["shift_mode"], "held")
            keys = [e for e in r["events"] if e["type"] == "key"]
            self.assertEqual([bool(set(e["modifiers"]) & {"ShiftLeft", "ShiftRight"}) for e in keys], [False, True, False, True])
            # contact-verified chord: the modifier's contact is still down when the key's contact begins
            down = set()
            for e in r["events"]:
                if e["type"] == "contact_onset":
                    down.add(e["key_id"])
                elif e["type"] == "contact_offset":
                    down.discard(e["key_id"])
                elif e["type"] == "key":
                    self.assertTrue(set(e["modifiers"]) <= down, (e, down))

    def test_fn_chords_del_home_end(self):
        # Fn is 13 to 14 pitches from Backspace / ArrowLeft / ArrowRight and the forelegs span about 11, so the
        # held chord is unreachable and the eval falls back to the sticky Fn tap (fn_mode="latch"), reporting it.
        for mode in ("mn", "claw"):
            for token, src in FN_TOKENS.items():
                r = type_tokens(src, "expert", mode, seed=2)
                check(self, r)
                self.assertTrue(r["fn_latch_used"], token)
                self.assertIn("Fn", [m for e in r["events"] if e["type"] == "key" for m in e["modifiers"]], token)

    def test_fn_held_is_unreachable_with_numbers(self):
        env = FlyTypingEnv(physics={"shift_mode": "held"})
        for key in ("Backspace", "ArrowLeft", "ArrowRight"):
            with self.assertRaises(UnreachableChord) as ctx:
                env.reset(options={"keys": [(key, False, ("Fn",))]})
            self.assertIn("key pitches", str(ctx.exception))
        with self.assertRaises(UnreachableChord):       # Shift+Fn+key would need three feet down
            FlyTypingEnv(physics={"shift_mode": "held"}).reset(options={"keys": [("ArrowLeft", True, ("Fn",))]})

    def test_held_fn_chord_when_reachable(self):
        env = FlyTypingEnv(action_mode="mn", physics={"shift_mode": "held"}, terminate_on_error=False)
        env.reset(seed=0, options={"keys": [("Space", False, ("Fn",))], "max_steps": 400})
        for _ in range(400):
            _, _, te, tr, info = env.step(env.expert_action())
            if te or tr:
                break
        self.assertTrue(info["exact"])
        key = [e for e in env.events if e["type"] == "key"][0]
        self.assertEqual(key["modifiers"], ["Fn"])

    def test_shift_selection_then_replace_holds_shift(self):
        toks = ["hello", "<SEL_START>", "<LEFT>", "<LEFT>", "<LEFT>", "<SEL_END>", "p"]
        for mode in ("mn", "claw"):
            r = type_tokens(toks, "expert", mode, seed=3)
            check(self, r, "hep")
            self.assertEqual(r["shift_mode"], "held")
            # Shift stays down across the arrows: one Shift contact covers several key events
            ups = [e for e in r["events"] if e["type"] == "modifier" and e["state"] == "shift_up"]
            downs = [e for e in r["events"] if e["type"] == "modifier" and e["state"] == "shift_down"]
            arrows = [e for e in r["events"] if e["type"] == "key" and e["key_id"] == "ArrowLeft"]
            self.assertEqual(len(arrows), 3)
            self.assertLess(len(downs), len(arrows))
            self.assertEqual(len(ups), len(downs))

    def test_selection_up_down_and_shift_fn(self):
        r = type_tokens(["ab", "<ENTER>", "cd", "<SEL_START>", "<UP>", "<LEFT>", "<SEL_END>", "<BS>"], "expert", "mn", seed=4)
        check(self, r)
        r = type_tokens(["ab cd", "<SEL_START>", "<HOME>", "<SEL_END>", "Z"], "expert", "mn", seed=4)
        check(self, r, "Z")

    def test_unreachable_shift_chord_falls_back_to_latch(self):
        # "_" (Shift + a number-row key) is too far from both Shift keys for two forelegs: auto mode latches Shift.
        r = type_tokens("a_b", "expert", "mn", seed=0, shift_mode="held")
        self.assertFalse(r["exact"])
        self.assertIn("Shift+", r["unreachable"])
        r = type_tokens("a_b", "expert", "mn", seed=0)
        check(self, r, "a_b")
        self.assertTrue(r["shift_latch_used"] or r["shift_mode"] == "latch")

    def test_default_latch_unchanged(self):
        r = type_tokens("Hi\nx", "expert", "mn", seed=0)
        check(self, r, "Hi\nx")
        self.assertEqual(r["shift_mode"], "latch")
        self.assertTrue(any(e.get("shift_latched") for e in r["events"] if e["type"] == "key"))


if __name__ == "__main__":
    unittest.main()
