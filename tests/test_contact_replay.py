"""Safety properties of the authoritative contact and replay path."""

import copy
import unittest

from backend.embodiment import replay_text, scripted_episode
from backend.keyboard import BY_ID, ContactKeyboard


class ContactReplayTest(unittest.TestCase):
    def test_hover_and_visual_motion_never_type(self) -> None:
        keyboard = ContactKeyboard()
        key = BY_ID["a"]
        x, y = key.x + 0.2, key.y + 0.2
        for tick, z in enumerate((1.0, 0.3, 0.01, 0.3, 1.0), start=1):
            self.assertEqual(keyboard.update(tick, x, y, z), [])

    def test_one_character_per_full_press(self) -> None:
        keyboard = ContactKeyboard()
        key = BY_ID["a"]
        x, y = key.x + 0.2, key.y + 0.2
        events = []
        for tick, z in enumerate((0.6, -0.1, -0.1, -0.1, 0.6), start=1):
            events.extend(keyboard.update(tick, x, y, z))
        self.assertEqual([event["char"] for event in events if event["type"] == "key"], ["a"])
        self.assertEqual([event["type"] for event in events], ["contact_onset", "key", "contact_offset"])

    def test_sliding_between_keys_while_down_does_not_rearm(self) -> None:
        keyboard = ContactKeyboard()
        a, s = BY_ID["a"], BY_ID["s"]
        ax, ay = a.x + 0.2, a.y + 0.2
        sx, sy = s.x + 0.2, s.y + 0.2
        events = keyboard.update(1, ax, ay, -0.1)
        events += keyboard.update(2, sx, sy, -0.1)
        events += keyboard.update(3, sx, sy, -0.1)
        self.assertEqual([e["char"] for e in events if e["type"] == "key"], ["a"])
        keyboard.update(4, sx, sy, 0.6)
        self.assertEqual([e["char"] for e in keyboard.update(5, sx, sy, -0.1) if e["type"] == "key"], ["s"])

    def test_replay_rejects_character_without_contact_and_tampering(self) -> None:
        episode = scripted_episode("test", "module m;\nendmodule\n")
        self.assertEqual(replay_text(episode["events"]), episode["final_text"])
        no_contact = [event for event in episode["events"] if event["type"] != "contact_onset"]
        with self.assertRaises(ValueError):
            replay_text(no_contact)
        tampered = copy.deepcopy(episode["events"])
        next(event for event in tampered if event["type"] == "key")["text"] = "wrong"
        with self.assertRaises(ValueError):
            replay_text(tampered)
        duplicated = copy.deepcopy(episode["events"])
        position = next(i for i, event in enumerate(duplicated) if event["type"] == "key")
        duplicated.insert(position + 1, copy.deepcopy(duplicated[position]))
        with self.assertRaises(ValueError):
            replay_text(duplicated)

    def test_key_events_carry_c4_fields_and_contact_ids(self) -> None:
        episode = scripted_episode("test", "aB")
        onsets = {e["contact_id"]: e for e in episode["events"] if e["type"] == "contact_onset"}
        keys = [e for e in episode["events"] if e["type"] == "key"]
        for event in keys:
            self.assertEqual(onsets[event["contact_id"]]["key_id"], event["key_id"])
            self.assertEqual(event["key"], event["key_id"])
            self.assertEqual(event["buffer_hash"], event["text_hash"])
            self.assertIn(event["op"], ("insert", "backspace", "delete", "move", "select", "none"))
        self.assertEqual(keys[-1]["cursor_after"], 2)
        self.assertEqual(keys[-1]["selection_after"], None)
        forged = copy.deepcopy(episode["events"])
        next(e for e in forged if e["type"] == "key")["contact_id"] += 1000
        with self.assertRaises(ValueError):
            replay_text(forged)

    def test_shifted_character_requires_shift_key_contact(self) -> None:
        episode = scripted_episode("test", "A")
        onsets = [e["key_id"] for e in episode["events"] if e["type"] == "contact_onset"]
        self.assertEqual(onsets, ["ShiftLeft", "a"])
        self.assertEqual(replay_text(episode["events"]), "A")


if __name__ == "__main__":
    unittest.main()
