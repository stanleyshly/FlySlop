"""NeuroMechFly posing, gait, and laptop layout properties of the scripted demo."""

import unittest

import numpy as np

from backend.embodiment import KEY_TRAVEL, LEGS, scripted_episode
from backend.flybody import default_body
from backend.keyboard import LAYOUT, LAYOUT_WIDTH


def tips(episode):
    frames = episode["frames"]
    rows = np.array(frames["rows"])
    start = frames["fields"].index("LF.tip.x")
    return rows, rows[:, start:start + 18].reshape(len(rows), 6, 3)


class EmbodimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episode = scripted_episode("test", "module m;\n")
        cls.rows, cls.tips = tips(cls.episode)

    def test_every_keyboard_row_spans_the_same_width(self):
        rows = {}
        starts = [0.0, 0.6, 1.6, 2.6, 3.6, 4.6]
        for key in LAYOUT:
            row = max(s for s in starts if s <= key.y)
            rows.setdefault(row, []).append(key.x + key.width)
        self.assertEqual(len(rows), 6)
        for right_edges in rows.values():
            self.assertAlmostEqual(max(right_edges), LAYOUT_WIDTH - 0.05, places=3)

    def test_ik_reaches_posed_targets(self):
        fly = default_body()
        target = fly.tip("RF", fly.rest_q["RF"]) + np.array([0.3, -0.1, 0.2])
        q, error = fly.solve_batch(np.array([LEGS.index("RF")]), target[None])
        self.assertLess(error[0], 1e-4)
        self.assertTrue(np.allclose(fly.tip("RF", q[0]), target, atol=1e-4))

    def test_stance_feet_do_not_slide(self):
        z = self.tips[:, :, 2]
        on_ground = (np.abs(z) < 0.005) | (np.abs(z + 0.04) < 0.005)  # keycap top or deck
        planted = on_ground[1:] & on_ground[:-1]
        slip = np.linalg.norm(self.tips[1:, :, :2] - self.tips[:-1, :, :2], axis=2)[planted]
        self.assertLess(slip.max(), 0.02)

    def test_only_forelegs_press_and_only_through_posed_tips(self):
        onsets = [e for e in self.episode["events"] if e["type"] == "contact_onset"]
        self.assertTrue(onsets)
        self.assertEqual({e["foot"] for e in onsets} - {"LF", "RF"}, set())
        for event in onsets:
            leg = LEGS.index(event["foot"])
            self.assertLessEqual(self.tips[event["tick"], leg, 2], -0.05)
        self.assertGreaterEqual(self.tips[:, :, 2].min(), -KEY_TRAVEL - 1e-3)

    def test_types_with_both_forelegs_mostly_without_walking(self):
        from backend.embodiment import LOOKAHEAD, Planner, key_sequence
        text = "module m;\n  assign y = a & b;\nendmodule\n"
        planner, sequence, walks = Planner(), key_sequence(text), 0
        for i, key_id in enumerate(sequence):
            start = len(planner.phase)
            planner.type_key(key_id, tuple(sequence[i + 1:i + 1 + LOOKAHEAD]))
            walks += "walk" in planner.phase[start:]
        self.assertLess(walks / len(sequence), 0.25, "the fly should mostly reach keys with its legs")
        presses = [e["foot"] for e in self.episode["events"] if e["type"] == "key"]
        self.assertIn("LF", presses)
        self.assertIn("RF", presses)
        self.assertEqual(self.episode["final_text"], "module m;\n")


if __name__ == "__main__":
    unittest.main()
