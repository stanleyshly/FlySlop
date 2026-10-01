import unittest

from backend.embodiment import replay_text, scripted_episode


class TrainingScaffoldTests(unittest.TestCase):
    def test_oracle_demo_is_contact_backed_and_replayable(self):
        result = scripted_episode("demo", "aA ")
        self.assertEqual(result["final_text"], "aA ")
        self.assertEqual(replay_text(result["events"]), "aA ")
        self.assertEqual(result["metadata"]["mode"], "scripted_kinematic_neuromechfly")
        self.assertTrue(any(event["type"] == "contact_onset" for event in result["events"]))

    def test_gymnasium_env_uses_contact_state_machine(self):
        try:
            from backend.env import KeyboardReachEnv
            from backend.keyboard import BY_ID, CHAR_TO_KEY
        except ImportError as exc:
            if "optional 'gymnasium'" in str(exc):
                self.skipTest("install the training extra to exercise Gymnasium")
            raise
        env = KeyboardReachEnv(targets=("a",), max_steps=10)
        obs, info = env.reset(seed=7, options={"target": "a"})
        self.assertEqual(info["mode"], "state_assisted_kinematic")
        self.assertTrue(env.observation_space.contains(obs))
        key_id, _ = CHAR_TO_KEY["a"]
        key = BY_ID[key_id]
        x, y = key.x + key.width / 2, key.y + key.height / 2
        env.step([x, y, 0.6])
        _obs, _reward, terminated, _truncated, _info = env.step([x, y, -0.1])
        self.assertTrue(terminated)
        self.assertEqual(env.buffer, "a")
        env.close()

    def test_training_oracle_uses_shift_contact(self):
        try:
            from training.evaluate_oracle import episode
        except ImportError as exc:
            if "optional 'gymnasium'" in str(exc):
                self.skipTest("install the training extra to exercise Gymnasium")
            raise
        self.assertTrue(episode("A")["exact"])


if __name__ == "__main__":
    unittest.main()
