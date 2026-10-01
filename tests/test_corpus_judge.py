import unittest

from backend.corpus import get_target, list_targets
from backend.judge import judge_text, validate_sv


class CorpusJudgeTests(unittest.TestCase):
    def test_manifest_targets_have_matching_hashes(self):
        targets = list_targets()
        self.assertTrue(targets)
        for item in targets:
            target = get_target(item["id"])
            self.assertEqual(target["id"], item["id"])
            self.assertTrue(target["text"])

    def test_exact_and_sv_results_are_separate(self):
        target = get_target("processor_drop_unit")
        result = judge_text(target["text"], target["id"])
        self.assertTrue(result["exact"]["passed"])
        self.assertTrue(result["syntax"]["passed"])
        self.assertTrue(result["elaboration"]["passed"])
        self.assertIn("expected", result["text"])

    def test_malformed_control_fails_parse(self):
        result = validate_sv("module broken(input logic a; assign a = ; endmodule")
        self.assertFalse(result["syntax"]["passed"])

    def test_unresolved_module_fails_elaboration_after_parse(self):
        result = validate_sv("module top; missing_unit u(); endmodule")
        self.assertTrue(result["syntax"]["passed"])
        self.assertFalse(result["elaboration"]["passed"])


if __name__ == "__main__":
    unittest.main()
