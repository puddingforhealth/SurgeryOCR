import unittest

from utils.ocr_prediction_policy import conservative_prediction
from tools.compare_ocr_models import comparison_metrics


class PredictionPolicyTests(unittest.TestCase):
    def test_confident_valid_baseline_is_preserved(self):
        result = conservative_prediction("24", 0.99, "29", 0.999, "region1", 1)
        self.assertEqual(result["text"], "24")
        self.assertTrue(result["needs_review"])

    def test_invalid_old_decimal_can_be_replaced(self):
        result = conservative_prediction("323", 0.999, "3.23", 0.8, "region6", 2)
        self.assertEqual(result["text"], "3.23")

    def test_punctuation_is_not_invented(self):
        result = conservative_prediction("3.23", 0.6, "323", 0.99, "region6", 1)
        self.assertEqual(result["text"], "3.23")
        self.assertEqual(result["source"], "baseline_after_invalid_candidate")
        result = conservative_prediction("323", 0.6, "1.223", 0.99, "region6", 1)
        self.assertEqual(result["text"], "323")
        self.assertEqual(result["source"], "both_invalid")
        self.assertTrue(result["needs_review"])

    def test_policy_is_not_applied_to_machine_three(self):
        with self.assertRaisesRegex(ValueError, "machines 1/2"):
            conservative_prediction("22", 0.9, "2.2", 0.9, "INCISION", 3)

    def test_policy_rejects_unverified_fields(self):
        with self.assertRaisesRegex(ValueError, "region"):
            conservative_prediction("22", 0.9, "2.2", 0.9, "new_field", 1)

    def test_paired_improvements_and_regressions_are_both_reported(self):
        rows = [
            {
                "id": "a",
                "label": "1",
                "baseline": "2",
                "candidate": "1",
                "video_id": "v1",
                "selection": "challenge",
            },
            {
                "id": "b",
                "label": "2",
                "baseline": "2",
                "candidate": "3",
                "video_id": "v2",
                "selection": "high_confidence_control",
            },
        ]
        result = comparison_metrics(rows, bootstrap_samples=20)
        self.assertEqual(result["improved"], ["a"])
        self.assertEqual(result["regressed"], ["b"])
        self.assertFalse(result["controls_preserved_in_this_sample"])
        self.assertEqual(result["by_region"]["unspecified"]["baseline"]["samples"], 2)
        self.assertIn("video_cluster_bootstrap_delta_95_interval", result)


if __name__ == "__main__":
    unittest.main()
