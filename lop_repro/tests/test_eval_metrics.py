from __future__ import annotations

import unittest

from lop.eval import score_mme, score_response


class EvalMetricTest(unittest.TestCase):
    def test_yes_no_metric(self) -> None:
        sample = {"dataset": "mme", "fields": {"answer": "Yes"}}
        result = score_response(sample, "Yes.")
        self.assertTrue(result.correct)
        self.assertEqual(result.predicted, "yes")

    def test_option_metric(self) -> None:
        sample = {"dataset": "mmbench", "fields": {"answer": "B"}}
        result = score_response(sample, "The answer is B.")
        self.assertTrue(result.correct)
        self.assertEqual(result.predicted, "B")

    def test_mme_accuracy_plus_groups_by_image(self) -> None:
        samples = [
            {
                "dataset": "mme",
                "fields": {"answer": "yes", "category": "existence"},
                "images": [{"path": "a.jpg"}],
            },
            {
                "dataset": "mme",
                "fields": {"answer": "no", "category": "existence"},
                "images": [{"path": "a.jpg"}],
            },
        ]
        score = score_mme(samples, ["yes", "no"])
        self.assertEqual(score.perception, 200.0)
        self.assertEqual(score.total, 200.0)


if __name__ == "__main__":
    unittest.main()
