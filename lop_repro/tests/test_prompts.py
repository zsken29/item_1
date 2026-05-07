from __future__ import annotations

import unittest

from lop.data.prompts import build_question, expected_answer


class PromptTest(unittest.TestCase):
    def test_mme_prompt_uses_question(self) -> None:
        sample = {
            "dataset": "mme",
            "fields": {"question": "Is there text?", "answer": "Yes"},
        }

        self.assertEqual(build_question(sample), "Is there text?")
        self.assertEqual(expected_answer(sample), "Yes")

    def test_mmbench_prompt_includes_valid_options(self) -> None:
        sample = {
            "dataset": "mmbench",
            "fields": {
                "question": "Choose.",
                "answer": "A",
                "A": "Alpha",
                "B": "Beta",
                "C": "nan",
                "D": "Delta",
            },
        }

        prompt = build_question(sample)

        self.assertIn("A. Alpha", prompt)
        self.assertIn("B. Beta", prompt)
        self.assertNotIn("C. nan", prompt)
        self.assertIn("Answer with the option letter only.", prompt)


if __name__ == "__main__":
    unittest.main()
