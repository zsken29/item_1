from __future__ import annotations

import re
from dataclasses import dataclass

from lop.data.prompts import expected_answer


@dataclass(frozen=True)
class EvaluationResult:
    correct: bool
    expected: str
    predicted: str
    metric: str


def score_response(sample: dict, response: str) -> EvaluationResult:
    dataset = sample["dataset"]
    expected = expected_answer(sample)
    if dataset in {"mme", "pope"}:
        predicted = _extract_yes_no(response)
        return EvaluationResult(
            correct=predicted == _extract_yes_no(expected),
            expected=expected,
            predicted=predicted,
            metric="yes_no_accuracy",
        )
    if dataset in {"mmbench", "mmmu"}:
        predicted = _extract_option(response)
        return EvaluationResult(
            correct=predicted == _extract_option(expected),
            expected=expected,
            predicted=predicted,
            metric="option_accuracy",
        )
    predicted = _normalize_text(response)
    return EvaluationResult(
        correct=predicted == _normalize_text(expected),
        expected=expected,
        predicted=predicted,
        metric="exact_match",
    )


def _extract_yes_no(text: str) -> str:
    normalized = _normalize_text(text)
    tokens = normalized.split()
    if not tokens:
        return ""
    first = tokens[0]
    if first in {"yes", "y"}:
        return "yes"
    if first in {"no", "n"}:
        return "no"
    if "yes" in tokens:
        return "yes"
    if "no" in tokens:
        return "no"
    return first


def _extract_option(text: str) -> str:
    match = re.search(r"\b([A-D])\b", text.upper())
    if match:
        return match.group(1)
    normalized = _normalize_text(text)
    return normalized[:1].upper()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().strip(".,;:!?'\"")).strip()
