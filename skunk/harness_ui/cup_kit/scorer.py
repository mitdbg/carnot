"""Deterministic scoring for the OfficeQA Grounded Reasoning Cup.

Wraps the vendored upstream ``reward.py``. Any two ``Scorer`` instances
constructed with identical arguments produce byte-identical
``ScoreResult`` values for the same inputs (modulo ``latency_ms``).

This is byte-identical to the server-side scorer; a contract test in
the cup's research CI compares both wrappers across a fixed test grid
and fails on drift.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import ClassVar

from cup_kit.reward import score_answer as _score_answer

_DASH_LIKE = {
    "−": "-",  # MINUS SIGN
    "–": "-",  # EN DASH
    "—": "-",  # EM DASH
}

_PAREN_RE = re.compile(r"\([^)]*\)")
_WHITESPACE_RE = re.compile(r"\s+")
_THOUSANDS_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")


@dataclass(frozen=True)
class ScoreResult:
    correct: bool
    points_awarded: float
    rationale: str
    scorer_version: str
    latency_ms: float


class Scorer:
    VERSION: ClassVar[str] = "officeqa.cup.scoring.v4"

    def __init__(self, tolerance: float = 0.0) -> None:
        if not 0.0 <= tolerance <= 1.0:
            raise ValueError(f"tolerance must be in [0.0, 1.0]; got {tolerance!r}")
        self._tolerance = tolerance

    @property
    def tolerance(self) -> float:
        return self._tolerance

    def score(self, canonical: str, submitted: str) -> ScoreResult:
        if not canonical:
            raise ValueError("canonical answer must be non-empty")

        start_ns = time.monotonic_ns()
        correct = _score_answer(canonical, submitted, self._tolerance) == 1.0
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0

        return ScoreResult(
            correct=correct,
            points_awarded=1.0 if correct else 0.0,
            rationale="" if correct else f"Answer is incorrect at {self._tolerance} tolerance.",
            scorer_version=self.VERSION,
            latency_ms=elapsed_ms,
        )

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        normalized = text
        for dash, replacement in _DASH_LIKE.items():
            normalized = normalized.replace(dash, replacement)
        normalized = _THOUSANDS_RE.sub(lambda m: m.group().replace(",", ""), normalized)
        normalized = normalized.strip().lower().strip('"').strip("'")
        normalized = _PAREN_RE.sub("", normalized).strip()
        normalized = _WHITESPACE_RE.sub(" ", normalized)
        return normalized

    def is_same_answer(self, a: str, b: str) -> bool:
        return self.normalize(a) == self.normalize(b)
