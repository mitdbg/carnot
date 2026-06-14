"""Deterministic scoring for the OfficeQA Grounded Reasoning Cup.

Delegates to :mod:`officeqa.cup.scoring.reward` — a verbatim vendored copy of
the public ``reward.py`` that participants score against
(https://github.com/databricks/officeqa/blob/main/reward.py; pinned SHA in
``reward.py``'s module header). Wrapping it gives us a stable, versioned
interface consumed by:

- The live FastAPI submission path, which calls ``Scorer.score`` once per
  submission and records the result in SQLite.
- The offline MLflow replay path, which reruns the same ``Scorer.score`` over
  event traces to reproduce the leaderboard.

Because both paths call the same method, the live and replayed scores are
byte-identical. Any behavior-affecting change — including re-vendoring a new
upstream SHA — requires bumping ``Scorer.VERSION``; tests in this package pin
the version so drift is caught.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import ClassVar

from cup_kit.reward import _VALID_THOUSANDS_RE, score_answer

# Characters that render as a minus sign but have distinct code points.
# ``reward.normalize_text`` only normalizes U+2212 and U+2212-like dashes for
# scoring; the resubmit-on-change check covers the most common extras here so
# that visually-identical answers normalize equal.
_DASH_LIKE = {
    "\u2212": "-",  # MINUS SIGN
    "\u2013": "-",  # EN DASH
    "\u2014": "-",  # EM DASH
}

# Parenthetical annotations (e.g., "(OASI)", "(FY)") are stripped before text
# comparison. Matches ``reward.fuzzy_match_answer`` behavior.
_PAREN_RE = re.compile(r"\([^)]*\)")

# Collapse all internal whitespace runs to a single space.
_WHITESPACE_RE = re.compile(r"\s+")

# Match the thousands-separator pattern only — never strip commas from text
# (e.g., "Portland, Oregon"). Compiled from ``reward._VALID_THOUSANDS_RE`` so the
# scoring and resubmit-detection paths share a single source of truth and cannot
# drift; ``reward.py`` is itself synced byte-for-byte from the upstream public
# reward. The ``(?<![\d.])`` lookbehind and ``(?!\d)`` lookahead keep a thousands
# group from starting after a digit or decimal point, so a numeric list element
# like "44.00,231.52" is not corrupted into "44.00231.52" (a decimal before a
# comma means the comma is a list delimiter, not a thousands separator).
_THOUSANDS_RE = re.compile(_VALID_THOUSANDS_RE)


@dataclass(frozen=True)
class ScoreResult:
    """Outcome of scoring a single submission.

    Attributes:
        correct: Whether the submission matches the canonical answer under the
            configured tolerance.
        points_awarded: Base points (1.0 for correct, 0.0 otherwise). The speed
            bonus is awarded separately by the server's atomic SQL recompute.
        rationale: Human-readable justification from the underlying tolerance
            metric. Useful for admin-facing dispute review.
        scorer_version: Version of the scorer used. Logged with every score so
            replays can verify they are reproducing the live event faithfully.
        latency_ms: Wall-clock duration of the ``score`` call, in milliseconds.
    """

    correct: bool
    points_awarded: float
    rationale: str
    scorer_version: str
    latency_ms: float


class Scorer:
    """Deterministic scorer for the OfficeQA Grounded Reasoning Cup.

    Any two ``Scorer`` instances constructed with identical arguments produce
    byte-identical ``ScoreResult`` values for the same inputs, modulo
    ``latency_ms``. The default ``tolerance=0.0`` gives strict numeric matching,
    which is the behavior needed for the live competition; other tolerances are
    supported for evaluation experiments but are never used in production.
    """

    VERSION: ClassVar[str] = "officeqa.cup.scoring.v5"
    """Stable version string. Bump on any behavior-affecting change.

    Logged with every ``ScoreResult`` so offline replays can verify they used
    the same code path as the live event. The test suite pins this string so
    an unintentional behavior change surfaces as a test failure.

    Versions:

    - v1: internal ``ToleranceMetric`` with an "unable to determine"
      short-circuit; never shipped past pre-PR development.
    - v2: switched to the vendored public ``reward.py`` for ``score()``;
      ``normalize()`` (used for resubmit no-op detection) still stripped
      every comma — including bare commas in text — which collapsed
      "Portland, Oregon" and "Portland Oregon" to the same key.
    - v3: ``normalize()`` narrows the comma strip to thousands-separator
      patterns only, mirroring ``reward.extract_numbers_with_context``.
      Score outcomes are unchanged across v2 → v3 (``score()`` was
      already using the vendored helper); only the ``is_same_answer``
      contract changes, so resubmit-with-only-comma-difference now burns
      a token instead of being treated as a no-op. Replay code that
      reproduces resubmit decisions must branch on ``scorer_version``.
    - v4: re-vendored upstream ``reward.py`` with accounting-notation
      normalization (e.g., ``(123)`` → ``-123``) and currency-symbol
      stripping. ``score()`` now calls ``reward.score_answer`` directly
      instead of composing ``extract_final_answer`` + ``fuzzy_match_answer``,
      and the rationale string is the terse upstream form. Edge-case
      verdicts change for any answer involving accounting parens or
      currency-prefixed numbers; everyday integer/decimal matches are
      unchanged from v3.
    - v5: re-vendored upstream ``reward.py`` with a bounded thousands
      regex (a leading lookbehind + trailing lookahead so a thousands
      group cannot start right after a digit or decimal point) and the
      bracketed-numeric-list matching helpers. A numeric-list element
      ending in a decimal (e.g., the canonical ``[44.00,231.52]``) is no
      longer corrupted into ``[44.00231, 52.0]`` by stripping the list
      delimiter as a thousands separator. ``score()`` verdicts change for
      such list answers (they now match correctly), and ``normalize()``
      keys change accordingly (``[44.00,231.52]`` no longer collapses to
      ``[44.00231.52]``). ``normalize()`` also now shares the regex with
      ``reward._VALID_THOUSANDS_RE`` so scoring and resubmit detection
      cannot drift. Everyday scalar matches are unchanged from v4.

      v5 also relaxes the direct-answer guard (``_is_direct_answer_only``).
      The prior version rejected a numeric-only ground truth whenever the
      prediction carried any residual text after removing numbers and a
      fixed set of unit words ("prose outside the answer value"), so a
      submission like ``0.88525 percentage points`` (residual "points"),
      ``2.5 basis points``, ``11.60 million dollars``, or framed prose like
      ``The answer is 5`` was marked incorrect. v5 drops that prose check;
      the only remaining surrounding-content rejection is HTML/XML markup
      (e.g., ``<b>543</b>``). A single-line prediction (<=250 chars)
      containing the correct number now scores correct even with unit
      suffixes or natural-language framing, including hedged answers like
      ``approximately 1.25%``. This affects ``score()`` verdicts for any
      numeric-only canonical whose submission adds surrounding text.
    """

    def __init__(self, tolerance: float = 0.0) -> None:
        if not 0.0 <= tolerance <= 1.0:
            raise ValueError(f"tolerance must be in [0.0, 1.0]; got {tolerance!r}")
        self._tolerance = tolerance

    @property
    def tolerance(self) -> float:
        return self._tolerance

    def score(self, canonical: str, submitted: str) -> ScoreResult:
        """Score a submitted answer against a canonical answer.

        Args:
            canonical: The correct answer for the question. Must be non-empty —
                a missing canonical is a server configuration bug, not a
                scoring outcome, so this method raises rather than returning
                "incorrect".
            submitted: The team's submitted answer. Empty strings are scored as
                incorrect without raising.

        Returns:
            A ``ScoreResult`` with the correctness outcome, rationale, version,
            and measured latency.

        Raises:
            ValueError: If ``canonical`` is empty.
        """
        if not canonical:
            raise ValueError("canonical answer must be non-empty")

        start_ns = time.monotonic_ns()
        # Delegate to upstream ``reward.score_answer`` which combines
        # FINAL_ANSWER extraction, fuzzy_match_answer, and the
        # accounting/currency normalization added in the v4 reward.
        correct = score_answer(canonical, submitted, self._tolerance) == 1.0
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0

        return ScoreResult(
            correct=correct,
            points_awarded=1.0 if correct else 0.0,
            rationale="" if correct else f"Answer is incorrect at {self._tolerance} tolerance.",
            scorer_version=self.VERSION,
            latency_ms=elapsed_ms,
        )

    def normalize(self, text: str) -> str:
        """Return the canonical form used for resubmit-on-change detection.

        Two answers that ``normalize`` to the same string are treated as the
        same submission — a resubmit consumes no token. This mirrors the
        transformations the scorer applies before comparison; see
        ``NORMALIZER_SPEC.md`` for the exact contract teams can rely on.

        Empty input normalizes to ``""``.
        """
        if not text:
            return ""
        normalized = text
        for dash, replacement in _DASH_LIKE.items():
            normalized = normalized.replace(dash, replacement)
        # Strip commas only from thousands-separator patterns (e.g., "1,234"
        # -> "1234"). Bare commas in text like "Portland, Oregon" are left
        # alone so distinct text answers don't collapse to the same key.
        normalized = _THOUSANDS_RE.sub(lambda m: m.group().replace(",", ""), normalized)
        # Case-fold and strip surrounding whitespace + quotes.
        normalized = normalized.strip().lower().strip('"').strip("'")
        # Remove parenthetical annotations like "(OASI)".
        normalized = _PAREN_RE.sub("", normalized).strip()
        # Collapse internal whitespace runs to a single space.
        normalized = _WHITESPACE_RE.sub(" ", normalized)
        return normalized

    def is_same_answer(self, a: str, b: str) -> bool:
        """Whether two answer strings normalize to the same canonical form.

        Used by the server's resubmit handler: if the new answer is the same
        as the prior answer (after normalization), the submission is a no-op
        and no resubmit token is consumed.
        """
        return self.normalize(a) == self.normalize(b)
