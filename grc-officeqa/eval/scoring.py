"""Answer scoring for the OfficeQA e2e harness.

Delegates to qatfd's vendored copy of the competition reward
(``qatfd/qatfd/benchmarks/officeqa_scoring.py``, byte-identical v5 body from
cup_kit 0.1.6) — the single scoring implementation in this monorepo. The skunk
*library* deliberately ships no scorer: the old ``skunk.eval.scoring`` module
vendored a stale v4 copy and was removed; harness_ui (which carried cup_kit)
was deleted 2026-07-07.
"""

from __future__ import annotations

import sys
from pathlib import Path

# qatfd is the sibling checkout in the carnot monorepo; put its package root on
# sys.path so the vendored reward imports without qatfd being installed.
_QATFD = str(Path(__file__).resolve().parents[2] / "qatfd")
if _QATFD not in sys.path:
    sys.path.insert(0, _QATFD)

from qatfd.benchmarks.officeqa_scoring import score_answer  # noqa: E402

# The Scorer.VERSION the cup kit's scorer.py wrapper reported for this reward.
SCORER_VERSION: str = "officeqa.cup.scoring.v5"


def score_correct(gold_answer, predicted, tolerance: float = 0.0) -> int:
    """Grade one prediction against the gold answer with the official scorer.

    Returns 1 if ``predicted`` matches ``gold_answer`` at the given tolerance
    (default 0.0 -> 0.0% absolute relative error, the competition's strictest
    threshold), else 0.

    Never raises: a missing/NaN gold answer or an empty prediction scores 0.
    Inputs are coerced to ``str`` so pandas cell values (floats, NaN) are safe.
    """
    gold = "" if gold_answer is None else str(gold_answer)
    pred = "" if predicted is None else str(predicted)
    # A non-empty canonical answer is required; an absent/NaN gold (str of a
    # pandas NaN is "nan") can't be graded, so treat it as incorrect.
    if not gold.strip() or gold.strip().lower() == "nan":
        return 0
    if not pred.strip():
        return 0
    return 1 if score_answer(gold, pred, tolerance) == 1.0 else 0
