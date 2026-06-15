"""question_explainer — whole-question concept selection. One LLM call reads the
question and selects which canonical references (from the fixed `PRECOMPUTED_CONCEPTS`
catalog) a code-writing agent needs, injected into the compute codegen prompt as
`## Concept references`.

The model only emits the indices of the relevant catalog entries (cheap generation); the
definitions themselves are fixed, authoritative text — no per-question generation."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from skunk.common import ExecutionContext
from skunk.prompted_call import PromptedCall


class ConceptExplanation(BaseModel):
    """One non-obvious concept paired with its operational definition."""

    model_config = ConfigDict(frozen=True)
    concept: str
    explanation: str


# A fixed catalog of canonical formulas for the named / ambiguous operations the benchmark
# asks for — the ones where a defensible-but-wrong variant yields the wrong answer (Zipf
# orientation, Box-Cox form, percentile type, pop-vs-sample std, …). Each pins ONE
# convention per operation, backed by an official US-gov / NIST definition or the
# convention the gold answers actually use (see docs/computation_catalog.md). The
# QuestionExplainer selects the relevant subset per question; the chosen entries become the
# compute `## Concept references` block.
PRECOMPUTED_CONCEPTS: list[tuple[str, str]] = [
    (
        "Zipf / power-law exponent",
        "Rank the values in descending order (ranks 1..n) and run an OLS regression of log(value) on log(rank); the exponent is the absolute value of the slope. Do NOT regress log(rank) on log(value) (that inverts the exponent), and do NOT use the MLE / Hill estimator unless the question explicitly asks for it.",
    ),
    (
        "CAGR (compound annual growth rate)",
        "CAGR = (end / begin) ** (1 / n) - 1, where n is the number of YEAR intervals between the endpoints (end_year - begin_year), not the count of data points.",
    ),
    (
        "Continuously compounded (logarithmic) growth rate",
        "r = ln(end / begin) / n over n periods; annualize by multiplying the per-period log change by periods-per-year. Relation to CAGR: r = ln(1 + CAGR).",
    ),
    (
        "Annual decay / growth factor",
        "The per-year multiplicative factor is (1 + CAGR); it is below 1 for a declining series. The reciprocal 1 / (1 + CAGR) is a discount factor, not the decay factor.",
    ),
    (
        "Geometric mean",
        "exp(mean(ln x)) = (product of x) ** (1 / n), over strictly positive values. For an average growth RATE, take the geometric mean of the growth factors (1 + gᵢ) and subtract 1.",
    ),
    (
        "Arc elasticity (midpoint method)",
        "[(Q2 - Q1) / ((Q1 + Q2) / 2)] / [(P2 - P1) / ((P1 + P2) / 2)] — each percent change uses the average of its two endpoints in the denominator (the factor of 2 cancels). Keep the sign unless the question asks for the absolute value.",
    ),
    (
        "Population vs sample standard deviation / variance",
        "Population divides by N (numpy default, ddof=0); sample divides by N-1 (ddof=1). Use exactly what the question names: \"population standard deviation\" => ddof=0; a \"sample\" standard deviation or a z-score taken off a sample => ddof=1.",
    ),
    (
        "Coefficient of variation of log growth rates",
        "First build the year-to-year log-growth series rₜ = ln(Xₜ / Xₜ₋₁), then CV = std(r) / mean(r). Use the population standard deviation (ddof=0) unless told otherwise.",
    ),
    (
        "Mean absolute deviation (about the mean)",
        "mean(|xᵢ - mean(x)|): absolute deviations from the MEAN, averaged (divide by N).",
    ),
    (
        "Median absolute deviation",
        "median(|xᵢ - median(x)|): the median of the absolute deviations from the median. Do NOT apply the 1.4826 scaling constant unless the question asks to normalize it to a standard-deviation estimate.",
    ),
    (
        "Percentiles / quartiles / H-spread (Type 7)",
        "Use the Type 7 method (linear interpolation, the numpy/R default): index h = (N - 1) * p + 1, value = x[floor(h)] + (h - floor(h)) * (x[floor(h)+1] - x[floor(h)]) on the ascending-sorted values. The H-spread is the interquartile range Q3 - Q1.",
    ),
    (
        "Hazen percentile (Hazen plotting position)",
        "Sort ascending; assign each x[i] (i = 1..n) the nonexceedance probability p_i = (i - 0.5) / n, then linearly interpolate between the bracketing (p_i, x[i]) points for the requested percentile. Equivalent to numpy.quantile(..., method=\"hazen\").",
    ),
    (
        "Herfindahl-Hirschman Index (HHI)",
        "HHI = sum of squared market shares. Choose one share scale and stay consistent: whole-number percents give HHI in 0..10000 (DOJ/FTC convention); fractional shares give 0..1. The \"effective number\" of competitors is the reciprocal of the HHI computed on FRACTIONAL shares (1 / sum(shareᵢ²)).",
    ),
    (
        "Gini coefficient",
        "Sort values ascending (i = 1..n); the population Gini is G_pop = (2 * sum(i * xᵢ)) / (n * sum(xᵢ)) - (n + 1) / n. Then apply the small-sample (bias) correction: G = G_pop * n / (n - 1). The correction is what the gold values use and matters at small n (it doubles G at n = 2).",
    ),
    (
        "Value at Risk (historical)",
        "Sort the historical returns / P&L ascending; VaR at confidence c is the loss at the (1 - c) lower-tail percentile (1st percentile for 99%, 5th for 95%).",
    ),
    (
        "Expected shortfall / CVaR (historical)",
        "Expected shortfall at confidence c is the mean of the outcomes in the worst (1 - c) tail — average all returns at or beyond the VaR cutoff (e.g. the mean of the worst 5% for 95%).",
    ),
    (
        "Realized variance / realized volatility",
        "Form log returns rᵢ = ln(Pᵢ / Pᵢ₋₁); realized variance = sum(rᵢ ** 2); realized volatility = sqrt(realized variance). Annualize volatility by multiplying by sqrt(periods-per-year).",
    ),
    (
        "Box-Cox transformation",
        "Use the scaled (NIST / original Box-Cox) form: T(y) = (y ** lambda - 1) / lambda for lambda != 0, and T(y) = ln(y) for lambda = 0. Not the bare power y ** lambda.",
    ),
    (
        "Hodrick-Prescott filter",
        "Use statsmodels.api.tsa.filters.hpfilter(series, lamb=LAMBDA) with the lambda the question states (e.g. 100 for annual data), applied separately to each series; the trend is the second return value.",
    ),
    (
        "Simple exponential smoothing",
        "Level recursion Sₜ = alpha * y_{t-1} + (1 - alpha) * S_{t-1}, initialized S₂ = y₁ (NIST convention); the one-step-ahead forecast is the latest level Sₜ.",
    ),
    (
        "Centered moving average",
        "For an odd window the average centers on the middle period. For an even window of length N, take a further 2-term moving average of the N-term averages (the 2xN centered MA, with weights (1/2, 1, ..., 1, 1/2) / N).",
    ),
    (
        "OLS simple linear regression",
        "slope = sum((x - x̄)(y - ȳ)) / sum((x - x̄)²); intercept = ȳ - slope * x̄. Forecast by substituting x into ŷ = intercept + slope * x. Use the exact predictor coding the question specifies (e.g. \"treat 1990 as year 0\").",
    ),
    (
        "Pearson correlation coefficient",
        "r = sum((x - x̄)(y - ȳ)) / sqrt(sum((x - x̄)²) * sum((y - ȳ)²)).",
    ),
    (
        "Partial correlation controlling for a third variable z",
        "r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz²)(1 - r_yz²)); use the time index as z when controlling for time.",
    ),
    (
        "Macaulay duration",
        "PV-weighted average time to cash flows: sum(t * PV(CF_t)) / sum(PV(CF_t)); for a zero-coupon instrument it equals the time to maturity.",
    ),
    (
        "CPI inflation adjustment (nominal -> real)",
        "real = nominal * CPI_base / CPI_period. Use the annual-average CPI-U for annual / multi-year figures and the specific monthly CPI-U when converting a named month; use the not-seasonally-adjusted series.",
    ),
    (
        "Fisher Ideal symmetric growth rate",
        "The Fisher index is the geometric mean of the Laspeyres and Paasche indexes; the symmetric growth between two values uses the index form (growth = Fisher_index - 1), not a plain percent change.",
    ),
]


def _concept_block(concept: str, explanation: str) -> str:
    """Render one catalog entry as the `### <Concept>` markdown the compute prompt expects."""
    return f"### {concept}\n{explanation}"


# The full catalog as a single markdown string (every entry). Used when selection is
# bypassed (config.compute_precomputed_concept_refs) and kept for back-compat imports.
PRECOMPUTED_CONCEPT_REFERENCES = "\n\n".join(
    _concept_block(name, body) for name, body in PRECOMPUTED_CONCEPTS
)

# Numbered menu (names + definitions) shown to the selector. The bodies are needed so the
# model can tell near-duplicates apart (Type-7 vs Hazen percentile, pop vs sample std); it
# is a fixed prefix, so it prompt-caches and its input cost is paid once, not per question.
_CATALOG_MENU = "\n\n".join(
    f"[{i}] {name}\n{body}" for i, (name, body) in enumerate(PRECOMPUTED_CONCEPTS)
)


_SELECT_SYSTEM_PROMPT = f"""\
Below is a numbered catalog of canonical concept references. Read the question
and select the entries a code-writing agent needs to answer it correctly — and
ONLY those.

Selection rules:
  - Include an entry when misapplying that concept would change the answer —
    named statistical operations, domain-specific conventions, or any phrase
    whose canonical meaning a non-expert might guess wrong.
  - Skip trivial operations (sum, average, max, ratio, difference) and pure
    unit / scope modifiers ("in millions of dollars", "for FY 2023").
  - When two entries are close, pick the one the question's wording implies; do
    not select alternative formulations the question doesn't mention.

Output a JSON array of the selected indices, e.g. [2, 14]. Output [] if none
apply. Output the array and nothing else.

## Catalog
{_CATALOG_MENU}
"""


def _parse_selection(raw: str, _ctx: ExecutionContext) -> list[ConceptExplanation]:
    """Map the model's index reply to catalog entries. Tolerant: pulls every integer out
    of the reply, dedupes, drops out-of-range / hallucinated indices, and returns the
    entries in catalog order. Empty / no valid indices → `[]`."""
    del _ctx  # parse-hook signature requires it; selection needs no context
    idxs = sorted({int(m) for m in re.findall(r"\d+", raw)})
    return [
        ConceptExplanation(
            concept=PRECOMPUTED_CONCEPTS[i][0], explanation=PRECOMPUTED_CONCEPTS[i][1]
        )
        for i in idxs
        if 0 <= i < len(PRECOMPUTED_CONCEPTS)
    ]


_ALL_CONCEPTS: list[ConceptExplanation] = [
    ConceptExplanation(concept=name, explanation=body)
    for name, body in PRECOMPUTED_CONCEPTS
]


class QuestionExplainer:
    """One LLM call per question → the relevant subset of `PRECOMPUTED_CONCEPTS` (empty
    list when none apply). With `config.compute_precomputed_concept_refs` set, skips the
    call and returns the entire catalog (full-dump A/B baseline)."""

    _prompt = PromptedCall(
        name="question_explainer",
        system_prompt=_SELECT_SYSTEM_PROMPT,
        default_effort="low",
        parse=_parse_selection,
        output_instruction=(
            "Output a JSON array of the selected catalog indices (e.g. [2, 14]), "
            "or [] if only obvious concepts apply."
        ),
    )

    async def run(self, ctx: ExecutionContext, *, question: str) -> list[ConceptExplanation]:
        if ctx.config.compute_precomputed_concept_refs:
            return list(_ALL_CONCEPTS)
        return await self._prompt.call(ctx, f"Question:\n{question}", temperature=0.0)
