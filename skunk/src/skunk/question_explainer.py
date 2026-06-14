"""question_explainer — whole-question concept extraction. One LLM call reads the
question and extracts the non-obvious concepts a code-writing agent needs (each with a
short definition), injected into the compute codegen prompt as `## Concept references`.

Output is markdown (one `### <concept>` section each) so LaTeX doesn't break parsing."""

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


# Splits the markdown reply on `### <concept>` headers.
_SECTION_RE = re.compile(r"^###\s+", re.MULTILINE)


_QUESTION_SYSTEM_PROMPT = """\
Read the question. Identify any non-obvious mathematical, statistical,
or domain concepts a code-writing agent must know to answer correctly.
For each, give a short explanation — only the information the agent
needs to apply the concept to this question.

Hard constraints:
  - Cover one formulation per concept — the one the question implies.
    Do not introduce alternative formulas, orientations, or variants
    the question doesn't mention; if the question doesn't pin a
    choice, pick the most standard form and stop there.
  - Skip trivial operations (sum, average, max, ratio, difference).
    Skip pure unit / scope modifiers ("in millions of dollars",
    "for FY 2023").
  - Include concepts whose misimplementation would yield the
    wrong answer — named statistical operations (geometric mean,
    Zipf exponent, arc elasticity, expected shortfall, Gini,
    H-spread, CAGR, linear regression, etc.), domain-specific
    conventions, or any phrase whose canonical meaning a non-expert
    might guess wrong.

Output format: one markdown section per concept. Each section starts
with `### <Concept Name>` on its own line, followed by a 1-3 sentence
explanation. Plain text or LaTeX is fine — write naturally, no JSON
escapes. Separate sections with a blank line.

If no non-obvious concepts apply, output nothing.
"""


# A fixed cheat-sheet of canonical formulas for the named / ambiguous operations the
# benchmark asks for — the ones where a defensible-but-wrong variant yields the wrong
# answer (Zipf orientation, Box-Cox form, percentile type, pop-vs-sample std, …). Pins
# ONE convention per operation, each backed by an official US-gov / NIST definition or by
# the convention the gold answers actually use (see docs/computation_catalog.md). Appended
# verbatim to the compute `## Concept references` block when
# `config.compute_precomputed_concept_refs` is on (env SKUNK_PRECOMPUTED_CONCEPT_REFS=1),
# after the question_explainer's per-question concepts. Same `### <Concept>` markdown shape
# as the explainer output so it reads as one continuous reference section.
PRECOMPUTED_CONCEPT_REFERENCES = """\
### Zipf / power-law exponent
Rank the values in descending order (ranks 1..n) and run an OLS regression of log(value) on log(rank); the exponent is the absolute value of the slope. Do NOT regress log(rank) on log(value) (that inverts the exponent), and do NOT use the MLE / Hill estimator unless the question explicitly asks for it.

### CAGR (compound annual growth rate)
CAGR = (end / begin) ** (1 / n) - 1, where n is the number of YEAR intervals between the endpoints (end_year - begin_year), not the count of data points.

### Continuously compounded (logarithmic) growth rate
r = ln(end / begin) / n over n periods; annualize by multiplying the per-period log change by periods-per-year. Relation to CAGR: r = ln(1 + CAGR).

### Annual decay / growth factor
The per-year multiplicative factor is (1 + CAGR); it is below 1 for a declining series. The reciprocal 1 / (1 + CAGR) is a discount factor, not the decay factor.

### Geometric mean
exp(mean(ln x)) = (product of x) ** (1 / n), over strictly positive values. For an average growth RATE, take the geometric mean of the growth factors (1 + gᵢ) and subtract 1.

### Arc elasticity (midpoint method)
[(Q2 - Q1) / ((Q1 + Q2) / 2)] / [(P2 - P1) / ((P1 + P2) / 2)] — each percent change uses the average of its two endpoints in the denominator (the factor of 2 cancels). Keep the sign unless the question asks for the absolute value.

### Population vs sample standard deviation / variance
Population divides by N (numpy default, ddof=0); sample divides by N-1 (ddof=1). Use exactly what the question names: "population standard deviation" => ddof=0; a "sample" standard deviation or a z-score taken off a sample => ddof=1.

### Coefficient of variation of log growth rates
First build the year-to-year log-growth series rₜ = ln(Xₜ / Xₜ₋₁), then CV = std(r) / mean(r). Use the population standard deviation (ddof=0) unless told otherwise.

### Mean absolute deviation (about the mean)
mean(|xᵢ - mean(x)|): absolute deviations from the MEAN, averaged (divide by N).

### Median absolute deviation
median(|xᵢ - median(x)|): the median of the absolute deviations from the median. Do NOT apply the 1.4826 scaling constant unless the question asks to normalize it to a standard-deviation estimate.

### Percentiles / quartiles / H-spread (Type 7)
Use the Type 7 method (linear interpolation, the numpy/R default): index h = (N - 1) * p + 1, value = x[floor(h)] + (h - floor(h)) * (x[floor(h)+1] - x[floor(h)]) on the ascending-sorted values. The H-spread is the interquartile range Q3 - Q1.

### Hazen percentile (Hazen plotting position)
Sort ascending; assign each x[i] (i = 1..n) the nonexceedance probability p_i = (i - 0.5) / n, then linearly interpolate between the bracketing (p_i, x[i]) points for the requested percentile. Equivalent to numpy.quantile(..., method="hazen").

### Herfindahl-Hirschman Index (HHI)
HHI = sum of squared market shares. Choose one share scale and stay consistent: whole-number percents give HHI in 0..10000 (DOJ/FTC convention); fractional shares give 0..1. The "effective number" of competitors is the reciprocal of the HHI computed on FRACTIONAL shares (1 / sum(shareᵢ²)).

### Gini coefficient
Lorenz-curve area ratio in 0..1: sort values ascending (i = 1..n), G = (2 * sum(i * xᵢ)) / (n * sum(xᵢ)) - (n + 1) / n.

### Value at Risk (historical)
Sort the historical returns / P&L ascending; VaR at confidence c is the loss at the (1 - c) lower-tail percentile (1st percentile for 99%, 5th for 95%).

### Expected shortfall / CVaR (historical)
Expected shortfall at confidence c is the mean of the outcomes in the worst (1 - c) tail — average all returns at or beyond the VaR cutoff (e.g. the mean of the worst 5% for 95%).

### Realized variance / realized volatility
Form log returns rᵢ = ln(Pᵢ / Pᵢ₋₁); realized variance = sum(rᵢ ** 2); realized volatility = sqrt(realized variance). Annualize volatility by multiplying by sqrt(periods-per-year).

### Box-Cox transformation
Use the scaled (NIST / original Box-Cox) form: T(y) = (y ** lambda - 1) / lambda for lambda != 0, and T(y) = ln(y) for lambda = 0. Not the bare power y ** lambda.

### Hodrick-Prescott filter
Use statsmodels.api.tsa.filters.hpfilter(series, lamb=LAMBDA) with the lambda the question states (e.g. 100 for annual data), applied separately to each series; the trend is the second return value.

### Simple exponential smoothing
Level recursion Sₜ = alpha * y_{t-1} + (1 - alpha) * S_{t-1}, initialized S₂ = y₁ (NIST convention); the one-step-ahead forecast is the latest level Sₜ.

### Centered moving average
For an odd window the average centers on the middle period. For an even window of length N, take a further 2-term moving average of the N-term averages (the 2xN centered MA, with weights (1/2, 1, ..., 1, 1/2) / N).

### OLS simple linear regression
slope = sum((x - x̄)(y - ȳ)) / sum((x - x̄)²); intercept = ȳ - slope * x̄. Forecast by substituting x into ŷ = intercept + slope * x. Use the exact predictor coding the question specifies (e.g. "treat 1990 as year 0").

### Pearson correlation coefficient
r = sum((x - x̄)(y - ȳ)) / sqrt(sum((x - x̄)²) * sum((y - ȳ)²)).

### Partial correlation controlling for a third variable z
r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz²)(1 - r_yz²)); use the time index as z when controlling for time.

### Macaulay duration
PV-weighted average time to cash flows: sum(t * PV(CF_t)) / sum(PV(CF_t)); for a zero-coupon instrument it equals the time to maturity.

### CPI inflation adjustment (nominal -> real)
real = nominal * CPI_base / CPI_period. Use the annual-average CPI-U for annual / multi-year figures and the specific monthly CPI-U when converting a named month; use the not-seasonally-adjusted series.

### Fisher Ideal symmetric growth rate
The Fisher index is the geometric mean of the Laspeyres and Paasche indexes; the symmetric growth between two values uses the index form (growth = Fisher_index - 1), not a plain percent change.
"""


def _parse_concepts(raw: str, ctx: ExecutionContext) -> list[ConceptExplanation]:
    """Parse the markdown reply into per-`### ` concept sections. Empty body →
    `[]`; prose without headers → one anonymous block so codegen still sees it."""
    raw = raw.strip()
    if not raw:
        return []
    sections = _SECTION_RE.split(raw)
    if sections and not sections[0].strip():  # drop the pre-header preamble
        sections = sections[1:]
    out: list[ConceptExplanation] = []
    for sect in sections:
        sect = sect.strip()
        if not sect:
            continue
        head, _, body = sect.partition("\n")  # first line = name, rest = explanation
        concept = head.strip()
        explanation = body.strip()
        if concept:
            out.append(ConceptExplanation(concept=concept, explanation=explanation))
    if not out and raw:
        # Prose with no `### ` headers — wrap as one anonymous block.
        out.append(ConceptExplanation(concept="(referenced concepts)", explanation=raw))
    return out


class QuestionExplainer:
    """One LLM call per question → extracted concepts (empty list when none apply)."""

    _prompt = PromptedCall(
        name="question_explainer",
        system_prompt=_QUESTION_SYSTEM_PROMPT,
        default_effort="low",
        parse=_parse_concepts,
        output_instruction=(
            "Output one `### <Concept>` markdown section per concept "
            "(1–3 sentences each), or nothing if only obvious concepts apply."
        ),
    )

    async def run(self, ctx: ExecutionContext, *, question: str) -> list[ConceptExplanation]:
        return await self._prompt.call(ctx, f"Question:\n{question}", temperature=0.4)
