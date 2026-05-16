"""Question → Plan via a single LLM call.

`PlannerOperator.plan(question, ctx)` turns a natural-language question
into a `Plan`: one or more retrieve branches feeding a compute node with
plan-level constraints (`method`, `units_out`, `precision`, `answer_form`).

When the orchestrator detects MissingData during execution, it appends a
one-shot recovery lesson to `ctx.prompt_overrides` (targeting "planner")
and re-invokes `PlannerOperator.plan` for a fresh attempt.
"""

from __future__ import annotations

from pathlib import Path

from skunk.common import HarnessContext, extract_json_object
from skunk.dsl import Plan, from_dict, validate
from skunk.operator import SkunkOperator
from skunk.subagents.base import StepFailed


# ---------------------------------------------------------------------------
# Static system prompt blocks
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are the planner. Given a question, emit a JSON plan that, when executed,
produces the answer.

## Output

{
  "branches": [
    {"kind": "retrieve",
     "concept": "<snake_case>", "period": "<period>",
     "period_type": "<str>", "value_kind": "<scalar|vector|table>",
     "index_name": "<str>", "visual_only": <bool>}
  ],
  "method": "<method>",
  "transforms": ["<transform>", ...],
  "units_out": "<unit>", "precision": <int>,
  "answer_form": "<scalar|bracketed_list|labeled_pair|string>"
}

Branches run in parallel. Each is a two-stage pipeline: `retrieve` narrows
the corpus to ~10 candidate pages (cheap ANN + rerank); `extract` reads
those pages (heavyweight; tries parsed tables, OCR, then vision). The
plan-level fields describe the final answer and feed an implicit compute.

(`lookup_external` branches are temporarily disabled — do not emit them.)

## Field semantics

retrieve.concept    snake_case topical hook embedded into the ANN query.
retrieve.period     temporal mask. Format is corpus-specific (e.g.
                    `CY1940`, `2025-03`, `1990-01..1995-12`).
retrieve.period_type
                    frequency hint for retrieval (e.g. `month`, `FY`,
                    `calendar_quarter`). Vocab is corpus-specific.

extract.value_kind  shape extract should aim for:
                      vector for 1-D series aggregations (means, growth
                        rates, regressions, argmax/argmin over a category).
                      table for two-dim joint analyses (correlations,
                        cross-tabulations).
                      scalar for single-point lookups.
extract.index_name  axis label for vectors — the dim the question varies
                    over (e.g. `month`, `fiscal_year`, `bureau`). Only
                    meaningful when value_kind=vector.
extract.visual_only true for charts/figures (skip text+OCR tiers).

method              the ONE computation the question pins, from a
                    corpus-specific vocabulary (e.g. `cagr`,
                    `pearson_correlation`, `geometric_mean`); null when
                    no specific method is named.
transforms          orthogonal modifiers on top of `method` or the raw
                    value (e.g. `per_capita`, `log_of`,
                    `inflation_adjusted`). A transform that needs an
                    extra input series (population for per-capita, a
                    deflator for inflation-adjustment) implies an extra
                    retrieve branch for that series.
units_out           output unit (e.g. `usd_millions`, `pct`, `year`),
                    from a corpus-specific vocabulary. Set only when the
                    question explicitly names a unit.
precision           int | null. Decimal places of the final answer.
answer_form ∈ { scalar, bracketed_list, labeled_pair, string }. Default
                    scalar. bracketed_list for "[a, b, c]"; labeled_pair
                    for "[year, value]" mixing types; string for entity
                    names.

The dataset section below enumerates the controlled vocabularies for
this corpus (period syntax, period_type, method, transforms, units_out)
and how to pick value_kind / index_name from them.

## Shaping rules

- Choose `concept` / `period` / `period_type` so retrieve finds the right
  pages — wrong-page-no-data is a common failure.
- Choose `value_kind` / `index_name` so extract knows what to pull —
  right-page-wrong-value is the other common failure.
- A `transform` that needs an external series implies an extra retrieve
  branch for that series.
"""


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class PlannerOperator(SkunkOperator):
    name: str = "planner"
    system: str = _SYSTEM

    def plan(self, question: str, ctx: HarnessContext) -> Plan:
        """Generate a Plan from a natural-language question."""
        system = self.build_system(ctx)

        manifest_summary = ""
        if ctx.config.manifest_path and Path(ctx.config.manifest_path).exists():
            try:
                import pandas as pd
                df = pd.read_csv(ctx.config.manifest_path)
                years = sorted(df["year"].dropna().unique().astype(int))
                if years:
                    manifest_summary = (
                        f"\n\nCorpus manifest covers years: {min(years)}–{max(years)} "
                        f"({len(df)} bulletins)"
                    )
            except Exception:
                pass

        user = f"""\
Question: {question}
{manifest_summary}

Produce the Plan JSON. Output ONLY a JSON code block — no prose, no explanation.
"""

        attempt_errors: list[str] = []
        for attempt in range(2):
            resp = ctx.llm_client.call(system, user, thinking_budget=-1, ctx=ctx)
            raw = resp.text
            try:
                plan_dict = extract_json_object(raw)
                p = from_dict(plan_dict)
                if not isinstance(p, Plan):
                    raise ValueError(f"from_dict returned {type(p).__name__}, expected Plan")
                result = validate(p, max_compute_depth=ctx.config.max_compute_depth)
                if not result.ok:
                    raise ValueError(f"Plan validation errors: {result.errors}")
                return p
            except Exception as e:
                attempt_errors.append(f"Attempt {attempt + 1}: {e}")
                if attempt == 0:
                    user += (
                        f"\n\nYour previous output had errors:\n{e}\n"
                        "Fix and return valid JSON only."
                    )

        raise StepFailed(
            "planner",
            f"Failed to produce valid Plan after {len(attempt_errors)} attempts: "
            + "; ".join(attempt_errors),
        )


