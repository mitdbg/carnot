"""Top-level planner — single LLM call to generate a flat Plan from a question.

Also exposes plan_recovery(): a small LLM call to propose ONE additional Branch
when compute reports MissingData mid-execution.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from skunk.common.context import HarnessContext
from skunk.dsl import Branch, Plan, from_dict, validate
from skunk.subagents.base import StepFailed

# ---------------------------------------------------------------------------
# DSL spec for the planner system prompt
# ---------------------------------------------------------------------------

_DSL_SPEC = """\
## Plan shape

A plan is a flat list of data-gathering branches that feed an implicit compute()
at the end. Every plan is exactly one of:

  - Single branch:
      retrieve(concept, period[, source_bulletin, visual_only]) --> compute()
      lookup_external(nl) --> compute()
  - Multiple branches in parallel:
      [ branch_1 ; branch_2 ; ... ] --> compute()

Each branch is either a `retrieve` (always followed by `extract`) or a `lookup_external`.
compute() is always the terminator, always implicit, always takes no args.

## Branch types

| kind            | fields                                               |
|-----------------|------------------------------------------------------|
| retrieve        | concept (str), period (str), source_bulletin? (YYYY-MM), visual_only? (bool) |
| lookup_external | nl (str) — natural-language description of external data to look up |

Set visual_only=true for questions about charts/figures/scanned images.

## Period strings (retrieve.period)

  'CY1940'         calendar year 1940
  'FY1939'         fiscal year 1939
  'CY1953..CY1955' range, inclusive
  '2025-03'        single bulletin month
  '2025-03-31'     specific date

## JSON shape

Emit a single JSON object:

  {
    "branches": [
      {"kind": "retrieve", "concept": "...", "period": "..."},
      {"kind": "lookup_external", "nl": "..."}
    ]
  }

A simple chain has one element in `branches`. A parallel has 2+.
"""

_FEW_SHOTS = [
    {
        "question": "What were total U.S. national defense expenditures (millions, nominal) in calendar year 1940?",
        "plan": {
            "branches": [
                {"kind": "retrieve", "concept": "national_defense_expenditures", "period": "CY1940"},
            ],
        },
    },
    {
        "question": "Absolute percent change in national defense expenditures between CY1940 and CY1953, rounded to hundredths.",
        "plan": {
            "branches": [
                {"kind": "retrieve", "concept": "national_defense_expenditures", "period": "CY1940"},
                {"kind": "retrieve", "concept": "national_defense_expenditures", "period": "CY1953"},
            ],
        },
    },
    {
        "question": "Geometric mean of weekly average discount rates for new 91-day bills, September 1953–1955.",
        "plan": {
            "branches": [
                {"kind": "retrieve", "concept": "91day_bill_discount_rate", "period": "1953..1955"},
            ],
        },
    },
    {
        "question": "How much does the U.S. Treasury have invested in Japanese Yen as of March 31 2025? Convert to JPY using Macrotrends FX data.",
        "plan": {
            "branches": [
                {"kind": "retrieve", "concept": "fx_investments", "period": "2025-03", "source_bulletin": "2025-03"},
                {"kind": "lookup_external", "nl": "USD/JPY exchange rate on 2025-03-31"},
            ],
        },
    },
    {
        "question": "Read the total debt held by the public from the September 1990 Treasury Bulletin chart on page 5.",
        "plan": {
            "branches": [
                {
                    "kind": "retrieve",
                    "concept": "public_debt_chart",
                    "period": "1990-09",
                    "source_bulletin": "1990-09",
                    "visual_only": True,
                },
            ],
        },
    },
]

_CAPABILITY_VOCAB = [
    "single_doc", "multi_doc", "cross_temporal",
    "tabular_extraction", "visual_reasoning", "text_extraction",
    "external_lookup", "arithmetic", "statistical_modeling", "forecasting",
    "unit_conversion", "inflation_adjustment", "fx_conversion",
    "definitional_disambiguation", "multi_part_answer", "temporal_alignment",
]

_SYSTEM = f"""\
You are the planner for the OfficeQA harness. Given a question about U.S. \
Treasury Monthly Bulletins, produce a flat Plan JSON that, when executed, \
will produce the correct answer.

{_DSL_SPEC}

## Capability hints (think about which apply, then build the plan accordingly)
{chr(10).join(f'- {c}' for c in _CAPABILITY_VOCAB)}

## Few-shot examples
"""


def _build_system(ctx: HarnessContext) -> str:
    system = _SYSTEM
    for ex in _FEW_SHOTS:
        system += f"\n### Example\nQ: {ex['question']}\nPlan:\n```json\n{json.dumps(ex['plan'], indent=2)}\n```\n"
    return system


def _build_user(question: str, ctx: HarnessContext) -> str:
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

    return f"""\
Question: {question}
{manifest_summary}

Produce the Plan JSON. Output ONLY a JSON code block — no prose, no explanation.
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plan(question: str, ctx: HarnessContext) -> Plan:
    """Generate a flat Plan from a natural-language question."""
    system = _build_system(ctx)
    user = _build_user(question, ctx)

    attempt_errors: list[str] = []
    for attempt in range(2):
        raw = ctx.llm_client.call(system, user, thinking_budget=-1)
        try:
            plan_dict = _extract_plan_json(raw)
            p = from_dict(plan_dict)
            if not isinstance(p, Plan):
                raise ValueError(f"from_dict returned {type(p).__name__}, expected Plan")
            result = validate(p)
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


def _extract_plan_json(text: str) -> dict:
    """Extract the outermost JSON object from planner output."""
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in planner response:\n{text[:400]}")
    depth = 0
    end = -1
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        raise ValueError(f"Unbalanced braces in planner response:\n{text[:400]}")
    return json.loads(text[start:end])


# ---------------------------------------------------------------------------
# Recovery planner — called when compute reports MissingData
# ---------------------------------------------------------------------------

_RECOVERY_SYSTEM = """\
You diagnose a compute failure and propose ONE additional data-gathering branch.

Compute ran on the plan's existing branches and reported MISSING:<reason> — it
needs one more piece of information to answer the question. Your job: emit one
JSON object describing the supplemental branch.

Two valid branch shapes:

  {"kind": "lookup_external", "nl": "<natural-language description>"}
      Use when the missing data is an external fact (CPI, FX rate, event year,
      named entity) NOT typically found in Treasury Bulletins.

  {"kind": "retrieve", "concept": "<snake_case>", "period": "<period>",
   "source_bulletin": "<YYYY-MM>"?, "visual_only": <bool>?}
      Use when an existing retrieve missed the right bulletin or a related
      bulletin row is needed (e.g., a comparison year was overlooked).

If recovery is not feasible (the question is genuinely unanswerable from
available sources), output exactly:
  {"kind": "decline", "reason": "<short reason>"}

Output ONLY the JSON object. No prose.
"""


def plan_recovery(
    question: str, plan_obj: Plan, missing_reason: str, ctx: HarnessContext
) -> Branch | None:
    """Propose one supplemental Branch given a MissingData reason. Returns None if recovery declines."""
    existing = json.dumps(
        {"branches": [_branch_summary(b) for b in plan_obj.branches]},
        indent=2,
    )
    user = (
        f"Question: {question}\n\n"
        f"Existing plan:\n```json\n{existing}\n```\n\n"
        f"Compute reported MISSING: {missing_reason}\n\n"
        "Propose ONE supplemental branch, or decline."
    )

    raw = ctx.llm_client.call(_RECOVERY_SYSTEM, user, thinking_budget=-1)
    ctx.emit("planner", "recovery response", raw=raw[:400])

    try:
        d = _extract_plan_json(raw)
    except Exception as e:
        raise StepFailed("planner", f"recovery JSON parse failed: {e}") from e

    kind = d.get("kind")
    if kind == "decline":
        ctx.emit("planner", "recovery declined", reason=d.get("reason", "(no reason)"))
        return None
    if kind in ("retrieve", "lookup_external"):
        from skunk.dsl import _branch_from_dict
        try:
            return _branch_from_dict(d)
        except Exception as e:
            raise StepFailed("planner", f"recovery branch construction failed: {e}") from e
    raise StepFailed("planner", f"recovery emitted unknown kind {kind!r}: {d!r}")


def _branch_summary(b: Branch) -> dict:
    """Compact dict describing a branch for the recovery prompt."""
    from skunk.dsl import LookupBranch as _Lookup
    from skunk.dsl import RetrieveBranch as _Retrieve

    if isinstance(b, _Retrieve):
        d = {"kind": "retrieve", "concept": b.concept, "period": b.period}
        if b.source_bulletin:
            d["source_bulletin"] = b.source_bulletin
        if b.visual_only:
            d["visual_only"] = True
        return d
    if isinstance(b, _Lookup):
        return {"kind": "lookup_external", "nl": b.nl}
    raise TypeError(f"Unknown branch type: {type(b)}")
