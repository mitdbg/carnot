"""Top-level planner — single LLM call to generate a DSL AST from a question."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skunk.dsl import ChainNode, from_dict, validate
from skunk.subagents.base import HarnessContext, LLMConfig, StepFailed, call_llm

# ---------------------------------------------------------------------------
# DSL spec (6-op grammar embedded in the system prompt)
# ---------------------------------------------------------------------------

_DSL_SPEC = """\
## DSL Op Grammar

6 operations. Steps composed left-to-right with `-->`. Parallel branches with
`[ chain1 ; chain2 ]` feeding the next op. Nesting allowed.

| op                   | in → out            | purpose / args |
|----------------------|---------------------|----------------|
| retrieve(...)        | () → DocHandle      | concept (str), period (str), source_bulletin? (YYYY-MM) |
| extract(...)         | Handle → TypedValue | concept (str), mode? ('value'\\|'list'\\|'table') |
| read_visual(...)     | Handle → TypedValue | concept (str) — vision read of charts/figures |
| lookup_external(...) | () → TypedValue     | resource (str), **params — CPI-U, FX, BLS, event dates |
| compute(...)         | Value(s) → Value    | code (Python str) — sets `result`; `prev` is the input |
| format(...)          | Value → String      | precision? (int), unit? (str), layout? (str) |

### Period string conventions
  'CY1940'         → calendar year 1940
  'FY1939'         → fiscal year 1939
  '1953..1955'     → range 1953 to 1955 inclusive
  '2025-03'        → single bulletin month (YYYY-MM)
  '2025-03-31'     → specific date

### Key rules
- retrieve and lookup_external are chain heads (no prev input).
- For parallel-branch results, prev is a list; use prev[0], prev[1], etc. in compute code.
- compute code must assign `result`; it runs in a sandbox with numpy (np), pandas (pd), math.
- format is always the chain terminator.
- Use single-quoted string args only.
"""

_FEW_SHOTS = [
    {
        "question": "What were total U.S. national defense expenditures (millions, nominal) in calendar year 1940?",
        "ast": {
            "type": "chain",
            "global_constraints": ["nominal dollars (not inflation-adjusted)"],
            "steps": [
                {
                    "type": "op", "op": "retrieve",
                    "args": {"concept": "national_defense", "period": "CY1940"},
                    "concepts": ["U.S. National Defense (Treasury budget category)"],
                    "constraints": ["calendar year, not fiscal year"]
                },
                {
                    "type": "op", "op": "extract",
                    "args": {"concept": "total_expenditure", "mode": "value"},
                    "concepts": [], "constraints": []
                },
                {
                    "type": "op", "op": "format",
                    "args": {"precision": 0, "unit": "millions_usd"},
                    "concepts": [], "constraints": []
                },
            ]
        }
    },
    {
        "question": "Absolute percent change in national defense expenditures between CY1940 and CY1953, rounded to hundredths.",
        "ast": {
            "type": "chain",
            "global_constraints": [],
            "steps": [
                {
                    "type": "parallel",
                    "branches": [
                        {
                            "type": "chain", "global_constraints": [],
                            "steps": [
                                {"type": "op", "op": "retrieve",
                                 "args": {"concept": "national_defense", "period": "CY1940"},
                                 "concepts": [], "constraints": []},
                                {"type": "op", "op": "extract",
                                 "args": {"concept": "total_expenditure", "mode": "value"},
                                 "concepts": [], "constraints": []},
                            ]
                        },
                        {
                            "type": "chain", "global_constraints": [],
                            "steps": [
                                {"type": "op", "op": "retrieve",
                                 "args": {"concept": "national_defense", "period": "CY1953"},
                                 "concepts": [], "constraints": []},
                                {"type": "op", "op": "extract",
                                 "args": {"concept": "total_expenditure", "mode": "value"},
                                 "concepts": [], "constraints": []},
                            ]
                        }
                    ]
                },
                {
                    "type": "op", "op": "compute",
                    "args": {"code": "a, b = prev[0].value, prev[1].value; result = abs((b - a) / a * 100)"},
                    "concepts": [], "constraints": ["absolute value of percent change"]
                },
                {
                    "type": "op", "op": "format",
                    "args": {"precision": 2, "unit": "percent"},
                    "concepts": [], "constraints": []
                },
            ]
        }
    },
    {
        "question": "Geometric mean of weekly average discount rates for new 91-day bills, September 1953–1955, rounded to nearest thousandths.",
        "ast": {
            "type": "chain",
            "global_constraints": [],
            "steps": [
                {
                    "type": "op", "op": "retrieve",
                    "args": {"concept": "91day_bill_discount_rate", "period": "1953..1955"},
                    "concepts": ["91-day Treasury bill weekly issuance", "average discount rate (Thursday quote convention)"],
                    "constraints": ["September months only"]
                },
                {
                    "type": "op", "op": "extract",
                    "args": {"concept": "weekly_discount_rate", "mode": "list"},
                    "concepts": [], "constraints": []
                },
                {
                    "type": "op", "op": "compute",
                    "args": {"code": "import math; vals = [v for v in prev.value if v is not None]; result = math.exp(sum(math.log(v) for v in vals) / len(vals))"},
                    "concepts": [], "constraints": ["geometric mean, not arithmetic"]
                },
                {
                    "type": "op", "op": "format",
                    "args": {"precision": 3},
                    "concepts": [], "constraints": []
                },
            ]
        }
    },
    {
        "question": "How much does the U.S. Treasury have invested in Japanese Yen as of March 31 2025? Convert to actual JPY using Macrotrends FX data. No commas, round to nearest whole yen.",
        "ast": {
            "type": "chain",
            "global_constraints": [],
            "steps": [
                {
                    "type": "parallel",
                    "branches": [
                        {
                            "type": "chain", "global_constraints": [],
                            "steps": [
                                {"type": "op", "op": "retrieve",
                                 "args": {"concept": "fx_investments", "period": "2025-03", "source_bulletin": "2025-03"},
                                 "concepts": ["Treasury Foreign Exchange and Securities investments"],
                                 "constraints": []},
                                {"type": "op", "op": "extract",
                                 "args": {"concept": "japanese_yen_holdings", "mode": "value"},
                                 "concepts": [], "constraints": []},
                            ]
                        },
                        {
                            "type": "chain", "global_constraints": [],
                            "steps": [
                                {"type": "op", "op": "lookup_external",
                                 "args": {"resource": "fx_rate", "pair": "USD/JPY", "date": "2025-03-31"},
                                 "concepts": ["Macrotrends FX data"], "constraints": []},
                            ]
                        }
                    ]
                },
                {
                    "type": "op", "op": "compute",
                    "args": {"code": "result = prev[0].value * prev[1].value"},
                    "concepts": [], "constraints": []
                },
                {
                    "type": "op", "op": "format",
                    "args": {"precision": 0, "layout": "no_commas"},
                    "concepts": [], "constraints": ["no commas", "round to nearest whole yen"]
                },
            ]
        }
    },
    {
        "question": "Read the total debt held by the public figure from the September 1990 Treasury Bulletin chart on page 5.",
        "ast": {
            "type": "chain",
            "global_constraints": [],
            "steps": [
                {
                    "type": "op", "op": "retrieve",
                    "args": {"concept": "public_debt_chart", "period": "1990-09", "source_bulletin": "1990-09"},
                    "concepts": ["debt held by the public (chart/figure)"],
                    "constraints": []
                },
                {
                    "type": "op", "op": "read_visual",
                    "args": {"concept": "total_debt_held_by_public"},
                    "concepts": [], "constraints": []
                },
                {
                    "type": "op", "op": "format",
                    "args": {"precision": 1, "unit": "billions_usd"},
                    "concepts": [], "constraints": []
                },
            ]
        }
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
Treasury Monthly Bulletins, produce a DSL JSON pipeline that, when executed, \
will produce the correct answer.

{_DSL_SPEC}

## Capability hints (think about which apply, then build the plan accordingly)
{chr(10).join(f'- {c}' for c in _CAPABILITY_VOCAB)}

## Few-shot examples
"""


def _build_system(ctx: HarnessContext) -> str:
    system = _SYSTEM
    for ex in _FEW_SHOTS:
        system += f"\n### Example\nQ: {ex['question']}\nAST:\n```json\n{json.dumps(ex['ast'], indent=2)}\n```\n"
    return system


def _build_user(question: str, ctx: HarnessContext) -> str:
    manifest_summary = ""
    if ctx.manifest_path and Path(ctx.manifest_path).exists():
        try:
            import pandas as pd
            df = pd.read_csv(ctx.manifest_path)
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

Produce the DSL JSON pipeline. Output ONLY a JSON code block with the \
ChainNode AST — no prose, no explanation.
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plan(question: str, ctx: HarnessContext) -> ChainNode:
    """Generate a 6-op AST ChainNode from a natural-language question."""
    system = _build_system(ctx)
    user = _build_user(question, ctx)

    last_err: Exception | None = None
    for attempt in range(2):
        raw = call_llm(system, user, ctx.llm_config)
        try:
            ast_dict = _extract_ast_json(raw)
            chain = from_dict(ast_dict)
            if not isinstance(chain, ChainNode):
                chain = ChainNode(steps=[chain] if hasattr(chain, "op") else [])
            result = validate(chain)
            if not result.ok:
                raise ValueError(f"AST validation errors: {result.errors}")
            return chain
        except Exception as e:
            last_err = e
            if attempt == 0:
                user += f"\n\nYour previous output had errors: {e}\nFix and return valid JSON."

    raise StepFailed("planner", f"Failed to produce valid AST: {last_err}")


def _extract_ast_json(text: str) -> dict:
    import re
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    raise ValueError(f"No JSON object found in planner response:\n{text[:400]}")
