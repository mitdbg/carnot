"""Top-level planner — single LLM call to generate a DSL AST from a question."""

from __future__ import annotations

import json
import re
from pathlib import Path

from skunk.common.context import HarnessContext
from skunk.dsl import ChainNode, from_dict, validate
from skunk.subagents.base import StepFailed, call_gemini

# ---------------------------------------------------------------------------
# DSL spec (5-op grammar embedded in the system prompt)
# ---------------------------------------------------------------------------

_DSL_SPEC = """\
## DSL Op Grammar

5 operations. Steps composed left-to-right with `-->`. Parallel branches with
`[ chain1 ; chain2 ]` feeding the next op. Nesting allowed.

| op                   | in → out                | purpose / args |
|----------------------|-------------------------|----------------|
| retrieve(...)        | () → DocHandle          | concept (str), period (str), source_bulletin? (YYYY-MM) |
| extract(...)         | DocHandle → TypedValue  | (no args) — emits a dict of named typed values relevant to the question |
| read_visual(...)     | DocHandle → TypedValue  | (no args) — vision read of charts/figures, same output shape |
| lookup_external(...) | () → TypedValue         | nl (str) — natural language description of the data to look up |
| compute(...)         | Value(s) → FormattedString | (no args) — chain terminator. Self-plans, generates Python, verifies output format/unit. |

### Period string conventions
  'CY1940'         → calendar year 1940
  'FY1939'         → fiscal year 1939
  '1953..1955'     → range 1953 to 1955 inclusive
  '2025-03'        → single bulletin month (YYYY-MM)
  '2025-03-31'     → specific date

### Key rules
- retrieve and lookup_external are chain heads (no prev input).
- extract / read_visual emit a TypedValue with .dtype='named' whose .value is a dict mapping
  snake_case names to scalars/lists/tables. Names disambiguate (e.g. national_defense_cy1940).
- compute is ALWAYS the chain terminator. It receives the question + extracted values, plans the
  computation itself, generates Python, and a verifier LLM checks the output format/unit. Plans
  do NOT need to spell out the computation — compute figures it out from the question.
- For parallel-branch chains, the value flowing into compute is a list[TypedValue].
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
                {"type": "op", "op": "extract", "args": {}},
                {"type": "op", "op": "compute", "args": {}},
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
                                 "args": {"concept": "national_defense", "period": "CY1940"}},
                                {"type": "op", "op": "extract", "args": {}},
                            ]
                        },
                        {
                            "type": "chain", "global_constraints": [],
                            "steps": [
                                {"type": "op", "op": "retrieve",
                                 "args": {"concept": "national_defense", "period": "CY1953"}},
                                {"type": "op", "op": "extract", "args": {}},
                            ]
                        }
                    ]
                },
                {"type": "op", "op": "compute", "args": {}},
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
                {"type": "op", "op": "extract", "args": {}},
                {"type": "op", "op": "compute", "args": {}},
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
                                 "concepts": ["Treasury Foreign Exchange and Securities investments"]},
                                {"type": "op", "op": "extract", "args": {}},
                            ]
                        },
                        {
                            "type": "chain", "global_constraints": [],
                            "steps": [
                                {"type": "op", "op": "lookup_external",
                                 "args": {"nl": "USD/JPY exchange rate on 2025-03-31"},
                                 "concepts": ["Macrotrends FX data"]},
                            ]
                        }
                    ]
                },
                {"type": "op", "op": "compute", "args": {}},
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
                    "concepts": ["debt held by the public (chart/figure)"]
                },
                {"type": "op", "op": "read_visual", "args": {}},
                {"type": "op", "op": "compute", "args": {}},
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
    """Generate a 5-op AST ChainNode from a natural-language question."""
    system = _build_system(ctx)
    user = _build_user(question, ctx)

    attempt_errors: list[str] = []
    for attempt in range(2):
        raw = call_gemini(system, user)
        try:
            ast_dict = _extract_ast_json(raw)
            chain = from_dict(ast_dict)
            if not isinstance(chain, ChainNode):
                raise ValueError(
                    f"from_dict returned {type(chain).__name__}, expected ChainNode"
                )
            result = validate(chain)
            if not result.ok:
                raise ValueError(f"AST validation errors: {result.errors}")
            return chain
        except Exception as e:
            attempt_errors.append(f"Attempt {attempt + 1}: {e}")
            if attempt == 0:
                user += (
                    f"\n\nYour previous output had errors:\n{e}\n"
                    "Fix and return valid JSON only."
                )

    raise StepFailed(
        "planner",
        f"Failed to produce valid AST after {len(attempt_errors)} attempts: "
        + "; ".join(attempt_errors),
    )


def _extract_ast_json(text: str) -> dict:
    """Extract the outermost JSON object from planner output.

    Tries a fenced code block first, then falls back to scanning for balanced braces.
    The old single-regex approach truncated multi-line nested JSON at the first '}'.
    """
    # Fenced code block: grab everything between the fences, then parse as JSON.
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Balanced-brace scan: find the outermost { ... } regardless of line breaks.
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
