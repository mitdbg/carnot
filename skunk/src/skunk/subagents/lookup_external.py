"""lookup_external subagent — single Gemini call, typed Python value output."""

from __future__ import annotations

from skunk.common.context import HarnessContext
from skunk.dsl import OpNode, TypedValue
from skunk.subagents.base import StepFailed, call_gemini, parse_llm_value

_SYSTEM = """\
You are a precise data assistant with knowledge of economic indicators, historical FX rates,
world history dates, named entities (agencies, bureaus, people, places), and other factual data.
Your training data covers through mid-2025, so treat any date before July 2025 as historical —
never refuse on grounds that a date is "future."

Return exactly two lines — no labels, no JSON, no prose:
  Line 1: the value as a Python literal — int, float, list of numbers, or a "double-quoted string"
  Line 2: the unit as a single lowercase word

Unit vocabulary:
  year         — integer calendar year
  cpi          — CPI-U index value (base 1982-84=100)
  fx_rate      — exchange rate (always a positive float)
  usd_millions — dollar amount in millions
  usd_billions — dollar amount in billions
  pct          — percentage
  count        — integer count
  rate         — generic rate or yield
  text         — string answers: names, places, identifiers (Line 1 must be wrapped in double quotes)

Rules:
- FX rates are always positive floats. Never return 0 or -1.
- Event years are plain integers.
- For multiple dates: line 1 is a Python list in request order.
- For named-entity questions ("Which Bureau...", "What city..."): wrap the answer in double quotes
  on line 1 (e.g. "Bureau of the Public Debt") and use unit `text` on line 2.
- Never return float('nan'), None, or other non-literal expressions; if you genuinely don't know,
  return the string "unknown" with unit text.
"""


def run(op: OpNode, prev: None, ctx: HarnessContext) -> TypedValue:
    nl = op.args.get("nl", "")
    if not nl:
        raise StepFailed("lookup_external", "Missing 'nl' arg")
    if ctx.cache_only:
        raise StepFailed("lookup_external", "cache_only mode: live external calls disabled")

    ctx.emit("lookup_external", "calling gemini", nl=nl)
    raw = call_gemini(_SYSTEM, nl)
    ctx.emit("lookup_external", "gemini response", raw=raw[:500])

    try:
        value, dtype, unit = parse_llm_value(raw)
    except ValueError as e:
        raise StepFailed("lookup_external", f"Cannot parse LLM response: {e}\nRaw: {raw[:200]}") from e

    ctx.emit("lookup_external", "parsed", value=repr(value)[:200], dtype=dtype, unit=unit)
    return TypedValue(value=value, dtype=dtype, unit=unit, desc=nl)
