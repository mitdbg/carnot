"""format subagent — LLM generates Python formatting code; exec applies it."""

from __future__ import annotations

from skunk.common.context import HarnessContext
from skunk.dsl import FormattedString, OpNode, TypedValue
from skunk.subagents.base import call_gemini, exec_python

_SYSTEM = """\
You are a Python code generation assistant. Write a short Python block that formats \
a numeric value (or list) into a concise, bare answer string.

The input value is available as the variable `value`.
You MUST set a variable named `result` containing the final formatted string.

Rules:
- result must be a plain Python str — a single line, no prose, no explanation
- For a single number: just the formatted number and any required symbol (e.g. "1608.80%", "81.406")
- For a list: bracket notation with each element formatted, e.g. "[1.2, 3.4, 5.6]"
- Precision means decimal places; use round() or f-strings
- For unit='pct': append "%" directly to the number (e.g. "1608.80%")
- For dollar units: no "$" prefix unless the question specifically asks; plain number is fine
- Do NOT output sentences, units as words, or explanations — only the bare answer
- Output ONLY executable Python code, no markdown fences, no explanation
"""


def run(op: OpNode, prev: object, ctx: HarnessContext) -> FormattedString:
    if isinstance(prev, TypedValue):
        value, unit, desc = prev.value, prev.unit, prev.desc
    elif isinstance(prev, FormattedString):
        return prev
    else:
        value, unit, desc = prev, "", ""

    args_desc = ", ".join(f"{k}={v!r}" for k, v in op.args.items())
    unit_hint = f"\nInput unit: {unit}" if unit else ""
    prompt = (
        f"Format the value into a string.\n\n"
        f"Formatting spec (DSL args): {args_desc}{unit_hint}\n"
        f"value = {repr(value)[:200]}\n\n"
        f"Write Python code that sets `result`."
    )
    code = call_gemini(_SYSTEM, prompt)
    result_str = exec_python(code, {"value": value})
    if not isinstance(result_str, str):
        result_str = str(result_str)
    return FormattedString(text=result_str, desc=desc)
