"""format subagent — LLM generates Python formatting code; exec applies it."""

from __future__ import annotations

from skunk.dsl import FormattedString, OpNode, TypedValue
from skunk.subagents.base import HarnessContext, Subagent, StepFailed, call_gemini, exec_python

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


def _build_prompt(op: OpNode, prev_value: object, unit: str = "") -> str:
    args_desc = ", ".join(f"{k}={v!r}" for k, v in op.args.items())
    value_repr = repr(prev_value)[:200]
    unit_hint = f"\nInput unit: {unit}" if unit else ""
    return (
        f"Format the value into a string.\n\n"
        f"Formatting spec (DSL args): {args_desc}{unit_hint}\n"
        f"value = {value_repr}\n\n"
        f"Write Python code that sets `result`."
    )


class FormatSubagent(Subagent):
    op_name = "format"

    def run(
        self,
        op: OpNode,
        prev: object,
        ctx: HarnessContext,
    ) -> FormattedString:
        if isinstance(prev, TypedValue):
            value = prev.value
            unit = prev.unit
            desc = prev.desc
        elif isinstance(prev, FormattedString):
            return prev
        else:
            value = prev
            unit = ""
            desc = ""

        prompt = _build_prompt(op, value, unit=unit)
        code = call_gemini(_SYSTEM, prompt)

        # Strip any accidental markdown fences the model adds
        code = code.strip()
        if code.startswith("```"):
            lines = code.splitlines()
            code = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        try:
            result_str = exec_python(code, {"value": value})
        except StepFailed as e:
            raise StepFailed("format", f"Generated code failed: {e}\nCode:\n{code}") from e

        if not isinstance(result_str, str):
            result_str = str(result_str)

        return FormattedString(text=result_str, desc=desc)
