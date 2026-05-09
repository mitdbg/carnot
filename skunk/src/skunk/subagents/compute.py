"""compute subagent — Gemini generates Python from NL; sandboxed exec with error feedback loop."""

from __future__ import annotations

import textwrap

from skunk.dsl import OpNode, TypedValue
from skunk.subagents.base import HarnessContext, Subagent, StepFailed, call_gemini, exec_python

_MAX_ATTEMPTS = 3

_SYSTEM = """\
You are a Python code generation assistant for a financial data pipeline.

The input value is available as `prev`. It is one of:
- A TypedValue object: .value (the data), .dtype (str), .desc (str)
- A list of TypedValue objects (parallel branches): prev[0].value, prev[1].value, …

Write a short Python block that performs the computation described and assigns the answer to `result`.

Available: numpy (np), pandas (pd), math, statsmodels. Do NOT import anything else. No print statements.
Output ONLY executable Python code — no markdown fences, no explanation.
"""


def _prev_desc(prev: object) -> str:
    if isinstance(prev, TypedValue):
        return (
            f"TypedValue(\n"
            f"  dtype={prev.dtype!r},\n"
            f"  value={repr(prev.value)[:200]},\n"
            f"  desc={prev.desc!r}\n"
            f")"
        )
    if isinstance(prev, list):
        parts = []
        for i, p in enumerate(prev):
            if isinstance(p, TypedValue):
                parts.append(
                    f"  prev[{i}] = TypedValue(dtype={p.dtype!r}, value={repr(p.value)[:120]})"
                )
            else:
                parts.append(f"  prev[{i}] = {repr(p)[:120]}")
        return "[\n" + "\n".join(parts) + "\n]"
    return repr(prev)[:300]


class ComputeSubagent(Subagent):
    op_name = "compute"

    def run(
        self,
        op: OpNode,
        prev: object,
        ctx: HarnessContext,
    ) -> TypedValue:
        nl = op.args.get("nl") or op.args.get("description", "")
        if not nl:
            raise StepFailed("compute", "Missing 'nl' arg describing the computation")

        prev_desc = _prev_desc(prev)
        base_user = (
            f"Computation: {nl}\n\n"
            f"prev =\n{prev_desc}\n\n"
            f"Write Python that assigns the answer to `result`."
        )

        user = base_user
        last_err: str = ""

        for attempt in range(_MAX_ATTEMPTS):
            code = call_gemini(_SYSTEM, user)

            try:
                value = exec_python(code, {"prev": prev})
            except StepFailed as e:
                last_err = str(e)
                # Feed the error back for the next attempt
                user = (
                    f"{base_user}\n\n"
                    f"Attempt {attempt + 1} produced this code:\n"
                    f"{textwrap.indent(code, '  ')}\n\n"
                    f"It failed with:\n  {last_err}\n\n"
                    f"Fix the error and return corrected Python code only."
                )
                continue

            if isinstance(prev, TypedValue):
                unit = prev.unit
            elif isinstance(prev, list):
                units = {p.unit for p in prev if isinstance(p, TypedValue)}
                unit = next(iter(units)) if len(units) == 1 else ""
            else:
                unit = ""
            dtype = "list[scalar]" if isinstance(value, list) else "scalar"
            return TypedValue(value=value, dtype=dtype, unit=unit, desc=nl)

        raise StepFailed(
            "compute",
            f"All {_MAX_ATTEMPTS} attempts failed. Last error: {last_err}\nNL: {nl!r}",
        )
