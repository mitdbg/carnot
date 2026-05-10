"""compute subagent — Gemini generates Python from NL; single exec."""

from __future__ import annotations

from skunk.dsl import OpNode, TypedValue
from skunk.subagents.base import HarnessContext, StepFailed, call_gemini, exec_python

_SYSTEM = """\
You are a Python code generation assistant for a financial data pipeline.

The input value is available as `prev`. It is one of:
- A TypedValue object: .value (the data), .dtype (str), .desc (str)
- A list of TypedValue objects (parallel branches): prev[0].value, prev[1].value, …

Write a short Python block that performs the computation described and assigns the answer to `result`.

Available: numpy (np), pandas (pd), math. Do NOT import anything else. No print statements.
Output ONLY executable Python code — no markdown fences, no explanation.
"""


def _prev_desc(prev: object) -> str:
    if isinstance(prev, TypedValue):
        return f"TypedValue(dtype={prev.dtype!r}, value={repr(prev.value)[:200]}, desc={prev.desc!r})"
    if isinstance(prev, list):
        parts = [
            f"  prev[{i}] = TypedValue(dtype={p.dtype!r}, value={repr(p.value)[:120]})"
            if isinstance(p, TypedValue) else f"  prev[{i}] = {repr(p)[:120]}"
            for i, p in enumerate(prev)
        ]
        return "[\n" + "\n".join(parts) + "\n]"
    return repr(prev)[:300]


def run(op: OpNode, prev: object, ctx: HarnessContext) -> TypedValue:
    nl = op.args.get("nl") or op.args.get("description", "")
    if not nl:
        raise StepFailed("compute", "Missing 'nl' arg describing the computation")

    user = (
        f"Computation: {nl}\n\n"
        f"prev =\n{_prev_desc(prev)}\n\n"
        f"Write Python that assigns the answer to `result`."
    )
    code = call_gemini(_SYSTEM, user)
    value = exec_python(code, {"prev": prev})

    if isinstance(prev, TypedValue):
        unit = prev.unit
    elif isinstance(prev, list):
        units = {p.unit for p in prev if isinstance(p, TypedValue)}
        unit = next(iter(units)) if len(units) == 1 else ""
    else:
        unit = ""
    dtype = "list[scalar]" if isinstance(value, list) else "scalar"
    return TypedValue(value=value, dtype=dtype, unit=unit, desc=nl)
