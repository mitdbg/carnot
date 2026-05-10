"""compute subagent — chain terminator. Self-plans, codegens, execs, verifies.

Subsumes the old `format` op. Reads ctx.question and prev (extracted values),
generates Python, runs it in the sandbox, then a verifier LLM checks that the
output's form/unit/precision match what the question asks for. Returns
FormattedString as the chain's final answer.

Failure modes (StepFailed):
- All N codegen+exec attempts raised an exception
- Verifier rejected all N attempts
- Codegen reported "MISSING:" — extracted values insufficient to answer
"""

from __future__ import annotations

import re

from skunk.common.context import HarnessContext
from skunk.dsl import FormattedString, OpNode, TypedValue
from skunk.subagents.base import StepFailed, call_gemini, exec_python

_MAX_ATTEMPTS = 3

_CODEGEN_SYSTEM = """\
You are the chain terminator for a financial QA pipeline.

You receive:
- The user's original question (free text).
- `prev`: extracted bulletin data, available in the sandbox. Shapes:
    * TypedValue with .dtype == 'named': .value is a dict {snake_case_name: scalar/list/table};
      the printed unit of each name is described in .desc.
    * TypedValue with another dtype: a single scalar/list/df; .unit gives the unit.
    * list[TypedValue]: parallel branches; index in question order.
- `lookup` (alias for prev when prev is a list-of-TypedValue): use prev[i].value.

Your job: produce the answer string the question asks for. This means
(a) plan the computation, (b) write Python that assigns the final answer
to `result` AS A STRING formatted exactly as the question requests.

Output format — exactly one of:
  CODE\\n<python code that sets `result` to a string>
  MISSING:<short explanation of which datum is missing>

Rules:
- If you can answer from `prev`, output CODE then the Python block.
- If `prev` is missing a value you need (e.g. only one of two periods present,
  no FX rate, no CPI), output MISSING:<reason>. Do NOT fabricate values.
- `result` MUST be a Python str. For numbers, format per the question's instructions
  (precision, with/without commas, with/without unit symbols). For lists, use the
  bracket form the question requests. For percent answers, append "%" only if the
  question asks for percent form.
- Convert units yourself when needed (e.g. usd_thousands → usd_millions: divide by 1000).
- Available imports: numpy (np), pandas (pd), math, statsmodels.api (sm). No other imports. No prints.
- Output ONLY the format above — no markdown fences around the whole response.
"""

_VERIFIER_SYSTEM = """\
You verify whether a computed answer is in the correct FORMAT and UNIT to answer the user's question.
You are NOT verifying numeric correctness — only that the *shape* of the answer matches what the
question literally asks for: precision, units, percent vs decimal, comma vs no-comma, list vs scalar,
required prefix/suffix symbols.

Reply with exactly one line:
  PASS
or
  FAIL: <one short reason describing the format/unit mismatch>

Examples:
- Question asks "rounded to the nearest hundredths place" but answer is "1.234" → FAIL: needs 2 decimals
- Question asks for percent value "12.34%" but answer is "0.1234" → FAIL: must be in percent form, not decimal
- Question asks "no commas" but answer is "1,234,567" → FAIL: contains commas
- Question asks "[a, b]" bracketed list but answer is "1.0 2.0" → FAIL: must be bracketed list
- Question asks rounded integer and answer is "8998" → PASS
"""

_CODE_RE = re.compile(r"^\s*CODE\s*\n(.*)$", re.DOTALL)
_MISSING_RE = re.compile(r"^\s*MISSING:\s*(.*)$", re.DOTALL)


def _prev_desc(prev: object) -> str:
    if isinstance(prev, TypedValue):
        if prev.dtype == "named" and isinstance(prev.value, dict):
            lines = [f"TypedValue(dtype='named', desc={prev.desc!r})", "  prev.value is a dict with keys:"]
            for k, v in prev.value.items():
                lines.append(f"    {k!r}: {repr(v)[:160]}")
            return "\n".join(lines)
        return f"TypedValue(dtype={prev.dtype!r}, value={repr(prev.value)[:200]}, desc={prev.desc!r})"
    if isinstance(prev, list):
        parts = []
        for i, p in enumerate(prev):
            if isinstance(p, TypedValue) and p.dtype == "named" and isinstance(p.value, dict):
                parts.append(f"  prev[{i}] = TypedValue(dtype='named', desc={p.desc!r})")
                for k, v in p.value.items():
                    parts.append(f"    prev[{i}].value[{k!r}] = {repr(v)[:120]}")
            elif isinstance(p, TypedValue):
                parts.append(f"  prev[{i}] = TypedValue(dtype={p.dtype!r}, value={repr(p.value)[:120]})")
            else:
                parts.append(f"  prev[{i}] = {repr(p)[:120]}")
        return "[\n" + "\n".join(parts) + "\n]"
    return repr(prev)[:300]


def _build_user(question: str, prev: object, prior_failure: str | None) -> str:
    msg = (
        f"Question:\n{question}\n\n"
        f"prev =\n{_prev_desc(prev)}\n\n"
        f"Produce CODE or MISSING:."
    )
    if prior_failure:
        msg += f"\n\nPrevious attempt failed:\n{prior_failure}\n\nTry again."
    return msg


def _verify(question: str, answer_text: str, ctx: HarnessContext) -> tuple[bool, str]:
    user = (
        f"Question:\n{question}\n\n"
        f"Computed answer:\n{answer_text}\n\n"
        f"Reply PASS or FAIL: <reason>."
    )
    raw = call_gemini(_VERIFIER_SYSTEM, user)
    ctx.emit("compute", "verifier response", raw=raw[:300])
    head = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    if head.upper().startswith("PASS"):
        return True, ""
    if head.upper().startswith("FAIL"):
        # Strip "FAIL:" prefix
        return False, head.split(":", 1)[1].strip() if ":" in head else head
    # Malformed verifier reply — treat as PASS rather than blocking the chain
    return True, ""


def run(op: OpNode, prev: object, ctx: HarnessContext) -> FormattedString:
    ctx.emit("compute", "starting", question=ctx.question, prev_summary=_prev_desc(prev)[:500])

    prior: str | None = None
    last_code: str = ""
    last_result_text: str = ""

    for attempt in range(_MAX_ATTEMPTS):
        raw = call_gemini(_CODEGEN_SYSTEM, _build_user(ctx.question, prev, prior))
        ctx.emit("compute", f"attempt {attempt + 1} codegen response", raw=raw[:600])

        m_missing = _MISSING_RE.match(raw.strip())
        if m_missing:
            reason = m_missing.group(1).strip()
            ctx.emit("compute", f"attempt {attempt + 1} reported MISSING", reason=reason)
            raise StepFailed("compute", f"missing data to answer: {reason}")

        m_code = _CODE_RE.match(raw.strip())
        if not m_code:
            prior = f"Response did not start with CODE or MISSING:. Raw: {raw[:300]}"
            ctx.emit("compute", f"attempt {attempt + 1} malformed response", prior=prior[:300])
            continue

        code = m_code.group(1).strip()
        last_code = code
        ctx.emit("compute", f"attempt {attempt + 1} code", code=code)

        try:
            result_value = exec_python(code, {"prev": prev})
        except Exception as e:
            prior = f"Attempt {attempt + 1} code:\n```python\n{code}\n```\nException: {e}"
            ctx.emit("compute", f"attempt {attempt + 1} exec failed", error=str(e))
            continue

        result_text = result_value if isinstance(result_value, str) else str(result_value)
        last_result_text = result_text
        ctx.emit("compute", f"attempt {attempt + 1} produced result", text=result_text)

        passed, reason = _verify(ctx.question, result_text, ctx)
        if passed:
            ctx.emit("compute", f"attempt {attempt + 1} verified", text=result_text)
            return FormattedString(text=result_text, desc=ctx.question[:120])

        prior = (
            f"Attempt {attempt + 1} produced result {result_text!r} but the verifier rejected it: {reason}\n"
            f"Code was:\n```python\n{code}\n```"
        )
        ctx.emit("compute", f"attempt {attempt + 1} verifier rejected", reason=reason)

    raise StepFailed(
        "compute",
        f"{_MAX_ATTEMPTS} attempts failed. Last code: {last_code!r}. Last result: {last_result_text!r}. Last error: {prior}"
    )
