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
from skunk.dsl import FormattedString, NamedEntry, OpNode, TypedValue
from skunk.subagents.base import StepFailed, call_gemini, exec_python, strip_code_fences

_MAX_ATTEMPTS = 3

_CODEGEN_SYSTEM = """\
You are the chain terminator for a financial QA pipeline.

You receive:
- The user's original question (free text).
- `prev`: extracted bulletin data, available in the sandbox. Shapes:
    * TypedValue with .dtype == 'named':
        .value is dict[str, scalar]   (each value is a number or string)
        .meta  is dict[str, NamedEntry] keyed identically, with:
              .unit  — semantic unit (usd, usd_millions, pct, count, year, text, ...)
              .quote — verbatim phrase from the page that anchors the value
              .dims  — small dict of categorical labels distinguishing siblings
                       (e.g. {"denomination": 1, "series": "Total"})
    * TypedValue with another dtype: a single scalar/list/df; .unit gives the unit.
    * list[TypedValue]: parallel branches indexed in question order; use prev[i].
- `lookup` (alias for prev when prev is a list-of-TypedValue): use prev[i].value.

Working with named entries:
- Every named scalar carries its unit in prev[i].meta[name].unit — trust that, never
  guess units from the key name.
- When the same kind of measurement was extracted for many siblings (e.g. one scalar
  per row of a table), the `dims` dict tells you what each scalar represents. To iterate
  the siblings, filter prev[i].value.items() by prev[i].meta[k].dims. Example:
      vals = [v for k, v in prev[i].value.items()
              if prev[i].meta[k].dims.get("series") == "Total"]
- Numeric `dims` labels (e.g. dims["denomination"] = 1) are the actual numeric facts —
  use them in arithmetic, not the key name.

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

# Match CODE / MISSING anywhere (after fence-stripping + light prose). Use re.search,
# not re.match, so leading commentary or whitespace doesn't break classification.
_CODE_RE = re.compile(r"\bCODE\s*\n(.*)\Z", re.DOTALL)
_MISSING_RE = re.compile(r"\bMISSING:\s*([^\n]*)", re.IGNORECASE)


def _render_named_entry(prefix: str, name: str, value: object, entry: NamedEntry | None) -> list[str]:
    """Render one named scalar with its metadata (untruncated)."""
    unit = entry.unit if entry is not None else ""
    quote = entry.quote if entry is not None else ""
    dims = entry.dims if entry is not None else {}
    out = [f"{prefix}{name} = {value!r}  (unit={unit!r}, dims={dims!r})"]
    if quote:
        out.append(f"{prefix}  quote: {quote!r}")
    return out


def _render_typed_value(tv: TypedValue, prefix: str) -> str:
    if tv.dtype == "named" and isinstance(tv.value, dict):
        lines = [f"{prefix}TypedValue(dtype='named', {len(tv.value)} entries)"]
        for k, v in tv.value.items():
            entry = (tv.meta or {}).get(k)
            lines.extend(_render_named_entry(prefix + "  ", k, v, entry))
        return "\n".join(lines)
    return (
        f"{prefix}TypedValue(dtype={tv.dtype!r}, unit={tv.unit!r}, "
        f"value={tv.value!r}, desc={tv.desc!r})"
    )


def _prev_desc(prev: object) -> str:
    if isinstance(prev, TypedValue):
        return _render_typed_value(prev, "")
    if isinstance(prev, list):
        parts = ["["]
        for i, p in enumerate(prev):
            parts.append(f"  prev[{i}] =")
            if isinstance(p, TypedValue):
                parts.append(_render_typed_value(p, "    "))
            else:
                parts.append(f"    {p!r}")
        parts.append("]")
        return "\n".join(parts)
    return repr(prev)


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
    """Ask the verifier LLM whether `answer_text` is in the right format/unit.

    Returns (passed, reason). Conservative when ambiguous: if both PASS and FAIL
    appear in the reply, FAIL wins. If neither appears, treat as FAIL with a
    diagnostic reason rather than silently accepting.
    """
    user = (
        f"Question:\n{question}\n\n"
        f"Computed answer:\n{answer_text}\n\n"
        f"Reply on a single line: PASS, or FAIL: <reason>."
    )
    raw = call_gemini(_VERIFIER_SYSTEM, user)
    ctx.emit("compute", "verifier response", raw=raw[:300])

    text = raw.strip()
    upper = text.upper()
    has_fail = "FAIL" in upper
    has_pass = "PASS" in upper

    if has_fail:
        # Extract a reason — text after the first FAIL token, up to a newline.
        m = re.search(r"FAIL[: ]?\s*([^\n]*)", text, re.IGNORECASE)
        reason = m.group(1).strip() if m and m.group(1).strip() else "verifier rejected (no reason)"
        return False, reason
    if has_pass:
        return True, ""
    return False, f"verifier produced malformed reply: {text[:120]!r}"


def run(op: OpNode, prev: object, ctx: HarnessContext) -> FormattedString:
    ctx.emit("compute", "starting", question=ctx.question, prev_summary=_prev_desc(prev)[:500])

    prior: str | None = None
    last_code: str = ""
    last_result_text: str = ""

    for attempt in range(_MAX_ATTEMPTS):
        raw = call_gemini(_CODEGEN_SYSTEM, _build_user(ctx.question, prev, prior))
        ctx.emit("compute", f"attempt {attempt + 1} codegen response", raw=raw[:600])

        # Strip outer markdown fences before scanning for CODE / MISSING tokens.
        cleaned = strip_code_fences(raw)

        m_missing = _MISSING_RE.search(cleaned)
        if m_missing:
            reason = m_missing.group(1).strip()
            ctx.emit("compute", f"attempt {attempt + 1} reported MISSING", reason=reason)
            raise StepFailed("compute", f"missing data to answer: {reason}")

        m_code = _CODE_RE.search(cleaned)
        if not m_code:
            prior = f"Response had neither CODE nor MISSING:. Raw: {raw[:300]}"
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
