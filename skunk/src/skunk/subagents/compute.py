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
from skunk.subagents.base import MissingData, StepFailed, exec_python, strip_code_fences

_CODEGEN_SYSTEM = """\
You are the chain terminator for a financial QA pipeline.

You receive:
- The user's original question (free text).
- `prev`: extracted bulletin data, available in the sandbox. Always a TypedValue or list[TypedValue].
    TypedValue:
        .value is dict[str, Any]  — keyed by name; a single unnamed result uses key ""
        .meta  is dict[str, NamedEntry] keyed identically, with per-entry:
              .unit       — semantic unit (usd_millions, pct, count, year, text, fx_rate, ...)
              .quote      — verbatim phrase from the page that anchors the value
              .dims       — small dict of categorical labels distinguishing siblings
                            (e.g. {"denomination": 1, "series": "Total"})
              .kind       — "scalar" | "vector" | "table"
              .index_name — vector only: name of the varying dim (e.g. "month")
              .row_name   — table only: name of the row dim
              .col_name   — table only: name of the column dim
    list[TypedValue]: parallel branches indexed in question order; use prev[i].

Payload shapes by kind:
  scalar:  prev.value[k] is the number or string itself.
  vector:  prev.value[k] is an insertion-ordered dict {index_label: scalar}.
           Use list(prev.value[k].values()) for whole-series ops, or .items() to filter
           by index label (e.g. a date range).
  table:   prev.value[k] is a 2-level dict {row_label: {col_label: scalar}}.
           Cast with pd.DataFrame.from_dict(prev.value[k], orient="index") for normal
           table operations.

Working with entries:
- For a single unnamed result: prev.value[""] gives the value; prev.meta[""].unit gives the unit.
- For named entries: iterate prev.value.items() and look up prev.meta[k] for metadata.
- Every entry's unit is in .meta[k].unit — trust that, never guess from the key name.
- For scalar siblings sharing a measurement type, filter by dims. Example:
      vals = [v for k, v in prev.value.items()
              if prev.meta[k].kind == "scalar"
              and prev.meta[k].dims.get("series") == "Total"]
- Numeric dims values (e.g. dims["denomination"] = 1) are actual facts — use them in
  arithmetic, not the key name.

Worked example — geometric mean over a vector:
    # prev has one entry "budget_expenditures" with kind="vector", index_name="month".
    vals = list(prev.value['budget_expenditures'].values())
    gm = float(np.exp(np.mean(np.log(vals))))
    result = f"{gm:.2f} millions of nominal dollars"

Worked example — date-range filter on a vector:
    # Keep only cells whose index label falls in [start, end] (ISO month strings sort lexically).
    series = prev.value['budget_expenditures']
    selected = [v for k, v in series.items() if "1942-03" <= k <= "1948-10"]
    gm = float(np.exp(np.mean(np.log(selected))))
    result = f"{gm:.2f}"

Worked example — column sum on a table:
    df = pd.DataFrame.from_dict(prev.value['receipts_by_year_month'], orient="index")
    total_1943 = float(df.loc["1943"].sum())
    result = f"{total_1943:,.0f}"

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
  Every cell in a vector or table shares one unit — convert once over the whole payload.
- Available imports: numpy (np), pandas (pd), math, statsmodels.api (sm). No other imports. No prints.
- Output ONLY the format above — no markdown fences around the whole response.
"""

# TODO: verifier currently checks format/unit only and will PASS clearly implausible magnitudes.
# Witnessed: UID0009 produced "$1.000" for "weighted average denomination of U.S. currency in
# circulation" (true value ≈ $32.66) because the compute LLM misread a column; the format was
# right (3 decimals + $) so the verifier accepted it. Future enhancement: extend the prompt with
# a common-sense magnitude clause — e.g. percent answers in [0%, 100%], years in plausible bounds,
# "average U.S. bill denomination" in [$1, $100]. Frame as "when in doubt, PASS" to avoid
# false-fail regressions on legitimate edge cases.
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


def _summarize_vector(value: dict) -> str:
    """Short, codegen-friendly summary of a vector payload. Shows index range + a sample of cells."""
    if not isinstance(value, dict) or not value:
        return repr(value)
    keys = list(value.keys())
    n = len(keys)
    head = ", ".join(f"{k!r}: {value[k]!r}" for k in keys[:3])
    if n <= 3:
        return f"{{{head}}}"
    return f"{{{head}, ...}} ({n} cells, index range {keys[0]!r}..{keys[-1]!r})"


def _summarize_table(value: dict) -> str:
    """Short summary of a table payload — row count, col count, a sample row."""
    if not isinstance(value, dict) or not value:
        return repr(value)
    rows = list(value.keys())
    first_row = value[rows[0]]
    cols = list(first_row.keys()) if isinstance(first_row, dict) else []
    n_rows, n_cols = len(rows), len(cols)
    sample_cells = ", ".join(f"{c!r}: {first_row[c]!r}" for c in cols[:3])
    more = ", ..." if n_cols > 3 else ""
    return (
        f"{{{rows[0]!r}: {{{sample_cells}{more}}}, ...}} "
        f"({n_rows} rows × {n_cols} cols)"
    )


def _render_named_entry(prefix: str, name: str, value: object, entry: NamedEntry | None) -> list[str]:
    unit = entry.unit if entry is not None else ""
    quote = entry.quote if entry is not None else ""
    dims = entry.dims if entry is not None else {}
    kind = entry.kind if entry is not None else "scalar"
    label = name if name else '""'

    if kind == "scalar":
        shape_str = "kind=scalar"
        value_str = repr(value)
    elif kind == "vector":
        idx = entry.index_name if entry is not None else None
        shape_str = f"kind=vector, index_name={idx!r}"
        value_str = _summarize_vector(value) if isinstance(value, dict) else repr(value)
    elif kind == "table":
        rn = entry.row_name if entry is not None else None
        cn = entry.col_name if entry is not None else None
        shape_str = f"kind=table, row_name={rn!r}, col_name={cn!r}"
        value_str = _summarize_table(value) if isinstance(value, dict) else repr(value)
    else:
        shape_str = f"kind={kind!r}"
        value_str = repr(value)

    out = [f"{prefix}{label} = {value_str}  ({shape_str}, unit={unit!r}, dims={dims!r})"]
    if quote:
        out.append(f"{prefix}  quote: {quote!r}")
    return out


def _render_typed_value(tv: TypedValue, prefix: str) -> str:
    lines = [f"{prefix}TypedValue({len(tv.value)} entries)"]
    for k, v in tv.value.items():
        lines.extend(_render_named_entry(prefix + "  ", k, v, tv.meta.get(k)))
    return "\n".join(lines)


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


def _build_user(question: str, prev: object, priors: list[str]) -> str:
    msg = (
        f"Question:\n{question}\n\n"
        f"prev =\n{_prev_desc(prev)}\n\n"
        f"Produce CODE or MISSING:."
    )
    if priors:
        msg += "\n\nPrior attempts (oldest first):\n"
        for i, p in enumerate(priors, 1):
            msg += f"\n[Attempt {i}]\n{p}\n"
        msg += "\nDo not repeat any of the above mistakes."
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
    raw = ctx.llm_client.call(_VERIFIER_SYSTEM, user)
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

    priors: list[str] = []
    last_code: str = ""
    last_result_text: str = ""

    for attempt in range(ctx.config.compute_max_attempts):
        raw = ctx.llm_client.call(_CODEGEN_SYSTEM, _build_user(ctx.question, prev, priors), thinking_budget=-1)
        ctx.emit("compute", f"attempt {attempt + 1} codegen response", raw=raw[:600])

        # Strip outer markdown fences before scanning for CODE / MISSING tokens.
        cleaned = strip_code_fences(raw)

        # Check CODE first: per the prompt's "exactly one of" contract, a CODE response
        # may legitimately contain the literal substring "MISSING:" inside Python string
        # literals (e.g. a defensive `raise ValueError("MISSING: ...")`). Only treat the
        # response as MISSING when no CODE block is present.
        m_code = _CODE_RE.search(cleaned)
        if not m_code:
            m_missing = _MISSING_RE.search(cleaned)
            if m_missing:
                reason = m_missing.group(1).strip()
                ctx.emit("compute", f"attempt {attempt + 1} reported MISSING", reason=reason)
                raise MissingData(reason)
            priors.append(f"Response had neither CODE nor MISSING:. Raw: {raw[:300]}")
            ctx.emit("compute", f"attempt {attempt + 1} malformed response", prior=priors[-1][:300])
            continue

        code = m_code.group(1).strip()
        last_code = code
        ctx.emit("compute", f"attempt {attempt + 1} code", code=code)

        try:
            result_value = exec_python(code, {"prev": prev})
        except Exception as e:
            priors.append(f"Code:\n```python\n{code}\n```\nException: {e}")
            ctx.emit("compute", f"attempt {attempt + 1} exec failed", error=str(e))
            continue

        result_text = result_value if isinstance(result_value, str) else str(result_value)
        last_result_text = result_text
        ctx.emit("compute", f"attempt {attempt + 1} produced result", text=result_text)

        passed, reason = _verify(ctx.question, result_text, ctx)
        if passed:
            ctx.emit("compute", f"attempt {attempt + 1} verified", text=result_text)
            return FormattedString(text=result_text)

        priors.append(
            f"Produced result {result_text!r} but the verifier rejected it: {reason}\n"
            f"Code was:\n```python\n{code}\n```"
        )
        ctx.emit("compute", f"attempt {attempt + 1} verifier rejected", reason=reason)

    raise StepFailed(
        "compute",
        f"{ctx.config.compute_max_attempts} attempts failed. Last code: {last_code!r}. "
        f"Last result: {last_result_text!r}. Priors: {priors}"
    )
