"""compute subagent — chain terminator. Self-plans, codegens, execs, self-critiques.

Subsumes the old `format` op. Reads ctx.question and prev (extracted values),
generates Python, runs it in the sandbox, then a self-critique LLM call (same
domain context as the producer) decides whether to ship the result or revise.
Returns FormattedString as the chain's final answer.

Failure modes (StepFailed):
- Attempt 1 produced no result across all transient retries
- Attempt 2 produced no result AND attempt 1 had no result either
- Codegen reported "MISSING:" on attempt 1 — extracted values insufficient
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

_CRITIQUE_SYSTEM = """\
You are reviewing code that you (the chain terminator) just wrote and the
string it produced. Decide whether to ship the result as-is or revise.

You receive the same context the producer had: the question, a summary of
`prev` (the TypedValue input — including each entry's unit, kind, dims, and
the originating quote), the python code that ran, and the produced
`result` string.

Reply on a single line:
  ACCEPT
  REVISE: <one short reason a re-run should address>

Bias toward ACCEPT. Only REVISE when the producer:
  - applied the wrong unit conversion (e.g. shipped usd_thousands while the
    question asked for usd_millions, or never converted),
  - produced a result whose form clearly contradicts the question (asked
    for "[a, b]" bracketed list, shipped "1.0 2.0"; asked for percent
    form, shipped a decimal like "0.1234"),
  - selected the wrong rows/columns from `prev` given the question's
    explicit constraints (wrong year, wrong series, wrong dim filter).

Do NOT REVISE on:
  - cosmetic precision when the question doesn't pin precision,
  - presence/absence of a trailing unit suffix when the magnitude is right
    and the question doesn't explicitly demand the suffix,
  - whitespace, capitalization, or punctuation nits.

When in doubt, ACCEPT.
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


def _build_user(question: str, prev_desc: str, priors: list[str]) -> str:
    msg = (
        f"Question:\n{question}\n\n"
        f"prev =\n{prev_desc}\n\n"
        f"Produce CODE or MISSING:."
    )
    if priors:
        msg += "\n\nPrior attempts (oldest first):\n"
        for i, p in enumerate(priors, 1):
            msg += f"\n[Attempt {i}]\n{p}\n"
        msg += "\nDo not repeat any of the above mistakes."
    return msg


def _self_critique(
    question: str, prev_desc: str, code: str, result_text: str, ctx: HarnessContext,
) -> tuple[bool, str]:
    """Same-actor review of (code, result) against the question.

    Returns (accept, reason). Conservative when ambiguous: malformed reply or both
    tokens present → REVISE with a diagnostic reason.
    """
    user = (
        f"Question:\n{question}\n\n"
        f"prev =\n{prev_desc}\n\n"
        f"Code that ran:\n```python\n{code}\n```\n\n"
        f"Produced result:\n{result_text}\n\n"
        f"Reply on a single line: ACCEPT, or REVISE: <reason>."
    )
    raw = ctx.llm_client.call(_CRITIQUE_SYSTEM, user)
    ctx.emit("compute", "self-critique response", raw=raw[:300])

    text = raw.strip()
    upper = text.upper()
    has_revise = "REVISE" in upper
    has_accept = "ACCEPT" in upper

    if has_revise:
        m = re.search(r"REVISE[: ]?\s*([^\n]*)", text, re.IGNORECASE)
        reason = m.group(1).strip() if m and m.group(1).strip() else "self-critique flagged revise (no reason)"
        return False, reason
    if has_accept:
        return True, ""
    return False, f"self-critique produced malformed reply: {text[:120]!r}"


def _try_codegen_and_exec(
    ctx: HarnessContext,
    prev: object,
    prev_desc: str,
    priors: list[str],
    retry_budget: int,
) -> tuple[str | None, str | None, list[str]]:
    """One logical attempt: codegen → parse → exec, with `retry_budget` retries
    for transient exec/parse failures (a malformed response or a code exception).

    MISSING: from codegen propagates as MissingData. Returns
    (result_text, code, accumulated_priors) on success, or
    (None, None, accumulated_priors) when the retry budget is exhausted.
    """
    priors = list(priors)
    # Total tries = 1 initial + retry_budget retries.
    for try_idx in range(retry_budget + 1):
        raw = ctx.llm_client.call(
            _CODEGEN_SYSTEM, _build_user(ctx.question, prev_desc, priors), thinking_budget=-1,
        )
        ctx.emit("compute", f"codegen try {try_idx + 1} response", raw=raw[:600])

        cleaned = strip_code_fences(raw)
        m_code = _CODE_RE.search(cleaned)
        if not m_code:
            m_missing = _MISSING_RE.search(cleaned)
            if m_missing:
                reason = m_missing.group(1).strip()
                ctx.emit("compute", f"codegen try {try_idx + 1} reported MISSING", reason=reason)
                raise MissingData(reason)
            priors.append(f"Response had neither CODE nor MISSING:. Raw: {raw[:300]}")
            ctx.emit("compute", f"codegen try {try_idx + 1} malformed", prior=priors[-1][:300])
            continue

        code = m_code.group(1).strip()
        ctx.emit("compute", f"codegen try {try_idx + 1} code", code=code)

        try:
            result_value = exec_python(code, {"prev": prev})
        except Exception as e:
            priors.append(f"Code:\n```python\n{code}\n```\nException: {e}")
            ctx.emit("compute", f"codegen try {try_idx + 1} exec failed", error=str(e))
            continue

        result_text = str(result_value)
        ctx.emit("compute", f"codegen try {try_idx + 1} produced result", text=result_text)
        return result_text, code, priors

    return None, None, priors


def run(op: OpNode, prev: object, ctx: HarnessContext) -> FormattedString:
    prev_desc = _prev_desc(prev)
    ctx.emit("compute", "starting", question=ctx.question, prev_summary=prev_desc[:500])

    # Attempt 1: codegen → exec → self-critique
    a1_result, a1_code, a1_priors = _try_codegen_and_exec(
        ctx, prev, prev_desc, priors=[], retry_budget=ctx.config.compute_max_attempts - 1,
    )
    if a1_result is None:
        raise StepFailed(
            "compute",
            f"could not produce a result on attempt 1: priors={a1_priors}",
        )

    accept, reason = _self_critique(ctx.question, prev_desc, a1_code, a1_result, ctx)
    if accept:
        ctx.emit("compute", "attempt 1 self-critique ACCEPT", text=a1_result)
        return FormattedString(text=a1_result)
    ctx.emit("compute", "attempt 1 self-critique REVISE", reason=reason)

    # Attempt 2: codegen with critique as prior, ship unconditionally (no re-critique → no flap)
    hint = (
        f"Produced result {a1_result!r}. Self-critique flagged: {reason}\n"
        f"Code was:\n```python\n{a1_code}\n```"
    )
    try:
        a2_result, _, _ = _try_codegen_and_exec(
            ctx, prev, prev_desc, priors=[hint], retry_budget=0,
        )
    except MissingData as e:
        # Codegen had data on attempt 1; on attempt 2 it's reacting to critique
        # feedback, not to a real absence. Fall back rather than mask the result.
        ctx.emit("compute", "attempt 2 MISSING; falling back to attempt 1",
                 reason=e.reason, fallback_text=a1_result)
        return FormattedString(text=a1_result)
    if a2_result is None:
        ctx.emit(
            "compute",
            "attempt 2 produced no result; falling back to attempt 1",
            fallback_text=a1_result,
        )
        return FormattedString(text=a1_result)
    ctx.emit("compute", "attempt 2 returned (no re-critique)", text=a2_result)
    return FormattedString(text=a2_result)
