"""compute subagent — codegen + execution over an extracted-value sandbox.

Two operators cover two roles. The terminal compute (`CodegenFinalOperator`)
takes the user's question plus `prev: list[AnnotatedValue]`, generates Python
that assigns `result` to the formatted answer string, runs it in a sandbox,
then self-critiques via `CritiqueOperator` and re-runs once on REVISE.
Returns a `FormattedString`.

The intermediate compute (`CodegenIntermediateOperator`) takes a sub-task
plus `prev`, generates Python that assigns `result` to a raw value (scalar,
vector dict, table dict-of-dict, or list[AnnotatedValue]) with optional
`result_unit` / `result_kind` / `result_description` / `result_index_name` /
`result_row_name` / `result_col_name` metadata vars. No critique. Returns
a `list[AnnotatedValue]` for the downstream aggregator.

Codegen reports `MISSING:<reason>` when `prev` lacks a needed value; the
orchestrator catches that and triggers a replan round.
"""

from __future__ import annotations

import re

from skunk.common import HarnessContext
from skunk.dsl import AnnotatedValue, FormattedString, OpNode
from skunk.operator import SkunkOperator
from skunk.subagents.base import (
    MissingData,
    StepFailed,
    exec_python_with_env,
    strip_code_fences,
)


# ---------------------------------------------------------------------------
# Static system prompt blocks
# ---------------------------------------------------------------------------

_CODEGEN_FINAL_SYSTEM = """\
You write Python that produces the final answer string for a financial QA
pipeline.

Inputs:
- The user's question.
- A "Parsed constraints" block in the user message — structured spec of
  the question (method, transforms, units_out, precision, answer_form,
  period_type). Treat as ground truth: every listed transform MUST appear
  as an explicit operation in your code; if method is set, implement its
  standard formula; if units_out is set, convert; if precision is set,
  round.
- `prev`: list[AnnotatedValue] available in the sandbox.

AnnotatedValue:
  .tag         snake_case key, e.g. "national_defense_expenditures:cy1940"
  .description natural-language label
  .value       scalar | vector dict {index: scalar} | table dict-of-dict
  .unit        usd_millions | pct | count | year | text | fx_rate | ...
  .kind        "scalar" | "vector" | "table"
  .index_name  vector only
  .row_name / .col_name   table only

Selecting from prev: PREFER exact tag matching:
  next(e for e in prev if e.tag == "<tag>")
Substring matching on description is fragile (first match wins); fall back
to it only when no tag is present (e.tag == "").

Payload access:
  vector: list(e.value.values()) or .items() to filter by index.
  table:  pd.DataFrame.from_dict(e.value, orient="index").

NUMERIC TRANSFORMS — every transform from the question MUST appear as an
explicit operation, not as a comment. The critic REVISEs on missing or
mis-applied transforms even when the magnitude looks plausible. Watch for:
  - "signed" vs "absolute" — never silently abs() a value meant signed.
  - "normalized" / "midpoint normalized" — apply explicitly (midpoint:
    (V2-V1)/((V1+V2)/2)).
  - "log of" / "ln(...)" — wrap in np.log.
  - "per capita" — divide by population.
  - "in percent form" vs "as a decimal" — scale by 100 and format the
    result string to match.

Worked example — geometric mean over a vector (tag-matched):
    e = next(e for e in prev if e.tag == "national_defense_expenditures:cy1940")
    vals = list(e.value.values())
    gm = float(np.exp(np.mean(np.log(vals))))
    result = f"{gm:.2f} millions of nominal dollars"

Worked example — column sum on a table (tag-matched):
    e = next(e for e in prev if e.tag == "internal_revenue_collections:fy1943")
    df = pd.DataFrame.from_dict(e.value, orient="index")
    total_1943 = float(df.loc["1943"].sum())
    result = f"{total_1943:,.0f}"

Output format — exactly one of:
  CODE\\n<python that sets `result` to a string>
  MISSING:<short reason a needed datum is absent>

Rules:
- If a value you need isn't in `prev` (e.g. only one period present, no FX
  rate, no CPI), output MISSING. Never fabricate.
- Prefer a printed total/summary row over re-aggregating its components
  when both are in `prev`.
- `result` is a Python str containing ONLY the requested answer — no
  prose, no "Answer:", no restatement of the question. Format numbers per
  the question (precision, commas, % suffix per "percent form" requests).
  For multi-part questions, emit only the ultimate quantity/identifier
  the question asks for.
- Every cell in a vector/table shares one unit — convert once over the
  whole payload (e.g. usd_thousands → usd_millions: divide by 1000).
- Foreign currency follows <iso3>_<scale> (jpy_billions, gbp_millions,
  cad_billions). When the question wants USD and an input is foreign,
  locate the matching fx_rate entry and apply it: value * rate if the
  rate is "USD per <iso>", value / rate if "<iso> per USD" — read the
  rate's description for direction.
- Available imports: numpy (np), pandas (pd), math, statsmodels.api (sm).
  `hp_filter(series, lamb=…)` is pre-defined in the sandbox; don't import it.
- Output ONLY the format above — no markdown fences around the whole response.
"""

_CODEGEN_INTERMEDIATE_SYSTEM = """\
You are an intermediate compute step. Your output feeds a downstream
aggregator — emit a raw value, not a pretty string.

Inputs:
- A natural-language sub-task (this step only, not the user's full question).
- `prev`: list[AnnotatedValue] available in the sandbox.
  Fields: .tag (snake_case key), .description, .value (scalar | vector dict
  | table dict-of-dict), .unit, .kind, .index_name (vector), .row_name/.col_name (table).

PREFER exact tag matching: `next(e for e in prev if e.tag == "<tag>")`.
Fall back to description substring matching only when no tag is present.

Output format — exactly one of:
  CODE\\n<python that sets `result` (and optionally metadata vars)>
  MISSING:<short reason a needed datum is absent>

Assign `result` to one of:
  (a) Python scalar (int / float / str)           → wrapped as kind=scalar
  (b) dict {index_label: scalar}                  → wrapped as kind=vector
  (c) dict-of-dict {row: {col: scalar}}           → wrapped as kind=table
  (d) list[AnnotatedValue] you construct explicitly → returned verbatim

Optional metadata vars (sensible defaults used when absent):
  result_unit, result_kind, result_description,
  result_index_name (vector), result_row_name / result_col_name (table).

Worked example — produce a 2-element dict for downstream pairwise math:
    a = next(e for e in prev if e.tag == "national_defense:cy1940").value
    b = next(e for e in prev if e.tag == "national_defense:cy1953").value
    result = {'1940': a, '1953': b}
    result_unit = 'usd_millions'
    result_kind = 'vector'
    result_index_name = 'year'
    result_description = 'national defense totals by year'

Rules:
- Do NOT format `result` as a pretty string — emit the raw numeric/dict.
- Prefer a printed total/summary row over re-aggregating its components
  when both are in `prev`.
- Every cell in a vector/table shares one unit — convert once
  (usd_thousands → usd_millions: divide by 1000) and set result_unit.
- Foreign currency follows <iso3>_<scale>. To answer in USD when the
  input is foreign, find the matching fx_rate entry and apply it: "USD
  per <iso>" → multiply, "<iso> per USD" → divide. Check the fx_rate
  description for direction.
- If `prev` is missing a value you need, output MISSING. Never fabricate.
- Available imports: numpy (np), pandas (pd), math, statsmodels.api (sm).
  No other imports. No prints. `hp_filter(series, lamb=…)` is pre-defined.
- Output ONLY the format above — no markdown fences around the whole response.
"""

_CRITIQUE_SYSTEM = """\
You review code that another agent wrote and the string it produced.
Decide ACCEPT or REVISE.

Inputs: the question, a summary of `prev` (descriptions + units + kinds),
the Python that ran, the produced `result` string, and a "Parsed
constraints" block (method, transforms, units_out, precision, answer_form,
period_type). Treat the parsed constraints as a checklist.

Reply on a single line:
  ACCEPT
  REVISE: <one short reason a re-run should address>

Numeric modifiers to scan for in the question: signed vs absolute,
normalized / midpoint-normalized, log_of, per_capita, year-over-year /
month-over-month, percent form vs decimal, weighted vs unweighted,
compound vs simple growth. Each modifier present in the question MUST
appear as an explicit operation in the code, not as a comment. If the
result's magnitude or sign disagrees with the natural reading of the
question under its modifiers, that is a STRONG REVISE signal.

When the question names a specific method, verify the code implements its
standard formula. Common gotchas:
  - expected_shortfall on a return/yield series is the SIGNED mean of the
    tail observations; expect a negative result in a loss context.
  - arc_elasticity = ((Q2-Q1)/((Q1+Q2)/2)) / ((P2-P1)/((P1+P2)/2));
    not point elasticity, not CAGR.
  - zipf: MLE is the default unless the question pins OLS log-rank vs
    log-size or regression of frequency on rank.
  - hazen_plotting_position = (i - 0.5) / n; not Weibull i/(n+1) or
    California i/n.
  - gini / theil / cv are distinct — use the one named.
  - pearson_correlation / partial_correlation / spearman are distinct.
  - h_spread / iqr = Q3 − Q1 using the percentile method named (default
    linear-interpolation / Tukey hinges).
  - cagr = (V_end / V_start)^(1/n) - 1, where n is the number of intervals
    (not years inclusive).
If a related-but-different formula was used, REVISE: "code uses <X> but
question asks for <named method>".

REVISE when the producer:
- shipped wrong / missing unit conversion;
- omitted or mis-applied a numeric modifier from the scan;
- used a formula that doesn't match the named statistical method;
- produced a result whose form contradicts the question (wanted
  "[a, b]" but shipped "1.0 2.0"; wanted percent but shipped 0.1234);
- selected wrong rows/columns from `prev` for the question's constraints;
- used a description substring that could match multiple entries —
  instruct the producer to switch to exact tag matching;
- wrapped the answer in narrative prose when a single value was asked
  for.

Do NOT REVISE on:
- cosmetic precision when the question doesn't pin precision;
- missing/extra trailing unit suffix when the magnitude is right;
- whitespace, capitalization, or punctuation nits.
"""


class CodegenFinalOperator(SkunkOperator):
    name: str = "compute.codegen.final"
    system: str = _CODEGEN_FINAL_SYSTEM


class CodegenIntermediateOperator(SkunkOperator):
    name: str = "compute.codegen.intermediate"
    system: str = _CODEGEN_INTERMEDIATE_SYSTEM


class CritiqueOperator(SkunkOperator):
    name: str = "compute.critique"
    system: str = _CRITIQUE_SYSTEM

# Match CODE / MISSING anywhere (after fence-stripping + light prose). Use re.search,
# not re.match, so leading commentary or whitespace doesn't break classification.
_CODE_RE = re.compile(r"\bCODE\s*\n(.*)\Z", re.DOTALL)
_MISSING_RE = re.compile(r"\bMISSING:\s*([^\n]*)", re.IGNORECASE)


def _summarize_vector(value: dict, expected_index_range: str = "") -> str:
    """Short, codegen-friendly summary of a vector payload. Shows index range + a sample of cells.
    When `expected_index_range` is set (e.g., '1969-01..1980-01') and the actual key span
    is narrower, the summary loudly flags the gap so compute can MISSING out of partial data."""
    if not isinstance(value, dict) or not value:
        return repr(value)
    keys = list(value.keys())
    n = len(keys)
    head = ", ".join(f"{k!r}: {value[k]!r}" for k in keys[:3])
    base = f"{{{head}}}" if n <= 3 else f"{{{head}, ...}}"
    actual_range = f"index range {keys[0]!r}..{keys[-1]!r}"

    # Gap detection: if the question-implied full range is set AND the actual span doesn't cover it.
    gap_note = ""
    if expected_index_range and ".." in expected_index_range:
        exp_first, _, exp_last = expected_index_range.partition("..")
        exp_first = exp_first.strip()
        exp_last = exp_last.strip()
        actual_first = str(keys[0])
        actual_last = str(keys[-1])
        # String comparison works for ISO-shaped keys (1942-03, FY1942, etc.).
        if exp_first < actual_first or exp_last > actual_last:
            gap_note = (
                f"; GAP: question asked for {exp_first!r}..{exp_last!r} "
                f"but data only covers {actual_first!r}..{actual_last!r} — likely MISSING"
            )

    if n <= 3:
        return f"{base} ({actual_range}{gap_note})"
    return f"{base} ({n} cells, {actual_range}{gap_note})"


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


def _render_annotated_value(prefix: str, idx: int, e: AnnotatedValue) -> list[str]:
    label = e.description if e.description else "(no description)"
    if e.kind == "scalar":
        shape_str = "kind=scalar"
        value_str = repr(e.value)
    elif e.kind == "vector":
        shape_str = f"kind=vector, index_name={e.index_name!r}"
        value_str = (_summarize_vector(e.value, e.expected_index_range)
                     if isinstance(e.value, dict) else repr(e.value))
    elif e.kind == "table":
        shape_str = f"kind=table, row_name={e.row_name!r}, col_name={e.col_name!r}"
        value_str = _summarize_table(e.value) if isinstance(e.value, dict) else repr(e.value)
    else:
        shape_str = f"kind={e.kind!r}"
        value_str = repr(e.value)
    tag_str = f"tag: {e.tag!r}  " if e.tag else ""
    return [
        f"{prefix}prev[{idx}]  {tag_str}description: {label!r}",
        f"{prefix}        value:       {value_str}  ({shape_str}, unit={e.unit!r})",
    ]


def _prev_desc(prev: list[AnnotatedValue]) -> str:
    lines = [f"prev ({len(prev)} entries)"]
    for i, e in enumerate(prev):
        lines.extend(_render_annotated_value("  ", i, e))
    return "\n".join(lines)


def _constraints_block(
    ctx: HarnessContext,
    method: str | None = None,
    transforms: list[str] | None = None,
) -> str:
    """Render constraint bullets for compute prompts. Plan-level fields come
    from ctx.plan; per-compute fields are passed as kwargs by the caller."""
    lines: list[str] = []
    plan = ctx.plan
    if plan is not None:
        if plan.units_out:
            lines.append(f"- units_out: {plan.units_out}")
        if plan.precision is not None:
            lines.append(f"- precision: {plan.precision} decimal places")
        if plan.answer_form != "scalar":
            lines.append(f"- answer_form: {plan.answer_form}")
    if method:
        lines.append(f"- method: {method}")
    if transforms:
        lines.append(f"- transforms (MUST apply each): {', '.join(transforms)}")
    if not lines:
        return ""
    return "Parsed constraints (apply each):\n" + "\n".join(lines) + "\n\n"


def _build_user(question: str, prev_desc: str, priors: list[str], spec_block: str = "") -> str:
    msg = (
        f"Question:\n{question}\n\n"
        f"{spec_block}"
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
    *, method: str | None = None, transforms: list[str] | None = None,
) -> tuple[bool, str]:
    """Same-actor review of (code, result) against the question.

    Returns (accept, reason). Conservative when ambiguous: malformed reply or both
    tokens present → REVISE with a diagnostic reason.
    """
    user = (
        f"Question:\n{question}\n\n"
        f"{_constraints_block(ctx, method=method, transforms=transforms)}"
        f"prev =\n{prev_desc}\n\n"
        f"Code that ran:\n```python\n{code}\n```\n\n"
        f"Produced result:\n{result_text}\n\n"
        f"Reply on a single line: ACCEPT, or REVISE: <reason>."
    )
    resp = ctx.llm_client.call(CritiqueOperator().build_system(ctx), user, ctx=ctx)
    raw = resp.text
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
    *,
    system_prompt: str,
    question: str,
    method: str | None = None,
    transforms: list[str] | None = None,
) -> tuple[dict | None, str | None, list[str]]:
    """One logical attempt: codegen → parse → exec, with `retry_budget` retries
    for transient exec/parse failures (a malformed response or a code exception).

    MISSING: from codegen propagates as MissingData. Returns
    (env, code, accumulated_priors) on success — where `env` is the post-exec
    sandbox dict (so callers can read `result`, `result_unit`, etc.) — or
    (None, None, accumulated_priors) when the retry budget is exhausted.
    """
    priors = list(priors)
    spec_block = _constraints_block(ctx, method=method, transforms=transforms)
    # Total tries = 1 initial + retry_budget retries.
    for try_idx in range(retry_budget + 1):
        resp = ctx.llm_client.call(
            system_prompt,
            _build_user(question, prev_desc, priors, spec_block=spec_block),
            thinking_budget=-1, ctx=ctx,
        )
        raw = resp.text
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
            env, _ = exec_python_with_env(code, {"prev": prev})
        except Exception as e:
            priors.append(f"Code:\n```python\n{code}\n```\nException: {e}")
            ctx.emit("compute", f"codegen try {try_idx + 1} exec failed", error=str(e))
            continue

        ctx.emit("compute", f"codegen try {try_idx + 1} produced result",
                 text=str(env.get("result"))[:300])
        return env, code, priors

    return None, None, priors


def _infer_kind(value: object) -> str:
    if isinstance(value, dict):
        if value and isinstance(next(iter(value.values()), None), dict):
            return "table"
        return "vector"
    return "scalar"


_TASK_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug_from_task(task: str) -> str:
    s = _TASK_SLUG_RE.sub("_", (task or "intermediate_result").lower()).strip("_")
    return (s[:48] or "intermediate_result")


def _wrap_intermediate_result(task: str, env: dict) -> list[AnnotatedValue]:
    """Turn the sandbox env produced by an intermediate compute into list[AnnotatedValue]."""
    result_value = env.get("result")
    if isinstance(result_value, list) and all(isinstance(x, AnnotatedValue) for x in result_value):
        return result_value
    kind = env.get("result_kind") or _infer_kind(result_value)
    description = env.get("result_description") or task or _slug_from_task(task)
    return [AnnotatedValue(
        description=str(description),
        value=result_value,
        unit=str(env.get("result_unit", "")),
        kind=str(kind),
        index_name=env.get("result_index_name"),
        row_name=env.get("result_row_name"),
        col_name=env.get("result_col_name"),
    )]


def _run_final(
    question: str, prev: object, ctx: HarnessContext,
    *, method: str | None = None, transforms: list[str] | None = None,
) -> FormattedString:
    """Terminator path — produces a single answer string via codegen + self-critique."""
    prev_desc = _prev_desc(prev)
    ctx.emit("compute", "starting", mode="final", question=question,
             method=method, transforms=transforms,
             prev_summary=prev_desc[:500])

    codegen_system = CodegenFinalOperator().build_system(ctx)
    a1_env, a1_code, a1_priors = _try_codegen_and_exec(
        ctx, prev, prev_desc, priors=[],
        retry_budget=ctx.config.compute_max_attempts - 1,
        system_prompt=codegen_system, question=question,
        method=method, transforms=transforms,
    )
    if a1_env is None:
        raise StepFailed(
            "compute",
            f"could not produce a result on attempt 1: priors={a1_priors}",
        )
    a1_result = str(a1_env["result"])

    accept, reason = _self_critique(question, prev_desc, a1_code, a1_result, ctx,
                                    method=method, transforms=transforms)
    if accept:
        ctx.emit("compute", "attempt 1 self-critique ACCEPT", text=a1_result)
        return FormattedString(text=a1_result)
    ctx.emit("compute", "attempt 1 self-critique REVISE", reason=reason)

    hint = (
        f"Produced result {a1_result!r}. Self-critique flagged: {reason}\n"
        f"Code was:\n```python\n{a1_code}\n```"
    )
    try:
        a2_env, _, _ = _try_codegen_and_exec(
            ctx, prev, prev_desc, priors=[hint], retry_budget=0,
            system_prompt=codegen_system, question=question,
            method=method, transforms=transforms,
        )
    except MissingData as e:
        ctx.emit("compute", "attempt 2 MISSING; falling back to attempt 1",
                 reason=e.reason, fallback_text=a1_result)
        return FormattedString(text=a1_result)
    if a2_env is None:
        ctx.emit(
            "compute",
            "attempt 2 produced no result; falling back to attempt 1",
            fallback_text=a1_result,
        )
        return FormattedString(text=a1_result)
    a2_result = str(a2_env["result"])
    ctx.emit("compute", "attempt 2 returned (no re-critique)", text=a2_result)
    return FormattedString(text=a2_result)


def _run_intermediate(
    task: str, prev: object, ctx: HarnessContext,
    *, method: str | None = None, transforms: list[str] | None = None,
) -> list[AnnotatedValue]:
    """Intermediate path — produces a list[AnnotatedValue] for the downstream aggregator."""
    prev_desc = _prev_desc(prev)
    ctx.emit("compute", "starting", mode="intermediate", task=task,
             method=method, transforms=transforms,
             prev_summary=prev_desc[:500])

    env, _, priors = _try_codegen_and_exec(
        ctx, prev, prev_desc, priors=[],
        retry_budget=ctx.config.compute_max_attempts - 1,
        system_prompt=CodegenIntermediateOperator().build_system(ctx), question=task,
        method=method, transforms=transforms,
    )
    if env is None:
        raise StepFailed(
            "compute",
            f"intermediate could not produce a result: priors={priors}",
        )
    wrapped = _wrap_intermediate_result(task, env)
    ctx.emit("compute", "intermediate produced entries",
             n=len(wrapped), descriptions=[e.description for e in wrapped])
    return wrapped


def run(op: OpNode, prev: object, ctx: HarnessContext) -> FormattedString | list[AnnotatedValue]:
    """Dispatch on op.args["final"]. Default is final=True (back-compat terminator)."""
    task = str(op.args.get("task") or "")
    is_final = bool(op.args.get("final", True))
    method = op.args.get("method") or None
    transforms = list(op.args.get("transforms") or []) or None
    if is_final:
        question = task or ctx.question
        return _run_final(question, prev, ctx, method=method, transforms=transforms)
    if not task:
        raise StepFailed("compute", "intermediate compute requires a non-empty task")
    return _run_intermediate(task, prev, ctx, method=method, transforms=transforms)
