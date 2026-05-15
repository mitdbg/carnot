"""compute subagent — two modes, switched by op.args["final"]:

  final=True  (default for back-compat): chain terminator. Reads ctx.question
              (or op.args["task"] if planner-supplied) and prev. Codegens
              Python that assigns `result` to the answer string, self-critiques,
              and ships FormattedString. This is the historical behavior.

  final=False: intermediate compute. Reads op.args["task"] and prev. Codegens
              Python that assigns `result` to a raw value (scalar / dict /
              dict-of-dict / list[AnnotatedValue]) plus optional metadata vars
              (`result_unit`, `result_kind`, `result_name`, `result_dims`,
              `result_index_name`, `result_row_name`, `result_col_name`).
              No self-critique. Returns list[AnnotatedValue] for the
              downstream aggregator to consume.

Failure modes (StepFailed):
- Attempt 1 produced no result across all transient retries
- Attempt 2 (final mode only) produced no result AND attempt 1 had no result either
- Codegen reported "MISSING:" on attempt 1 — extracted values insufficient
"""

from __future__ import annotations

import re

from skunk.common import HarnessContext
from skunk.dsl import AnnotatedValue, FormattedString, OpNode
from skunk.subagents.base import (
    MissingData,
    StepFailed,
    exec_python_with_env,
    strip_code_fences,
)

_CODEGEN_SYSTEM = """\
You are writing code for a financial data processing pipeline.

You receive:
- The user's original question (free text).
- `prev`: extracted data, available in the sandbox. Always list[AnnotatedValue].
    AnnotatedValue fields:
          .tag         — short snake_case selection key (e.g. "national_defense_expenditures:cy1940").
                          Two entries describing the same series + period share the same tag.
          .description — natural-language label uniquely identifying this datum
                          (e.g. "Total US national defense expenditures, monthly, CY1940")
          .value       — the payload (scalar, vector dict, or table dict-of-dict)
          .unit        — semantic unit (usd_millions, pct, count, year, text, fx_rate, ...)
          .kind        — "scalar" | "vector" | "table"
          .index_name  — vector only: name of the varying dim (e.g. "month")
          .row_name    — table only: name of the row dim
          .col_name    — table only: name of the column dim

Selecting from prev:
  PREFER exact tag matching: `next(e for e in prev if e.tag == "national_defense_expenditures:cy1940")`.
  When multiple entries match a substring, the first match wins — which is fragile.
  Tags are the unambiguous handle; use them when available. Fall back to description
  substring matching only when no tag is present (e.tag == "") or when you need to
  scan multiple entries.

Payload shapes by kind:
  scalar:  entry.value is the number or string itself.
  vector:  entry.value is an insertion-ordered dict {index_label: scalar}.
           Use list(entry.value.values()) for whole-series ops, or .items() to filter
           by index label (e.g. a date range).
  table:   entry.value is a 2-level dict {row_label: {col_label: scalar}}.
           Cast with pd.DataFrame.from_dict(entry.value, orient="index") for normal
           table operations.

Pay special attention to the question's explicit constraints:
  - year:  e.g. "1942-1948", fiscal year (FY) vs. calendar year (CY)
  - series: e.g. "national defense expenditures"
  - type of computation requested: e.g., geometric mean vs. arithmetic mean

You will also see a "Parsed constraints" block in the user message — a
pre-parsed structured spec of the question. Treat it as ground truth: every
transform listed there MUST appear as an explicit operation in your code.
If method is set, implement that exact method's standard formula. If
units_out is set, format/convert the answer to that unit. If precision is
set, round the final answer to that many decimal places.

NUMERIC TRANSFORMS — every transform present in the question MUST appear as
an explicit operation in the code, NOT just acknowledged in a comment.
Naming the transform without implementing it will be REVISEd downstream.
Common ones to watch for:
  - "normalized", "mid-point normalized" — apply the normalization (e.g.
    midpoint: (V2-V1) / ((V1+V2)/2))
  - "signed", "absolute" — preserve or strip the sign accordingly; never
    silently `abs()` a value the question wants signed
  - "log of", "ln(...)" — wrap in np.log; do not skip
  - "ratio of A to B" — divide A by B; do not subtract
  - "per capita" — divide by population
  - "year-over-year", "month-over-month" — pct change between consecutive
    periods, not raw diff
  - "in percent form" vs "as a decimal" — multiply/divide by 100 accordingly
    and format the result string to match
  - "rounded to N decimal places" — round explicitly
  - "in millions/billions of USD" — convert the .unit value to that scale
  - "geometric mean" vs "arithmetic mean" — use np.exp(np.mean(np.log(...)))
    vs np.mean
  - "weighted" vs "unweighted" — apply weights or don't, per the question
  - "compound" vs "simple" growth rate — use CAGR formula vs naive ratio

Worked example — geometric mean over a vector (tag-matched):
    e = next(e for e in prev if e.tag == "national_defense_expenditures:cy1940")
    vals = list(e.value.values())
    gm = float(np.exp(np.mean(np.log(vals))))
    result = f"{gm:.2f} millions of nominal dollars"

Worked example — date-range filter on a vector:
    series = next(e for e in prev
                  if "budget expenditures" in e.description.lower()).value
    selected = [v for k, v in series.items() if "1942-03" <= k <= "1948-10"]
    gm = float(np.exp(np.mean(np.log(selected))))
    result = f"{gm:.2f}"

Worked example — column sum on a table (tag-matched):
    e = next(e for e in prev if e.tag == "internal_revenue_collections:fy1943")
    df = pd.DataFrame.from_dict(e.value, orient="index")
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
- Prefer a printed value already in `prev` over re-deriving it from components
  also in `prev`. If the page prints both a total/summary row and its component
  rows, use the total — don't re-aggregate the components to compute it yourself.
- `result` MUST be a Python str containing ONLY the answer the question asks for —
  no surrounding prose, no restatement of the question, no labels like "Answer:"
  or "The bureau is...". For numbers, format per the question's instructions
  (precision, with/without commas, with/without unit symbols). For lists, use the
  bracket form the question requests. For percent answers, append "%" only if the
  question asks for percent form. If the question contains multiple sub-questions
  (e.g. "name the bureau AND compute X"), only emit the final answer the question
  ultimately asks for — typically the last quantity/identifier requested.
- Convert units yourself when needed (e.g. usd_thousands → usd_millions: divide by 1000).
  Every cell in a vector or table shares one unit — convert once over the whole payload.
- Units may follow the pattern <iso3>_<scale> for foreign currency (e.g. jpy_billions,
  gbp_millions, cad_billions). When the question asks for an answer in USD but an input
  is in foreign currency, locate the matching fx_rate entry in `prev` and apply it —
  e.g. usd = value * rate when the rate's units are "USD per <iso>", or value / rate when
  "<iso> per USD". Check the description of the fx_rate entry to determine direction.
- Available imports: numpy (np), pandas (pd), math, statsmodels.api (sm). `hp_filter(series, lamb=…)` is also pre-defined in the sandbox (pure-numpy HP filter). Do not import it.
- Output ONLY the format above — no markdown fences around the whole response.
"""

_CODEGEN_SYSTEM_INTERMEDIATE = """\
You are an intermediate compute step in a financial QA pipeline. Your output
feeds a downstream aggregator compute — emit raw values, not pretty strings.

You receive:
- A natural-language sub-task (the question for THIS step, not the user's
  original question).
- `prev`: extracted bulletin data, available in the sandbox. Always list[AnnotatedValue].
    AnnotatedValue fields:
          .tag         — short snake_case selection key (e.g. "national_defense_expenditures:cy1940")
          .description — natural-language label uniquely identifying this datum
          .value       — the payload (scalar, vector dict, or table dict-of-dict)
          .unit        — semantic unit (usd_millions, pct, count, year, text, fx_rate, ...)
          .kind        — "scalar" | "vector" | "table"
          .index_name  — vector only: name of the varying dim (e.g. "month")
          .row_name    — table only: name of the row dim
          .col_name    — table only: name of the column dim

Payload shapes by kind:
  scalar:  entry.value is the number or string itself.
  vector:  entry.value is an insertion-ordered dict {index_label: scalar}.
  table:   entry.value is a 2-level dict {row_label: {col_label: scalar}}.

Working with entries:
- PREFER exact tag matching: `next(e for e in prev if e.tag == "<tag>")`.
- Fall back to description substring matching only when no tag is present.
- Every entry's unit is in .unit — trust that, never guess.

Your job: produce ONE raw value (or a small set) that answers the sub-task and
will be wrapped as a downstream AnnotatedValue.

Output format — exactly one of:
  CODE\\n<python code that sets `result` (and optionally metadata vars)>
  MISSING:<short explanation of which datum is missing>

Rules:
- Assign `result` to one of:
    (a) a Python scalar (int / float / str)              → wrapped as kind=scalar
    (b) a Python dict {index_label: scalar}              → wrapped as kind=vector
    (c) a Python dict-of-dict {row: {col: scalar}}       → wrapped as kind=table
    (d) a list[AnnotatedValue] you construct explicitly  → returned verbatim
- Optionally set these metadata vars to refine the wrapping (sensible defaults
  used when absent):
    result_unit         — str, e.g. "usd_millions" or "pct"
    result_kind         — "scalar" | "vector" | "table" (only if inference is wrong)
    result_description  — natural-language label for this output
    result_index_name   — vector only: name of the varying dim
    result_row_name     — table only
    result_col_name     — table only
- Do NOT format `result` as a pretty string — emit the raw numeric/dict.
- Prefer a printed value already in `prev` over re-deriving it from components
  also in `prev`. If both a total/summary row and its component rows are present,
  use the total — don't re-aggregate components to compute it yourself.
- Convert units yourself when needed (e.g. usd_thousands → usd_millions: divide by 1000)
  and set result_unit accordingly.
- Units may follow <iso3>_<scale> for foreign currency (e.g. jpy_billions, gbp_millions).
  When the sub-task requires a USD answer but an input is in foreign currency, find the
  matching fx_rate entry in `prev` and apply it (read the fx_rate's description to
  determine direction: "USD per <iso>" → multiply, "<iso> per USD" → divide).
- If `prev` is missing a value you need, output MISSING:<reason>. Do NOT fabricate values.
- Available imports: numpy (np), pandas (pd), math, statsmodels.api (sm). No other imports. No prints.
- `hp_filter(series, lamb=…)` is also pre-defined in the sandbox (pure-numpy HP filter). Do not import it.
- Output ONLY the format above — no markdown fences around the whole response.

Worked example — find the year where a series is minimized:
    series = next(e for e in prev
                  if "yield spread" in e.description.lower()).value
    min_year, min_val = min(series.items(), key=lambda kv: kv[1])
    result = min_year
    result_unit = 'year'
    result_kind = 'scalar'
    result_description = 'year of minimum yield spread'

Worked example — produce a 2-element dict for downstream pairwise math:
    a = next(e for e in prev if "cy1940 total" in e.description.lower()).value
    b = next(e for e in prev if "cy1953 total" in e.description.lower()).value
    result = {'1940': a, '1953': b}
    result_unit = 'usd_millions'
    result_kind = 'vector'
    result_index_name = 'year'
    result_description = 'national defense totals by year'
"""


_CRITIQUE_SYSTEM = """\
You are reviewing code that another coding agent wrote and the
string it produced. Decide whether to ship the result as-is or revise.

You receive the question, a summary of `prev` (the list[AnnotatedValue] input — including each entry's description,
unit, and kind), the python code that ran, and the produced `result` string.

You will also see a "Parsed constraints" block — a pre-parsed structured spec
of the question (units_out, precision, answer_form, method, transforms,
period_type). Treat it as ground truth and use it as your checklist: for
each transform in the spec, verify the code applies it; for method, verify
the formula matches the standard form; for units_out, verify the result is
in that unit; for precision, verify rounding; for answer_form, verify the
result's shape.

Reply on a single line:
  ACCEPT
  REVISE: <one short reason a re-run should address>

Before deciding, scan the question for NUMERIC MODIFIERS. Common ones:
  "normalized", "mid-point normalized", "signed" (vs "absolute"), "log of",
  "ln(...)", "ratio of A to B", "per capita", "year-over-year",
  "month-over-month", "in percent form" vs "as a decimal", "rounded to N
  decimal places", "in millions/billions of USD", "geometric mean" vs
  "arithmetic mean", "weighted" vs "unweighted", "compound" vs "simple"
  growth rate.
For each modifier present in the question, verify the code applies it
correctly as an explicit operation (not just a comment). If any modifier
is omitted or misapplied, REVISE — even when the result "looks plausible"
in shape. Magnitude/sign sanity check: if the result's order of magnitude
or sign disagrees with the natural reading of the question under its
modifiers, that is a STRONG REVISE signal.

When the question NAMES A SPECIFIC METHOD, verify the code implements the
standard formula for that method. Specific gotchas:
  - "Expected shortfall" / ES on a return-or-yield series: ES is the
    *signed* mean of the tail-loss observations. If yields are positive
    on average but the question implies a "shortfall" / loss context,
    expect a NEGATIVE answer. Code that returns the mean of the lowest
    values without sign-handling is wrong.
  - "Arc elasticity": standard form is
    ((Q2-Q1)/((Q1+Q2)/2)) / ((P2-P1)/((P1+P2)/2)).
    Do NOT substitute point elasticity, CAGR, or growth-rate-as-elasticity.
  - "Zipf exponent": if the question pins a method (OLS log-rank vs log-size,
    MLE, regression of frequency on rank), use exactly that method. If the
    question doesn't pin one, MLE is the typical default for power-law fits.
  - "Hazen plotting position": (i - 0.5) / n. NOT Weibull i/(n+1) and NOT
    California i/n.
  - "Gini coefficient" vs "Theil index" vs "coefficient of variation": these
    are distinct; use the one named.
  - "Pearson correlation" vs "partial correlation" vs "Spearman": distinct.
  - "H-spread" / IQR: Q3 − Q1 using the percentile method named (default
    "linear interpolation" / Tukey hinges if unspecified).
  - "CAGR": (V_end/V_start)^(1/n) - 1, where n is the number of intervals
    (NOT years inclusive).
If a related-but-different formula was used, REVISE with: "code uses
<short description> but question asks for <named method>".

REVISE when the producer:
  - applied the wrong unit conversion (e.g. shipped usd_thousands while the
    question asked for usd_millions, or never converted),
  - omitted or misapplied a numeric modifier from the question (see scan
    above) — e.g. the question said "midpoint normalized" but the code
    shipped a raw difference; said "signed" but the code wrapped abs(...);
    said "log of" but the code skipped the log; said "per capita" but the
    code didn't divide by population,
  - used a formula that doesn't match a named statistical method (see
    method-name list above),
  - produced a result whose form clearly contradicts the question (asked
    for "[a, b]" bracketed list, shipped "1.0 2.0"; asked for percent
    form, shipped a decimal like "0.1234"),
  - selected the wrong rows/columns from `prev` given the question's
    explicit constraints (wrong year, wrong series, wrong dim filter),
  - used a description substring or tag that would match MULTIPLE entries
    in `prev` — `next(e for e in prev if "X" in e.description)` is fragile
    when several entries share that substring; REVISE and instruct the
    producer to use exact tag matching (e.tag == "<tag>") to disambiguate,
  - wrapped the answer in narrative prose ("The bureau is X and the
    average is Y") when the question asks for a single value — REVISE
    and instruct the producer to return ONLY the requested value.

Do NOT REVISE on:
  - cosmetic precision when the question doesn't pin precision,
  - presence/absence of a trailing unit suffix when the magnitude is right
    and the question doesn't explicitly demand the suffix,
  - whitespace, capitalization, or punctuation nits.
"""

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
    resp = ctx.llm_client.call(_CRITIQUE_SYSTEM, user, ctx=ctx)
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

    a1_env, a1_code, a1_priors = _try_codegen_and_exec(
        ctx, prev, prev_desc, priors=[],
        retry_budget=ctx.config.compute_max_attempts - 1,
        system_prompt=_CODEGEN_SYSTEM, question=question,
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
            system_prompt=_CODEGEN_SYSTEM, question=question,
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
        system_prompt=_CODEGEN_SYSTEM_INTERMEDIATE, question=task,
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
