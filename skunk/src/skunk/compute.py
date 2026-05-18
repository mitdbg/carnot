"""compute operator — codegen + execution over the extracted-value env.

Takes the user's question plus `prev: list[AnnotatedValue]`, generates Python
that assigns `result` to the formatted answer string, execs it in-process,
then self-critiques via `CritiqueExecutor` and re-runs once on REVISE.
Returns the answer as a plain `str`.

Two LLM contracts:
  - codegen: emits EITHER a fenced ```python``` block (success) OR a bare
    JSON object `{"missing": [...], "description": "..."}` (insufficient
    data). Dispatch is on the first non-whitespace character. The JSON
    form surfaces as `MissingData(description, missing=...)` out of `run()`.
  - critique: emits a single bare JSON object validated onto `CritiqueResult`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from skunk.common import strip_code_fence
from skunk.errors import MissingData, StepFailed
from skunk.executor import SkunkExecutor
from skunk.models import AnnotatedValue, HarnessContext
from skunk.plan import Plan
from skunk.pyexec import exec_python_with_env


class MissingResult(BaseModel):
    """Structured codegen signal: 'I cannot compute; here's what's absent.'

    `missing` is a list of short tags/keys naming the data the model would
    need (free-form; used only for the trace). `description` is a one-line
    human-readable explanation.
    """
    model_config = ConfigDict(frozen=True)
    missing: list[str] = []
    description: str


class CritiqueResult(BaseModel):
    """Parsed critique LLM output. `reason` is required when `verdict='revise'`."""
    model_config = ConfigDict(frozen=True)
    verdict: Literal["accept", "revise"]
    reason: str = ""

    @model_validator(mode="after")
    def _revise_needs_reason(self) -> CritiqueResult:
        if self.verdict == "revise" and not self.reason.strip():
            raise ValueError("verdict=revise requires non-empty 'reason'")
        return self


def _summarize_vector(value: dict, expected_index_range: str = "") -> str:
    """Short, codegen-friendly summary of a vector payload. Shows index range + a sample of cells.
    When `expected_index_range` is set (e.g., '1969-01..1980-01') and the actual key span
    is narrower, the summary loudly flags the gap so compute can MISSING out of partial data."""
    if not value:
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
    if not value:
        return repr(value)
    rows = list(value.keys())
    first_row = value[rows[0]]
    cols = list(first_row.keys())
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
        value_str = _summarize_vector(e.value, e.expected_index_range)
    else:  # table — VALUE_KIND_VOCAB is exhaustive; AnnotatedValue.from_dict guarantees this
        shape_str = f"kind=table, row_name={e.row_name!r}, col_name={e.col_name!r}"
        value_str = _summarize_table(e.value)
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
    computation: str = "",
    qualifiers: list[str] | None = None,
    *,
    units_out: str = "",
    precision: int | None = None,
    answer_form: str = "scalar",
) -> str:
    """Render constraint bullets for compute prompts. Plan-level presentation
    fields (`units_out` / `precision` / `answer_form`) and the per-compute
    `computation` / `qualifiers` all arrive as kwargs to `run()`."""
    lines: list[str] = []
    if units_out:
        lines.append(f"- units_out: {units_out}")
    if precision is not None:
        lines.append(f"- precision: {precision} decimal places")
    if answer_form != "scalar":
        lines.append(f"- answer_form: {answer_form}")
    if computation:
        lines.append(f"- computation: {computation}")
    for h in qualifiers or []:
        lines.append(f"- qualifier (MUST apply): {h}")
    if not lines:
        return ""
    return "Parsed constraints (apply each):\n" + "\n".join(lines) + "\n\n"


def _build_user(question: str, prev_desc: str, priors: list[str], spec_block: str = "") -> str:
    msg = (
        f"Question:\n{question}\n\n"
        f"{spec_block}"
        f"prev =\n{prev_desc}\n\n"
        f"Produce a fenced ```python``` block OR a bare missing-data JSON object."
    )
    if priors:
        msg += "\n\nPrior attempts (oldest first):\n"
        for i, p in enumerate(priors, 1):
            msg += f"\n[Attempt {i}]\n{p}\n"
        msg += "\nDo not repeat any of the above mistakes."
    return msg


class CodegenExecutor(SkunkExecutor):
    name: str = "compute.codegen"
    system_prompt: str = """\
You write Python that produces the final answer string. Given a question,
a list of extracted values `prev`, and a parsed-constraints block, emit
either runnable Python or a structured missing-data signal.

## Inputs

- The user's question.
- A "Parsed constraints" block in the user message — the planner's
  `computation` description plus presentation fields (units_out,
  precision, answer_form). Treat as ground truth: implement what the
  computation says, convert to units_out if set, round to precision if
  set, format to answer_form.
- `prev`: list[AnnotatedValue] available in the exec environment.

## AnnotatedValue API

  .tag         snake_case key shaped <series>:<period>
  .description natural-language label
  .value       scalar | vector dict {index: scalar}
               | table dict-of-dict {row: {col: scalar}}
  .unit        snake_case token (varies by corpus)
  .kind        "scalar" | "vector" | "table"
  .index_name  vector only
  .row_name / .col_name   table only

Selecting from prev — PREFER exact tag matching:
  next(e for e in prev if e.tag == "<tag>")
Substring matching on description is fragile (first match wins); fall
back to it only when no tag is present (e.tag == "").

Payload access:
  vector: list(e.value.values()) or .items() to filter by index.
  table:  pd.DataFrame.from_dict(e.value, orient="index").

## Output format

Emit exactly one of the two forms below and nothing else. No prose, no
commentary, no second block.

(a) On success — a single fenced ```python``` block. Assign the final
    answer string to `result`. The string must contain ONLY the
    requested answer — no prose, no "Answer:", no question restatement.

(b) On insufficient data — a single bare JSON object (NO fence):

      {"missing": [<short tag/key strings>],
       "description": "<one-line explanation>"}

    `missing` names the data you would need; `description` explains
    why the available inputs are insufficient. Never fabricate values
    to avoid this path — if a required value isn't in `prev`, emit (b).

Never emit both forms. Never wrap (b) in a fence. Never add prose
before, between, or after the output.

## Principles

- Every modifier in the question (or echoed in `computation`) MUST
  appear as an explicit operation in the code, not as a comment. The
  critic REVISEs on missing or mis-applied modifiers even when the
  magnitude looks plausible.
- Prefer a printed total/summary row over re-aggregating its
  components when both are in `prev`.
- Every cell in a vector/table shares one unit — apply any unit
  conversion once over the whole payload, never cell-by-cell.
- For multi-part questions, `result` contains only the ultimate
  quantity / identifier the question asks for.
- Available imports: numpy (np), pandas (pd), math, statsmodels.api (sm).
"""

    def try_once(
        self,
        ctx: HarnessContext,
        question: str,
        prev: list[AnnotatedValue],
        prev_desc: str,
        priors: list[str],
        spec_block: str,
        try_idx: int,
    ) -> tuple[dict | None, str | None, list[str]]:
        """One codegen → parse → exec attempt. Returns
        `(env, code, priors)` on success or `(None, None, priors)` on a
        transient parse/exec failure (caller may retry). `MissingData`
        propagates when the LLM reports kind="missing"."""
        priors = list(priors)
        system = self.assemble_system_prompt(ctx)
        resp = ctx.llm_client.call(
            system,
            _build_user(question, prev_desc, priors, spec_block=spec_block),
            thinking_budget=-1, ctx=ctx,
        )
        raw = resp.text
        ctx.emit("compute", f"codegen try {try_idx + 1} response", raw=raw)

        s = raw.strip()
        if s.startswith("{"):
            # Missing-data signal: bare JSON object, no fence.
            try:
                missing = MissingResult.model_validate_json(s)
            except ValidationError as e:
                priors.append(
                    f"Your previous attempt emitted malformed missing-data JSON. "
                    f"Raw: {raw[:300]}\nError: {e}\n"
                    f'Output ONLY a fenced ```python``` block OR a bare JSON '
                    f'{{"missing": [...], "description": "..."}} object.'
                )
                ctx.emit("compute", f"codegen try {try_idx + 1} malformed missing",
                         prior=priors[-1])
                return None, None, priors
            ctx.emit("compute", f"codegen try {try_idx + 1} reported MISSING",
                     missing=missing.missing, description=missing.description)
            raise MissingData(missing.description, missing=missing.missing)

        code = strip_code_fence(s).strip()
        if not code:
            priors.append(
                f"Your previous attempt produced empty or un-fenced code. "
                f"Raw: {raw[:300]}\n"
                f'Output ONLY a fenced ```python``` block OR a bare JSON '
                f'{{"missing": [...], "description": "..."}} object.'
            )
            ctx.emit("compute", f"codegen try {try_idx + 1} no code", prior=priors[-1])
            return None, None, priors
        ctx.emit("compute", f"codegen try {try_idx + 1} code", code=code)

        try:
            env, _ = exec_python_with_env(code, {"prev": prev})
        except Exception as e:
            priors.append(f"Code:\n```python\n{code}\n```\nException: {e}")
            ctx.emit("compute", f"codegen try {try_idx + 1} exec failed", error=str(e))
            return None, None, priors

        ctx.emit("compute", f"codegen try {try_idx + 1} produced result",
                 text=str(env.get("result")))
        return env, code, priors


class CritiqueExecutor(SkunkExecutor):
    name: str = "compute.critique"
    system_prompt: str = """\
You review code that another agent wrote and the string it produced.
Decide accept or revise.

## Inputs

The question, a summary of `prev` (descriptions + units + kinds), the
Python that ran, the produced `result` string, and a "Parsed
constraints" block (the planner's `computation` description plus
units_out / precision / answer_form). Treat the parsed constraints as
a checklist.

## Output format

A single bare JSON object. No fences, no prose. Exactly one of:

  {"verdict": "accept"}
  {"verdict": "revise",
   "reason": "<one short reason a re-run should address>"}

## What to check

- Numeric modifiers: every modifier named in the question (signed vs
  absolute, normalized / midpoint-normalized, log, per-capita, percent
  vs decimal, weighted vs unweighted, compound vs simple growth,
  year-over-year, etc.) MUST appear as an explicit operation in the
  code, not a comment. If the result's magnitude or sign disagrees
  with the natural reading of the question under its modifiers, that
  is a STRONG REVISE signal.
- Named operations: when the question names a specific statistical
  operation, verify the code implements its STANDARD formula. If a
  related-but-different formula was used, set
  reason="code uses <X> but question asks for <named operation>".
- Selection: code uses exact tag matching where possible. Flag
  fragile description-substring lookups that could match multiple
  entries.
- Unit handling: any conversion required by units_out was applied
  correctly over the whole payload.
- Result form: matches answer_form / precision; no narrative prose
  around a single requested value.

## REVISE when the producer

- shipped wrong / missing unit conversion;
- omitted or mis-applied a numeric modifier from the question;
- used a formula that doesn't match the named statistical operation;
- produced a result whose form contradicts the question (wanted
  "[a, b]" but shipped "1.0 2.0"; wanted percent but shipped 0.1234);
- selected wrong rows / columns from `prev`;
- used a fragile description substring where a tag was available;
- wrapped the answer in narrative prose when a single value was
  asked for.

## Do NOT REVISE on

- cosmetic precision when the question doesn't pin precision;
- missing / extra trailing unit suffix when the magnitude is right;
- whitespace, capitalization, or punctuation nits.
"""

    def critique(
        self,
        ctx: HarnessContext,
        question: str,
        prev_desc: str,
        code: str,
        result_text: str,
        spec_block: str,
    ) -> tuple[bool, str]:
        """Same-actor review of (code, result) against the question.

        Returns (accept, reason). Conservative when ambiguous: a malformed
        reply is treated as REVISE with a diagnostic reason — never silently
        accept.
        """
        user = (
            f"Question:\n{question}\n\n"
            f"{spec_block}"
            f"prev =\n{prev_desc}\n\n"
            f"Code that ran:\n```python\n{code}\n```\n\n"
            f"Produced result:\n{result_text}\n\n"
            f"Output a single bare JSON object with your verdict."
        )
        resp = ctx.llm_client.call(self.assemble_system_prompt(ctx), user, ctx=ctx)
        raw = resp.text
        ctx.emit("compute", "self-critique response", raw=raw)

        try:
            verdict = CritiqueResult.model_validate_json(raw.strip())
        except ValidationError as e:
            return False, f"self-critique produced malformed reply: {e}"
        if verdict.verdict == "accept":
            return True, ""
        return False, verdict.reason


class ComputeOperator:
    """Operator-level orchestrator for the compute step.

    Owns one `CodegenExecutor` + one `CritiqueExecutor` instance and drives
    the codegen → exec → critique → optional revise loop. Mirrors the
    `PlannerExecutor` shape at the operator boundary: one class, one
    public `run()` method, no module-level state.
    """

    def __init__(self) -> None:
        self._codegen = CodegenExecutor()
        self._critique = CritiqueExecutor()

    def _codegen_with_retries(
        self,
        ctx: HarnessContext,
        prev: list[AnnotatedValue],
        prev_desc: str,
        priors: list[str],
        retry_budget: int,
        *,
        question: str,
        spec_block: str,
    ) -> tuple[dict | None, str | None, list[str]]:
        """Drive `CodegenExecutor.try_once` with `retry_budget` retries for
        transient parse/exec failures. Returns `(env, code, priors)` on
        success or `(None, None, priors)` when the retry budget is exhausted.
        `MissingData` propagates from the executor."""
        priors = list(priors)
        # Total tries = 1 initial + retry_budget retries.
        for try_idx in range(retry_budget + 1):
            env, code, priors = self._codegen.try_once(
                ctx, question, prev, prev_desc, priors, spec_block, try_idx,
            )
            if env is not None:
                return env, code, priors
        return None, None, priors

    def run(
        self, prev: list[AnnotatedValue], ctx: HarnessContext, *, plan: Plan,
    ) -> str:
        """Takes the user's question + extracted prev values + the active Plan,
        returns the final answer string. Pulls `computation` (`task` /
        `qualifiers`) and `presentation` (`units_out` / `precision` /
        `answer_form`) directly off the Plan — `Plan.model_validate_json`
        has already coerced them to the right types."""
        question = ctx.question
        prev_desc = _prev_desc(prev)
        comp, pres = plan.computation, plan.presentation
        task = comp.task or ""
        ctx.emit("compute", "starting", question=question,
                 computation=task, qualifiers=comp.qualifiers,
                 prev_summary=prev_desc)

        spec_block = _constraints_block(
            computation=task, qualifiers=comp.qualifiers,
            units_out=pres.units_out or "", precision=pres.precision,
            answer_form=pres.answer_form or "scalar",
        )

        a1_env, a1_code, a1_priors = self._codegen_with_retries(
            ctx, prev, prev_desc, priors=[],
            retry_budget=ctx.config.compute_max_attempts - 1,
            question=question, spec_block=spec_block,
        )
        if a1_env is None:
            raise StepFailed(
                "compute",
                f"could not produce a result on attempt 1: priors={a1_priors}",
            )
        a1_result = str(a1_env["result"])

        accept, reason = self._critique.critique(
            ctx, question, prev_desc, a1_code, a1_result, spec_block,
        )
        if accept:
            ctx.emit("compute", "attempt 1 self-critique ACCEPT", text=a1_result)
            return a1_result
        ctx.emit("compute", "attempt 1 self-critique REVISE", reason=reason)

        revise_prior = (
            f"Produced result {a1_result!r}. Self-critique flagged: {reason}\n"
            f"Code was:\n```python\n{a1_code}\n```"
        )
        try:
            a2_env, _, _ = self._codegen_with_retries(
                ctx, prev, prev_desc, priors=[revise_prior], retry_budget=0,
                question=question, spec_block=spec_block,
            )
        except MissingData as e:
            ctx.emit("compute", "attempt 2 MISSING; falling back to attempt 1",
                     reason=e.reason, missing=e.missing, fallback_text=a1_result)
            return a1_result
        if a2_env is None:
            ctx.emit(
                "compute",
                "attempt 2 produced no result; falling back to attempt 1",
                fallback_text=a1_result,
            )
            return a1_result
        a2_result = str(a2_env["result"])
        ctx.emit("compute", "attempt 2 returned (no re-critique)", text=a2_result)
        return a2_result
