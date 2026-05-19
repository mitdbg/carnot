"""compute operator — codegen + execution over the extracted-value env.

Takes the user's question plus `prev: list[AnnotatedValue]`, generates Python
that assigns `result` to the formatted answer string, execs it in-process,
then self-critiques via `CritiquePromptedCall`. A single retry loop in
`ComputeExecutor.run` covers parse failures, exec exceptions, and critique
REVISE verdicts under one shared budget. Returns the answer as a plain `str`.

Two LLM contracts:
  - codegen: emits EITHER a fenced ```python``` block (success) OR a bare
    JSON object `{"missing": [...], "description": "..."}` (insufficient
    data). Dispatch is on the first non-whitespace character. The JSON
    form surfaces as `MissingData(description, missing=...)`. A malformed
    reply (neither valid missing-data JSON nor a fenced code block)
    raises `_ParseFailure`, which `run()` catches and feeds back as the
    next iteration's `prev_failure`.
  - critique: emits a single bare JSON object validated onto `CritiqueResult`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from skunk.common import strip_code_fence
from skunk.errors import MissingData, StepFailed
from skunk.prompted_call import PromptedCall
from skunk.models import AnnotatedValue, HarnessContext
from skunk.plan import Plan
from skunk.pyexec import exec_python_with_env


class _ParseFailure(Exception):
    """Codegen reply was neither valid missing-data JSON nor a fenced code
    block. Caught inside `ComputeExecutor.run`; `hint` is fed back as the
    next iteration's `prev_failure`, `tag` labels the trace event."""

    def __init__(self, hint: str, tag: Literal["malformed_missing", "no_code"]):
        super().__init__(tag)
        self.hint = hint
        self.tag = tag


class CritiqueResult(BaseModel):
    """Parsed critique verdict. `reason` required when `verdict='revise'`."""

    model_config = ConfigDict(frozen=True)
    verdict: Literal["accept", "revise"]
    reason: str = ""

    @model_validator(mode="after")
    def _revise_needs_reason(self) -> CritiqueResult:
        if self.verdict == "revise" and not self.reason.strip():
            raise ValueError("verdict=revise requires non-empty 'reason'")
        return self


class MissingDataSignal(BaseModel):
    """LLM-emitted insufficient-data signal. Wire shape:
      {"missing": [...], "description": "..."}
    `description` is required; `missing` defaults to []."""

    model_config = ConfigDict(frozen=True)
    description: str
    missing: list[str] = []


_PREVIEW_MAX_ROWS = 10
_PREVIEW_MAX_COLS = 8


def _prev_desc(prev: list[AnnotatedValue]) -> str:
    """Render `prev` for the codegen/critique prompts. Every non-scalar entry
    is shown as its `.frame.head().to_string()` so the prompt and the exec
    env see the same pandas object."""
    import pandas as pd

    lines = [f"prev ({len(prev)} entries)"]
    for i, e in enumerate(prev):
        label = e.description or "(no description)"
        if e.kind == "scalar":
            lines.append(f"  prev[{i}]  description: {label!r}")
            lines.append(f"         value={e.value!r}  (kind=scalar, unit={e.unit!r})")
            continue

        df = e.frame
        n_rows, n_cols = df.shape
        if e.kind == "vector":
            meta = (
                f"kind=vector, index_name={e.index_name!r}, "
                f"unit={e.unit!r}, shape=({n_rows}, {n_cols})"
            )
        else:
            meta = (
                f"kind=table, row_name={e.row_name!r}, col_name={e.col_name!r}, "
                f"unit={e.unit!r}, shape=({n_rows}, {n_cols})"
            )

        lines.append(f"  prev[{i}]  description: {label!r}")
        lines.append(f"         {meta}")

        if n_rows == 0:
            lines.append("         frame: (empty)")
            continue

        head = df.iloc[:_PREVIEW_MAX_ROWS, :_PREVIEW_MAX_COLS]
        with pd.option_context(
            "display.max_rows", _PREVIEW_MAX_ROWS,
            "display.max_columns", _PREVIEW_MAX_COLS,
            "display.width", 120,
        ):
            rendered = head.to_string()
        lines.append("         frame.head():")
        lines.extend("         " + ln for ln in rendered.splitlines())

        more_rows = max(0, n_rows - _PREVIEW_MAX_ROWS)
        more_cols = max(0, n_cols - _PREVIEW_MAX_COLS)
        if more_rows or more_cols:
            tail_bits = []
            if more_rows:
                tail_bits.append(f"{more_rows} more rows")
            if more_cols:
                tail_bits.append(f"{more_cols} more cols")
            lines.append(f"         ... {', '.join(tail_bits)}")
    return "\n".join(lines)


class CodegenPromptedCall(PromptedCall):
    name: str = "compute.codegen"
    system_prompt: str = """\
You write Python that produces the final answer string, or emit a
structured missing-data signal.

## Inputs

- The user's question.
- `computation` (JSON): the planner's `{task, qualifiers}`. `task` is
  what to compute; each `qualifier` is a MUST-apply modifier.
- `presentation` (JSON): the planner's `{units_out, precision,
  answer_form}`. Convert to `units_out`, round to `precision` decimal
  places, format to `answer_form`. Null fields mean unconstrained.
- `prev`: list[AnnotatedValue] in the exec environment; these are values previous agents deemed relevant for answering the question.

## AnnotatedValue API

  .description   natural-language label
  .frame         pd.DataFrame view of the payload (uniform across kinds)
  .unit          natural-language unit, e.g. "millions of dollars", "percent"
  .kind          "scalar" | "vector" | "table"  (rarely needed; prefer .frame)
  .index_name    (vector)   .row_name / .col_name (table)
  .value         raw payload — only use if you specifically need the dict/list form

## Payload access

Always read data through `e.frame`:

  scalar  →  1x1 DataFrame; `e.frame.iat[0, 0]` for the raw cell.
             (scalar from lookup_external may be N x 1 — a column of values.)
  vector  →  N x 1 DataFrame; index is `e.index_name`, the single column
             is named after the description. Use `e.frame.loc[label]`
             or `e.frame.iloc[:, 0]` for the Series form.
  table   →  R x C DataFrame; `index.name == e.row_name`,
             `columns.name == e.col_name`.

The `prev =` block below shows `frame.head()` for each entry, so what
you see is what you'll get in the exec env. Apply unit conversions
once over the whole frame (e.g. `df * 1e-6`), never cell-by-cell.

## Output format

Emit exactly one of the two forms below and nothing else — no prose,
no commentary, no second block, no fence around (b):

  (a) Success — a single fenced ```python``` block. Assign the final
      answer string to `result`. The string contains ONLY the requested
      answer — no prose, no "Answer:", no question restatement.
  (b) Insufficient data — a single bare JSON object:
        {"missing": [<short identifier strings>],
         "description": "<one-line explanation>"}
      Never fabricate values to avoid this path — if a required value
      isn't in `prev`, emit (b).

## Principles

- Every modifier in the question (or echoed in `computation`) MUST
  appear as an explicit operation in the code, not a comment.
- Prefer a printed total/summary row over re-aggregating its
  components when both are in `prev`.
- Every cell in a vector/table shares one unit — apply any unit
  conversion once over the whole payload, never cell-by-cell.
- For multi-part questions, `result` contains only the ultimate
  quantity / identifier the question asks for.
- Available imports: numpy (np), pandas (pd), math, statsmodels.api (sm).
"""

    def codegen(
        self,
        ctx: HarnessContext,
        plan: Plan,
        prev: list[AnnotatedValue],
        prev_code: str | None,
        prev_failure: str | None,
    ) -> str:
        """One LLM call. Returns the generated code on success.
        Raises:
          - `MissingData` — structured missing-data signal; propagates past
            `compute` to the orchestrator.
          - `_ParseFailure` — reply was unparseable; caller retries with
            `failure.hint` as the next `prev_failure`.

        `prev_code` + `prev_failure` describe the single most-recent
        failed attempt (parse/exec failure or critique REVISE verdict).
        Older attempts are deliberately omitted — accumulating them
        dilutes the actual issue to fix. `prev_code` is None when there
        was no parseable code on the prior attempt (parse failure)."""
        user_msg = (
            f"Question:\n{ctx.question}\n\n"
            f"computation = {plan.computation.model_dump_json()}\n"
            f"presentation = {plan.presentation.model_dump_json()}\n\n"
            f"prev =\n{_prev_desc(prev)}\n\n"
            f"Produce a fenced ```python``` block OR a bare missing-data JSON object."
        )
        if prev_failure:
            user_msg += "\n\nYour previous attempt failed."
            if prev_code:
                user_msg += f"\nPrevious code:\n```python\n{prev_code}\n```"
            user_msg += (
                f"\nSpecific issue to address:\n{prev_failure}\n"
                "Focus on fixing this specific issue without introducing new mistakes."
            )
        resp = ctx.llm_client.call(
            self.assemble_system_prompt(ctx), user_msg, thinking_budget=4096, ctx=ctx
        )
        raw = resp.text
        ctx.emit("compute", "codegen response", raw=raw)
        s = raw.strip()

        if s.startswith("{"):
            try:
                signal = MissingDataSignal.model_validate_json(s)
            except ValidationError as e:
                raise _ParseFailure(
                    hint=(
                        f"Your previous attempt emitted malformed missing-data JSON. "
                        f"Raw: {raw[:300]}\nError: {e}\n"
                        "Output ONLY a fenced ```python``` block OR a bare JSON "
                        '{"missing": [...], "description": "..."} object.'
                    ),
                    tag="malformed_missing",
                )
            raise MissingData(signal.description, missing=signal.missing)

        code = strip_code_fence(s).strip()
        if not code:
            raise _ParseFailure(
                hint=(
                    f"Your previous attempt produced empty or un-fenced code. "
                    f"Raw: {raw[:300]}\n"
                    "Output ONLY a fenced ```python``` block OR a bare JSON "
                    '{"missing": [...], "description": "..."} object.'
                ),
                tag="no_code",
            )
        return code


class CritiquePromptedCall(PromptedCall):
    name: str = "compute.critique"
    system_prompt: str = """\
You review code another agent wrote and the string it produced.
Decide accept or revise.

## Inputs

Question, `computation` and `presentation` JSON (treat them as a
checklist: each field is a constraint the result must satisfy),
`prev` summary (each non-scalar entry shown as a `frame.head()`
preview alongside kind/axis/unit metadata; the same `.frame` view
the code consumed), the Python that ran, and the produced `result`.

## Output format

A single bare JSON object. No fences, no prose. Exactly one of:

  {"verdict": "accept"}
  {"verdict": "revise",
   "reason": "<short reasons why the previous agent was incorrect and issues a re-run should address>"}

## Revise if

- A numeric modifier from the question (e.g. signed vs absolute,
  per-capita, percent vs decimal, compound vs simple growth) is
  missing or applied only in a comment. A result whose magnitude or
  sign disagrees with the question under its modifiers is a strong
  signal.
- The question names a specific statistical operation and the code
  uses a related-but-different formula. Reason format:
  "code uses <X> but question asks for <named operation>".
- A description substring used to pick entries from `prev` is not
  distinctive and could collide with another entry.
- A required unit conversion is missing or applied cell-by-cell.
- The result form contradicts the question (wanted "[a, b]" but
  shipped "1.0 2.0"; wanted percent but shipped 0.1234), wraps a
  single value in prose, or doesn't match answer_form / precision.
"""
    def critique(
        self,
        ctx: HarnessContext,
        plan: Plan,
        prev: list[AnnotatedValue],
        code: str,
        result_text: str,
    ) -> tuple[bool, str]:
        """Same-actor review of (code, result). Returns (accept, reason).
        Malformed reply → REVISE with diagnostic reason (never silently accept)."""
        user = (
            f"Question:\n{ctx.question}\n\n"
            f"computation = {plan.computation.model_dump_json()}\n"
            f"presentation = {plan.presentation.model_dump_json()}\n\n"
            f"prev =\n{_prev_desc(prev)}\n\n"
            f"Code that ran:\n```python\n{code}\n```\n\n"
            f"Produced result:\n{result_text}\n\n"
            f"Output a single bare JSON object with your verdict."
        )
        resp = ctx.llm_client.call(self.assemble_system_prompt(ctx), user, ctx=ctx)
        raw = resp.text
        ctx.emit("compute", "self-critique response", raw=raw)
        try:
            verdict = CritiqueResult.model_validate_json(strip_code_fence(raw))
        except ValidationError as e:
            return False, f"self-critique produced malformed reply: {e}"
        if verdict.verdict == "accept":
            return True, ""
        return False, verdict.reason


class ComputeExecutor:
    """Unified codegen → exec → critique loop. One public `run()`."""

    def __init__(self) -> None:
        self._codegen = CodegenPromptedCall()
        self._critique = CritiquePromptedCall()

    def run(
        self,
        prev: list[AnnotatedValue],
        ctx: HarnessContext,
        plan: Plan,
    ) -> str:
        """Single retry loop over `compute_max_attempts` iterations. Each
        iteration does codegen → exec → critique. Any failure (parse, exec,
        or critique REVISE) feeds the next iteration's `prev_failure`.
        Returns on critique ACCEPT, or — if the budget exhausts after at
        least one successful exec — the most recent uncritiqued result as
        a fallback. Raises `StepFailed` if no iteration ever execs cleanly,
        and propagates `MissingData` from the codegen signal."""
        ctx.emit(
            "compute",
            "starting",
            question=ctx.question,
            computation=plan.computation.model_dump(),
            presentation=plan.presentation.model_dump(),
            prev_summary=_prev_desc(prev),
        )

        prev_code: str | None = None
        prev_failure: str | None = None
        last_result: str | None = None

        for try_idx in range(ctx.config.compute_max_attempts):
            try:
                code = self._codegen.codegen(ctx, plan, prev, prev_code, prev_failure)
            except _ParseFailure as e:
                prev_code = None
                prev_failure = e.hint
                ctx.emit(
                    "compute",
                    f"codegen try {try_idx + 1} {e.tag}",
                    hint=e.hint,
                )
                continue
            except MissingData as e:
                ctx.emit(
                    "compute",
                    f"codegen try {try_idx + 1} reported MISSING",
                    missing=e.missing,
                    description=e.reason,
                )
                if last_result is not None:
                    ctx.emit(
                        "compute",
                        "MISSING after a prior successful exec; falling back",
                        fallback_text=last_result,
                    )
                    return last_result
                raise

            ctx.emit("compute", f"codegen try {try_idx + 1} code", code=code)
            try:
                env, _ = exec_python_with_env(code, {"prev": prev})
            except Exception as e:
                prev_code = code
                prev_failure = f"Exception during exec: {e}"
                ctx.emit(
                    "compute", f"codegen try {try_idx + 1} exec failed", error=str(e)
                )
                continue

            result = str(env.get("result"))
            last_result = result
            ctx.emit(
                "compute",
                f"codegen try {try_idx + 1} produced result",
                text=result,
            )

            accept, reason = self._critique.critique(
                ctx, plan, prev, code, result
            )
            if accept:
                ctx.emit(
                    "compute",
                    f"try {try_idx + 1} self-critique ACCEPT",
                    text=result,
                )
                return result
            ctx.emit(
                "compute",
                f"try {try_idx + 1} self-critique REVISE",
                reason=reason,
            )
            prev_code = code
            prev_failure = f"Produced result {result!r}. Self-critique flagged: {reason}"

        if last_result is not None:
            ctx.emit(
                "compute",
                "budget exhausted after critique REVISE; returning last uncritiqued result",
                fallback_text=last_result,
            )
            return last_result
        raise StepFailed(
            "compute",
            f"no successful exec within budget; last failure: {prev_failure}",
        )
