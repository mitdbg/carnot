"""compute operator — codegen + execution over the extracted-value env.

`ComputeOp.run` generates Python that assigns `result` (the answer string),
execs it in-process, then self-critiques — all under one retry loop covering parse
failures, exec exceptions, and critique REVISE verdicts.

Codegen emits either a fenced ```python``` block or a bare missing-data JSON object
(`{"missing": [...], "description": "..."}` → `MissingData`); a malformed reply raises
`_ParseFailure`, fed back as the next attempt's `prev_failure`."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from skunk.common import Effort, strip_code_fence
from skunk.errors import MissingData, ParseError, StepFailed
from skunk.prompted_call import PromptedCall
from skunk.common import AnnotatedValue, HarnessContext
from skunk.plan import Plan, Presentation
from skunk.pyexec import exec_python_with_env
from skunk.question_explainer import ConceptExplanation


class _ParseFailure(Exception):
    """Codegen reply was neither valid missing-data JSON nor a fenced code block.
    `hint` is fed back as the next attempt's `prev_failure`; `tag` labels the trace."""

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
    """LLM-emitted insufficient-data signal: `{"missing": [...], "description": "..."}`."""

    model_config = ConfigDict(frozen=True)
    description: str
    missing: list[str] = []


_PREVIEW_MAX_ROWS = 10
_PREVIEW_MAX_COLS = 8


def prev_desc(prev: list[AnnotatedValue], *, full: bool = False) -> str:
    """Render `prev` for the codegen/critique prompts; non-scalar entries shown as
    `.to_string()` of their DataFrame. `full=False` (codegen) truncates to a head
    preview; `full=True` (critique) emits every row/col for cell-level verification."""
    # TODO: try a metadata-only variant (schema, no frame rows) — showing cells may
    # bias the model toward those rows. Compare accuracy + token spend.
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

        if full:
            view = df
            max_rows: int | None = None
            max_cols: int | None = None
            header = "         frame:"
        else:
            view = df.iloc[:_PREVIEW_MAX_ROWS, :_PREVIEW_MAX_COLS]
            max_rows = _PREVIEW_MAX_ROWS
            max_cols = _PREVIEW_MAX_COLS
            header = "         frame.head():"
        with pd.option_context(
            "display.max_rows", max_rows,
            "display.max_columns", max_cols,
            "display.width", 120,
        ):
            rendered = view.to_string()
        lines.append(header)
        lines.extend("         " + ln for ln in rendered.splitlines())

        if not full:
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


def _parse_codegen(raw: str) -> str:
    """Parse a codegen reply into a Python code string. A `{`-leading JSON object is
    the missing-data signal (→ `MissingData`); malformed JSON or empty code →
    `_ParseFailure`. Not a `PromptedCall` parse hook — these are application signals,
    not format errors for the prompt layer to retry."""
    # Strip fences first so `{`-dispatch works for both bare and ```json-wrapped JSON.
    s = strip_code_fence(raw).strip()

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

    code = s
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


def _parse_critique(raw: str, ctx: HarnessContext) -> CritiqueResult:
    """Parse a critique reply into a `CritiqueResult`. Malformed JSON → `ParseError`
    (re-prompted once; `_decide` then degrades to REVISE, never silent accept)."""
    try:
        return CritiqueResult.model_validate_json(strip_code_fence(raw))
    except ValidationError as e:
        raise ParseError(raw, str(e)) from e


class Codegen:
    _SYSTEM_PROMPT = """\
You write Python that produces the final answer string, or emit a
structured missing-data signal.

## Inputs

- The user's question.
- `computation` (JSON): the planner's `{task, qualifiers}`. `task` is
  what to compute; each `qualifier` is a MUST-apply modifier.
- `presentation` (JSON): the planner's `{units_out, precision,
  answer_form}`. Convert to `units_out`, round to `precision` decimal
  places, format to `answer_form`. Null fields mean unconstrained.
- Optional `## Concept references` section: one block per non-obvious
  concept the question references — canonical definition + formula
  for named operations, domain-specific conventions, etc. Treat each
  as the authoritative reference; if your default implementation
  diverges from it, follow the reference.
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
{{ default_tail }}"""

    def __init__(self) -> None:
        self._prompt = PromptedCall(
            name="compute.codegen",
            system_prompt=self._SYSTEM_PROMPT,
            default_effort="medium",
        )

    def codegen(
        self,
        ctx: HarnessContext,
        plan: Plan,
        prev: list[AnnotatedValue],
        prev_code: str | None,
        prev_failure: str | None,
        concept_explanations: list[ConceptExplanation] = (),
        *,
        effort: Effort | None = None,
    ) -> str:
        """One LLM call returning the generated code. Raises `MissingData` (signal,
        propagates past compute) or `_ParseFailure` (caller retries with `hint`).
        `prev_code`/`prev_failure` describe only the most-recent failed attempt —
        accumulating older ones dilutes the issue to fix."""
        user_msg = (
            f"Question:\n{ctx.question}\n\n"
            f"computation = {plan.computation.model_dump_json()}\n"
            f"presentation = {plan.presentation.model_dump_json()}\n\n"
        )
        if concept_explanations:
            block = "\n\n".join(
                f"### {c.concept}\n{c.explanation}"
                for c in concept_explanations
            )
            user_msg += f"## Concept references\n{block}\n\n"
        user_msg += (
            f"prev =\n{prev_desc(prev)}\n\n"
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
        raw = self._prompt.call(ctx, user_msg, effort=effort)
        return _parse_codegen(raw)


class Critique:
    _SYSTEM_PROMPT = """\
You review code against ONE specific constraint from the plan.
Decide accept or revise.

## Inputs

- Question — context for what the agent is answering.
- One specific constraint — either a qualifier phrase or the
  presentation block. This is the ONLY constraint you are checking.
  Other constraints are someone else's job — do not flag issues
  outside this one.
- `prev` summary — the data the code consumed (full, every row/col).
- The Python code that ran.
- The produced `result` string.

## Output format

A single bare JSON object. No fences, no prose. Exactly one of:

  {"verdict": "accept"}
  {"verdict": "revise",
   "reason": "<short, specific reason tied to THIS constraint>"}

Accept iff the code AND result honor your assigned constraint.
Revise otherwise. Be specific about what the code did wrong in your
reason.
{{ default_tail }}"""

    def __init__(self) -> None:
        self._prompt = PromptedCall(
            name="compute.critique",
            system_prompt=self._SYSTEM_PROMPT,
            default_effort="medium",
            parse=_parse_critique,
        )

    def _decide(
        self,
        ctx: HarnessContext,
        question: str,
        focus_block: str,
        prev: list[AnnotatedValue],
        code: str,
        result_text: str,
    ) -> tuple[bool, str]:
        """Shared body for the focused critique. Returns (accept, reason); a
        malformed reply → REVISE with a diagnostic reason (never silent accept)."""
        user = (
            f"Question:\n{question}\n\n"
            f"{focus_block}\n\n"
            f"prev =\n{prev_desc(prev, full=True)}\n\n"
            f"Code that ran:\n```python\n{code}\n```\n\n"
            f"Produced result:\n{result_text}\n\n"
            f"Output a single bare JSON object with your verdict."
        )
        try:
            verdict = self._prompt.call(ctx, user)
        except ParseError as e:
            return False, f"self-critique produced malformed reply: {e.detail}"
        if verdict.verdict == "accept":
            return True, ""
        return False, verdict.reason

    def critique_qualifier(
        self,
        ctx: HarnessContext,
        question: str,
        qualifier: str,
        prev: list[AnnotatedValue],
        code: str,
        result_text: str,
    ) -> tuple[bool, str]:
        """Focused critique against a single qualifier phrase."""
        focus = f"Constraint to check (a qualifier):\n{qualifier}"
        return self._decide(ctx, question, focus, prev, code, result_text)

    def critique_presentation(
        self,
        ctx: HarnessContext,
        question: str,
        presentation: "Presentation",
        prev: list[AnnotatedValue],
        code: str,
        result_text: str,
    ) -> tuple[bool, str]:
        """Focused critique against the presentation block."""
        focus = (
            "Constraint to check (the presentation block):\n"
            f"{presentation.model_dump_json()}"
        )
        return self._decide(ctx, question, focus, prev, code, result_text)


class ComputeOp:
    """The compute operator — unified codegen → exec → critique loop. One public `run()`."""

    def __init__(self) -> None:
        self._codegen = Codegen()
        self._critique = Critique()

    def _critique_parallel(
        self,
        ctx: HarnessContext,
        plan: Plan,
        prev: list[AnnotatedValue],
        code: str,
        result: str,
    ) -> tuple[bool, str]:
        """Fan out N+1 focused critiques in parallel — one per qualifier plus one
        for the presentation block. Any REVISE → overall REVISE (reasons concatenated)."""
        qualifiers = plan.computation.qualifiers
        n_calls = len(qualifiers) + 1
        ctx.emit("compute", "critique_fanout", n_calls=n_calls)
        revises: list[tuple[str, str]] = []  # (focus_label, reason), in completion order
        with ThreadPoolExecutor(
            max_workers=ctx.config.max_parallel_workers
        ) as pool:
            futs: dict = {}
            for q in qualifiers:
                f = pool.submit(
                    self._critique.critique_qualifier,
                    ctx, ctx.question, q, prev, code, result,
                )
                futs[f] = ("qualifier", q)
            f = pool.submit(
                self._critique.critique_presentation,
                ctx, ctx.question, plan.presentation, prev, code, result,
            )
            futs[f] = ("presentation", "presentation")
            for fut in as_completed(futs):
                kind, label = futs[fut]
                try:
                    accept, reason = fut.result()
                except Exception as e:
                    accept, reason = False, f"critique error: {e}"
                if not accept:
                    revises.append((f"{kind} '{label}'", reason))
        if not revises:
            return True, ""
        return False, "; ".join(f"{label}: {reason}" for label, reason in revises)

    def run(
        self,
        prev: list[AnnotatedValue],
        ctx: HarnessContext,
        plan: Plan,
        concept_explanations: list[ConceptExplanation] = (),
    ) -> str:
        """Retry loop over `compute_max_attempts`: each iteration does codegen → exec
        → critique, and any failure feeds the next `prev_failure`. Returns on ACCEPT,
        or the last uncritiqued result if the budget exhausts after a clean exec.
        Raises `StepFailed` if no iteration ever execs cleanly; propagates `MissingData`."""
        # No "starting" boundary emit — the orchestrator's trace records this
        # step's boundary; the plan/computation are the planner step's output.

        prev_code: str | None = None
        prev_failure: str | None = None
        last_result: str | None = None

        for try_idx in range(ctx.config.compute_max_attempts):
            # Escalate effort after the first failed attempt.
            retry_effort: Effort | None = "high" if try_idx > 0 else None
            try:
                code = self._codegen.codegen(
                    ctx, plan, prev, prev_code, prev_failure,
                    concept_explanations, effort=retry_effort,
                )
            except _ParseFailure as e:
                prev_code = None
                prev_failure = e.hint
                ctx.emit(
                    "compute", "codegen_parse_failed",
                    attempt=try_idx + 1, tag=e.tag, hint=e.hint,
                )
                continue
            except MissingData as e:
                ctx.emit(
                    "compute", "codegen_missing",
                    attempt=try_idx + 1, missing=e.missing, description=e.reason,
                )
                if last_result is not None:
                    ctx.emit(
                        "compute", "missing_fallback",
                        attempt=try_idx + 1, fallback_text=last_result,
                    )
                    return last_result
                raise

            ctx.emit("compute", "codegen_code", attempt=try_idx + 1, code=code)
            try:
                env, _ = exec_python_with_env(code, {"prev": prev})
            except Exception as e:
                prev_code = code
                prev_failure = f"Exception during exec: {e}"
                ctx.emit("compute", "exec_failed", attempt=try_idx + 1, error=str(e))
                continue

            result = str(env.get("result"))
            last_result = result
            ctx.emit("compute", "exec_result", attempt=try_idx + 1, text=result)

            accept, reason = self._critique_parallel(ctx, plan, prev, code, result)
            if accept:
                ctx.emit("compute", "critique_accept", attempt=try_idx + 1, text=result)
                return result
            ctx.emit("compute", "critique_revise", attempt=try_idx + 1, reason=reason)
            prev_code = code
            prev_failure = f"Produced result {result!r}. Self-critique flagged: {reason}"

        if last_result is not None:
            ctx.emit("compute", "budget_exhausted", fallback_text=last_result)
            return last_result
        raise StepFailed(
            "compute",
            f"no successful exec within budget; last failure: {prev_failure}",
        )
