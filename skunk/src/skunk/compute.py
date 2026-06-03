from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from skunk.common import Effort, strip_code_fence
from skunk.errors import MissingData, ParseError, StepFailed
from skunk.prompted_call import PromptedCall
from skunk.common import AnnotatedValue, ExecutionContext, input_values_desc
from skunk.plan import Plan, Requirements
from skunk.pyexec import exec_python_with_env
from skunk.question_explainer import ConceptExplanation



class CritiqueResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    verdict: Literal["accept", "revise"]
    reason: str = ""

    @model_validator(mode="after")
    def _revise_needs_reason(self) -> CritiqueResult:
        if self.verdict == "revise" and not self.reason.strip():
            raise ValueError("verdict=revise requires non-empty 'reason'")
        return self


class MissingDataSignal(BaseModel):
    model_config = ConfigDict(frozen=True)
    description: str
    missing: list[str] = []


def _parse_codegen(raw: str) -> str:
    """Parse a codegen reply into a Python code string. A `{`-leading JSON object is
    the missing-data signal (→ `MissingData`); malformed JSON or empty code →
    `ParseError`. Not a `PromptedCall` parse hook — called after `call()` returns so
    the prompt layer never sees it; `ComputeOp.run()` catches it and retries."""
    # Strip fences first so `{`-dispatch works for both bare and ```json-wrapped JSON.
    s = strip_code_fence(raw).strip()

    if s.startswith("{"):
        try:
            signal = MissingDataSignal.model_validate_json(s)
        except ValidationError as e:
            raise ParseError(
                raw=raw,
                detail=(
                    f"malformed missing-data JSON — {e}\n"
                    "Output ONLY a fenced ```python``` block OR a bare JSON "
                    '{"missing": [...], "description": "..."} object.'
                ),
            )
        raise MissingData(signal.description, missing=signal.missing)

    if not s:
        raise ParseError(
            raw=raw,
            detail=(
                "empty or un-fenced code.\n"
                "Output ONLY a fenced ```python``` block OR a bare JSON "
                '{"missing": [...], "description": "..."} object.'
            ),
        )
    return s


def _parse_critique(raw: str, _: ExecutionContext) -> CritiqueResult:
    """Parse a critique reply into a `CritiqueResult`. Malformed JSON → `ParseError`
    (re-prompted once; `_decide` then degrades to REVISE, never silent accept)."""
    try:
        return CritiqueResult.model_validate_json(strip_code_fence(raw))
    except ValidationError as e:
        raise ParseError(raw, str(e)) from e


class Codegen:
    _SYSTEM_PROMPT = """\
You write Python that produces the final answer string, or emit a structured missing-data signal.

## Inputs
- The user's question — the authoritative statement of what to compute.
- `requirements` (JSON): the planner's distilled constraints on the answer.
  `qualifiers` — each a MUST-apply modifier from the question. The
  output-format fields `units_out` / `precision` / `answer_form` — convert
  to `units_out`, round to `precision` decimal places, format to
  `answer_form`. Null fields mean unconstrained.
- Optional `## Concept references` section: one block per non-obvious
  concept the question references — canonical definition + formula
  for named operations, domain-specific conventions, etc. Treat each
  as the authoritative reference; if your default implementation
  diverges from it, follow the reference.
- `input_values`: list[AnnotatedValue] in the exec environment; these are values previous agents deemed relevant for answering the question.

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

For each non-scalar entry the `input_values =` block below shows a schema view:
the full axis labels (index labels + column names), per-column dtypes, and a
2-row sample — NOT every cell. The full frame is what you get in the exec env;
write code against it (`.loc[...]`, `.idxmax()`, etc.) using the labels shown.
Apply unit conversions once over the whole frame (e.g. `df * 1e-6`), never
cell-by-cell.

## Output format

Emit exactly one of the two forms below and nothing else — no prose,
no commentary, no second block, no fence around (b):

  (a) Success — a single fenced ```python``` block. Assign the final
      answer string to `result`. The string contains ONLY the requested
      answer — no prose, no "Answer:", no question restatement. For a
      multi-part question, `result` is only the ultimate quantity /
      identifier asked for, not any intermediate.
  (b) Insufficient data — a single bare JSON object:
        {"missing": [<short identifier strings>],
         "description": "<one-line explanation>"}
      Never fabricate values to avoid this path — if a required value
      isn't in `input_values`, emit (b).

Available imports: numpy (np), pandas (pd), math, statsmodels.api (sm).
"""

    _prompt = PromptedCall(
        name="compute.codegen",
        system_prompt=_SYSTEM_PROMPT,
        default_effort="medium",
        output_instruction="Produce a fenced ```python``` block OR a bare missing-data JSON object.",
    )

    async def codegen(
        self,
        ctx: ExecutionContext,
        plan: Plan,
        input_values: list[AnnotatedValue],
        prev_code: str | None,
        prev_failure: str | None,
        concept_explanations: Sequence[ConceptExplanation] = (),
        *,
        effort: Effort | None = None,
    ) -> str:
        """One LLM call returning the generated code. Raises `MissingData` (signal,
        propagates past compute) or `_ParseFailure` (caller retries with `hint`).
        `prev_code`/`prev_failure` describe only the most-recent failed attempt —
        accumulating older ones dilutes the issue to fix."""
        user_msg = (
            f"Question:\n{ctx.question}\n\n"
            f"requirements = {plan.requirements.model_dump_json()}\n\n"
        )
        if concept_explanations:
            block = "\n\n".join(
                f"### {c.concept}\n{c.explanation}"
                for c in concept_explanations
            )
            user_msg += f"## Concept references\n{block}\n\n"
        user_msg += f"input_values =\n{input_values_desc(input_values)}"
        if prev_failure:
            user_msg += "\n\nYour previous attempt failed."
            if prev_code:
                user_msg += f"\nPrevious code:\n```python\n{prev_code}\n```"
            user_msg += (
                f"\nSpecific issue(s) to address:\n{prev_failure}\n"
                "Focus on fixing them without introducing new mistakes."
            )
        raw = await self._prompt.call(ctx, user_msg, effort=effort)
        return _parse_codegen(raw)


class Critique:
    _SYSTEM_PROMPT = """\
You review code against ONE specific constraint from the plan.
Decide accept or revise.

## Inputs

- Question — context for what the agent is answering.
- One specific constraint — either a qualifier phrase or the
  output-format fields (units_out / precision / answer_form). This is
  the ONLY constraint you are checking. Other constraints are someone
  else's job — do not flag issues outside this one.
- `input_values` summary — the data the code consumed (full, every row/col).
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
"""

    _prompt = PromptedCall(
        name="compute.critique",
        system_prompt=_SYSTEM_PROMPT,
        default_effort="off",
        parse=_parse_critique,
        output_instruction="Output a single bare JSON object with your verdict — no markdown fences, no prose.",
    )

    async def review(
        self,
        ctx: ExecutionContext,
        plan: Plan,
        input_values: list[AnnotatedValue],
        code: str,
        result: str,
    ) -> tuple[bool, str]:
        """Fan out N+1 focused critiques in parallel — one per qualifier plus one
        for the output-format fields. Any REVISE → overall REVISE (reasons concatenated)."""
        qualifiers = plan.requirements.qualifiers
        n_calls = len(qualifiers) + 1
        ctx.emit(f"critique_fanout n_calls={n_calls}")
        labels = [("qualifier", q) for q in qualifiers] + [
            ("output format", "units_out/precision/answer_form")
        ]
        coros = [
            self._critique_qualifier(ctx, ctx.question, q, input_values, code, result)
            for q in qualifiers
        ]
        coros.append(
            self._critique_format(
                ctx, ctx.question, plan.requirements, input_values, code, result,
            )
        )
        outcomes = await asyncio.gather(*coros, return_exceptions=True)
        revises: list[tuple[str, str]] = []  # (focus_label, reason)
        for (kind, label), outcome in zip(labels, outcomes):
            if isinstance(outcome, BaseException):
                accept, reason = False, f"critique error: {outcome}"
            else:
                accept, reason = outcome
            if not accept:
                revises.append((f"{kind} '{label}'", reason))
        if not revises:
            return True, ""
        return False, "; ".join(f"{label}: {reason}" for label, reason in revises)

    async def _decide(
        self,
        ctx: ExecutionContext,
        question: str,
        focus_block: str,
        input_values: list[AnnotatedValue],
        code: str,
        result_text: str,
    ) -> tuple[bool, str]:
        """Shared body for the focused critique. Returns (accept, reason); a
        malformed reply → REVISE with a diagnostic reason (never silent accept)."""
        user = (
            f"Question:\n{question}\n\n"
            f"{focus_block}\n\n"
            f"input_values =\n{input_values_desc(input_values)}\n\n"
            f"Code that ran:\n```python\n{code}\n```\n\n"
            f"Produced result:\n{result_text}"
        )
        try:
            verdict = await self._prompt.call(ctx, user)
        except ParseError as e:
            return False, f"self-critique produced malformed reply: {e.detail}"
        if verdict.verdict == "accept":
            return True, ""
        return False, verdict.reason

    async def _critique_qualifier(
        self,
        ctx: ExecutionContext,
        question: str,
        qualifier: str,
        input_values: list[AnnotatedValue],
        code: str,
        result_text: str,
    ) -> tuple[bool, str]:
        """Focused critique against a single qualifier phrase."""
        focus = f"Constraint to check:\n{qualifier}"
        return await self._decide(ctx, question, focus, input_values, code, result_text)

    async def _critique_format(
        self,
        ctx: ExecutionContext,
        question: str,
        requirements: "Requirements",
        input_values: list[AnnotatedValue],
        code: str,
        result_text: str,
    ) -> tuple[bool, str]:
        """Focused critique against the output-format fields (units_out / precision /
        answer_form), checked as one unit — the qualifiers are someone else's job."""
        focus = (
            "Output format to check:\n"
            f"{requirements.model_dump_json(include={'units_out', 'precision', 'answer_form'})}"
        )
        return await self._decide(ctx, question, focus, input_values, code, result_text)


class ComputeOp:
    """The compute operator — unified codegen → exec → critique loop. One public `run()`."""
    def __init__(self) -> None:
        self._codegen = Codegen()
        self._critique = Critique()

    async def run(
        self,
        input_values: list[AnnotatedValue],
        ctx: ExecutionContext,
        plan: Plan,
        concept_explanations: Sequence[ConceptExplanation] = (),
    ) -> str:
        """Retry loop over `compute_max_attempts`: each iteration does codegen → exec
        → critique, and any failure feeds the next `prev_failure`. Returns on ACCEPT,
        or — if the budget exhausts after at least one clean exec — the last result
        (which critique rejected) as a best-effort answer. Raises `StepFailed` if no
        iteration ever execs cleanly; propagates `MissingData` (codegen's give-up
        signal) so the orchestrator can recover by gathering more data and replanning."""
        # No "starting" boundary emit — the orchestrator's trace records this
        # step's boundary; the plan/requirements are the planner step's output.

        prev_code: str | None = None
        prev_failure: str | None = None
        last_result: str | None = None

        for try_idx in range(ctx.config.compute_max_attempts):
            try:
                code = await self._codegen.codegen(
                    ctx, plan, input_values, prev_code, prev_failure,
                    concept_explanations,
                )
            except ParseError as e:
                prev_code = None
                prev_failure = e.detail
                ctx.emit(f"codegen_parse_failed attempt={try_idx + 1} detail={e.detail!r}")
                continue
            except MissingData as e:
                # Trust the signal: propagate so the orchestrator's recovery loop can
                # gather the missing data and replan. We deliberately do NOT fall back
                # to a prior clean-but-critique-rejected result — that would suppress
                # recovery in favor of an answer the self-critique already flagged.
                ctx.emit(
                    f"codegen_missing attempt={try_idx + 1} missing={e.missing!r} "
                    f"description={e.reason!r}"
                )
                raise

            ctx.emit(f"codegen_code attempt={try_idx + 1} code={code!r}")
            try:
                env, _ = exec_python_with_env(code, {"input_values": input_values})
            except Exception as e:
                prev_code = code
                prev_failure = f"Exception during exec: {e}"
                ctx.emit(f"exec_failed attempt={try_idx + 1} error={str(e)!r}")
                continue

            result = str(env.get("result"))
            last_result = result
            ctx.emit(f"exec_result attempt={try_idx + 1} text={result!r}")

            accept, reason = await self._critique.review(ctx, plan, input_values, code, result)
            if accept:
                ctx.emit(f"critique_accept attempt={try_idx + 1} text={result!r}")
                return result
            ctx.emit(f"critique_revise attempt={try_idx + 1} reason={reason!r}")
            prev_code = code
            prev_failure = f"Produced result {result!r}. Self-critique flagged: {reason}"

        if last_result is not None:
            ctx.emit(f"budget_exhausted fallback_text={last_result!r}")
            return last_result
        raise StepFailed(
            "compute",
            f"no successful exec within budget; last failure: {prev_failure}",
        )
