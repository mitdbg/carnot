from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, ValidationError

from skunk.common import Effort, strip_code_fence
from skunk.errors import MissingData, ParseError, StepFailed
from skunk.prompted_call import PromptedCall
from skunk.common import AnnotatedValue, ExecutionContext, input_values_desc
from skunk.pyexec import exec_python_with_env
from skunk.question_explainer import ConceptExplanation


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


class Codegen:
    _SYSTEM_PROMPT = """\
You write Python that produces the final answer string, or emit a structured missing-data signal.

## Inputs
- The user's question — the authoritative statement of what to compute,
  including the output format (units, decimal places, list/bracket shape).
- Optional `## Concept references` section: one block per non-obvious
  concept the question references — canonical definition + formula
  for named operations, domain-specific conventions, etc. Treat each
  as the authoritative reference; if your default implementation
  diverges from it, follow the reference.
- `input_values`: list[AnnotatedValue] in the exec environment; these are values previous agents deemed relevant for answering the question.

## AnnotatedValue API

  .description   natural-language label
  .qualifiers    verbatim page fragments the value was read under — column
                 header, row label, footnote markers, print flags (p/r)
  .frame         pd.DataFrame view of the payload (uniform across kinds)
  .unit          natural-language unit, e.g. "millions of dollars", "percent"
  .kind          "scalar" | "vector" | "table"  (rarely needed; prefer .frame)
  .index_name    (vector)   .row_name / .col_name (table)
  .value         raw payload — only use if you specifically need the dict/list form
  .bulletin           source issue "YYYY-MM" the value was printed in
  .pages              source PDF page number(s)
  .as_of              issue the plan pinned (when the question named one), else None
  .requested_period   data window the value was retrieved for
  .retrieve_key       the concept this datum was retrieved for

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
the full axis labels (index labels + column names) and per-column dtypes — NOT
the cell values. The full frame is what you get in the exec env; write code
against it (`.loc[...]`, `.idxmax()`, etc.) using the labels shown.
Apply unit conversions once over the whole frame (e.g. `df * 1e-6`), never
cell-by-cell.

## Selecting inputs

- Pick the entries you need from the `input_values =` block and reference
  them by index (`input_values[7].frame`). Do not re-locate entries at
  runtime by filtering on `.description`.
- Entries may repeat. Pick the one whose description AND qualifiers best match
  the question's wording — a qualifier word in the question ("subject to
  limitation", "accepted", "issued", "total outstanding") must match the
  entry's qualifiers, not just its description.
  Do NOT use multiple entries for max/min/avg/sum/etc.
  
## Output format

Emit exactly one of the two forms below and nothing else — no prose,
no commentary, no second block, no fence around (b):

  (a) Success — a single fenced ```python``` block. Assign the final
      answer string to `result`. The string contains ONLY the requested
      answer — no prose, no "Answer:", no question restatement. For a
      multi-part question, `result` is only the ultimate quantity /
      identifier asked for, not any intermediate.
      Carry full precision through every intermediate; round or format only
      at the latest possible step — the final `result` string — to the
      decimal places the question states.
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
        user_msg = f"Question:\n{ctx.question}\n\n"
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


class ComputeOp:
    """The compute operator — a codegen → exec loop. One public `run()`."""
    def __init__(self) -> None:
        self._codegen = Codegen()

    async def run(
        self,
        input_values: list[AnnotatedValue],
        ctx: ExecutionContext,
        concept_explanations: Sequence[ConceptExplanation] = (),
    ) -> str:
        """Retry loop over `compute_max_attempts`: each iteration does codegen → exec,
        and any failure feeds the next attempt's `prev_failure`. Returns the result of
        the first clean exec. Raises `StepFailed` if no iteration ever execs cleanly;
        propagates `MissingData` (codegen's give-up signal) so the orchestrator can
        recover by gathering more data and replanning."""
        # No "starting" boundary emit — the orchestrator's trace records this
        # step's boundary; the plan is the planner step's output.

        prev_code: str | None = None
        prev_failure: str | None = None

        for try_idx in range(ctx.config.compute_max_attempts):
            try:
                code = await self._codegen.codegen(
                    ctx, input_values, prev_code, prev_failure,
                    concept_explanations,
                )
            except ParseError as e:
                prev_code = None
                prev_failure = e.detail
                ctx.emit(f"codegen_parse_failed attempt={try_idx + 1} detail={e.detail!r}")
                continue
            except MissingData as e:
                # Trust the signal: propagate so the orchestrator's recovery loop can
                # gather the missing data and replan.
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
            ctx.emit(f"exec_result attempt={try_idx + 1} text={result!r}")
            return result

        raise StepFailed(
            "compute",
            f"no successful exec within budget; last failure: {prev_failure}",
        )
