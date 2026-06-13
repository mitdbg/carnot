from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from skunk.common import Effort, strip_code_fence
from skunk.errors import ParseError, StepFailed
from skunk.prompted_call import PromptedCall
from skunk.common import (
    AnnotatedValue,
    ExecutionContext,
    Final,
    NeedsMore,
    input_values_desc,
)
from skunk.pyexec import exec_python_with_env
from skunk.question_explainer import ConceptExplanation


class MissingDataSignal(BaseModel):
    model_config = ConfigDict(frozen=True)
    description: str
    missing: list[str] = []


def _parse_codegen(raw: str) -> str | MissingDataSignal:
    """Parse a codegen reply into a Python code string (forms (a)/(c)) or a
    `MissingDataSignal` (form (b), a `{`-leading JSON object). Malformed JSON or
    empty code → `ParseError`. Not a `PromptedCall` parse hook — called after
    `call()` returns so the prompt layer never sees it; `ComputeOp.run()` catches
    the `ParseError` and retries."""
    # Strip fences first so `{`-dispatch works for both bare and ```json-wrapped JSON.
    s = strip_code_fence(raw).strip()

    if s.startswith("{"):
        try:
            return MissingDataSignal.model_validate_json(s)
        except ValidationError as e:
            raise ParseError(
                raw=raw,
                detail=(
                    f"malformed missing-data JSON — {e}\n"
                    "Output ONLY a fenced ```python``` block (form (a) or (c)) OR a "
                    'bare JSON {"missing": [...], "description": "..."} object (form (b)).'
                ),
            )

    if not s:
        raise ParseError(
            raw=raw,
            detail=(
                "empty or un-fenced code.\n"
                "Output ONLY a fenced ```python``` block (form (a) or (c)) OR a "
                'bare JSON {"missing": [...], "description": "..."} object (form (b)).'
            ),
        )
    return s


def _coerce_prim(v: Any) -> Any:
    """Unwrap numpy scalars (the usual product of pandas arithmetic) so committed
    values pass `AnnotatedValue`'s primitive-cell validation."""
    return v.item() if isinstance(v, np.generic) else v


def _needs_more_from_env(
    env: dict[str, Any], input_values: list[AnnotatedValue], round_idx: int
) -> NeedsMore:
    """Validate a partial-progress (form (c)) exec environment into a `NeedsMore`.
    `missing` must validate as `MissingDataSignal`; `keep` is REQUIRED — the explicit
    whitelist of entries that survive into the next round (everything else is dropped);
    `committed` values are wrapped as provenance-free `AnnotatedValue`s whose
    description marks them as computed intermediates. Raises `ValueError` with a
    fix-it detail on any malformed shape — fed back to the next codegen attempt
    via `prev_failure`."""
    try:
        signal = MissingDataSignal.model_validate(env["missing"])
    except ValidationError as e:
        raise ValueError(
            f'`missing` must be {{"missing": [...], "description": "..."}} — {e}'
        )

    if "keep" not in env:
        raise ValueError(
            "`keep` is required for partial progress: the explicit list of "
            "input_values indices to carry into the next round — everything not "
            "listed is dropped. Use [] to carry nothing forward."
        )
    keep = env["keep"]
    if not (
        isinstance(keep, list)
        and all(isinstance(i, int) and not isinstance(i, bool) for i in keep)
    ):
        raise ValueError("`keep` must be a list of ints (indices into input_values)")
    bad = [i for i in keep if not 0 <= i < len(input_values)]
    if bad:
        raise ValueError(
            f"`keep` indices out of range for input_values"
            f"[0..{len(input_values) - 1}]: {bad!r}"
        )

    committed_raw = env.get("committed", {})
    if not isinstance(committed_raw, dict) or not all(
        isinstance(k, str) for k in committed_raw
    ):
        raise ValueError("`committed` must be a dict keyed by str names")
    committed: list[AnnotatedValue] = []
    for name, v in committed_raw.items():
        desc = f"computed intermediate (round {round_idx}): {name}"
        try:
            if isinstance(v, dict):
                payload = {str(k): _coerce_prim(c) for k, c in v.items()}
                entry = AnnotatedValue(
                    description=desc, value=payload, kind="vector", index_name="label"
                )
            else:
                payload = (
                    [_coerce_prim(c) for c in v]
                    if isinstance(v, list)
                    else _coerce_prim(v)
                )
                entry = AnnotatedValue(description=desc, value=payload, kind="scalar")
        except ValidationError as e:
            raise ValueError(
                f"committed[{name!r}]: value must be a scalar, list of scalars, or "
                f"flat {{label: scalar}} dict — {e}"
            )
        committed.append(entry)
    return NeedsMore(
        keep=keep,
        committed=committed,
        missing_reason=signal.description,
        missing=signal.missing,
    )


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
  .notes         prose page context bearing on the question — footnotes,
                 headnotes, caveats, print-flag meanings (p/r); shared
                 across a vector/table's cells, so it is context not a
                 per-cell discriminator (that lives in .description)
  .frame         pd.DataFrame view of the payload (uniform across kinds)
  .unit          natural-language unit, e.g. "millions of dollars", "percent"
  .kind          "scalar" | "vector" | "table"  (rarely needed; prefer .frame)
  .index_name    (vector)   .row_name / .col_name (table)
  .value         raw payload — only use if you specifically need the dict/list form
  .bulletin           source issue "YYYY-MM" the value was printed in
  .pages              source PDF page number(s)
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

The `input_values =` block below shows each non-scalar entry's schema (axis
labels and dtypes, not cell values); the full frame exists in the exec
environment. Apply unit conversions once over the whole frame, never
cell-by-cell.

## Selecting inputs

- Reference entries by index (`input_values[7].frame`); do not re-locate them
  at runtime by filtering on `.description`.
- Entries may repeat. Pick the one whose description (and notes) match the
  question's wording, including its qualifier words ("subject to limitation",
  "accepted", "issued"). Do not combine multiple entries for max/min/avg/sum.
- Extra scope words in a description or its notes mark a different, broader
  series, not a looser label for the asked one: a question about "public debt"
  is not answered by "public debt and guaranteed obligations"; "savings bonds"
  is not "savings bonds and savings notes". Match the question's exact series
  even when the broader entry is more convenient to read (printed total row,
  fuller index) — convenience of access never outweighs a series mismatch.
- Pay attention to notes when chaining entries that cover adjacent
  sub-periods of one series: reprints of the same table tile cleanly, but
  entries whose notes name different tables usually define the series
  differently, and a value assembled across them drifts. Prefer covering the
  period from one table; when only a cross-table patchwork can cover it,
  weigh signaling missing data instead.

## Output format

Emit exactly one of the three forms below and nothing else — no prose,
no commentary, no second block, no fence around (b):

  (a) Success — a single fenced ```python``` block. Assign the final
      answer string to `result`. The string contains only the requested
      answer — no prose, no "Answer:", no question restatement; for a
      multi-part question, only the ultimate quantity asked for.
      Carry full precision through every intermediate; round or format
      only in the final `result` string, to the decimal places the
      question states.
  (b) Insufficient data, no computable progress — a single bare JSON object:
        {"missing": [<short identifier strings>],
         "description": "<one-line explanation>"}
      If a required value is not in `input_values`, never fabricate it.
      Real-world reference data (exchange rates, deflators, CPI, GDP,
      population, market prices) is data, not knowledge: if no entry
      carries it, signal missing rather than supplying it from memory.
  (c) Partial progress — a single fenced ```python``` block that does
      NOT assign `result` and instead assigns:
        missing   = {"missing": [...], "description": "..."}    # required
        committed = {name: scalar or flat {label: scalar} dict} # optional
        keep      = [indices into input_values to retain]       # required
      Use (c) when computation over the gathered values narrows what is
      missing — e.g. derive the qualifying month from a gathered series,
      commit it, and name only that month's value as missing. Committed
      values come back next round as new input_values entries; name them
      self-descriptively, unit included. `keep` is the explicit whitelist
      for the next round: every entry NOT listed is dropped for good.
      Keep exactly the entries you will combine with the missing data
      (committed values carry forward automatically); keep = [] carries
      nothing forward.
      The fabrication rule of (b) applies: commit only values COMPUTED
      from input_values, never from memory.

Available imports: numpy (np), pandas (pd), math, statsmodels.api (sm).
"""

    _prompt = PromptedCall(
        name="compute.codegen",
        system_prompt=_SYSTEM_PROMPT,
        default_effort="high",
        output_instruction=(
            "Produce a fenced ```python``` block (assigning `result`, or `missing` for "
            "partial progress) OR a bare missing-data JSON object."
        ),
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
    ) -> str | MissingDataSignal:
        """One LLM call returning the generated code, or the form-(b) give-up signal.
        Raises `ParseError` (caller retries with the detail echoed back).
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
        raw = await self._prompt.call(ctx, user_msg, effort=effort, temperature=1.0)
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
        *,
        round_idx: int = 0,
    ) -> Final | NeedsMore:
        """Best-of-N codegen for this compute call: run `compute_best_of_n` independent
        codegen→exec trials in parallel and vote on the outcome (see `_vote`). Each trial
        is a `_run_trial` retry loop returning `Final | NeedsMore` (or raising
        `StepFailed`). N≤1 runs a single trial — today's behavior. Raises `StepFailed`
        only if every trial does."""
        # No "starting" boundary emit — the orchestrator's trace records this
        # step's boundary; the plan is the planner step's output.

        # Source pages behind the values reaching compute (post-retry/replan) — the
        # final-stage survivor set for per-stage recall (eval/stage_report.py). Shared
        # across trials, so emitted once here rather than per trial.
        src_pages = sorted(
            {f"{e.bulletin}:{p}" for e in input_values if e.bulletin for p in e.pages}
        )
        ctx.emit(
            f"compute_inputs n_values={len(input_values)} n_pages={len(src_pages)}",
            data={"pages": src_pages},
        )

        n = ctx.config.compute_best_of_n
        if n <= 1:
            return await self._run_trial(
                input_values, ctx, concept_explanations,
                round_idx=round_idx, trial_idx=0,
            )

        results = await asyncio.gather(
            *(
                self._run_trial(
                    input_values, ctx, concept_explanations,
                    round_idx=round_idx, trial_idx=i,
                )
                for i in range(n)
            ),
            return_exceptions=True,
        )
        return self._vote(results, ctx)

    def _vote(
        self, results: Sequence[Final | NeedsMore | BaseException], ctx: ExecutionContext
    ) -> Final | NeedsMore:
        """Commit the majority answer across best-of-N trials. `Final` answers vote
        (most frequent wins; ties broken by first-seen via `Counter.most_common`).
        `NeedsMore` outcomes abstain — one is returned only when NO trial produced a
        `Final`. If every trial raised `StepFailed`, re-raise the first; any other
        exception (programming error / cancellation) is re-raised immediately."""
        finals: list[Final] = []
        needs: list[NeedsMore] = []
        failures: list[StepFailed] = []
        for r in results:
            if isinstance(r, Final):
                finals.append(r)
            elif isinstance(r, NeedsMore):
                needs.append(r)
            elif isinstance(r, StepFailed):
                failures.append(r)
            elif isinstance(r, BaseException):
                raise r

        if finals:
            counts = Counter(f.answer for f in finals)
            winner, votes = counts.most_common(1)[0]
            ctx.emit(
                f"compute_vote n_trials={len(results)} n_final={len(finals)} "
                f"n_needs_more={len(needs)} n_failed={len(failures)} "
                f"winner_votes={votes} answer={winner!r}",
                data={"counts": dict(counts)},
            )
            return next(f for f in finals if f.answer == winner)
        if needs:
            ctx.emit(
                f"compute_vote_needs_more n_trials={len(results)} "
                f"n_needs_more={len(needs)} n_failed={len(failures)} "
                f"missing={needs[0].missing!r} description={needs[0].missing_reason!r}"
            )
            return needs[0]
        ctx.emit(f"compute_vote_all_failed n_trials={len(results)}")
        raise failures[0]

    async def _run_trial(
        self,
        input_values: list[AnnotatedValue],
        ctx: ExecutionContext,
        concept_explanations: Sequence[ConceptExplanation],
        *,
        round_idx: int,
        trial_idx: int,
    ) -> Final | NeedsMore:
        """One codegen→exec retry loop over `compute_max_attempts`: each iteration does
        codegen → exec, and any failure feeds the next attempt's `prev_failure`. Returns
        `Final` on the first clean exec that set `result`, or `NeedsMore` when codegen
        gives up outright (form (b)) or executed code commits partial progress (form
        (c)). Raises `StepFailed` if no iteration ever resolves. `trial_idx` tags every
        emit so interleaved best-of-N trials stay attributable in the trace."""
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
                ctx.emit(
                    f"codegen_parse_failed trial={trial_idx} attempt={try_idx + 1} "
                    f"detail={e.detail!r}"
                )
                continue
            if isinstance(code, MissingDataSignal):
                # Trust the give-up signal (form (b)): keep everything, commit nothing.
                ctx.emit(
                    f"compute_needs_more trial={trial_idx} attempt={try_idx + 1} "
                    f"n_keep={len(input_values)} n_committed=0 "
                    f"missing={code.missing!r} description={code.description!r}"
                )
                return NeedsMore(
                    keep=list(range(len(input_values))),
                    committed=[],
                    missing_reason=code.description,
                    missing=code.missing,
                )

            ctx.emit(f"codegen_code trial={trial_idx} attempt={try_idx + 1} code={code!r}")
            try:
                env, _ = exec_python_with_env(
                    code, {"input_values": input_values}, require_result=False
                )
            except Exception as e:
                prev_code = code
                prev_failure = f"Exception during exec: {e}"
                ctx.emit(
                    f"exec_failed trial={trial_idx} attempt={try_idx + 1} "
                    f"error={str(e)!r}"
                )
                continue

            if "result" in env:
                result = str(env["result"])
                ctx.emit(
                    f"exec_result trial={trial_idx} attempt={try_idx + 1} text={result!r}"
                )
                return Final(result)
            if "missing" in env:
                try:
                    needs = _needs_more_from_env(env, input_values, round_idx)
                except ValueError as e:
                    prev_code = code
                    prev_failure = f"partial-progress block malformed: {e}"
                    ctx.emit(
                        f"compute_partial_malformed trial={trial_idx} "
                        f"attempt={try_idx + 1} detail={str(e)!r}"
                    )
                    continue
                ctx.emit(
                    f"compute_needs_more trial={trial_idx} attempt={try_idx + 1} "
                    f"n_keep={len(needs.keep)} n_committed={len(needs.committed)} "
                    f"missing={needs.missing!r} description={needs.missing_reason!r}"
                )
                return needs
            prev_code = code
            prev_failure = (
                "code set neither `result` nor `missing` — emit form (a), (b), or (c)"
            )
            ctx.emit(f"exec_no_output trial={trial_idx} attempt={try_idx + 1}")

        raise StepFailed(
            "compute",
            f"no successful exec within budget; last failure: {prev_failure}",
        )
