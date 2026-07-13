from __future__ import annotations

import asyncio
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from skunk.common import Effort
from skunk.errors import ParseError, StepFailed
from skunk.prompted_call import PromptedCall
from skunk.common import (
    AnnotatedValue,
    ExecutionContext,
    Final,
    NeedsMore,
    RetrievedDoc,
    documents_desc,
    input_values_desc,
    split_pool,
)
from skunk.sandbox.pyexec import exec_python_with_env, parse_codegen_reply


class MissingDataSignal(BaseModel):
    model_config = ConfigDict(frozen=True)
    description: str
    missing: list[str] = []


# `parse_codegen_reply`'s fix-it text for compute replies. Not a `PromptedCall` parse
# hook — parsed after `call()` returns so the prompt layer never sees it;
# `ComputeOp.run()` catches the `ParseError` and retries.
_CODEGEN_EXPECTATION = (
    "expected a fenced ```python``` block — emit (a) success (assign `result`) "
    "or (b) missing data (assign `missing`). Do not emit a bare JSON object."
)


# A standalone NaN/inf token in the answer string means a non-finite `result`
# (str(float("nan")) == "nan", and f-strings format nan/inf into composite answers like
# "[nan, nan]"). Word-boundary guards keep it from tripping on substrings of real answers
# ("Nanjing", "infants"); a bare "infinity" in an answer is itself non-finite, so catching
# it is correct.
_NONFINITE_RE = re.compile(
    r"(?<![A-Za-z0-9])[+-]?(nan|inf(?:inity)?)(?![A-Za-z0-9])", re.IGNORECASE
)


def _needs_more_from_env(env: dict[str, Any]) -> NeedsMore:
    """Validate a missing-data exec environment into a `NeedsMore`. `missing` must validate
    as `MissingDataSignal` (`{"missing": [...], "description": "..."}`). Raises `ValueError`
    with a fix-it detail on a malformed shape. The orchestrator keeps everything gathered so
    far, so there is no per-value carry — the model only signals what is still missing."""
    try:
        signal = MissingDataSignal.model_validate(env["missing"])
    except ValidationError as e:
        raise ValueError(
            f'`missing` must be {{"missing": [...], "description": "..."}} — {e}'
        )
    return NeedsMore(missing_reason=signal.description, missing=signal.missing)


class Codegen:
    _SYSTEM_PROMPT = """\
You answer a question from a set of retrieved corpus pages, using a Python interpreter for any arithmetic. You either produce the final answer string, or — when the pages are not enough — signal what is still missing.

## Inputs
- The user's question — the authoritative statement of what to answer,
  including the output format (units, decimal places, list/bracket shape).
- Optional `## Concept references` section: one block per non-obvious
  concept the question references. If the question itself pins a specific variant of an operation,
  use that variant instead of the reference; otherwise follow the reference.
- `retrieved pages`: the corpus pages a search agent judged relevant, as plain
  text. READ THEM DIRECTLY. To compute over them, copy the relevant numbers out
  of the page text into your code as literals, then compute — do NOT do arithmetic
  in your head. The pages are text only; they are NOT variables in the exec environment.
- `input_values` (only when external lookups ran): list[AnnotatedValue] present in
  the exec environment — values a `lookup_external` agent pulled from outside the
  corpus. Read each through `e.frame` (a pd.DataFrame); `e.description`/`e.unit`/
  `e.source` label it. Reference by index (`input_values[i].frame`).

## Selecting data

- Match the question's EXACT series, including its qualifier words ("subject to
  limitation", "accepted", "issued", "net" vs "gross"). Extra scope words mark a
  different, broader series, not a looser label: "public debt" is not answered by
  "public debt and guaranteed obligations"; "savings bonds" is not "savings bonds
  and savings notes". A convenient printed total never outweighs a series mismatch.
- Watch for footnotes, headnotes, print flags (p/r = preliminary/revised), and
  "n/a" cells in the page text — they change what a number means.

## Output format

Emit exactly one of the two forms below as a single fenced ```python``` block
and nothing else — no prose, no commentary, no second block:

  (a) Success — assign the final answer string to `result`: exactly what the
      question asks for, formatted as it requests (units, decimal places,
      list/bracket shape, every part in order) — no prose, no "Answer:", no
      question restatement. Carry full precision through every intermediate;
      round or format only in the final `result` string.
      Never emit a non-finite answer: if a computation yields NaN or infinity —
      "n/a" cells, or division by an empty/zero quantity — do not format it into
      `result`. Drop or skip those cells before aggregating; if the pages
      genuinely cannot support a finite answer, use form (b) missing instead.

  (b) Missing data — when you cannot finish from the retrieved pages (and any
      `input_values`) alone. Do NOT assign `result`; instead assign:
        missing = {"missing": [<short identifier strings>], "description": "<one-line reason>"}
      First do as much as you can, then signal what is still needed: the short
      identifiers + a one-line reason. Never fabricate: use only values present in
      the pages / `input_values` or COMPUTED from them, never from memory.
      Real-world reference data (exchange rates, deflators, CPI, GDP, population,
      market prices) is data, not knowledge — if no page carries it, list it under
      `missing` rather than supplying it.

      Rules for signaling:
      - Never signal missing data because a page number differs from one named in
        the question — the retrieved pages are what you have.
      - DO signal missing data when the supplied data does NOT align with what the
        question asks for (reported for a different date than asked, or a different
        source/series), and clearly state this in your signal.

Available imports: numpy (np), pandas (pd), math, statsmodels.api (sm).
"""

    _prompt = PromptedCall(
        name="compute.codegen",
        system_prompt=_SYSTEM_PROMPT,
        default_effort="high",
        output_instruction=(
            "Produce a fenced ```python``` block — assign `result` for the answer, "
            "or `missing` when the pages are insufficient."
        ),
    )

    async def codegen(
        self,
        ctx: ExecutionContext,
        documents: list[RetrievedDoc],
        input_values: list[AnnotatedValue],
        prev_code: str | None,
        prev_failure: str | None,
        *,
        effort: Effort | None = None,
    ) -> str:
        """One LLM call returning the generated code (form (a) success or form (b) missing
        data). Raises `ParseError` (caller retries with the detail echoed back).
        `prev_code`/`prev_failure` describe only the most-recent failed attempt —
        accumulating older ones dilutes the issue to fix."""
        user_msg = f"Question:\n{ctx.question}\n\n"
        user_msg += documents_desc(documents)
        if input_values:
            user_msg += f"\n\ninput_values =\n{input_values_desc(input_values)}"
        if prev_failure:
            user_msg += "\n\nYour previous attempt failed."
            if prev_code:
                user_msg += f"\nPrevious code:\n```python\n{prev_code}\n```"
            user_msg += (
                f"\nSpecific issue(s) to address:\n{prev_failure}\n"
                "Focus on fixing them without introducing new mistakes."
            )
        raw = await self._prompt.call(ctx, user_msg, effort=effort, temperature=1.0)
        return parse_codegen_reply(raw, expectation=_CODEGEN_EXPECTATION)


class ComputeOp:
    """The compute operator — a codegen → exec loop. One public `run()`."""
    def __init__(self) -> None:
        self._codegen = Codegen()

    async def run(
        self,
        pool: list,
        ctx: ExecutionContext,
        *,
        round_idx: int = 0,
    ) -> Final | NeedsMore:
        """Best-of-N codegen for this compute call: run `compute_best_of_n` independent
        codegen→exec trials in parallel and vote on the outcome (see `_vote`). `pool` is
        the orchestrator's mixed gathered set — retrieved pages (`RetrievedDoc`) plus any
        `lookup_external` values (`AnnotatedValue`); it is split into the codegen prompt's
        `retrieved pages` and `input_values`. Each trial is a `_run_trial` retry loop
        returning `Final | NeedsMore` (or raising `StepFailed`). N≤1 runs a single trial.
        Raises `StepFailed` only if every trial does."""
        # No "starting" boundary emit — the orchestrator's trace records this
        # step's boundary; the plan is the planner step's output.
        documents, input_values = split_pool(pool)

        # Source pages behind the content reaching compute (post-retry/replan) — the
        # final-stage survivor set for per-stage recall (eval/stage_report.py). Shared
        # across trials, so emitted once here rather than per trial.
        src_pages = sorted(
            {f"{d.ref.stem}:{d.ref.page}" for d in documents if d.ref.stem}
        )
        ctx.emit(
            f"compute_inputs n_pages={len(src_pages)} n_values={len(input_values)}",
            data={"pages": src_pages},
        )

        n = ctx.config.compute_best_of_n
        if n <= 1:
            return await self._run_trial(
                documents, input_values, ctx,
                round_idx=round_idx, trial_idx=0,
            )

        results = await asyncio.gather(
            *(
                self._run_trial(
                    documents, input_values, ctx,
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
        """Vote across best-of-N trials. Each distinct `Final` answer competes with a single
        pooled MISSING-DATA candidate — all `NeedsMore` trials count equally toward it. Most
        frequent wins; a tie NEVER breaks in favor of missing-data (an actual answer beats a
        give-up at equal votes). When missing-data wins, the `NeedsMore` trials are combined:
        their `missing` identifiers are UNIONed and the reason is rendered per trial
        ("agent 1: …, agent 2: …"). If every trial raised `StepFailed`, re-raise the first;
        any other exception (programming error / cancellation) is re-raised immediately."""
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

        if not finals and not needs:
            ctx.emit(f"compute_vote_all_failed n_trials={len(results)}")
            raise failures[0]

        final_counts = Counter(f.answer for f in finals)
        best_answer, best_votes = (
            final_counts.most_common(1)[0] if final_counts else (None, 0)
        )
        # Missing-data wins only by a STRICT majority over the top answer — at a tie the
        # answer wins (never break in favor of missing-data).
        if best_answer is not None and len(needs) <= best_votes:
            ctx.emit(
                f"compute_vote n_trials={len(results)} n_final={len(finals)} "
                f"n_needs_more={len(needs)} n_failed={len(failures)} "
                f"winner_votes={best_votes} answer={best_answer!r}",
                data={"counts": dict(final_counts)},
            )
            return next(f for f in finals if f.answer == best_answer)

        # Missing-data wins (or no trial finalized): combine every `NeedsMore` trial. UNION
        # their missing identifiers and render the reasons per agent.
        missing = list(dict.fromkeys(m for n in needs for m in n.missing))
        reason = "\n".join(
            f"agent {i}: {n.missing_reason}" for i, n in enumerate(needs, 1)
        )
        ctx.emit(
            f"compute_vote_needs_more n_trials={len(results)} n_final={len(finals)} "
            f"n_needs_more={len(needs)} n_failed={len(failures)} "
            f"best_final_votes={best_votes} missing={missing!r}"
        )
        return NeedsMore(missing_reason=reason, missing=missing)

    async def _run_trial(
        self,
        documents: list[RetrievedDoc],
        input_values: list[AnnotatedValue],
        ctx: ExecutionContext,
        *,
        round_idx: int,
        trial_idx: int,
    ) -> Final | NeedsMore:
        """One codegen→exec retry loop over `compute_max_attempts`: each iteration does
        codegen → exec, and any failure feeds the next attempt's `prev_failure`. Returns
        `Final` on the first clean exec that set `result`, or `NeedsMore` when codegen
        gives up outright (form (b)). Raises `StepFailed` if no iteration ever resolves.
        `trial_idx` tags every emit so interleaved best-of-N trials stay attributable in
        the trace. Only `input_values` (lookup values) enter the exec env; the retrieved
        pages are text in the prompt and the model transcribes numbers into its code."""
        prev_code: str | None = None
        prev_failure: str | None = None

        for try_idx in range(ctx.config.compute_max_attempts):
            try:
                code = await self._codegen.codegen(
                    ctx, documents, input_values, prev_code, prev_failure,
                )
            except ParseError as e:
                prev_code = None
                prev_failure = e.detail
                ctx.emit(
                    f"codegen_parse_failed trial={trial_idx} attempt={try_idx + 1} "
                    f"detail={e.detail!r}"
                )
                continue

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
                if _NONFINITE_RE.search(result):
                    prev_code = code
                    prev_failure = (
                        f"`result` is non-finite ({result!r}). A NaN/inf answer means the "
                        "computation is undefined — typically NaN cells in the inputs "
                        "(e.g. 'n/a'), or division by an empty/zero quantity. Never emit "
                        "nan/inf as the answer: drop or skip the NaN cells before "
                        "aggregating, or — if the inputs genuinely cannot support the "
                        "answer — emit form (b) missing instead."
                    )
                    ctx.emit(
                        f"exec_result_nonfinite trial={trial_idx} "
                        f"attempt={try_idx + 1} text={result!r}"
                    )
                    continue
                ctx.emit(
                    f"exec_result trial={trial_idx} attempt={try_idx + 1} text={result!r}"
                )
                return Final(result)
            if "missing" in env:
                try:
                    needs = _needs_more_from_env(env)
                except ValueError as e:
                    prev_code = code
                    prev_failure = f"missing-data block malformed: {e}"
                    ctx.emit(
                        f"compute_partial_malformed trial={trial_idx} "
                        f"attempt={try_idx + 1} detail={str(e)!r}"
                    )
                    continue
                ctx.emit(
                    f"compute_needs_more round={round_idx} trial={trial_idx} "
                    f"attempt={try_idx + 1} "
                    f"missing={needs.missing!r} description={needs.missing_reason!r}"
                )
                return needs
            prev_code = code
            prev_failure = (
                "code set neither `result` nor `missing` — emit form (a) or (b)"
            )
            ctx.emit(f"exec_no_output trial={trial_idx} attempt={try_idx + 1}")

        raise StepFailed(
            "compute",
            f"no successful exec within budget; last failure: {prev_failure}",
        )
