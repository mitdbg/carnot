from __future__ import annotations

import asyncio
import re
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


def _parse_codegen(raw: str) -> str:
    """Parse a codegen reply into a Python code string. Empty / un-fenced / bare-JSON →
    `ParseError`. Not a `PromptedCall` parse hook — called after `call()` returns so the
    prompt layer never sees it; `ComputeOp.run()` catches the `ParseError` and retries."""
    s = strip_code_fence(raw).strip()
    if not s or s.startswith("{"):
        raise ParseError(
            raw=raw,
            detail=(
                "expected a fenced ```python``` block — emit (a) success (assign `result`) "
                "or (b) missing data (assign `keep` and `missing`). Do not emit a bare "
                "JSON object."
            ),
        )
    return s


# A standalone NaN/inf token in the answer string means a non-finite `result`
# (str(float("nan")) == "nan", and f-strings format nan/inf into composite answers like
# "[nan, nan]"). Word-boundary guards keep it from tripping on substrings of real answers
# ("Nanjing", "infants"); a bare "infinity" in an answer is itself non-finite, so catching
# it is correct.
_NONFINITE_RE = re.compile(
    r"(?<![A-Za-z0-9])[+-]?(nan|inf(?:inity)?)(?![A-Za-z0-9])", re.IGNORECASE
)


def _coerce_prim(v: Any) -> Any:
    """Unwrap numpy scalars (the usual product of pandas arithmetic) so kept
    values pass `AnnotatedValue`'s primitive-cell validation."""
    return v.item() if isinstance(v, np.generic) else v


def _needs_more_from_env(env: dict[str, Any]) -> NeedsMore:
    """Validate a missing-data exec environment into a `NeedsMore`. `missing` must validate
    as `MissingDataSignal`. `keep` (optional, default `{}`) is the model's drop-by-default
    choice of what to carry into the next round, a `{name: value}` dict where each value is
    EITHER an `input_values` entry — carried verbatim so its provenance (bulletin/pages)
    survives — OR a freshly computed scalar / [scalars] / flat {label: scalar} dict,
    which becomes a provenance-free `AnnotatedValue` marked "computed". Anything not in `keep`
    is dropped. Raises `ValueError` with a fix-it detail on any malformed shape."""
    try:
        signal = MissingDataSignal.model_validate(env["missing"])
    except ValidationError as e:
        raise ValueError(
            f'`missing` must be {{"missing": [...], "description": "..."}} — {e}'
        )

    keep_raw = env.get("keep", {})
    if not isinstance(keep_raw, dict) or not all(isinstance(k, str) for k in keep_raw):
        raise ValueError(
            "`keep` must be a dict keyed by str names — each value is either an "
            "`input_values` entry (carried with its source) or a computed scalar / "
            "[scalars] / flat {label: scalar} dict. Omit or use {} to carry nothing."
        )
    keep: list[AnnotatedValue] = []
    for name, v in keep_raw.items():
        if isinstance(v, AnnotatedValue):
            # A re-stated input entry — carry it verbatim so provenance survives.
            keep.append(v)
            continue
        # Otherwise a value the model computed — provenance-free, marked "computed".
        try:
            if isinstance(v, dict):
                payload = {str(k): _coerce_prim(c) for k, c in v.items()}
                entry = AnnotatedValue(
                    description=name, value=payload, kind="vector",
                    index_name="label", notes="computed",
                )
            else:
                payload = (
                    [_coerce_prim(c) for c in v]
                    if isinstance(v, list)
                    else _coerce_prim(v)
                )
                entry = AnnotatedValue(
                    description=name, value=payload, kind="scalar", notes="computed"
                )
        except ValidationError as e:
            raise ValueError(
                f"keep[{name!r}]: value must be an `input_values` entry, a scalar, a "
                f"list of scalars, or a flat {{label: scalar}} dict — {e}"
            )
        keep.append(entry)
    return NeedsMore(
        keep=keep, missing_reason=signal.description, missing=signal.missing
    )


class Codegen:
    _SYSTEM_PROMPT = """\
You write Python that either produces the final answer string, or — when the inputs are not enough — keeps the values still worth using and signals what is missing.

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
  .source             publisher/origin of an external-lookup value (empty for corpus extracts)
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

The `input_values =` block below shows each non-scalar entry's full frame (axis
labels, dtypes, and every cell); the same frame also exists in the exec
environment. Read the cells to spot NaN / "n/a" values and handle them before
aggregating. Apply unit conversions once over the whole frame, never cell-by-cell.

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



## Output format

Emit exactly one of the two forms below as a single fenced ```python``` block
and nothing else — no prose, no commentary, no second block:

  (a) Success — assign the final answer string to `result`. The string contains
      only the requested answer — no prose, no "Answer:", no question
      restatement; for a multi-part question, only the ultimate quantity asked
      for. Carry full precision through every intermediate; round or format only
      in the final `result` string, to the decimal places the question states.
      Never emit a non-finite answer: if a computation yields NaN or infinity — NaN
      cells in the inputs ("n/a"), or division by an empty/zero quantity — do not
      format it into `result`. Drop or skip those cells before aggregating; if the
      inputs genuinely cannot support a finite answer, use form (b) missing instead.

  (b) Missing data — when you cannot finish from `input_values` alone. Do NOT
      assign `result`; instead assign both:
        keep    = {name: value}   # each value: an input_values entry OR a computed value
        missing = {"missing": [<short identifier strings>], "description": "<one-line reason>"}
      First do as much as you can, then signal. `keep` is drop-by-default —
      anything you do not put in it is dropped for good and never reaches the next
      round (use `keep = {}` only to start over):
        - to retain an input you will reuse, REFERENCE the whole entry so its
          source survives:  keep["..."] = input_values[i]   (NOT input_values[i].value)
        - for a value you derived, assign the raw number / [list] / {label: value}
          dict:  keep["..."] = <value>   (recorded as "computed", no source pages)
        - state what is missing: the short identifiers + a one-line reason.
      Name computed values self-descriptively, unit included. Never fabricate: keep
      only values present in `input_values` or COMPUTED from them, never from
      memory. Real-world reference data (exchange rates, deflators, CPI, GDP,
      population, market prices) is data, not knowledge — if no input carries it,
      list it under `missing` rather than supplying it.

      Rules for signaling:
      - Never signal missing data because an input's `.pages` differ from a page
        number named in the question. 
      - DO signal missing data if the supplied data does align with what the question asks for
        (e.g., data was reported on a different date than what the question asked for, or from a different source),
        and clearly state this in your signal,

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
    ) -> str:
        """One LLM call returning the generated code (form (a) success or form (b) missing
        data). Raises `ParseError` (caller retries with the detail echoed back).
        `prev_code`/`prev_failure` describe only the most-recent failed attempt —
        accumulating older ones dilutes the issue to fix."""
        user_msg = f"Question:\n{ctx.question}\n\n"
        # `concept_explanations` are the canonical references the QuestionExplainer
        # selected from PRECOMPUTED_CONCEPTS for this question (or the full catalog when
        # config.compute_precomputed_concept_refs bypasses selection).
        ref_blocks = [
            f"### {c.concept}\n{c.explanation}" for c in concept_explanations
        ]
        if ref_blocks:
            block = "\n\n".join(ref_blocks)
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
        """Vote across best-of-N trials. Each distinct `Final` answer competes with a single
        pooled MISSING-DATA candidate — all `NeedsMore` trials count equally toward it. Most
        frequent wins; a tie NEVER breaks in favor of missing-data (an actual answer beats a
        give-up at equal votes). When missing-data wins, the `NeedsMore` trials are combined:
        their `keep` values are UNIONed (nothing any trial asked to keep is dropped) and the
        reason is rendered per trial ("agent 1: …, agent 2: …"). If every trial raised
        `StepFailed`, re-raise the first; any other exception
        (programming error / cancellation) is re-raised immediately."""
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
        # their kept values (dedup identical ones) so nothing any trial chose to keep is
        # dropped, and render the reasons per agent.
        keep: list[AnnotatedValue] = []
        seen: set[str] = set()
        for n in needs:
            for e in n.keep:
                key = e.model_dump_json()
                if key not in seen:
                    seen.add(key)
                    keep.append(e)
        missing = list(dict.fromkeys(m for n in needs for m in n.missing))
        reason = "\n".join(
            f"agent {i}: {n.missing_reason}" for i, n in enumerate(needs, 1)
        )
        ctx.emit(
            f"compute_vote_needs_more n_trials={len(results)} n_final={len(finals)} "
            f"n_needs_more={len(needs)} n_failed={len(failures)} "
            f"best_final_votes={best_votes} n_keep={len(keep)} missing={missing!r}"
        )
        return NeedsMore(keep=keep, missing_reason=reason, missing=missing)

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
                    f"attempt={try_idx + 1} n_keep={len(needs.keep)} "
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
