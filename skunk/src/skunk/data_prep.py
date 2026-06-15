"""Data-prep gate (prototype): a Flash "thinking" pass that cleans the union of extracted
`AnnotatedValue`s before they reach compute.

It mirrors the compute operator's codegen→exec shape: one LLM call emits a fenced
```python``` block that builds `result` — the cleaned list — over `input_values`, then the
host execs it in the same sandbox. The code never constructs `AnnotatedValue`s itself
(pydantic construction is unfriendly to the AST-walked sandbox); instead each `result`
element is a plain dict of value fields the host validates into a fresh `AnnotatedValue`.
The agent emits dicts (not `input_values[i]` references) because it cleans labels/typos and
coalesces overlapping prints — the kept value is rarely the original object — and it authors
each value's provenance (bulletin/pages/...) too: copied from the source entry, or UNIONed
across sources on a coalesce, so source-tracing and vintage selection downstream survive.

The op fails safe: on any parse / exec / validation error after the attempt budget, the
original pool passes through unchanged. Cleaning is an optimization, not a correctness
requirement, so a bad data-prep run must never starve compute of its inputs.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from skunk.common import (
    AnnotatedValue,
    Effort,
    ExecutionContext,
    input_values_desc,
)
from skunk.errors import ParseError
from skunk.prompted_call import PromptedCall
from skunk.pyexec import exec_python_with_env, strip_code_fences

# The agent emits each kept value as a full dict (it may have cleaned labels/typos or
# coalesced prints, so it can't pass the original object through). It authors provenance
# (bulletin/pages/...) too — copied from the source entry, or UNIONed across sources when it
# coalesces — so here, unlike elsewhere, provenance is LLM-written rather than machine-stamped.
# Allowed keys are exactly `AnnotatedValue`'s fields (kept in sync with the model); any other
# key is a typo and rejected.
_VALUE_FIELDS = set(AnnotatedValue.model_fields)


def _parse_codegen(raw: str) -> str:
    """A fenced ```python``` block is required. Empty / bare-JSON → `ParseError` (the caller
    retries with the detail echoed back)."""
    s = strip_code_fences(raw).strip()
    if not s or s.startswith("{"):
        raise ParseError(
            raw=raw,
            detail=(
                "expected a single fenced ```python``` block assigning `result` — a list of "
                "dicts, one per kept value. Do not emit a bare JSON object or prose."
            ),
        )
    return s


def _result_from_env(env: dict[str, Any]) -> list[AnnotatedValue]:
    """Validate the exec environment's `result` into a `list[AnnotatedValue]`. Each entry is a
    dict of value fields (including provenance copied from its source) validated into a fresh
    `AnnotatedValue`; a bare `input_values[i]` object is also accepted and carried verbatim.
    Raises `ValueError` with a fix-it detail on any malformed shape."""
    result = env.get("result")
    if not isinstance(result, list):
        raise ValueError("`result` must be a list (assign `result = [...]`).")
    out: list[AnnotatedValue] = []
    for i, item in enumerate(result):
        if isinstance(item, AnnotatedValue):
            out.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError(
                f"result[{i}]: each entry must be a dict of value fields (or a bare "
                f"`input_values[...]` object), got {type(item).__name__}."
            )
        extra = set(item) - _VALUE_FIELDS
        if extra:
            raise ValueError(
                f"result[{i}]: unknown field(s) {sorted(extra)} — a value dict may only set "
                f"{sorted(_VALUE_FIELDS)}."
            )
        try:
            out.append(AnnotatedValue.model_validate(item))
        except ValidationError as e:
            raise ValueError(f"result[{i}]: invalid AnnotatedValue dict — {e}")
    return out


class DataPrep:
    _SYSTEM_PROMPT = """\
You clean a pool of already-extracted data values before it is handed to a downstream compute step. You do NOT answer the question, compute anything, or look anything up — your only job is to return a tidier version of the SAME values.

## Inputs
- The question, for context on which values are about to be used and how.
- `input_values`: list[AnnotatedValue] in the exec environment — the union of everything previous extract/lookup steps gathered. Entries frequently repeat: the same table is reprinted across many monthly bulletins, so the same series+period value appears again and again, sometimes with identical and sometimes with slightly different `description`/`notes`.

## AnnotatedValue API (read-only)
  .description   natural-language label distinguishing this datum from its siblings
  .value         raw payload (scalar | vector dict | table dict)
  .frame         pd.DataFrame view of the payload (uniform across kinds)
  .unit          natural-language unit, e.g. "millions of dollars", "percent"
  .notes         prose page context — footnotes, caveats, print-flag (p/r) meaning
  .kind          "scalar" | "vector" | "table"
  .bulletin      source issue "YYYY-MM" the value was printed in (publication date)
  .pages         source PDF page number(s)
  .requested_period / .retrieve_key   the data window / concept this datum served

## What to do
- REMOVE duplicates: when two entries are the same series, same period, and EXACT same value, keep one.
- REMOVE irrelevant data: when a piece of data is clearly not what the question asks for -- remove it 
- COALESCE overlapping ranges: sometimes the requested time period is reported across different prints in overlapping ranges.
  Coalesce them, remove the duplicates, and present one complete, contiguous series. Be careful to only coalesce data from
  the same series with the same accounting and reporting methods. When overlapping numbers disagree, replace any provisional/estimates
  with the later actual values, unless the question specifically requests data of a particular version. Other disagreement may be caused by OCR errors, etc. Use your best judgement.
- CLEAN data and remove any formatting issues, fix inconsistent labels, typos.
- PRESERVE notes and other important description of the data.
- PRESERVE each value's shape: keep the same `kind` and payload structure as the input (scalar primitive, vector `dict[str, scalar]`, table `dict[str, dict[str, scalar]]` — never re-nest, wrap, or restructure it).
- NEVER fabricate, compute, convert units, or pull in outside data. Every value you emit must already be present in `input_values` (verbatim) or be a faithful coalescing of input entries.

## Output
Emit exactly one fenced ```python``` block assigning `result` — the cleaned list — and nothing else (no prose, no second block). `result` is a list of dicts, ONE PER value you are keeping. Build each dict yourself (read the source data from `input_values[i].value`) — you are cleaning and coalescing, so emit the actual value, never an `input_values[i]` reference:
  {"description": "...", "value": <scalar | vector dict | table dict>, "unit": "...",
   "notes": "...", "kind": "scalar"|"vector"|"table",
   "index_name"/"row_name"/"col_name": ... as the kind requires,
   "bulletin": "YYYY-MM", "pages": [<int>, ...],
   "requested_period": "...", "retrieve_key": "..."}
Always fill the provenance fields (`bulletin`, `pages`, `requested_period`, `retrieve_key`) yourself, from the input entry/entries a value came from — downstream compute reads them to sort by publication date and trace sources. For a value kept from ONE entry, copy them across. For a value you coalesced from SEVERAL entries, UNION them:
  - `pages`: every source page combined (deduped).
  - `bulletin`: the issues the data was drawn from — a single "YYYY-MM", or "<earliest>..<latest>" when it spans several.
  - `requested_period`: the combined period the merged value now covers.
  - `retrieve_key`: the shared concept key (identical across true coalesce candidates).
Reproduce the data exactly — your only changes are removing duplicates, coalescing, and fixing labels/typos.
"""

    _prompt = PromptedCall(
        name="data_prep.codegen",
        system_prompt=_SYSTEM_PROMPT,
        default_effort="medium",
        output_instruction=(
            "Produce one fenced ```python``` block assigning `result` (a list of value "
            "dicts, one per kept value)."
        ),
    )

    async def codegen(
        self,
        ctx: ExecutionContext,
        input_values: list[AnnotatedValue],
        prev_code: str | None,
        prev_failure: str | None,
        *,
        effort: Effort | None = None,
    ) -> str:
        """One LLM call returning the cleaning code. Raises `ParseError` (caller retries
        with the detail echoed back). `prev_code`/`prev_failure` describe only the most
        recent failed attempt."""
        user_msg = (
            f"Question:\n{ctx.question}\n\n"
            f"input_values =\n{input_values_desc(input_values)}"
        )
        if prev_failure:
            user_msg += "\n\nYour previous attempt failed."
            if prev_code:
                user_msg += f"\nPrevious code:\n```python\n{prev_code}\n```"
            user_msg += (
                f"\nSpecific issue(s) to address:\n{prev_failure}\n"
                "Fix them without introducing new mistakes."
            )
        raw = await self._prompt.call(ctx, user_msg, effort=effort, temperature=0.4)
        return _parse_codegen(raw)


class DataPrepOp:
    """The data-prep gate — a codegen→exec pass that cleans `input_values` before compute.
    One public `run()`. Fails safe: returns the original pool on any unrecoverable error."""

    def __init__(self) -> None:
        self._codegen = DataPrep()

    async def run(
        self,
        input_values: list[AnnotatedValue],
        ctx: ExecutionContext,
    ) -> list[AnnotatedValue]:
        """Clean `input_values` and return the new pool. On a clean run emits a summary and
        returns the cleaned list; on exhausting the attempt budget emits a note and returns
        `input_values` unchanged (the gate never starves compute)."""
        if not input_values:
            return input_values

        prev_code: str | None = None
        prev_failure: str | None = None
        for try_idx in range(ctx.config.compute_max_attempts):
            try:
                code = await self._codegen.codegen(
                    ctx, input_values, prev_code, prev_failure
                )
            except ParseError as e:
                prev_code = None
                prev_failure = e.detail
                ctx.emit(
                    f"data_prep_parse_failed attempt={try_idx + 1} detail={e.detail!r}"
                )
                continue

            ctx.emit(f"data_prep_code attempt={try_idx + 1} code={code!r}")
            try:
                env, _ = exec_python_with_env(
                    code, {"input_values": input_values}, require_result=False
                )
            except Exception as e:
                prev_code = code
                prev_failure = f"Exception during exec: {e}"
                ctx.emit(
                    f"data_prep_exec_failed attempt={try_idx + 1} error={str(e)!r}"
                )
                continue

            try:
                cleaned = _result_from_env(env)
            except ValueError as e:
                prev_code = code
                prev_failure = f"`result` malformed: {e}"
                ctx.emit(
                    f"data_prep_result_malformed attempt={try_idx + 1} detail={str(e)!r}"
                )
                continue

            ctx.emit(
                f"data_prep_cleaned in={len(input_values)} out={len(cleaned)} "
                f"attempt={try_idx + 1}",
                data={"descriptions": [e.description for e in cleaned]},
            )
            return cleaned

        # Fail safe — never let a bad data-prep run drop compute's inputs.
        ctx.emit(
            f"data_prep_giveup attempts={ctx.config.compute_max_attempts} "
            f"last_failure={prev_failure!r} — passing the original pool through unchanged"
        )
        return input_values
