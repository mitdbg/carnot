"""lookup_external operator — Gemini with Google Search.

The model replies with a single bare JSON object validated onto
`LookupResult` (value + unit + source). The `source` field is
parsed-and-discarded; Gemini's `grounding_titles` already constrain
which domain the answer came from, so we don't re-validate the token.

TODO: The FRED and BLS REST-API helpers (`_fetch_fred` / `_fetch_bls`) were temporarily removed
2026-05-17 — the date dispatch (len-based), the annual-mean reducer,
and the BLS monthly-only assumption were silently wrong for several
series classes. Recover from git history when the rest of this file
has been tightened and we're ready to put a typed wrapper around them.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, ValidationError

from skunk.errors import MissingData, StepFailed
from skunk.prompted_call import PromptedCall
from skunk.models import AnnotatedValue, HarnessContext
from skunk.plan import LookupBranch


class LookupResult(BaseModel):
    """One Gemini lookup reply. `value` is the publisher's printed figure
    (number, entity string, or list for multi-value requests), or `None`
    when the model could not find a value; `unit` is a short label;
    `source` is the publisher token (parsed-and-discarded at the call
    site)."""

    model_config = ConfigDict(frozen=True)
    value: float | int | str | list[float | int | str] | None
    unit: str = ""
    source: str = "none"


class LookupExternalPromptedCall(PromptedCall):
    name: str = "lookup_external"
    system_prompt: str = """\
You find data points from a reliable external source. You must
ground every answer in a publisher's page returned by Google Search —
never invent data.

## Input

The user message is a single bare JSON object:

  {"target": "<natural-language request for a single value>",
   "src":    "<natural-language description of required source | null>"}

`target` is a natural-language request for a single value (or a small
list of values in a fixed order). If 'src' is not null, you must strictly use data
 from the named source; otherwise, pick the most authoritative public site for the
figure — typically the issuing statistical agency.

## Output format

A single bare JSON object. No markdown fences. No prose.

  {"value":  <number | string | array of those>,
   "unit":   "<string>",
   "source": "<string>"}

## Field semantics

value     the publisher's full printed precision. JSON number for
          numeric answers, JSON string for entity names, JSON array
          for multiple values in request order. Return null if you are
          unable to find such a value.
unit      natural-language label for the value's scale and base,
          e.g. "millions of dollars", "percent", "year". Leave blank
          ("") if the value is not a measurement (e.g. a name or
          other string answer).
source    short token naming the publisher domain, e.g. "bls",
          "bankofengland".

## Procedure

(a) If `src` is present, search that site (use a `site:` filter or
    include the publisher name). Do not substitute a different
    aggregator even if you know it carries the same series.
(b) If `src` is null, pick the most authoritative public site and
    search there.
(c) Issue the search via your Google Search tool. Answering from
    training memory alone is not allowed — your response is accepted
    only if a grounding chunk from the publisher's site appears.
(d) Read the snippets and extract the value at the publisher's full
    printed precision. Never round, format, or simplify.
"""

    def run(self, ctx: HarnessContext, branch: LookupBranch) -> list[AnnotatedValue]:
        user_msg = branch.model_dump_json(
            include={"target", "src"}, indent=2, exclude_none=True
        )

        ctx.emit(self.name, "calling gemini", target=branch.target, src=branch.src)
        resp = ctx.llm_client.call(
            self.assemble_system_prompt(ctx),
            user_msg,
            thinking_budget=-1,
            use_google_search=True,
            ctx=ctx,
        )
        raw = resp.text
        ctx.emit(
            self.name,
            "gemini response",
            raw=raw,
            grounding_titles=resp.grounding_titles,
        )

        try:
            result = LookupResult.model_validate_json(raw.strip())
        except ValidationError as e:
            raise StepFailed(
                self.name,
                f"Cannot parse response: {e}\nRaw: {raw}",
            ) from e

        # System prompt forbids answering from training memory alone; an empty
        # grounding-titles list means the model violated that contract.
        if not resp.grounding_titles:
            raise StepFailed(
                self.name,
                f"Ungrounded reply (no grounding_titles) for target: {branch.target}",
            )

        # Null or empty payload → the model looked and found nothing. Signal
        # MissingData so compute can react, rather than passing a degenerate
        # value downstream.
        v = result.value
        if v is None or (isinstance(v, (str, list)) and len(v) == 0):
            raise MissingData(
                f"lookup_external found no value for target: {branch.target}",
                missing=[branch.target],
            )

        # TODO: when `src` is set, verify the reported `result.source` (and/or
        # grounding-titles domains) plausibly overlaps with the requested `src`.
        # Hard to do robustly on raw strings — defer until we have a domain
        # normalizer (publisher → canonical domain stem).

        ctx.emit(self.name, "parsed", value=result.value, unit=result.unit)
        return [
            AnnotatedValue(
                description=branch.target, value=result.value, unit=result.unit
            )
        ]
