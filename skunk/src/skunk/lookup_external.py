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

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from skunk.errors import StepFailed
from skunk.executor import SkunkExecutor
from skunk.models import AnnotatedValue, HarnessContext


class LookupResult(BaseModel):
    """One Gemini lookup reply. `value` is the publisher's printed figure
    (number, entity string, or list for multi-value requests); `unit` is a
    snake_case token; `source` is the publisher token (parsed-and-discarded
    at the call site)."""
    model_config = ConfigDict(frozen=True)
    value: float | int | str | list[float | int | str]
    unit: str = ""
    source: str = "none"


class LookupExternalExecutor(SkunkExecutor):
    name: str = "lookup_external"
    system_prompt: str = """\
You find a single value from a reliable external source. You must
ground every answer in a publisher's page returned by Google Search —
never invent data.

## Input

The user message is a single bare JSON object:

  {"target": "<natural-language request for a single value>",
   "src":    "<natural-language description of preferred source | null>"}

`target` is a natural-language request for a single value (or a small
list of values in a fixed order). Do NOT re-parse `target` for
"according to X" / "from X" phrases — the planner has already
extracted any such hint into `src`. If `src` is null, the question
did not pin a source; pick the most authoritative public site for the
figure — typically the issuing statistical agency.

## Output format

A single bare JSON object. No markdown fences. No prose.

  {"value":  <number | string | array of those>,
   "unit":   "<snake_case_token>",
   "source": "<publisher_token>"}

## Field semantics

value     the publisher's full printed precision. JSON number for
          numeric answers, JSON string for entity names, JSON array
          for multiple values in request order. NEVER `null`, NEVER
          `NaN`, NEVER a string like "N/A". If the page lacked the
          value, use the JSON string "unknown" (and `source` must
          still name the site you searched).
unit      lowercase snake_case token describing what the value
          represents. Use `text` for named-entity answers. Invent a
          similar token when no standard one fits.
source    bare publisher token (the domain stem, e.g. `bls`,
          `bankofengland`). Use "none" only when `src` was null and
          no specific site applied.

## Procedure

(a) If `src` is present, search that site (use a `site:` filter or
    include the publisher name). Do not substitute a different
    aggregator even if you know it carries the same series.
(b) If `src` is null, pick the most authoritative public site and
    search there.
(c) Issue the search via your Google Search tool. Answering from
    training memory alone is rejected — your response is accepted
    only if a grounding chunk from the publisher's site appears.
(d) Read the snippets and extract the value at the publisher's full
    printed precision. Never round, format, or simplify — that's the
    final output agent's job. If the source prints 26.766, return
    26.766; if 26.8, return 26.8. FX rates are positive floats.
"""

    def _build_user_message(self, target: str, src: str | None) -> str:
        req: dict[str, Any] = {"target": target, "src": src}
        return json.dumps(req, indent=2, ensure_ascii=False)

    def run(
        self, prev: None, ctx: HarnessContext, *,
        target: str = "", src: str | None = None,
    ) -> list[AnnotatedValue]:
        if not target:
            raise StepFailed(self.name, "Missing 'target' arg")

        user_msg = self._build_user_message(target, src)

        ctx.emit(self.name, "calling gemini", target=target, src=src)
        resp = ctx.llm_client.call(
            self.assemble_system_prompt(ctx),
            user_msg, thinking_budget=-1, use_google_search=True, ctx=ctx,
        )
        raw = resp.text
        ctx.emit(self.name, "gemini response",
                 raw=raw, grounding_titles=resp.grounding_titles)

        try:
            result = LookupResult.model_validate_json(raw.strip())
        except ValidationError as e:
            raise StepFailed(
                self.name,
                f"Cannot parse response: {e}\nRaw: {raw}",
            ) from e

        ctx.emit(self.name, "parsed", value=result.value, unit=result.unit)
        return [AnnotatedValue(description=target, value=result.value, unit=result.unit)]
