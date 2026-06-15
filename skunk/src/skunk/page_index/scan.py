"""Build pass: one LLM call parses a whole page into structured metadata.

A page's tagged element string → a `PageScan`: the catalog payload (blocks,
date span) plus build-routing signals (role / continuation /
unparsed-graphics / printed page). Driven per page by the pipeline's `scan` stage."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from skunk.common import B64Image, ExecutionContext, strip_code_fence
from skunk.errors import ParseError
from skunk.prompted_call import PromptedCall

from .data_model import CONTENT_BLOCK_FIELDS, ContentBlock

_MONTH_RE = re.compile(r"\d{4}-(0[1-9]|1[0-2])")


class PageScan(BaseModel):
    """One page's scan: the catalog payload plus build-only routing signals."""

    page_role: Literal["content", "toc", "non_content"]
    printed_page: str | None = None
    is_continuation: bool = False
    has_unparsed_graphics: bool = False
    parse_broken: bool = (
        False  # parsed elements too mangled to trust → re-read from the PDF
    )
    date_interval: tuple[str, str] | None = None
    blocks: list[ContentBlock] = Field(default_factory=list)
    continuation_pages: list[int] = Field(default_factory=list)
    vision_rescanned: bool = False

    @field_validator("date_interval")
    @classmethod
    def _months(cls, v: tuple[str, str] | None) -> tuple[str, str] | None:
        if v is None:
            return None
        lo, hi = str(v[0])[:7], str(v[1])[:7]  # tolerate a stray day component
        if not (_MONTH_RE.fullmatch(lo) and _MONTH_RE.fullmatch(hi)):
            raise ValueError(f"date_interval must be two YYYY-MM months, got {v!r}")
        if lo > hi:
            raise ValueError(f"date_interval start after end: {v!r}")
        return (lo, hi)

    @model_validator(mode="after")
    def _headers_tables_only(self) -> "PageScan":
        for b in self.blocks:
            if b.kind != "table" and (b.column_headers or b.row_headers):
                raise ValueError(
                    f"{b.kind} block carries table-only column/row headers"
                )
        return self


# Prompt blurb for the PageScan object the scan emits (its `blocks` are content blocks, described
# by `CONTENT_BLOCK_FIELDS`).
PAGE_SCAN_FIELDS = """\
- page_role: content = a body page of data or substantive prose (indexed); toc = the issue's table
  of contents; non_content = front/back matter, indexes, forms, blank pages (not indexed). A section
  / chapter DIVIDER page — just a chapter title on an otherwise empty page — is non_content; record
  that chapter title as a single prose block (the name as the block title) so a missing outline can
  be reconstructed from it.
- printed_page: the page's own printed footer label (e.g. "A-1", "27"); null if unlabeled.
- is_continuation: true ONLY when this page contains a table or figure fragment that cannot be read on its own —
  key headers or information lives on the previous page. For example, a table whose data rows carry over from the
  previous page WITHOUT restating their column headers. A self-contained table that is logically a continuation but
  is legible by itself is NOT a continuation, regardless of what the caption says.
- has_unparsed_graphics: a content page with a chart/figure whose data is not in its own text/tables.
- parse_broken: the parsed elements are too mangled to trust — scrambled or merged cells, numbers
  with no row/column labels. When in doubt set this true rather than guessing, and leave blocks empty.
- date_interval: the [lowest, highest] months the page covers — each "YYYY-MM", or null when
  undatable. Resolve fiscal/calendar years against the publication month. Any mention of a date, implicit or explict,
  in data or in prose, counts towards the date. 
- blocks: the page's content blocks (content pages, plus a divider's one naming block; see below)."""


# Lead-in paragraph for the parsed-text scan: describes the tagged-element input.
_TEXT_LEAD = """\
You parse ONE page of a statistical publication into retrieval metadata.

The page is tagged elements ("[type] content" in reading order; tables are verbatim HTML, a bare
"[figure]" is a graphic the parser could not extract). You also get the publication month and the
corpus notes below.
"""

# Lead-in paragraph for the vision re-pass: same task, but the input is a rendered page image of a
# page the text parser handled badly (missing a graphic, or scrambled cells). Only this paragraph
# differs from the text tier; the schema, fields, and block rules below (`_SCAN_BODY`) are shared.
_VISION_LEAD = """\
You parse ONE page of a statistical publication into retrieval metadata.

The page is a rendered IMAGE of a single page whose text the parser handled badly — it dropped a
chart/figure or scrambled the cells. You also get the publication month and the corpus notes below.
Because you can now SEE the page, read its charts, figures, and any garbled table directly into
`blocks`, and set `has_unparsed_graphics`/`parse_broken` false unless the image itself is genuinely
illegible.
"""

# Shared body (schema + field/block definitions), identical for both the text and vision tiers.
_SCAN_BODY = (
    """\

## Output

{
  "page_role": "content" | "toc" | "non_content",
  "printed_page": "<this page's printed footer label, e.g. 'A-1'> | null",
  "is_continuation": <bool>,
  "has_unparsed_graphics": <bool>,
  "parse_broken": <bool>,
  "date_interval": ["YYYY-MM", "YYYY-MM"] | null,
  "blocks": [
    {"kind": "table" | "chart" | "prose", "title": "<caption | null>",
     "column_headers": ["<col>", ...], "row_headers": ["<row>", ...],
     "summary": "<what this block reports>"}
  ]
}

## Fields

"""
    + PAGE_SCAN_FIELDS
    + """

## Blocks

"""
    + CONTENT_BLOCK_FIELDS
    + """

Only content pages (and a divider's one naming block) carry blocks. Leave summary and title null
for a figure you cannot see.
"""
)

_SYSTEM = _TEXT_LEAD + _SCAN_BODY
_VISION_SYSTEM = _VISION_LEAD + _SCAN_BODY


def _parse(raw: str, _ctx: ExecutionContext) -> PageScan:
    try:
        return PageScan.model_validate_json(strip_code_fence(raw).strip())
    except ValidationError as e:
        raise ParseError(raw, str(e)) from e


_scan: PromptedCall[PageScan] = PromptedCall(
    name="page_scan",
    system_prompt=_SYSTEM,
    parse=_parse,
    output_instruction="Output a single bare JSON object — no markdown fences, no prose.",
    # 3 attempts: a small fraction of pages derail the model into a degenerate reply (prose-as-JSON,
    # an unescaped backslash) that a single temp-0 reprompt regenerates verbatim. PromptedCall
    # escalates temperature across these retries (0.0 → 0.4 → 0.6), which is what breaks the loop.
    max_parse_retries=2,
)

# Vision re-pass: same parser/output, image-based system prompt. Distinct name so it can be
# separately routed (SKUNK_MODEL_OVERRIDES) and traced; inherits the `page_scan` corpus blurb +
# lessons via its own target entries in the prompt-overrides YAML.
_vision_scan: PromptedCall[PageScan] = PromptedCall(
    name="page_scan_vision",
    system_prompt=_VISION_SYSTEM,
    parse=_parse,
    output_instruction="Output a single bare JSON object — no markdown fences, no prose.",
    # Vision output is non-deterministic even at temp 0, and the densest wide tables occasionally
    # emit a single malformed field; allow extra reprompts (3 attempts, with PromptedCall's
    # temperature escalation) so these self-correct within a run rather than on the next resume.
    max_parse_retries=2,
)


async def scan_page(
    ctx: ExecutionContext,
    page_string: str,
    *,
    bulletin: str,
) -> PageScan:
    """One LLM call → a validated `PageScan` (auto-reprompts on invalid output).
    The Treasury corpus blurb + lessons come from `ctx.prompt_overrides` (the
    `page_scan` target), appended to the system prompt by `PromptedCall`."""
    user = f"publication month: {bulletin}\n\nPAGE:\n{page_string}"
    return await _scan.call(ctx, user)


async def vision_scan_page(
    ctx: ExecutionContext,
    image: B64Image,
    *,
    bulletin: str,
) -> PageScan:
    """Re-scan one page from its rendered IMAGE → a validated `PageScan`. Used by the
    pipeline's `vision_rescan` stage to redo pages the text scan flagged
    (`has_unparsed_graphics` / `parse_broken`). Same output object as `scan_page`."""
    return await _vision_scan.call(
        ctx, f"publication month: {bulletin}", images=[image]
    )
