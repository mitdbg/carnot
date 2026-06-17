"""Build pass: one LLM call parses a whole page into structured metadata.

A page's tagged element string → a `PageScan`: the catalog payload (blocks,
date span) plus build-routing signals (role / continuation /
unparsed-graphics / printed page). Driven per page by the pipeline's `scan` stage."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from skunk.common import B64Image, ExecutionContext, strip_code_fence
from skunk.errors import ParseError
from skunk.prompted_call import PromptedCall

from .data_model import CONTENT_BLOCK_FIELDS, ContentBlock


class PageScan(BaseModel):
    """One page's scan: the catalog payload plus build-only routing signals."""

    page_role: Literal["content", "toc", "non_content"]
    printed_page: str | None = None
    is_continuation: bool = False
    references_external_notes: bool = False
    is_notes_page: bool = False
    has_unparsed_graphics: bool = False
    parse_broken: bool = (
        False  # parsed elements too mangled to trust → re-read from the PDF
    )
    date_interval: tuple[int, int] | None = None
    blocks: list[ContentBlock] = Field(default_factory=list)
    continuation_pages: list[int] = Field(default_factory=list)
    # Set by the build's `notes_link` pass (never the scan LLM): the pages whose footnote
    # definitions qualify this page's data. Projected onto the catalog row's `notes_pages`.
    notes_pages: list[int] = Field(default_factory=list)
    vision_rescanned: bool = False

    @field_validator("date_interval")
    @classmethod
    def _years(cls, v: tuple[int, int] | None) -> tuple[int, int] | None:
        if v is None:
            return None
        try:
            lo, hi = int(str(v[0])[:4]), int(str(v[1])[:4])  # tolerate a "YYYY-MM" string
        except (TypeError, ValueError) as e:
            raise ValueError(f"date_interval must be two years, got {v!r}") from e
        if not (1700 <= lo <= 2100 and 1700 <= hi <= 2100):
            raise ValueError(f"date_interval years out of range: {v!r}")
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
- is_continuation: true ONLY when this page's table cannot be read on its own because the COLUMN HEADERS (and any
  units) that make its numbers interpretable are on a PREVIOUS page and are NOT restated here — e.g. a page that
  opens straight into unlabeled data columns. Judge by whether those headers are physically present on THIS page,
  never by the caption: a "Continued" title means nothing. If the page restates its column headers it is legible
  standalone, so is_continuation is FALSE — even if it is captioned "Continued" and its first rows carry on an
  account or category begun on the previous page. A few carried-over rows do NOT make a header-bearing page a continuation.
- references_external_notes: true when this page's tables/data carry footnote markers — superscripts, trailing
  reference digits, or symbols (*, †, ‡) — whose definitions are NOT on this page (they sit on an end-of-chapter
  "Footnotes" page or elsewhere). False when the page restates the definitions of every marker it uses.
- is_notes_page: true when the page is wholly or mostly footnotes, notes, or explanatory text that qualify DATA
  on OTHER pages — an end-of-chapter "Footnotes" section, a "Note.—" block. A data page that carries only its own
  self-contained footnotes is NOT a notes page.
- has_unparsed_graphics: a content page with a chart/figure whose data is not in its own text/tables.
- parse_broken: the parsed elements are too mangled to trust — scrambled or merged cells, numbers
  with no row/column labels. When in doubt set this true rather than guessing, and leave blocks empty.
- date_interval: the [lowest, highest] YEARS the page covers — each an integer year (e.g.
  [2024, 2024] for a single fiscal year; [1995, 2004] for a 10-year comparative table), or null
  when undatable. The publication year is given to you. Any mention of a year, implicit or explicit,
  in data or in prose, counts toward the span. A fiscal year is named by its end year (FY2024 -> 2024).
- blocks: the page's content blocks (content pages, plus a divider's one naming block; see below)."""


# Lead-in paragraph for the parsed-text scan: describes the tagged-element input.
_TEXT_LEAD = """\
You parse ONE page of a statistical publication into retrieval metadata.

The page is tagged elements ("[type] content" in reading order; tables are verbatim HTML, a bare
"[figure]" is a graphic the parser could not extract). You also get the document id and its
publication year, plus the corpus notes below.
"""

# Lead-in paragraph for the vision re-pass: same task, but the input is a rendered page image of a
# page the text parser handled badly (missing a graphic, or scrambled cells). Only this paragraph
# differs from the text tier; the schema, fields, and block rules below (`_SCAN_BODY`) are shared.
_VISION_LEAD = """\
You parse ONE page of a statistical publication into retrieval metadata.

The page is a rendered IMAGE of a single page whose text the parser handled badly — it dropped a
chart/figure or scrambled the cells. You also get the document id and its publication year, plus
the corpus notes below.
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
  "references_external_notes": <bool>,
  "is_notes_page": <bool>,
  "has_unparsed_graphics": <bool>,
  "parse_broken": <bool>,
  "date_interval": [<start_year>, <end_year>] | null,
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
    doc: str,
    pub_year: int | None,
) -> PageScan:
    """One LLM call → a validated `PageScan` (auto-reprompts on invalid output).
    The Treasury corpus blurb + lessons come from `ctx.prompt_overrides` (the
    `page_scan` target), appended to the system prompt by `PromptedCall`."""
    user = f"document: {doc}\npublication year: {pub_year}\n\nPAGE:\n{page_string}"
    return await _scan.call(ctx, user)


async def vision_scan_page(
    ctx: ExecutionContext,
    image: B64Image,
    *,
    doc: str,
    pub_year: int | None,
) -> PageScan:
    """Re-scan one page from its rendered IMAGE → a validated `PageScan`. Used by the
    pipeline's `vision_rescan` stage to redo pages the text scan flagged
    (`has_unparsed_graphics` / `parse_broken`). Same output object as `scan_page`."""
    return await _vision_scan.call(
        ctx, f"document: {doc}\npublication year: {pub_year}", images=[image]
    )
