"""Per-issue table of contents: extraction, reconstruction, and placement.

  - `outline_issue` (LLM `toc_outline`) reads an issue's table-of-contents text and returns
    its chapter hierarchy with each chapter's PRINTED page range (or flags a non-ToC).
  - `reconstruct_outline` (LLM `toc_reconstruct`) handles ToC-less issues — it reads the
    section-divider pages and emits the same hierarchy with physical `start_page`s.
  - `place_pages` (plain Python) then files each content page under its chapter: resolve each
    chapter's start to a physical page, lay the chapters out as contiguous physical ranges,
    and assign by range membership.

The cross-issue era merge that turns these placements into `concept_tree.json` lives in
`eras.py`.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from skunk.common import ExecutionContext, strip_code_fence
from skunk.errors import ParseError
from skunk.prompted_call import PromptedCall

from .scan import PageScan


# ---------------------------------------------------------------------------
# ToC hierarchy — LLM output (from the ToC text alone)
# ---------------------------------------------------------------------------

class TocChapter(BaseModel):
    name: str
    children: list[str] = Field(default_factory=list)
    start_printed: str = ""       # printed page label where the chapter begins (real ToC)
    end_printed: str | None = None
    start_page: int | None = None  # physical PDF page (the reconstructed path binds here)


# Prompt blurb for a TocChapter — shared by the ToC-outline prompt (real ToC, uses the printed
# page labels) and the ToC-reconstruct prompt (no ToC, uses the physical `start_page`).
TOC_CHAPTER_FIELDS = """\
A chapter:
  - name: its heading, verbatim (fix only obvious OCR typos; do not invent or renumber).
  - children: immediate sub-section names — one level only, [] if none.
  - start_printed / end_printed: the chapter's PRINTED start/end page labels exactly as a real
    table of contents shows them ("27", "A-1"); end_printed is null when unclear.
  - start_page: the PHYSICAL page index where the chapter begins, used when the outline is
    reconstructed from an issue that has no table of contents."""


class TocHierarchy(BaseModel):
    # Whether the candidate page(s) really are this issue's table of contents. The
    # model sets this false to FLAG a non-ToC — a content/section-start page the
    # upstream scan mislabeled `toc`, a cross-issue cumulative index, etc. — instead
    # of extracting; a flagged non-ToC carries no chapters.
    is_toc: bool = True
    # "toc" = extracted from a real ToC page; "reconstructed" = inferred from section
    # banners for a ToC-less issue (chapters carry a physical `start_page`).
    source: Literal["toc", "reconstructed"] = "toc"
    chapters: list[TocChapter] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "TocHierarchy":
        if not self.is_toc:
            self.chapters = []   # a flagged non-ToC has no chapters
            return self
        for c in self.chapters:
            # A chapter needs a name and SOME start anchor: a printed label (real ToC)
            # or a physical page (reconstructed).
            if not c.name.strip() or not (c.start_printed.strip() or c.start_page):
                raise ValueError(f"chapter needs a name and a start anchor: {c!r}")
        return self


_OUTLINE_SYSTEM = """\
You read the candidate table-of-contents page(s) of ONE issue of a periodical and EITHER
extract its chapter structure OR flag the pages as not a real table of contents.

You get the verbatim text of the candidate page(s) — normally a list of sections, each
followed by the PRINTED page number (or page range) where that section begins. They were
auto-detected upstream and may be wrong.

## Step 1 — is this actually a table of contents?

A genuine table of contents:
  - is headed "Table of Contents" / "Contents" (tolerate OCR garbling — "C O N T E N T S",
    "TabIe of Contents", "CONTEN TS" all count), AND
  - lists THIS issue's sections/articles, each paired with a printed page number or range
    ("27", "A-1", "21-29").

It is NOT a table of contents when the upstream tagger grabbed the wrong page — e.g. a data
table, the FIRST page of a content section (a banner title like "SUMMARY OF FISCAL
STATISTICS" sitting over a table, with no list of sections-and-page-numbers), a cross-issue
cumulative index, or front/back matter. If so, set `"is_toc": false` and return
`"chapters": []` — do not invent chapters from a non-ToC page.

If several candidate pages are given and only some are real ToC pages, extract from the real
ones and ignore the rest; set `is_toc` true as long as at least one is a genuine ToC.

## Step 2 — if it IS a table of contents, extract the chapters

Return the issue's TOP-LEVEL chapters, in body order, each with its printed page range.

## Output

{"is_toc": <bool>,
 "chapters": [
  {"name": "<chapter name>", "children": ["<sub-section>", ...],
   "start_printed": "<printed page label where the chapter begins>",
   "end_printed": "<printed page label where it ends, or null>"}
]}

Each chapter:
""" + TOC_CHAPTER_FIELDS + """

Here use the printed labels (start_printed / end_printed). A chapter ends just before the next
begins; the last ends at the ToC's final page reference.

## Rules

- Skip entries that are not body chapters (cover, contents, masthead, index, lists of tables).
- When `is_toc` is false, `chapters` MUST be empty.
"""


def _parse_outline(raw: str, _ctx: ExecutionContext) -> TocHierarchy:
    try:
        return TocHierarchy.model_validate_json(strip_code_fence(raw).strip())
    except ValidationError as e:
        raise ParseError(raw, str(e)) from e


_outline: PromptedCall[TocHierarchy] = PromptedCall(
    name="toc_outline",
    system_prompt=_OUTLINE_SYSTEM,
    parse=_parse_outline,
    output_instruction='Output a single bare JSON object {"is_toc": ..., "chapters": [...]} — no fences, no prose.',
)


async def outline_issue(
    ctx: ExecutionContext, *, bulletin: str, toc_texts: dict[int, str]
) -> TocHierarchy:
    """One LLM call → the issue's chapter hierarchy, from its candidate `toc` pages.
    `toc_texts` is `{page_idx: page_string}` for the issue's `toc`-role pages. The call
    also validates the candidates: it returns `is_toc=False` (no chapters) when they
    aren't a real table of contents. Empty `toc_texts` → `is_toc=False` with no call."""
    if not toc_texts:
        return TocHierarchy(is_toc=False)
    toc = "\n\n".join(f"--- TOC page {idx} ---\n{t}" for idx, t in sorted(toc_texts.items()))
    user = f"issue: {bulletin}\n\nTABLE-OF-CONTENTS TEXT:\n{toc}"
    return await _outline.call(ctx, user)


# ---------------------------------------------------------------------------
# ToC reconstruction — for issues that publish no table of contents
# ---------------------------------------------------------------------------

_RECONSTRUCT_SYSTEM = """\
You reconstruct the chapter outline of ONE issue of a statistical periodical that has NO table of
contents, from a per-page summary of the issue — each page's role, printed page label, and the
titles/summaries of any blocks on it (no raw text).

The issue is organized into top-level CHAPTERS. Each chapter opens with a DIVIDER page whose
purpose is just to announce the chapter — essentially only the chapter name, little or no data. A
divider is usually tagged role "non_content" carrying a single prose block whose title is the
chapter name; its printed label may reset (e.g. "-1-").

You also get REFERENCE chapters — the top-level chapters of neighboring issues of this same
publication. Use them two ways:
  1. NORMALIZE a divider's chapter name to the reference wording when it clearly matches.
  2. COVERAGE — if a reference chapter plainly has content in this issue (by the block titles on
     its pages) but no divider page, still emit it, anchored to the first page of that content.

## What is NOT a chapter

Do not emit data pages, table captions ("Table 1.- ...", or a single table's name like "Net
Capital Movement to the United States"), continuation pages, charts, or front/back matter — the
library stamp, masthead, or the cumulative table of contents (a multi-issue index, usually near
the front or back).

## Output

{"is_toc": true, "chapters": [{"name": "<chapter>", "start_page": <physical page>}, ...]}

`start_page` is the physical `page` value where the chapter begins, in page order. If the issue
has no recognizable chapter structure, return {"is_toc": false, "chapters": []}.
"""


def _parse_reconstruct(raw: str, _ctx: ExecutionContext) -> TocHierarchy:
    try:
        h = TocHierarchy.model_validate_json(strip_code_fence(raw).strip())
    except ValidationError as e:
        raise ParseError(raw, str(e)) from e
    h.source = "reconstructed"
    return h


_reconstruct: PromptedCall[TocHierarchy] = PromptedCall(
    name="toc_reconstruct",
    system_prompt=_RECONSTRUCT_SYSTEM,
    parse=_parse_reconstruct,
    output_instruction='Output a single bare JSON object {"is_toc": ..., "chapters": [...]} — no fences, no prose.',
)


async def reconstruct_outline(
    ctx: ExecutionContext, *, bulletin: str, page_views: list[dict], reference: list[str]
) -> TocHierarchy:
    """One `toc_reconstruct` LLM call → a `TocHierarchy` (source='reconstructed') for an issue
    with no ToC. Chapters are read from the issue's section-DIVIDER pages (a chapter name on an
    otherwise empty page), named against `reference` — the top-level chapters of neighboring
    real-ToC issues — for consistent wording and coverage. Each chapter binds to a physical
    `start_page`, so placement skips printed→physical resolution. Empty `page_views` →
    `is_toc=False` with no call."""
    if not page_views:
        return TocHierarchy(is_toc=False, source="reconstructed")
    ref = "\n".join(f"  - {r}" for r in reference) or "  (none)"
    user = (f"issue: {bulletin}\n\nREFERENCE CHAPTERS (neighboring issues):\n{ref}\n\n"
            f"PAGES (JSON, {len(page_views)} entries):\n{json.dumps(page_views, ensure_ascii=False)}")
    return await _reconstruct.call(ctx, user)


def coalesce_toc_ranges(
    toc_pages: list[int], *, max_skip: int = 1
) -> list[tuple[int, int]]:
    """ToC build: group a bulletin's `toc`-tagged pages into contiguous physical page
    ranges, bridging gaps of at most `max_skip` untagged page(s) — so a real
    multi-page ToC the scan tagged with one interstitial missed (e.g. [4, 5, 7])
    coalesces into the single range (4, 7).

    Purely structural: NO position or cluster-size heuristics, every tagged page is
    kept. Deciding which ranges are a genuine table of contents (vs a content page
    the scan mislabeled `toc`) is left to `outline_issue`, which reads the coalesced
    pages and flags non-ToCs via its `is_toc` output. Returns `(start, end)`
    inclusive ranges in page order."""
    pages = sorted(set(toc_pages))
    if not pages:
        return []
    ranges: list[tuple[int, int]] = []
    start = prev = pages[0]
    for p in pages[1:]:
        if p - prev <= max_skip + 1:
            prev = p
        else:
            ranges.append((start, prev))
            start = prev = p
    ranges.append((start, prev))
    return ranges


# ---------------------------------------------------------------------------
# Placement — plain Python (printed → physical, contiguous ranges)
# ---------------------------------------------------------------------------

_DIGITS = re.compile(r"\s*0*(\d+)\s*$")


def _printed_key(s: str | None):
    """Comparable key for a printed label: numeric when it's digits, else lowercased text.
    Lets a ToC ref ("27") match a page label ("27") regardless of leading zeros/whitespace."""
    if not s:
        return None
    m = _DIGITS.match(s)
    return ("n", int(m.group(1))) if m else ("s", s.strip().lower())


def _resolve_start(start_printed: str, scans: dict[int, PageScan]) -> int | None:
    """First physical page whose printed label matches `start_printed`, or None."""
    key = _printed_key(start_printed)
    if key is None:
        return None
    return next((i for i in sorted(scans) if _printed_key(scans[i].printed_page) == key), None)


def _chapter_start_phys(ch: TocChapter, scans: dict[int, PageScan]) -> int | None:
    """Physical start page of a chapter. Reconstructed chapters carry `start_page`
    directly (the printed labels of ToC-less issues are too noisy to resolve); real
    ToC chapters resolve their `start_printed` label via `_resolve_start`."""
    if ch.start_page is not None:
        return ch.start_page if ch.start_page in scans else None
    return _resolve_start(ch.start_printed, scans)


def place_pages(
    hierarchy: TocHierarchy, scans: dict[int, PageScan]
) -> tuple[dict[int, str], dict]:
    """Resolve each chapter's printed start to a physical page, lay the chapters out as
    contiguous physical ranges, and file every content page by range membership.
    Returns `({page_idx: chapter_name}, coverage_stats)`."""
    bounds: list[tuple[int, str]] = []
    unresolved: list[str] = []
    seen: set[int] = set()
    for ch in hierarchy.chapters:
        phys = _chapter_start_phys(ch, scans)
        if phys is None:
            unresolved.append(ch.name)
        elif phys in seen:
            # a later ToC entry resolving to an already-claimed start page is a
            # sub-section listed as a sibling — fold it into the earlier chapter.
            continue
        else:
            seen.add(phys)
            bounds.append((phys, ch.name))
    bounds.sort()

    last = max(scans) if scans else 0
    ranges = [
        (start, (bounds[i + 1][0] - 1 if i + 1 < len(bounds) else last), name)
        for i, (start, name) in enumerate(bounds)
    ]

    assign: dict[int, str] = {}
    unfiled: list[int] = []
    for idx, s in scans.items():
        if s.page_role != "content":
            continue
        name = next((nm for st, en, nm in ranges if st <= idx <= en), None)
        if name is not None:
            assign[idx] = name
        else:
            unfiled.append(idx)

    n_content = len(assign) + len(unfiled)
    return assign, {
        "n_content": n_content,
        "n_filed": len(assign),
        "n_unfiled": len(unfiled),
        "unfiled": sorted(unfiled),
        "unresolved_chapters": unresolved,
        "ranges": ranges,
    }
