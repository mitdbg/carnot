"""On-disk artifact: the catalog row (`PageCatalogRow`), concept tree
(`ConceptTree`), and the filename layout shared by the build and query paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Literal, Protocol

from pydantic import BaseModel, Field, model_validator

from skunk.common import PageRef
from skunk.errors import StepFailed


# Artifact filenames under the build/query root, joined at each call site.
CATALOG_SUBDIR = "catalog"  # slim shipped rows (PageCatalogRow); query-facing
SCANS_SUBDIR = "scans"  # raw per-doc page scans (PageScan): the page store's content source
PAGES_SUBDIR = "pages"  # per-bulletin page store: anchor -> member texts + figures
RENDERS_SUBDIR = (
    "renders"  # lazy 200-DPI page-image cache (written on first vision read)
)
TREE_FILE = "concept_tree.json"  # era-keyed concept tree (ConceptTree)


def page_index_root() -> Path:
    """The artifact root — `SKUNK_PAGE_INDEX_DIR`, the single source of truth shared by the
    retriever (`query.py`) and the page store (`store.py`)."""
    env = os.environ.get("SKUNK_PAGE_INDEX_DIR")
    if not env:
        raise StepFailed(
            "retrieve",
            "SKUNK_PAGE_INDEX_DIR is not set; point it at a built page-index artifact.",
        )
    return Path(env)


class ContentBlock(BaseModel):
    """One detected content block on a page: a table, a chart, or prose. A page
    may carry several."""

    kind: Literal["table", "chart", "prose"]
    title: str | None = None
    # Tables only — empty for chart/prose.
    column_headers: list[str] = Field(default_factory=list)
    row_headers: list[str] = Field(default_factory=list)

    # What the block reports; set by the scan, read by the semantic filter.
    summary: str | None = None

    # Parse quality: the min VLM confidence (0..1) of the parsed elements backing this block,
    # carried from the corpus parse by the scan stage. Low values flag a noisy parse (old
    # govinfo scans run ~0.94; clean modern tables ~0.99) — a routing signal for vision rescan
    # and a caveat for downstream extract. None when the parse carried no confidence.
    confidence: float | None = None

    # Cross-page table merge (the build's `table_merge` pass; the scan LLM never sets
    # these). A table block with NO row and NO column labels is a genuine continuation —
    # a fragment whose headers live on an earlier page, or a "Footnotes to Table X"
    # spillover. The pass links it to its parent block:
    #   - `extra_pages` (on the PARENT) — the later pages whose fragment blocks were folded
    #     into this block. Retrieval expands a selected block to [its own page, *extra_pages]
    #     so the text tier reads and the vision tier renders every page of the table.
    #   - `merged_into` (on the FRAGMENT) — the parent block's page. The catalog projection
    #     drops merged fragments (the parent's row covers them); scans keep them, lossless.
    extra_pages: list[int] = Field(default_factory=list)
    merged_into: int | None = None


# Prompt blurb for the ContentBlock object — one source of truth shared by every prompt that
# emits or reads a content block (the page scan that produces them, the semantic filter that
# reads them). Drop it under a prompt's `## Fields` / `## Input` header.
CONTENT_BLOCK_FIELDS = """\
A content block is one indexable unit of a page — one "table" (a data table), one "chart" (a
figure/graph/plot), or one "prose" block (the narrative text on a page that has no table):
  - kind: "table" | "chart" | "prose".
  - title: the block's own caption, copied verbatim from the page (fix only obvious OCR typos);
    null when it has none — never fabricated.
  - column_headers / row_headers: the table's column and row labels, verbatim (tables only;
    empty for charts and prose). Stacked header levels are joined top-to-bottom into one label
    — a "1944" group over "Nov." becomes "1944 Nov." — so each header stays complete and distinct.
  - summary: a short free-text description of what the block reports — its subject, breakdown,
    and time basis — never the numeric values."""


class PageCatalogRow(BaseModel):
    """One shipped catalog row per retrievable PDF page — the query-facing schema,
    projected from each content page's `PageScan` by the catalog stage."""

    # The corpus document this page belongs to, identified by its parsed-JSON/PDF filename
    # stem (e.g. "combined_statement__modern__2024__c40"). The page index keys on (source,
    # page); source/era/year/section are all parseable from it (see corpus.parse_doc_id), so
    # the row carries only `source` and lets readers infer the rest.
    source: str
    page: int  # 1-based PDF page index
    # The page's own printed footer label ("5", "A-1"); None when unlabeled. Carried from the
    # scan so a page-pin can resolve a question's stated page number against the printed label
    # (the planner's `page_pin`), not just the PDF index. Optional → old catalogs load as None.
    printed_page: str | None = None
    # True when this page's table is header-less — its column headers live on the PREVIOUS page
    # and aren't restated here (set by the scan, pruned by the `continuation_check` pass). The
    # flag is self-describing: when a reader accesses this page it also fetches the predecessor
    # chain that carries the headers (`continuation_chain`). Nothing is merged or folded.
    is_continuation: bool = False
    # Pages whose footnote/notes definitions qualify THIS page's data (set by the `notes_link`
    # pass: an end-of-section "Footnotes" page this data page cites). A reader expands a selected
    # data page to also read these — see `notes_refs`. Empty when the page cites no external notes.
    notes_pages: list[int] = Field(default_factory=list)

    content_blocks: list[ContentBlock] = Field(default_factory=list)
    # `(low, high)` data-year span from the scan (inclusive calendar/fiscal years); read by the
    # year filter. Defaults to the document's own year when the scan can't narrow it.
    date_interval: tuple[int, int] | None = None
    # Continuation pages folded into this row by the build's merge pass (the dropped pages
    # this anchor represents). They carry no row of their own; the retriever expands them
    # back into the returned refs so extract reads the full continued table.
    continuation_pages: list[int] = Field(default_factory=list)

    @property
    def ref(self) -> PageRef:
        """This row's canonical page coordinate (the catalog dict key). The `month` slot of
        `PageRef` carries the doc `source` stem — the page index's opaque per-doc key."""
        return PageRef(month=self.source, page=self.page)

    @property
    def member_pages(self) -> list[int]:
        """The physical pages this row represents — the anchor page first, then its folded
        continuation pages. The page store keys its per-row text/figures on this list."""
        return [self.page, *self.continuation_pages]

    def member_refs(self) -> list[PageRef]:
        """This row's member pages as `PageRef`s (a contiguous run). The retriever expands a
        kept anchor into these so extract reads the merged text once (continuation pages carry
        no text of their own) and the vision tier renders every page of the continued table."""
        return [PageRef(month=self.source, page=p) for p in self.member_pages]

    def notes_refs(self) -> list[PageRef]:
        """The notes/footnotes pages whose definitions qualify this page's data (`notes_pages`),
        as `PageRef`s. A reader appends these to a selected data page so extract/compute sees the
        footnote definitions; kept distinct from `member_refs` so notes never enter the table's
        own page span or `date_interval`."""
        return [PageRef(month=self.source, page=p) for p in self.notes_pages]

    def block_refs(self, block: ContentBlock) -> list[PageRef]:
        """The physical pages ONE block spans: the row's member pages plus the block's own
        cross-page merge tail (`extra_pages`), deduped in order. This is what a selected block
        expands to — so extract's text tier reads, and its vision tier renders, every page of
        a table that continues across pages, while unmerged blocks stay a single page."""
        pages = list(dict.fromkeys(self.member_pages + list(block.extra_pages)))
        return [PageRef(month=self.source, page=p) for p in pages]

    @property
    def primary_title(self) -> str | None:
        """First non-empty title across content_blocks, or None."""
        return next((b.title for b in self.content_blocks if b.title), None)

    def all_column_headers(self) -> list[str]:
        out: list[str] = []
        for b in self.content_blocks:
            out.extend(b.column_headers)
        return out

    def to_json(self) -> str:
        """Serialize to one JSON line."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, line: str) -> "PageCatalogRow":
        """Parse one JSONL catalog row."""
        return cls.model_validate_json(line)


class _Continuable(Protocol):
    """Structural type for `continuation_chain`'s lookup result — anything carrying the
    `is_continuation` flag. Satisfied by both `PageCatalogRow` and the raw `PageScan` the page
    store now serves, so the walk works regardless of which one the caller hands it."""

    is_continuation: bool


def continuation_chain(
    ref: PageRef, get_row: Callable[[PageRef], "_Continuable | None"]
) -> list[PageRef]:
    """The predecessor pages a header-less continuation page needs to be readable — the run of
    earlier physical pages back to (and including) the header-bearing page that carries its
    column labels. Walks physical pages backward from `ref`: each step is included; the walk
    stops once it reaches a page whose row is NOT `is_continuation` (the head) or that has no
    row. Returns the chain in ascending page order, EXCLUDING `ref` itself; empty when `ref`
    is not a continuation (or unknown). This is the read-side of the self-describing flag — no
    page is ever merged at build time; readers fetch the chain on access."""
    row = get_row(ref)
    if row is None or not row.is_continuation or ref.page is None:
        return []
    chain: list[PageRef] = []
    page = ref.page
    while page > 1:
        page -= 1
        prev = PageRef(month=ref.month, page=page)
        prow = get_row(prev)
        if prow is None:
            break
        chain.append(prev)
        if not prow.is_continuation:
            break  # reached the header-bearing head page
    chain.reverse()
    return chain


# Prompt blurb for the catalog row the semantic filter sees, one per candidate page (its
# `content_blocks` are described by `CONTENT_BLOCK_FIELDS`).
PAGE_CATALOG_ROW_FIELDS = """\
source: the corpus document this page belongs to, its filename stem (e.g. "combined_statement__modern__2024__c40").
page: the 1-based PDF page index.
date_interval: the [start, end] YEARS the page's DATA covers — each an integer year, or null when undatable.
content_blocks: the page's content blocks (see below)."""


# --- Concept tree -----------------------------------------------------------
# The era-keyed bucket tree the build's chapter-merge stage writes to
# `concept_tree.json`, and the query path reads back to pick chapters. Each era
# is a span of years with a stable reporting convention; within an era, pages are
# partitioned across chapters (each page lands in exactly one). A chapter's `pages`
# are `PageRef` postings resolved to full `PageCatalogRow`s at query time via the
# catalog. The query picker scopes the chapter listing to the era(s) overlapping a
# question's period.


class PageRange(BaseModel):
    """A chapter bucket's run of pages within ONE issue (inclusive, 1-based). Placement
    files a chapter as a contiguous physical range per issue, so a bucket's membership is
    naturally a list of these — one per issue it appears in."""

    bulletin: str  # "YYYY-MM"
    start: int  # first 1-based PDF page (inclusive)
    end: int  # last 1-based PDF page (inclusive)

    def refs(self) -> list[PageRef]:
        return [
            PageRef(month=self.bulletin, page=p)
            for p in range(self.start, self.end + 1)
        ]


class Chapter(BaseModel):
    """One concept-tree bucket: a named group of catalog pages. The bucket's name is the key
    in `EraTree.chapters`, not a field here. `description` + `examples` are the picker-facing
    scope blurb the describe pass distills by sampling the table titles on this chapter's pages;
    `pages` are its page ranges (its membership). (The build-time `members` (raw headings) and
    sub-section `children` are intermediates only — neither is persisted; the picker relies on
    description/examples.)"""

    n_pages: int = 0
    # Picker-facing scope blurb (describe pass) — summarized from sampled page table titles.
    description: str = ""
    examples: list[str] = Field(default_factory=list)
    pages: list[PageRange] = Field(default_factory=list)

    @property
    def refs(self) -> list[PageRef]:
        """Expand the page ranges to individual `PageRef`s (the query path's page keys)."""
        return [r for pr in self.pages for r in pr.refs()]


# Prompt blurb for a concept-tree chapter — read by the chapter-pick prompt at query time.
CHAPTER_FIELDS = """\
A chapter is one top-level bucket of the era's table of contents, keyed by its name:
  - description: a short scope summary of what the chapter covers.
  - examples: concrete sub-area / table names under it — string-match anchors for matching a
    question to the right chapter.
  - n_pages: how many catalog pages it holds (a size hint)."""


class EraTree(BaseModel):
    """One era's chapter tree: an inclusive `YYYY-MM` span, a label, and that era's
    `name -> Chapter` buckets (pages partitioned within the era)."""

    span: tuple[str, str]
    label: str = ""
    chapters: dict[str, Chapter] = Field(default_factory=dict)


class ConceptTree(BaseModel):
    """Era-keyed chapter trees on disk under a top-level `"eras"` key. A legacy
    single-tree file (`{"chapters": {...}}`) loads as one all-spanning era."""

    eras: list[EraTree] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy(cls, v):
        if isinstance(v, dict) and "eras" not in v and "chapters" in v:
            return {
                "eras": [
                    {
                        "span": ["0000-00", "9999-99"],
                        "label": "all",
                        "chapters": v["chapters"],
                    }
                ]
            }
        return v
