"""On-disk artifact: the catalog row (`PageCatalogRow`), concept tree
(`ConceptTree`), and the filename layout shared by the build and query paths."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from skunk.common import PageRef


# Artifact layout — filenames under the artifact root, joined onto a root path
# (`build --output-dir` writing, `page_index_root()` reading) at each call site.
CATALOG_SUBDIR = "catalog"      # slim shipped rows (PageCatalogRow); query-facing
BUILD_SUBDIR = "build"          # rich build-only intermediate (BuildPage)
L1_SUBDIR = "l1"
TREE_FILE = "concept_tree.json"
MANIFEST_FILE = "manifest.json"
BUILD_STATS_FILE = "build_stats.json"


class ContentBlock(BaseModel):
    """One detected content block on a page: a table, a chart, or prose. A page
    may carry several."""
    kind: Literal["table", "chart", "prose"]
    title: str | None = None
    # Tables only — empty for chart/prose.
    column_headers: list[str] = Field(default_factory=list)
    row_headers: list[str] = Field(default_factory=list)

    # What the block is about; set by the `summarize` pass, read by the semantic filter.
    summary: str | None = None


class PageCatalogRow(BaseModel):
    """One shipped catalog row per retrievable PDF page — the query-facing schema.
    Build-only state (placement, banners, diagnostics) lives on `BuildPage` and
    never reaches this on-disk catalog."""
    bulletin: str                                   # "YYYY-MM"
    page: int                                       # 1-based PDF page index

    content_blocks: list[ContentBlock] = Field(default_factory=list)
    # Salient retrieval terms, produced by the `summarize` LLM pass.
    keywords: list[str] = Field(default_factory=list)
    # `YYYY-MM` `(low, high)` data span from `summarize`; read by the year filter.
    date_interval: tuple[str, str] | None = None

    @property
    def ref(self) -> PageRef:
        """This row's canonical page coordinate (the catalog dict key)."""
        return PageRef(month=self.bulletin, page=self.page)

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


class BuildPage(PageCatalogRow):
    """A `PageCatalogRow` plus the build-only metadata the pipeline threads
    across stages but never ships. Persisted to `build/` for stage resume;
    `to_row()` projects the slim shipped row at finalize."""

    # `section`: the page's level-1 bulletin section (assigned by `place_pages`).
    # `printed_page`: page-footer label (e.g. "A-1"), mapped onto a ToC span.
    section: str | None = None
    printed_page: str | None = None

    # `char_count`: drives summarize's blank-page gate. `banner_self`: the page's
    # running-head, used to derive L1 vocab + banner-exact placement.
    # `is_content`: summarize sets it False for blank/ToC/front-matter; finalize
    # drops those rows.
    char_count: int = 0
    banner_self: str | None = None
    is_content: bool = True

    def to_row(self) -> PageCatalogRow:
        """Project to the slim, shippable query row (build-only fields dropped)."""
        return PageCatalogRow(
            bulletin=self.bulletin,
            page=self.page,
            content_blocks=self.content_blocks,
            keywords=self.keywords,
            date_interval=self.date_interval,
        )


# --- Concept tree -----------------------------------------------------------
# The flat bucket tree the build's chapter-merge stage writes to
# `concept_tree.json`, and the query path reads back to pick chapters. Pages are
# partitioned across chapters (each page lands in exactly one); a chapter's
# `pages` are `PageRef` postings resolved to full `PageCatalogRow`s at query time
# via the catalog.


class Chapter(BaseModel):
    """One concept-tree bucket: a named group of catalog pages, with the
    `description` + `examples` shown to the chapter-pick LLM. The bucket's name
    is the key in `ConceptTree.chapters`, not a field here."""
    n_pages: int = 0
    description: str = ""
    examples: list[str] = Field(default_factory=list)
    pages: list[PageRef] = Field(default_factory=list)

    @field_validator("pages", mode="before")
    @classmethod
    def _page_refs(cls, v: list) -> list:
        """On-disk page entries are `{bulletin, page, key_phrases}` dicts; map the
        bulletin→month identity into a `PageRef` (the build-only `key_phrases` is
        never read by the query path)."""
        return [PageRef(month=p["bulletin"], page=p["page"]) if isinstance(p, dict) else p
                for p in v]


class ConceptTree(BaseModel):
    """`name -> Chapter`, nested on disk under a top-level `"chapters"` key."""
    chapters: dict[str, Chapter] = Field(default_factory=dict)
