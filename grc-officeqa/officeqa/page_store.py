"""The Treasury-corpus implementation of skunk's `PageContentStore` protocol.

Wraps the page-index `PageStore` (parsed-JSON text + on-demand 200-DPI page
renders) and supplies the Treasury structure knowledge the generic extract
operator reads through the protocol: content-block metadata lines, header-less
continuation-page context (inherited column grammar + own block summaries), and
read-group expansion (folded continuation tails + linked footnote pages).

This logic lived inside `skunk/extract.py` until 2026-07-07 (Phase 5.4 moved it
behind the protocol seam); the wording of the injected prompt context is
unchanged.
"""

from __future__ import annotations

from functools import lru_cache

from skunk.common import B64Image, PageRef

from officeqa.page_index.data_model import inherited_column_headers
from officeqa.page_index.store import PageStore, get_page_store


class OfficeQAPageStore:
    """`skunk.page_store.PageContentStore` implementation for the Treasury corpus."""

    def __init__(self, pdf_dir: str):
        self._store: PageStore = get_page_store(pdf_dir)

    def text(self, ref: PageRef) -> str | None:
        return self._store.text(ref)

    def image(self, ref: PageRef) -> B64Image | None:
        return self._store.image(ref)

    def page_metadata(self, refs: list[PageRef]) -> str:
        """One line per content block — kind and title only. Headers and summary are
        selection-stage context; at read time they restate ~half the page's tokens out of
        layout order, so the model reads structure from the table markup in the text
        instead. Empty string when no scan metadata is available."""
        lines: list[str] = []
        for ref in refs:
            sc = self._store.summary(ref)
            if sc is None:
                continue
            for block in sc.blocks:
                lines.append(
                    f"- page {ref.page}: {block.kind}: {block.title or '(untitled)'}"
                )
        return "\n".join(lines)

    def extra_read_context(self, ref: PageRef) -> str:
        """Extra read context for a header-less continuation page, in place of fetching its
        predecessor pages: (1) the column grammar inherited from the run head
        (`inherited_column_headers`) so unlabeled cells can be placed, and (2) the page's OWN
        block summaries — the account it reports, which a banner-only continuation page's text
        may not state (its account heading is on an earlier page). Only UNTITLED blocks are
        summarized: a titled block is already named by the page-metadata line
        (`- page N: table: <title>`), so repeating its summary here just duplicates that line;
        an untitled block has no title to show, so its summary is the only account signal.
        Empty string for a normal (non-continuation) page."""
        sc = self._store.summary(ref)
        if sc is None or not getattr(sc, "is_continuation", False):
            return ""
        lines: list[str] = []
        inh = inherited_column_headers(ref, self._store.summary)
        if inh:
            # Only the column ORDER is inherited (stable across the run); the account is NOT —
            # it can change mid-run, so it's left to the page's own summaries/titles below,
            # not the head's.
            cols = inh[2]
            lines.append(
                "This page's table opens directly into data columns; its header row is on an earlier "
                f"page (not reprinted here). Columns, left to right: {cols}."
            )
        # Only summarize blocks the page metadata can't already name (untitled ones) — a titled
        # block's "- page N: table: <title>" line makes its summary here redundant.
        summaries = [b.summary for b in sc.blocks if b.summary and not b.title]
        if summaries:
            lines.append("This page reports: " + " / ".join(summaries))
        return "\n".join(lines)

    def read_group(self, ref: PageRef) -> list[PageRef]:
        """The page itself (and any folded continuation pages), then its linked `notes_pages`
        (footnote definitions) appended. A header-less continuation page does NOT pull in its
        predecessor text — the column grammar it needs is injected as a compact metadata line
        (`extra_read_context`) instead, so a deep continuation reads one page rather than its
        whole multi-page run."""
        sc = self._store.summary(ref)
        # The page itself plus any folded continuation pages (the table's tail).
        members = [ref.page, *sc.continuation_pages] if sc is not None else [ref.page]
        refs = [PageRef(stem=ref.stem, page=pg) for pg in members]
        if sc is not None:  # linked footnote/notes pages last (auxiliary context)
            for n in sc.notes_pages:
                r = PageRef(stem=ref.stem, page=n)
                if r not in refs:
                    refs.append(r)
        return refs


@lru_cache(maxsize=4)
def get_officeqa_page_store(pdf_dir: str) -> OfficeQAPageStore:
    """Process-wide store for `pdf_dir` (cached so all branch workers share one
    instance and its caches — mirrors `page_index.store.get_page_store`)."""
    return OfficeQAPageStore(pdf_dir)
