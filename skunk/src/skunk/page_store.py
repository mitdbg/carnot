"""`PageContentStore` — the protocol extract reads page content through.

The library's extract operator is corpus-agnostic: everything it knows about a
page comes through this seam. An application supplies the implementation (e.g.
grc-officeqa's Treasury store, which serves parsed-JSON text, renders PDF pages
on demand, and augments continuation pages with inherited column headers) and
injects it via `Orchestrator(page_store=...)` → `ExecutionContext.page_store`.

The two content methods (`text` / `image`) are the essential surface. The three
structure methods let a corpus with page-level structure metadata (continuation
runs, linked footnote pages, content-block titles) improve extraction; a corpus
without it returns the trivial values noted on each method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from skunk.common import B64Image, PageRef


@runtime_checkable
class PageContentStore(Protocol):
    def text(self, ref: PageRef) -> str | None:
        """The page's extracted text, or None when the store has none (the vision
        tier can still read the page image)."""
        ...

    def image(self, ref: PageRef) -> B64Image | None:
        """The page rendered as an image, or None when it cannot be rendered."""
        ...

    def page_metadata(self, refs: list[PageRef]) -> str:
        """A compact structural summary of the pages (e.g. one line per content
        block: kind + title), prepended to the extract prompt as context — never a
        source of values. "" when the corpus has no structure metadata."""
        ...

    def extra_read_context(self, ref: PageRef) -> str:
        """Extra context a page needs to be read standalone (e.g. a header-less
        continuation page's inherited column grammar). "" for a normal page."""
        ...

    def read_group(self, ref: PageRef) -> list[PageRef]:
        """Every page one extract call must read to interpret `ref` — `ref` itself
        plus any dependent pages (folded continuation tails, linked footnote
        pages). `[ref]` when pages are self-contained."""
        ...
