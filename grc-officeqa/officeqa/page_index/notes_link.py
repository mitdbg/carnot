"""Build pass: link data pages to the notes pages that define their footnotes.

The scan sets two flags per page: `references_external_notes` (this page's data carries
footnote markers — superscripts, `*`, `†` — defined elsewhere) and `is_notes_page` (this page
IS those footnote definitions, an end-of-section "Footnotes" block qualifying data on other
pages). This pass connects the two with a deterministic sequence heuristic: a notes page
"closes" the run of referencing data pages before it.

Walking each bulletin in physical-page order, every `references_external_notes` content page is
held pending; on reaching an `is_notes_page` that page is recorded as the `notes_pages` link for
all pending data pages, and the pending set clears. (Notes sit at the END of their section, so a
data page's notes are the next notes page after it, not crossing an intervening notes page.) Data
pages trailing the final notes page stay unlinked.

Annotation-only and no LLM: it writes `notes_pages` into each data page's scan record in place.
A reader expands a selected data page to also fetch its `notes_refs` so extract/compute sees the
footnote definitions. Driven per bulletin by the pipeline's `notes_link` stage.
"""

from __future__ import annotations


def link_notes(scans: dict[str, dict]) -> dict[int, list[int]]:
    """`{data page: [notes page, ...]}` for one bulletin's scans. Each `is_notes_page` is linked
    to the `references_external_notes` content pages since the previous notes page. Pages with no
    following notes page are omitted (no link)."""
    links: dict[int, list[int]] = {}
    pending: list[int] = []
    for p in sorted((int(p) for p in scans), key=int):
        s = scans[str(p)]
        if s.get("page_role") != "content":
            continue
        if s.get("is_notes_page"):
            for d in pending:
                links.setdefault(d, []).append(p)
            pending = []
        elif s.get("references_external_notes"):
            pending.append(p)
    return links


def apply_links(scans: dict[str, dict], links: dict[int, list[int]]) -> list[int]:
    """Write each data page's `notes_pages` into its scan record (in place). Returns the linked
    pages (ascending) — the catalog patch reprojects exactly these rows."""
    for p, notes in links.items():
        scans[str(p)]["notes_pages"] = notes
    return sorted(links)
