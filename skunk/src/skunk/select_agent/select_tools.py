"""Tool implementations for the SelectAgent.

The SelectAgent is the precision stage that runs over the page-index sem-filter
survivors. Unlike the SearchAgent it has NO ChromaDB and NO embeddings: its corpus
index is the in-memory page-index **catalog** (one `PageCatalogRow` per page, each
carrying the structured per-block summaries the scan produced), and page CONTENT is
read on demand through the `PageStore` (the same source extract reads). The five tools
mirror the SearchAgent's surface, re-pointed at those two stores:

  - `search_corpus`  — a METADATA search over the structured catalog summaries (whole
    corpus reach; lexical match over titles/summaries/headers + date/kind/bulletin
    filters). No vectors.
  - `grep_corpus`    — a regex over the full TEXT of the SURVIVING pages only (the exact
    string the LLM summaries can't give); scoped to the flagged subset.
  - `read_document`  — the full text of any page(s), via `PageStore.text`.
  - `view_figure`    — the rendered page image, via `PageStore.image`.
  - `prune`          — record pages / blocks ruled out (excluded from later searches and
    redacted from the visible context).

Structured returns: every tool returns a *tagged dict* (the tag constants discriminate
the payload shape) so `SelectAgent._blocks_from_output` can turn chunk-bearing payloads
into redactable `ChunkBlock`s — exactly the SearchAgent contract. Prune state is two
mutable sets (`pruned_doc_ids` / `pruned_block_ids`) shared by the search/grep/prune
tools and read by the agent for render-time redaction; `PruneTool` is the single writer.
One SelectAgent (and tool set) is built per branch, so the sets never cross-talk.
"""

from __future__ import annotations

import re
from collections import defaultdict

from skunk.common import PageRef, page_key_to_pageref, pageref_to_doc_key
from skunk.multi_turn_agent import Tool
from skunk.page_index.data_model import ContentBlock, PageCatalogRow
from skunk.page_index.store import PageStore

SEARCH_RESULT_TAG = "__search_result__"
GREP_RESULT_TAG = "__grep_result__"
READ_DOCUMENT_RESULT_TAG = "__read_document_result__"
VIEW_FIGURE_RESULT_TAG = "__view_figure_result__"
PRUNE_RESULT_TAG = "__prune__"

EMPTY_RESULT_MESSAGE = "No matching catalog blocks found."
EMPTY_GREP_MESSAGE = (
    "No matches in the flagged pages' text (grep searches only the flagged subset; "
    "use search_corpus to reach pages beyond the flagged set)."
)


def block_id(doc_id: str, block_index: int) -> str:
    """Stable id for one catalog block — the page's doc_id plus its block position
    (`"1946_11_41#2"`). Carried as a `ChunkBlock.chunk_id` so a `prune` of it redacts the
    block from the visible context."""
    return f"{doc_id}#{block_index}"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def render_block_line(doc_id: str, row: PageCatalogRow, bi: int, block: ContentBlock) -> str:
    """One catalog block rendered as a compact candidate line (no numbers): id, issue/page,
    data span, kind/title, then axis labels and the free-text summary. Shared by the seed
    listing and `search_corpus` results so the agent sees one consistent shape."""
    dates = (
        f"{row.date_interval[0]}..{row.date_interval[1]}" if row.date_interval else "none"
    )
    line = (
        f"[{block_id(doc_id, bi)}] {row.bulletin} p.{row.page}  dates={dates}  "
        f"{block.kind}: {block.title or '(untitled)'}"
    )
    if block.column_headers:
        line += f"\n    cols: {', '.join(block.column_headers)}"
    if block.row_headers:
        line += f"\n    rows: {', '.join(block.row_headers)}"
    if block.summary:
        line += f"\n    summary: {block.summary}"
    return line


class CatalogView:
    """Read-only query surface over the loaded page-index catalog + the surviving pages'
    text + the `PageStore`. Built once per branch and shared by that branch's tools."""

    def __init__(
        self,
        catalog: dict[PageRef, PageCatalogRow],
        survivor_texts: dict[str, str],
        page_store: PageStore,
    ) -> None:
        self._catalog = catalog
        self._survivor_texts = survivor_texts  # doc_id -> full page text (flagged pages only)
        self._page_store = page_store

    def iter_blocks(self):
        """Every catalog block as `(doc_id, row, block_index, block)`, in (bulletin, page)
        order — the whole-corpus surface `search_corpus` ranks/filters over."""
        for ref in sorted(self._catalog, key=lambda r: (r.month or "", r.page or 0)):
            row = self._catalog[ref]
            doc_id = pageref_to_doc_key(ref)
            for bi, block in enumerate(row.content_blocks):
                yield doc_id, row, bi, block

    @property
    def survivor_texts(self) -> dict[str, str]:
        return self._survivor_texts

    @property
    def page_store(self) -> PageStore:
        return self._page_store


def _overlaps_interval(
    interval: tuple[str, str] | None, lo: str | None, hi: str | None
) -> bool:
    """True iff a `(low, high)` YYYY-MM data span overlaps the `[lo, hi]` query window
    (either bound optional → open-ended)."""
    if interval is None:
        return False
    i_lo, i_hi = interval[0][:7], interval[1][:7]
    if lo and i_hi < lo[:7]:
        return False
    if hi and i_lo > hi[:7]:
        return False
    return True


class SearchCorpusTool(Tool):
    name = "search_corpus"

    def __init__(self, view: CatalogView, pruned_doc_ids: set[str], pruned_block_ids: set[str]):
        self._view = view
        self._pruned_doc_ids = pruned_doc_ids
        self._pruned_block_ids = pruned_block_ids

    def __call__(
        self,
        query: str | None = None,
        top_k: int = 50,
        metadata_filter: dict | None = None,
    ) -> dict:
        mf = metadata_filter or {}
        want_bulletin = mf.get("bulletin")
        want_kind = mf.get("kind")
        date_from, date_to = mf.get("date_from"), mf.get("date_to")
        q_tokens = set(_tokens(query)) if query else set()

        # (-score, ordinal) sort key keeps score-desc then catalog order (a stable tie-break,
        # since iter_blocks yields in (bulletin, page) order).
        scored: list[tuple[int, int, str, str, str]] = []  # key fields + (bid, doc_id, line)
        for ordinal, (doc_id, row, bi, block) in enumerate(self._view.iter_blocks()):
            bid = block_id(doc_id, bi)
            if doc_id in self._pruned_doc_ids or bid in self._pruned_block_ids:
                continue
            if want_bulletin and row.bulletin != want_bulletin:
                continue
            if want_kind and block.kind != want_kind:
                continue
            if (date_from or date_to) and not _overlaps_interval(
                row.date_interval, date_from, date_to
            ):
                continue
            if q_tokens:
                haystack = set(
                    _tokens(
                        " ".join(
                            [block.title or "", block.summary or ""]
                            + block.column_headers
                            + block.row_headers
                        )
                    )
                )
                score = len(q_tokens & haystack)
                if score == 0:
                    continue
            else:
                score = 0
            scored.append(
                (-score, ordinal, bid, doc_id, render_block_line(doc_id, row, bi, block))
            )

        scored.sort(key=lambda t: (t[0], t[1]))
        if not scored:
            return {SEARCH_RESULT_TAG: True, "chunks": []}
        chunks = [
            {"chunk_id": bid, "doc_id": doc_id, "text": line}
            for _, _, bid, doc_id, line in scored[: max(1, top_k)]
        ]
        return {SEARCH_RESULT_TAG: True, "chunks": chunks}

    doc = """\
### search_corpus(query: str | None = None, top_k: int = 50, metadata_filter: dict | None = None)
A METADATA search over the structured catalog summaries of the WHOLE corpus (every page of every bulletin) — use it to look beyond the flagged candidates (e.g. to find an earlier/later reprint that tiles a period, or a page a stage-1 filter missed). There are no embeddings: `query` is matched lexically against each block's title, summary, and column/row labels, and results are ranked by how many query terms hit. Each result is one catalog block, labelled with its block id and page; it shows the summary only — `read_document` the page to see the actual text. Already-pruned pages/blocks are excluded.

`metadata_filter` keys (all optional, ANDed):
- `bulletin`: "YYYY-MM" — restrict to one issue.
- `kind`: "table" | "chart" | "prose".
- `date_from` / `date_to`: "YYYY-MM" — keep blocks whose DATA span (what the page reports on) overlaps the window.

```python
# tables reporting on 1940 data, anywhere in the corpus, matching "public debt"
search_corpus("public debt", top_k=30, metadata_filter={"kind": "table", "date_from": "1940-01", "date_to": "1940-12"})
```"""


class GrepCorpusTool(Tool):
    name = "grep_corpus"

    _CHARS_PER_TOKEN = 4

    def __init__(
        self,
        view: CatalogView,
        pruned_doc_ids: set[str],
        max_output_tokens: int,
    ):
        self._view = view
        self._pruned_doc_ids = pruned_doc_ids
        self._max_output_chars = max_output_tokens * self._CHARS_PER_TOKEN

    def __call__(self, pattern: str, limit: int | None = None) -> dict:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return {GREP_RESULT_TAG: True, "groups": [], "error": f"bad regex: {e}"}

        # Match within each flagged page's full text; return the matching lines per page.
        grouped: dict[str, list[str]] = defaultdict(list)
        total = 0
        for doc_id, text in self._view.survivor_texts.items():
            if doc_id in self._pruned_doc_ids:
                continue
            hits = [ln for ln in text.splitlines() if rx.search(ln)]
            if hits:
                grouped[doc_id] = hits
                total += len(hits)

        groups: list[dict] = []
        used = 0
        kept = 0
        truncated = False
        emitted = 0
        for doc_id in sorted(grouped):
            if truncated:
                break
            header = f"\n# doc_id={doc_id}"
            lines: list[str] = []
            for ln in grouped[doc_id]:
                if limit is not None and emitted >= limit:
                    break
                snippet = f"  {ln.strip()}"
                cost = len(snippet) + (len(header) if not lines else 0)
                if (lines or groups) and used + cost > self._max_output_chars:
                    truncated = True
                    break
                used += cost
                lines.append(snippet)
                kept += 1
                emitted += 1
            if lines:
                groups.append({"doc_id": doc_id, "header": header, "text": "\n".join(lines)})
            if limit is not None and emitted >= limit:
                break

        result: dict = {GREP_RESULT_TAG: True, "groups": groups}
        if truncated:
            cap_k = self._max_output_chars // self._CHARS_PER_TOKEN // 1000
            result["truncation_note"] = (
                f"[grep_corpus output truncated: showing {kept} of {total} matching line(s) "
                f"(~{cap_k}k-token cap); narrow the pattern or pass limit=N.]"
            )
        return result

    doc = """\
### grep_corpus(pattern: str, limit: int | None = None)
A regex search over the FULL TEXT of the flagged candidate pages (the exact strings the summaries don't show — a specific series name, footnote, or value). It searches ONLY the flagged subset, not the whole corpus; to reach other pages use `search_corpus`, then `read_document`. Returns the matching lines grouped by page (doc_id). Output is capped; a broad pattern is truncated with a note — narrow it or pass `limit=N`. Pruned pages are excluded.

```python
# which flagged pages mention this exact series, case-insensitive
grep_corpus(r"(?i)statutory debt limitation")
```"""


class ReadDocumentTool(Tool):
    name = "read_document"
    _DOC_TEMPLATE = """\
### read_document(doc_id: str | list[str])
Returns the full text of one or more pages by `doc_id` — use it to CONFIRM a candidate actually carries the target series at the needed granularity before selecting it. Works for any page in the corpus (flagged or found via search_corpus). Don't read more than ~{{ max_pages }} pages per call.

```python
read_document(["1946_11_41", "1947_01_38"])
```"""

    def __init__(self, view: CatalogView, max_pages: int, max_output_chars: int):
        self._view = view
        self._max_output_chars = max_output_chars
        self.doc = self._DOC_TEMPLATE.replace("{{ max_pages }}", str(max_pages))

    def __call__(self, doc_id: str | list[str]) -> dict:
        doc_ids = [doc_id] if isinstance(doc_id, str) else list(doc_id)
        docs: list[dict] = []
        used = 0
        for n, did in enumerate(doc_ids):
            try:
                ref = page_key_to_pageref(did)
                text = self._view.page_store.text(ref)
            except ValueError:
                text = None
            body = text if text is not None else "[no such page (or no text for it)]"
            rendered = f"=== doc_id={did} ===\n{body}"
            remaining = self._max_output_chars - used
            if len(rendered) > remaining:
                dropped = len(doc_ids) - n - 1
                note = (
                    f"\n[truncated: read_document output exceeded {self._max_output_chars} chars"
                    + (f"; {dropped} more requested doc(s) not shown" if dropped else "")
                    + " — read fewer doc_ids per call.]"
                )
                docs.append({"doc_id": did, "text": rendered[: max(0, remaining)] + note})
                break
            docs.append({"doc_id": did, "text": rendered})
            used += len(rendered)
        return {READ_DOCUMENT_RESULT_TAG: True, "docs": docs}


class ViewFigureTool(Tool):
    """Render the full page containing a `<figure id=N>` placeholder and return it as an
    image observation — so the agent can judge a chart/figure candidate it can't read as
    text. Whole-page render (not a bbox crop) for context, via the `PageStore` cache."""

    name = "view_figure"

    def __init__(self, view: CatalogView):
        self._view = view

    @staticmethod
    def _figure_ids(page_text: str) -> list[str]:
        return re.findall(r"<figure id=([^>]+)>", page_text)

    def __call__(self, doc_id: str, figure_id: int | str) -> dict:
        try:
            ref = page_key_to_pageref(doc_id)
        except ValueError:
            return {
                VIEW_FIGURE_RESULT_TAG: True,
                "error": f"doc_id {doc_id!r} is not in the '<year>_<month>_<page>' format",
            }
        page_text = self._view.page_store.text(ref)
        if page_text is None:
            return {VIEW_FIGURE_RESULT_TAG: True, "error": f"no such page: {doc_id!r}"}
        if f"<figure id={figure_id}>" not in page_text:
            visible = self._figure_ids(page_text)
            hint = f"figure ids on this page: {visible}" if visible else "this page has no figures"
            return {
                VIEW_FIGURE_RESULT_TAG: True,
                "error": f"no figure with id={figure_id} on doc_id={doc_id}; {hint}",
            }
        img = self._view.page_store.image(ref)
        if img is None:
            return {
                VIEW_FIGURE_RESULT_TAG: True,
                "error": f"could not render page for doc_id={doc_id} (PDF missing)",
            }
        return {
            VIEW_FIGURE_RESULT_TAG: True,
            "doc_id": doc_id,
            "figure_id": figure_id,
            "mime": img.mime,
            "data": img.data,
        }

    doc = """\
### view_figure(doc_id: str, figure_id: int | str)
When you read a page and see a `<figure id=N>` placeholder (a chart NOT in the text), call this to actually *see* it — it returns an image of the whole page. Use it to judge a candidate whose answer may live in a chart (the catalog marks such blocks `kind=chart`).

```python
view_figure("2002_12_8", 5)
```"""


class PruneTool(Tool):
    name = "prune"

    def __init__(self, pruned_doc_ids: set[str], pruned_block_ids: set[str]):
        self._pruned_doc_ids = pruned_doc_ids
        self._pruned_block_ids = pruned_block_ids

    def __call__(
        self,
        doc_ids: list[str] | None = None,
        block_ids: list[str] | None = None,
    ) -> dict:
        new_docs = set(doc_ids or ()) - self._pruned_doc_ids
        new_blocks = set(block_ids or ()) - self._pruned_block_ids
        self._pruned_doc_ids.update(new_docs)
        self._pruned_block_ids.update(new_blocks)
        return {
            PRUNE_RESULT_TAG: True,
            "new_doc_count": len(new_docs),
            "new_block_count": len(new_blocks),
            "total_docs": len(self._pruned_doc_ids),
            "total_blocks": len(self._pruned_block_ids),
        }

    doc = """\
### prune(doc_ids: list[str] | None = None, block_ids: list[str] | None = None)
Record pages (`doc_id`) and/or blocks (block id, e.g. "1946_11_41#2") you've ruled out. Pruned items are excluded from later `search_corpus` / `grep_corpus` results and dropped from your visible context. Use it aggressively on flagged candidates you've rejected to stay focused.

```python
prune(block_ids=["1946_11_41#0"], doc_ids=["1947_01_38"])
```"""
