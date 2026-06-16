"""Tool implementations for the SelectAgent.

The SelectAgent is the precision stage that runs over the page-index sem-filter
survivors. Unlike the SearchAgent it has NO ChromaDB and NO embeddings: whole-corpus reach
is a prebuilt **SQLite FTS5 index** the agent queries through a fixed-column search tool, and
page CONTENT is read on demand through the `PageStore` (the same source extract reads). The four tools:

  - `search_corpus` — query the corpus search index (one FTS5 table `pages`): the agent supplies
    only the `WHERE` / `ORDER BY` / `LIMIT` clauses and the tool always returns `doc_id, bulletin,
    title` (full-text rank with MATCH / bm25, filter on metadata) — triage on title, then confirm.
  - `grep_corpus`   — a regex over the full TEXT of pages (the exact string the summaries
    can't give); defaults to the flagged subset, or pass `doc_ids` to grep any pages.
  - `read_document` — the full text of any page(s), via `PageStore.text`.
  - `view_figure`   — the rendered page image, via `PageStore.image`.

Structured returns: every tool returns a *tagged dict* (the tag constants discriminate
the payload shape) so `SelectAgent._blocks_from_output` can render each payload. One
SelectAgent (and tool set) is built per branch.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections import defaultdict

from skunk.common import page_key_to_pageref
from skunk.multi_turn_agent import Tool
from skunk.page_index.store import PageStore

SEARCH_RESULT_TAG = "__search_result__"
GREP_RESULT_TAG = "__grep_result__"
READ_DOCUMENT_RESULT_TAG = "__read_document_result__"
VIEW_FIGURE_RESULT_TAG = "__view_figure_result__"

EMPTY_GREP_MESSAGE = (
    "No matches in the searched pages' text. By default grep_corpus searches only the flagged "
    "candidate pages; pass doc_ids=[...] (e.g. pages found via search_corpus) to grep others."
)


# One read-only SQLite connection per worker thread, keyed by db path, lazily opened and
# reused across questions. The search index is built once and never written at runtime, so we
# open it `immutable` (SQLite skips ALL locking → lock-free concurrent readers) and keep one
# connection per thread — sharing a single connection across threads either errors
# (check_same_thread) or serializes on SQLite's internal mutex, defeating the parallelism.
_conns = threading.local()

def _ro_connection(path: str) -> sqlite3.Connection:
    cache: dict[str, sqlite3.Connection] | None = getattr(_conns, "by_path", None)
    if cache is None:
        cache = _conns.by_path = {}
    con = cache.get(path)
    if con is None:
        con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        # Read-path tuning for a hot, read-only FTS workload: memory-map the whole file so
        # reads come from the OS page cache (shared across the per-thread connections — mmap
        # is virtual address space, NOT N× physical RAM), and keep the BM25 `ORDER BY` sort
        # b-trees in memory.
        con.execute("PRAGMA mmap_size=536870912")  # 512 MiB ≥ the artifact
        con.execute("PRAGMA temp_store=MEMORY")
        con.execute("PRAGMA cache_size=-32000")  # 32 MiB page cache per connection
        cache[path] = con
    return con


class CatalogView:
    """Read-only query surface over the surviving pages' text + the `PageStore`. Built once
    per branch and shared by that branch's tools (grep / read / view)."""

    def __init__(
        self,
        survivor_texts: dict[str, str],
        page_store: PageStore,
    ) -> None:
        self._survivor_texts = survivor_texts  # doc_id -> full page text (flagged pages only)
        self._page_store = page_store

    @property
    def survivor_texts(self) -> dict[str, str]:
        return self._survivor_texts

    @property
    def page_store(self) -> PageStore:
        return self._page_store


def _render_rows(columns: list[str], rows: list[tuple], truncated: bool, max_chars: int) -> str:
    """Render a SQL result set as a compact `col | col` table, char-capped, with a note when
    rows were dropped (hit the row cap) or the text was clipped."""
    if not columns:
        return "[query ran; no columns returned]"
    if not rows:
        return "[0 rows]"
    out = [" | ".join(columns)]
    for r in rows:
        out.append(" | ".join("" if v is None else str(v) for v in r))
    text = "\n".join(out)
    note = ""
    if len(text) > max_chars:
        text = text[:max_chars]
        note = "\n[... output clipped; narrow your WHERE or lower LIMIT]"
    if truncated:
        note += f"\n[showing first {len(rows)} rows; add or lower LIMIT for a complete set]"
    return text + note


class SearchCorpusTool(Tool):
    name = "search_corpus"
    _COLUMNS = "doc_id, bulletin, title"  # the ONLY columns this tool ever returns
    _MAX_ROWS = 50  # hard cap on rows returned, regardless of the query's LIMIT
    _MAX_CHARS = 30_000  # char cap on the rendered result text (a backstop)
    _TIMEOUT_S = 5.0  # wall-clock budget per query (a pathological scan is interrupted)
    # Reject anything that would break out of the single assembled SELECT — statement
    # separators and SQL comments. The read-only `immutable` connection already blocks writes;
    # this keeps the fixed `SELECT doc_id, bulletin, title` column contract intact.
    _FORBIDDEN = re.compile(r";|--|/\*")

    def __init__(self, search_index_path: str):
        self._index_path = str(search_index_path)

    def __call__(self, where: str, order_by: str | None = None, limit: int | None = None) -> dict:
        where_s = (where or "").strip()
        if not where_s:
            return {
                SEARCH_RESULT_TAG: True,
                "text": "[error] `where` is required, e.g. \"pages MATCH 'public debt' AND has_table=1\"",
            }
        order_s = (order_by or "").strip()
        for frag in (where_s, order_s):
            if frag and self._FORBIDDEN.search(frag):
                return {
                    SEARCH_RESULT_TAG: True,
                    "text": "[error] `where`/`order_by` take a single clause only — no ';', SQL comments, or extra statements",
                }
        sql = f"SELECT {self._COLUMNS} FROM pages WHERE {where_s}"
        if order_s:
            sql += f" ORDER BY {order_s}"
        if limit is not None:
            try:
                n = int(limit)
            except (TypeError, ValueError):
                return {SEARCH_RESULT_TAG: True, "text": "[error] `limit` must be an integer"}
            if n > 0:
                sql += f" LIMIT {n}"
        con = _ro_connection(self._index_path)
        # Bound a pathological query with a watchdog that `interrupt()`s from another thread.
        # (A SQLite progress handler would also work, but it calls back into Python every N
        # opcodes — reacquiring the GIL and serializing the otherwise-parallel readers.)
        watchdog = threading.Timer(self._TIMEOUT_S, con.interrupt)
        watchdog.daemon = True
        watchdog.start()
        try:
            cur = con.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(self._MAX_ROWS + 1)
        except sqlite3.Error as e:
            # SQL / operational errors (a bad MATCH/clause, an interrupt timeout) are handed
            # back so the agent can fix its query and retry, REPL-style.
            return {SEARCH_RESULT_TAG: True, "text": f"[error] {type(e).__name__}: {e}"}
        finally:
            watchdog.cancel()
        truncated = len(rows) > self._MAX_ROWS
        text = _render_rows(columns, rows[: self._MAX_ROWS], truncated, self._MAX_CHARS)
        return {SEARCH_RESULT_TAG: True, "text": text}

    doc = """\
### search_corpus(where: str, order_by: str | None = None, limit: int | None = None)
Find candidate pages beyond your seeded shortlist. You write only the SQL `WHERE` clause (required) and optional `ORDER BY` / `LIMIT`; the tool always returns the columns `doc_id, bulletin, title` — triage on the title, then use read_document / grep_corpus to confirm a page actually carries the target series. The index is one SQLite FTS5 table `pages`, one row per page of every bulletin. At most 50 rows are returned.

Columns you can filter / order on:
- `pages MATCH '<q>'` — full-text over each page's block titles, summaries, and table row/column headers. Supports `AND` / `OR` / `NOT`, `"exact phrase"`, `prefix*`, `NEAR(a b, 5)`. For best-first ranking pass `order_by="bm25(pages)"` (more-negative = better).
- `bulletin` — issue 'YYYY-MM'.  `page` — 1-based PDF page.  `title` — primary table title.
- `lo`, `hi` — the page's DATA span bounds 'YYYY-MM' (what it REPORTS ON; NULL if undatable). Date overlap = `lo<='<end>' AND hi>='<start>'`.
- `has_table`, `has_chart`, `has_prose` — 1 if the page has a block of that kind.

```python
# tables about public debt reporting on 1940 data, best matches first
search_corpus(where="pages MATCH 'public AND debt' AND has_table=1 AND lo<='1940-12' AND hi>='1940-01'", order_by="bm25(pages)", limit=20)
# an exact phrase, any issue
search_corpus(where="pages MATCH '\\"statutory debt limitation\\"'", limit=20)
# every page of one issue, in printed order
search_corpus(where="bulletin='1946-11'", order_by="page")
```"""


class GrepCorpusTool(Tool):
    name = "grep_corpus"

    _CHARS_PER_TOKEN = 4

    def __init__(
        self,
        view: CatalogView,
        max_output_tokens: int,
    ):
        self._view = view
        self._max_output_chars = max_output_tokens * self._CHARS_PER_TOKEN

    def __call__(
        self,
        pattern: str,
        doc_ids: str | list[str] | None = None,
        limit: int | None = None,
    ) -> dict:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return {GREP_RESULT_TAG: True, "groups": [], "error": f"bad regex: {e}"}

        # Which pages to grep: the explicit `doc_ids` (any page, read via the PageStore — e.g.
        # pages found with search_corpus), else the flagged survivor subset.
        if doc_ids is not None:
            ids = [doc_ids] if isinstance(doc_ids, str) else list(doc_ids)
            texts: dict[str, str] = {}
            for did in ids:
                try:
                    text = self._view.page_store.text(page_key_to_pageref(did))
                except ValueError:
                    text = None
                if text is not None:
                    texts[did] = text
        else:
            texts = self._view.survivor_texts

        # Match within each page's full text; return the matching lines per page.
        grouped: dict[str, list[str]] = defaultdict(list)
        total = 0
        for doc_id, text in texts.items():
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
### grep_corpus(pattern: str, doc_ids: str | list[str] | None = None, limit: int | None = None)
A regex search over the FULL TEXT of pages (the exact strings the summaries don't show — a specific series name, footnote, or value). By DEFAULT it searches only the flagged candidate pages; pass `doc_ids=[...]` to grep specific pages instead (any page in the corpus, e.g. ones you found with `search_corpus`). Returns the matching lines grouped by page (doc_id). Output is capped; a broad pattern is truncated with a note — narrow it or pass `limit=N`.

```python
# which flagged pages mention this exact series, case-insensitive
grep_corpus(r"(?i)statutory debt limitation")
# grep a specific page found via search_corpus
grep_corpus(r"(?i)statutory debt limitation", doc_ids=["1980_04_85"])
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
