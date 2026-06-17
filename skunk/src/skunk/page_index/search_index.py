"""Corpus search index — the prebuilt SQLite FTS5 page index and the `search_corpus`
tool that runs SQL against it.

Salvaged from the `skunk-qatfd-sa` SelectAgent and re-keyed for the new PageScan-based
page index (per-doc `scans/<source>.json`, integer-year `date_interval`s). It is the one
whole-corpus reach primitive that needs NO ChromaDB and NO embeddings: an offline build
projects every content page's scan into one FTS5 table `pages`, and the agent reaches the
corpus by writing SQL `WHERE` clauses against it.

Two halves, mirroring the branch:

  - BUILD path (`build_search_index` / `iter_scan_rows` / the `build` CLI): read the
    `scans/*.json` artifact, project each `content` page into one denormalized FTS5 row —
    the searchable `summary` column (block titles + summaries + table headers) plus
    `UNINDEXED` metadata the agent filters on — and write `search_index.sqlite`.
  - QUERY path (`SearchCorpusTool`): a fixed-column read-only search tool. The agent supplies
    only the `WHERE` / `ORDER BY` / `LIMIT`; the tool always returns `source, page, title`.

This module is intentionally NOT wired into retrieval — it builds and queries standalone.
Page CONTENT (full text / images) is still served by `store.PageStore`; this index only
answers "which pages might carry the target series".

Build it against the competition artifact with:

    python3 -m skunk.page_index.search_index build competition_page_index

and smoke-test a query with:

    python3 -m skunk.page_index.search_index query competition_page_index \\
        --where "pages MATCH 'public AND debt' AND has_table=1" --order-by "bm25(pages)" --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Iterator

from skunk.multi_turn_agent import Tool

from .data_model import SCANS_SUBDIR, SEARCH_INDEX_FILE, PageCatalogRow

# ---------------------------------------------------------------------------
# Build path — project the scans into the FTS5 index
# ---------------------------------------------------------------------------


def iter_scan_rows(scans_dir: Path) -> Iterator[PageCatalogRow]:
    """Every content page across the `scans/*.json` artifact, projected to its catalog row.

    Sources the FTS build straight from the raw scans (the competition artifact ships no
    `catalog/` dir). Projection is delegated to the pipeline's `_catalog_row` — the single
    source of truth for scan → catalog — imported lazily so importing this module for the
    query tool never pulls the build's heavy LLM deps. Non-content pages project to None and
    are dropped."""
    from .pipeline import _catalog_row  # lazy: heavy import, build-time only

    for path in sorted(Path(scans_dir).glob("*.json")):
        data = json.loads(path.read_text())
        source = data.get("bulletin") or path.stem
        for page_str, scan in sorted(
            data.get("scans", {}).items(), key=lambda kv: int(kv[0])
        ):
            row = _catalog_row(source, int(page_str), scan)
            if row is not None:
                yield row


def build_search_index(rows: Iterable[PageCatalogRow], out: Path) -> int:
    """Build the read-only SQLite FTS5 index the `search_corpus` tool runs SQL against: ONE
    FTS5 table `pages`, one row per catalog PAGE. The single indexed `summary` column carries
    the page's searchable text (block titles + summaries + table headers, for MATCH / bm25);
    the rest are `UNINDEXED` columns the agent filters/selects on — `source` (the doc stem),
    `page`, `printed_page`, the integer-year `date_interval` bounds `lo`/`hi`, and the per-kind
    `has_*` flags, plus `title`.

    One denormalized table (no joins for the agent's SQL); metadata is `UNINDEXED` because the
    workload is MATCH-driven (filters apply post-match on the small matched set). Built to a
    temp file then atomically renamed. Returns the page count. No LLM calls — a pure
    projection of the catalog."""
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.executescript(
            """
            PRAGMA page_size=8192;
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE VIRTUAL TABLE pages USING fts5(
                summary,
                source UNINDEXED,
                page UNINDEXED,
                printed_page UNINDEXED,
                lo UNINDEXED,
                hi UNINDEXED,
                has_table UNINDEXED,
                has_chart UNINDEXED,
                has_prose UNINDEXED,
                title UNINDEXED
            );
            """
        )
        batch: list[tuple] = []
        rowid = 0
        for row in rows:
            rowid += 1
            kinds = {b.kind for b in row.content_blocks}
            parts: list[str] = []
            for b in row.content_blocks:
                if b.title:
                    parts.append(b.title)
                if b.summary:
                    parts.append(b.summary)
                parts.extend(b.column_headers)
                parts.extend(b.row_headers)
            lo = row.date_interval[0] if row.date_interval else None
            hi = row.date_interval[1] if row.date_interval else None
            batch.append(
                (
                    rowid,
                    " ".join(parts),
                    row.source,
                    row.page,
                    row.printed_page,
                    lo,
                    hi,
                    int("table" in kinds),
                    int("chart" in kinds),
                    int("prose" in kinds),
                    row.primary_title,
                )
            )
        con.executemany(
            "INSERT INTO pages(rowid, summary, source, page, printed_page, lo, hi, "
            "has_table, has_chart, has_prose, title) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            batch,
        )
        # Merge the FTS5 index into a single segment — fewer b-tree segments to scan per MATCH
        # (faster queries) and a smaller file. One-time build cost; pays off on every read.
        con.execute("INSERT INTO pages(pages) VALUES('optimize')")
        con.commit()
    finally:
        con.close()
    os.replace(tmp, out)
    return rowid


# ---------------------------------------------------------------------------
# Query path — the read-only FTS search tool
# ---------------------------------------------------------------------------

SEARCH_RESULT_TAG = "__search_result__"

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


def _render_rows(
    columns: list[str], rows: list[tuple], truncated: bool, max_chars: int
) -> str:
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
    _COLUMNS = "source || '#' || page AS doc_id, page, title"  # the ONLY columns this tool ever returns
    _MAX_ROWS = 50  # hard cap on rows returned, regardless of the query's LIMIT
    _MAX_CHARS = 30_000  # char cap on the rendered result text (a backstop)
    _TIMEOUT_S = 5.0  # wall-clock budget per query (a pathological scan is interrupted)
    # Reject anything that would break out of the single assembled SELECT — statement
    # separators and SQL comments. The read-only `immutable` connection already blocks writes;
    # this keeps the fixed `SELECT source, page, title` column contract intact.
    _FORBIDDEN = re.compile(r";|--|/\*")

    def __init__(self, search_index_path: str | Path):
        self._index_path = str(search_index_path)

    def __call__(
        self, where: str, order_by: str | None = None, limit: int | None = None
    ) -> dict:
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
Find candidate pages beyond your seeded shortlist. You write only the SQL `WHERE` clause (required) and optional `ORDER BY` / `LIMIT`; the tool always returns the columns `doc_id, page, title` (where `doc_id` is `<source>#<page>`) — triage on the title, then copy `doc_id` VERBATIM into read_document / grep_corpus / view_figure to confirm a page actually carries the target series. The index is one SQLite FTS5 table `pages`, one row per content page of every document. At most 50 rows are returned.

Columns you can filter / order on:
- `pages MATCH '<q>'` — full-text over each page's block titles, summaries, and table row/column headers. Supports `AND` / `OR` / `NOT`, `"exact phrase"`, `prefix*`, `NEAR(a b, 5)`. For best-first ranking pass `order_by="bm25(pages)"` (more-negative = better).
- `source` — the document filename stem (e.g. 'combined_statement__modern__2024__c40').  `page` — 1-based PDF page.  `printed_page` — the page's own printed footer label.  `title` — primary table title.
- `lo`, `hi` — the page's DATA span bounds as integer YEARS (what it REPORTS ON; NULL if undatable). Year overlap = `lo<=<end> AND hi>=<start>`.
- `has_table`, `has_chart`, `has_prose` — 1 if the page has a block of that kind.

```python
# tables about public debt reporting on 1940 data, best matches first
search_corpus(where="pages MATCH 'public AND debt' AND has_table=1 AND lo<=1940 AND hi>=1940", order_by="bm25(pages)", limit=20)
# an exact phrase, any document
search_corpus(where="pages MATCH '\\"statutory debt limitation\\"'", limit=20)
# every content page of one document, in printed order
search_corpus(where="source='combined_statement__historical__cs-1872'", order_by="page")
```"""


# ---------------------------------------------------------------------------
# CLI — build / query the index standalone
# ---------------------------------------------------------------------------


def _default_index_path(root: Path) -> Path:
    return Path(root) / SEARCH_INDEX_FILE


def _cmd_build(args: argparse.Namespace) -> None:
    root = Path(args.root)
    scans_dir = Path(args.scans_dir) if args.scans_dir else root / SCANS_SUBDIR
    out = Path(args.out) if args.out else _default_index_path(root)
    if out.exists() and not args.force:
        print(f"{out} exists; pass --force to rebuild")
        return
    n = build_search_index(iter_scan_rows(scans_dir), out)
    print(f"built {out} with {n} page rows from {scans_dir}")


def _cmd_query(args: argparse.Namespace) -> None:
    root = Path(args.root)
    index_path = Path(args.index) if args.index else _default_index_path(root)
    if not index_path.exists():
        raise SystemExit(f"index not found: {index_path} (build it first)")
    tool = SearchCorpusTool(index_path)
    res = tool(args.where, order_by=args.order_by, limit=args.limit)
    print(res["text"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="build the FTS5 search index from the scans")
    pb.add_argument("root", help="page-index artifact root (holds scans/)")
    pb.add_argument("--scans-dir", help="override the scans dir (default: <root>/scans)")
    pb.add_argument("--out", help="output index path (default: <root>/search_index.sqlite)")
    pb.add_argument("--force", action="store_true", help="rebuild even if the index exists")
    pb.set_defaults(func=_cmd_build)

    pq = sub.add_parser("query", help="run one search_corpus query against the index")
    pq.add_argument("root", help="page-index artifact root (holds search_index.sqlite)")
    pq.add_argument("--index", help="override the index path")
    pq.add_argument("--where", required=True, help="the SQL WHERE clause")
    pq.add_argument("--order-by", help="optional ORDER BY clause, e.g. 'bm25(pages)'")
    pq.add_argument("--limit", type=int, help="optional LIMIT")
    pq.set_defaults(func=_cmd_query)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
