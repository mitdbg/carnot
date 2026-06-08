"""Page store — the page-index artifact's source of truth for page CONTENT.

The metadata catalog (`catalog/*.jsonl`) says *which* pages answer a query; this store
serves *what's on them*, with exactly two access paths per page:

    text(ref)   -> the page's text (read from `pages/<bulletin>.json`)
    image(ref)  -> the page's rendered image (rendered on demand at 200 DPI, cached to disk)

It is dumb, directory-backed, and thread-safe — multiple parallel branch workers hit one
shared instance. Continuation merge is NOT the store's concern: the build writes each anchor
page's already-merged text (its + folded continuation pages' text, plus the figure note)
under the anchor's page key, and drops the continuation pages. The store just reads it back.

On-disk layout under the artifact root (`page_index_root()` / the build's `--build-dir`):

    pages/<bulletin>.json          {"<page>": "<text>"}    (written by the `page_store` stage)
    renders/<bulletin>/<page>.png  200-DPI image cache, written on first `image()` read

`extract.py` reads text and vision-tier images only through `PageStore`, so the request path
never touches the corpus parsed-JSON/PDFs directly (the store encapsulates rendering)."""

from __future__ import annotations

import base64
import json
import os
import threading
from functools import lru_cache
from pathlib import Path

from skunk.common import B64Image, PageRef
from skunk.corpus import render_page_b64
from skunk.errors import StepFailed

from .data_model import CATALOG_SUBDIR, PAGES_SUBDIR, RENDERS_SUBDIR, PageCatalogRow, page_index_root

RENDER_DPI = 200


def summarize_row(row: PageCatalogRow) -> str:
    """Render a catalog row's content blocks as one compact, greppable summary line — the
    kind/title/headers/summary digest the page-select index shows (no numeric values).
    Public so callers holding a row can format it without a second catalog lookup."""
    blocks: list[str] = []
    for b in row.content_blocks:
        part = f"{b.kind}: {b.title or '(untitled)'}"
        if b.column_headers:
            part += f" [cols: {', '.join(b.column_headers)}]"
        if b.row_headers:
            part += f" [rows: {', '.join(b.row_headers)}]"
        if b.summary:
            part += f" — {b.summary}"
        blocks.append(part)
    return " ;; ".join(blocks) if blocks else "(no content blocks)"


class PageStore:
    """Thread-safe, directory-backed reader over the page store. Per-bulletin text maps are
    loaded once and memoized; images are rendered on demand at `RENDER_DPI` and cached to
    `renders/`. Built from the artifact `root` and the corpus `pdf_dir` (the latter only for
    on-demand rendering)."""

    def __init__(self, root: Path, pdf_dir: Path | str) -> None:
        self._pages_dir = Path(root) / PAGES_SUBDIR
        self._catalog_dir = Path(root) / CATALOG_SUBDIR
        self._renders_dir = Path(root) / RENDERS_SUBDIR
        self._pdf_dir = pdf_dir
        self._lock = threading.Lock()
        self._text: dict[str, dict[int, str]] = {}       # bulletin -> {page: text}
        self._catalog: dict[str, dict[int, PageCatalogRow]] = {}  # bulletin -> {page: anchor row}
        self._page_locks: dict[str, threading.Lock] = {}  # render key -> lock

    # -- text -----------------------------------------------------------------

    def text(self, ref: PageRef) -> str | None:
        """The page's stored text (the anchor's already-merged text), or None when absent."""
        if ref.month is None or ref.page is None:
            return None
        return self._bulletin_text(ref.month).get(int(ref.page))

    def _bulletin_text(self, bulletin: str) -> dict[int, str]:
        """`{page: text}` for one bulletin, loaded once and memoized (thread-safe). Raises if
        the store was never built; a single missing bulletin file is treated as empty."""
        with self._lock:
            cached = self._text.get(bulletin)
        if cached is not None:
            return cached
        if not self._pages_dir.exists():
            raise StepFailed("extract", f"page store not built ({self._pages_dir} missing); "
                             "run the page-index build pipeline first.")
        path = self._pages_dir / f"{bulletin}.json"
        raw = json.loads(path.read_text()) if path.exists() else {}
        entries = {int(p): t for p, t in raw.items()}
        with self._lock:
            return self._text.setdefault(bulletin, entries)

    # -- catalog summaries ----------------------------------------------------

    def catalog_row(self, ref: PageRef) -> PageCatalogRow | None:
        """The page's catalog row, or None when absent. An anchor page and any of its folded
        continuation pages both resolve to the anchor's row."""
        if ref.month is None or ref.page is None:
            return None
        return self._bulletin_catalog(ref.month).get(int(ref.page))

    def summary(self, ref: PageRef) -> str | None:
        """A compact one-line summary of the page's content blocks (titles / headers / block
        summaries — no numbers), read from the catalog. None when the page has no catalog row."""
        row = self.catalog_row(ref)
        return None if row is None else summarize_row(row)

    def _bulletin_catalog(self, bulletin: str) -> dict[int, PageCatalogRow]:
        """`{page: anchor row}` for one bulletin, loaded once and memoized (thread-safe). Every
        member page (anchor + folded continuations) keys the anchor's row, so a continuation ref
        resolves to its anchor. Raises if the catalog was never built; a single missing bulletin
        file is treated as empty."""
        with self._lock:
            cached = self._catalog.get(bulletin)
        if cached is not None:
            return cached
        if not self._catalog_dir.exists():
            raise StepFailed("retrieve", f"catalog not built ({self._catalog_dir} missing); "
                             "run the page-index build pipeline first.")
        path = self._catalog_dir / f"{bulletin}.jsonl"
        rows = (
            [PageCatalogRow.from_json(line) for line in path.read_text().splitlines() if line]
            if path.exists() else []
        )
        entries: dict[int, PageCatalogRow] = {}
        for row in rows:
            for p in row.member_pages:
                entries[p] = row
        with self._lock:
            return self._catalog.setdefault(bulletin, entries)

    # -- image (render on demand + cache) -------------------------------------

    def image(self, ref: PageRef) -> B64Image | None:
        """The page's 200-DPI PNG: served from the `renders/` cache, else rendered once and
        cached. None when the ref is incomplete or the PDF can't be rendered. Concurrent calls
        for the same page render at most once (per-page lock); different pages run in parallel."""
        if ref.month is None or ref.page is None:
            return None
        img = read_cached_image(self._renders_dir, ref.month, ref.page)
        if img is not None:
            return img
        with self._lock_for(f"{ref.month}/{ref.page}"):
            img = read_cached_image(self._renders_dir, ref.month, ref.page)  # re-check under lock
            if img is not None:
                return img
            if render_to_cache(ref.month, ref.page, self._pdf_dir, self._renders_dir) == "skipped":
                return None
            return read_cached_image(self._renders_dir, ref.month, ref.page)

    def _lock_for(self, key: str) -> threading.Lock:
        with self._lock:
            return self._page_locks.setdefault(key, threading.Lock())


def render_cache_path(renders_dir: str | Path, month: str, page: int) -> Path:
    """The page's image-cache path: `<renders_dir>/<month>/<page>.png`."""
    return Path(renders_dir) / month / f"{page}.png"


def read_cached_image(renders_dir: str | Path, month: str, page: int) -> B64Image | None:
    """The page's cached PNG as a `B64Image`, or None if it hasn't been rendered yet."""
    return _read_png(render_cache_path(renders_dir, month, page))


def render_to_cache(month: str, page: int, pdf_dir: str | Path, renders_dir: str | Path) -> str:
    """Ensure the page's `renders/<month>/<page>.png` exists at `RENDER_DPI`, rendering it once
    if absent. Returns "cached" (already present), "rendered" (newly written), or "skipped" (no
    PDF / incomplete ref). Top-level and picklable so a pre-render stage can fan it out across a
    process pool — page rasterization is CPU-bound and GIL-serialized, so it needs processes, not
    threads; the atomic write keeps concurrent renders of the same page safe."""
    cache_path = render_cache_path(renders_dir, month, page)
    if cache_path.exists():
        return "cached"
    try:
        img = render_page_b64(month, page, dpi=RENDER_DPI, fmt="png", pdf_dir=pdf_dir)
    except Exception:  # noqa: BLE001 — bad page index / corrupt PDF: skip, never abort the batch
        return "skipped"
    if img is None:
        return "skipped"
    _atomic_write_bytes(cache_path, base64.standard_b64decode(img.data))
    return "rendered"


def _read_png(path: Path) -> B64Image | None:
    """Load a cached PNG into a `B64Image`, or None if it isn't there."""
    if not path.exists():
        return None
    return B64Image(mime="image/png", data=base64.standard_b64encode(path.read_bytes()).decode())


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` via a tmp file + rename (no partial files; safe under concurrency)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.rename(path)


@lru_cache(maxsize=4)
def get_page_store(pdf_dir: str) -> PageStore:
    """Process-wide `PageStore` for the current artifact root + `pdf_dir` (cached so all
    branch workers share one instance and its caches)."""
    return PageStore(page_index_root(), pdf_dir)
