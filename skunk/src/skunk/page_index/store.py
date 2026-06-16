"""Page store — the page-index artifact's source of truth for page CONTENT.

It serves *what's on a page*, with three access paths per page, all keyed by `PageRef`
(`month` carries the doc-id stem, `page` the 1-based PDF page):

    text(ref)    -> the page's text   (read from `pages/<doc_id>.json`)
    image(ref)   -> the page's image  (rendered on demand at 200 DPI, cached to `renders/`)
    summary(ref) -> the page's raw `PageScan` (read from `scans/<doc_id>.json`)

It is dumb, directory-backed, and thread-safe — multiple parallel branch workers hit one
shared instance. The store does NOT expand or merge pages: it serves one physical page's
content. Continuation chains and notes-page expansion are the reader's concern — `extract.py`
walks the `PageScan` flags (`is_continuation` / `continuation_pages` / `notes_pages`) and
fetches the dependent pages itself.

On-disk layout under the artifact root (`page_index_root()` / `SKUNK_PAGE_INDEX_DIR`,
pointed at `competition_page_index/`):

    scans/<doc_id>.json            {"scans": {"<page>": PageScan, ...}, ...}
    pages/<doc_id>.json            {"<page>": "<text>"}
    renders/<doc_id>/<page>.png    200-DPI image cache (populated by `scripts/prerender_cache.py`;
                                   a miss is rendered on demand)

`extract.py` reads scans, text, and vision-tier images only through `PageStore`, so the
request path never touches the corpus parsed-JSON/PDFs directly (the store encapsulates
rendering)."""

from __future__ import annotations

import base64
import json
import os
import threading
from functools import lru_cache
from pathlib import Path

from skunk.common import B64Image, PageRef, render_page_b64
from skunk.errors import StepFailed

from .data_model import (
    PAGES_SUBDIR,
    RENDERS_SUBDIR,
    SCANS_SUBDIR,
    page_index_root,
)
from .scan import PageScan

RENDER_DPI = 200


class PageStore:
    """Thread-safe, directory-backed reader over the page store. Per-bulletin text maps are
    loaded once and memoized; images are rendered on demand at `RENDER_DPI` and cached to
    `renders/`. Built from the artifact `root` and the corpus `pdf_dir` (the latter only for
    on-demand rendering)."""

    def __init__(self, root: Path, pdf_dir: Path | str) -> None:
        self._pages_dir = Path(root) / PAGES_SUBDIR
        self._scans_dir = Path(root) / SCANS_SUBDIR
        self._renders_dir = Path(root) / RENDERS_SUBDIR
        self._pdf_dir = pdf_dir
        self._lock = threading.Lock()
        self._text: dict[str, dict[int, str]] = {}  # doc_id -> {page: text}
        self._scans: dict[str, dict[int, PageScan]] = {}  # doc_id -> {page: PageScan}
        self._page_locks: dict[str, threading.Lock] = {}  # render key -> lock

    # -- text -----------------------------------------------------------------

    def text(self, ref: PageRef) -> str | None:
        """One PHYSICAL page's stored text, or None when absent. (Under the block-level table
        merge the page store keys text per physical page — a continued table's tail lives on its
        own `extra_pages` entry, not folded into the anchor — so a multi-page block's text is the
        concatenation of its `block_refs` pages, which extract feeds whole.)"""
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
            raise StepFailed(
                "extract",
                f"page store not built ({self._pages_dir} missing); "
                "run the page-index build pipeline first.",
            )
        path = self._pages_dir / f"{bulletin}.json"
        raw = json.loads(path.read_text()) if path.exists() else {}
        entries = {int(p): t for p, t in raw.items()}
        with self._lock:
            return self._text.setdefault(bulletin, entries)

    # -- scans ----------------------------------------------------------------

    def summary(self, ref: PageRef) -> PageScan | None:
        """The page's raw `PageScan`, or None when absent — the blocks/continuation/notes
        metadata extract reads to orient a page and expand it to its dependent pages."""
        if ref.month is None or ref.page is None:
            return None
        return self._doc_scans(ref.month).get(int(ref.page))

    def _doc_scans(self, doc_id: str) -> dict[int, PageScan]:
        """`{page: PageScan}` for one document, loaded once and memoized (thread-safe). Raises if
        the scans dir was never built; a single missing doc file is treated as empty."""
        with self._lock:
            cached = self._scans.get(doc_id)
        if cached is not None:
            return cached
        if not self._scans_dir.exists():
            raise StepFailed(
                "extract",
                f"scans not built ({self._scans_dir} missing); "
                "run the page-index build pipeline first.",
            )
        path = self._scans_dir / f"{doc_id}.json"
        raw = json.loads(path.read_text()).get("scans", {}) if path.exists() else {}
        entries = {int(p): PageScan.model_validate(s) for p, s in raw.items()}
        with self._lock:
            return self._scans.setdefault(doc_id, entries)

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
            img = read_cached_image(
                self._renders_dir, ref.month, ref.page
            )  # re-check under lock
            if img is not None:
                return img
            if (
                render_to_cache(ref.month, ref.page, self._pdf_dir, self._renders_dir)
                == "skipped"
            ):
                return None
            return read_cached_image(self._renders_dir, ref.month, ref.page)

    def _lock_for(self, key: str) -> threading.Lock:
        with self._lock:
            return self._page_locks.setdefault(key, threading.Lock())


def render_cache_path(renders_dir: str | Path, month: str, page: int) -> Path:
    """The page's image-cache path: `<renders_dir>/<month>/<page>.png`."""
    return Path(renders_dir) / month / f"{page}.png"


def read_cached_image(
    renders_dir: str | Path, month: str, page: int
) -> B64Image | None:
    """The page's cached PNG as a `B64Image`, or None if it hasn't been rendered yet."""
    return _read_png(render_cache_path(renders_dir, month, page))


def render_to_cache(
    month: str, page: int, pdf_dir: str | Path, renders_dir: str | Path
) -> str:
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
    return B64Image(
        mime="image/png", data=base64.standard_b64encode(path.read_bytes()).decode()
    )


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
