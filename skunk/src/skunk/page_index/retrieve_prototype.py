"""Prototype page-index retriever — preserved while we integrate an
external retrieve implementation. Not wired into `Orchestrator` by
default; use it directly for regression comparison or to keep BM25
/ year-filter experiments alive.

Two-step deterministic-after-LLM pipeline:

  1. L1 chapter pick (one LLM call) — chooses a canonical chapter from
     the concept tree given the branch's question/key/period.
  2. Year-window filter (deterministic) — drops chapter pages whose
     `(min_year, max_year)` envelope doesn't intersect the period's year
     window. No-op when the period has no parseable year window.

Optional BM25 rerank (default OFF; gated on `ctx.config.bm25_enabled`)
runs after the year filter. The period parser is supplied by the active
corpus profile (default: treasury; override via `SKUNK_CORPUS_PROFILE`).
"""

from __future__ import annotations

import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from skunk.errors import StepFailed
from skunk.models import HarnessContext, PageRef
from skunk.plan import RetrieveBranch
from skunk.page_index import default_profile
from skunk.page_index.bm25 import Bm25Index
from skunk.page_index.bm25_runtime import bm25_rerank, build_chapter_index
from skunk.page_index.retrieve_probe import (
    load_catalog,
    load_concept_tree,
    one_shot_parent_chapter_retrieve,
)
from skunk.page_index.schema import PageCatalogRow


class PageIndexRetrievePrototype:
    """Page-index retrieval prototype. One instance per orchestrator;
    catalog/concept-tree are lazy-loaded and cached per-instance behind a
    lock so a parallel branch fan-out doesn't load the index twice.

    Not a `PromptedCall` subclass — the only LLM call (the L1 chapter
    pick) is owned by `page_index.retrieve_probe`, which has its own
    static prompt. This class is the operator-level orchestrator: catalog
    loading, year-window filter, BM25 rerank, golden-bypass.
    """

    def __init__(self) -> None:
        self._profile = default_profile()
        self._period_parser = self._profile.period_parser
        self._index_lock = threading.Lock()
        self._index_cache: dict[
            Path,
            tuple[dict[str, Any], dict[tuple[str, int], PageCatalogRow]],
        ] = {}
        # BM25 scaffold (default OFF; gated on ctx.config.bm25_enabled).
        # Per-chapter index built lazily on first hit. Per-chapter lock so
        # different chapters can build in parallel.
        self._bm25_cache: dict[tuple[Path, str], Bm25Index] = {}
        self._bm25_locks_master = threading.Lock()
        self._bm25_locks: dict[tuple[Path, str], threading.Lock] = {}

    def _bm25_chapter_lock(self, key: tuple[Path, str]) -> threading.Lock:
        with self._bm25_locks_master:
            lock = self._bm25_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._bm25_locks[key] = lock
            return lock

    def _get_bm25_index(
        self,
        catalog_dir: Path,
        chapter: str,
        tree: dict[str, Any],
        catalog_index: dict[tuple[str, int], PageCatalogRow],
    ) -> Bm25Index:
        """Lazy-build a per-chapter BM25 index. Cached on this executor."""
        key = (catalog_dir.resolve(), chapter)
        cached = self._bm25_cache.get(key)
        if cached is not None:
            return cached
        with self._bm25_chapter_lock(key):
            cached = self._bm25_cache.get(key)
            if cached is not None:
                return cached
            pages = tree.get("chapters", {}).get(chapter, {}).get("pages", [])
            idx = build_chapter_index(pages, catalog_index)
            self._bm25_cache[key] = idx
            return idx

    def _catalog_dir(self) -> Path:
        """Resolve the page-index catalog directory.

        Override with `SKUNK_PAGE_INDEX_DIR`. Default points at the
        in-repo cache built by the page-index pipeline.
        """
        env = os.environ.get("SKUNK_PAGE_INDEX_DIR")
        if env:
            return Path(env)
        return Path(__file__).resolve().parents[3] / "cache" / "page_index"

    def _tree_path(self, catalog_dir: Path) -> Path:
        """Locate `concept_tree.json` — sibling of the catalog dir, with
        a fallback to the catalog dir itself for older layouts."""
        sibling = catalog_dir.parent / "concept_tree.json"
        if sibling.exists():
            return sibling
        return catalog_dir / "concept_tree.json"

    def _load_index(
        self, catalog_dir: Path,
    ) -> tuple[dict[str, Any], dict[tuple[str, int], PageCatalogRow]]:
        """Lazy-load + cache (concept tree, catalog index) by resolved dir."""
        key = catalog_dir.resolve()
        with self._index_lock:
            cached = self._index_cache.get(key)
            if cached is not None:
                return cached
            tree = load_concept_tree(self._tree_path(catalog_dir))
            catalog_rows = load_catalog(catalog_dir)
            catalog_index = {(r.bulletin, r.page): r for r in catalog_rows}
            self._index_cache[key] = (tree, catalog_index)
            return tree, catalog_index

    def _year_filter(
        self,
        candidates: list[dict[str, Any]],
        catalog_index: dict[tuple[str, int], PageCatalogRow],
        period: str | None,
    ) -> list[dict[str, Any]]:
        """Drop candidates whose `(min_year, max_year)` envelope doesn't
        intersect `period_year_window(period)`. Pages with no envelope
        (build-time gap; shouldn't happen post-bulletin-fallback) are kept
        so callers can decide. No-op when the period is unparseable.
        """
        window = self._period_parser.year_window(period)
        if window is None:
            return list(candidates)
        q_lo, q_hi = window
        kept: list[dict[str, Any]] = []
        for c in candidates:
            row = catalog_index.get((c["bulletin"], c["page"]))
            if row is None or row.min_year is None or row.max_year is None:
                kept.append(c)
                continue
            if row.max_year >= q_lo and row.min_year <= q_hi:
                kept.append(c)
        return kept

    def run(self, prev: None, ctx: HarnessContext, *, branch: RetrieveBranch) -> list[PageRef]:
        if ctx.config.golden_pages is not None:
            ctx.emit("retrieve", "golden bypass",
                     n_pages=len(ctx.config.golden_pages),
                     refs=[str(r) for r in ctx.config.golden_pages])
            return ctx.config.golden_pages

        key, period = branch.key, branch.period
        catalog_dir = self._catalog_dir()

        try:
            tree, catalog_index = self._load_index(catalog_dir)
        except FileNotFoundError as e:
            raise StepFailed(
                "retrieve",
                f"page index not built ({e}); run the page-index build "
                "pipeline first.",
            ) from e

        # 1. L1 chapter pick. (`concept=` is the helper's internal kwarg name;
        # we pass the NL `key` through unchanged.)
        chapter_top, trace = one_shot_parent_chapter_retrieve(
            tree, question=ctx.question, concept=key, period=period,
            llm=ctx.llm_client, catalog_index=catalog_index,
        )

        # 2. Year-window filter.
        filtered = self._year_filter(chapter_top, catalog_index, period)

        # 3. Optional BM25 rerank (scaffold, default OFF — see SkunkConfig).
        # Only runs when the flag is set and the LLM actually picked a
        # chapter (otherwise there's nothing to score against).
        bm25_meta: dict[str, Any] = {"enabled": False}
        if ctx.config.bm25_enabled and filtered and trace.picked_chapters:
            try:
                # Use the first picked chapter for the BM25 index. The
                # tree-page list of that chapter is a superset of `filtered`,
                # so the index covers every candidate page.
                index = self._get_bm25_index(
                    catalog_dir, trace.picked_chapters[0], tree, catalog_index,
                )
                filtered, bm25_meta = bm25_rerank(
                    filtered, index, question=ctx.question, key=key,
                    top_k=ctx.config.bm25_top_k,
                    threshold=ctx.config.bm25_dominance_threshold,
                )
            except Exception as e:
                # Scaffold safety: a BM25 failure must not break the
                # baseline path. Surface in the trace and pass through.
                bm25_meta = {"enabled": True, "error": f"{type(e).__name__}: {e}"}

        trace.candidate_count = len(filtered)
        trace.top_k = filtered[:50]

        ctx.emit(
            "retrieve", "page-index retrieve",
            key=key, period=period,
            catalog_size=trace.catalog_size,
            chapter_size=len(chapter_top),
            candidate_count=trace.candidate_count,
            top_k=trace.top_k,
            levels=[asdict(lvl) for lvl in trace.levels],
            picked_chapters=trace.picked_chapters,
            bm25=bm25_meta,
        )

        refs: list[PageRef] = [
            PageRef(month=r["bulletin"], page=int(r["page"])) for r in filtered
        ]
        if not refs:
            raise StepFailed(
                "retrieve",
                f"no pages matched key={key!r} period={period!r} "
                f"(chapter={len(chapter_top)})",
            )

        return refs
