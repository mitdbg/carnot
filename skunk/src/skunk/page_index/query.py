"""Page-index retriever — the query path over the offline-built index.

One of the two retrieval methods (`config.retriever == "page_index"`;
the other is the iterative `search_agent`). Three deterministic passes
after the catalog/concept-tree are loaded:

  1. ToC chapter pick (one LLM call) — picks up to two canonical
     chapter(s) from the concept tree given the branch's question / key /
     period; every page under them becomes a candidate. See
     `page_index.query_toc`.
  2. Year filter (deterministic) — drops candidate pages whose structured
     `dates` (verbatim strings parsed back into ISO intervals) don't
     intersect any of the period's intervals. Pages without dates are
     kept (recall safety net). No-op when the period is unparseable.
  3. Semantic filter (coarse → fine, parallel) — prunes the survivors to
     a tight candidate set. See `page_index.query_semfilter`. Gated on
     `config.semfilter_enabled` (off = ToC + year-filter only, for
     ablation).

The period parser is supplied by the active corpus profile (default:
treasury; override via `SKUNK_CORPUS_PROFILE`). Golden bypass is handled
one level up in `RetrieveDispatcher`, so it isn't repeated here.
"""

from __future__ import annotations

import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from skunk.errors import StepFailed
from skunk.common import HarnessContext, PageRef
from skunk.plan import RetrieveBranch
from skunk.page_index import default_profile
from skunk.page_index.query_semfilter import semantic_filter
from skunk.page_index.query_toc import (
    load_catalog,
    load_concept_tree,
    one_shot_parent_chapter_retrieve,
)
from skunk.page_index.schema import PageCatalogRow


class PageIndexRetriever:
    """Page-index retrieval method. One instance per orchestrator;
    catalog/concept-tree are lazy-loaded and cached per-instance behind a
    lock so a parallel branch fan-out doesn't load the index twice.

    Not a `PromptedCall` subclass — the only operator-owned LLM call (the
    ToC chapter pick) lives in `page_index.query_toc`, which has its own
    static prompt; the semantic filter's calls live in
    `page_index.query_semfilter`. This class is the operator-level
    orchestrator: catalog loading, year filter, and stage sequencing.
    """

    def __init__(self) -> None:
        self._profile = default_profile()
        self._period_parser = self._profile.period_parser
        self._index_lock = threading.Lock()
        self._index_cache: dict[
            Path,
            tuple[dict[str, Any], dict[tuple[str, int], PageCatalogRow]],
        ] = {}

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
        """Drop candidates whose `row.dates` don't intersect any of the
        period's ISO intervals. Verbatim dates on the page are parsed via
        the period parser's `verbatim_date_to_intervals`, preserving
        range semantics ("1932-1939" is one closed interval, not min/max
        years 1932/1939) and month/day granularity.

        Pages with no dates and pages whose dates all fail to parse are
        kept — recall safety net for ToC / continuation pages and pages
        that survived catalog build with an unrecognized date shape.
        No-op when the period is unparseable.
        """
        period_intervals = self._period_parser.intervals(period)
        if not period_intervals:
            return list(candidates)
        kept: list[dict[str, Any]] = []
        for c in candidates:
            row = catalog_index.get((c["bulletin"], c["page"]))
            if row is None or not row.dates:
                kept.append(c)
                continue
            if self._period_parser.dates_overlap_period(
                row.dates, period_intervals,
            ):
                kept.append(c)
        return kept

    def run(self, prev: None, ctx: HarnessContext, *, branch: RetrieveBranch) -> list[PageRef]:
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

        # 1. ToC chapter pick. (`concept=` is the helper's internal kwarg
        # name; we pass the NL `key` through unchanged.)
        chapter_top, trace = one_shot_parent_chapter_retrieve(
            tree, question=ctx.question, concept=key, period=period,
            llm=ctx.llm_client, catalog_index=catalog_index,
        )

        # 2. Year filter.
        filtered = self._year_filter(chapter_top, catalog_index, period)

        # 3. Semantic filter (coarse → fine), unless disabled for ablation.
        sem_meta: dict[str, Any] = {"enabled": False}
        if ctx.config.semfilter_enabled and filtered:
            survivors = [(c["bulletin"], int(c["page"])) for c in filtered]
            kept_keys, sem_meta = semantic_filter(
                survivors, catalog_index, key, period, ctx,
            )
            kept_set = set(kept_keys)
            filtered = [c for c in filtered
                        if (c["bulletin"], int(c["page"])) in kept_set]

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
            semfilter=sem_meta,
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
