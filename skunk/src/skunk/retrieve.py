"""retrieve operator — page-index retrieval.

Two-step deterministic-after-LLM pipeline:

  1. L1 chapter pick (one LLM call) — chooses a canonical chapter from
     the concept tree given the branch's question/key/period.
  2. Year-window filter (deterministic) — drops chapter pages whose
     `(min_year, max_year)` envelope doesn't intersect the period's year
     window. No-op when the period has no parseable year window.

The survivors are returned as the candidate set; no reranking, no caps.
See `skunk.page_index.retrieve_probe` for the L1 helper and
`skunk.page_index.period.period_year_window` for the period → year-
envelope reduction.
"""

from __future__ import annotations

import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from skunk.common import HarnessContext
from skunk.plan import PageRef
from skunk.page_index.period import period_year_window
from skunk.page_index.retrieve_probe import (
    load_catalog,
    load_concept_tree,
    one_shot_parent_chapter_retrieve,
)
from skunk.page_index.schema import PageCatalogRow
from skunk.operator import OpNode, StepFailed

_INDEX_LOCK = threading.Lock()
_INDEX_CACHE: dict[Path, tuple[dict[str, Any],
                               dict[tuple[str, int], PageCatalogRow]]] = {}

def _catalog_dir() -> Path:
    """Resolve the page-index catalog directory.

    Override with `SKUNK_PAGE_INDEX_DIR`. Default points at the
    in-repo cache built by the page-index pipeline.
    """
    env = os.environ.get("SKUNK_PAGE_INDEX_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "cache" / "page_index"

def _tree_path(catalog_dir: Path) -> Path:
    """Locate `concept_tree.json` — sibling of the catalog dir, with
    a fallback to the catalog dir itself for older layouts."""
    sibling = catalog_dir.parent / "concept_tree.json"
    if sibling.exists():
        return sibling
    return catalog_dir / "concept_tree.json"

def _load_index(
    catalog_dir: Path,
) -> tuple[dict[str, Any], dict[tuple[str, int], PageCatalogRow]]:
    """Process-singleton load of (concept tree, catalog index)."""
    key = catalog_dir.resolve()
    with _INDEX_LOCK:
        cached = _INDEX_CACHE.get(key)
        if cached is not None:
            return cached
        tree = load_concept_tree(_tree_path(catalog_dir))
        catalog_rows = load_catalog(catalog_dir)
        catalog_index = {(r.bulletin, r.page): r for r in catalog_rows}
        _INDEX_CACHE[key] = (tree, catalog_index)
        return tree, catalog_index

def _year_filter(
    candidates: list[dict[str, Any]],
    catalog_index: dict[tuple[str, int], PageCatalogRow],
    period: str,
) -> list[dict[str, Any]]:
    """Drop candidates whose `(min_year, max_year)` envelope doesn't
    intersect `period_year_window(period)`. Pages with no envelope
    (build-time gap; shouldn't happen post-bulletin-fallback) are kept
    so callers can decide. No-op when the period is unparseable.
    """
    window = period_year_window(period)
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

def run(op: OpNode, prev: None, ctx: HarnessContext) -> list[PageRef]:
    if ctx.config.golden_pages is not None:
        ctx.emit("retrieve", "golden bypass",
                 n_pages=len(ctx.config.golden_pages),
                 refs=[str(r) for r in ctx.config.golden_pages])
        return ctx.config.golden_pages

    key = str(op.args.get("key", "?"))
    period = str(op.args.get("period", "?"))
    catalog_dir = _catalog_dir()

    try:
        tree, catalog_index = _load_index(catalog_dir)
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
    filtered = _year_filter(chapter_top, catalog_index, period)
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
