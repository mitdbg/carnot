"""retrieve subagent — vector-retrieval over the page-index catalog.

Loads (lazily, once per process) a dense vector index built by
`skunk.page_index.vector_index.build_vector_index`. Per call: embed the
question, apply the symbolic period mask, take cosine top-K, and rerank
once with the leaf-rank LLM prompt. Returns a `DocHandle` of the picked
pages.
"""

from __future__ import annotations

import os
import threading
from dataclasses import asdict
from pathlib import Path

from skunk.common import HarnessContext
from skunk.dsl import DocHandle, OpNode, PageRef
from skunk.page_index.retrieve_probe import load_catalog
from skunk.page_index.schema import PageCatalogRow
from skunk.page_index.vector_index import VectorIndex, load_vector_index
from skunk.page_index.vector_retrieve import retrieve_vector
from skunk.subagents.base import StepFailed


_INDEX_LOCK = threading.Lock()
_INDEX_CACHE: dict[Path, tuple[VectorIndex, dict[tuple[str, int], PageCatalogRow]]] = {}


def _index_dir() -> Path:
    """Resolve the vector-index directory. Override with SKUNK_PAGE_INDEX_DIR."""
    env = os.environ.get("SKUNK_PAGE_INDEX_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "cache" / "page_index"


def _load_index(
    out_dir: Path,
) -> tuple[VectorIndex, dict[tuple[str, int], PageCatalogRow]]:
    """Process-singleton load of (vector index, catalog index) keyed by path."""
    key = out_dir.resolve()
    with _INDEX_LOCK:
        cached = _INDEX_CACHE.get(key)
        if cached is not None:
            return cached
        idx = load_vector_index(out_dir)
        catalog_rows = load_catalog(out_dir)
        catalog_index = {(r.bulletin, r.page): r for r in catalog_rows}
        _INDEX_CACHE[key] = (idx, catalog_index)
        return idx, catalog_index


def run(op: OpNode, prev: None, ctx: HarnessContext) -> DocHandle:
    if ctx.config.golden_handle is not None:
        ctx.emit("retrieve", "golden bypass",
                 n_pages=len(ctx.config.golden_handle.refs),
                 refs=[str(r) for r in ctx.config.golden_handle.refs])
        return ctx.config.golden_handle

    concept = str(op.args.get("concept") or op.args.get("source", "?"))
    period = str(op.args.get("period", "?"))
    out_dir = _index_dir()

    try:
        index, catalog_index = _load_index(out_dir)
    except FileNotFoundError as e:
        raise StepFailed(
            "retrieve",
            f"vector index not built ({e}); run "
            "`python -m skunk.page_index.vector_index build` first.",
        ) from e

    top, trace = retrieve_vector(
        question=ctx.question,
        concept=concept,
        period=period,
        llm=ctx.llm_client,
        index=index,
        catalog_index=catalog_index,
        period_type=op.args.get("period_type") or None,
    )

    ctx.emit(
        "retrieve", "vector retrieve",
        concept=concept, period=period,
        catalog_size=trace.catalog_size,
        candidate_count=trace.candidate_count,
        prefilter_s=round(trace.prefilter_s, 3),
        total_walk_s=round(trace.total_walk_s, 3),
        top_k=trace.top_k,
        levels=[asdict(lvl) for lvl in trace.levels],
    )

    refs: list[PageRef] = []
    for r in top:
        bulletin = r["bulletin"]
        page = int(r["page"])
        row = catalog_index.get((bulletin, page))
        file_path = row.file_path if row is not None else None
        refs.append(PageRef(month=bulletin, page=page, file_path=file_path))

    if not refs:
        raise StepFailed(
            "retrieve",
            f"no pages matched concept={concept!r} period={period!r} "
            f"(candidates={trace.candidate_count})",
        )

    desc = f"vector retrieve concept={concept!r} period={period!r}"
    return DocHandle(refs=refs, desc=desc)
