"""retrieve operator — one op over three swappable backends.

`RetrieveOp` turns a `RetrieveBranch` into the pages that answer it, picking a
backend per call:

- golden bypass — returns `ctx.config.golden_pages` verbatim (eval ablation;
  no backend built), taken first when `golden_pages` is set.
- `search_agent` — iterative ChromaDB + LLM loop under `skunk.search_agent`;
  returns page keys adapted to `PageRef`.
- `page_index` — ToC pick → year filter → semantic filter (wraps
  `skunk.page_index.query.PageIndexRetriever`).

The two real backends are built lazily so an unused one (notably ChromaDB) is
never opened. Backends raise `StepFailed("retrieve", …)` on failure and do not
emit their own start/done boundaries (the orchestrator's trace owns those).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from skunk.config import SkunkConfig
from skunk.errors import StepFailed
from skunk.common import ExecutionContext, PageRef, page_key_to_pageref
from skunk.plan import RetrieveBranch


class RetrieveOp:
    """The retrieve operator. Picks a backend per call: golden bypass first
    (when `ctx.config.golden_pages` is set), otherwise on `config.retriever`.
    `run` is also the single seam where a future per-branch backend override
    would plug in. The search-agent and page-index backends are built lazily so
    an unused one is never opened."""

    def __init__(self, config: SkunkConfig) -> None:
        self._config = config
        self._agent = None  # skunk.search_agent.SearchAgent
        self._agent_lock = threading.Lock()
        self._page_index_retriever = None  # skunk.page_index.query.PageIndexRetriever

    async def run(self, ctx: ExecutionContext, branch: RetrieveBranch) -> list[PageRef]:
        if ctx.config.golden_pages is not None:
            return self._run_golden(ctx, branch)
        match ctx.config.retriever:
            case "search_agent":
                return await self._run_search_agent(ctx, branch)
            case "page_index":
                return await self._run_page_index(ctx, branch)
            case other:
                raise StepFailed(
                    "retrieve",
                    f"unknown retriever {other!r}; expected 'search_agent' or 'page_index'",
                )

    def _run_golden(self, ctx: ExecutionContext, branch: RetrieveBranch) -> list[PageRef]:
        """Golden-pages bypass (eval ablation only). `run` only routes here when
        `golden_pages` is set, but we guard defensively anyway (also narrows the type)."""
        if ctx.config.golden_pages is None:
            raise StepFailed("retrieve", "golden bypass reached without golden_pages set")
        ctx.emit(
            f"golden_bypass n_pages={len(ctx.config.golden_pages)} "
            f"refs={[str(r) for r in ctx.config.golden_pages]!r}"
        )
        return ctx.config.golden_pages

    async def _run_search_agent(self, ctx: ExecutionContext, branch: RetrieveBranch) -> list[PageRef]:
        """Iterative search agent (`skunk.search_agent`); maps its page keys to `PageRef`."""
        agent = self._ensure_agent(ctx.config)
        page_keys = await agent.retrieve(
            ctx,
            ctx.question,
            branch_key=branch.key,
            branch_period=branch.period,
            branch_as_of=branch.as_of,
        )
        refs: list[PageRef] = []
        bad: list[str] = []
        for key in page_keys:
            try:
                refs.append(page_key_to_pageref(key))
            except ValueError:
                bad.append(key)
        if bad:
            # Callee-only diagnostic the trace can't show: keys the agent
            # returned that didn't map to a PageRef.
            ctx.emit(f"bad_page_keys n_bad={len(bad)} keys={bad[:5]!r}")
        if not refs:
            raise StepFailed(
                "retrieve",
                f"search_agent returned no usable page keys (raw={page_keys!r})",
            )
        return refs

    def _ensure_agent(self, config: SkunkConfig):
        # Single-flight: parallel branches must not race to open ChromaDB and
        # load the clean-page map.
        with self._agent_lock:
            if self._agent is None:
                self._agent = _build_search_agent(config)
            return self._agent

    async def _run_page_index(self, ctx: ExecutionContext, branch: RetrieveBranch) -> list[PageRef]:
        """Page-index retriever (ToC pick → year filter → semantic filter). The
        inner retriever is built lazily and caches the catalog/concept-tree."""
        from skunk.page_index.query import PageIndexRetriever

        if self._page_index_retriever is None:
            self._page_index_retriever = PageIndexRetriever()
        return await self._page_index_retriever.run(None, ctx, branch=branch)


def _build_search_agent(config: SkunkConfig):
    """Open ChromaDB, load the clean-page map, and construct the agent. Raises
    `StepFailed` with a clear message if either artifact is missing."""
    import chromadb

    from skunk.search_agent import SearchAgent

    chromadb_dir = Path(config.chromadb_dir)
    clean_page_map_path = Path(config.clean_page_map_path)

    if not chromadb_dir.exists():
        raise StepFailed(
            "retrieve",
            f"chromadb_dir {chromadb_dir!s} does not exist; build the "
            "vector DB first (see src/skunk/search_agent/prep/) or set "
            "SKUNK_CHROMADB_DIR / config.chromadb_dir.",
        )
    if not clean_page_map_path.exists():
        raise StepFailed(
            "retrieve",
            f"clean_page_map_path {clean_page_map_path!s} does not exist; "
            "run the page cleaner first (see "
            "src/skunk/search_agent/prep/page_cleaner.py) or set "
            "SKUNK_CLEAN_PAGE_MAP / config.clean_page_map_path.",
        )

    with clean_page_map_path.open() as f:
        clean_page_map = json.load(f)

    chroma_client = chromadb.PersistentClient(path=str(chromadb_dir))
    try:
        collection = chroma_client.get_collection(name=config.chromadb_collection)
    except Exception as e:  # chromadb raises a custom NotFound-style error
        raise StepFailed(
            "retrieve",
            f"chromadb collection {config.chromadb_collection!r} not found "
            f"under {chromadb_dir!s}: {e}",
        ) from e

    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        raise StepFailed(
            "retrieve",
            "GOOGLE_CLOUD_PROJECT not set — required for Vertex AI. "
            "Set it in your .env and run `gcloud auth application-default login`.",
        )
    return SearchAgent(
        config=config,
        clean_page_map=clean_page_map,
        chroma_collection=collection,
    )
