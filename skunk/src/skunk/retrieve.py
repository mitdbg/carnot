"""retrieve operator — strategy pattern over swappable backends.

Three interchangeable executors implement the `RetrieveStrategy` interface
(`run(ctx, branch) -> list[PageRef]`):

- `GoldenRetrieveExecutor` — returns `ctx.config.golden_pages` verbatim (eval
  ablation; no backend built).
- `SearchAgentRetrieveExecutor` — iterative ChromaDB + LLM loop under
  `skunk.search_agent`; returns page keys adapted to `PageRef`.
- `PageIndexRetrieveExecutor` — ToC pick → year filter → semantic filter
  (wraps `skunk.page_index.query.PageIndexRetriever`).

`RetrieveDispatcher` is what the orchestrator holds: it selects a strategy per
call — golden bypass first (when `ctx.config.golden_pages` is set), otherwise
on `ctx.config.retriever`. Strategies are built lazily so an unused backend
(notably ChromaDB) is never opened.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Protocol

from skunk.config import SkunkConfig
from skunk.errors import StepFailed
from skunk.common import HarnessContext, PageRef, page_key_to_pageref
from skunk.plan import RetrieveBranch


class RetrieveStrategy(Protocol):
    """One retrieve backend. Implementations turn a `RetrieveBranch` into the
    pages that answer it; they raise `StepFailed("retrieve", …)` on failure and
    must not emit their own start/done boundaries (the orchestrator's trace owns
    those)."""

    def run(self, ctx: HarnessContext, branch: RetrieveBranch) -> list[PageRef]: ...


class GoldenRetrieveExecutor:
    """Golden-pages bypass (eval ablation only). Returns the page refs pinned on
    the config and opens no backend. The dispatcher only routes here when
    `golden_pages` is set, but `run` guards defensively anyway."""

    def run(self, ctx: HarnessContext, branch: RetrieveBranch) -> list[PageRef]:
        if ctx.config.golden_pages is None:
            raise StepFailed(
                "retrieve", "GoldenRetrieveExecutor called without golden_pages set"
            )
        ctx.emit(
            "retrieve",
            "golden_bypass",
            n_pages=len(ctx.config.golden_pages),
            refs=[str(r) for r in ctx.config.golden_pages],
        )
        return ctx.config.golden_pages


class SearchAgentRetrieveExecutor:
    """Wraps the iterative search agent (`skunk.search_agent`). Builds the agent
    lazily behind a lock — single-flight so a parallel branch fan-out doesn't
    race to open ChromaDB and load the clean-page map."""

    def __init__(self) -> None:
        self._agent = None  # skunk.search_agent.SearchAgent
        self._agent_lock = threading.Lock()

    def run(self, ctx: HarnessContext, branch: RetrieveBranch) -> list[PageRef]:
        agent = self._ensure_agent(ctx.config)
        page_keys = agent.retrieve(
            ctx,
            ctx.question,
            branch_key=branch.key,
            branch_period=branch.period,
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
            ctx.emit("retrieve", "bad_page_keys", n_bad=len(bad), keys=bad[:5])
        if not refs:
            raise StepFailed(
                "retrieve",
                f"search_agent returned no usable page keys (raw={page_keys!r})",
            )
        return refs

    def _ensure_agent(self, config: SkunkConfig):
        # Single-flight: parallel branches must not race to build the agent.
        with self._agent_lock:
            if self._agent is None:
                self._agent = _build_search_agent(config)
            return self._agent


class PageIndexRetrieveExecutor:
    """Wraps `skunk.page_index.query.PageIndexRetriever` (ToC pick → year filter
    → semantic filter). The inner retriever is built lazily and caches the
    catalog/concept-tree per-instance."""

    def __init__(self) -> None:
        self._inner = None  # skunk.page_index.query.PageIndexRetriever

    def run(self, ctx: HarnessContext, branch: RetrieveBranch) -> list[PageRef]:
        from skunk.page_index.query import PageIndexRetriever

        if self._inner is None:
            self._inner = PageIndexRetriever()
        return self._inner.run(None, ctx, branch=branch)


class RetrieveDispatcher:
    """Selects a `RetrieveStrategy` per call and delegates. Golden bypass takes
    precedence over `config.retriever`; this is also the single seam where a
    future per-branch backend override would plug in. Strategies are
    instantiated lazily so an unused backend is never built."""

    def __init__(self, config: SkunkConfig) -> None:
        self._config = config
        self._golden_exec: GoldenRetrieveExecutor | None = None
        self._search_agent_exec: SearchAgentRetrieveExecutor | None = None
        self._page_index_exec: PageIndexRetrieveExecutor | None = None

    def run(self, ctx: HarnessContext, branch: RetrieveBranch) -> list[PageRef]:
        if ctx.config.golden_pages is not None:
            return self._golden().run(ctx, branch)
        match ctx.config.retriever:
            case "search_agent":
                return self._search_agent().run(ctx, branch)
            case "page_index":
                return self._page_index().run(ctx, branch)
            case other:
                raise StepFailed(
                    "retrieve",
                    f"unknown retriever {other!r}; expected 'search_agent' or 'page_index'",
                )

    def _golden(self) -> GoldenRetrieveExecutor:
        if self._golden_exec is None:
            self._golden_exec = GoldenRetrieveExecutor()
        return self._golden_exec

    def _search_agent(self) -> SearchAgentRetrieveExecutor:
        if self._search_agent_exec is None:
            self._search_agent_exec = SearchAgentRetrieveExecutor()
        return self._search_agent_exec

    def _page_index(self) -> PageIndexRetrieveExecutor:
        if self._page_index_exec is None:
            self._page_index_exec = PageIndexRetrieveExecutor()
        return self._page_index_exec


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
