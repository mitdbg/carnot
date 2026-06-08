"""retrieve operator — one op over three swappable backends.

`RetrieveOp` turns a `RetrieveBranch` into the pages that answer it, picking a
backend per call:

- golden bypass — returns `ctx.config.golden_pages` verbatim (eval ablation;
  no backend built), taken first when `golden_pages` is set.
- `search_agent` — iterative ChromaDB + LLM loop under `skunk.search_agent`;
  returns page keys adapted to `PageRef`.
- `page_index_old` — ToC pick → year filter → semantic filter (wraps
  `skunk.page_index_old.query.PageIndexRetriever`).

The two real backends are built lazily so an unused one (notably ChromaDB) is
never opened. Backends raise `StepFailed("retrieve", …)` on failure and do not
emit their own start/done boundaries (the orchestrator's trace owns those).
"""

from __future__ import annotations

import asyncio
import json
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
        # Cached read-only resources (ChromaDB collection + doc_id→text map),
        # shared across branches and opened once. A fresh SearchAgent is built
        # per branch since it now carries per-question state (block trajectory +
        # prune sets), so it cannot be shared across concurrent branches.
        self._resources = None  # tuple[Collection, dict[str, str]]
        self._resources_lock = threading.Lock()
        self._page_index_retriever = (
            None  # skunk.page_index_old.query.PageIndexRetriever
        )

    async def run(self, ctx: ExecutionContext, branch: RetrieveBranch) -> list[PageRef]:
        if ctx.config.golden_pages is not None:
            return self._run_golden(ctx, branch)
        match ctx.config.retriever:
            case "search_agent":
                return await self._run_search_agent(ctx, branch)
            case "page_index_old":
                return (await self._page_index().retrieve_all(ctx, [branch]))[0]
            case other:
                raise StepFailed(
                    "retrieve",
                    f"unknown retriever {other!r}; expected 'search_agent' or 'page_index_old'",
                )

    async def run_all(
        self, ctx: ExecutionContext, branches: list[RetrieveBranch]
    ) -> list[list[PageRef] | StepFailed]:
        """Retrieve for several branches at once, result aligned to `branches`. A slot is
        that branch's page refs, or a `StepFailed` to attribute to that branch alone — so a
        single branch failing does not sink its siblings (the orchestrator routes each slot
        to its own extract / records the error). The page-index backend scans each unique
        candidate page with the LLM semantic filter at most once (judging it against all
        branch targets) and routes survivors back per branch; the other backends have no
        shared-scan benefit and run per branch via the same `run` seam. A whole-sweep /
        infra failure (unknown retriever, missing index) raises instead."""
        if ctx.config.golden_pages is not None:
            out: list[list[PageRef] | StepFailed] = [
                self._run_golden(ctx, b) for b in branches
            ]
            return out
        match ctx.config.retriever:
            case "search_agent":
                # Independent per branch (fresh agent + per-question state); isolate
                # per-branch failures so one branch's StepFailed doesn't fail the rest.
                settled = await asyncio.gather(
                    *(self._run_search_agent(ctx, b) for b in branches),
                    return_exceptions=True,
                )
                results: list[list[PageRef] | StepFailed] = []
                for r in settled:
                    if isinstance(r, StepFailed):
                        results.append(r)
                    elif isinstance(r, BaseException):
                        raise r
                    else:
                        results.append(r)
                return results
            case "page_index_old":
                return [
                    refs
                    for refs in await self._page_index().retrieve_all(ctx, branches)
                ]
            case other:
                raise StepFailed(
                    "retrieve",
                    f"unknown retriever {other!r}; expected 'search_agent' or 'page_index_old'",
                )

    def _run_golden(
        self, ctx: ExecutionContext, branch: RetrieveBranch
    ) -> list[PageRef]:
        """Golden-pages bypass (eval ablation only). `run` only routes here when
        `golden_pages` is set, but we guard defensively anyway (also narrows the type)."""
        if ctx.config.golden_pages is None:
            raise StepFailed(
                "retrieve", "golden bypass reached without golden_pages set"
            )
        ctx.emit(
            f"golden_bypass n_pages={len(ctx.config.golden_pages)} "
            f"refs={[str(r) for r in ctx.config.golden_pages]!r}"
        )
        return ctx.config.golden_pages

    async def _run_search_agent(
        self, ctx: ExecutionContext, branch: RetrieveBranch
    ) -> list[PageRef]:
        """Iterative search agent (`skunk.search_agent`); maps its page keys to `PageRef`."""
        from skunk.search_agent import SearchAgent

        collection, document_map = self._ensure_resources(ctx.config)
        # Fresh agent per branch: it holds per-question state (trajectory + prune
        # sets); the ChromaDB collection + document map underneath are shared.
        agent = SearchAgent(
            config=ctx.config,
            document_map=document_map,
            chroma_collection=collection,
        )
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

    def _ensure_resources(self, config: SkunkConfig):
        # Single-flight: parallel branches must not race to open ChromaDB and
        # load the document-text map. These resources are read-only + shared;
        # only the per-branch SearchAgent built around them holds mutable state.
        with self._resources_lock:
            if self._resources is None:
                self._resources = _build_resources(config)
            return self._resources

    def _page_index(self):
        """Lazily build + cache the page-index retriever (loads the catalog/concept-tree
        once). Used by both the single-branch `run` and the batched `run_all`."""
        from skunk.page_index.query import PageIndexRetriever

        if self._page_index_retriever is None:
            self._page_index_retriever = PageIndexRetriever()
        return self._page_index_retriever


def _build_resources(config: SkunkConfig):
    """Open ChromaDB and build the doc_id→text map from the clean-page map.
    Returns (collection, document_map). Raises `StepFailed` with a clear message
    if either artifact is missing."""
    import chromadb

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

    # clean_page_map: doc_id -> [clean_page_path, element_id_order]. Eagerly load
    # each cleaned page's text so `read_document` can serve it by doc_id. Built
    # once and shared across branches. Missing/unreadable pages are skipped here
    # and surface as "no such document" at read time (mirrors the old lazy read).
    document_map: dict[str, str] = {}
    for doc_id, entry in clean_page_map.items():
        path = entry[0] if isinstance(entry, (list, tuple)) else entry
        try:
            with open(path) as pf:
                document_map[doc_id] = pf.read()
        except OSError:
            continue

    chroma_client = chromadb.PersistentClient(path=str(chromadb_dir))
    try:
        collection = chroma_client.get_collection(name=config.chromadb_collection)
    except Exception as e:  # chromadb raises a custom NotFound-style error
        raise StepFailed(
            "retrieve",
            f"chromadb collection {config.chromadb_collection!r} not found "
            f"under {chromadb_dir!s}: {e}",
        ) from e

    return collection, document_map
