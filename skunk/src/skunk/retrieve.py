"""retrieve operator — one op over three swappable backends.

`RetrieveOp` turns a `RetrieveBranch` into the pages that answer it:
- golden bypass — returns `ctx.config.golden_pages` verbatim (eval ablation).
- `search_agent` — iterative ChromaDB + LLM loop under `skunk.search_agent`.
- `page_index` — ToC pick → year filter → semantic filter (all within
  `PageIndexRetriever.retrieve_all`); the selection agent narrows downstream.

The two real backends are built lazily.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from skunk.config import SkunkConfig
from skunk.errors import StepFailed
from skunk.common import (
    BlockRef,
    BranchRetrieval,
    ExecutionContext,
    PageRef,
    page_key_to_pageref,
    traced_step,
)
from skunk.plan import RetrieveBranch

if TYPE_CHECKING:
    from skunk.page_index.query import BlockRef


def _whole_page_blocks(refs: list[PageRef]) -> list[BlockRef]:
    """Wrap bare page refs (golden / search-agent) as whole-page `BlockRef`s (`block_index=None`),
    so extract reads them the same as page-index blocks — just with no specific block to scope to."""
    return [
        BlockRef(page=r, block_index=None, member_refs=(r,), block=None) for r in refs
    ]


class RetrieveOp:
    def __init__(self, config: SkunkConfig) -> None:
        self._config = config
        self._resources = None  # (Collection, dict[str, str]) — shared across branches
        self._resources_lock = threading.Lock()
        self._page_index_retriever = None  # skunk.page_index.query.PageIndexRetriever

    async def run_all(
        self,
        ctx: ExecutionContext,
        branches: list[RetrieveBranch],
        branch_ids: list[int] | None = None,
        *,
        document_scopes: list[list[str] | None] | None = None,
    ) -> list[BranchRetrieval | StepFailed]:
        """The single retrieval seam: retrieve for several branches at once, one result per
        branch aligned to `branches`. Each slot is a `BranchRetrieval` (its blocks +
        `pre_selected`) or a `StepFailed` — a single branch failing does not sink its
        siblings. A whole-sweep failure (unknown retriever, missing index) raises. This is
        the ONLY place that dispatches on `config.retriever`:

        - golden bypass (`golden_pages`) → whole-page blocks, `pre_selected=True`.
        - `search_agent` → agent rollout → page keys → whole-page blocks, `pre_selected=True`.
        - `page_index` → sem-filter survivor blocks, `pre_selected=False`; an empty survivor
          list for a branch surfaces as a per-branch `StepFailed`.

        `branch_ids` (search-agent backend only) aligns each branch to its stable id so its
        retrieve runs in a per-branch `traced_step`: the rollout + the returned pages then
        attach to that branch in the trace viewer. The page-index backend is a single shared
        sweep, so it owns no per-branch step here (the orchestrator traces the whole phase).
        `document_scopes` (search-agent backend) hard-scopes a branch's corpus to a set of
        bulletins (HITL human-required documents)."""
        if ctx.config.golden_pages is not None:
            # --golden ablation: inject the benchmark pages verbatim, already final.
            pages = ctx.config.golden_pages
            ctx.emit(
                f"golden_bypass n_pages={len(pages)} refs={[str(r) for r in pages]!r}"
            )
            blocks = tuple(_whole_page_blocks(pages))
            return [BranchRetrieval(blocks=blocks, pre_selected=True) for _ in branches]
        scopes = document_scopes or [None] * len(branches)
        match str(ctx.config.retriever):
            case "search_agent":
                ids = branch_ids if branch_ids is not None else [None] * len(branches)

                async def _one(
                    b: RetrieveBranch, bid: int | None, scope: list[str] | None
                ) -> BranchRetrieval:
                    # Per-branch `retrieve` step (branch_id=bid) so the SearchAgent rollout
                    # and its `pages` summary group under this branch in the viewer. `scope`
                    # (human-required bulletins, if any) hard-scopes the agent's corpus.
                    refs = await traced_step(
                        ctx, "retrieve",
                        lambda: self._run_search_agent(ctx, b, required_bulletins=scope),
                        branch_id=bid,
                    )
                    return BranchRetrieval(
                        blocks=tuple(_whole_page_blocks(refs)), pre_selected=True
                    )

                settled = await asyncio.gather(
                    *(
                        _one(b, bid, scope)
                        for b, bid, scope in zip(branches, ids, scopes)
                    ),
                    return_exceptions=True,
                )
                out: list[BranchRetrieval | StepFailed] = []
                for r in settled:
                    if isinstance(r, StepFailed):
                        out.append(r)
                    elif isinstance(r, BaseException):
                        raise r
                    else:
                        out.append(r)
                return out
            case "page_index":
                survivors = await self._page_index().retrieve_all(
                    ctx,
                    branches,
                    document_scopes=scopes,
                )
                results: list[BranchRetrieval | StepFailed] = []
                for b, brs in zip(branches, survivors):
                    if not brs:
                        reason = (
                            f"page_pin {b.page_pin.bulletin}:{b.page_pin.page} resolved to no catalog page"
                            if b.page_pin is not None
                            else f"semantic filter kept no blocks for branch {b.key!r}"
                        )
                        results.append(StepFailed("retrieve", reason))
                    else:
                        # A page-pinned branch is a deterministic positional FETCH: mark it
                        # pre_selected so block_select keeps both resolved pages whole instead
                        # of dropping them on a topic mismatch — they go straight to extract.
                        results.append(
                            BranchRetrieval(
                                blocks=tuple(brs), pre_selected=b.page_pin is not None
                            )
                        )
                return results
            case other:
                raise StepFailed(
                    "retrieve",
                    f"unknown retriever {other!r}; expected 'search_agent' or 'page_index'",
                )

    async def _run_search_agent(
        self,
        ctx: ExecutionContext,
        branch: RetrieveBranch,
        *,
        required_bulletins: list[str] | None = None,
    ) -> list[PageRef]:
        from skunk.search_agent import SearchAgent

        collection, document_map = self._ensure_resources(ctx.config)
        # Optional: give the agent a `page_index_search` tool backed by the shared, lazily
        # built PageIndexRetriever (loads the concept tree/catalog once, not per branch).
        page_index_retriever = (
            self._page_index() if ctx.config.search_agent_pageindex else None
        )
        agent = SearchAgent(
            config=ctx.config,
            document_map=document_map,
            chroma_collection=collection,
            human_intervention_handler=(
                ctx.human_intervention_handler
                if ctx.human_intervention_enabled
                else None
            ),
            required_bulletins=required_bulletins,
            page_index_retriever=page_index_retriever,
        )
        page_keys = await agent.retrieve(
            ctx,
            ctx.question,
            branch_key=branch.key,
            branch_period=branch.period,
            required_bulletins=required_bulletins,
        )
        refs: list[PageRef] = []
        bad: list[str] = []
        for key in page_keys:
            try:
                ref = page_key_to_pageref(key)
                if required_bulletins and ref.month not in required_bulletins:
                    bad.append(key)
                    continue
                refs.append(ref)
            except ValueError:
                bad.append(key)
        if bad:
            ctx.emit(f"bad_page_keys n_bad={len(bad)} keys={bad[:5]!r}")
        if not refs:
            raise StepFailed(
                "retrieve",
                f"search_agent returned no usable page keys (raw={page_keys!r})",
            )
        return refs

    def _ensure_resources(self, config: SkunkConfig):
        # Single-flight: neither parallel branches (same op) nor parallel UID
        # workers (eval's thread pool, each with its own RetrieveOp) must race to
        # open ChromaDB. Concurrent `PersistentClient` construction against the
        # same SQLite fails with "Could not connect to tenant default_tenant", so
        # construction is serialized + cached process-wide by
        # `_get_shared_resources`. The resources are read-only + shared; only the
        # per-branch SearchAgent built around them holds mutable state.
        with self._resources_lock:
            if self._resources is None:
                self._resources = _get_shared_resources(config)
            return self._resources

    def _page_index(self):
        from skunk.page_index.query import PageIndexRetriever

        if self._page_index_retriever is None:
            self._page_index_retriever = PageIndexRetriever()
        return self._page_index_retriever


# Process-wide ChromaDB/document-map cache. The vector DB is a large read-only
# SQLite (tens of GB); constructing a `chromadb.PersistentClient` against it is not
# safe to do concurrently — parallel first-opens race on the tenant-bootstrap SELECT
# and fail with "Could not connect to tenant default_tenant". The eval runs many UIDs
# through one ThreadPoolExecutor, each UID with its own RetrieveOp, so we cache the
# opened collection + document map process-wide and single-flight construction under
# one global lock. The resources are read-only, so sharing the collection across
# worker threads is safe (and avoids N in-memory copies of the document map).
_SHARED_RESOURCES_LOCK = threading.Lock()
_SHARED_RESOURCES: dict[tuple[str, str, str], tuple] = {}


def _get_shared_resources(config: SkunkConfig):
    """Process-wide single-flight wrapper over `_build_resources`, keyed by the
    ChromaDB dir + collection + clean-page-map path. Serializes ChromaDB opens
    across all RetrieveOps (i.e. across all UID worker threads)."""
    key = (
        str(Path(config.chromadb_dir).resolve()),
        config.chromadb_collection,
        str(Path(config.clean_page_map_path).resolve()),
    )
    with _SHARED_RESOURCES_LOCK:
        if key not in _SHARED_RESOURCES:
            _SHARED_RESOURCES[key] = _build_resources(config)
        return _SHARED_RESOURCES[key]


def _build_resources(config: SkunkConfig):
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
    except Exception as e:
        raise StepFailed(
            "retrieve",
            f"chromadb collection {config.chromadb_collection!r} not found "
            f"under {chromadb_dir!s}: {e}",
        ) from e

    return collection, document_map
