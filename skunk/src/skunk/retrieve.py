"""retrieve operator — one op over two swappable backends.

`RetrieveOp` turns a `RetrieveBranch` into the pages that answer it:
- golden bypass — returns `ctx.config.golden_pages` verbatim (eval ablation).
- `search_agent` — iterative ChromaDB + LLM loop under `skunk.search_agent`.
- `page_index` — ToC pick → year filter → semantic filter (all within
  `PageIndexRetriever.retrieve_all`), then ONE `SelectAgent` does the precision selection
  over the flagged survivors and emits the final pages.

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
    AnnotatedValue,
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
    """Wrap bare page refs (golden / search-agent / select-agent) as whole-page `BlockRef`s (`block_index=None`),
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
        prior_values: list[AnnotatedValue] | None = None,
    ) -> list[BranchRetrieval | StepFailed]:
        """The single retrieval seam: retrieve for several branches at once, one result per
        branch aligned to `branches`. Each slot is a `BranchRetrieval` (its final blocks) or a
        `StepFailed` — a single branch failing does not sink its siblings. A whole-sweep
        failure (unknown retriever, missing index) raises. This is the ONLY place that
        dispatches on `config.retriever`:

        - golden bypass (`golden_pages`) → whole-page blocks.
        - `search_agent` → agent rollout → page keys → whole-page blocks.
        - `page_index` → ToC + year + sem filter sweep, then ONE `SelectAgent` selects the
          final pages over the union of every branch's survivors; an empty survivor list for
          a branch surfaces as a per-branch `StepFailed`.

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
            return [BranchRetrieval(blocks=blocks) for _ in branches]
        scopes: list[list[str] | None] = document_scopes or [None] * len(branches)
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
                    return BranchRetrieval(blocks=tuple(_whole_page_blocks(refs)))

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
                # ToC + year + sem filter sweep (shared across branches), then ONE SelectAgent
                # — given the full question and the UNION of every (non-pinned) branch's
                # flagged survivors — selects the pages that answer the whole question. Its
                # pages are broadcast to every non-pinned branch (each branch's extract then
                # pulls its own series from them). Pinned branches stay deterministic positional
                # fetches and skip the agent.
                from skunk.page_index.query import PageIndexRetriever
                from skunk.page_index.store import get_page_store

                retriever = self._page_index()
                survivors = await retriever.retrieve_all(
                    ctx, branches, document_scopes=scopes
                )
                catalog = retriever.catalog
                page_store = get_page_store(str(ctx.config.pdf_dir))
                search_index_path = str(retriever.search_index_path)

                slots: list[BranchRetrieval | StepFailed | None] = [None] * len(branches)
                union: list[BlockRef] = []
                seen_union: set[BlockRef] = set()
                agent_positions: list[int] = []
                for i, (b, brs) in enumerate(zip(branches, survivors)):
                    if b.page_pin is not None:
                        slots[i] = (
                            BranchRetrieval(blocks=tuple(brs))
                            if brs
                            else StepFailed(
                                "retrieve",
                                f"page_pin {b.page_pin.bulletin}:{b.page_pin.page} resolved to no catalog page",
                            )
                        )
                    elif not brs:
                        slots[i] = StepFailed(
                            "retrieve",
                            f"semantic filter kept no blocks for branch {b.key!r}",
                        )
                    else:
                        agent_positions.append(i)
                        for blk in brs:
                            if blk not in seen_union:
                                seen_union.add(blk)
                                union.append(blk)

                if agent_positions:
                    pool = PageIndexRetriever.pool_for_blocks(
                        union, str(ctx.config.pdf_dir)
                    )
                    if not pool:
                        fail = StepFailed("retrieve", "semantic filter kept no blocks")
                        for i in agent_positions:
                            slots[i] = fail
                    else:
                        try:
                            refs, targets = await traced_step(
                                ctx, "select_agent",
                                lambda: self._run_select_agent(
                                    ctx, pool, catalog, page_store, search_index_path,
                                    prior_values=prior_values,
                                ),
                            )
                            shared = BranchRetrieval(
                                blocks=tuple(_whole_page_blocks(refs)),
                                page_targets=targets or None,
                            )
                            for i in agent_positions:
                                slots[i] = shared
                        except StepFailed as e:
                            for i in agent_positions:
                                slots[i] = e
                return [
                    r if r is not None else StepFailed("retrieve", "no retrieval")
                    for r in slots
                ]
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

    async def _run_select_agent(
        self,
        ctx: ExecutionContext,
        pool: list,
        catalog,
        page_store,
        search_index_path: str,
        *,
        prior_values: list[AnnotatedValue] | None = None,
    ) -> tuple[list[PageRef], dict[str, str]]:
        """ONE agent for the whole question over the union of all branches' candidates.
        `prior_values` (replan sweeps) is what earlier attempts already gathered — surfaced
        to the agent so it selects pages only for the data still missing. Returns the selected
        page refs and a (doc_id → retrieval target) map: the per-page natural-language target
        the agent wrote, which drives that page's extraction."""
        from skunk.select_agent import SelectAgent

        agent = SelectAgent(
            config=ctx.config,
            catalog=catalog,
            page_store=page_store,
            candidates=pool,
            search_index_path=search_index_path,
            prior_values=prior_values,
        )
        pages = await agent.retrieve(ctx, ctx.question)
        refs: list[PageRef] = []
        targets: dict[str, str] = {}
        bad: list[str] = []
        for key, target in pages:
            try:
                refs.append(page_key_to_pageref(key))
            except ValueError:
                bad.append(key)
                continue
            if target:
                targets[key] = target
        if bad:
            ctx.emit(f"bad_page_keys n_bad={len(bad)} keys={bad[:5]!r}")
        if not refs:
            raise StepFailed(
                "retrieve",
                f"select_agent returned no usable page keys (raw={pages!r})",
            )
        return refs, targets

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
