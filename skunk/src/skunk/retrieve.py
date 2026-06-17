"""retrieve operator — one op over two swappable backends.

`RetrieveOp` turns a `RetrieveBranch` into the pages that answer it:
- golden bypass — returns `ctx.config.golden_pages` verbatim (eval ablation).
- `search_agent` — iterative FTS5 + PageStore + LLM loop under `skunk.search_agent`.

The search-agent backend is built lazily.
"""

from __future__ import annotations

import asyncio
import threading

from skunk.config import SkunkConfig
from skunk.errors import StepFailed
from skunk.common import (
    BranchRetrieval,
    ExecutionContext,
    PageRef,
    traced_step,
)
from skunk.plan import RetrieveBranch


class RetrieveOp:
    def __init__(self, config: SkunkConfig) -> None:
        self._config = config
        self._resources = None  # (PageStore, search_index_path) — shared across branches
        self._resources_lock = threading.Lock()

    async def run_all(
        self,
        ctx: ExecutionContext,
        branches: list[RetrieveBranch],
        branch_ids: list[int] | None = None,
        *,
        document_scopes: list[list[str] | None] | None = None,
    ) -> list[BranchRetrieval | StepFailed]:
        """The single retrieval seam: retrieve for several branches at once, one result per
        branch aligned to `branches`. Each slot is a `BranchRetrieval` (its pages) or a
        `StepFailed` — a single branch failing does not sink its siblings. A whole-sweep
        failure (unknown retriever, missing index) raises. This is the ONLY place that
        dispatches on `config.retriever`:

        - golden bypass (`golden_pages`) → the benchmark pages verbatim.
        - `search_agent` → agent rollout → page keys → the retrieved pages.

        `branch_ids` aligns each branch to its stable id so its retrieve runs in a per-branch
        `traced_step`: the rollout + the returned pages then attach to that branch in the
        trace viewer. `document_scopes` hard-scopes a branch's corpus to a set of documents
        (HITL human-required documents)."""
        if ctx.config.golden_pages is not None:
            # --golden ablation: inject the benchmark pages verbatim, already final.
            pages = ctx.config.golden_pages
            ctx.emit(
                f"golden_bypass n_pages={len(pages)} refs={[str(r) for r in pages]!r}"
            )
            return [BranchRetrieval(pages=tuple(pages)) for _ in branches]
        scopes = document_scopes or [None] * len(branches)
        match str(ctx.config.retriever):
            case "search_agent":
                ids = branch_ids if branch_ids is not None else [None] * len(branches)

                async def _one(
                    b: RetrieveBranch, bid: int | None, scope: list[str] | None
                ) -> BranchRetrieval:
                    # Per-branch `retrieve` step (branch_id=bid) so the SearchAgent rollout
                    # and its `pages` summary group under this branch in the viewer. `scope`
                    # (human-required documents, if any) hard-scopes the agent's corpus.
                    refs = await traced_step(
                        ctx, "retrieve",
                        lambda: self._run_search_agent(ctx, b, required_docs=scope),
                        branch_id=bid,
                    )
                    return BranchRetrieval(pages=tuple(refs))

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
            case other:
                raise StepFailed(
                    "retrieve",
                    f"unknown retriever {other!r}; expected 'search_agent'",
                )

    async def _run_search_agent(
        self,
        ctx: ExecutionContext,
        branch: RetrieveBranch,
        *,
        required_docs: list[str] | None = None,
    ) -> list[PageRef]:
        from skunk.search_agent import SearchAgent
        from skunk.search_agent.local_tools import doc_key_to_ref

        page_store, search_index_path = self._ensure_resources(ctx.config)
        agent = SearchAgent(
            config=ctx.config,
            page_store=page_store,
            search_index_path=search_index_path,
            human_intervention_handler=(
                ctx.human_intervention_handler
                if ctx.human_intervention_enabled
                else None
            ),
            required_docs=required_docs,
        )
        page_keys = await agent.retrieve(
            ctx,
            ctx.question,
            branch_key=branch.key,
            branch_period=branch.period,
            required_docs=required_docs,
        )
        refs: list[PageRef] = []
        bad: list[str] = []
        for key in page_keys:
            try:
                ref = doc_key_to_ref(key)
                if required_docs and ref.month not in required_docs:
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
        # Single-flight: neither parallel branches (same op) nor parallel UID workers
        # (eval's thread pool, each with its own RetrieveOp) need to rebuild the PageStore /
        # re-resolve the search index. The store + index path are serialized + cached
        # process-wide by `_get_shared_resources`. The resources are read-only + shared; only
        # the per-branch SearchAgent built around them holds mutable state.
        with self._resources_lock:
            if self._resources is None:
                self._resources = _get_shared_resources(config)
            return self._resources


# Process-wide page-index resource cache. The chroma-free retriever reads page CONTENT
# through one shared, thread-safe `PageStore` and reaches the corpus via the prebuilt
# read-only SQLite FTS5 search index. The eval runs many UIDs through one ThreadPoolExecutor,
# each with its own RetrieveOp, so we cache the (PageStore, index path) process-wide and
# single-flight construction under one global lock: this builds the store once and keeps a
# single in-memory copy of its (read-only) page caches shared across worker threads.
_SHARED_RESOURCES_LOCK = threading.Lock()
_SHARED_RESOURCES: dict[tuple[str, str], tuple] = {}


def _get_shared_resources(config: SkunkConfig):
    """Process-wide single-flight wrapper over `_build_resources`, keyed by the page-index
    artifact root + the corpus pdf dir. Serializes the store/index build across all
    RetrieveOps (i.e. across all UID worker threads)."""
    from skunk.page_index.data_model import page_index_root

    key = (str(page_index_root()), str(config.pdf_dir))
    with _SHARED_RESOURCES_LOCK:
        if key not in _SHARED_RESOURCES:
            _SHARED_RESOURCES[key] = _build_resources(config)
        return _SHARED_RESOURCES[key]


def _build_resources(config: SkunkConfig):
    from skunk.page_index.data_model import SEARCH_INDEX_FILE, page_index_root
    from skunk.page_index.store import get_page_store

    search_index_path = page_index_root() / SEARCH_INDEX_FILE
    if not search_index_path.exists():
        raise StepFailed(
            "retrieve",
            f"search index {search_index_path!s} does not exist; build it first with "
            "`python3 -m skunk.page_index.search_index build <root>` (see "
            "src/skunk/page_index/search_index.py), or point SKUNK_PAGE_INDEX_DIR at a "
            "built page-index artifact.",
        )

    page_store = get_page_store(str(config.pdf_dir))
    return page_store, str(search_index_path)
