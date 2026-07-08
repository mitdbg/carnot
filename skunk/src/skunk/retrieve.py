"""retrieve operator — one op over two swappable backends.

`RetrieveOp` turns a `RetrieveBranch` into the pages that answer it:
- golden bypass — returns `ctx.config.golden_pages` verbatim (eval ablation).
- `search_agent` — iterative ChromaDB + LLM loop under `skunk.search_agent`.

The search-agent backend is built lazily.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from skunk.config import SkunkConfig
from skunk.errors import StepFailed
from skunk.common import (
    BranchRetrieval,
    ExecutionContext,
    PageRef,
    page_key_to_pageref,
    traced_step,
)
from skunk.plan import RetrieveBranch


class RetrieveOp:
    def __init__(self, config: SkunkConfig) -> None:
        self._config = config
        self._resources = None  # (Collection, dict[str, str]) — shared across branches
        self._resources_lock = threading.Lock()

    async def run_all(
        self,
        ctx: ExecutionContext,
        branches: list[RetrieveBranch],
        branch_ids: list[int] | None = None,
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
        trace viewer."""
        if ctx.config.golden_pages is not None:
            # --golden ablation: inject the benchmark pages verbatim, already final.
            pages = ctx.config.golden_pages
            ctx.emit(
                f"golden_bypass n_pages={len(pages)} refs={[str(r) for r in pages]!r}"
            )
            return [BranchRetrieval(pages=tuple(pages)) for _ in branches]
        match str(ctx.config.retriever):
            case "search_agent":
                ids = branch_ids if branch_ids is not None else [None] * len(branches)

                async def _one(b: RetrieveBranch, bid: int | None) -> BranchRetrieval:
                    # Per-branch `retrieve` step (branch_id=bid) so the SearchAgent rollout
                    # and its `pages` summary group under this branch in the viewer.
                    refs = await traced_step(
                        ctx, "retrieve",
                        lambda: self._run_search_agent(ctx, b),
                        branch_id=bid,
                    )
                    return BranchRetrieval(pages=tuple(refs))

                settled = await asyncio.gather(
                    *(_one(b, bid) for b, bid in zip(branches, ids)),
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
    ) -> list[PageRef]:
        from skunk.search_agent import SearchAgent

        collection, document_map = self._ensure_resources(ctx.config)
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
        )
        refs: list[PageRef] = []
        bad: list[str] = []
        for key in page_keys:
            try:
                refs.append(page_key_to_pageref(key))
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
        # (eval's thread pool, each with its own RetrieveOp) need to re-connect to the
        # ChromaDB server / re-fetch the collection. Connection + collection handle are
        # serialized + cached process-wide by `_get_shared_resources`. The resources are
        # read-only + shared; only the per-branch SearchAgent built around them holds
        # mutable state.
        with self._resources_lock:
            if self._resources is None:
                self._resources = _get_shared_resources(config)
            return self._resources


# Process-wide ChromaDB/document-map cache. Reads go through a ChromaDB *server*
# (HttpClient) — the embedded PersistentClient deadlocks under the eval's 15-way
# in-process concurrency (worker threads wedge inside ChromaDB's Rust core), whereas
# the server process owns ChromaDB's concurrency. The eval runs many UIDs through one
# ThreadPoolExecutor, each with its own RetrieveOp, so we cache the (collection handle,
# document map) process-wide and single-flight construction under one global lock: this
# avoids re-connecting / re-fetching the collection per UID and keeps a single in-memory
# copy of the (read-only) document map shared across worker threads.
_SHARED_RESOURCES_LOCK = threading.Lock()
_SHARED_RESOURCES: dict[tuple[str, int, str, str], tuple] = {}


def _get_shared_resources(config: SkunkConfig):
    """Process-wide single-flight wrapper over `_build_resources`, keyed by the
    ChromaDB server (host, port) + collection + clean-page-map path. Serializes the
    connect/collection-fetch across all RetrieveOps (i.e. across all UID worker threads)."""
    key = (
        config.chroma_server_host,
        config.chroma_server_port,
        config.chromadb_collection,
        str(Path(config.clean_page_map_path).resolve()),
    )
    with _SHARED_RESOURCES_LOCK:
        if key not in _SHARED_RESOURCES:
            _SHARED_RESOURCES[key] = _build_resources(config)
        return _SHARED_RESOURCES[key]


def _build_resources(config: SkunkConfig):
    from skunk.chroma_client import make_chroma_client

    clean_page_map_path = Path(config.clean_page_map_path)

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

    chroma_client = make_chroma_client(
        config.chroma_server_host, config.chroma_server_port
    )
    try:
        collection = chroma_client.get_collection(name=config.chromadb_collection)
    except Exception as e:
        raise StepFailed(
            "retrieve",
            f"chromadb collection {config.chromadb_collection!r} not found on the "
            f"server at {config.chroma_server_host}:{config.chroma_server_port}: {e}",
        ) from e

    return collection, document_map
