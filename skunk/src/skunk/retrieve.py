"""retrieve operator — plain functions over two swappable backends.

`run_retrieve_all` turns `RetrieveBranch`es into the pages that answer them:
- golden bypass — returns `ctx.config.golden_pages` verbatim (eval ablation).
- `search_agent` — iterative ChromaDB + LLM loop under `skunk.search_agent`.

Stateless by design (everything flows through `ctx` and the process-wide
resources cache below); the old `RetrieveOp` class held no state and was
retired for these functions.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from skunk.config import PipelineConfig
from skunk.errors import StepFailed
from skunk.common import (
    BranchRetrieval,
    ExecutionContext,
    PageRef,
    RetrievedDoc,
    page_key_to_pageref,
    traced_step,
)
from skunk.plan import RetrieveBranch


async def run_retrieve_all(
    ctx: ExecutionContext,
    branches: list[RetrieveBranch],
    branch_ids: list[int] | None = None,
) -> list[BranchRetrieval | StepFailed]:
    """The single retrieval seam: retrieve for several branches at once, one result per
    branch aligned to `branches`. Each slot is a `BranchRetrieval` (its retrieved pages
    with text) or a `StepFailed` — a single branch failing does not sink its siblings. A
    whole-sweep failure (missing index) raises. The search agent is the sole retriever:

    - golden bypass (`golden_pages`) → the benchmark pages verbatim (text attached).
    - search agent → agent rollout → page keys → the retrieved pages (text attached).

    Each page's text comes from the same `document_map` the search agent reads, so compute
    can read it directly (there is no separate extract step).

    `branch_ids` aligns each branch to its stable id so its retrieve runs in a per-branch
    `traced_step`: the rollout + the returned pages then attach to that branch in the
    trace viewer."""
    if ctx.config.golden_pages is not None:
        # --golden ablation: inject the benchmark pages verbatim (text attached), already final.
        pages = ctx.config.golden_pages
        document_map = _load_document_map(ctx.config)
        docs = tuple(_docs_for_refs(pages, document_map))
        ctx.emit(
            f"golden_bypass n_pages={len(pages)} refs={[str(r) for r in pages]!r}"
        )
        return [BranchRetrieval(documents=docs) for _ in branches]

    ids = branch_ids if branch_ids is not None else [None] * len(branches)

    async def _one(b: RetrieveBranch, bid: int | None) -> BranchRetrieval:
        # Per-branch `retrieve` step (branch_id=bid) so the SearchAgent rollout
        # and its `pages` summary group under this branch in the viewer.
        return await traced_step(
            ctx, "retrieve",
            lambda: _run_search_agent(ctx, b),
            branch_id=bid,
        )

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


def _docs_for_refs(
    refs: list[PageRef], document_map: dict[str, str]
) -> list[RetrievedDoc]:
    """Attach each ref's cleaned page text (from `document_map`, keyed by page key
    `<stem>_<page>`) — empty string when the map has no entry for it."""
    docs: list[RetrievedDoc] = []
    for r in refs:
        key = f"{r.stem}_{r.page}"
        docs.append(RetrievedDoc(ref=r, text=document_map.get(key, "")))
    return docs


async def _run_search_agent(
    ctx: ExecutionContext,
    branch: RetrieveBranch,
) -> BranchRetrieval:
    from skunk.search_agent import SearchAgent, doc_ids_from_payload

    collection, document_map = _get_shared_resources(ctx.config)
    agent = SearchAgent(
        config=ctx.config,
        document_map=document_map,
        chroma_collection=collection,
    )
    # This operator owns the retrieval user message (the agent's `call()` is the
    # generic entry point; the branch framing below is pipeline vocabulary).
    parts = [f"Question: {ctx.question}"]
    if branch.key:
        parts.append(f"Search focus: {branch.key}")
    if branch.period:
        parts.append(f"Time period (of the data): {branch.period}")
    payload = await agent.call(ctx, "\n".join(parts))
    page_keys = doc_ids_from_payload(payload)
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
    return BranchRetrieval(documents=tuple(_docs_for_refs(refs, document_map)))


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


def _get_shared_resources(config: PipelineConfig):
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


# Process-wide cache of the (read-only) document map, keyed by clean-page-map path, so the
# golden bypass (which needs the page text but not ChromaDB) doesn't re-read it per UID.
_DOC_MAP_LOCK = threading.Lock()
_DOC_MAP_CACHE: dict[str, dict[str, str]] = {}


def _load_document_map(config: PipelineConfig) -> dict[str, str]:
    """Load `doc_id (`<stem>_<page>` page key) → cleaned page text` from
    `config.clean_page_map_path`. Cached process-wide. Shared by the search-agent path
    (via `_build_resources`) and the golden bypass; needs no ChromaDB."""
    path_key = str(Path(config.clean_page_map_path).resolve())
    with _DOC_MAP_LOCK:
        cached = _DOC_MAP_CACHE.get(path_key)
    if cached is not None:
        return cached

    clean_page_map_path = Path(config.clean_page_map_path)
    if not clean_page_map_path.exists():
        raise StepFailed(
            "retrieve",
            f"clean_page_map_path {clean_page_map_path!s} does not exist; "
            "build the corpus page map offline first, or point "
            "config.clean_page_map_path (env SKUNK_CLEAN_PAGE_MAP) at it.",
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

    with _DOC_MAP_LOCK:
        _DOC_MAP_CACHE[path_key] = document_map
    return document_map


def _build_resources(config: PipelineConfig):
    from skunk.chroma_client import make_chroma_client

    document_map = _load_document_map(config)

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
