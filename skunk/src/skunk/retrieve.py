"""retrieve operator — one op over three swappable backends.

`RetrieveOp` turns a `RetrieveBranch` into the pages that answer it:
- golden bypass — returns `ctx.config.golden_pages` verbatim (eval ablation).
- `search_agent` — iterative ChromaDB + LLM loop under `skunk.search_agent`.
- `page_index` — ToC pick → year filter → semantic filter → block selection
  (all within `PageIndexRetriever.retrieve_all`).

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
from skunk.common import BlockRef, ExecutionContext, PageRef, page_key_to_pageref
from skunk.plan import RetrieveBranch

if TYPE_CHECKING:
    from skunk.page_index.query import BlockRef


# TODO: this is just a hack to make blocks work with search agents. We should probably fix this at some point
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
        self._page_index_retriever = (
            None  # skunk.page_index_old.query.PageIndexRetriever
        )

    async def run(
        self, ctx: ExecutionContext, branch: RetrieveBranch
    ) -> list[BlockRef]:
        if ctx.config.golden_pages is not None:
            return self._golden_blocks(ctx)
        match str(ctx.config.retriever):
            case "search_agent":
                return _whole_page_blocks(await self._run_search_agent(ctx, branch))
            case "page_index":
                return (await self._page_index().retrieve_all(ctx, [branch]))[0]
            case other:
                raise StepFailed(
                    "retrieve",
                    f"unknown retriever {other!r}; expected 'search_agent' or 'page_index_old'",
                )

    async def run_all(
        self, ctx: ExecutionContext, branches: list[RetrieveBranch]
    ) -> list[list[BlockRef] | StepFailed]:
        """Retrieve for several branches at once, result aligned to `branches`. A slot is
        that branch's blocks or a `StepFailed` — a single branch failing does not
        sink its siblings. A whole-sweep failure (unknown retriever, missing index) raises."""
        if ctx.config.golden_pages is not None:
            return [self._golden_blocks(ctx) for _ in branches]
        match str(ctx.config.retriever):
            case "search_agent":
                settled = await asyncio.gather(
                    *(self._run_search_agent(ctx, b) for b in branches),
                    return_exceptions=True,
                )
                out: list[list[BlockRef] | StepFailed] = []
                for r in settled:
                    if isinstance(r, StepFailed):
                        out.append(r)
                    elif isinstance(r, BaseException):
                        raise r
                    else:
                        out.append(_whole_page_blocks(r))
                return out
            case "page_index":
                return list(await self._page_index().retrieve_all(ctx, branches))
            case other:
                raise StepFailed(
                    "retrieve",
                    f"unknown retriever {other!r}; expected 'search_agent' or 'page_index'",
                )

    def _golden_blocks(self, ctx: ExecutionContext) -> list[BlockRef]:
        """Golden / replay bypass → `BlockRef`s. A block-aware cache replay injects the cached
        blocks (`config.cached_blocks`) so extract block-scopes as the live run did; plain
        `--golden` (no blocks) wraps its pages as whole-page blocks."""
        pages = ctx.config.golden_pages
        assert pages is not None
        ctx.emit(f"golden_bypass n_pages={len(pages)} refs={[str(r) for r in pages]!r}")
        if ctx.config.cached_blocks is not None:
            return ctx.config.cached_blocks
        return _whole_page_blocks(pages)

    async def _run_search_agent(
        self, ctx: ExecutionContext, branch: RetrieveBranch
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
            ctx.emit(f"bad_page_keys n_bad={len(bad)} keys={bad[:5]!r}")
        if not refs:
            raise StepFailed(
                "retrieve",
                f"search_agent returned no usable page keys (raw={page_keys!r})",
            )
        return refs

    def _ensure_resources(self, config: SkunkConfig):
        # Single-flight: parallel branches must not race to open ChromaDB.
        with self._resources_lock:
            if self._resources is None:
                self._resources = _build_resources(config)
            return self._resources

    def _page_index(self):
        from skunk.page_index.query import PageIndexRetriever

        if self._page_index_retriever is None:
            self._page_index_retriever = PageIndexRetriever()
        return self._page_index_retriever

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
