"""retrieve operator — dispatches to one of two backends based on
`ctx.config.retriever`:

- `"search_agent"` (default) — the teammate's iterative ChromaDB +
  LLM-loop retriever, vendored under `skunk.search_agent`. Returns
  page keys; this module adapts them to `PageRef`.
- `"page_index"` — the legacy chapter-pick + year-filter retriever
  preserved under `skunk.page_index.retrieve_prototype` for
  ablation/regression comparison.

`ctx.config.golden_pages` short-circuits both — for `--golden` eval
runs the backend is never built.

"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from skunk.config import SkunkConfig
from skunk.errors import StepFailed
from skunk.models import HarnessContext, PageRef, page_key_to_pageref
from skunk.plan import RetrieveBranch


class RetrieveExecutor:
    """Dispatcher. Holds lazily-built backends so the golden path never
    pays the cost of opening ChromaDB or loading the clean-page map."""

    def __init__(self, config: SkunkConfig) -> None:
        self._config = config
        self._agent = None                # skunk.search_agent.SearchAgent
        self._agent_lock = threading.Lock()
        self._page_index_proto = None     # PageIndexRetrievePrototype

    def run(self, ctx: HarnessContext, branch: RetrieveBranch) -> list[PageRef]:
        if ctx.config.golden_pages is not None:
            ctx.emit(
                "retrieve", "golden bypass",
                n_pages=len(ctx.config.golden_pages),
                refs=[str(r) for r in ctx.config.golden_pages],
            )
            return ctx.config.golden_pages

        if ctx.config.retriever == "page_index":
            return self._run_page_index(ctx, branch)
        if ctx.config.retriever == "search_agent":
            return self._run_search_agent(ctx, branch)
        raise StepFailed(
            "retrieve",
            f"unknown retriever {ctx.config.retriever!r}; "
            "expected 'search_agent' or 'page_index'",
        )

    # ------------------------------------------------------------------
    # search_agent backend
    # ------------------------------------------------------------------

    def _run_search_agent(
        self, ctx: HarnessContext, branch: RetrieveBranch
    ) -> list[PageRef]:
        agent = self._ensure_agent(ctx.config)
        ctx.emit(
            "retrieve", "search_agent start",
            key=branch.key, period=branch.period,
        )
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
        ctx.emit(
            "retrieve", "search_agent done",
            n_pages=len(refs), n_bad_keys=len(bad),
            bad_keys=bad[:5] if bad else None,
        )
        if not refs:
            raise StepFailed(
                "retrieve",
                f"search_agent returned no usable page keys (raw={page_keys!r})",
            )
        return refs

    def _ensure_agent(self, config: SkunkConfig):
        # Single-flight: first non-golden retrieve in a parallel branch
        # set must not race with a second one.
        with self._agent_lock:
            if self._agent is None:
                self._agent = _build_search_agent(config)
            return self._agent

    # ------------------------------------------------------------------
    # page_index backend (legacy prototype)
    # ------------------------------------------------------------------

    def _run_page_index(
        self, ctx: HarnessContext, branch: RetrieveBranch
    ) -> list[PageRef]:
        from skunk.page_index.retrieve_prototype import PageIndexRetrievePrototype

        if self._page_index_proto is None:
            self._page_index_proto = PageIndexRetrievePrototype()
        # PageIndexRetrievePrototype.run() takes (prev, ctx, *, branch);
        # call it directly with prev=None.
        return self._page_index_proto.run(None, ctx, branch=branch)


def _build_search_agent(config: SkunkConfig):
    """Open ChromaDB, load the clean-page map, and construct the agent.
    Raises `StepFailed` with a clear message if either artifact is
    missing — first non-golden run is where corpus-prep bugs surface."""
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
