"""retrieve operator — plain functions over a search-agent backend.

`run_retrieve_all` turns `RetrieveBranch`es into the pages that answer them via
`search_agent` — an iterative ChromaDB + LLM loop under `skunk.search_agent`.

Stateless by design: the retrieval substrate (chroma collection + document map,
plus the optional figure-tool paths) is built ONCE by the application and injected
onto `ctx`; these functions only read it. The old `RetrieveOp` class held no state
and was retired for these functions.
"""

from __future__ import annotations

import asyncio

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
from skunk.storage.document_map import DocumentMap

async def run_retrieve_all(
    ctx: ExecutionContext,
    branches: list[RetrieveBranch],
    branch_ids: list[int] | None = None,
) -> list[BranchRetrieval | StepFailed]:
    """The single retrieval seam: retrieve for several branches at once, one result per
    branch aligned to `branches`. Each slot is a `BranchRetrieval` (its retrieved pages
    with text) or a `StepFailed` — a single branch failing does not sink its siblings. A
    whole-sweep failure (missing index) raises. The search agent is the sole retriever:

    - search agent → agent rollout → page keys → the retrieved pages (text attached).

    Each page's text comes from the same `document_map` the search agent reads, so compute
    can read it directly (there is no separate extract step).

    `branch_ids` aligns each branch to its stable id so its retrieve runs in a per-branch
    `traced_step`: the rollout + the returned pages then attach to that branch in the
    trace viewer."""
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
    refs: list[PageRef], document_map: DocumentMap
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

    if ctx.chroma_collection is None:
        raise StepFailed(
            "retrieve",
            "no chroma_collection on ctx; the application must build the retrieval "
            "substrate (chroma collection + document_map) and inject it into the "
            "Orchestrator.",
        )
    agent = SearchAgent(
        config=ctx.config.search,
        document_map=ctx.document_map,
        chroma_collection=ctx.chroma_collection,
        pdf_dir=ctx.config.storage.pdf_dir,
        page_renders_dir=ctx.config.storage.page_renders_dir,
        llm_client=ctx.llm_client,
        emb_model_id=ctx.config.inference.emb_model_id,
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
    return BranchRetrieval(documents=tuple(_docs_for_refs(refs, ctx.document_map)))
