"""Vector retriever — query → masked cosine top-K → single LLM rerank.

Drop-in alternative to the section→cluster→leaf walker in `retrieve_probe.py`.
Reuses the catalog's blob text and the leaf-rank prompt verbatim; replaces
three LLM calls with one embedding + one rerank.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from skunk.common import LLMClient

from .retrieve_probe import (
    LevelTrace,
    RetrieveTrace,
    leaf_rank_postings,
)
from .schema import PageCatalogRow
from .vector_index import VectorIndex, period_mask, search


_FREQUENCY_HINT_BY_PERIOD_TYPE = {
    "month": "Frequency: monthly table",
    "FY": "Frequency: annual rollup (fiscal year)",
    "CY": "Frequency: annual rollup (calendar year)",
    "fiscal_quarter": "Frequency: quarterly table",
    "calendar_quarter": "Frequency: quarterly table",
    "specific_date": "Frequency: dated snapshot (end-of-period table)",
}


def retrieve_vector(
    question: str,
    concept: str,
    period: str,
    llm: LLMClient,
    index: VectorIndex,
    catalog_index: dict[tuple[str, int], PageCatalogRow],
    *,
    period_type: str | None = None,
    uid: str | None = None,
    retrieve_idx: int = 0,
    top_k_ann: int = 50,
) -> tuple[list[dict[str, Any]], RetrieveTrace]:
    """End-to-end vector retrieve: embed → mask → cosine → rerank.

    Returns `(top, trace)` with the same `RetrieveTrace` schema the
    hierarchical walker emits, so eval/diff tooling stays compatible.
    `trace.levels` carries two entries: one synthetic `ann_search` level
    (no LLM, just timings + candidate count) and the real `leaf_rank`
    rerank trace.
    """
    trace = RetrieveTrace(
        uid=uid, retrieve_idx=retrieve_idx,
        concept=concept, period=period,
        catalog_size=index.vectors.shape[0],
    )

    t_walk = time.monotonic()

    # 1. Symbolic period prefilter.
    t0 = time.monotonic()
    mask = period_mask(index.keys, period, catalog_index=catalog_index)
    trace.prefilter_s = time.monotonic() - t0
    n_survivors = int(mask.sum())
    trace.candidate_count = n_survivors

    if n_survivors == 0:
        trace.total_walk_s = time.monotonic() - t_walk
        return [], trace

    # 2. Embed the question (RETRIEVAL_QUERY).
    t_embed = time.monotonic()
    frequency_hint = ""
    if period_type:
        hint = _FREQUENCY_HINT_BY_PERIOD_TYPE.get(period_type)
        if hint:
            frequency_hint = f"\n{hint}"
    embed_text = f"{question}\n\nConcept: {concept}\nPeriod: {period}{frequency_hint}"
    vecs = llm.embed([embed_text], task_type="RETRIEVAL_QUERY", dim=index.dim)
    embed_latency = time.monotonic() - t_embed
    if not vecs:
        trace.total_walk_s = time.monotonic() - t_walk
        return [], trace
    query_vec = np.asarray(vecs[0], dtype=np.float32)

    # 3. Masked cosine top-K.
    t_ann = time.monotonic()
    hits = search(index, query_vec, mask, top_k=top_k_ann)
    ann_latency = time.monotonic() - t_ann

    ann_trace = LevelTrace(
        level="ann_search",
        input_count=n_survivors,
        input_chars=len(embed_text),
        latency_s=embed_latency + ann_latency,
        output_count=len(hits),
        prompt_excerpt=embed_text[:800],
        response_excerpt="; ".join(
            f"{b}/p{p}:{score:.3f}" for score, (b, p, _) in hits[:10]
        ),
    )
    trace.levels.append(ann_trace)

    if not hits:
        trace.total_walk_s = time.monotonic() - t_walk
        return [], trace

    # 4. Single rerank call over the ANN candidates.
    candidate_keys: list[tuple[str, int]] = [(b, p) for _score, (b, p, _fp) in hits]
    top, rerank_trace = leaf_rank_postings(
        question, concept, period, candidate_keys, catalog_index, llm,
    )
    trace.levels.append(rerank_trace)
    trace.top_k = top
    trace.total_walk_s = time.monotonic() - t_walk
    return top, trace
