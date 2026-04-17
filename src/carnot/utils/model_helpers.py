"""Helpers for context-window-aware input truncation.

Operators whose serialised input exceeds the model's context window use
these utilities to chunk and rank text by embedding similarity, then
reassemble a truncated version that fits within the token budget.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import litellm
import numpy as np

from carnot.agents.models import LiteLLMModel
from carnot.core.models import LLMCallStats
from carnot.optimizer.model_ids import get_api_key_for_model

logger = logging.getLogger(__name__)

_DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
_CHARS_PER_TOKEN = 4

# we determine the chunk size to be the minimum of (input_tokens / 32) and (token_budget / 10)
_DEFAULT_CHUNKS_WHEN_INPUT_IS_SLIGHTLY_ABOVE_LIMIT = 32
_DEFAULT_CHUNKS_WHEN_INPUT_IS_WELL_ABOVE_LIMIT = 10

# ── Token counting & model limits ───────────────────────────────────

def get_chunk_size(input_tokens: int, token_budget: int, is_join: bool = False) -> int:
    """Determine chunk size based on input size and token budget.
    For joins we double the target number of chunks to retain information from the left and right inputs.
    """
    chunk_size = min(
        input_tokens // _DEFAULT_CHUNKS_WHEN_INPUT_IS_SLIGHTLY_ABOVE_LIMIT,
        token_budget // _DEFAULT_CHUNKS_WHEN_INPUT_IS_WELL_ABOVE_LIMIT,
    )
    if is_join:
        chunk_size //= 2

    return max(chunk_size, 1)


def get_model_max_input_tokens(model_id: str, default: int = 128_000) -> int:
    """Return the maximum input-token limit for *model_id*.

    Requires:
        - *model_id* is a non-empty string.

    Returns:
        The ``max_input_tokens`` value from ``litellm.model_cost`` if
        found, otherwise *default*.

    Raises:
        None.
    """
    entry = litellm.model_cost.get(model_id)
    if entry is not None:
        return entry.get("max_input_tokens", default)
    return default


def count_tokens(text: str, model_id: str) -> int:
    """Count tokens in *text* for *model_id*.

    Requires:
        - *text* is a string.
        - *model_id* is a non-empty string.

    Returns:
        The token count. Falls back to ``len(text) // _CHARS_PER_TOKEN + 1`` on
        failure.

    Raises:
        None.
    """
    try:
        return litellm.token_counter(model=model_id, text=text)
    except Exception:
        return len(text) // _CHARS_PER_TOKEN + 1


# ── Chunking ────────────────────────────────────────────────────────

def chunk_item(item: dict, chunk_size: int, model_id: str) -> tuple[dict[str, list[str] | Any], list[str]]:
    """Chunk string values in *item* into lists of strings of at most *chunk_size* tokens.

    Each string value is replaced with a list of its chunks.  Non-string
    values are left unchanged.

    Requires:
        - *item* is a dict.
        - *chunk_size* >= 1.

    Returns:
        A new dict with the same keys as *item* where string values are
        replaced by lists of chunk strings and a list of keys that were chunked.

    Raises:
        None.
    """
    out, chunk_keys = {}, []
    for key, value in item.items():
        if isinstance(value, str):
            out[key] = chunk_text(value, chunk_size, model_id)
            chunk_keys.append(key)
        else:
            out[key] = value
    return out, chunk_keys


def chunk_text(text: str, chunk_size_tokens: int, model_id: str) -> list[str]:
    """Split *text* into non-overlapping chunks of at most *chunk_size_tokens*.

    Uses a character estimate (4 chars/token) then validates with
    ``count_tokens``, halving the character budget until all chunks fit.

    Requires:
        - *text* is a non-empty string.
        - *chunk_size_tokens* >= 1.

    Returns:
        A list of non-empty string chunks each within the token limit.

    Raises:
        None.
    """
    chunk_chars = max(chunk_size_tokens * _CHARS_PER_TOKEN, 1)
    while chunk_chars > 0:
        chunks = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]
        counts = [count_tokens(c, model_id) for c in chunks]
        if all(c <= chunk_size_tokens for c in counts):
            return chunks
        chunk_chars //= 2
    # Last resort — should never happen.
    return [text[i:i + 1] for i in range(len(text))]


# ── Embedding similarity ranking ────────────────────────────────────

def _get_embedding_model(llm_config: dict) -> LiteLLMModel:
    """Create a ``LiteLLMModel`` for the default embedding model."""
    api_key = get_api_key_for_model(_DEFAULT_EMBEDDING_MODEL, llm_config)
    return LiteLLMModel(model_id=_DEFAULT_EMBEDDING_MODEL, api_key=api_key)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def rank_chunks_by_similarity(
    chunks: list[str],
    query: str,
    llm_config: dict,
) -> tuple[list[tuple[int, float]], list[LLMCallStats]]:
    """Rank *chunks* by cosine similarity to *query* using embeddings.

    Requires:
        - *chunks* is a non-empty list of strings.
        - *query* is a non-empty string.

    Returns:
        A tuple ``(ranked, stats)`` where *ranked* is a list of
        ``(chunk_index, similarity_score)`` sorted descending by score,
        and *stats* is a list of ``LLMCallStats`` from embedding calls.

    Raises:
        Whatever ``LiteLLMModel.embed`` raises on unrecoverable failure.
    """
    embed_model = _get_embedding_model(llm_config)
    all_texts = chunks + [query]
    embeddings, stats = embed_model.embed(texts=all_texts)
    query_emb = embeddings[-1]
    ranked = [
        (i, _cosine_similarity(embeddings[i], query_emb))
        for i in range(len(chunks))
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked, [stats]


# ── High-level truncation entry points ──────────────────────────────

def truncate_item_to_fit(
    item: dict,
    task: str,
    model_id: str,
    llm_config: dict,
    overhead_tokens: int = 0,
) -> tuple[dict, list[LLMCallStats]]:
    """Return a (possibly truncated) copy of *item* that fits the model's context window.

    If the serialised item fits within the budget, it is returned unchanged.
    Otherwise the item's values are chunked, ranked by embedding similarity
    to *task*, and the most relevant chunks for each key are retained.

    Requires:
        - *item* is a dict.
        - *task* is a non-empty string describing the operator's objective.
        - *model_id* is a non-empty string.

    Returns:
        A tuple ``(item, embed_stats)`` where *item* may be a shallow copy
        with truncated string values, and *embed_stats* is a list of
        ``LLMCallStats`` from any embedding calls made (empty if no
        truncation was needed).

    Raises:
        None. Falls back gracefully on embedding errors.
    """
    # get the token budget for the input size and return if it fits without truncation
    token_budget = get_model_max_input_tokens(model_id) - overhead_tokens
    input_str = json.dumps(item, indent=2)
    input_tokens = count_tokens(input_str, model_id)
    if input_tokens <= token_budget:
        return item, []

    # otherwise, compute chunk size and chunk the item's string fields
    logger.info(
        "Item exceeds context window (%d tokens vs %d budget); truncating via embedding similarity.",
        input_tokens,
        token_budget,
    )
    chunk_size = get_chunk_size(input_tokens, token_budget)
    item_chunks, chunk_keys = chunk_item(item, chunk_size, model_id)

    # rank chunks by similarity to the task for string fields, leaving other fields unchanged
    key_to_ranked: dict[str, list[tuple[int, float]]] = {}
    embed_stats: list[LLMCallStats] = []
    for key in chunk_keys:
        chunks = item_chunks[key]
        try:
            ranked, chunk_embed_stats = rank_chunks_by_similarity(chunks, task, llm_config)
            embed_stats.extend(chunk_embed_stats)
            key_to_ranked[key] = ranked
        except Exception:
            logger.warning("Embedding similarity ranking failed; using natural order instead of sort.")
            key_to_ranked[key] = [(idx, 0.0) for idx in range(len(chunks))]

    # take top chunks in ranked order until we hit the token budget, then reassemble.
    item = _reassemble_top_chunks(item_chunks, chunk_keys, key_to_ranked, token_budget, model_id)

    return item, embed_stats


def truncate_join_inputs_to_fit(
    left_item: dict,
    right_item: dict,
    task: str,
    model_id: str,
    llm_config: dict,
    overhead_tokens: int = 0,
) -> tuple[dict, dict, list[LLMCallStats]]:
    """Return (possibly truncated) copies of *left_item* and *right_item* for a join.

    If the serialised items fit within the budget, they are returned unchanged.
    Otherwise the items' values are chunked, ranked by embedding similarity
    to *task*, and the most relevant chunks for each key are retained.

    Requires:
        - *left_item* and *right_item* are dicts.
        - *task* is the join condition string.

    Returns:
        A tuple ``(left, right, embed_stats)``.

    Raises:
        None.
    """
    # get the token budget for the input size and return if both inputs fit without truncation
    token_budget = get_model_max_input_tokens(model_id) - overhead_tokens
    left_str = json.dumps(left_item, indent=2)
    right_str = json.dumps(right_item, indent=2)
    left_tokens = count_tokens(left_str, model_id)
    right_tokens = count_tokens(right_str, model_id)
    combined_tokens = left_tokens + right_tokens
    if combined_tokens <= token_budget:
        return left_item, right_item, []

    # otherwise, compute chunk size and chunk the items' string fields
    logger.info(
        "Join inputs exceed context window (%d tokens vs %d budget); truncating.",
        combined_tokens,
        token_budget,
    )
    chunk_size = get_chunk_size(combined_tokens, token_budget, is_join=True)
    left_item_chunks, left_chunk_keys = chunk_item(left_item, chunk_size, model_id)
    right_item_chunks, right_chunk_keys = chunk_item(right_item, chunk_size, model_id)

    # rank chunks by similarity to the task for string fields, leaving other fields unchanged
    def rank_item_chunks(chunk_keys: list[str], item_chunks: dict[str, list[str] | Any], embed_stats: list[LLMCallStats]) -> dict[str, list[tuple[int, float]]]:
        key_to_ranked: dict[str, list[tuple[int, float]]] = {}
        for key in chunk_keys:
            chunks = item_chunks[key]
            try:
                ranked, chunk_embed_stats = rank_chunks_by_similarity(chunks, task, llm_config)
                embed_stats.extend(chunk_embed_stats)
                key_to_ranked[key] = ranked
            except Exception:
                logger.warning("Embedding similarity ranking failed; using natural order instead of sort.")
                key_to_ranked[key] = [(idx, 0.0) for idx in range(len(chunks))]
        return key_to_ranked

    embed_stats: list[LLMCallStats] = []
    left_key_to_ranked = rank_item_chunks(left_chunk_keys, left_item_chunks, embed_stats)
    right_key_to_ranked = rank_item_chunks(right_chunk_keys, right_item_chunks, embed_stats)

    # take top chunks in ranked order until we hit the token budget, then reassemble.
    left_item = _reassemble_top_chunks(left_item_chunks, left_chunk_keys, left_key_to_ranked, token_budget // 2, model_id)
    right_item = _reassemble_top_chunks(right_item_chunks, right_chunk_keys, right_key_to_ranked, token_budget // 2, model_id)

    return left_item, right_item, embed_stats


def truncate_agg_items_to_fit(
    items: list[dict],
    task: str,
    model_id: str,
    llm_config: dict,
    overhead_tokens: int = 0,
) -> tuple[list[dict], list[LLMCallStats]]:
    """Return (possibly truncated) copies of *items* for aggregation.

    For each item, we set a token budget proportional equal to (item_tokens / total_tokens) * (token_budget)
    and truncate via chunking and embedding similarity ranking as in the other functions.

    Requires:
        - *items* is a non-empty list of dicts.

    Returns:
        A tuple ``(items, embed_stats)``.

    Raises:
        None.
    """
    # get the total token budget for the input size and return if the items fit without truncation
    total_token_budget = get_model_max_input_tokens(model_id) - overhead_tokens
    input_str = json.dumps(items, indent=2)
    input_tokens = count_tokens(input_str, model_id)
    if input_tokens <= total_token_budget:
        return items, []

    # otherwise, chunk each item in accordance with its proportional share of the token budget,
    logger.info(
        "Agg inputs exceed context window (%d tokens vs %d budget); truncating.",
        input_tokens,
        total_token_budget,
    )
    all_embed_stats: list[LLMCallStats] = []
    truncated_items: list[dict] = []
    for item in items:
        item_str = json.dumps(item, indent=2)
        item_tokens = count_tokens(item_str, model_id)
        item_token_budget = int(item_tokens / input_tokens * total_token_budget)
        if item_tokens <= item_token_budget:
            truncated_items.append(item)
            continue

        # compute chunk size and chunk the item's string fields
        chunk_size = get_chunk_size(item_tokens, item_token_budget)
        item_chunks, chunk_keys = chunk_item(item, chunk_size, model_id)

        # rank chunks by similarity to the task for string fields, leaving other fields unchanged
        key_to_ranked: dict[str, list[tuple[int, float]]] = {}
        embed_stats: list[LLMCallStats] = []
        for key in chunk_keys:
            chunks = item_chunks[key]
            try:
                ranked, chunk_embed_stats = rank_chunks_by_similarity(chunks, task, llm_config)
                embed_stats.extend(chunk_embed_stats)
                key_to_ranked[key] = ranked
            except Exception:
                logger.warning("Embedding similarity ranking failed; using natural order instead of sort.")
                key_to_ranked[key] = [(idx, 0.0) for idx in range(len(chunks))]

        # take top chunks in ranked order until we hit the token budget, then reassemble.
        item = _reassemble_top_chunks(item_chunks, chunk_keys, key_to_ranked, item_token_budget, model_id)
        truncated_items.append(item)
        all_embed_stats.extend(embed_stats)

    return truncated_items, all_embed_stats


# ── Internal helpers ─────────────────────────────────────────────────

def _assemble_chunks_in_order(item: dict, chunked_keys: list[str]) -> dict:
    """Join selected chunks in their original order with '...' separators.
    
    Requires:
        - *item* is a dict where values for keys in *chunked_keys* are lists of (chunk, original_index) tuples.
        - *chunked_keys* is a list of keys in *item* whose values are lists of (chunk, original_index) tuples.

    Returns:
        A new dict with the same keys as *item* where chunked fields are reassembled in original order with
        '...' separators and non-chunked fields are unchanged.
    """
    for key in chunked_keys:
        chunks = item[key]
        # sort by original index
        sorted_chunks = sorted(chunks, key=lambda x: x[1])
        item[key] = "...".join(c for c, _ in sorted_chunks)
    return item


def _reassemble_top_chunks(
    item_chunks: dict[str, list[str] | Any],
    chunk_keys: list[str],
    key_to_ranked: dict[str, list[tuple[int, float]]],
    token_budget: int,
    model_id: str,
) -> dict:
    """Select as many top-ranked chunks as fit, reassemble, return truncated item."""
    item = {}

    # first, assign all fields which are not chunked and set a placeholder for chunked fields
    for key, value in item_chunks.items():
        if key in chunk_keys:
            item[key] = []
        else:
            item[key] = value

    # then, as long as the item is below the token budget, place the top-ranked chunk for each
    # chunked field in round-robin fashion until we run out of budget or chunks
    next_key_idx, more_chunks = 0, True
    while more_chunks and count_tokens(json.dumps(item, indent=2), model_id) <= token_budget:
        key = chunk_keys[next_key_idx]
        next_key_idx = (next_key_idx + 1) % len(chunk_keys)
        ranked = key_to_ranked.get(key)
        try:
            idx, _ = ranked.pop(0)
            item[key].append((item_chunks[key][idx], idx))
        except IndexError:
            more_chunks = any(ranked for ranked in key_to_ranked.values())

    # pop the last chunk which caused the overflow
    last_key = chunk_keys[(next_key_idx - 1) % len(chunk_keys)]
    item[last_key].pop()

    # finally, reassemble the retained chunks in original order for each field
    return _assemble_chunks_in_order(item, chunk_keys)
