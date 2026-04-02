import argparse
import importlib.util
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import carnot
from carnot.agents.models import ChatMessage, LiteLLMModel
from carnot.index.models import HierarchicalIndexConfig
from carnot.index.summary_layer import SummaryLayer
from carnot.operators.sem_filter import SemFilterOperator
from carnot.operators.sem_topk import SemTopKOperator

load_dotenv()

_QUEST_EVAL_DIR = Path(__file__).resolve().parent
_BCP_EVAL_DIR = _QUEST_EVAL_DIR.parent / "browsercomp-plus"
_spec_qdt = importlib.util.spec_from_file_location(
    "browsercomp_qdt_retrieval_quest",
    _BCP_EVAL_DIR / "qdt_retrieval.py",
)
_qdt_mod = importlib.util.module_from_spec(_spec_qdt)
assert _spec_qdt.loader is not None
_spec_qdt.loader.exec_module(_qdt_mod)
generate_qdt_from_query = _qdt_mod.generate_qdt_from_query
dedupe_qdt_tasks = _qdt_mod.dedupe_qdt_tasks

DATASET_ID = 1

# Modes that load per-file summaries (cached under SummaryLayer).
SUMMARY_PREP_MODES = frozenset(
    {
        "summary-filter-only",
        "summary-filter-batched",
        "chroma-summary-filter",
        "summary-embedding-topk",
        "summary-embedding-topk-filter-batched",
    }
)
# Subset that loads summary embeddings for dense retrieval.
SUMMARY_MODES_WITH_EMBEDDINGS = frozenset(
    {"summary-embedding-topk", "summary-embedding-topk-filter-batched"}
)


def add_paths_to_items(items: list[dict]) -> list[dict]:
    """Add uri and path to items for index cache consistency with build_indices.py."""
    items_with_paths = []
    for i, item in enumerate(items):
        item_copy = dict(item)
        fake_path = f"/tmp/quest_docs/{i}_{item.get('title', 'doc')[:50]}.txt"
        item_copy["uri"] = fake_path
        item_copy["path"] = fake_path
        items_with_paths.append(item_copy)
    return items_with_paths


def aggregate_llm_stats(llm_calls: list) -> dict:
    """Aggregate cost and token usage from a list of LLM call stats."""
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for stat in llm_calls:
        total_cost += stat.cost_usd
        total_input_tokens += stat.total_input_tokens
        total_output_tokens += stat.total_output_tokens

    return {
        "total_cost_usd": total_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


def build_summary_items(
    items: list[dict],
    api_key: str,
    summary_model_id: str = "openai/gpt-5-mini-2025-08-07",
    embedding_model_id: str = "openai/text-embedding-3-small",
    include_embeddings: bool = False,
) -> tuple[list[dict], dict]:
    """Build summary-backed items, reusing cached summaries when available.

    When *include_embeddings* is True, each item includes an ``embedding`` list
    (same model as *embedding_model_id*, matching :class:`SummaryLayer` cache).
    """
    start_time = time.perf_counter()

    config = HierarchicalIndexConfig(
        summary_model=summary_model_id,
        embedding_model=embedding_model_id,
    )
    layer = SummaryLayer(config=config, api_key=api_key)
    summaries = layer.get_or_build_summaries(items)

    item_by_path = {item["path"]: item for item in items if item.get("path")}
    summary_items = []
    for entry in summaries:
        original_item = item_by_path.get(entry.path)
        if original_item is None:
            continue
        row = {
            "title": original_item.get("title", ""),
            "docid": original_item.get("docid", ""),
            "url": original_item.get("url", ""),
            "uri": original_item.get("uri", ""),
            "path": original_item.get("path", ""),
            "summary": entry.summary,
        }
        if include_embeddings:
            row["embedding"] = list(entry.embedding)
        summary_items.append(row)

    stats = aggregate_llm_stats(layer.llm_call_stats)
    stats["summary_build_wall_clock_secs"] = time.perf_counter() - start_time
    stats["summary_items_out"] = len(summary_items)
    return summary_items, stats


def _topk_summary_items_by_embedding(
    query: str,
    summary_items: list[dict],
    api_key: str,
    topk: int,
    embedding_model_id: str = "openai/text-embedding-3-small",
) -> tuple[list[dict], dict]:
    """Select up to *topk* summary rows by cosine(query_emb, summary_emb)."""
    start_time = time.perf_counter()

    with_embedding = [
        item
        for item in summary_items
        if item.get("embedding") is not None
        and isinstance(item["embedding"], list)
        and len(item["embedding"]) > 0
    ]
    empty_stats = {
        "total_cost_usd": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_wall_clock_secs": time.perf_counter() - start_time,
        "topk_wall_clock_secs": time.perf_counter() - start_time,
        "topk_items_out": 0,
        "summary_items_out": len(summary_items),
        "summary_items_with_embedding": 0,
    }
    if not with_embedding:
        return [], empty_stats

    model = LiteLLMModel(model_id=embedding_model_id, api_key=api_key)
    query_embeddings, embed_stats = model.embed(texts=[query], model=embedding_model_id)
    query_vec = np.asarray(query_embeddings[0], dtype=np.float64)
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-12)

    matrix = np.asarray([item["embedding"] for item in with_embedding], dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    matrix_n = matrix / norms
    scores = matrix_n @ qn

    k = max(0, min(topk, len(with_embedding)))
    if k == 0:
        top_items: list[dict] = []
    else:
        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        top_items = [with_embedding[i] for i in top_indices]

    end_time = time.perf_counter()
    stats = {
        "total_cost_usd": embed_stats.cost_usd,
        "total_input_tokens": embed_stats.total_input_tokens,
        "total_output_tokens": embed_stats.total_output_tokens,
        "total_wall_clock_secs": end_time - start_time,
        "topk_wall_clock_secs": end_time - start_time,
        "topk_items_out": len(top_items),
        "summary_items_out": len(summary_items),
        "summary_items_with_embedding": len(with_embedding),
    }

    return top_items, stats


def run_summary_embedding_topk(
    query: str,
    summary_items: list[dict],
    api_key: str,
    topk: int,
    embedding_model_id: str = "openai/text-embedding-3-small",
) -> tuple[list[str], dict]:
    """Top-*k* titles by cosine similarity of query embedding to summary embeddings (no SemFilter)."""
    top_items, stats = _topk_summary_items_by_embedding(
        query, summary_items, api_key, topk, embedding_model_id
    )
    titles = [str(item.get("title", "")) for item in top_items]
    return titles, stats


def run_summary_embedding_topk_then_batched_filter(
    query: str,
    summary_items: list[dict],
    model_id: str,
    api_key: str,
    topk: int,
    embedding_model_id: str,
    batch_size: int,
) -> tuple[list[str], dict]:
    """Summary embedding top-*k*, then batched SemFilter on those *k* rows only."""
    top_with_emb, retr_stats = _topk_summary_items_by_embedding(
        query, summary_items, api_key, topk, embedding_model_id
    )
    top_for_filter = [
        {k: v for k, v in item.items() if k != "embedding"} for item in top_with_emb
    ]

    if not top_for_filter:
        return [], {
            **retr_stats,
            "filter_wall_clock_secs": 0.0,
            "filter_items_out": 0,
            "filter_batch_size": max(1, batch_size),
        }

    titles, filter_stats = run_summary_filter_only(
        query=query,
        summary_items=top_for_filter,
        model_id=model_id,
        api_key=api_key,
        batch_size=max(1, batch_size),
    )

    combined = {
        "total_cost_usd": retr_stats["total_cost_usd"] + filter_stats["total_cost_usd"],
        "total_input_tokens": retr_stats["total_input_tokens"]
        + filter_stats["total_input_tokens"],
        "total_output_tokens": retr_stats["total_output_tokens"]
        + filter_stats["total_output_tokens"],
        "total_wall_clock_secs": retr_stats["total_wall_clock_secs"]
        + filter_stats["total_wall_clock_secs"],
        "topk_wall_clock_secs": retr_stats["topk_wall_clock_secs"],
        "filter_wall_clock_secs": filter_stats["filter_wall_clock_secs"],
        "topk_items_out": retr_stats["topk_items_out"],
        "summary_items_out": retr_stats["summary_items_out"],
        "summary_items_with_embedding": retr_stats["summary_items_with_embedding"],
        "filter_items_out": filter_stats["filter_items_out"],
        "filter_batch_size": filter_stats["filter_batch_size"],
    }

    return titles, combined


def generate_query_variations(
    query: str,
    api_key: str,
    model_id: str = "openai/gpt-5-mini-2025-08-07",
    total_queries: int = 3,
) -> tuple[list[str], dict]:
    """Generate meaning-preserving query rewrites for recall experiments."""
    start_time = time.perf_counter()

    if total_queries <= 1:
        return [query], {
            "total_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "variation_generation_wall_clock_secs": 0.0,
        }

    model = LiteLLMModel(model_id=model_id, api_key=api_key)
    num_new_variations = total_queries - 1
    prompt = f"""Generate {num_new_variations} semantically equivalent rewrites of the query below for semantic filtering.

Goal:
- Produce rewrites that preserve the exact meaning but are materially different in wording or sentence structure.
- These rewrites should help catch documents that might be missed due to phrasing sensitivity.

Hard constraints:
- Preserve the meaning exactly.
- Preserve all logical conditions, including AND, OR, and NOT.
- Preserve all entities, dates, quantities, countries, genres, attributes, and constraints.
- Do not broaden the query.
- Do not narrow the query.
- Do not add examples, explanations, assumptions, or new constraints.
- Each rewrite must be a standalone query string.

Diversity requirements:
- Make each rewrite noticeably different from the original and from the other rewrites.
- Use different wording, synonyms, clause ordering, or sentence structure where possible.
- If the original query is very short or highly constrained, still vary phrasing as much as possible without changing meaning.

Return JSON only in exactly this format:
{{"variations": ["...", "..."]}}

Rules for the JSON:
- The "variations" array must contain EXACTLY {num_new_variations} strings.
- Do not include the original query in the array.
- Do not include duplicate or near-duplicate rewrites.

Original query:
{query}
"""
    llm_calls = []
    variations = [query]
    try:
        message = ChatMessage(role="user", content=prompt)
        response = model.generate(
            messages=[message],
            temperature=1.0 if "gpt-5" in model_id else 0.2,
        )
        if response.llm_call_stats is not None:
            llm_calls.append(response.llm_call_stats)

        payload = json.loads(response.content.strip())
        raw_variations = payload.get("variations", [])
        for variation in raw_variations:
            normalized = str(variation).strip()
            if normalized and normalized not in variations:
                variations.append(normalized)
        variations = variations[:total_queries]
    except Exception as e:
        print(f"  Warning: query variation generation failed: {e}")

    stats = aggregate_llm_stats(llm_calls)
    stats["variation_generation_wall_clock_secs"] = time.perf_counter() - start_time
    return variations, stats


def get_item_key(item: dict) -> str:
    """Return a stable key for deduplicating result items."""
    return (
        item.get("uri")
        or item.get("path")
        or item.get("title")
        or json.dumps(item, sort_keys=True)
    )


def decompose_or_query(
    query: str,
    api_key: str,
    model_id: str = "openai/gpt-5-mini-2025-08-07",
) -> tuple[list[str], dict]:
    """Decompose an OR query into exactly two meaning-preserving clauses."""
    start_time = time.perf_counter()
    model = LiteLLMModel(model_id=model_id, api_key=api_key)
    llm_calls = []
    decomposition_method = "llm"

    prompt = f"""Decompose the following search query into exactly two standalone clauses A and B such that the original query means "A OR B".

Requirements:
- Preserve the original meaning exactly.
- Return exactly two clause strings.
- Do not broaden the query.
- Do not narrow the query.
- Keep each clause standalone and understandable on its own.
- Remove only the OR composition; do not paraphrase more than necessary.
- If the query already contains "either ... or ...", normalize that into two clean clauses.

Return JSON only in exactly this format:
{{"clauses": ["A", "B"]}}

Original query:
{query}
"""

    try:
        response = model.generate(messages=[ChatMessage(role="user", content=prompt)])
        if response.llm_call_stats is not None:
            llm_calls.append(response.llm_call_stats)

        payload = json.loads(response.content.strip())
        raw_clauses = payload.get("clauses", [])
        clauses = [str(clause).strip(" ,.") for clause in raw_clauses if str(clause).strip()]
        if len(clauses) != 2:
            raise ValueError(f"Expected exactly 2 clauses, got {clauses}")
    except Exception as e:
        print(f"  Warning: query decomposition failed, falling back to string split: {e}")
        if " or " not in query.lower():
            raise ValueError(f"Could not decompose OR query: {query}") from e

        lower_query = query.lower()
        split_idx = lower_query.rfind(" or ")
        clause_a = query[:split_idx].strip(" ,.")
        clause_b = query[split_idx + 4 :].strip(" ,.")
        if not clause_a or not clause_b:
            raise ValueError(f"Could not decompose OR query via fallback: {query}") from e
        clauses = [clause_a, clause_b]
        decomposition_method = "string_split_fallback"

    stats = aggregate_llm_stats(llm_calls)
    stats["decomposition_wall_clock_secs"] = time.perf_counter() - start_time
    stats["decomposition_method"] = decomposition_method
    stats["clauses"] = clauses
    return clauses, stats


def build_documents_input_datasets(corpus_items: list[dict]) -> dict[str, carnot.Dataset]:
    """Build the standard documents dataset used in QUEST evals."""
    dataset = carnot.Dataset(
        name="Documents",
        annotation="A set of documents with their titles and content.",
        items=corpus_items,
        dataset_id=DATASET_ID,
    )
    return {"Documents": dataset}


def run_topk_for_query(
    query: str,
    input_datasets: dict[str, carnot.Dataset],
    index_name: str,
    api_key: str,
    topk: int,
    output_dataset_id: str,
    embedding_model_id: str = "openai/text-embedding-3-small",
) -> tuple[dict[str, carnot.Dataset], object]:
    """Run a single SemTopK retrieval over the Documents dataset."""
    topk_op = SemTopKOperator(
        task=query,
        k=topk,
        output_dataset_id=output_dataset_id,
        max_workers=64,
        model_id=embedding_model_id,
        llm_config={"OPENAI_API_KEY": api_key},
        index_name=index_name,
    )
    return topk_op("Documents", input_datasets)


def run_topk_union_for_clauses(
    clauses: list[str],
    corpus_items: list[dict],
    index_name: str,
    api_key: str,
    topk: int,
    embedding_model_id: str = "openai/text-embedding-3-small",
) -> tuple[dict[str, carnot.Dataset], dict]:
    """Run SemTopK for each clause and union the retrieved items."""
    input_datasets = build_documents_input_datasets(corpus_items)
    union_items = {}
    all_topk_calls = []
    clause_topk_items_out = []

    for idx, clause in enumerate(clauses):
        topk_datasets, topk_stats = run_topk_for_query(
            query=clause,
            input_datasets=input_datasets,
            index_name=index_name,
            api_key=api_key,
            topk=topk,
            output_dataset_id=f"topk_clause_{idx}",
            embedding_model_id=embedding_model_id,
        )
        input_datasets = topk_datasets
        all_topk_calls.extend(topk_stats.llm_calls)
        clause_items = topk_datasets[f"topk_clause_{idx}"].items
        clause_topk_items_out.append(len(clause_items))
        for item in clause_items:
            union_items[get_item_key(item)] = item

    union_dataset = carnot.Dataset(
        name="topk_union_output",
        annotation="Union of per-clause SemTopK outputs.",
        items=list(union_items.values()),
        dataset_id=DATASET_ID,
    )
    output_datasets = {**input_datasets, union_dataset.name: union_dataset}

    topk_agg = aggregate_llm_stats(all_topk_calls)
    stats = {
        "total_cost_usd": topk_agg["total_cost_usd"],
        "total_input_tokens": topk_agg["total_input_tokens"],
        "total_output_tokens": topk_agg["total_output_tokens"],
        "topk_items_out": len(union_dataset.items),
        "clause_topk_items_out": clause_topk_items_out,
    }
    return output_datasets, stats


def run_filter_union(
    filter_queries: list[str],
    source_dataset_id: str,
    input_datasets: dict[str, carnot.Dataset],
    model_id: str,
    api_key: str,
) -> tuple[list[dict], dict]:
    """Run SemFilter for each query and union the passing items."""
    llm_config = {"OPENAI_API_KEY": api_key}
    union_items = {}
    all_filter_calls = []
    clause_filter_items_out = []

    for idx, filter_query in enumerate(filter_queries):
        filter_op = SemFilterOperator(
            task=filter_query,
            output_dataset_id=f"filter_clause_{idx}",
            model_id=model_id,
            llm_config=llm_config,
            max_workers=64,
        )
        final_datasets, filter_stats = filter_op(source_dataset_id, input_datasets)
        all_filter_calls.extend(filter_stats.llm_calls)
        clause_items = final_datasets[f"filter_clause_{idx}"].items
        clause_filter_items_out.append(len(clause_items))
        for item in clause_items:
            union_items[get_item_key(item)] = item

    results = list(union_items.values())
    filter_agg = aggregate_llm_stats(all_filter_calls)
    stats = {
        "total_cost_usd": filter_agg["total_cost_usd"],
        "total_input_tokens": filter_agg["total_input_tokens"],
        "total_output_tokens": filter_agg["total_output_tokens"],
        "filter_items_out": len(results),
        "clause_filter_items_out": clause_filter_items_out,
    }
    return results, stats


def run_with_or_decomposition_topk_union_filter_query(
    query: str,
    corpus_items: list[dict],
    index_name: str,
    model_id: str,
    api_key: str,
    topk: int = 50,
    decomposition_model_id: str = "openai/gpt-5-mini-2025-08-07",
    embedding_model_id: str = "openai/text-embedding-3-small",
) -> tuple[list[str], dict]:
    """Decompose query, union TopK(A)/TopK(B), then SemFilter(original query)."""
    start_time = time.perf_counter()
    llm_config = {"OPENAI_API_KEY": api_key}

    clauses, decomposition_stats = decompose_or_query(
        query=query,
        api_key=api_key,
        model_id=decomposition_model_id,
    )
    print(f"  Decomposed clauses: {clauses}")

    topk_datasets, topk_stats = run_topk_union_for_clauses(
        clauses=clauses,
        corpus_items=corpus_items,
        index_name=index_name,
        api_key=api_key,
        topk=topk,
        embedding_model_id=embedding_model_id,
    )

    topk_titles = [item.get("title", "") for item in topk_datasets["topk_union_output"].items[:10]]
    print(f"  TopK union sample: {topk_titles}")

    filter_op = SemFilterOperator(
        task=query,
        output_dataset_id="final_output",
        model_id=model_id,
        llm_config=llm_config,
        max_workers=64,
    )
    final_datasets, filter_stats = filter_op("topk_union_output", topk_datasets)

    end_time = time.perf_counter()
    results = final_datasets["final_output"].items
    titles = [item.get("title", "") for item in results]

    stats = {
        "total_cost_usd": (
            decomposition_stats["total_cost_usd"]
            + topk_stats["total_cost_usd"]
            + aggregate_llm_stats(filter_stats.llm_calls)["total_cost_usd"]
        ),
        "total_input_tokens": (
            decomposition_stats["total_input_tokens"]
            + topk_stats["total_input_tokens"]
            + aggregate_llm_stats(filter_stats.llm_calls)["total_input_tokens"]
        ),
        "total_output_tokens": (
            decomposition_stats["total_output_tokens"]
            + topk_stats["total_output_tokens"]
            + aggregate_llm_stats(filter_stats.llm_calls)["total_output_tokens"]
        ),
        "total_wall_clock_secs": end_time - start_time,
        "decomposition_wall_clock_secs": decomposition_stats["decomposition_wall_clock_secs"],
        "decomposition_method": decomposition_stats["decomposition_method"],
        "decomposed_queries": clauses,
        "topk_items_out": topk_stats["topk_items_out"],
        "clause_topk_items_out": topk_stats["clause_topk_items_out"],
        "filter_items_out": filter_stats.items_out,
    }
    return titles, stats


def run_with_or_decomposition_topk_union_only(
    query: str,
    corpus_items: list[dict],
    index_name: str,
    api_key: str,
    topk: int = 50,
    decomposition_model_id: str = "openai/gpt-5-mini-2025-08-07",
    embedding_model_id: str = "openai/text-embedding-3-small",
) -> tuple[list[str], dict]:
    """Decompose query, union TopK(A)/TopK(B), and return retrieval-only results."""
    start_time = time.perf_counter()

    clauses, decomposition_stats = decompose_or_query(
        query=query,
        api_key=api_key,
        model_id=decomposition_model_id,
    )
    print(f"  Decomposed clauses: {clauses}")

    topk_datasets, topk_stats = run_topk_union_for_clauses(
        clauses=clauses,
        corpus_items=corpus_items,
        index_name=index_name,
        api_key=api_key,
        topk=topk,
        embedding_model_id=embedding_model_id,
    )

    end_time = time.perf_counter()
    results = topk_datasets["topk_union_output"].items
    titles = [item.get("title", "") for item in results]
    topk_titles = titles[:10]
    print(f"  TopK union sample: {topk_titles}")

    stats = {
        "total_cost_usd": (
            decomposition_stats["total_cost_usd"]
            + topk_stats["total_cost_usd"]
        ),
        "total_input_tokens": (
            decomposition_stats["total_input_tokens"]
            + topk_stats["total_input_tokens"]
        ),
        "total_output_tokens": (
            decomposition_stats["total_output_tokens"]
            + topk_stats["total_output_tokens"]
        ),
        "total_wall_clock_secs": end_time - start_time,
        "decomposition_wall_clock_secs": decomposition_stats["decomposition_wall_clock_secs"],
        "decomposition_method": decomposition_stats["decomposition_method"],
        "decomposed_queries": clauses,
        "topk_items_out": topk_stats["topk_items_out"],
        "clause_topk_items_out": topk_stats["clause_topk_items_out"],
    }
    return titles, stats


def run_with_or_decomposition_topk_union_filter_union(
    query: str,
    corpus_items: list[dict],
    index_name: str,
    model_id: str,
    api_key: str,
    topk: int = 50,
    decomposition_model_id: str = "openai/gpt-5-mini-2025-08-07",
    embedding_model_id: str = "openai/text-embedding-3-small",
) -> tuple[list[str], dict]:
    """Decompose query, union TopK(A)/TopK(B), then union SemFilter(A)/SemFilter(B)."""
    start_time = time.perf_counter()

    clauses, decomposition_stats = decompose_or_query(
        query=query,
        api_key=api_key,
        model_id=decomposition_model_id,
    )
    print(f"  Decomposed clauses: {clauses}")

    topk_datasets, topk_stats = run_topk_union_for_clauses(
        clauses=clauses,
        corpus_items=corpus_items,
        index_name=index_name,
        api_key=api_key,
        topk=topk,
        embedding_model_id=embedding_model_id,
    )

    topk_titles = [item.get("title", "") for item in topk_datasets["topk_union_output"].items[:10]]
    print(f"  TopK union sample: {topk_titles}")

    results, filter_stats = run_filter_union(
        filter_queries=clauses,
        source_dataset_id="topk_union_output",
        input_datasets=topk_datasets,
        model_id=model_id,
        api_key=api_key,
    )

    end_time = time.perf_counter()
    titles = [item.get("title", "") for item in results]

    stats = {
        "total_cost_usd": (
            decomposition_stats["total_cost_usd"]
            + topk_stats["total_cost_usd"]
            + filter_stats["total_cost_usd"]
        ),
        "total_input_tokens": (
            decomposition_stats["total_input_tokens"]
            + topk_stats["total_input_tokens"]
            + filter_stats["total_input_tokens"]
        ),
        "total_output_tokens": (
            decomposition_stats["total_output_tokens"]
            + topk_stats["total_output_tokens"]
            + filter_stats["total_output_tokens"]
        ),
        "total_wall_clock_secs": end_time - start_time,
        "decomposition_wall_clock_secs": decomposition_stats["decomposition_wall_clock_secs"],
        "decomposition_method": decomposition_stats["decomposition_method"],
        "decomposed_queries": clauses,
        "topk_items_out": topk_stats["topk_items_out"],
        "clause_topk_items_out": topk_stats["clause_topk_items_out"],
        "filter_items_out": filter_stats["filter_items_out"],
        "clause_filter_items_out": filter_stats["clause_filter_items_out"],
    }
    return titles, stats


def run_with_or_decomposition_topk_query_filter_union(
    query: str,
    corpus_items: list[dict],
    index_name: str,
    model_id: str,
    api_key: str,
    topk: int = 50,
    decomposition_model_id: str = "openai/gpt-5-mini-2025-08-07",
    embedding_model_id: str = "openai/text-embedding-3-small",
) -> tuple[list[str], dict]:
    """Run TopK(original query), then union SemFilter(A)/SemFilter(B)."""
    start_time = time.perf_counter()

    clauses, decomposition_stats = decompose_or_query(
        query=query,
        api_key=api_key,
        model_id=decomposition_model_id,
    )
    print(f"  Decomposed clauses: {clauses}")

    input_datasets = build_documents_input_datasets(corpus_items)
    topk_datasets, topk_stats_obj = run_topk_for_query(
        query=query,
        input_datasets=input_datasets,
        index_name=index_name,
        api_key=api_key,
        topk=topk,
        output_dataset_id="topk_output",
        embedding_model_id=embedding_model_id,
    )
    topk_agg = aggregate_llm_stats(topk_stats_obj.llm_calls)

    topk_titles = [item.get("title", "") for item in topk_datasets["topk_output"].items[:10]]
    print(f"  TopK sample: {topk_titles}")

    results, filter_stats = run_filter_union(
        filter_queries=clauses,
        source_dataset_id="topk_output",
        input_datasets=topk_datasets,
        model_id=model_id,
        api_key=api_key,
    )

    end_time = time.perf_counter()
    titles = [item.get("title", "") for item in results]

    stats = {
        "total_cost_usd": (
            decomposition_stats["total_cost_usd"]
            + topk_agg["total_cost_usd"]
            + filter_stats["total_cost_usd"]
        ),
        "total_input_tokens": (
            decomposition_stats["total_input_tokens"]
            + topk_agg["total_input_tokens"]
            + filter_stats["total_input_tokens"]
        ),
        "total_output_tokens": (
            decomposition_stats["total_output_tokens"]
            + topk_agg["total_output_tokens"]
            + filter_stats["total_output_tokens"]
        ),
        "total_wall_clock_secs": end_time - start_time,
        "decomposition_wall_clock_secs": decomposition_stats["decomposition_wall_clock_secs"],
        "decomposition_method": decomposition_stats["decomposition_method"],
        "decomposed_queries": clauses,
        "topk_items_out": topk_stats_obj.items_out,
        "filter_items_out": filter_stats["filter_items_out"],
        "clause_filter_items_out": filter_stats["clause_filter_items_out"],
    }
    return titles, stats


def run_with_index(
    query: str,
    corpus_items: list[dict],
    index_name: str,
    model_id: str,
    api_key: str,
    topk: int = 50,
) -> tuple[list[str], dict]:
    """Run SemTopK + SemFilter pipeline.
    
    Returns:
        Tuple of (predicted_titles, stats_dict)
    """
    start_time = time.perf_counter()
    
    dataset = carnot.Dataset(
        name="Documents",
        annotation="A set of documents with their titles and content.",
        items=corpus_items,
        dataset_id=DATASET_ID,
    )
    input_datasets = {"Documents": dataset}
    llm_config = {"OPENAI_API_KEY": api_key}
    
    # Run SemTopK
    # Note: model_id is used by ChromaIndex for query embedding.
    # Flat/Hierarchical indices use config.llm_routing_model (default: gpt-5-mini)
    # for LLM-based selection/routing during search.
    topk_op = SemTopKOperator(
        task=query,
        k=topk,
        output_dataset_id="topk_output",
        max_workers=64,
        model_id="openai/text-embedding-3-small",  # Embedding model for ChromaIndex
        llm_config=llm_config,
        index_name=index_name,
    )

    topk_datasets, topk_stats = topk_op("Documents", input_datasets)
    
    topk_titles = [item.get("title", "") for item in topk_datasets["topk_output"].items[:10]]
    print(f"  TopK sample: {topk_titles}")
    # Run SemFilter on TopK results
    filter_op = SemFilterOperator(
        task=query,
        output_dataset_id="final_output",
        model_id=model_id,
        llm_config=llm_config,
        max_workers=64,
    )
    final_datasets, filter_stats = filter_op("topk_output", topk_datasets)
    
    end_time = time.perf_counter()
    
    # Extract results
    results = final_datasets["final_output"].items
    titles = [item.get("title", "") for item in results]
    
    # Aggregate stats
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    
    for stat in topk_stats.llm_calls:
        total_cost += stat.cost_usd
        total_input_tokens += stat.total_input_tokens
        total_output_tokens += stat.total_output_tokens
    
    for stat in filter_stats.llm_calls:
        total_cost += stat.cost_usd
        total_input_tokens += stat.total_input_tokens
        total_output_tokens += stat.total_output_tokens
    
    stats = {
        "total_cost_usd": total_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_wall_clock_secs": end_time - start_time,
        "topk_wall_clock_secs": topk_stats.wall_clock_secs,
        "filter_wall_clock_secs": filter_stats.wall_clock_secs,
        "topk_items_out": topk_stats.items_out,
        "filter_items_out": filter_stats.items_out,
    }
    
    return titles, stats


def run_vector_topk_then_batched_filter(
    query: str,
    corpus_items: list[dict],
    index_name: str,
    model_id: str,
    api_key: str,
    topk: int = 50,
    embedding_model_id: str = "openai/text-embedding-3-small",
    batch_size: int = 16,
) -> tuple[list[str], dict]:
    """Vector top-*K* over full documents (SemTopK / Chroma), then batched SemFilter on those *K* rows only."""
    start_time = time.perf_counter()

    dataset = carnot.Dataset(
        name="Documents",
        annotation="A set of documents with their titles and content.",
        items=corpus_items,
        dataset_id=DATASET_ID,
    )
    input_datasets = {"Documents": dataset}
    llm_config = {"OPENAI_API_KEY": api_key}

    topk_op = SemTopKOperator(
        task=query,
        k=topk,
        output_dataset_id="topk_output",
        max_workers=64,
        model_id=embedding_model_id,
        llm_config=llm_config,
        index_name=index_name,
    )
    topk_datasets, topk_stats = topk_op("Documents", input_datasets)

    topk_titles = [item.get("title", "") for item in topk_datasets["topk_output"].items[:10]]
    print(f"  TopK sample: {topk_titles}")

    filter_op = SemFilterOperator(
        task=query,
        output_dataset_id="final_output",
        model_id=model_id,
        llm_config=llm_config,
        max_workers=64,
        batch_size=max(1, batch_size),
    )
    final_datasets, filter_stats = filter_op("topk_output", topk_datasets)

    end_time = time.perf_counter()

    results = final_datasets["final_output"].items
    titles = [item.get("title", "") for item in results]

    topk_agg = aggregate_llm_stats(topk_stats.llm_calls)
    filter_agg = aggregate_llm_stats(filter_stats.llm_calls)
    stats = {
        "total_cost_usd": topk_agg["total_cost_usd"] + filter_agg["total_cost_usd"],
        "total_input_tokens": topk_agg["total_input_tokens"] + filter_agg["total_input_tokens"],
        "total_output_tokens": topk_agg["total_output_tokens"] + filter_agg["total_output_tokens"],
        "total_wall_clock_secs": end_time - start_time,
        "topk_wall_clock_secs": topk_stats.wall_clock_secs,
        "filter_wall_clock_secs": filter_stats.wall_clock_secs,
        "topk_items_out": topk_stats.items_out,
        "filter_items_out": filter_stats.items_out,
        "filter_batch_size": max(1, batch_size),
    }
    return titles, stats


def _run_sem_filter_items_titles(
    query: str,
    items: list[dict],
    model_id: str,
    api_key: str,
    batch_size: int,
    dataset_name: str,
    annotation: str,
    max_workers: int = 64,
) -> tuple[list[str], dict]:
    """Run SemFilter on an item list; return passing items' titles (QUEST metrics)."""
    start_time = time.perf_counter()

    dataset = carnot.Dataset(
        name=dataset_name,
        annotation=annotation,
        items=items,
        dataset_id=DATASET_ID,
    )
    input_datasets = {dataset_name: dataset}
    llm_config = {"OPENAI_API_KEY": api_key}

    filter_op = SemFilterOperator(
        task=query,
        output_dataset_id="final_output",
        model_id=model_id,
        llm_config=llm_config,
        max_workers=max_workers,
        batch_size=max(1, batch_size),
    )
    final_datasets, filter_stats = filter_op(dataset_name, input_datasets)

    end_time = time.perf_counter()

    results = final_datasets["final_output"].items
    titles = [item.get("title", "") for item in results]

    agg = aggregate_llm_stats(filter_stats.llm_calls)
    stats = {
        "total_cost_usd": agg["total_cost_usd"],
        "total_input_tokens": agg["total_input_tokens"],
        "total_output_tokens": agg["total_output_tokens"],
        "total_wall_clock_secs": end_time - start_time,
        "filter_wall_clock_secs": filter_stats.wall_clock_secs,
        "filter_items_out": filter_stats.items_out,
        "filter_batch_size": max(1, batch_size),
        "semfilter_items_in": len(items),
    }

    return titles, stats


def run_vector_topk_then_qdt_semfilter(
    query: str,
    corpus_items: list[dict],
    index_name: str,
    model_id: str,
    qdt_model_id: str,
    api_key: str,
    topk: int,
    embedding_model_id: str,
    qdt_semfilter_batch_size: int,
) -> tuple[list[str], dict]:
    """Vector top-*K* over full docs → QDT (anchor + probes) → SemFilter per task on survivors; union titles."""
    start_total = time.perf_counter()

    dataset = carnot.Dataset(
        name="Documents",
        annotation="A set of documents with their titles and content.",
        items=corpus_items,
        dataset_id=DATASET_ID,
    )
    input_datasets = {"Documents": dataset}
    llm_config = {"OPENAI_API_KEY": api_key}

    topk_op = SemTopKOperator(
        task=query,
        k=topk,
        output_dataset_id="topk_output",
        max_workers=64,
        model_id=embedding_model_id,
        llm_config=llm_config,
        index_name=index_name,
    )
    topk_datasets, topk_stats = topk_op("Documents", input_datasets)

    topk_items = list(topk_datasets["topk_output"].items)
    topk_titles = [item.get("title", "") for item in topk_items[:10]]
    print(f"  TopK sample: {topk_titles}")
    print("TOP K DONE")

    topk_agg = aggregate_llm_stats(topk_stats.llm_calls)

    if not topk_items:
        return [], {
            "total_cost_usd": topk_agg["total_cost_usd"],
            "total_input_tokens": topk_agg["total_input_tokens"],
            "total_output_tokens": topk_agg["total_output_tokens"],
            "total_wall_clock_secs": time.perf_counter() - start_total,
            "topk_wall_clock_secs": topk_stats.wall_clock_secs,
            "topk_items_out": topk_stats.items_out,
            "qdt_union_titles_count": 0,
            "qdt_tasks": 0,
            "qdt_passes": [],
            "qdt_wall_clock_secs": 0.0,
            "qdt_semfilter_wall_clock_secs": 0.0,
            "filter_items_out": 0,
            "qdt_semfilter_batch_size": max(1, qdt_semfilter_batch_size),
        }

    qdt, qdt_stat = generate_qdt_from_query(
        query=query,
        api_key=api_key,
        model_id=qdt_model_id,
    )
    print("QDT GENERATION DONE")
    tasks = dedupe_qdt_tasks(qdt["anchor"], qdt["probes"])

    union: set[str] = set()
    passes: list[dict] = []
    qdt_sf_cost = 0.0
    qdt_sf_in = 0
    qdt_sf_out = 0

    def _qdt_semfilter_one(pair: tuple[str, str]) -> tuple[str, str, list[str], dict]:
        label, task = pair
        titles, st = _run_sem_filter_items_titles(
            query=task,
            items=topk_items,
            model_id=model_id,
            api_key=api_key,
            batch_size=max(1, qdt_semfilter_batch_size),
            dataset_name="TopKDocuments",
            annotation="Top-K full documents from vector retrieval (QUEST corpus).",
            max_workers=16,
        )
        return label, task, titles, st

    n_qdt_workers = min(64, max(1, len(tasks)))
    qdt_sf_t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_qdt_workers) as pool:
        qdt_pass_results = list(pool.map(_qdt_semfilter_one, tasks))
    qdt_sf_wall = time.perf_counter() - qdt_sf_t0

    for label, task, titles, st in qdt_pass_results:
        for t in titles:
            if t:
                union.add(str(t))
        qdt_sf_cost += float(st.get("total_cost_usd", 0.0))
        qdt_sf_in += int(st.get("total_input_tokens", 0))
        qdt_sf_out += int(st.get("total_output_tokens", 0))
        preview = task if len(task) <= 500 else task[:500] + "..."
        passes.append(
            {
                "label": label,
                "task": preview,
                "filter_items_out": st.get("filter_items_out"),
                "semfilter_items_in": st.get("semfilter_items_in"),
            }
        )

    qdt_cost = float(qdt_stat.cost_usd) if qdt_stat else 0.0
    qdt_in = int(qdt_stat.total_input_tokens) if qdt_stat else 0
    qdt_out = int(qdt_stat.total_output_tokens) if qdt_stat else 0
    qdt_wall = float(qdt_stat.duration_secs) if qdt_stat else 0.0

    combined = {
        "total_cost_usd": topk_agg["total_cost_usd"] + qdt_cost + qdt_sf_cost,
        "total_input_tokens": topk_agg["total_input_tokens"] + qdt_in + qdt_sf_in,
        "total_output_tokens": topk_agg["total_output_tokens"] + qdt_out + qdt_sf_out,
        "total_wall_clock_secs": time.perf_counter() - start_total,
        "topk_wall_clock_secs": topk_stats.wall_clock_secs,
        "qdt_wall_clock_secs": qdt_wall,
        "qdt_semfilter_wall_clock_secs": qdt_sf_wall,
        "topk_items_out": topk_stats.items_out,
        "qdt_semfilter_batch_size": max(1, qdt_semfilter_batch_size),
        "qdt": qdt,
        "qdt_tasks": len(tasks),
        "qdt_passes": passes,
        "qdt_union_titles_count": len(union),
        "filter_items_out": len(union),
    }

    return sorted(union), combined


def run_only_topk(
    query: str,
    corpus_items: list[dict],
    index_name: str,
    api_key: str,
    topk: int = 50,
    embedding_model_id="openai/text-embedding-3-small"
) -> tuple[list[str], dict]:
    """Run SemTopK only (no SemFilter).

    Returns:
        Tuple of (predicted_titles, stats_dict)
    """
    start_time = time.perf_counter()

    dataset = carnot.Dataset(
        name="Documents",
        annotation="A set of documents with their titles and content.",
        items=corpus_items,
        dataset_id=DATASET_ID,
    )
    input_datasets = {"Documents": dataset}
    llm_config = {"OPENAI_API_KEY": api_key}

    topk_op = SemTopKOperator(
        task=query,
        k=topk,
        output_dataset_id="topk_output",
        max_workers=64,
        model_id="openai/text-embedding-3-small",  # Embedding model for ChromaIndex
        llm_config=llm_config,
        index_name=index_name,
    )
    topk_datasets, topk_stats = topk_op("Documents", input_datasets)

    end_time = time.perf_counter()

    results = topk_datasets["topk_output"].items
    titles = [item.get("title", "") for item in results]

    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for stat in topk_stats.llm_calls:
        total_cost += stat.cost_usd
        total_input_tokens += stat.total_input_tokens
        total_output_tokens += stat.total_output_tokens

    stats = {
        "total_cost_usd": total_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_wall_clock_secs": end_time - start_time,
        "topk_wall_clock_secs": topk_stats.wall_clock_secs,
        "topk_items_out": topk_stats.items_out,
    }

    return titles, stats


def run_no_index(
    query: str,
    corpus_items: list[dict],
    model_id: str,
    api_key: str,
) -> tuple[list[str], dict]:
    """Run SemFilter only (no index).
    
    Returns:
        Tuple of (predicted_titles, stats_dict)
    """
    start_time = time.perf_counter()
    
    dataset = carnot.Dataset(
        name="Documents",
        annotation="A set of documents with their titles and content.",
        items=corpus_items,
        dataset_id=DATASET_ID,
    )
    input_datasets = {"Documents": dataset}
    llm_config = {"OPENAI_API_KEY": api_key}
    
    # Run SemFilter directly on all documents
    filter_op = SemFilterOperator(
        task=query,
        output_dataset_id="final_output",
        model_id=model_id,
        llm_config=llm_config,
        max_workers=64,
    )
    final_datasets, filter_stats = filter_op("Documents", input_datasets)
    
    end_time = time.perf_counter()
    
    # Extract results
    results = final_datasets["final_output"].items
    titles = [item.get("title", "") for item in results]
    
    # Aggregate stats
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    
    for stat in filter_stats.llm_calls:
        total_cost += stat.cost_usd
        total_input_tokens += stat.total_input_tokens
        total_output_tokens += stat.total_output_tokens
    
    stats = {
        "total_cost_usd": total_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_wall_clock_secs": end_time - start_time,
        "filter_wall_clock_secs": filter_stats.wall_clock_secs,
        "filter_items_out": filter_stats.items_out,
    }
    
    return titles, stats


def run_summary_filter_only(
    query: str,
    summary_items: list[dict],
    model_id: str,
    api_key: str,
    batch_size: int = 1,
) -> tuple[list[str], dict]:
    """Run SemFilter directly over summary-backed items."""
    start_time = time.perf_counter()

    dataset = carnot.Dataset(
        name="SummaryDocuments",
        annotation="A set of document summaries with titles and metadata.",
        items=summary_items,
        dataset_id=DATASET_ID,
    )
    input_datasets = {"SummaryDocuments": dataset}
    llm_config = {"OPENAI_API_KEY": api_key}

    filter_op = SemFilterOperator(
        task=query,
        output_dataset_id="final_output",
        model_id=model_id,
        llm_config=llm_config,
        max_workers=64,
        batch_size=max(1, batch_size),
    )
    final_datasets, filter_stats = filter_op("SummaryDocuments", input_datasets)

    end_time = time.perf_counter()

    results = final_datasets["final_output"].items
    titles = [item.get("title", "") for item in results]

    agg = aggregate_llm_stats(filter_stats.llm_calls)
    stats = {
        "total_cost_usd": agg["total_cost_usd"],
        "total_input_tokens": agg["total_input_tokens"],
        "total_output_tokens": agg["total_output_tokens"],
        "total_wall_clock_secs": end_time - start_time,
        "filter_wall_clock_secs": filter_stats.wall_clock_secs,
        "summary_items_out": len(summary_items),
        "filter_items_out": filter_stats.items_out,
        "filter_batch_size": max(1, batch_size),
    }
    return titles, stats


def run_with_index_summary_filter(
    query: str,
    corpus_items: list[dict],
    summary_items_by_path: dict[str, dict],
    index_name: str,
    model_id: str,
    api_key: str,
    topk: int = 50,
    embedding_model_id: str = "openai/text-embedding-3-small",
) -> tuple[list[str], dict]:
    """Run SemTopK over raw docs, then SemFilter over cached summaries of the top-k docs."""
    start_time = time.perf_counter()

    dataset = carnot.Dataset(
        name="Documents",
        annotation="A set of documents with their titles and content.",
        items=corpus_items,
        dataset_id=DATASET_ID,
    )
    input_datasets = {"Documents": dataset}
    llm_config = {"OPENAI_API_KEY": api_key}

    topk_op = SemTopKOperator(
        task=query,
        k=topk,
        output_dataset_id="topk_output",
        max_workers=64,
        model_id=embedding_model_id,
        llm_config=llm_config,
        index_name=index_name,
    )
    topk_datasets, topk_stats = topk_op("Documents", input_datasets)

    topk_items = topk_datasets["topk_output"].items
    topk_titles = [item.get("title", "") for item in topk_items[:10]]
    print(f"  TopK sample: {topk_titles}")

    topk_summary_items = []
    for item in topk_items:
        path = item.get("path")
        if path and path in summary_items_by_path:
            topk_summary_items.append(summary_items_by_path[path])

    summary_dataset = carnot.Dataset(
        name="SummaryDocuments",
        annotation="Summaries of top-k retrieved documents.",
        items=topk_summary_items,
        dataset_id=DATASET_ID,
    )
    summary_input_datasets = {**topk_datasets, "SummaryDocuments": summary_dataset}

    filter_op = SemFilterOperator(
        task=query,
        output_dataset_id="final_output",
        model_id=model_id,
        llm_config=llm_config,
        max_workers=64,
    )
    final_datasets, filter_stats = filter_op("SummaryDocuments", summary_input_datasets)

    end_time = time.perf_counter()

    results = final_datasets["final_output"].items
    titles = [item.get("title", "") for item in results]

    topk_agg = aggregate_llm_stats(topk_stats.llm_calls)
    filter_agg = aggregate_llm_stats(filter_stats.llm_calls)
    stats = {
        "total_cost_usd": topk_agg["total_cost_usd"] + filter_agg["total_cost_usd"],
        "total_input_tokens": topk_agg["total_input_tokens"] + filter_agg["total_input_tokens"],
        "total_output_tokens": topk_agg["total_output_tokens"] + filter_agg["total_output_tokens"],
        "total_wall_clock_secs": end_time - start_time,
        "topk_wall_clock_secs": topk_stats.wall_clock_secs,
        "filter_wall_clock_secs": filter_stats.wall_clock_secs,
        "topk_items_out": topk_stats.items_out,
        "summary_items_out": len(topk_summary_items),
        "filter_items_out": filter_stats.items_out,
    }
    return titles, stats


def run_with_index_paraphrase_union(
    query: str,
    corpus_items: list[dict],
    index_name: str,
    model_id: str,
    api_key: str,
    topk: int = 50,
    num_query_variations: int = 3,
    paraphrase_model_id: str = "openai/gpt-5-mini-2025-08-07",
) -> tuple[list[str], dict]:
    """Run SemTopK once, then SemFilter for each query rewrite and union results."""
    start_time = time.perf_counter()

    dataset = carnot.Dataset(
        name="Documents",
        annotation="A set of documents with their titles and content.",
        items=corpus_items,
        dataset_id=DATASET_ID,
    )
    input_datasets = {"Documents": dataset}
    llm_config = {"OPENAI_API_KEY": api_key}

    topk_op = SemTopKOperator(
        task=query,
        k=topk,
        output_dataset_id="topk_output",
        max_workers=64,
        model_id="openai/text-embedding-3-small",  # Embedding model for ChromaIndex
        llm_config=llm_config,
        index_name=index_name,
    )
    topk_datasets, topk_stats = topk_op("Documents", input_datasets)

    query_variations, variation_stats = generate_query_variations(
        query=query,
        api_key=api_key,
        model_id=paraphrase_model_id,
        total_queries=num_query_variations,
    )
    print(f"  Query variations: {query_variations}")

    all_filter_calls = []
    union_items = {}
    for variation in query_variations:
        filter_op = SemFilterOperator(
            task=variation,
            output_dataset_id="final_output",
            model_id=model_id,
            llm_config=llm_config,
            max_workers=64,
        )
        final_datasets, filter_stats = filter_op("topk_output", topk_datasets)
        all_filter_calls.extend(filter_stats.llm_calls)

        for item in final_datasets["final_output"].items:
            key = item.get("uri") or item.get("path") or item.get("title") or json.dumps(item, sort_keys=True)
            union_items[key] = item

    end_time = time.perf_counter()

    results = list(union_items.values())
    titles = [item.get("title", "") for item in results]

    topk_agg = aggregate_llm_stats(topk_stats.llm_calls)
    filter_agg = aggregate_llm_stats(all_filter_calls)
    stats = {
        "total_cost_usd": (
            topk_agg["total_cost_usd"]
            + filter_agg["total_cost_usd"]
            + variation_stats["total_cost_usd"]
        ),
        "total_input_tokens": (
            topk_agg["total_input_tokens"]
            + filter_agg["total_input_tokens"]
            + variation_stats["total_input_tokens"]
        ),
        "total_output_tokens": (
            topk_agg["total_output_tokens"]
            + filter_agg["total_output_tokens"]
            + variation_stats["total_output_tokens"]
        ),
        "total_wall_clock_secs": end_time - start_time,
        "topk_wall_clock_secs": topk_stats.wall_clock_secs,
        "variation_generation_wall_clock_secs": variation_stats["variation_generation_wall_clock_secs"],
        "topk_items_out": topk_stats.items_out,
        "filter_items_out": len(results),
        "num_query_variations_used": len(query_variations),
        "query_variations": query_variations,
    }

    return titles, stats


def compute_metrics(predicted: list[str], ground_truth: list[str]) -> dict:
    """Compute precision, recall, and F1 score."""
    preds = set(predicted)
    labels = set(ground_truth)
    
    tp = len(preds & labels)
    fp = len(preds - labels)
    fn = len(labels - preds)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Carnot with different index configurations")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "chroma",
            "flat",
            "hierarchical",
            "vector-only",
            "vector-topk-filter-batched",
            "vector-topk-qdt-semfilter",
            "vector-paraphrase-union",
            "vector-or-topk-union-only",
            "vector-or-topk-union-filter-query",
            "vector-or-topk-union-filter-union",
            "vector-or-topk-query-filter-union",
            "summary-filter-only",
            "summary-filter-batched",
            "summary-embedding-topk",
            "summary-embedding-topk-filter-batched",
            "chroma-summary-filter",
            "no-index",
        ],
        help="Index mode to use",
    )
    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Path to queries JSONL file",
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain to filter queries (e.g., 'films', 'books')",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="data/corpus_first25",
        help="Path to corpus directory",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Number of queries to run (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-5-2025-08-07",
        help="Model to use for SemFilter execution",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="openai/text-embedding-3-small",
        help="Model to use for embedding (default: openai/text-embedding-3-small)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help=(
            "Top-k for SemTopK, vector-topk-filter-batched, vector-topk-qdt-semfilter, "
            "summary-embedding-topk, and summary-embedding-topk-filter-batched (default: 50). "
            "Match --embedding-model to cached summaries for embedding modes."
        ),
    )
    parser.add_argument(
        "--filter-batch-size",
        type=int,
        default=16,
        help=(
            "SemFilter batch size for summary-filter-batched, "
            "summary-embedding-topk-filter-batched, vector-topk-filter-batched, and "
            "vector-topk-qdt-semfilter QDT passes (default: 16; overridden by --qdt-filter-batch-size)"
        ),
    )
    parser.add_argument(
        "--qdt-filter-batch-size",
        type=int,
        default=None,
        help=(
            "SemFilter batch size for QDT anchor/probe passes on vector-topk survivors "
            "(vector-topk-qdt-semfilter; default: same as --filter-batch-size)"
        ),
    )
    parser.add_argument(
        "--qdt-model",
        type=str,
        default=None,
        help="Model for QDT anchor/probe generation (vector-topk-qdt-semfilter; default: same as --model)",
    )
    parser.add_argument(
        "--num-query-variations",
        type=int,
        default=3,
        help="Total query variations to evaluate in paraphrase-union mode, including the original query",
    )
    parser.add_argument(
        "--paraphrase-model",
        type=str,
        default="openai/gpt-5-mini-2025-08-07",
        help="Model used to generate safe paraphrases for paraphrase-union mode",
    )
    parser.add_argument(
        "--decomposition-model",
        type=str,
        default="openai/gpt-5-mini-2025-08-07",
        help="Model used to decompose OR queries into two clauses",
    )
    parser.add_argument(
        "--summary-model",
        type=str,
        default="openai/gpt-5-mini-2025-08-07",
        help="Model used to build/load summaries for summary filter modes",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    args = parser.parse_args()

    if args.qdt_model is None:
        args.qdt_model = args.model
    if args.qdt_filter_batch_size is None:
        args.qdt_filter_batch_size = args.filter_batch_size

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Load corpus
    corpus_path = f"{args.corpus_dir}/corpus.jsonl"
    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path) as f:
        raw_items = [json.loads(line) for line in f]
    corpus_items = add_paths_to_items(raw_items)
    print(f"Loaded {len(corpus_items)} items")

    summary_items = None
    summary_items_by_path = None
    summary_prep_stats = None
    if args.mode in SUMMARY_PREP_MODES:
        print(f"Loading cached summaries with model: {args.summary_model}...")
        summary_items, summary_prep_stats = build_summary_items(
            items=corpus_items,
            api_key=api_key,
            summary_model_id=args.summary_model,
            embedding_model_id=args.embedding_model,
            include_embeddings=(args.mode in SUMMARY_MODES_WITH_EMBEDDINGS),
        )
        summary_items_by_path = {
            item["path"]: item for item in summary_items if item.get("path")
        }
        print(f"Prepared {len(summary_items)} summary items")
    
    # Load queries
    print(f"Loading queries from {args.queries}...")
    queries = []
    with open(args.queries) as f:
        for line in f:
            d = json.loads(line)
            if d['metadata']['domain'] == args.domain:
                queries.append(d)
    
    # Limit queries if specified
    if args.num_queries is not None:
        queries = queries[:args.num_queries]
    
    print(f"Running {len(queries)} queries in mode: {args.mode}")
    print(f"Execution model: {args.model}")
    print("=" * 60)
    
    # Run evaluation
    results = []
    total_stats = {
        "total_cost_usd": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_wall_clock_secs": 0.0,
    }
    if summary_prep_stats is not None:
        total_stats["total_cost_usd"] += summary_prep_stats["total_cost_usd"]
        total_stats["total_input_tokens"] += summary_prep_stats["total_input_tokens"]
        total_stats["total_output_tokens"] += summary_prep_stats["total_output_tokens"]
        total_stats["total_wall_clock_secs"] += summary_prep_stats["summary_build_wall_clock_secs"]
    
    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] {query['query'][:70]}...")
        
        try:
            if args.mode == "no-index":
                pred_titles, exec_stats = run_no_index(
                    query=query['query'],
                    corpus_items=corpus_items,
                    model_id=args.model,
                    api_key=api_key,
                )
            elif args.mode == "vector-only":
                pred_titles, exec_stats = run_only_topk(
                    query=query['query'],
                    corpus_items=corpus_items,
                    embedding_model_id=args.embedding_model,
                    index_name="chroma",
                    api_key=api_key,
                    topk=args.topk,
                )
            elif args.mode == "vector-topk-filter-batched":
                pred_titles, exec_stats = run_vector_topk_then_batched_filter(
                    query=query['query'],
                    corpus_items=corpus_items,
                    index_name="chroma",
                    model_id=args.model,
                    api_key=api_key,
                    topk=args.topk,
                    embedding_model_id=args.embedding_model,
                    batch_size=max(1, args.filter_batch_size),
                )
            elif args.mode == "vector-topk-qdt-semfilter":
                pred_titles, exec_stats = run_vector_topk_then_qdt_semfilter(
                    query=query['query'],
                    corpus_items=corpus_items,
                    index_name="chroma",
                    model_id=args.model,
                    qdt_model_id=args.qdt_model,
                    api_key=api_key,
                    topk=args.topk,
                    embedding_model_id=args.embedding_model,
                    qdt_semfilter_batch_size=max(1, args.qdt_filter_batch_size),
                )
            elif args.mode == "vector-paraphrase-union":
                pred_titles, exec_stats = run_with_index_paraphrase_union(
                    query=query['query'],
                    corpus_items=corpus_items,
                    index_name="chroma",
                    model_id=args.model,
                    api_key=api_key,
                    topk=args.topk,
                    num_query_variations=args.num_query_variations,
                    paraphrase_model_id=args.paraphrase_model,
                )
            elif args.mode == "vector-or-topk-union-only":
                pred_titles, exec_stats = run_with_or_decomposition_topk_union_only(
                    query=query['query'],
                    corpus_items=corpus_items,
                    index_name="chroma",
                    api_key=api_key,
                    topk=args.topk,
                    decomposition_model_id=args.decomposition_model,
                    embedding_model_id=args.embedding_model,
                )
            elif args.mode == "vector-or-topk-union-filter-query":
                pred_titles, exec_stats = run_with_or_decomposition_topk_union_filter_query(
                    query=query['query'],
                    corpus_items=corpus_items,
                    index_name="chroma",
                    model_id=args.model,
                    api_key=api_key,
                    topk=args.topk,
                    decomposition_model_id=args.decomposition_model,
                    embedding_model_id=args.embedding_model,
                )
            elif args.mode == "vector-or-topk-union-filter-union":
                pred_titles, exec_stats = run_with_or_decomposition_topk_union_filter_union(
                    query=query['query'],
                    corpus_items=corpus_items,
                    index_name="chroma",
                    model_id=args.model,
                    api_key=api_key,
                    topk=args.topk,
                    decomposition_model_id=args.decomposition_model,
                    embedding_model_id=args.embedding_model,
                )
            elif args.mode == "vector-or-topk-query-filter-union":
                pred_titles, exec_stats = run_with_or_decomposition_topk_query_filter_union(
                    query=query['query'],
                    corpus_items=corpus_items,
                    index_name="chroma",
                    model_id=args.model,
                    api_key=api_key,
                    topk=args.topk,
                    decomposition_model_id=args.decomposition_model,
                    embedding_model_id=args.embedding_model,
                )
            elif args.mode == "summary-filter-only":
                pred_titles, exec_stats = run_summary_filter_only(
                    query=query['query'],
                    summary_items=summary_items or [],
                    model_id=args.model,
                    api_key=api_key,
                    batch_size=1,
                )
            elif args.mode == "summary-filter-batched":
                pred_titles, exec_stats = run_summary_filter_only(
                    query=query['query'],
                    summary_items=summary_items or [],
                    model_id=args.model,
                    api_key=api_key,
                    batch_size=max(1, args.filter_batch_size),
                )
            elif args.mode == "summary-embedding-topk":
                pred_titles, exec_stats = run_summary_embedding_topk(
                    query=query['query'],
                    summary_items=summary_items or [],
                    api_key=api_key,
                    topk=args.topk,
                    embedding_model_id=args.embedding_model,
                )
            elif args.mode == "summary-embedding-topk-filter-batched":
                pred_titles, exec_stats = run_summary_embedding_topk_then_batched_filter(
                    query=query['query'],
                    summary_items=summary_items or [],
                    model_id=args.model,
                    api_key=api_key,
                    topk=args.topk,
                    embedding_model_id=args.embedding_model,
                    batch_size=max(1, args.filter_batch_size),
                )
            elif args.mode == "chroma-summary-filter":
                pred_titles, exec_stats = run_with_index_summary_filter(
                    query=query['query'],
                    corpus_items=corpus_items,
                    summary_items_by_path=summary_items_by_path or {},
                    index_name="chroma",
                    model_id=args.model,
                    api_key=api_key,
                    topk=args.topk,
                    embedding_model_id=args.embedding_model,
                )
            else:
                pred_titles, exec_stats = run_with_index(
                    query=query['query'],
                    corpus_items=corpus_items,
                    index_name=args.mode,
                    model_id=args.model,
                    api_key=api_key,
                    topk=args.topk,
                )
            
            # Compute metrics
            metrics = compute_metrics(pred_titles, query['docs'])
            
            # Accumulate stats
            total_stats["total_cost_usd"] += exec_stats["total_cost_usd"]
            total_stats["total_input_tokens"] += exec_stats["total_input_tokens"]
            total_stats["total_output_tokens"] += exec_stats["total_output_tokens"]
            total_stats["total_wall_clock_secs"] += exec_stats["total_wall_clock_secs"]
            
            print(f"  P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")
            print(f"  Cost: ${exec_stats['total_cost_usd']:.8f}, Time: {exec_stats['total_wall_clock_secs']:.2f}s")
            print(f"  Predicted: {len(pred_titles)}, GT: {len(query['docs'])}")
            print(
                f"  TopK out: {exec_stats.get('topk_items_out', 'n/a')}, "
                f"Filter out: {exec_stats.get('filter_items_out', 'n/a')}"
            )
            if args.mode == "vector-topk-qdt-semfilter":
                qdt_obj = exec_stats.get("qdt") or {}
                n_probes = len(qdt_obj.get("probes", []))
                print(
                    f"  QDT: {exec_stats.get('qdt_tasks', 'n/a')} tasks "
                    f"({1 + n_probes} anchor+probes), "
                    f"union titles: {exec_stats.get('qdt_union_titles_count', 'n/a')}"
                )

            result = {
                "query": query['query'],
                "predicted_docs": pred_titles,
                "ground_truth_docs": query['docs'],
                "metrics": metrics,
                "execution_stats": exec_stats,
            }
            results.append(result)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({
                "query": query['query'],
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
    
    # Compute averages
    successful_results = [r for r in results if "metrics" in r]
    n = len(successful_results)
    
    if n > 0:
        avg_precision = sum(r['metrics']['precision'] for r in successful_results) / n
        avg_recall = sum(r['metrics']['recall'] for r in successful_results) / n
        avg_f1 = sum(r['metrics']['f1'] for r in successful_results) / n
        avg_latency = total_stats["total_wall_clock_secs"] / n
        avg_cost = total_stats["total_cost_usd"] / n
    else:
        avg_precision = avg_recall = avg_f1 = avg_latency = avg_cost = 0.0
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY - Mode: {args.mode}")
    print("=" * 60)
    print(f"Queries executed: {n}/{len(queries)}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall:    {avg_recall:.4f}")
    print(f"Average F1 Score:  {avg_f1:.4f}")
    print("-" * 60)
    print(f"Average Latency:   {avg_latency:.2f}s")
    print(f"Average Cost:      ${avg_cost:.8f}")
    print(f"Total Cost:        ${total_stats['total_cost_usd']:.8f}")
    print(f"Total Time:        {total_stats['total_wall_clock_secs']:.2f}s")
    print(f"Total Input Tokens:  {total_stats['total_input_tokens']:,}")
    print(f"Total Output Tokens: {total_stats['total_output_tokens']:,}")
    
    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    output_path = f"{args.output_dir}/eval_{args.mode}_{args.domain}_{ts}.json"
    
    output_data = {
        "mode": args.mode,
        "domain": args.domain,
        "model": args.model,
        "topk": args.topk
        if args.mode
        not in {"no-index", "summary-filter-only", "summary-filter-batched"}
        else None,
        "embedding_model": args.embedding_model
        if args.mode
        in {
            "vector-topk-filter-batched",
            "vector-topk-qdt-semfilter",
            "summary-embedding-topk",
            "summary-embedding-topk-filter-batched",
        }
        else None,
        "filter_batch_size": args.filter_batch_size
        if args.mode
        in {
            "summary-filter-batched",
            "summary-embedding-topk-filter-batched",
            "vector-topk-filter-batched",
        }
        else None,
        "qdt_model": args.qdt_model if args.mode == "vector-topk-qdt-semfilter" else None,
        "qdt_filter_batch_size": args.qdt_filter_batch_size
        if args.mode == "vector-topk-qdt-semfilter"
        else None,
        "paraphrase_model": args.paraphrase_model if args.mode == "vector-paraphrase-union" else None,
        "num_query_variations": args.num_query_variations if args.mode == "vector-paraphrase-union" else None,
        "decomposition_model": args.decomposition_model if args.mode in {
            "vector-or-topk-union-only",
            "vector-or-topk-union-filter-query",
            "vector-or-topk-union-filter-union",
            "vector-or-topk-query-filter-union",
        } else None,
        "summary_model": args.summary_model if args.mode in SUMMARY_PREP_MODES else None,
        "summary_prep_stats": summary_prep_stats if args.mode in SUMMARY_PREP_MODES else None,
        "num_queries": len(queries),
        "num_successful": n,
        "summary": {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "avg_latency_secs": avg_latency,
            "avg_cost_usd": avg_cost,
            "total_cost_usd": total_stats["total_cost_usd"],
            "total_time_secs": total_stats["total_wall_clock_secs"],
            "total_input_tokens": total_stats["total_input_tokens"],
            "total_output_tokens": total_stats["total_output_tokens"],
        },
        "results": results,
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
