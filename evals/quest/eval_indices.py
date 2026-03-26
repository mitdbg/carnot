import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

import carnot
from carnot.agents.models import ChatMessage, LiteLLMModel
from carnot.index.models import HierarchicalIndexConfig
from carnot.index.summary_layer import SummaryLayer
from carnot.operators.sem_filter import SemFilterOperator
from carnot.operators.sem_topk import SemTopKOperator

load_dotenv()


DATASET_ID = 1


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
) -> tuple[list[dict], dict]:
    """Build summary-backed items, reusing cached summaries when available."""
    start_time = time.perf_counter()

    config = HierarchicalIndexConfig(summary_model=summary_model_id)
    layer = SummaryLayer(config=config, api_key=api_key)
    summaries = layer.get_or_build_summaries(items)

    item_by_path = {item["path"]: item for item in items if item.get("path")}
    summary_items = []
    for entry in summaries:
        original_item = item_by_path.get(entry.path)
        if original_item is None:
            continue
        summary_items.append(
            {
                "title": original_item.get("title", ""),
                "docid": original_item.get("docid", ""),
                "url": original_item.get("url", ""),
                "uri": original_item.get("uri", ""),
                "path": original_item.get("path", ""),
                "summary": entry.summary,
            }
        )

    stats = aggregate_llm_stats(layer.llm_call_stats)
    stats["summary_build_wall_clock_secs"] = time.perf_counter() - start_time
    stats["summary_items_out"] = len(summary_items)
    return summary_items, stats


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
            "vector-paraphrase-union",
            "vector-or-topk-union-only",
            "vector-or-topk-union-filter-query",
            "vector-or-topk-union-filter-union",
            "vector-or-topk-query-filter-union",
            "summary-filter-only",
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
        help="Number of items to retrieve with SemTopK (default: 50)",
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
    if args.mode in {"summary-filter-only", "chroma-summary-filter"}:
        print(f"Loading cached summaries with model: {args.summary_model}...")
        summary_items, summary_prep_stats = build_summary_items(
            items=corpus_items,
            api_key=api_key,
            summary_model_id=args.summary_model,
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
            results.append({
                "query": query['query'],
                "error": str(e),
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
        "topk": args.topk if args.mode not in {"no-index", "summary-filter-only"} else None,
        "paraphrase_model": args.paraphrase_model if args.mode == "vector-paraphrase-union" else None,
        "num_query_variations": args.num_query_variations if args.mode == "vector-paraphrase-union" else None,
        "decomposition_model": args.decomposition_model if args.mode in {
            "vector-or-topk-union-only",
            "vector-or-topk-union-filter-query",
            "vector-or-topk-union-filter-union",
            "vector-or-topk-query-filter-union",
        } else None,
        "summary_model": args.summary_model if args.mode in {"summary-filter-only", "chroma-summary-filter"} else None,
        "summary_prep_stats": summary_prep_stats if args.mode in {"summary-filter-only", "chroma-summary-filter"} else None,
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
