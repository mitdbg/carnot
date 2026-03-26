#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

import carnot
from carnot.index.models import HierarchicalIndexConfig
from carnot.index.summary_layer import SummaryLayer
from carnot.operators.sem_filter import SemFilterOperator
from carnot.operators.sem_topk import SemTopKOperator

load_dotenv()


DATASET_ID = 3


def add_paths_to_items(items: list[dict]) -> list[dict]:
    """Add uri/path fields so cached indices match build_indices.py."""
    items_with_paths = []
    for i, item in enumerate(items):
        item_copy = dict(item)
        fake_path = f"/tmp/browsercomp_plus_docs/{i}_{item.get('docid', 'doc')}.txt"
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


def run_with_index(
    query: str,
    corpus_items: list[dict],
    index_name: str,
    model_id: str,
    api_key: str,
    topk: int = 50,
    embedding_model_id: str = "openai/text-embedding-3-small",
) -> tuple[list[str], dict]:
    """Run SemTopK + SemFilter pipeline."""
    start_time = time.perf_counter()

    dataset = carnot.Dataset(
        name="Documents",
        annotation="A set of BrowserComp+ documents with docids, URLs, and text.",
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

    topk_docids = [
        item.get("docid", "") for item in topk_datasets["topk_output"].items[:10]
    ]
    print(f"  TopK sample: {topk_docids}")

    filter_op = SemFilterOperator(
        task=query,
        output_dataset_id="final_output",
        model_id=model_id,
        llm_config=llm_config,
        max_workers=64,
    )
    final_datasets, filter_stats = filter_op("topk_output", topk_datasets)

    end_time = time.perf_counter()

    results = final_datasets["final_output"].items
    docids = [item.get("docid", "") for item in results]

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

    return docids, stats


def run_only_topk(
    query: str,
    corpus_items: list[dict],
    index_name: str,
    api_key: str,
    topk: int = 50,
    embedding_model_id: str = "openai/text-embedding-3-small",
) -> tuple[list[str], dict]:
    """Run SemTopK only (no SemFilter)."""
    start_time = time.perf_counter()

    dataset = carnot.Dataset(
        name="Documents",
        annotation="A set of BrowserComp+ documents with docids, URLs, and text.",
        items=corpus_items,
        dataset_id=3,
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

    end_time = time.perf_counter()

    results = topk_datasets["topk_output"].items
    docids = [item.get("docid", "") for item in results]

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

    return docids, stats


def run_no_index(
    query: str,
    corpus_items: list[dict],
    model_id: str,
    api_key: str,
) -> tuple[list[str], dict]:
    """Run SemFilter only (no index)."""
    start_time = time.perf_counter()

    dataset = carnot.Dataset(
        name="Documents",
        annotation="A set of BrowserComp+ documents with docids, URLs, and text.",
        items=corpus_items,
        dataset_id=DATASET_ID,
    )
    input_datasets = {"Documents": dataset}
    llm_config = {"OPENAI_API_KEY": api_key}

    filter_op = SemFilterOperator(
        task=query,
        output_dataset_id="final_output",
        model_id=model_id,
        llm_config=llm_config,
        max_workers=64,
    )
    final_datasets, filter_stats = filter_op("Documents", input_datasets)

    end_time = time.perf_counter()

    results = final_datasets["final_output"].items
    docids = [item.get("docid", "") for item in results]

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

    return docids, stats


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
        annotation="A set of BrowserComp+ document summaries with docids and URLs.",
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
    docids = [item.get("docid", "") for item in results]

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

    return docids, stats


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
    """Run SemTopK over raw docs, then SemFilter over cached summaries of top-k docs."""
    start_time = time.perf_counter()

    dataset = carnot.Dataset(
        name="Documents",
        annotation="A set of BrowserComp+ documents with docids, URLs, and text.",
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
    topk_docids = [item.get("docid", "") for item in topk_items[:10]]
    print(f"  TopK sample: {topk_docids}")

    topk_summary_items = []
    for item in topk_items:
        path = item.get("path")
        if path and path in summary_items_by_path:
            topk_summary_items.append(summary_items_by_path[path])

    summary_dataset = carnot.Dataset(
        name="SummaryDocuments",
        annotation="Summaries of top-k BrowserComp+ documents.",
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
    docids = [item.get("docid", "") for item in results]

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

    return docids, stats


def compute_metrics(predicted: list[str], ground_truth: list[str]) -> dict:
    """Compute precision, recall, and F1 score."""
    preds = set(predicted)
    labels = set(ground_truth)

    tp = len(preds & labels)
    fp = len(preds - labels)
    fn = len(labels - preds)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Carnot on BrowserComp+ with different index configurations"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "chroma",
            "flat",
            "hierarchical",
            "vector-only",
            "summary-filter-only",
            "chroma-summary-filter",
            "no-index",
        ],
        help="Index mode to use",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default="data/corpus_first25/queries.jsonl",
        help="Path to queries JSONL file",
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

    print(f"Loading queries from {args.queries}...")
    queries = []
    with open(args.queries) as f:
        for line in f:
            queries.append(json.loads(line))

    if args.num_queries is not None:
        queries = queries[:args.num_queries]

    print(f"Running {len(queries)} queries in mode: {args.mode}")
    print(f"Execution model: {args.model}")
    print("=" * 60)

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
                pred_docids, exec_stats = run_no_index(
                    query=query["query"],
                    corpus_items=corpus_items,
                    model_id=args.model,
                    api_key=api_key,
                )
            elif args.mode == "vector-only":
                pred_docids, exec_stats = run_only_topk(
                    query=query["query"],
                    corpus_items=corpus_items,
                    index_name="chroma",
                    api_key=api_key,
                    topk=args.topk,
                    embedding_model_id=args.embedding_model,
                )
            elif args.mode == "summary-filter-only":
                pred_docids, exec_stats = run_summary_filter_only(
                    query=query["query"],
                    summary_items=summary_items or [],
                    model_id=args.model,
                    api_key=api_key,
                )
            elif args.mode == "chroma-summary-filter":
                pred_docids, exec_stats = run_with_index_summary_filter(
                    query=query["query"],
                    corpus_items=corpus_items,
                    summary_items_by_path=summary_items_by_path or {},
                    index_name="chroma",
                    model_id=args.model,
                    api_key=api_key,
                    topk=args.topk,
                    embedding_model_id=args.embedding_model,
                )
            else:
                pred_docids, exec_stats = run_with_index(
                    query=query["query"],
                    corpus_items=corpus_items,
                    index_name=args.mode,
                    model_id=args.model,
                    api_key=api_key,
                    topk=args.topk,
                    embedding_model_id=args.embedding_model,
                )

            metrics = compute_metrics(pred_docids, query["docs"])

            total_stats["total_cost_usd"] += exec_stats["total_cost_usd"]
            total_stats["total_input_tokens"] += exec_stats["total_input_tokens"]
            total_stats["total_output_tokens"] += exec_stats["total_output_tokens"]
            total_stats["total_wall_clock_secs"] += exec_stats["total_wall_clock_secs"]

            print(
                f"  P: {metrics['precision']:.3f}, "
                f"R: {metrics['recall']:.3f}, "
                f"F1: {metrics['f1']:.3f}"
            )
            print(
                f"  Cost: ${exec_stats['total_cost_usd']:.8f}, "
                f"Time: {exec_stats['total_wall_clock_secs']:.2f}s"
            )
            print(f"  Predicted: {len(pred_docids)}, GT: {len(query['docs'])}")
            print(
                f"  TopK out: {exec_stats.get('topk_items_out', 'n/a')}, "
                f"Filter out: {exec_stats.get('filter_items_out', 'n/a')}"
            )

            results.append(
                {
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "predicted_docs": pred_docids,
                    "ground_truth_docs": query["docs"],
                    "metrics": metrics,
                    "execution_stats": exec_stats,
                }
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(
                {
                    "query_id": query.get("query_id"),
                    "query": query["query"],
                    "error": str(e),
                }
            )

    successful_results = [r for r in results if "metrics" in r]
    n = len(successful_results)

    if n > 0:
        avg_precision = sum(r["metrics"]["precision"] for r in successful_results) / n
        avg_recall = sum(r["metrics"]["recall"] for r in successful_results) / n
        avg_f1 = sum(r["metrics"]["f1"] for r in successful_results) / n
        avg_latency = total_stats["total_wall_clock_secs"] / n
        avg_cost = total_stats["total_cost_usd"] / n
    else:
        avg_precision = avg_recall = avg_f1 = avg_latency = avg_cost = 0.0

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

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    output_path = f"{args.output_dir}/eval_{args.mode}_{ts}.json"

    output_data = {
        "mode": args.mode,
        "model": args.model,
        "embedding_model": args.embedding_model,
        "topk": args.topk if args.mode not in {"no-index", "summary-filter-only"} else None,
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
