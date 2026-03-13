#!/usr/bin/env python3
"""Evaluate Carnot with different index configurations.

Runs 4 query plan modes:
- chroma: SemTopK(chroma) + SemFilter
- flat: SemTopK(flat) + SemFilter  
- hierarchical: SemTopK(hierarchical) + SemFilter
- no-index: SemFilter only (no SemTopK)

Usage:
    python eval_indices.py --mode flat --queries data/quest_all.jsonl --domain films
    python eval_indices.py --mode no-index --num-queries 5 --model openai/gpt-4o
"""
import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import carnot
from carnot.operators.sem_topk import SemTopKOperator
from carnot.operators.sem_filter import SemFilterOperator


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
        choices=["chroma", "flat", "hierarchical", "no-index"],
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
        "--topk",
        type=int,
        default=50,
        help="Number of items to retrieve with SemTopK (default: 50)",
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
            print(f"  Cost: ${exec_stats['total_cost_usd']:.4f}, Time: {exec_stats['total_wall_clock_secs']:.2f}s")
            print(f"  Predicted: {len(pred_titles)}, GT: {len(query['docs'])}")
            
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
    print(f"Average Cost:      ${avg_cost:.4f}")
    print(f"Total Cost:        ${total_stats['total_cost_usd']:.4f}")
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
        "topk": args.topk if args.mode != "no-index" else None,
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
