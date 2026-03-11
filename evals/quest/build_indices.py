#!/usr/bin/env python3
"""Build and benchmark indices for QUEST evaluation.

This script builds each index type on the corpus and tracks:
- Wall clock time
- Actual cost (from LLM API responses)
- Actual token counts

Usage:
    python build_indices.py --corpus-dir data/corpus_first25 --output-dir indices

The indices are persisted automatically by the index classes.
Results are saved to indices/build_stats.json.
"""
import argparse
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def load_corpus(corpus_path: str) -> list[dict]:
    """Load corpus from JSONL file."""
    items = []
    with open(corpus_path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def collect_stats_from_index(index) -> dict:
    """Collect actual LLM call stats from an index object.
    
    Returns a dict with total tokens and costs from actual API calls.
    """
    stats = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_embedding_tokens": 0,
        "total_cost_usd": 0.0,
        "num_llm_calls": 0,
        "num_embedding_calls": 0,
    }
    
    llm_call_stats = getattr(index, "_llm_call_stats", [])
    for call in llm_call_stats:
        stats["total_cost_usd"] += call.cost_usd
        if call.call_type == "embedding":
            stats["total_embedding_tokens"] += call.embedding_input_tokens
            stats["num_embedding_calls"] += 1
        else:
            stats["total_input_tokens"] += call.input_text_tokens
            stats["total_output_tokens"] += call.output_text_tokens
            stats["num_llm_calls"] += 1
    
    return stats


def build_chroma_index(
    items: list[dict],
    name: str,
    api_key: str,
) -> tuple[float, dict]:
    """Build ChromaIndex and return (time, stats)."""
    from carnot.index.index import ChromaIndex
    
    '''
    # Clear existing collection if it exists (force fresh build)
    home_dir = Path.home() / ".carnot"
    chroma_dir = home_dir / "chroma"
    if chroma_dir.exists():
        import chromadb
        client = chromadb.PersistentClient(str(chroma_dir))
        try:
            client.delete_collection(name)
            print(f"  Deleted existing collection '{name}'")
        except Exception:
            pass
    '''
    start = time.perf_counter()
    index = ChromaIndex(name=name, items=items, api_key=api_key)
    elapsed = time.perf_counter() - start
    
    # Collect actual stats from the index
    actual_stats = collect_stats_from_index(index)
    
    stats = {
        "index_type": "ChromaIndex",
        "num_items": len(items),
        "build_time_seconds": elapsed,
        "total_embedding_tokens": actual_stats["total_embedding_tokens"],
        "num_embedding_calls": actual_stats["num_embedding_calls"],
        "total_cost_usd": actual_stats["total_cost_usd"],
    }
    
    return elapsed, stats


def build_flat_index(
    items: list[dict],
    name: str,
    api_key: str,
    summary_model: str = "openai/gpt-5-mini-2025-08-07",
) -> tuple[float, dict]:
    """Build FlatCarnotIndex and return (time, stats).
    
    Note: FlatCarnotIndex requires items with 'path' field.
    For QUEST docs without paths, we create temporary paths.
    """
    from carnot.index.index import FlatCarnotIndex
    from carnot.index.models import HierarchicalIndexConfig
    
    # Add uri and path to items (uri for index mapping, path for summary caching)
    items_with_paths = []
    for i, item in enumerate(items):
        item_copy = dict(item)
        fake_path = f"/tmp/quest_docs/{i}_{item.get('title', 'doc')[:50]}.txt"
        item_copy["uri"] = fake_path  # needed by _build_uri_to_idx
        item_copy["path"] = fake_path  # needed by SummaryLayer
        items_with_paths.append(item_copy)
    
    # Configure with specified summary model
    config = HierarchicalIndexConfig(
        summary_model=summary_model
    )
    
    start = time.perf_counter()
    try:
        index = FlatCarnotIndex(
            name=name,
            items=items_with_paths,
            config=config,
            api_key=api_key,
            use_persistence=True,
        )
        elapsed = time.perf_counter() - start
        
        # Collect actual stats from the index
        actual_stats = collect_stats_from_index(index)
        
        stats = {
            "index_type": "FlatCarnotIndex",
            "num_items": len(items),
            "summary_model": summary_model,
            "build_time_seconds": elapsed,
            "total_input_tokens": actual_stats["total_input_tokens"],
            "total_output_tokens": actual_stats["total_output_tokens"],
            "total_embedding_tokens": actual_stats["total_embedding_tokens"],
            "num_llm_calls": actual_stats["num_llm_calls"],
            "num_embedding_calls": actual_stats["num_embedding_calls"],
            "total_cost_usd": actual_stats["total_cost_usd"],
        }
        
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  Warning: FlatCarnotIndex build failed: {e}")
        stats = {
            "index_type": "FlatCarnotIndex",
            "error": str(e),
            "build_time_seconds": elapsed,
            "total_cost_usd": 0.0,
        }
    
    return elapsed, stats


def build_hierarchical_index(
    items: list[dict],
    name: str,
    api_key: str,
    summary_model: str = "openai/gpt-5-mini-2025-08-07",
) -> tuple[float, dict]:
    """Build HierarchicalCarnotIndex and return (time, stats).
    
    Note: HierarchicalCarnotIndex requires items with 'path' field.
    """
    from carnot.index.index import HierarchicalCarnotIndex
    from carnot.index.models import HierarchicalIndexConfig
    
    # Add uri and path to items (uri for index mapping, path for summary caching)
    items_with_paths = []
    for i, item in enumerate(items):
        item_copy = dict(item)
        fake_path = f"/tmp/quest_docs/{i}_{item.get('title', 'doc')[:50]}.txt"
        item_copy["uri"] = fake_path  # needed by _build_uri_to_idx
        item_copy["path"] = fake_path  # needed by SummaryLayer
        items_with_paths.append(item_copy)
    
    # Configure with specified summary model
    config = HierarchicalIndexConfig(
        summary_model=summary_model,
        embedding_model="openai/text-embedding-3-small",
    )
    
    start = time.perf_counter()
    try:
        index = HierarchicalCarnotIndex(
            name=name,
            items=items_with_paths,
            config=config,
            api_key=api_key,
            use_persistence=True,
        )
        elapsed = time.perf_counter() - start
        
        # Collect actual stats from the index
        actual_stats = collect_stats_from_index(index)
        
        stats = {
            "index_type": "HierarchicalCarnotIndex",
            "num_items": len(items),
            "summary_model": summary_model,
            "build_time_seconds": elapsed,
            "total_input_tokens": actual_stats["total_input_tokens"],
            "total_output_tokens": actual_stats["total_output_tokens"],
            "total_embedding_tokens": actual_stats["total_embedding_tokens"],
            "num_llm_calls": actual_stats["num_llm_calls"],
            "num_embedding_calls": actual_stats["num_embedding_calls"],
            "total_cost_usd": actual_stats["total_cost_usd"],
        }
        
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  Warning: HierarchicalCarnotIndex build failed: {e}")
        stats = {
            "index_type": "HierarchicalCarnotIndex",
            "error": str(e),
            "build_time_seconds": elapsed,
            "total_cost_usd": 0.0,
        }
    
    return elapsed, stats


def main():
    parser = argparse.ArgumentParser(description="Build indices for QUEST evaluation")
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="data/corpus_first25",
        help="Directory containing corpus.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="indices",
        help="Output directory for stats",
    )
    parser.add_argument(
        "--index-types",
        type=str,
        nargs="+",
        default=["chroma", "flat", "hierarchical"],
        choices=["chroma", "flat", "hierarchical"],
        help="Index types to build",
    )
    parser.add_argument(
        "--summary-model",
        type=str,
        default="openai/gpt-5-mini-2025-08-07",
        help="Model for summarization (flat/hierarchical indices)",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=1,
        help="Dataset ID for index naming (must match eval.py). "
             "Index names will be 'ds{dataset_id}_{index_type}' to match SemTopKOperator.",
    )
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load corpus
    corpus_path = Path(args.corpus_dir) / "corpus.jsonl"
    print(f"Loading corpus from {corpus_path}...")
    items = load_corpus(corpus_path)
    print(f"Loaded {len(items)} items")
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    all_stats = {
        "corpus_dir": args.corpus_dir,
        "dataset_id": args.dataset_id,
        "num_items": len(items),
        "summary_model": args.summary_model,
        "indices": {},
    }
    
    # Build each index type
    # Use naming convention that matches SemTopKOperator: "ds{dataset_id}_{index_name}"
    for idx_type in args.index_types:
        name = f"ds{args.dataset_id}_{idx_type}"
        print(f"\nBuilding {idx_type} index (name='{name}')...")
        
        if idx_type == "chroma":
            elapsed, stats = build_chroma_index(items, name, api_key)
        elif idx_type == "flat":
            elapsed, stats = build_flat_index(items, name, api_key, args.summary_model)
        elif idx_type == "hierarchical":
            elapsed, stats = build_hierarchical_index(items, name, api_key, args.summary_model)
        else:
            print(f"  Unknown index type: {idx_type}")
            continue
        
        all_stats["indices"][idx_type] = stats
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Actual cost: ${stats.get('total_cost_usd', 0):.6f}")
    
    # Save stats
    stats_path = output_dir / "build_stats.json"
    print(f"\nSaving stats to {stats_path}...")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BUILD SUMMARY")
    print("=" * 60)
    total_time = 0
    total_cost = 0
    for idx_type, stats in all_stats["indices"].items():
        t = stats.get("build_time_seconds", 0)
        c = stats.get("total_cost_usd", 0)
        total_time += t
        total_cost += c
        print(f"{idx_type:20s}: {t:8.2f}s  ${c:.4f}")
    print("-" * 60)
    print(f"{'TOTAL':20s}: {total_time:8.2f}s  ${total_cost:.4f}")


if __name__ == "__main__":
    main()
