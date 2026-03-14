import argparse
import json
import os
import time

import carnot


def add_paths_to_items(items: list[dict]) -> list[dict]:
    """Add uri and path to items for index cache consistency with build_indices.py."""
    items_with_paths = []
    for i, item in enumerate(items):
        item_copy = dict(item)
        fake_path = f"/tmp/quest_docs/{i}_{item.get('title', 'doc')[:50]}.txt"
        item_copy["uri"] = fake_path  # needed by _build_uri_to_idx
        item_copy["path"] = fake_path  # needed by SummaryLayer
        items_with_paths.append(item_copy)
    return items_with_paths


# Dataset ID must match build_indices.py for index cache to be hit
DATASET_ID = 1


def carnot_run_query(query: dict, corpus_items: list[dict]) -> tuple[list[str], dict]:
    """Run a query using the pre-loaded corpus.
    
    Args:
        query: Query dict with 'query' and 'docs' fields
        corpus_items: Pre-loaded corpus items (with paths added)
    
    Returns:
        Tuple of (predicted_titles, execution_stats_dict)
    """
    dataset = carnot.Dataset(
        name="Documents",
        annotation="A set of documents with their titles and content.",
        items=corpus_items,
        dataset_id=DATASET_ID,
    )
    execution = carnot.Execution(
        query=f"Query: {query['query']}\n\nReturn the (list of) document title(s) under the column `title`.",
        datasets=[dataset],
        llm_config={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
        execution_model="openai/gpt-5-2025-08-07",
    )
    _, plan = execution.plan()
    execution._plan = plan
    output, _, stats = execution.run()

    stats_dict = {
        "total_cost_usd": stats.total_cost_usd,
        "total_input_tokens": stats.total_input_tokens,
        "total_output_tokens": stats.total_output_tokens,
        "total_wall_clock_secs": stats.total_wall_clock_secs,
        "planning_cost_usd": stats.planning.total_cost_usd,
        "execution_cost_usd": stats.execution.total_cost_usd,
    }
    
    return [out['title'] for out in output], stats_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QUEST")
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="The domain to evalute (one of 'films' or 'books').",
    )
    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Path to the file containing the queries (one query per line).",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="data/corpus_first25",
        help="Path to the corpus directory (must match build_indices.py)",
    )
    args = parser.parse_args()

    # extract args
    domain = args.domain
    queries_path = args.queries
    corpus_dir = args.corpus_dir

    # Load corpus once (with paths for index cache consistency)
    corpus_path = f"{corpus_dir}/corpus.jsonl"
    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path) as f:
        raw_items = [json.loads(line) for line in f]
    corpus_items = add_paths_to_items(raw_items)
    print(f"Loaded {len(corpus_items)} items")

    # load queries
    queries = []
    with open(queries_path) as f:
        for line in f:
            d = json.loads(line)
            if d['metadata']['domain'] == domain:
                queries.append(d)

    results = []
    total_stats = {
        "total_cost_usd": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_wall_clock_secs": 0.0,
    }
    
    for i, query in enumerate(queries):
        # execute the query
        print(f"\n[{i+1}/{len(queries)}] Executing query: {query['query'][:80]}...")
        pred_docs, exec_stats = carnot_run_query(query, corpus_items)
        
        # Accumulate stats
        total_stats["total_cost_usd"] += exec_stats["total_cost_usd"]
        total_stats["total_input_tokens"] += exec_stats["total_input_tokens"]
        total_stats["total_output_tokens"] += exec_stats["total_output_tokens"]
        total_stats["total_wall_clock_secs"] += exec_stats["total_wall_clock_secs"]
        
        print(f"  Cost: ${exec_stats['total_cost_usd']:.6f}, Time: {exec_stats['total_wall_clock_secs']:.2f}s")
        
        # compute the precision, recall, and F1 score
        gt_docs = query['docs']
        preds = set(pred_docs)
        labels = set(gt_docs)

        tp, fp, fn = 0, 0, 0
        for pred in preds:
            if pred in labels:
                tp += 1
            else:
                fp += 1
        
        for label in labels:
            if label not in preds:
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")

        result = {
            "query": query['query'],
            "predicted_docs": pred_docs,
            "ground_truth_docs": gt_docs,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "execution_stats": exec_stats,
        }
        results.append(result)

        # TODO: remove this break statement after testing
        break

    # save results to a file
    ts = int(time.time())
    output_data = {
        "domain": domain,
        "num_queries": len(results),
        "total_stats": total_stats,
        "results": results,
    }
    output_path = f"results_{domain}_{ts}.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"\nResults saved to {output_path}")

    # compute and print summary
    avg_precision = sum(result['precision'] for result in results) / len(results)
    avg_recall = sum(result['recall'] for result in results) / len(results)
    avg_f1 = sum(result['f1_score'] for result in results) / len(results)
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Queries executed: {len(results)}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print("-" * 60)
    print(f"Total Cost: ${total_stats['total_cost_usd']:.6f}")
    print(f"Total Time: {total_stats['total_wall_clock_secs']:.2f}s")
    print(f"Total Input Tokens: {total_stats['total_input_tokens']:,}")
    print(f"Total Output Tokens: {total_stats['total_output_tokens']:,}")
