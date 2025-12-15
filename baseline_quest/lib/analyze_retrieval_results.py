import os
import numpy as np
import logging
from typing import List, Dict, Union, Any

# Re-use the existing jsonl reader from your library
from chroma_utils import read_jsonl

def calculate_recall_stats(gold_path: str, pred_path: str) -> Dict[str, Any]:
    """
    Calculates Recall metrics for predictions against a gold standard based on the full predicted list.
    
    Args:
        gold_path: Path to the ground truth JSONL (must have "query" and "docs").
        pred_path: Path to the prediction JSONL (must have "query" and "docs").
                  
    Returns:
        A dictionary containing:
        - 'per_query': List of dicts with detailed stats per query.
        - 'summary': Dict of average scores (e.g., {"avg_recall": 0.75, "avg_pred_size": 12.5}).
        - 'missing_count': Number of queries in gold but missing in pred.
    """
    if not os.path.exists(gold_path):
        raise FileNotFoundError(f"Gold file not found: {gold_path}")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    # Load Data
    gold_data = list(read_jsonl(gold_path))
    pred_data = list(read_jsonl(pred_path))
    
    # Map predictions by query for O(1) lookup
    pred_map = {item['query']: item for item in pred_data}

    stats = {
        "per_query": [],
        "summary": {},
        "missing_count": 0
    }
    
    recall_scores = []
    pred_sizes = []
    
    for gold in gold_data:
        query = gold['query']
        gold_docs = set(gold.get('docs', []))
        
        if not gold_docs:
            stats["per_query"].append({"query": query, "error": "Gold example has no docs"})
            continue

        if query not in pred_map:
            stats["missing_count"] += 1
            recall_scores.append(0.0)
            pred_sizes.append(0)
            stats["per_query"].append({"query": query, "error": "Missing in predictions"})
            continue

        raw_pred = pred_map[query].get('docs', [])
        pred_docs = []
        for d in raw_pred:
            if isinstance(d, dict):
                pred_docs.append(d.get('title', ''))
            else:
                pred_docs.append(d)

        pred_set = set(pred_docs)
        intersection = gold_docs.intersection(pred_set)
        
        recall = len(intersection) / len(gold_docs)
        
        query_res = {
            "query": query, 
            "gold_count": len(gold_docs),
            "recall": recall,
            "pred_size": len(pred_docs),
            "covered": len(intersection)
        }

        recall_scores.append(recall)
        pred_sizes.append(len(pred_docs))
        stats["per_query"].append(query_res)

    if recall_scores:
        stats["summary"]["avg_recall"] = np.mean(recall_scores)
        stats["summary"]["avg_pred_size"] = np.mean(pred_sizes)
    else:
        stats["summary"]["avg_recall"] = 0.0
        stats["summary"]["avg_pred_size"] = 0.0

    return stats

def write_analysis_report(stats: Dict, output_path: str):
    """Writes the calculated stats to a human-readable text file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        # 1. Summary Section
        f.write("--- Analysis Report ---\n\n")
        f.write("Summary Metrics:\n")
        for key, val in stats['summary'].items():
            f.write(f"  {key}: {val:.4f}\n")
        f.write(f"  Missing Queries: {stats['missing_count']}\n\n")
        
        # 2. Detailed Per-Query Section
        f.write("Detailed Results:\n")
        for item in stats['per_query']:
            q = item.get('query', '')
            
            if 'error' in item:
                f.write(f"  [ERROR] {item['error']} | Query: \"{q}\"\n")
            else:
                recall = item.get('recall', 0.0)
                size = item.get('pred_size', 0)
                covered = item.get('covered', 0)
                gold_count = item.get('gold_count', 0)
                
                # Format: Recall=0.80 (Size: 50) (4/5) | Query: "..."
                f.write(f"  - Recall={recall:.2f} (Size: {size}) ({covered}/{gold_count}) | Query: \"{q}\"\n")
                
    print(f"Report written to: {output_path}")