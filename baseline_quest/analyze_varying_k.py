import os
import csv
import argparse
from typing import List, Set

from lib.chroma_utils import read_jsonl

def calculate_recall(gold_docs: Set[str], predicted_docs: List[str]) -> float:
    """
    Calculates Recall: |Intersection| / |Gold|
    Consistent with logic in lib.analyze_retrieval_results.
    """
    if not gold_docs:
        return 1.0  # Trivial success if gold set is empty
    
    pred_set = set(predicted_docs)
    intersection = gold_docs.intersection(pred_set)
    return len(intersection) / len(gold_docs)

def run_analysis(gold_path: str, pred_path: str, output_path: str, query_index: int):
    """
    Main logic: Loads data using project utils, calculates stats, and writes CSV.
    """
    # 2. Load Gold Data (using chroma_utils)
    gold_examples = list(read_jsonl(gold_path))
    
    # 3. Extract the target Query
    target_gold = gold_examples[query_index]
    gold_docs_set = set(target_gold.get("docs", []))
    query_text = target_gold.get("query", "Unknown Query")
    
    pred_examples = list(read_jsonl(pred_path))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f_csv:
        csv_writer = csv.writer(f_csv)
        # CSV Header
        csv_writer.writerow(["query", "k1", "k2", "recall_docs_1", "recall_docs_2", "recall_final"])

        stats = {
            "total_recall_1": 0.0,
            "total_recall_2": 0.0,
            "total_recall_final": 0.0,
            "count": 0
        }
            
        for pred in pred_examples:
            # Extract parameters
            k1 = pred.get("k1")
            k2 = pred.get("k2")
                
            if k1 is None or k2 is None:
                continue

            # Extract doc lists
            # Note: Handling both 'docs' (final) and intermediate steps if available
            docs_1 = pred.get("docs_1", [])
            docs_2 = pred.get("docs_2", [])
            docs_final = pred.get("docs", []) 

            # Calculate Recall
            r1 = calculate_recall(gold_docs_set, docs_1)
            r2 = calculate_recall(gold_docs_set, docs_2)
            r_final = calculate_recall(gold_docs_set, docs_final)
                
            # Write row
            csv_writer.writerow([
                query_text, 
                k1, 
                k2, 
                f"{r1:.6f}", 
                f"{r2:.6f}", 
                f"{r_final:.6f}"
            ])
                
            # Accumulate stats
            stats["total_recall_1"] += r1
            stats["total_recall_2"] += r2
            stats["total_recall_final"] += r_final
            stats["count"] += 1

def main():
    parser = argparse.ArgumentParser(description="Analyze Recall for varying k parameters.")
    
    # Default args match your file structure
    parser.add_argument("--gold", type=str, default="../../train_subset.jsonl", 
                        help="Path to gold standard JSONL.")
    parser.add_argument("--pred", type=str, default="pred_query_10_varying_k_modified.jsonl", 
                        help="Path to predictions JSONL.")
    parser.add_argument("--out", type=str, default="query_10_varying_k_results_modified.csv", 
                        help="Output CSV path.")
    parser.add_argument("--index", type=int, default=10, 
                        help="Index of the query in the gold file (0-based). Default 10.")

    args = parser.parse_args()

    run_analysis(args.gold, args.pred, args.out, args.index)

if __name__ == "__main__":
    main()