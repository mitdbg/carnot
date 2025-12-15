import json
import os
import csv
from typing import Dict, Iterable, List, Any
import numpy as np

GOLD_FILE_PATH = "../../train_subset.jsonl"
PRED_FILE_PATH = "pred_query_10_varying_k_modified.jsonl" 
OUTPUT_CSV_PATH = "query_10_varying_k_results_modified.csv" 

QUERY_INDEX = 10

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Reads a JSONL file line by line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line in {path}")
                    continue

def calculate_recall(gold_docs: set, predicted_docs: List[str]) -> float:
    """Calculates Recall @ |Pred|."""
    predicted_docs_set = set(predicted_docs)
    
    gold_size = len(gold_docs)
    
    if gold_size == 0:
        return 1.0
    
    covered_docs = gold_docs.intersection(predicted_docs_set)
    recall = len(covered_docs) / gold_size
    return recall

def main():
    if not os.path.isfile(GOLD_FILE_PATH):
        print(f"Error: Gold file not found at '{GOLD_FILE_PATH}'")
        return
    if not os.path.isfile(PRED_FILE_PATH):
        print(f"Error: Prediction file not found at '{PRED_FILE_PATH}'")
        return

    # 1. Load Gold Data
    print(f"Loading gold examples from: {GOLD_FILE_PATH}")
    gold_examples = list(read_jsonl(GOLD_FILE_PATH))
    
    # 2. Extract the gold set for Query 10 (index 9)
    try:
        gold_example_10 = gold_examples[QUERY_INDEX]
        gold_docs_set = set(gold_example_10.get("docs", []))
        query_text = gold_example_10["query"]
    except IndexError:
        print(f"Error: Gold file does not contain query at index {QUERY_INDEX}.")
        return
    
    if not gold_docs_set:
        print(f"Warning: Gold set for Query 10 ('{query_text}') is empty. Evaluation may be trivial.")
        
    # 3. Load Prediction Data (all k1, k2 variations for Query 10)
    print(f"Loading predicted examples from: {PRED_FILE_PATH}")
    pred_examples = list(read_jsonl(PRED_FILE_PATH))

    # 4. Prepare for CSV output
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f_csv:
        csv_writer = csv.writer(f_csv)
        
        csv_writer.writerow(["query", "k1", "k2", "recall_docs_1", "recall_docs_2", "recall_final"])

        total_recall_1 = 0.0
        total_recall_2 = 0.0
        total_recall_final = 0.0
        processed_count = 0
        
        print(f"\n--- Calculating Recall for {len(pred_examples)} (k1, k2) variations ---")
        
        for pred_example in pred_examples:
            k1 = pred_example.get("k1")
            k2 = pred_example.get("k2")
            
            predicted_docs_1 = pred_example.get("docs_1", [])
            predicted_docs_2 = pred_example.get("docs_2", [])
            final_docs = pred_example.get("docs", []) # This is the final merged list
            
            if k1 is None or k2 is None:
                print(f"Warning: Skipping entry missing k1 or k2: {pred_example}")
                continue

            recall_1 = calculate_recall(gold_docs_set, predicted_docs_1)
            recall_2 = calculate_recall(gold_docs_set, predicted_docs_2)
            recall_final = calculate_recall(gold_docs_set, final_docs)
            
            csv_writer.writerow([
                query_text, 
                k1, 
                k2, 
                f"{recall_1:.6f}", 
                f"{recall_2:.6f}", 
                f"{recall_final:.6f}"
            ])
            
            total_recall_1 += recall_1
            total_recall_2 += recall_2
            total_recall_final += recall_final
            processed_count += 1
            
        print("\n--- Summary ---")
        print(f"Total combinations evaluated: {processed_count}")

        if processed_count > 0:
            avg_recall_1 = total_recall_1 / processed_count
            avg_recall_2 = total_recall_2 / processed_count
            avg_recall_final = total_recall_final / processed_count
            print(f"Average Recall (docs_1): {avg_recall_1:.4f}")
            print(f"Average Recall (docs_2): {avg_recall_2:.4f}")
            print(f"Average Recall (final):  {avg_recall_final:.4f}")
        else:
            print("No valid prediction entries were processed.")

    print(f"\nResults successfully written to CSV file: '{OUTPUT_CSV_PATH}'")

if __name__ == "__main__":
    main()