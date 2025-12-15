# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A script to calculate Recall @ K for the QUEST dataset for multiple K values, 
printing the recall for each individual query with the full query text, and 
saving the output to a human-readable text file in a 'results' subdirectory.
"""

import json
from typing import Dict, Iterable, List
import numpy as np
import os

GOLD_FILE_PATH = "../data/train_subset2.jsonl"
PRED_FILE_PATH = "../semantic/pred_unranked_limited_subset2.jsonl"
OUTPUT_REPORT_PATH = "../semantic/results/recall_report_limited_subset2.txt"

K_VALUES = [20, 50, 100]

def read_jsonl(path: str) -> Iterable[Dict]:
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

def main():
    if not os.path.isfile(GOLD_FILE_PATH):
        print(f"Error: Gold file not found at '{GOLD_FILE_PATH}'")
        return
    if not os.path.isfile(PRED_FILE_PATH):
        print(f"Error: Prediction file not found at '{PRED_FILE_PATH}'")
        return

    output_dir = os.path.dirname(OUTPUT_REPORT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f_report:

        def log_and_print(message: str):
            """Helper function to write to file and print to console."""
            print(message)
            f_report.write(message + "\n")

        log_and_print(f"Loading gold examples from: {GOLD_FILE_PATH}")
        gold_examples = list(read_jsonl(GOLD_FILE_PATH))

        log_and_print(f"Loading predicted examples from: {PRED_FILE_PATH}")
        pred_examples = list(read_jsonl(PRED_FILE_PATH))

        query_to_pred_example = {ex["query"]: ex for ex in pred_examples}
        
        # MODIFIED: Use a dictionary to store lists of recall values for each K.
        recall_values = {k: [] for k in K_VALUES}

        log_and_print(f"\n--- Per-Query Recall @ {', '.join(map(str, K_VALUES))} ---")

        for gold_example in gold_examples:
            query = gold_example["query"]
            log_line = ""

            if query not in query_to_pred_example:
                log_line = f"  - Query NOT FOUND in predictions: \"{query}\""
            elif not gold_example.get("docs"):
                log_line = f"  - WARNING: Gold example has no docs. Skipping query: \"{query}\""
            else:
                pred_example = query_to_pred_example[query]
                all_predicted_docs = pred_example.get("docs", [])
                gold_docs = set(gold_example.get("docs", []))
                
                recall_strings = []
                for k in K_VALUES:
                    predicted_docs_at_k = set(all_predicted_docs[:k])
                    covered_docs = gold_docs.intersection(predicted_docs_at_k)

                    if len(gold_docs) == 0:
                        recall = 1.0 if len(predicted_docs_at_k) == 0 else 0.0
                    else:
                        recall = len(covered_docs) / len(gold_docs)
                    
                    recall_values[k].append(recall)
                    recall_strings.append(f"@{k}={recall:.2f}")

                recall_summary = ", ".join(recall_strings)
                covered_at_max_k = len(gold_docs.intersection(set(all_predicted_docs[:max(K_VALUES)])))
                log_line = f"  - Recalls: {recall_summary} ({covered_at_max_k}/{len(gold_docs)}) | Query: \"{query}\""
            
            log_and_print(log_line)

        if not recall_values[K_VALUES[0]]:
            log_and_print("\nNo valid queries were processed. Cannot calculate average recall.")
            return

        log_and_print("\n--- Summary ---")
        num_evaluated = len(recall_values[K_VALUES[0]])
        log_and_print(f"Total queries evaluated: {num_evaluated}")

        for k in K_VALUES:
            average_recall = np.mean(recall_values[k])
            log_and_print(f"Average Recall @ {k}: {average_recall:.4f}")

    print(f"\nReport successfully written to '{OUTPUT_REPORT_PATH}'")

if __name__ == "__main__":
    main()
