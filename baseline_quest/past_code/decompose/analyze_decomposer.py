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
A script to calculate Recall @ |Pred| for the QUEST dataset, where |Pred| is 
the arbitrary size of the predicted document list (no truncation).
It prints the recall and list size for each individual query with the full 
query text, and saves the output to a human-readable text file in a 'results' 
subdirectory.
"""

import json
from typing import Dict, Iterable, List
import numpy as np
import os

# --- Hardcoded File Paths ---
GOLD_FILE_PATH = "../../../data/train_subset.jsonl"
PRED_FILE_PATH = "pred_retrieved_set_ops_limited.jsonl" 
OUTPUT_REPORT_PATH = "recall_report_decomposer_limited.txt" 


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

    # Ensure the output directory exists before trying to write to it.
    output_dir = os.path.dirname(OUTPUT_REPORT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Open the output report file for writing
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
        
        # MODIFIED: Store all recall and list size values.
        recall_values: List[float] = []
        pred_list_sizes: List[int] = []

        log_and_print(f"\n--- Per-Query Recall @ |Pred| (Arbitrary List Size) ---")

        for gold_example in gold_examples:
            query = gold_example["query"]
            log_line = ""

            if query not in query_to_pred_example:
                log_line = f"  - Query NOT FOUND in predictions: \"{query}\""
            elif not gold_example.get("docs"):
                log_line = f"  - WARNING: Gold example has no docs. Skipping query: \"{query}\""
            else:
                pred_example = query_to_pred_example[query]
                # Use the full list of predicted documents
                all_predicted_docs = pred_example.get("docs", [])
                
                gold_docs = set(gold_example.get("docs", []))
                
                # The prediction list size is the arbitrary K for this query
                list_size = len(all_predicted_docs)
                predicted_docs_set = set(all_predicted_docs)
                
                covered_docs = gold_docs.intersection(predicted_docs_set)
                
                # Calculate recall
                if len(gold_docs) == 0:
                    # R = 1.0 if the gold set is empty, unless predictions exist (shouldn't happen here)
                    recall = 1.0 if list_size == 0 else 0.0
                else:
                    recall = len(covered_docs) / len(gold_docs)
                
                recall_values.append(recall)
                pred_list_sizes.append(list_size)

                # Create a single log line summarizing results for the query.
                # Show covered docs, gold set size, recall, and the predicted list size.
                log_line = (
                    f"  - Recall @{list_size}: {recall:.4f} "
                    f"({len(covered_docs)}/{len(gold_docs)}) | Query: \"{query}\""
                )
            
            log_and_print(log_line)

        # Check if any queries were processed.
        if not recall_values:
            log_and_print("\nNo valid queries were processed. Cannot calculate average recall.")
            return

        log_and_print("\n--- Summary ---")
        num_evaluated = len(recall_values)
        log_and_print(f"Total queries evaluated: {num_evaluated}")

        # Calculate and print the final average recall and list size.
        average_recall = np.mean(recall_values)
        average_list_size = np.mean(pred_list_sizes)
        
        log_and_print(f"Average Predicted List Size (|Pred|): {average_list_size:.2f}")
        log_and_print(f"Average Recall @ |Pred|: {average_recall:.4f}")

    print(f"\nReport successfully written to '{OUTPUT_REPORT_PATH}'")

if __name__ == "__main__":
    main()