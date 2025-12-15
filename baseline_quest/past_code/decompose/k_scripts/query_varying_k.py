import json
import sys
import os
from tqdm import tqdm
from itertools import product

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from decompose.retrieve import initialize_retriever, retrieve

query_num = 10
from query_10 import execute_query

GOLD_FILE_PATH = "../../data/train_subset.jsonl"
OUTPUT_PATH = f"pred_query_{query_num}_varying_k.jsonl"

def main():
    initialize_retriever()

    try:
        with open(GOLD_FILE_PATH, "r", encoding="utf-8") as f:
            queries = [json.loads(line) for line in f if line.strip()]
        QUERY_TEXT = queries[query_num-1]["query"]
        print(f"Loaded Query {query_num} text: '{QUERY_TEXT}'")
        
    except FileNotFoundError:
        print(f"Error: Gold file not found at '{GOLD_FILE_PATH}'. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading query from gold file: {e}")
        return

    k_values = list(range(100, 1001, 100)) # 100, 200, ..., 1000 (10 values)
    k_combinations = list(product(k_values, k_values))
    
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    print(f"--- Starting execution of Query {query_num} for {len(k_combinations)} combinations ---")
    
    for k1, k2 in tqdm(k_combinations, desc=f"Executing Query {query_num} Strategy"):
        
        docs_1, docs_2, final_docs = execute_query(retrieve, k1, k2)
        
        prediction = {
            "query": QUERY_TEXT, 
            "k1": k1,
            "docs_1": docs_1,
            "k2": k2, 
            "docs_2": docs_2,
            "docs": final_docs
        }
        
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(prediction) + "\n")
    
    print(f"\nResults saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
