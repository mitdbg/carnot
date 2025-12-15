import importlib.util
import json
import os
from tqdm import tqdm

from retrieve import initialize_retriever, retrieve

GOLD_FILE_PATH = "../../data/train_subset.jsonl"
OUTPUT_PATH = "pred_retrieved_set_ops.jsonl"
DECOMPOSITION_DIR = os.path.join(os.path.dirname(__file__), "k_50")

def load_strategy_functions(strategy_dir):
    """Import all query_*.py files in the given folder."""
    strategy_funcs = []

    for filename in sorted(os.listdir(strategy_dir)):
        if filename.startswith("query_") and filename.endswith(".py"):
            file_path = os.path.join(strategy_dir, filename)
            module_name = f"{os.path.basename(strategy_dir)}_{filename[:-3]}"

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "execute_query"):
                strategy_funcs.append(module.execute_query)
            else:
                print(f"WARNING: {filename} has no execute_query(retrieve)")
    
    return strategy_funcs

def main():
    print(f"Loading strategies from: {DECOMPOSITION_DIR}")

    # Init retriever once
    initialize_retriever()

    # Load gold queries
    with open(GOLD_FILE_PATH, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    # Output file
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    # Load strategy functions from k_50 folder
    strategy_funcs = load_strategy_functions(DECOMPOSITION_DIR)
    num_to_process = len(strategy_funcs)

    print(f"Found {num_to_process} strategy files.")

    # Run each query
    for i in tqdm(range(num_to_process), desc="Executing strategies"):
        strategy_func = strategy_funcs[i]
        query_text = queries[i]["query"]

        final_docs = strategy_func(retrieve)

        prediction = {
            "query": query_text,
            "docs": final_docs,
        }

        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(prediction) + "\n")

    print("\n--- All strategies executed. ---")
    print(f"Results saved in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
