import importlib.util
import json
import os
import sys
from tqdm import tqdm
from functools import partial

from chroma_utils import get_db_collection
from retriever import retrieve

def load_strategy_functions(strategy_dir):
    """
    Import all query_*.py files in the given folder.
    Returns a dictionary mapping the query index (0-based) to the execute function.
    """
    strategy_funcs = {} 

    if not os.path.exists(strategy_dir):
        print(f"Strategy directory not found: {strategy_dir}")
        return {}

    # Get all python files starting with "query_"
    files = sorted([f for f in os.listdir(strategy_dir) if f.startswith("query_") and f.endswith(".py")])

    for filename in files:
        # Extract index from filename (e.g., "query_1.py" -> index 0)
        try:
            # Split "query_1.py" -> ["query", "1.py"] -> "1" -> int(1) -> 0
            idx = int(filename.split('_')[1].split('.')[0]) - 1 
        except ValueError:
            continue

        file_path = os.path.join(strategy_dir, filename)
        module_name = f"strategy_{filename[:-3]}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "execute_query"):
                strategy_funcs[idx] = module.execute_query
            else:
                print(f"WARNING: {filename} has no execute_query(retrieve)")
        
        except SyntaxError as e:
            print(f"ERROR: Syntax error in {filename} (line {e.lineno}): {e.msg}")
            print(f"       Please fix this file before running execution.")
        except Exception as e:
            print(f"ERROR: Failed to load {filename}: {e}")

    return strategy_funcs

def execute_all_strategies(strategies_dir: str, queries_path: str, output_path: str, config: dict):
    """
    Main execution loop:
    1. Initializes collection using standard utils.
    2. Loads all strategy scripts.
    3. Runs them against the queries using a wrapped retrieve function.
    4. Saves results.
    """
    # 1. Initialize Collection (using chroma_utils)
    print("Initializing ChromaDB collection...")
    collection = get_db_collection(config)

    # 2. Load gold queries
    try:
        with open(queries_path, "r", encoding="utf-8") as f:
            queries = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Queries file not found at {queries_path}")
        return

    # 3. Load strategies
    strategy_funcs = load_strategy_functions(strategies_dir)
    print(f"Found {len(strategy_funcs)} valid strategy functions.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    def retrieve_wrapper(query_text, k):
        return retrieve(collection, query_text, k, include_chunks=False)

    # 4. Execute
    processed_count = 0
    with open(output_path, "w", encoding="utf-8") as f_out:
        for i, q_data in enumerate(tqdm(queries, desc="Executing Strategies")):
            query_text = q_data["query"]
            if i in strategy_funcs:
                try:
                    final_docs_set = strategy_funcs[i](retrieve_wrapper)
                    final_docs = list(final_docs_set) if isinstance(final_docs_set, set) else final_docs_set
                    prediction = {
                        "query": query_text,
                        "docs": final_docs
                    }
                    f_out.write(json.dumps(prediction) + "\n")
                    processed_count += 1
                except Exception as e:
                    print(f"Runtime Error in strategy {i+1}: {e}")
            else:
                pass

    print(f"Execution complete. {processed_count}/{len(queries)} queries processed.")
    print(f"Results saved to {output_path}")