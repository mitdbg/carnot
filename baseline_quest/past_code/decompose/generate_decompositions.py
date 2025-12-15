"""
Uses DSPy to generate modular Python strategy files for each QUEST query.
Generates a master `execute_all.py` script in each output directory
to run the full pipeline using a separate retriever module and set operations,
without reranking, and without truncating the final result list.
"""
import os
import json
import dspy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm

# --- Configuration ---
INPUT_QUERIES_PATH = "../data/train_subset3.jsonl"
OUTPUT_STRATEGIES_DIR = "decomposition_scripts"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = 'gpt-5-mini'
TARGET_K = 100

@dataclass
class QuestQuery:
    query: str
    docs: List[str]
    original_query: str
    metadata: Optional[Dict[str, Any]] = None

def load_quest_queries(path: str) -> List[QuestQuery]:
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                queries.append(QuestQuery(**{k: v for k, v in data.items() if k in QuestQuery.__annotations__}))
    return queries

# --- DSPy Module for Query Decomposition (RETRIEVAL ONLY) ---
class QueryDecomposerWithPython(dspy.Signature):
    query = dspy.InputField(desc="A natural language retrieval task with implicit set operations.")
    target_k = dspy.InputField(desc="The target number of final documents to return. The final list will NOT be truncated to this k.")
    
    output = dspy.OutputField(
        desc=(
            "A Python program that decomposes the query into a sequence of smaller retrieval "
            "tasks and combines the results using set operations (&, |, -). The program must "
            "call the function retrieve(query_string, k), where k is the top-k documents to "
            "retrieve and the function returns a set of document IDs (str). "
            "Tune k according to the mix of set operations, using larger k when more restrictive "
            "operations (&, -) are involved. "
            "Return the results from the set operations as a list, without truncation by target_k. "
            "Store the final list of document IDs in a variable named result. "
            "The last line of the program MUST be: return result. "
            "Write ONLY the Python code inside the function body. Do NOT write `def execute_query(...)`. "
            "Wrap the code block in <python> and </python>."
        )
    )

class QuestQueryDecomposer(dspy.Module):
    few_shot_examples = [
        dspy.Example(
            query="Birds of Kolombangara or of the Western Province (Solomon Islands)",
            target_k=50,
            output="""<python>
birds_of_kolombangara_ids = retrieve('Birds of Kolombangara', k=100)
western_province_birds_ids = retrieve('Birds of the Western Province (Solomon Islands)', k=100)
final_doc_ids = birds_of_kolombangara_ids | western_province_birds_ids
result = list(final_doc_ids)
return result
</python>"""
        ),
        dspy.Example(
            query="Trees of South Africa that are also in the south-central Pacific",
            target_k=50,
            output="""<python>
trees_sa_ids = retrieve('Trees of South Africa', k=100)
trees_scp_ids = retrieve('south-central Pacific', k=100)
final_doc_ids = trees_sa_ids & trees_scp_ids
result = list(final_doc_ids)
return result
</python>"""
        ),
        dspy.Example(
            query="2010's adventure films set in the Southwestern United States but not in California",
            target_k=50,
            output="""<python>
adventure_films_ids = retrieve('2010s adventure films', k=200)
sw_us_films_ids = retrieve('films set in the Southwestern United States', k=200)
california_films_ids = retrieve('films set in California', k=200)
inclusive_films = adventure_films_ids & sw_us_films_ids
final_doc_ids = inclusive_films - california_films_ids
result = list(final_doc_ids)
return result
</python>"""
        )
    ]

    def __init__(self):
        super().__init__()
        # Kept at 1.0, but consider lowering to 0.7 or 0.5 if the model ignores the k-sizing guidance.
        self.generate_code = dspy.Predict(QueryDecomposerWithPython, n=1, temperature=1.0) 

    def forward(self, query: str, target_k: int):
        return self.generate_code(query=query, target_k=target_k, Demos=self.few_shot_examples)

def generate_strategies():
    if not OPENAI_API_KEY:
        print("FATAL ERROR: OPENAI_API_KEY environment variable not set.")
        return

    try:
        lm = dspy.LM(
            f'openai/{LLM_MODEL}', 
            temperature=1.0, 
            max_tokens=16000,
            api_key=OPENAI_API_KEY
        )
        dspy.configure(lm=lm)
        print(f"Successfully configured dspy.LM with model: {LLM_MODEL}")
    except Exception as e:
        print(f"Error configuring DSPy language model: {e}")
        return

    queries = load_quest_queries(INPUT_QUERIES_PATH)
    
    queries_to_process = queries

    decomposer = QuestQueryDecomposer()
    
    output_dir_for_k = os.path.join(OUTPUT_STRATEGIES_DIR, f"k_{TARGET_K}")
    os.makedirs(output_dir_for_k, exist_ok=True)
        
    import_statements = []
    strategy_list = []

    print(f"\n--- Generating Python strategies for k={TARGET_K} into '{output_dir_for_k}/' (No Truncation) ---")
    for i, quest_query in enumerate(queries_to_process):
        print(f"  Processing query {i+1}/{len(queries_to_process)}: {quest_query.query}")
        try:
            result = decomposer(query=quest_query.query, target_k=TARGET_K)
            python_code = result.output.strip().replace("<python>", "").replace("</python>", "").strip()
        except Exception as e:
            print(f"    ERROR generating strategy for query {i+1}: {e}")
            print("    Skipping this query.")
            continue

        filename = f"query_{i+1}.py"
        filepath = os.path.join(output_dir_for_k, filename)
        with open(filepath, "w", encoding="utf-8") as f_out:
            f_out.write(f"# Strategy for Query {i+1}: {quest_query.query} (No Truncation)\n")
            f_out.write(f'def execute_query(retrieve):\n') 
            f_out.write("    " + python_code.replace("\n", "\n    ") + "\n")
            
        import_statements.append(f"from query_{i+1} import execute_query as execute_query_{i+1}")
        strategy_list.append(f"execute_query_{i+1},")

    print("\n--- All strategies scripts generated successfully! ---")

if __name__ == "__main__":
    generate_strategies()