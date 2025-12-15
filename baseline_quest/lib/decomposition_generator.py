import os
import json
import dspy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# --- Data Structures ---

@dataclass
class QuestQuery:
    query: str
    docs: List[str]
    original_query: str
    metadata: Optional[Dict[str, Any]] = None

# --- DSPy Module for Query Decomposition ---

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
        self.generate_code = dspy.Predict(QueryDecomposerWithPython, n=1, temperature=1.0) 

    def forward(self, query: str, target_k: int):
        return self.generate_code(query=query, target_k=target_k, Demos=self.few_shot_examples)

# --- Main Generation Logic ---

def generate_strategies(queries_path: str, output_dir: str, target_k: int, llm_model: str):
    """
    Generates Python strategy files for each query in the input file using DSPy.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("FATAL ERROR: OPENAI_API_KEY environment variable not set.")
        return

    # 1. Configure DSPy
    try:
        lm = dspy.LM(
            f'openai/{llm_model}', 
            temperature=1.0, 
            max_tokens=2000, 
            api_key=api_key
        )
        dspy.configure(lm=lm)
        print(f"DSPy configured with model: {llm_model}")
    except Exception as e:
        print(f"Error configuring DSPy: {e}")
        return

    # 2. Load Queries
    queries = []
    try:
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Safe load into dataclass
                    queries.append(QuestQuery(**{k: v for k, v in data.items() if k in QuestQuery.__annotations__}))
    except FileNotFoundError:
        print(f"Error: Query file not found at {queries_path}")
        return

    # 3. Prepare Output Directory
    os.makedirs(output_dir, exist_ok=True)
    decomposer = QuestQueryDecomposer()
    
    print(f"Generating {len(queries)} strategies into '{output_dir}'...")

    # 4. Generate Strategies Loop
    generated_count = 0
    for i, quest_query in enumerate(queries):
        try:
            # Generate code using DSPy
            result = decomposer(query=quest_query.query, target_k=target_k)
            
            # Clean up the output to extract pure Python code
            python_code = result.output.strip().replace("<python>", "").replace("</python>", "").strip()
            
            # Write to file
            filename = f"query_{i+1}.py"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f_out:
                f_out.write(f"# Strategy for Query {i+1}: {quest_query.query}\n")
                # We wrap the code in a function definition
                f_out.write(f'def execute_query(retrieve):\n') 
                # Indent the generated code
                f_out.write("    " + python_code.replace("\n", "\n    ") + "\n")
            
            generated_count += 1
            
        except Exception as e:
            print(f"  [Error] Query {i+1} ('{quest_query.query}'): {e}")
            continue

    print(f"Successfully generated {generated_count}/{len(queries)} strategy files.")