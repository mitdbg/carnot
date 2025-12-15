import sys
import os
import json
import logging
import yaml
import argparse

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from lib
from lib.chroma_utils import get_db_collection, read_jsonl
from lib.retriever import retrieve_batch
from baseline_quest.lib.retrieval_analyzer import analyze_results

CONFIG_PATH = "config.yaml"

def load_config(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. Load Configuration
    logger.info(f"Loading configuration from {CONFIG_PATH}...")
    config = load_config(CONFIG_PATH)

    # Extract key settings from the 'retrieval' block
    queries_path = config['data']['queries_file']
    top_k = config['retrieval'].get('top_k', 100)
    include_chunks = config['retrieval'].get('include_chunks', False)
    is_limited = config['indexing'].get('index_first_512', False)
    
    suffix = "_limited" if is_limited else ""
    subset_name = os.path.splitext(os.path.basename(queries_path))[0] 
    output_pred_path = f"results/semantic_retrieval/pred_unranked{suffix}_{subset_name}.jsonl"
    output_report_path = f"results/semantic_retrieval/recall_report{suffix}_{subset_name}.txt"

    # 2. Initialize Collection
    logger.info("Initializing ChromaDB collection...")
    collection = get_db_collection(config)

    # 3. Load Queries
    logger.info(f"Reading queries from {queries_path}...")
    queries_data = list(read_jsonl(queries_path))

    query_texts = [item['query'] for item in queries_data]
    logger.info(f"Found {len(query_texts)} queries. Starting retrieval (k={top_k}, chunks={include_chunks})...")

    # 4. Run Batch Retrieval
    results = retrieve_batch(
        collection, 
        query_texts, 
        k=top_k, 
        include_chunks=include_chunks
    )

    # 5. Write Predictions
    logger.info(f"Writing predictions to {output_pred_path}...")
    os.makedirs(os.path.dirname(output_pred_path), exist_ok=True)
    
    with open(output_pred_path, 'w', encoding='utf-8') as f_out:
        for q_text, res in zip(query_texts, results):
            if include_chunks:
                # result: Dict[title, chunk] -> Convert to [{"title": t, "chunk": c}, ...]
                docs_output = [{"title": t, "chunk": c} for t, c in res.items()]
            else:
                # result: Set[title] -> Convert to ["title1", "title2", ...]
                docs_output = list(res)

            f_out.write(json.dumps({
                "query": q_text,
                "docs": docs_output
            }) + "\n")

    # 6. Analyze Results
    logger.info("Calculating recall statistics...")
    analyze_results(queries_path, output_pred_path, output_report_path)
    logger.info(f'Semantic retrieval completed successfully. Results saved to{output_report_path}')

if __name__ == "__main__":
    main()