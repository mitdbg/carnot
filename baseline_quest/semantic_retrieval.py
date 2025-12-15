import sys
import os
import json
import logging
import yaml
import argparse

# SQLite compatibility
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from lib
from lib.chroma_utils import get_db_collection, read_jsonl
from lib.retriever import retrieve_batch
from lib.analyze_retrieval_results import calculate_recall_stats, write_analysis_report

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
    try:
        collection = get_db_collection(config) #
    except Exception as e:
        logger.error(f"Failed to load collection: {e}")
        return

    # 3. Load Queries
    logger.info(f"Reading queries from {queries_path}...")
    try:
        queries_data = list(read_jsonl(queries_path))
    except FileNotFoundError:
        logger.error(f"Query file not found: {queries_path}")
        return

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
    try:
        stats = calculate_recall_stats(queries_path, output_pred_path) #
        
        # 7. Write Report
        logger.info(f"Writing analysis report to {output_report_path}...")
        write_analysis_report(stats, output_report_path) #
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

    logger.info("Semantic retrieval completed successfully.")

if __name__ == "__main__":
    main()