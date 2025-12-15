import argparse
import logging
import os
import sys
import torch
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util

from lib.chroma_utils import read_jsonl

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_gold_chunk(gold_item: Dict, title: str, query: str) -> str:
    """Extracts the 'best' gold chunk from the complex attributions field."""
    try:
        attributions = gold_item.get('metadata', {}).get('attributions', {})
        attrib_list = attributions.get(title, [])
        
        if not attrib_list:
            return None

        # The attribution is a list, usually of one dict
        attrib_dict = attrib_list[0]
        
        # Ideal case: the query itself is the key
        if query in attrib_dict:
            return attrib_dict[query]
        
        # Fallback: concatenate all chunk snippets
        return " ".join(attrib_dict.values())
        
    except Exception as e:
        logger.warning(f"Error parsing gold chunk for {title}: {e}")
        return None

def log_and_write(f_handle, message: str):
    """Helper to write to the report file and log to console."""
    logger.info(message)
    f_handle.write(message + "\n")

def run_similarity_analysis(
    gold_path: str,
    retrieved_path: str,
    output_path: str,
    target_query: str,
    unrelated_query: str,
    model_name: str
):
    """Main execution logic for similarity analysis."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Validation
    if not os.path.isfile(gold_path):
        logger.error(f"Gold file not found at '{gold_path}'")
        return
    if not os.path.isfile(retrieved_path):
        logger.error(f"Retrieved file not found at '{retrieved_path}'")
        return

    # 2. Load Model
    logger.info(f"Loading embedding model '{model_name}' on device '{device}'...")
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 3. Load Data using chroma_utils.read_jsonl
    logger.info("Loading dataset files...")
    try:
        gold_data = {item['query']: item for item in read_jsonl(gold_path)}
        retrieved_data = {item['query']: item for item in read_jsonl(retrieved_path)}
    except Exception as e:
        logger.error(f"Error reading JSONL files: {e}")
        return

    # 4. Check Query Existence
    if target_query not in gold_data:
        logger.error(f"Target query not found in GOLD file: '{target_query}'")
        return
    if target_query not in retrieved_data:
        logger.error(f"Target query not found in RETRIEVED file: '{target_query}'")
        return

    gold_item = gold_data[target_query]
    retrieved_item = retrieved_data[target_query]
    
    gold_titles = set(gold_item.get("docs", []))
    retrieved_docs_list = retrieved_item.get("docs", [])

    # Map retrieved chunks for lookup
    retrieved_chunks_map = {
        d['title']: d['chunk'] 
        for d in retrieved_docs_list if d.get('title')
    }

    # Identify False Positives (documents retrieved but not in gold set)
    fp_data = [
        {'title': d['title'], 'chunk': d['chunk']}
        for d in retrieved_docs_list 
        if d.get('title') not in gold_titles and d.get('chunk')
    ]
    rep_fp_doc = fp_data[0] if fp_data else None

    # Load Unrelated Data (Baseline)
    unrelated_docs_data = []
    if unrelated_query in retrieved_data:
        unrelated_item = retrieved_data[unrelated_query]
        unrelated_list = unrelated_item.get('docs', [])
        if unrelated_list:
            for doc in unrelated_list:
                unrelated_docs_data.append({
                    'query': unrelated_query,
                    'title': doc.get('title'),
                    'chunk': doc.get('chunk')
                })
        else:
            logger.warning(f"Unrelated query found, but has no docs: '{unrelated_query}'")
    else:
        logger.warning(f"Unrelated query not found in retrieved file: '{unrelated_query}'")

    # 5. Prepare Batch Encoding
    texts_to_encode = [target_query]
    text_info = []

    # A. Gold Documents
    for title in gold_titles:
        gold_chunk = get_gold_chunk(gold_item, title, target_query)
        if not gold_chunk:
            logger.warning(f"Skipping {title}: No gold chunk text found.")
            continue
        
        texts_to_encode.append(gold_chunk)
        text_info.append({'title': title, 'type': 'gold_chunk', 'chunk': gold_chunk})

        # Check if retrieved (True Positive) or missed (False Negative)
        if title in retrieved_chunks_map:
            retrieved_chunk = retrieved_chunks_map[title]
            texts_to_encode.append(retrieved_chunk)
            text_info.append({'title': title, 'type': 'retrieved_chunk', 'chunk': retrieved_chunk})
        else:
            # If missed, compare against a "Representative False Positive" if available
            if rep_fp_doc:
                texts_to_encode.append(rep_fp_doc['chunk'])
                text_info.append({
                    'title': title, # Linked to gold title context
                    'type': 'rep_fp_chunk',
                    'chunk': rep_fp_doc['chunk'],
                    'fp_title': rep_fp_doc['title']
                })

    # B. Pure False Positives (not linked to any gold doc)
    other_fp_data = fp_data[1:] if rep_fp_doc else fp_data
    for fp_doc in other_fp_data:
        texts_to_encode.append(fp_doc['chunk'])
        text_info.append({'title': fp_doc['title'], 'type': 'pure_fp_chunk', 'chunk': fp_doc['chunk']})

    # C. Unrelated Chunks
    for doc in unrelated_docs_data:
        if doc['chunk']:
            texts_to_encode.append(doc['chunk'])
            text_info.append({
                'title': doc['title'],
                'type': 'unrelated_chunk',
                'chunk': doc['chunk'],
                'orig_query': doc['query']
            })

    # 6. Calculate Similarities
    logger.info(f"Encoding {len(texts_to_encode)} texts...")
    embeddings = model.encode(texts_to_encode, convert_to_tensor=True, normalize_embeddings=True)
    
    query_embedding = embeddings[0]
    chunk_embeddings = embeddings[1:]
    
    # Compute cosine similarity
    similarities = util.cos_sim(query_embedding, chunk_embeddings)[0].tolist()

    # 7. Generate Report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        log_and_write(f, "--- Similarity Analysis Report ---")
        log_and_write(f, f"Query: {target_query}")
        log_and_write(f, f"Gold Titles: {list(gold_titles)}\n")
        log_and_write(f, "="*80)
        
        # Organize results
        report_map = {title: {} for title in gold_titles}
        pure_fps = []
        unrelateds = []

        sim_idx = 0
        for info in text_info:
            sim = similarities[sim_idx]
            sim_idx += 1
            
            t_type = info['type']
            title = info['title']
            
            if t_type == 'gold_chunk':
                report_map[title]['gold'] = {'sim': sim, 'chunk': info['chunk']}
            elif t_type == 'retrieved_chunk':
                report_map[title]['retrieved'] = {'sim': sim, 'chunk': info['chunk']}
            elif t_type == 'rep_fp_chunk':
                report_map[title]['fp'] = {'sim': sim, 'chunk': info['chunk'], 'title': info['fp_title']}
            elif t_type == 'pure_fp_chunk':
                pure_fps.append({'sim': sim, 'chunk': info['chunk'], 'title': title})
            elif t_type == 'unrelated_chunk':
                unrelateds.append({'sim': sim, 'chunk': info['chunk'], 'title': title})

        # Write Analysis: Gold Docs
        log_and_write(f, "--- Analysis of Gold Set Documents ---")
        for title in gold_titles:
            log_and_write(f, f"\nDocument: {title}")
            data = report_map.get(title, {})
            
            if 'gold' in data:
                g_sim = data['gold']['sim']
                g_chunk = data['gold']['chunk'][:200].replace('\n', ' ')
                log_and_write(f, f"  Sim (Query <-> Gold Chunk):      {g_sim:.4f}")
                log_and_write(f, f"    Chunk: {g_chunk}...")
            else:
                log_and_write(f, "  [!] Error: Gold chunk missing.")
                continue

            if 'retrieved' in data:
                r_sim = data['retrieved']['sim']
                r_chunk = data['retrieved']['chunk'][:200].replace('\n', ' ')
                log_and_write(f, f"  Sim (Query <-> Retrieved Chunk): {r_sim:.4f}")
                log_and_write(f, f"    Chunk: {r_chunk}...")
                log_and_write(f, f"  Status: TRUE POSITIVE")
            elif 'fp' in data:
                fp_sim = data['fp']['sim']
                fp_chunk = data['fp']['chunk'][:200].replace('\n', ' ')
                fp_title = data['fp']['title']
                log_and_write(f, f"  Sim (Query <-> Random FP Chunk): {fp_sim:.4f}")
                log_and_write(f, f"    FP Title: {fp_title}")
                log_and_write(f, f"    Chunk: {fp_chunk}...")
                log_and_write(f, f"  Status: FALSE NEGATIVE (Doc not retrieved)")
            else:
                log_and_write(f, f"  Status: FALSE NEGATIVE (No FP available for comparison)")

        # Write Analysis: Other FPs
        log_and_write(f, "\n" + "="*80)
        log_and_write(f, "--- Analysis of Other False Positives (Not in Gold Set) ---")
        if pure_fps:
            pure_fps.sort(key=lambda x: x['sim'], reverse=True)
            for item in pure_fps:
                log_and_write(f, f"\n  Sim: {item['sim']:.4f} | Title: {item['title']}")
                log_and_write(f, f"    Chunk: {item['chunk'][:200].replace('\n', ' ')}...")
        else:
            log_and_write(f, "  None found.")

        # Write Analysis: Unrelated
        log_and_write(f, "\n" + "="*80)
        log_and_write(f, "--- Analysis of Unrelated Chunks (Baseline) ---")
        log_and_write(f, f"  (Chunks from query: '{unrelated_query}')")
        
        if unrelateds:
            unrelateds.sort(key=lambda x: x['sim'], reverse=True)
            for item in unrelateds:
                log_and_write(f, f"\n  Sim: {item['sim']:.4f} | Title: {item['title']}")
                log_and_write(f, f"    Chunk: {item['chunk'][:200].replace('\n', ' ')}...")
        else:
            log_and_write(f, "  No unrelated chunks found.")

    logger.info(f"Report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze embedding similarity between Query, Gold Chunks, and Retrieved Chunks.")
    
    # Default parameters based on user's original script
    parser.add_argument("--gold", type=str, default="data/train_subset3.jsonl",
                        help="Path to gold standard JSONL file.")
    parser.add_argument("--retrieved", type=str, default="full_document/pred_unranked.jsonl",
                        help="Path to retrieved predictions JSONL file.")
    parser.add_argument("--out", type=str, default="gold_vs_retrieved_similarity_report.txt",
                        help="Path to output text report.")
    parser.add_argument("--query", type=str, 
                        default="cultural geography and Science books but not about creativity",
                        help="The specific query string to analyze.")
    parser.add_argument("--unrelated-query", type=str, 
                        default="1947 Science Linguistics books",
                        help="An unrelated query string to fetch baseline chunks.")
    parser.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5",
                        help="HuggingFace model name for embeddings.")

    args = parser.parse_args()

    run_similarity_analysis(
        gold_path=args.gold,
        retrieved_path=args.retrieved,
        output_path=args.out,
        target_query=args.query,
        unrelated_query=args.unrelated_query,
        model_name=args.model
    )

if __name__ == "__main__":
    main()