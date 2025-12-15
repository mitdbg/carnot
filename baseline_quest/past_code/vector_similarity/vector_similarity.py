try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import json
import os
import torch
from typing import Dict, Iterable, List
from sentence_transformers import SentenceTransformer, util

GOLD_FILE_PATH = "data/train_subset3.jsonl"
RETRIEVED_FILE_PATH = "full_document/pred_unranked.jsonl"
OUTPUT_REPORT_PATH = "gold_vs_retrieved_similarity_report.txt"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# set query and baseline/unrelated query
QUERY_TO_ANALYZE = "cultural geography and Science books but not about creativity"
UNRELATED_QUERY = "1947 Science Linguistics books"

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

def log_and_print(message: str, f_handle):
    """Helper function to write to file and print to console."""
    print(message)
    f_handle.write(message + "\n")

def get_gold_chunk(gold_item: Dict, title: str, query: str) -> str:
    """
    Extracts the "best" gold chunk from the complex attributions field.
    """
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
        print(f"Error parsing gold chunk for {title}: {e}")
        return None

def main():
    if not os.path.isfile(GOLD_FILE_PATH):
        print(f"Error: Gold file not found at '{GOLD_FILE_PATH}'")
        return
    if not os.path.isfile(RETRIEVED_FILE_PATH):
        print(f"Error: Retrieved file not found at '{RETRIEVED_FILE_PATH}'")
        return

    # 1. Load the Embedding Model
    print(f"Loading embedding model '{MODEL_NAME}' on device '{DEVICE}'...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print("Model loaded.")

    # 2. Load all data
    print("Loading data...")
    gold_data = {item['query']: item for item in read_jsonl(GOLD_FILE_PATH)}
    retrieved_data = {item['query']: item for item in read_jsonl(RETRIEVED_FILE_PATH)}
    print("Data loaded.")

    # 3. Find the specific query
    query = QUERY_TO_ANALYZE
    if query not in retrieved_data or query not in gold_data:
        print(f"Error: Query '{query}' not found in one or both files.")
        return

    gold_item = gold_data[query]
    retrieved_item = retrieved_data[query]
    
    gold_titles = set(gold_item.get("docs", []))
    retrieved_docs_list = retrieved_item.get("docs", [])
    
    # Create a fast lookup map for retrieved chunks
    retrieved_chunks_map = {
        d['title']: d['chunk'] 
        for d in retrieved_docs_list if d.get('title')
    }
    
    # Get a list of "False Positive" documents (title and chunk)
    fp_data = [
        {'title': d['title'], 'chunk': d['chunk']}
        for d in retrieved_docs_list 
        if d.get('title') not in gold_titles and d.get('chunk')
    ]
    
    # Select one "representative" False Positive document
    rep_fp_doc = fp_data[0] if fp_data else None

    unrelated_docs_data = []
    if UNRELATED_QUERY in retrieved_data:
        unrelated_item = retrieved_data[UNRELATED_QUERY]
        unrelated_docs_list = unrelated_item.get('docs', [])
        if unrelated_docs_list:
            for doc_data in unrelated_docs_list:
                unrelated_docs_data.append({
                    'query': UNRELATED_QUERY,
                    'title': doc_data.get('title'),
                    'chunk': doc_data.get('chunk')
                })
        else:
            print(f"Warning: Specified unrelated query '{UNRELATED_QUERY}' has no retrieved docs.")
    else:
        print(f"Warning: Specified unrelated query '{UNRELATED_QUERY}' not found in retrieved file.")

    # 4. Prepare all texts for batch embedding
    texts_to_encode = [query]
    text_info: List[Dict] = []

    for title in gold_titles:
        # Get the gold chunk
        gold_chunk = get_gold_chunk(gold_item, title, query)
        if not gold_chunk:
            print(f"Warning: Could not find gold chunk for {title}")
            continue
            
        texts_to_encode.append(gold_chunk)
        text_info.append({'title': title, 'type': 'gold_chunk', 'chunk': gold_chunk})

        # CASE 1: True Positive (document was retrieved)
        if title in retrieved_chunks_map:
            retrieved_chunk = retrieved_chunks_map[title]
            texts_to_encode.append(retrieved_chunk)
            text_info.append({'title': title, 'type': 'retrieved_chunk', 'chunk': retrieved_chunk})
        
        # CASE 2: False Negative (document was NOT retrieved)
        else:
            if rep_fp_doc:
                texts_to_encode.append(rep_fp_doc['chunk'])
                text_info.append({
                    'title': title, # Still keyed to the gold title
                    'type': 'rep_fp_chunk', 
                    'chunk': rep_fp_doc['chunk'],
                    'fp_title': rep_fp_doc['title']
                })
    
    # Add all other false positives
    other_fp_data = fp_data[1:] if rep_fp_doc else fp_data
    for fp_doc in other_fp_data:
        texts_to_encode.append(fp_doc['chunk'])
        text_info.append({'title': fp_doc['title'], 'type': 'pure_fp_chunk', 'chunk': fp_doc['chunk']})

    # Add all the unrelated chunks
    if unrelated_docs_data:
        for doc_data in unrelated_docs_data:
            if doc_data['chunk']:
                texts_to_encode.append(doc_data['chunk'])
                text_info.append({
                    'title': doc_data['title'], 
                    'type': 'unrelated_chunk', 
                    'chunk': doc_data['chunk'],
                    'orig_query': doc_data['query']
                })

    # 5. Calculate all embeddings and similarities
    print("Calculating similarities...")
    embeddings = model.encode(
        texts_to_encode, 
        convert_to_tensor=True, 
        normalize_embeddings=True
    )
    
    query_embedding = embeddings[0]
    chunk_embeddings = embeddings[1:]
    
    similarities = util.cos_sim(query_embedding, chunk_embeddings)[0].tolist()
    
    # 6. Process results and write the report
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f_report:
        log_and_print("--- Similarity Analysis Report ---", f_report)
        log_and_print(f"\nQUERY: {query}", f_report)
        log_and_print(f"\nGOLD TITLES: {list(gold_titles)}", f_report)
        log_and_print("\n" + "="*80, f_report)
        log_and_print("--- Analysis of Gold Set Documents ---", f_report)

        # Store results for organized printing
        report_data = {title: {} for title in gold_titles}
        pure_fp_data = []
        unrelated_data = []

        sim_index = 0
        for info in text_info:
            sim = similarities[sim_index]
            chunk = info['chunk']
            title = info['title']
            sim_index += 1

            if info['type'] == 'gold_chunk':
                report_data[title]['gold_sim'] = sim
                report_data[title]['gold_chunk'] = chunk
            elif info['type'] == 'retrieved_chunk':
                report_data[title]['retrieved_sim'] = sim
                report_data[title]['retrieved_chunk'] = chunk
            elif info['type'] == 'rep_fp_chunk':
                report_data[title]['fp_sim'] = sim
                report_data[title]['fp_chunk'] = chunk
                report_data[title]['fp_title'] = info['fp_title']
            elif info['type'] == 'pure_fp_chunk':
                pure_fp_data.append({'title': title, 'sim': sim, 'chunk': chunk})
            elif info['type'] == 'unrelated_chunk':
                unrelated_data.append({
                    'title': title, 
                    'sim': sim, 
                    'chunk': chunk,
                    'orig_query': info['orig_query']
                })

        # Print the analysis for each gold document
        for title in gold_titles:
            log_and_print(f"\n--- Document: {title} ---", f_report)
            data = report_data.get(title, {})

            if 'gold_sim' not in data:
                log_and_print("  [!] Error: Gold chunk was not found or processed.", f_report)
                continue
                
            log_and_print(f"  Sim (Query <-> Gold Chunk):      {data['gold_sim']:.4f}", f_report)
            log_and_print(f"    Gold Chunk: {data['gold_chunk'][:200]}...", f_report)

            if 'retrieved_sim' in data:
                log_and_print(f"  Sim (Query <-> Retrieved Chunk): {data['retrieved_sim']:.4f}", f_report)
                log_and_print(f"    Retrieved Chunk: {data['retrieved_chunk'][:200]}...", f_report)
                log_and_print(f"  Status: TRUE POSITIVE", f_report)
            elif 'fp_sim' in data:
                log_and_print(f"  Sim (Query <-> Random FP Chunk): {data['fp_sim']:.4f}", f_report)
                log_and_print(f"    FP Title: {data['fp_title']}", f_report)
                log_and_print(f"    FP Chunk: {data['fp_chunk'][:200]}...", f_report)
                log_and_print(f"  Status: FALSE NEGATIVE", f_report)
            else:
                log_and_print(f"  Status: FALSE NEGATIVE (No FP chunks found for comparison)", f_report)
                
        # Print the extra False Positives
        log_and_print("\n" + "="*80, f_report)
        log_and_print("--- Analysis of Other False Positives (Not in Gold Set) ---", f_report)
        
        if pure_fp_data:
            pure_fp_data.sort(key=lambda x: x['sim'], reverse=True)
            for item in pure_fp_data:
                log_and_print(f"\n  Sim: {item['sim']:.4f} | Title: {item['title']}", f_report)
                log_and_print(f"    Chunk: {item['chunk'][:200]}...", f_report)
        else:
            log_and_print("  None found.", f_report)
            
        # Print the Unrelated Chunk analysis
        log_and_print("\n" + "="*80, f_report)
        log_and_print("--- Analysis of Unrelated Chunks (Baseline) ---", f_report)
        
        if unrelated_data:
            log_and_print(f"  (Chunks from query: '{UNRELATED_QUERY}')", f_report)
            # Sort by similarity and print all
            unrelated_data.sort(key=lambda x: x['sim'], reverse=True)
            for item in unrelated_data:
                log_and_print(f"\n  Sim (Target Query <-> Unrelated Chunk): {item['sim']:.4f}", f_report)
                log_and_print(f"    Unrelated Title: {item['title']}", f_report)
                log_and_print(f"    Unrelated Chunk: {item['chunk'][:200]}...", f_report)
        else:
            log_and_print("  Could not find the specified unrelated chunks to compare.", f_report)

    print(f"\nAnalysis complete. Report saved to '{OUTPUT_REPORT_PATH}'")

if __name__ == "__main__":
    main()