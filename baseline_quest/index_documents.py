__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import signal
import logging
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

# Import tools from the library
from lib.chroma_utils import (
    get_db_collection, 
    read_jsonl, 
    stable_entity_id, 
    chunk_by_tokens
)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("indexing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def upsert_in_batches(collection, ids, documents, metadatas, batch_size):
    for i in range(0, len(ids), batch_size):
        j = i + batch_size
        collection.upsert(
            ids=ids[i:j], 
            documents=documents[i:j], 
            metadatas=metadatas[i:j]
        )

def main():
    # 1. Load Config
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        logger.error("config.yaml not found.")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    idx_conf = config['indexing']
    doc_path = config['data']['document_path']

    logger.info(f"Source file: {doc_path}")
    logger.info(f"Collection: {idx_conf['chroma']['collection']}")
    logger.info(f"Strategy: {'First 512 Only' if idx_conf['index_first_512'] else 'Full Sliding Window'}")

    # 2. Initialize Resources
    if not os.path.isfile(doc_path):
        logger.error(f"Data file not found: {doc_path}")
        sys.exit(1)

    collection = get_db_collection(config, clear_existing=True)
    tokenizer = AutoTokenizer.from_pretrained(idx_conf['embedding_model'], use_fast=True)

    # 3. Prepare Loop
    total_docs = sum(1 for _ in read_jsonl(doc_path))
    docs_iter = read_jsonl(doc_path)
    
    p_docs = tqdm(total=total_docs, desc="Docs", unit="doc")
    p_units = tqdm(total=0, desc="Chunks", unit="chunk")

    batch_ids, batch_docs, batch_metas = [], [], []
    running_total = 0
    
    # Graceful Interrupt Handling
    interrupted = False
    def handle_sigint(sig, frame):
        nonlocal interrupted
        interrupted = True
        logger.warning("Interrupt received. Finishing current batch...")
    
    original_sigint = signal.signal(signal.SIGINT, handle_sigint)

    # 4. Processing Loop
    try:
        for raw in docs_iter:
            if interrupted: break

            title = (raw.get("title") or "").strip() or "untitled"
            text = (raw.get("text") or "").strip()
            entity_id = stable_entity_id(title, text)
            
            # --- STRATEGY A: First 512 Tokens Only ---
            if idx_conf['index_first_512']:
                toks = tokenizer.encode(text, add_special_tokens=False)
                truncated = toks[:idx_conf['chunk_size']]
                chunk_text = tokenizer.decode(truncated, skip_special_tokens=True).strip()
                if not chunk_text: chunk_text = title

                batch_ids.append(entity_id)
                batch_docs.append(chunk_text)
                batch_metas.append({
                    "entity_id": entity_id,
                    "title": title,
                    "chunk_index": 0,
                    "n_chunks": 1,
                    "source": os.path.basename(doc_path),
                })
                running_total += 1

            # --- STRATEGY B: Full Sliding Window ---
            else:
                chunks = chunk_by_tokens(
                    text, tokenizer, 
                    idx_conf['chunk_size'], 
                    idx_conf['overlap']
                )
                if not chunks: chunks = [title]
                
                n_chunks = len(chunks)
                for idx, ch in enumerate(chunks):
                    # Create unique ID for chunk
                    cid = f"{entity_id}__{idx:04d}"
                    batch_ids.append(cid)
                    batch_docs.append(ch)
                    batch_metas.append({
                        "entity_id": entity_id,
                        "title": title,
                        "chunk_index": idx,
                        "n_chunks": n_chunks,
                        "source": os.path.basename(doc_path),
                    })
                running_total += n_chunks

            p_docs.update(1)
            p_units.total = running_total
            p_units.refresh()

            # Flush Batch
            if len(batch_ids) >= idx_conf['batch_size']:
                upsert_in_batches(collection, batch_ids, batch_docs, batch_metas, idx_conf['batch_size'])
                p_units.update(len(batch_ids))
                batch_ids.clear(); batch_docs.clear(); batch_metas.clear()

        # Final Flush
        if batch_ids:
            upsert_in_batches(collection, batch_ids, batch_docs, batch_metas, idx_conf['batch_size'])
            p_units.update(len(batch_ids))

    finally:
        signal.signal(signal.SIGINT, original_sigint)
        p_docs.close()
        p_units.close()
        logger.info(f"Indexing complete. Total units: {p_units.n}")

if __name__ == "__main__":
    main()