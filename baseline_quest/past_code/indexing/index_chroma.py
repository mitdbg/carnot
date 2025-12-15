#!/usr/bin/env python3

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os, json, re, unicodedata, hashlib, signal, logging
from typing import List, Dict, Iterable
from tqdm import tqdm
from unidecode import unidecode

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DOCUMENT_PATH = "/orcd/home/002/joycequ/quest_data/documents.jsonl"
DEVICE = None # = "cuda" if GPU
INDEX_FIRST_512 = True
# INDEX_FIRST_512 = True  --> index the first 512 tokens of each document
# INDEX_FIRST_512 = False --> index entire document (512 tokens with 80-token overlap)

if INDEX_FIRST_512:
    # from limited script
    PERSIST_DIR = "./chroma_quest_limited_2"
    COLLECTION_NAME = "quest_documents_limited_2"
    LOG_FILE = "indexing_progress_limited.log"
    CHUNK_TOKENS = 512
    BATCH_SIZE = 2048
    CLEAR_COLLECTION = True
else:
    # from verbose script
    PERSIST_DIR = "./chroma_quest"
    COLLECTION_NAME = "quest_documents"
    LOG_FILE = "indexing_progress.log"
    CHUNK_TOKENS = 512
    OVERLAP_TOKENS = 80
    BATCH_SIZE = 256
    CLEAR_COLLECTION = True

logger = logging.getLogger(__name__)

def normalize_title_slug(s: str) -> str:
    if not s:
        return "untitled"
    t = unicodedata.normalize("NFC", s).strip()
    t = unidecode(t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^A-Za-z0-9 _\-.]", "", t).strip().replace(" ", "_")
    return t or "untitled"

def stable_entity_id(title: str, text: str) -> str:
    slug = normalize_title_slug(title)
    h = hashlib.sha1((title + "\n" + (text or "")).encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{h}"

def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON on line {idx}: {e}")

def build_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def chunk_by_tokens(text: str, tokenizer, chunk_tokens: int, overlap_tokens: int) -> List[str]:
    toks = tokenizer.encode(text, add_special_tokens=False)
    if not toks:
        return []
    chunks = []
    step = max(1, chunk_tokens - overlap_tokens)
    for start in range(0, len(toks), step):
        end = min(start + chunk_tokens, len(toks))
        sub = toks[start:end]
        if not sub:
            break
        chunk_text = tokenizer.decode(sub, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(toks):
            break
    return chunks

class STEmbeddingFn:
    def __init__(self, model_name: str, device: str = None, batch_size: int = 64):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def __call__(self, input: List[str]) -> List[List[float]]:
        if not input:
            return []
        embs = self.model.encode(
            input,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embs.tolist()

    def name(self) -> str:
        return f"sentence-transformers:{self.model_name}"

def upsert_in_batches(collection, ids: List[str], documents: List[str],
                      metadatas: List[Dict], batch_size: int):
    for i in range(0, len(ids), batch_size):
        j = i + batch_size
        collection.upsert(ids=ids[i:j], documents=documents[i:j], metadatas=metadatas[i:j])

def index_jsonl(jsonl_path: str):
    logger.info("Starting indexing process...")
    logger.info(f"Mode: {'LIMITED (first 512 tokens per doc)' if INDEX_FIRST_512 else 'VERBOSE (full chunking)'}")
    logger.info(f"Source file: {jsonl_path}")
    logger.info(f"Chroma DB directory: {PERSIST_DIR}")
    logger.info(f"Collection: {COLLECTION_NAME}")

    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    embed_fn = STEmbeddingFn(model_name=MODEL_NAME, device=DEVICE)

    if CLEAR_COLLECTION:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    tokenizer = build_tokenizer(MODEL_NAME)

    total_docs = sum(1 for _ in read_jsonl(jsonl_path))
    docs_iter = read_jsonl(jsonl_path)
    logger.info(f"Found {total_docs} documents.")

    p_docs = tqdm(total=total_docs, desc="Documents processed", unit="doc")
    p_units = tqdm(total=0, desc="Units indexed", unit="unit")

    batch_ids, batch_docs, batch_metas = [], [], []
    running_total = 0

    interrupted = {"flag": False}

    def handle_sigint(sig, frame):
        interrupted["flag"] = True
        logger.warning("Interrupt received; flushing remaining batch...")

    old_handler = signal.signal(signal.SIGINT, handle_sigint)

    try:
        for raw in docs_iter:
            if interrupted["flag"]:
                break

            title = (raw.get("title") or "").strip() or "untitled"
            text = (raw.get("text") or "").strip()
            entity_id = stable_entity_id(title, text)
            
            # first 512 tokens
            if INDEX_FIRST_512:
                toks = tokenizer.encode(text, add_special_tokens=False)
                truncated = toks[:CHUNK_TOKENS]
                chunk_text = tokenizer.decode(truncated, skip_special_tokens=True).strip()

                if not chunk_text:
                    chunk_text = title

                # limited script used a single ID = entity_id
                batch_ids.append(entity_id)
                batch_docs.append(chunk_text)
                batch_metas.append({
                    "entity_id": entity_id,
                    "title": title,
                    "chunk_index": 0,
                    "n_chunks": 1,
                    "source": os.path.basename(jsonl_path),
                })

                running_total += 1

            # full document
            else:
                chunks = chunk_by_tokens(text, tokenizer, CHUNK_TOKENS, OVERLAP_TOKENS)
                if not chunks:
                    chunks = [title]

                n_chunks = len(chunks)
                for idx, ch in enumerate(chunks):
                    cid = f"{entity_id}__{idx:04d}"
                    batch_ids.append(cid)
                    batch_docs.append(ch)
                    batch_metas.append({
                        "entity_id": entity_id,
                        "title": title,
                        "chunk_index": idx,
                        "n_chunks": n_chunks,
                        "source": os.path.basename(jsonl_path),
                    })

                running_total += n_chunks

            p_docs.update(1)
            p_units.total = running_total
            p_units.refresh()

            if len(batch_ids) >= BATCH_SIZE:
                upsert_in_batches(collection, batch_ids, batch_docs, batch_metas, BATCH_SIZE)
                p_units.update(len(batch_ids))
                batch_ids.clear(); batch_docs.clear(); batch_metas.clear()

        if batch_ids:
            upsert_in_batches(collection, batch_ids, batch_docs, batch_metas, BATCH_SIZE)
            p_units.update(len(batch_ids))

    finally:
        signal.signal(signal.SIGINT, old_handler)
        p_docs.close()
        p_units.close()

    logger.info(f"Done. Indexed {p_units.n} units.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )

    if not os.path.isfile(DOCUMENT_PATH):
        logger.error(f"File not found: {DOCUMENT_PATH}")
        sys.exit(1)

    index_jsonl(DOCUMENT_PATH)
