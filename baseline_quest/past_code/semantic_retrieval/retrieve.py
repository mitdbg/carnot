# =========================
# SQLite compatibility
# =========================
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    print("pysqlite3 not found, using system sqlite3. This might fail on some systems.")

import os
import json
from typing import Iterable, Dict
from tqdm import tqdm

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
INDEX_FIRST_512 = True
INCLUDE_CHUNKS = False
DEVICE = None   # or "cuda"

if INDEX_FIRST_512:
    PERSIST_DIR = "./chroma_quest_limited"
    COLLECTION_NAME = "quest_documents_limited"
    INPUT_QUERIES_PATH = "data/train_subset2.jsonl"
    OUTPUT_PREDICTIONS_PATH = "pred_unranked_limited.jsonl"
else:
    PERSIST_DIR = "./chroma_quest"
    COLLECTION_NAME = "quest_documents"
    INPUT_QUERIES_PATH = "data/train_subset2.jsonl"
    OUTPUT_PREDICTIONS_PATH = "pred_unranked.jsonl"

def read_jsonl(path: str) -> Iterable[Dict]:
    """Reads a JSONL file line by line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def retrieve(queries_path: str, output_path: str):
    print("\n=== Retrieval Mode ===")
    print(f"INDEX_FIRST_512 = {INDEX_FIRST_512}")
    print(f"INCLUDE_CHUNKS = {INCLUDE_CHUNKS}")
    print(f"Using ChromaDB directory: {PERSIST_DIR}")

    if not os.path.exists(PERSIST_DIR):
        print(f"Error: ChromaDB directory '{PERSIST_DIR}' not found.")
        print("Please run the indexing step first.")
        return

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Loaded Chroma collection '{COLLECTION_NAME}' with {collection.count()} chunks.")

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(f"Loaded embedding model: {MODEL_NAME}")

    queries = list(read_jsonl(queries_path))
    print(f"Found {len(queries)} queries from {queries_path}")

    n_to_retrieve = 100 if INDEX_FIRST_512 else 200

    include_fields = ["metadatas"]
    if INCLUDE_CHUNKS:
        include_fields.append("documents")

    with open(output_path, "w", encoding="utf-8") as f_out:

        for item in tqdm(queries, desc="Retrieving…"):
            text = item["query"]

            q_emb = model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).tolist()

            results = collection.query(
                query_embeddings=[q_emb],
                n_results=n_to_retrieve,
                include=include_fields
            )

            metas = results["metadatas"][0]
            docs_text = results["documents"][0] if INCLUDE_CHUNKS else None

            # Case 1: first 512 tokens, titles only
            if INDEX_FIRST_512 and not INCLUDE_CHUNKS:
                top_docs = [m["title"] for m in metas][:100]

            # Case 2: first 512 tokens, return titles and text chunks
            elif INDEX_FIRST_512 and INCLUDE_CHUNKS:
                top_docs = [
                    {"title": m.get("title", "No Title"), "chunk": chunk}
                    for m, chunk in zip(metas, docs_text)
                ]

            # Case 3: full document, titles only (deduplicated)
            elif not INDEX_FIRST_512 and not INCLUDE_CHUNKS:
                seen = set()
                dedup_titles = []
                for m in metas:
                    t = m["title"]
                    if t not in seen:
                        seen.add(t)
                        dedup_titles.append(t)
                top_docs = dedup_titles[:100]

            # Case 4: full index, titles and chunks
            else:
                top_docs = [
                    {"title": m.get("title", "No Title"), "chunk": chunk}
                    for m, chunk in zip(metas, docs_text)
                ]

            f_out.write(json.dumps({
                "query": text,
                "docs": top_docs
            }) + "\n")

    print(f"\n--- Done ---")
    print(f"Output written to: {output_path}")

if __name__ == "__main__":
    if not os.path.isfile(INPUT_QUERIES_PATH):
        print(f"Error: Missing query file: {INPUT_QUERIES_PATH}")
    else:
        retrieve(INPUT_QUERIES_PATH, OUTPUT_PREDICTIONS_PATH)
