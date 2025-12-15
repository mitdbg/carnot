import os
import re
import json
import hashlib
import unicodedata
import chromadb
import logging
from typing import List, Dict, Iterable
from unidecode import unidecode
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# --- Text Processing Utils ---

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

# --- ChromaDB & Embedding Utils ---

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

def get_db_collection(config: Dict, clear_existing: bool = False):
    """
    Initializes ChromaDB client and returns the collection based on config.
    """
    # Extract config values
    persist_dir = config['indexing']['chroma']['persist_dir']
    collection_name = config['indexing']['chroma']['collection']
    model_name = config['indexing']['embedding_model']
    
    # Ensure directory exists
    os.makedirs(persist_dir, exist_ok=True)
    
    # Connect
    client = chromadb.PersistentClient(path=persist_dir)
    embed_fn = STEmbeddingFn(model_name=model_name)

    if clear_existing:
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection '{collection_name}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
    )
    
    return collection