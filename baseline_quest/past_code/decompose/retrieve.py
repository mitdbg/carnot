import os
import sys
from typing import Dict, List, Set

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
PERSIST_DIR = os.path.join(project_root, "chroma_quest_limited_2")

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from sentence_transformers import SentenceTransformer

# --- Configuration ---
RETRIEVER_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# PERSIST_DIR = "./chroma_quest"
COLLECTION_NAME = "quest_documents_limited_2"
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# --- Global Variables for Singleton Pattern ---
retriever_model = None
collection = None

def initialize_retriever():
    """
    Initializes the SentenceTransformer model and ChromaDB client.
    Uses a singleton pattern to avoid reloading models on subsequent calls.
    """
    global retriever_model, collection
    if retriever_model and collection:
        return

    print("--- Initializing Retriever (ChromaDB and bge-small) ---")
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError(f"ChromaDB persistence directory not found at '{PERSIST_DIR}'.")
    
    retriever_model = SentenceTransformer(RETRIEVER_MODEL_NAME, device=DEVICE)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    print("--- Retriever Initialized ---")

# --- Heuristic for over-retrieval ---
# To find K unique documents, we must retrieve more than K chunks.
# A multiplier of 10 is a common, conservative choice for dense vector stores.
CHUNK_MULTIPLIER = 10 

def retrieve(query: str, k: int) -> Set[str]:
    """
    Retrieves enough chunks to find k unique documents for a single query and 
    returns the set of those unique document IDs (titles).

    Args:
        query (str): The query text to search for.
        k (int): The target number of unique documents (titles) to return.

    Returns:
        A set of unique document IDs (titles) that contain the retrieved chunks.
    """
    if not retriever_model or not collection:
        raise RuntimeError("Retriever not initialized. Call initialize_retriever() first.")
    
    if k <= 0:
        return set()

    # Calculate the number of chunks to retrieve (over-retrieve)
    n_chunks = k * CHUNK_MULTIPLIER
    
    query_embedding = retriever_model.encode(
        query, convert_to_numpy=True, normalize_embeddings=True
    ).tolist()

    # Query the collection with a large n_results
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=n_chunks, 
        include=["metadatas"] # Only retrieve metadatas
    )
    
    unique_doc_titles = set()
    
    # Extract unique document titles until we reach the target k
    if results.get("metadatas") and results["metadatas"][0]:
        chunk_metadatas = results["metadatas"][0]
        for meta in chunk_metadatas:
            # Assumes the document ID is stored in the 'title' key of the metadata
            if "title" in meta:
                unique_doc_titles.add(meta["title"])
                # Stop as soon as we have enough unique documents
                if len(unique_doc_titles) >= k:
                    break
        
    return unique_doc_titles