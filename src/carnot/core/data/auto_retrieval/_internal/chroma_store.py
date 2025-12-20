import chromadb
from chromadb.utils import embedding_functions
import uuid
from typing import List, Dict, Any, Optional

class ChromaStore:
    """    
    Handles:
    - Persistent storage (saved to disk).
    - Automatic embedding generation (via SentenceTransformers).
    - Automatic ID generation (via UUID4) if not provided.
    - Safe metadata updates (Read-Modify-Write).
    """

    def __init__(self, 
                 collection_name: Optional[str] = None, 
                 persist_directory: str = "./chroma_db", 
                 embedding_model_name: str = "BAAI/bge-small-en-v1.5",
                 distance_metric: str = "cosine",
                 collection: Optional[Any] = None):
        """
        Initialize the ChromaDB backend using the v0.6.0+ Configuration API.

        Args:
            collection_name: The unique name of the collection.
            persist_directory: Where to save data on disk.
            embedding_model_name: Name of the SentenceTransformer model (e.g., 'all-MiniLM-L6-v2').
            distance_metric: Distance function: 'cosine', 'l2', or 'ip'.
            collection: Existing collection object (optional).
        """
        if collection is not None:
            self.collection = collection
            if collection_name and collection.name!= collection_name:
                print(f"Warning: Provided name '{collection_name}' matches existing '{collection.name}'")
            return

        if not collection_name:
            raise ValueError("collection_name is required if no existing collection is provided.")

        self.client = chromadb.PersistentClient(path=persist_directory)

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={
                "embedding_function": self.embedding_fn,
                "hnsw": {
                    "space": distance_metric,
                }
            }
        )
        
        print(f"Connected to collection '{collection_name}' at '{persist_directory}'")

    def upsert_documents(self, 
                         documents: List[str], 
                         metadatas: Optional[List[Dict[str, Any]]] = None, 
                         ids: Optional[List[str]] = None,
                         batch_size: int = 5000) -> List[str]:
        """
        Add or update documents. 
        
        - If 'ids' are not provided, UUIDs are automatically generated.
        - If 'ids' exist in the DB, the records are updated.
        - If 'ids' do not exist, new records are created.
        
        Returns:
            List[str]: The list of IDs used (generated or provided).
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        if len(documents) != len(ids):
            raise ValueError(f"Count mismatch: {len(documents)} docs vs {len(ids)} IDs")
        if metadatas and len(metadatas) != len(ids):
            raise ValueError(f"Count mismatch: {len(metadatas)} metadata entries vs {len(ids)} IDs")

        # Chroma's Rust backend enforces a maximum batch size for upsert.
        # To make ingestion robust for large corpora, we automatically chunk.
        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))
            batch_ids = ids[start:end]
            batch_docs = documents[start:end]
            batch_metas = metadatas[start:end] if metadatas is not None else None

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
            )
        
        return ids

    def query(self, 
              query_text: str, 
              n_results: int = 100, 
              where_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        clean_results = []
        
        if results['ids'] and len(results['ids']) > 0:
            ids_list = results['ids'][0]
            docs_list = results['documents'][0] if results['documents'] else []
            metas_list = results['metadatas'][0] if results['metadatas'] else []
            dists_list = results['distances'][0] if results['distances'] else []

            for i in range(len(ids_list)):
                clean_results.append({
                    "id": ids_list[i],
                    "document": docs_list[i] if i < len(docs_list) else None,
                    "metadata": metas_list[i] if i < len(metas_list) else None,
                    "distance": dists_list[i] if i < len(dists_list) else None
                })
        
        return clean_results

    def update_document_metadata(self, doc_id: str, new_metadata: Dict[str, Any], id_field: str = "entity_id"):
        """
        Update metadata for a specific document without overwriting unrelated keys.
        
        Args:
            doc_id: The value of the ID to look for (e.g. the specific entity_id).
            new_metadata: The new metadata fields to merge in.
            id_field: The metadata key to use for looking up the document. 
                      Defaults to 'entity_id'. 
        """
        internal_id = None
        
        results = self.collection.get(
            where={id_field: doc_id},
            include=["metadatas"]
        )
        if not results['ids']:
            raise ValueError(f"Document with {id_field}='{doc_id}' not found.")
        
        if len(results['ids']) > 1:
            raise ValueError(f"Multiple documents found for {id_field}='{doc_id}'.")
        
        internal_id = results['ids'][0]
        current_meta = results['metadatas'][0] if results['metadatas'] and results['metadatas'][0] else {}


        updated_meta = {**current_meta, **new_metadata}
        
        self.collection.update(
            ids=[internal_id],
            metadatas=[updated_meta]
        )

    def delete(self, ids: List[str]):
        """Remove documents by ID."""
        self.collection.delete(ids=ids)

    def count(self) -> int:
        """Get the total number of documents."""
        return self.collection.count()

