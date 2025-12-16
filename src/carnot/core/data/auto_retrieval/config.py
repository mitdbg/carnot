from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import yaml

@dataclass
class Config:
    """Top-level configuration for SearchClient and internal modules."""
    # Chroma settings
    chroma_persist_dir: str
    chroma_collection_name: str
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    clear_chroma_collection: bool = False
    
    # Data settings
    quest_documents_path: str = ""
    
    # Chunking settings
    index_first_512: bool = True
    chunk_size: int = 512
    overlap: int = 80
    tokenizer_model: str = "BAAI/bge-small-en-v1.5"
    
    # Batching
    batch_size: int = 64
    
    # Concept generation
    concept_generation_mode: str = "two_stage"
    concept_cluster_count: int = 50
    concept_embedding_model: str = "all-MiniLM-L6-v2"

    # Other
    dataset_name: str = "quest"
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        # Filter for known fields to avoid errors with extra config keys
        known_fields = cls.__annotations__.keys()
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

def load_config(path: str) -> Config:
    """Load a Config object from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config.from_dict(raw)
