from __future__ import annotations

from typing import Any, List, Dict, Optional
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    """A single chunk of text with its embedding."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="The text content of the chunk")
    start_char_idx: int = Field(..., description="Starting character index in the original file")
    end_char_idx: int = Field(..., description="Ending character index in the original file")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the chunk text")

class ChunkIndex(BaseModel):
    """Materialized view for vector search over file contents."""
    file_id: str = Field(..., description="ID of the source file")
    chunks: List[Chunk] = Field(default_factory=list, description="List of chunks for this file")
    model_name: str = Field(..., description="Name of the embedding model used")

class FileSummary(BaseModel):
    """Materialized view for high-level file understanding."""
    file_id: str = Field(..., description="ID of the source file")
    global_summary: str = Field(..., description="Concise summary of the entire file")
    dense_blocks: List[str] = Field(default_factory=list, description="Key extracted sections (e.g. Abstract, Conclusion)")
    summary_embedding: Optional[List[float]] = Field(None, description="Vector embedding of the global summary")

class TaggedFiles(BaseModel):
    """Materialized view for generalized topic tagging (solving the 'Ice Cream vs Waffles' reuse problem)."""
    file_id: str = Field(..., description="ID of the source file")
    tags: List[str] = Field(default_factory=list, description="List of generalized topics this file answers (e.g. 'Q4 Sales Data')")
    provenance_queries: List[str] = Field(default_factory=list, description="List of original queries that led to these tags")

class MetadataRegistry(BaseModel):
    """Materialized view for fast filtering and heuristics."""
    file_id: str = Field(..., description="ID of the source file")
    file_size_bytes: int = Field(..., description="Size of the file in bytes")
    file_type: str = Field(..., description="Extension or MIME type")
    creation_date: str = Field(..., description="Creation date string")
    topic_clusters: List[str] = Field(default_factory=list, description="Assigned topic tags or cluster IDs")
    quality_score: float = Field(0.0, description="Heuristic score of information density (0-1)")
