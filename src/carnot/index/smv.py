"""Structured metadata and summaries for files (SMV - Structured Metadata Views)."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A chunk of text from a file, optionally with embedding."""

    chunk_id: str
    text: str
    start_char_idx: int
    end_char_idx: int
    embedding: list[float] | None = None


@dataclass
class ChunkIndex:
    """Index of chunks for a single file."""

    file_id: str
    chunks: list[Chunk]
    model_name: str


@dataclass
class FileSummary:
    """Summary of a file, optimized for routing and search."""

    file_id: str
    global_summary: str
    dense_blocks: list[Any] = field(default_factory=list)
    summary_embedding: list[float] | None = None


@dataclass
class TaggedFiles:
    """Collection of files with tags."""

    file_ids: list[str] = field(default_factory=list)
    tags: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class MetadataRegistry:
    """Registry of file metadata."""

    file_id: str
    file_size_bytes: int
    file_type: str
    creation_date: str
    topic_clusters: list[str] = field(default_factory=list)
    quality_score: float = 0.0
