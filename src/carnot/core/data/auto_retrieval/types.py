from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, TypedDict


@dataclass
class Query:
    """User-facing representation of a search query."""
    text: str
    metadata_filters: Optional[Mapping[str, Any]] = None


@dataclass
class SearchResult:
    """Single search result with a score and metadata."""
    doc_id: str
    score: float
    metadata: Optional[Mapping[str, Any]] = None


class SearchError(Exception):
    """Raised when the search API fails or is misconfigured."""
    pass


class DocumentChunk(TypedDict):
    """
    This is NOT an in-memory store—just the schema contract for ingestion/query IO.
    """
    id: str
    text: str
    metadata: Mapping[str, Any]


@dataclass
class VectorQueryResult:
    """
    Structured result from a vector query with optional projections.

    Backends should populate only the requested fields; missing fields are
    represented as None.
    """
    ids: Sequence[str]
    documents: Optional[Sequence[str]] = None
    metadatas: Optional[Sequence[Mapping[str, Any]]] = None
    distances: Optional[Sequence[float]] = None
