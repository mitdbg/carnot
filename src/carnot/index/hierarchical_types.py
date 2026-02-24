"""Shared types for hierarchical indexing (no dependencies on persistence or summary_indices)."""

from __future__ import annotations

from dataclasses import dataclass

# Default context limit for router model (tokens) - conservative for gpt-4o-mini
DEFAULT_ROUTER_CONTEXT_LIMIT = 32_000
# Target: top-level summaries should use at most this fraction of context
CONTEXT_USAGE_FRACTION = 0.5
# Min files before we build hierarchy (otherwise use flat search)
MIN_FILES_FOR_HIERARCHY = 20
# Max children per internal node
MAX_CHILDREN_PER_NODE = 10


@dataclass
class FileSummaryEntry:
    """A file with its summary and embedding for indexing."""

    path: str
    summary: str
    embedding: list[float]


@dataclass
class InternalNode:
    """Internal node summarizing a cluster of files or child nodes."""

    summary: str
    embedding: list[float]
    child_paths: list[str]  # file paths when is_leaf_cluster=True
    is_leaf_cluster: bool  # True if direct children are files, False if child InternalNodes
    children: list[InternalNode] | None = None  # child InternalNodes when is_leaf_cluster=False


@dataclass
class HierarchicalIndexConfig:
    """Configuration for building the hierarchical index."""

    router_context_limit: int = DEFAULT_ROUTER_CONTEXT_LIMIT
    context_usage_fraction: float = CONTEXT_USAGE_FRACTION
    min_files_for_hierarchy: int = MIN_FILES_FOR_HIERARCHY
    max_children_per_node: int = MAX_CHILDREN_PER_NODE
    embedding_model: str = "openai/text-embedding-3-small"
    summary_model: str = "openai/gpt-5-mini"
    tokens_per_summary_estimate: int = 80  # rough chars/4 for internal node summaries
    use_llm_routing: bool = True  # use LLM to select nodes when they fit in context
    llm_routing_model: str = "openai/gpt-5-mini"
    llm_routing_max_nodes: int = 15  # max nodes to send to LLM at once (context limit)
