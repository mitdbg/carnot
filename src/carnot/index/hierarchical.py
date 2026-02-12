"""Hierarchical (B-tree-like) file index for semantic routing.

Leaves are file summaries; internal nodes summarize their children.
The top level is sized to fit within the router model's context limit.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from carnot.data.item import DataItem
from carnot.utils.hash_helpers import hash_for_id

logger = logging.getLogger(__name__)

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
    children: list["InternalNode"] | None = None  # child InternalNodes when is_leaf_cluster=False


@dataclass
class HierarchicalIndexConfig:
    """Configuration for building the hierarchical index."""

    router_context_limit: int = DEFAULT_ROUTER_CONTEXT_LIMIT
    context_usage_fraction: float = CONTEXT_USAGE_FRACTION
    min_files_for_hierarchy: int = MIN_FILES_FOR_HIERARCHY
    max_children_per_node: int = MAX_CHILDREN_PER_NODE
    embedding_model: str = "openai/text-embedding-3-small"
    summary_model: str = "openai/gpt-4o-mini"
    tokens_per_summary_estimate: int = 80  # rough chars/4 for internal node summaries
    use_llm_routing: bool = True  # use LLM to select nodes when they fit in context
    llm_routing_model: str = "openai/gpt-4o-mini"
    llm_routing_max_nodes: int = 15  # max nodes to send to LLM at once (context limit)


class HierarchicalFileIndex:
    """
    B-tree-like index over file summaries.

    - Leaves: file summaries (path, summary, embedding)
    - Internal nodes: LLM-generated summaries of children
    - Top level sized to fit router context
    - Search: top-down traversal by query embedding similarity
    """

    def __init__(
        self,
        name: str,
        file_summaries: list[FileSummaryEntry],
        config: HierarchicalIndexConfig | None = None,
        api_key: str | None = None,
        build: bool = True,
    ):
        self.name = name
        self.file_summaries = file_summaries
        self.config = config or HierarchicalIndexConfig()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # path -> FileSummaryEntry
        self._path_to_summary: dict[str, FileSummaryEntry] = {e.path: e for e in file_summaries}
        # node_id -> InternalNode (for non-root levels)
        self._nodes: dict[str, InternalNode] = {}
        # Root level: list of InternalNodes (or FileSummaryEntry if flat)
        self._root_level: list[InternalNode | FileSummaryEntry] = []
        self._embeddings: np.ndarray | None = None  # for flat or root-level search

        if build:
            self._build()

    def _max_root_nodes(self) -> int:
        """Max nodes at root level that fit in router context."""
        return max(
            2,
            int(
                self.config.router_context_limit
                * self.config.context_usage_fraction
                / self.config.tokens_per_summary_estimate
            ),
        )

    def _build(self) -> None:
        n = len(self.file_summaries)
        if n == 0:
            return

        if n < self.config.min_files_for_hierarchy:
            # Flat: root is just all file summaries
            self._root_level = self.file_summaries
            self._embeddings = np.array([e.embedding for e in self.file_summaries], dtype="float32")
            return

        # Recursively build hierarchy until top level fits in context
        nodes = self._build_level(self.file_summaries, are_leaves=True)
        self._root_level = nodes
        self._embeddings = np.array([n.embedding for n in nodes], dtype="float32")

    def _build_level(
        self,
        members: list[FileSummaryEntry] | list[InternalNode],
        are_leaves: bool,
    ) -> list[InternalNode]:
        """
        Cluster members and create internal nodes. Recurse if result
        exceeds context limit.
        """
        if not members:
            return []

        embeddings = np.array([m.embedding for m in members], dtype="float32")
        n = len(members)
        max_root = self._max_root_nodes()

        n_clusters = min(
            max(2, n // self.config.max_children_per_node),
            max(2, max_root),
        )

        try:
            import faiss

            d = embeddings.shape[1]
            kmeans = faiss.Kmeans(d, n_clusters, niter=20)
            kmeans.train(embeddings.astype(np.float32))
            _, cluster_ids = kmeans.index.search(embeddings.astype(np.float32), 1)
            cluster_ids = cluster_ids.flatten()
        except (ImportError, Exception) as e:
            logger.warning("faiss clustering failed (%s), using random clustering", e)
            n_clusters = min(n_clusters, n)
            cluster_ids = np.random.randint(0, n_clusters, size=n)

        # Group by cluster
        clusters: dict[int, list] = {}
        for i, m in enumerate(members):
            cid = int(cluster_ids[i])
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(m)

        # Build internal node for each cluster
        level_nodes: list[InternalNode] = []
        for members_in_cluster in clusters.values():
            if not members_in_cluster:
                continue
            if are_leaves:
                summary_text = self._summarize_file_members(members_in_cluster)
                child_paths = [m.path for m in members_in_cluster]
                node = InternalNode(
                    summary=summary_text,
                    embedding=self._embed_texts([summary_text])[0],
                    child_paths=child_paths,
                    is_leaf_cluster=True,
                    children=None,
                )
            else:
                summary_text = self._summarize_internal_members(members_in_cluster)
                child_paths = self._collect_all_paths(members_in_cluster)
                node = InternalNode(
                    summary=summary_text,
                    embedding=self._embed_texts([summary_text])[0],
                    child_paths=child_paths,
                    is_leaf_cluster=False,
                    children=members_in_cluster,
                )
            level_nodes.append(node)

        # Recurse if we still have too many nodes for context
        if len(level_nodes) > max_root:
            return self._build_level(level_nodes, are_leaves=False)
        return level_nodes

    def _collect_all_paths(self, nodes: list[InternalNode]) -> list[str]:
        """Collect all file paths under these internal nodes."""
        paths: list[str] = []
        for n in nodes:
            paths.extend(n.child_paths)
        return paths

    def _summarize_file_members(self, members: list[FileSummaryEntry]) -> str:
        """Summarize a cluster of file summaries."""
        return self._summarize_members(members)

    def _summarize_members(self, members: list[FileSummaryEntry]) -> str:
        """Use LLM to create a concise summary of file summaries in this cluster."""
        if len(members) == 1:
            return members[0].summary

        import litellm

        combined = "\n".join(f"- {m.summary}" for m in members[:20])  # cap for context
        if len(members) > 20:
            combined += f"\n... and {len(members) - 20} more files"

        prompt = f"""Summarize the following file summaries into 2-3 sentences that capture the main themes and would help route queries to these files.

File summaries:
{combined}

Concise cluster summary:"""

        try:
            response = litellm.completion(
                model=self.config.summary_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                api_key=self.api_key,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}, using first summary")
            return members[0].summary

    def _summarize_internal_members(self, members: list[InternalNode]) -> str:
        """Summarize a cluster of internal node summaries (higher-level aggregation)."""
        if len(members) == 1:
            return members[0].summary

        import litellm

        combined = "\n".join(f"- {m.summary}" for m in members[:20])
        if len(members) > 20:
            combined += f"\n... and {len(members) - 20} more clusters"

        prompt = f"""Summarize the following cluster summaries into 2-3 sentences that capture the main themes and would help route queries to relevant files.

Cluster summaries:
{combined}

Concise meta-cluster summary:"""

        try:
            response = litellm.completion(
                model=self.config.summary_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                api_key=self.api_key,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}, using first summary")
            return members[0].summary

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the configured model."""
        import litellm

        response = litellm.embedding(
            model=self.config.embedding_model,
            input=texts,
            api_key=self.api_key,
        )
        return [item["embedding"] for item in response.data]

    def _llm_select_node_indices(
        self,
        query: str,
        nodes: list[InternalNode | FileSummaryEntry],
        top_k: int,
    ) -> list[int]:
        """
        Use LLM to select the most relevant node indices for the query.
        Returns indices into nodes in descending relevance order.
        Falls back to empty list on parse failure (caller should use embedding).
        """
        if not nodes or top_k <= 0:
            return []
        if len(nodes) <= top_k:
            return list(range(len(nodes)))

        import re

        import litellm

        numbered = "\n".join(
            f"{i + 1}. {self._get_node_summary(n)}"
            for i, n in enumerate(nodes)
        )
        prompt = f"""Given the user query, which of the following items are most relevant? Return ONLY the numbers of the top {min(top_k, len(nodes))} most relevant items, in order of relevance, as a comma-separated list (e.g. 3,1,7).

Query: {query}

Items:
{numbered}

Return only the comma-separated numbers, nothing else:"""

        try:
            response = litellm.completion(
                model=self.config.llm_routing_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                api_key=self.api_key,
            )
            text = response.choices[0].message.content.strip()
            # Parse "3, 1, 7" or "3,1,7" or "1. 3 2. 1" etc
            numbers = re.findall(r"\b(\d+)\b", text)
            indices = []
            for n in numbers:
                idx = int(n) - 1  # 1-based in prompt
                if 0 <= idx < len(nodes) and idx not in indices:
                    indices.append(idx)
                if len(indices) >= top_k:
                    break
            return indices
        except Exception as e:
            logger.warning("LLM routing failed: %s", e)
            return []

    def _get_node_summary(self, node: InternalNode | FileSummaryEntry) -> str:
        """Get summary text for a node (truncated if very long)."""
        summary = node.summary
        max_chars = 1200  # ~300 tokens
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "..."
        return summary

    def _order_nodes_by_relevance(
        self,
        query: str,
        query_emb: np.ndarray,
        nodes: list[InternalNode | FileSummaryEntry],
        top_k: int,
    ) -> list[int]:
        """
        Return indices of nodes ordered by relevance (best first).
        Uses LLM when enabled and node count fits; otherwise embedding similarity.
        """
        max_llm = self.config.llm_routing_max_nodes
        if self.config.use_llm_routing and len(nodes) <= max_llm:
            indices = self._llm_select_node_indices(query, nodes, top_k)
            if indices:
                return indices

        # Embedding fallback
        embeddings = np.array([n.embedding for n in nodes], dtype="float32")
        sims = -np.dot(embeddings, query_emb)
        order = np.argsort(sims)
        return order[:top_k].tolist()

    def search(self, query: str, k: int = 50) -> list[str]:
        """
        Return top-k file paths most relevant to the query.

        Uses top-down traversal: LLM or embedding to rank nodes at each level,
        expand best first, recurse until we collect k file paths.
        """
        if not self.file_summaries:
            return []

        k = min(k, len(self.file_summaries))

        if not self._root_level:
            return []

        query_emb = np.array(self._embed_texts([query])[0], dtype="float32")

        if self._embeddings is None:
            return [e.path for e in self.file_summaries[:k]]

        # Order root level by relevance (LLM when fits, else embedding)
        top_n = min(k * 2, len(self._root_level))
        top_indices = self._order_nodes_by_relevance(
            query, query_emb, self._root_level, top_n
        )

        seen_paths: set[str] = set()
        result_paths: list[str] = []
        candidates: list[InternalNode | FileSummaryEntry] = []

        for idx in top_indices:
            node = self._root_level[idx]
            if isinstance(node, FileSummaryEntry):
                if node.path not in seen_paths:
                    seen_paths.add(node.path)
                    result_paths.append(node.path)
            else:
                candidates.append(node)

        # If flat (all FileSummaryEntry), we're done
        if all(isinstance(n, FileSummaryEntry) for n in self._root_level):
            return result_paths[:k]

        # Expand internal nodes in relevance order
        for inode in candidates:
            if len(result_paths) >= k:
                break
            self._collect_paths_from_node(
                inode, query, query_emb, k, seen_paths, result_paths
            )

        return result_paths[:k]

    def _collect_paths_from_node(
        self,
        node: InternalNode,
        query: str,
        query_emb: np.ndarray,
        k: int,
        seen_paths: set[str],
        result_paths: list[str],
    ) -> None:
        """Recursively collect paths from internal node, LLM or embedding for child order."""
        if len(result_paths) >= k:
            return

        if node.is_leaf_cluster:
            for path in node.child_paths:
                if path not in seen_paths:
                    seen_paths.add(path)
                    result_paths.append(path)
                if len(result_paths) >= k:
                    return
            return

        # Nested: order children by relevance (LLM when fits, else embedding)
        if not node.children:
            for path in node.child_paths:
                if path not in seen_paths:
                    seen_paths.add(path)
                    result_paths.append(path)
                if len(result_paths) >= k:
                    return
            return

        child_order = self._order_nodes_by_relevance(
            query, query_emb, node.children, len(node.children)
        )
        for idx in child_order:
            if len(result_paths) >= k:
                break
            self._collect_paths_from_node(
                node.children[idx], query, query_emb, k, seen_paths, result_paths
            )


class FileRouter:
    """
    High-level router that builds a hierarchical index from datasets and routes
    queries to a subset of relevant files.

    Uses persistent caches (per-file summaries, per-path-set index) so that
    once a dataset's files are summarized, they are reused for all future queries.
    """

    def __init__(
        self,
        summary_generator=None,
        config: HierarchicalIndexConfig | None = None,
        api_key: str | None = None,
        storage_dir: Path | None = None,
        use_persistence: bool = True,
    ):
        from carnot.core.data.smv_generator import SMVGenerator

        self.summary_generator = summary_generator or SMVGenerator()
        self.config = config or HierarchicalIndexConfig()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._index_cache: dict[str, HierarchicalFileIndex] = {}  # in-memory fallback
        self.use_persistence = use_persistence
        if use_persistence:
            from carnot.index.persistence import FileSummaryCache, HierarchicalIndexCache

            base = storage_dir or (Path.home() / ".carnot" / "routing")
            if os.getenv("CARNOT_HOME"):
                base = Path(os.getenv("CARNOT_HOME")) / "routing"
            self._summary_cache = FileSummaryCache(base / "summaries")
            self._index_cache_persistent = HierarchicalIndexCache(base / "indices")
        else:
            self._summary_cache = None
            self._index_cache_persistent = None

    def _get_file_text(self, item: DataItem) -> str:
        """Extract text from a DataItem (file path)."""
        try:
            d = item.to_dict()
            return d.get("contents", "") or ""
        except Exception:
            return ""

    def _build_file_summaries(
        self, items: list[DataItem], skip_suffixes: set[str] | None = None
    ) -> list[FileSummaryEntry]:
        """Generate summaries for each file."""
        skip_suffixes = skip_suffixes or {".jpg", ".jpeg", ".png", ".gif", ".zip", ".exe", ".bin"}
        entries: list[FileSummaryEntry] = []

        for item in items:
            if not item.path:
                continue
            suffix = Path(item.path).suffix.lower()
            if suffix in skip_suffixes:
                continue

            try:
                text = self._get_file_text(item)
                if not text.strip():
                    continue

                fs = self.summary_generator.generate_file_summary(
                    file_id=item.path,
                    text_content=text,
                    file_path=item.path,
                )
                emb = fs.summary_embedding
                if emb is None and self.summary_generator.emb_fn:
                    emb = self.summary_generator.emb_fn([fs.global_summary])[0]
                if emb is None:
                    continue  # skip files we can't embed

                entry = FileSummaryEntry(
                    path=item.path,
                    summary=fs.global_summary,
                    embedding=emb,
                )
                entries.append(entry)
                # Save immediately so incremental progress is visible and cost is preserved
                if self._summary_cache:
                    try:
                        self._summary_cache.save(entry)
                    except Exception as e:
                        logger.warning("Failed to persist summary for %s: %s", item.path, e)
            except Exception as e:
                logger.warning(f"Failed to summarize {item.path}: {e}")
                continue

        return entries

    def _get_or_build_summaries(
        self, items: list[DataItem], skip_suffixes: set[str] | None = None
    ) -> list[FileSummaryEntry]:
        """
        Get file summaries, loading from persistent cache when available and
        computing only for missing files.
        """
        skip_suffixes = skip_suffixes or {".jpg", ".jpeg", ".png", ".gif", ".zip", ".exe", ".bin"}
        paths = [
            i.path for i in items
            if i.path and Path(i.path).suffix.lower() not in skip_suffixes
        ]

        if self._summary_cache:
            loaded, missing_paths = self._summary_cache.load_many(paths)
            items_to_compute = [i for i in items if i.path in missing_paths]
        else:
            loaded = {}
            items_to_compute = [i for i in items if i.path and Path(i.path).suffix.lower() not in skip_suffixes]

        # Compute summaries for missing files (saves incrementally inside _build_file_summaries)
        if items_to_compute:
            new_entries = self._build_file_summaries(items_to_compute, skip_suffixes)
            for entry in new_entries:
                loaded[entry.path] = entry

        return list(loaded.values())

    def route(
        self,
        query: str,
        items: list[DataItem],
        k: int = 100,
        index_name: str | None = None,
        min_files_to_route: int = 30,
    ) -> tuple[list[DataItem], HierarchicalFileIndex | None]:
        """
        Route a query to the top-k most relevant files.

        Returns (routed_items, index). routed_items are DataItems in relevance order.
        index is the HierarchicalFileIndex used (or None if routing was skipped/failed).
        If the index cannot be built (e.g. no embeddings), returns (all items, None).
        Skips routing when item count is below min_files_to_route.
        """
        if not items:
            return [], None
        if len(items) < min_files_to_route:
            return list(items), None

        paths = [i.path for i in items if i.path]
        cache_key = index_name or hash_for_id("|".join(sorted(paths)))

        # Try in-memory cache first
        if cache_key in self._index_cache:
            index = self._index_cache[cache_key]
        else:
            index = None

            # Try persistent index cache
            if index is None and self._index_cache_persistent:
                index = self._index_cache_persistent.load(
                    paths, config=self.config, api_key=self.api_key
                )
                if index is not None:
                    logger.debug("Loaded index from persistent cache for %d files", len(paths))

            # Build index if not cached
            if index is None:
                summaries = self._get_or_build_summaries(items)
                if not summaries:
                    logger.warning("No file summaries could be generated, returning all items")
                    return list(items), None

                idx = HierarchicalFileIndex(
                    name=cache_key,
                    file_summaries=summaries,
                    config=self.config,
                    api_key=self.api_key,
                )
                # Save to persistent index cache
                if self._index_cache_persistent:
                    try:
                        self._index_cache_persistent.save(idx)
                    except Exception as e:
                        logger.warning("Failed to persist index cache: %s", e)
                self._index_cache[cache_key] = idx
                index = idx

        paths_result = index.search(query, k=k)
        path_to_item = {item.path: item for item in items if item.path}
        routed = [path_to_item[p] for p in paths_result if p in path_to_item]
        return routed, index
