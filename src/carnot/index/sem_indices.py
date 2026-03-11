from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from carnot.index.models import (
    FileSummaryEntry,
    HierarchicalIndexConfig,
    InternalNode,
)
from carnot.storage.config import StorageConfig
from carnot.utils.hash_helpers import hash_for_id

if TYPE_CHECKING:
    from carnot.agents.models import LiteLLMModel

logger = logging.getLogger(__name__)


# Max file summaries to send to LLM at once (context limit)
FLAT_INDEX_MAX_LLM_ITEMS = 40

# Number of concurrent workers for parallel cluster summarization
_CLUSTER_SUMMARIZATION_WORKERS = 64

class FlatFileIndex:
    """Single-level index over file summaries for query-time retrieval.

    All file summaries are stored in a flat list.  At query time, if the
    number of summaries fits within the LLM context limit (controlled by
    ``max_llm_items``), they are sent directly to the LLM for top-k
    selection.  Otherwise, embedding cosine similarity pre-filters to
    ``max_llm_items`` candidates, and the LLM selects from that subset.

    Public attributes after construction:

    - ``file_summaries`` (``list[FileSummaryEntry]``): the stored summaries.
    - ``max_llm_items`` (``int``): defaults to ``FLAT_INDEX_MAX_LLM_ITEMS``.

    Representation invariant:
        - ``_embeddings`` is ``None`` iff ``file_summaries`` is empty.
        - When not ``None``, ``_embeddings.shape == (len(file_summaries), embedding_dim)``.
        - ``_model`` is a :class:`LiteLLMModel` instance.
    """

    def __init__(
        self,
        name: str,
        file_summaries: list[FileSummaryEntry],
        model: LiteLLMModel | None = None,
        config: HierarchicalIndexConfig | None = None,
        api_key: str | None = None,
        max_llm_items: int = FLAT_INDEX_MAX_LLM_ITEMS,
    ):
        self.name = name
        self.file_summaries = file_summaries
        self.config = config or HierarchicalIndexConfig()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_llm_items = max_llm_items
        self._embeddings = (
            np.array([e.embedding for e in file_summaries], dtype="float32") if file_summaries else None
        )

        if model is not None:
            self._model = model
        else:
            from carnot.agents.models import LiteLLMModel as _LiteLLMModel

            self._model = _LiteLLMModel(
                model_id=self.config.llm_routing_model,
                api_key=self.api_key,
            )
        self._llm_call_stats: list = []

    @classmethod
    def from_items(
        cls,
        name: str,
        items: list[dict],
        model: LiteLLMModel | None = None,
        config: HierarchicalIndexConfig | None = None,
        api_key: str | None = None,
        storage_dir: Path | None = None,
        summary_layer=None,
    ) -> FlatFileIndex | None:
        """Build a :class:`FlatFileIndex` from dict objects.

        Summarises each item via *summary_layer* (defaulting to a new
        :class:`SummaryLayer`), caches summaries on disk, and constructs
        the index.

        Requires:
            - *items* is a non-empty list of dict instances.

        Returns:
            A :class:`FlatFileIndex`, or ``None`` if no summaries could
            be generated.

        Raises:
            None.  Failures for individual items are logged and skipped.
        """
        if not items:
            return None

        from carnot.index.summary_layer import SummaryLayer

        layer = summary_layer or SummaryLayer(
            model=model, config=config, api_key=api_key, storage_dir=storage_dir,
        )
        summaries = layer.get_or_build_summaries(items)
        if not summaries:
            logger.warning("No file summaries could be generated for FlatFileIndex")
            return None
        index = cls(name=name, file_summaries=summaries, model=model, config=config, api_key=api_key)
        index._llm_call_stats.extend(layer.llm_call_stats)
        return index

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings, embed_stats = self._model.embed(
            texts=texts,
            model=self.config.embedding_model,
        )
        self._llm_call_stats.append(embed_stats)
        return embeddings

    def _llm_select_indices(self, query: str, entries: list[FileSummaryEntry], top_k: int) -> list[int]:
        """Send summaries to LLM; return indices of top-k most relevant."""
        if not entries or top_k <= 0:
            return []
        if len(entries) <= top_k:
            return list(range(len(entries)))

        import re

        max_chars = 1200
        numbered = "\n".join(
            f"{i + 1}. {(e.summary[:max_chars] + '...' if len(e.summary) > max_chars else e.summary)}"
            for i, e in enumerate(entries)
        )
        prompt = f"""Given the user query, which of the following file summaries are most relevant? Return ONLY the numbers of the top {min(top_k, len(entries))} most relevant items, in order of relevance, as a comma-separated list (e.g. 3,1,7).

Query: {query}

File summaries:
{numbered}

Return only the comma-separated numbers, nothing else:"""

        try:
            from carnot.agents.models import ChatMessage

            message = ChatMessage(role="user", content=prompt)
            response = self._model.generate(
                messages=[message],
                temperature=0.1,
            )
            text = response.content.strip()
            numbers = re.findall(r"\b(\d+)\b", text)
            indices = []
            for n in numbers:
                idx = int(n) - 1
                if 0 <= idx < len(entries) and idx not in indices:
                    indices.append(idx)
                if len(indices) >= top_k:
                    break
            return indices
        except Exception as e:
            logger.warning("FlatFileIndex LLM selection failed: %s", e)
            return []

    def search(self, query: str, k: int = 50) -> list[str]:
        """Return top k file paths, uses LLM to pick, pre-filters by embedding when summaries exceed context limit."""
        if not self.file_summaries:
            return []
        k = min(k, len(self.file_summaries))

        if len(self.file_summaries) <= self.max_llm_items:
            candidates = self.file_summaries
        else:
            query_emb = np.array(self._embed_texts([query])[0], dtype="float32")
            sims = -np.dot(self._embeddings, query_emb)
            order = np.argsort(sims)
            top_n = min(self.max_llm_items, len(self.file_summaries))
            indices = order[:top_n].tolist()
            candidates = [self.file_summaries[i] for i in indices]
        indices = self._llm_select_indices(query, candidates, k)
        if not indices:
            query_emb = np.array(self._embed_texts([query])[0], dtype="float32")
            emb = np.array([candidates[i].embedding for i in range(len(candidates))], dtype="float32")
            sims = -np.dot(emb, query_emb)
            indices = np.argsort(sims)[:k].tolist()
        return [candidates[i].path for i in indices]


class HierarchicalFileIndex:
    """B-tree-like index over file summaries.

    - Leaves: file summaries (path, summary, embedding)
    - Internal nodes: LLM-generated summaries of children
    - Top level sized to fit router context
    - Search: top-down traversal by query embedding similarity

    Use ``from_items()`` to build from raw dictionaries (summaries, embeds, caches).

    Construction parameters:

    - *build* (``bool``, default ``True``): if ``True``, ``_build()`` is
      called automatically.  Pass ``False`` to defer building (useful for
      testing or when loading from cache).

    After construction (with ``build=True`` or after calling ``_build()``):

    - ``_path_to_summary``: ``dict[str, FileSummaryEntry]`` mapping each
      file path to its summary entry.
    - ``_root_level``: ``list[InternalNode | FileSummaryEntry]``.
      When the file count is below ``config.min_files_for_hierarchy``,
      the root level is the raw ``FileSummaryEntry`` list (flat mode).
      Otherwise it is a list of ``InternalNode`` objects.
    - ``_embeddings``: ``np.ndarray | None``.  ``None`` only when
      ``file_summaries`` is empty.

    ``_max_root_nodes()`` returns the maximum number of nodes that fit
    in the router context, calculated as::

        max(2, router_context_limit * context_usage_fraction
                / tokens_per_summary_estimate)
    """

    def __init__(
        self,
        name: str,
        file_summaries: list[FileSummaryEntry],
        model: LiteLLMModel | None = None,
        config: HierarchicalIndexConfig | None = None,
        api_key: str | None = None,
        build: bool = True,
        **kwargs
    ):
        self.name = name
        self.file_summaries = file_summaries
        self.config = config or HierarchicalIndexConfig()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if model is not None:
            self._model = model
        else:
            from carnot.agents.models import LiteLLMModel as _LiteLLMModel

            self._model = _LiteLLMModel(
                model_id=self.config.llm_routing_model,
                api_key=self.api_key,
            )

        # path -> FileSummaryEntry
        self._path_to_summary: dict[str, FileSummaryEntry] = {e.path: e for e in file_summaries}
        # node_id -> InternalNode (for non-root levels)
        self._nodes: dict[str, InternalNode] = {}
        # Root level: list of InternalNodes (or FileSummaryEntry if flat)
        self._root_level: list[InternalNode | FileSummaryEntry] = []
        self._embeddings: np.ndarray | None = None  # for flat or root-level search
        self._llm_call_stats: list = []
        self._stats_lock = threading.Lock()

        if build:
            self._build()

    @classmethod
    def from_items(
        cls,
        name: str,
        items: list[dict],
        model: LiteLLMModel | None = None,
        config: HierarchicalIndexConfig | None = None,
        api_key: str | None = None,
        storage_dir: Path | None = None,
        summary_layer=None,
    ) -> HierarchicalFileIndex | None:
        """Build a :class:`HierarchicalFileIndex` from dict objects.

        Summarises each item, caches summaries and the built index on
        disk, and constructs the hierarchy.

        Requires:
            - *items* is a non-empty list of dict instances.

        Returns:
            A :class:`HierarchicalFileIndex`, or ``None`` if no
            summaries could be generated.

        Raises:
            None.  Failures for individual items are logged and skipped.
        """
        if not items:
            return None

        from carnot.index.sem_indices_cache import HierarchicalIndexCache
        from carnot.index.summary_layer import SummaryLayer

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        config = config or HierarchicalIndexConfig()

        storage_config = StorageConfig()
        hier_dir = storage_dir or storage_config.hierarchical_dir
        index_cache = HierarchicalIndexCache(storage_dir=hier_dir)

        paths = [i.get("path") for i in items if i.get("path")]
        cache_key = hash_for_id("|".join(sorted(paths)))

        index = index_cache.load(paths, config=config, api_key=api_key)
        if index is not None:
            logger.debug("Loaded index from cache for %d files", len(paths))
            return index

        layer = summary_layer or SummaryLayer(
            model=model, config=config, api_key=api_key, storage_dir=storage_dir,
        )
        summaries = layer.get_or_build_summaries(items)
        if not summaries:
            logger.warning("No file summaries could be generated")
            return None

        index = cls(
            name=cache_key,
            file_summaries=summaries,
            model=model,
            config=config,
            api_key=api_key,
        )
        index._llm_call_stats.extend(layer.llm_call_stats)
        try:
            index_cache.save(index)
        except Exception as e:
            logger.warning("Failed to persist index: %s", e)
        return index

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

        Uses parallel processing for cluster summarization.
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

        # Filter out empty clusters
        cluster_list = [c for c in clusters.values() if c]

        # Build internal nodes in parallel
        level_nodes: list[InternalNode] = []
        with ThreadPoolExecutor(max_workers=_CLUSTER_SUMMARIZATION_WORKERS) as executor:
            futures = {
                executor.submit(self._build_cluster_node, members_in_cluster, are_leaves): idx
                for idx, members_in_cluster in enumerate(cluster_list)
            }

            for future in as_completed(futures):
                node = future.result()
                if node is not None:
                    level_nodes.append(node)

        # Recurse if we still have too many nodes for context
        if len(level_nodes) > max_root:
            return self._build_level(level_nodes, are_leaves=False)
        return level_nodes

    def _build_cluster_node(
        self,
        members_in_cluster: list,
        are_leaves: bool,
    ) -> InternalNode | None:
        """Build a single internal node for a cluster. Thread-safe helper."""
        try:
            if are_leaves:
                summary_text = self._summarize_file_members(members_in_cluster)
                child_paths = [m.path for m in members_in_cluster]
                return InternalNode(
                    summary=summary_text,
                    embedding=self._embed_texts([summary_text])[0],
                    child_paths=child_paths,
                    is_leaf_cluster=True,
                    children=None,
                )
            else:
                summary_text = self._summarize_internal_members(members_in_cluster)
                child_paths = self._collect_all_paths(members_in_cluster)
                return InternalNode(
                    summary=summary_text,
                    embedding=self._embed_texts([summary_text])[0],
                    child_paths=child_paths,
                    is_leaf_cluster=False,
                    children=members_in_cluster,
                )
        except Exception as e:
            logger.warning(f"Failed to build cluster node: {e}")
            return None

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

        combined = "\n".join(f"- {m.summary}" for m in members[:20])  # cap for context
        if len(members) > 20:
            combined += f"\n... and {len(members) - 20} more files"

        prompt = f"""Summarize the following file summaries into 2-3 sentences that capture the main themes and would help route queries to these files.

File summaries:
{combined}

Concise cluster summary:"""

        try:
            from carnot.agents.models import ChatMessage

            message = ChatMessage(role="user", content=prompt)
            response = self._model.generate(
                messages=[message],
                temperature=0.2,
            )
            if response.llm_call_stats is not None:
                with self._stats_lock:
                    self._llm_call_stats.append(response.llm_call_stats)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}, using first summary")
            return members[0].summary

    def _summarize_internal_members(self, members: list[InternalNode]) -> str:
        """Summarize a cluster of internal node summaries (higher-level aggregation)."""
        if len(members) == 1:
            return members[0].summary

        combined = "\n".join(f"- {m.summary}" for m in members[:20])
        if len(members) > 20:
            combined += f"\n... and {len(members) - 20} more clusters"

        prompt = f"""Summarize the following cluster summaries into 2-3 sentences that capture the main themes and would help route queries to relevant files.

Cluster summaries:
{combined}

Concise meta-cluster summary:"""

        try:
            from carnot.agents.models import ChatMessage

            message = ChatMessage(role="user", content=prompt)
            response = self._model.generate(
                messages=[message],
                temperature=0.2,
            )
            if response.llm_call_stats is not None:
                with self._stats_lock:
                    self._llm_call_stats.append(response.llm_call_stats)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}, using first summary")
            return members[0].summary

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the configured model."""
        embeddings, embed_stats = self._model.embed(
            texts=texts,
            model=self.config.embedding_model,
        )
        with self._stats_lock:
            self._llm_call_stats.append(embed_stats)
        return embeddings

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
            from carnot.agents.models import ChatMessage

            message = ChatMessage(role="user", content=prompt)
            response = self._model.generate(
                messages=[message],
                temperature=0.1,
            )
            text = response.content.strip()
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
