"""Persistent caching for the hierarchical file index.

- Per-file summary cache: once a file is summarized, reuse for all future queries
- Per-path-set index cache: reuse built index when the same file set is queried
"""

from __future__ import annotations

import base64
import json
import logging
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
    from carnot.index.sem_indices import HierarchicalFileIndex

logger = logging.getLogger(__name__)


def _embedding_to_json_safe(emb) -> list:
    """Convert embedding (ndarray or list) to JSON-serializable list."""
    if hasattr(emb, "tolist"):
        return emb.tolist()
    return emb


class FileSummaryCache:
    """Persistent cache for per-file summaries (path, summary, embedding).

    Each summary is stored as a JSON file in ``storage_dir``, keyed by a
    hash of the file path.

    - ``save(entry)`` writes the summary to disk.
    - ``load(path)`` returns a :class:`FileSummaryEntry` or ``None`` if
      not found or corrupt.
    - ``load_many(paths)`` returns ``(dict[path, entry], list[missing_paths])``.

    Parameters
    ----------
    storage_dir:
        Explicit directory override. When *None*, uses ``StorageConfig().summaries_dir``.
    config:
        A :class:`StorageConfig` instance. Used only when *storage_dir* is not supplied.
    """

    def __init__(
        self,
        storage_dir: Path | None = None,
        config: StorageConfig | None = None,
    ):
        if storage_dir is not None:
            self.storage_dir = storage_dir
        else:
            self.storage_dir = (config or StorageConfig()).summaries_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _path_to_key(self, path: str) -> str:
        """Convert file path to cache filename (safe for filesystem)."""
        return hash_for_id(path, max_chars=32) + ".json"

    def load(self, path: str) -> FileSummaryEntry | None:
        """Load cached summary for a file path. Returns None if not found."""
        key = self._path_to_key(path)
        filepath = self.storage_dir / key
        if not filepath.exists():
            return None
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("path") != path:
                # Hash collision - path mismatch
                return None
            return FileSummaryEntry(
                path=data["path"],
                summary=data["summary"],
                embedding=data["embedding"],
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Corrupt cache for %s: %s", path, e)
            return None

    def save(self, entry: FileSummaryEntry) -> None:
        """Save a file summary to the cache."""
        key = self._path_to_key(entry.path)
        filepath = self.storage_dir / key
        data = {
            "path": entry.path,
            "summary": entry.summary,
            "embedding": _embedding_to_json_safe(entry.embedding),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=0)

    def load_many(
        self, paths: list[str]
    ) -> tuple[dict[str, FileSummaryEntry], list[str]]:
        """
        Load cached summaries for multiple paths.
        Returns (path -> entry for loaded, list of missing paths).
        """
        loaded: dict[str, FileSummaryEntry] = {}
        missing: list[str] = []
        for p in paths:
            entry = self.load(p)
            if entry is not None:
                loaded[p] = entry
            else:
                missing.append(p)
        return loaded, missing


class HierarchicalIndexCache:
    """Persistent cache for built HierarchicalFileIndex by path set.

    Parameters
    ----------
    storage_dir:
        Explicit directory override. When *None*, uses
        ``StorageConfig().hierarchical_dir``.
    config:
        A :class:`StorageConfig` instance. Used only when *storage_dir*
        is not supplied.
    """

    def __init__(
        self,
        storage_dir: Path | None = None,
        config: StorageConfig | None = None,
    ):
        if storage_dir is not None:
            self.storage_dir = storage_dir
        else:
            self.storage_dir = (config or StorageConfig()).hierarchical_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _path_set_to_key(self, paths: list[str]) -> str:
        """Hash of sorted path set for cache key."""
        return hash_for_id("|".join(sorted(p for p in paths if p)), max_chars=32)

    def load(
        self,
        paths: list[str],
        config: HierarchicalIndexConfig | None = None,
        api_key: str | None = None,
    ) -> HierarchicalFileIndex | None:
        """Load a cached index for the given path set. Returns None if not found."""
        key = self._path_set_to_key(paths)
        filepath = self.storage_dir / f"{key}.json"
        if not filepath.exists():
            return None
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            return self._deserialize(data, config, api_key)
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.warning("Failed to load index cache %s: %s", key, e)
            return None

    def save(self, index: HierarchicalFileIndex) -> None:
        """Save an index to the cache (keyed by its path set)."""
        paths = [e.path for e in index.file_summaries]
        key = self._path_set_to_key(paths)
        filepath = self.storage_dir / f"{key}.json"
        data = self._serialize(index)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=0)

    def _serialize(self, index: HierarchicalFileIndex) -> dict:
        """Serialize index to JSON-safe dict, including nested tree structure."""
        file_summaries = [
            {
                "path": e.path,
                "summary": e.summary,
                "embedding": _embedding_to_json_safe(e.embedding),
            }
            for e in index.file_summaries
        ]
        # Serialize root level (recursively for nested InternalNodes)
        root_serialized = []
        for node in index._root_level:
            if hasattr(node, "path"):
                # FileSummaryEntry
                root_serialized.append(
                    {
                        "type": "file",
                        "path": node.path,
                        "summary": node.summary,
                        "embedding": _embedding_to_json_safe(node.embedding),
                    }
                )
            else:
                root_serialized.append(self._serialize_internal_node(node))
        embeddings_b64 = None
        if index._embeddings is not None:
            embeddings_b64 = base64.b64encode(index._embeddings.tobytes()).decode("ascii")
            embeddings_shape = list(index._embeddings.shape)
            embeddings_dtype = str(index._embeddings.dtype)
        else:
            embeddings_shape = []
            embeddings_dtype = "float32"
        return {
            "file_summaries": file_summaries,
            "root_level": root_serialized,
            "embeddings_b64": embeddings_b64,
            "embeddings_shape": embeddings_shape,
            "embeddings_dtype": embeddings_dtype,
        }

    def _serialize_internal_node(self, node: InternalNode) -> dict:
        """Recursively serialize an InternalNode and its children."""
        children_serialized = None
        if node.children:
            children_serialized = [
                self._serialize_internal_node(c) for c in node.children
            ]
        return {
            "type": "internal",
            "summary": node.summary,
            "embedding": _embedding_to_json_safe(node.embedding),
            "child_paths": node.child_paths,
            "is_leaf_cluster": node.is_leaf_cluster,
            "children": children_serialized,
        }

    def _deserialize(
        self,
        data: dict,
        config: HierarchicalIndexConfig | None = None,
        api_key: str | None = None,
    ) -> HierarchicalFileIndex:
        """Deserialize index from dict."""
        from carnot.index.sem_indices import HierarchicalFileIndex

        file_summaries = [
            FileSummaryEntry(
                path=e["path"],
                summary=e["summary"],
                embedding=e["embedding"],
            )
            for e in data["file_summaries"]
        ]
        path_to_summary = {e.path: e for e in file_summaries}
        root_level = []
        for node_data in data["root_level"]:
            if node_data["type"] == "file":
                root_level.append(path_to_summary[node_data["path"]])
            else:
                root_level.append(self._deserialize_internal_node(node_data))
        if data.get("embeddings_b64"):
            arr = np.frombuffer(
                base64.b64decode(data["embeddings_b64"]),
                dtype=np.dtype(data.get("embeddings_dtype", "float32")),
            )
            shape = data.get("embeddings_shape", [])
            if shape:
                arr = arr.reshape(shape)
        else:
            arr = None
        index = HierarchicalFileIndex(
            name="",
            file_summaries=file_summaries,
            config=config or HierarchicalIndexConfig(),
            api_key=api_key,
            build=False,
        )
        index._root_level = root_level
        index._embeddings = arr
        index._path_to_summary = path_to_summary
        return index

    def _deserialize_internal_node(self, data: dict) -> InternalNode:
        """Recursively deserialize an InternalNode and its children."""
        children = None
        if data.get("children"):
            children = [self._deserialize_internal_node(c) for c in data["children"]]
        return InternalNode(
            summary=data["summary"],
            embedding=data["embedding"],
            child_paths=data["child_paths"],
            is_leaf_cluster=data["is_leaf_cluster"],
            children=children,
        )
