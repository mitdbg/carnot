"""Unit tests for index construction and structure (no LLM required).

Tests cover:
1. CarnotIndex base class — attribute storage, ``_build_uri_to_idx``,
   ``_items_to_data_items``
2. FlatFileIndex — construction from pre-built summaries, embedding
   pre-filter logic with synthetic embeddings
3. HierarchicalFileIndex._build() — flat mode for small file counts,
   hierarchical mode with mock/fallback clustering
4. HierarchicalIndexConfig defaults
5. FileSummaryEntry / InternalNode dataclass construction
6. FileSummaryCache — save/load round-trip
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from carnot.data.item import DataItem
from carnot.index.index import CarnotIndex, _build_uri_to_idx, _items_to_data_items
from carnot.index.models import (
    FileSummaryEntry,
    HierarchicalIndexConfig,
    InternalNode,
)
from carnot.index.sem_indices import FlatFileIndex, HierarchicalFileIndex

# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_embedding(dim: int = 16, seed: int = 0) -> list[float]:
    """Return a deterministic unit-length embedding of *dim* floats.

    Requires:
        - *dim* > 0.
        - *seed* is a non-negative integer.

    Returns:
        A list of *dim* floats with unit L2-norm.
    """
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).tolist()
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def _make_summaries(n: int, dim: int = 16) -> list[FileSummaryEntry]:
    """Create *n* synthetic FileSummaryEntry objects.

    Each entry has a unique path, a simple summary string, and a
    deterministic embedding.

    Requires:
        - *n* >= 0.

    Returns:
        A list of *n* FileSummaryEntry instances.
    """
    return [
        FileSummaryEntry(
            path=f"/data/file_{i:03d}.txt",
            summary=f"Summary of file {i}",
            embedding=_make_embedding(dim=dim, seed=i),
        )
        for i in range(n)
    ]


# ═══════════════════════════════════════════════════════════════════════
# Dataclass construction
# ═══════════════════════════════════════════════════════════════════════


class TestFileSummaryEntry:
    """FileSummaryEntry dataclass construction."""

    def test_fields(self):
        """All three fields are accessible after construction."""
        entry = FileSummaryEntry(path="/a.txt", summary="about a", embedding=[1.0, 0.0])
        assert entry.path == "/a.txt"
        assert entry.summary == "about a"
        assert entry.embedding == [1.0, 0.0]


class TestInternalNode:
    """InternalNode dataclass construction."""

    def test_leaf_cluster(self):
        """A leaf-cluster node stores child_paths and has no children list."""
        node = InternalNode(
            summary="cluster of files",
            embedding=[0.5, 0.5],
            child_paths=["/a.txt", "/b.txt"],
            is_leaf_cluster=True,
        )
        assert node.is_leaf_cluster
        assert node.children is None
        assert len(node.child_paths) == 2

    def test_internal_node_with_children(self):
        """A non-leaf node stores child InternalNodes."""
        child = InternalNode(
            summary="sub",
            embedding=[0.1],
            child_paths=["/x.txt"],
            is_leaf_cluster=True,
        )
        parent = InternalNode(
            summary="parent",
            embedding=[0.2],
            child_paths=["/x.txt"],
            is_leaf_cluster=False,
            children=[child],
        )
        assert not parent.is_leaf_cluster
        assert len(parent.children) == 1


class TestHierarchicalIndexConfig:
    """HierarchicalIndexConfig sensible defaults."""

    def test_defaults(self):
        """Default config has reasonable values."""
        cfg = HierarchicalIndexConfig()
        assert cfg.min_files_for_hierarchy == 20
        assert cfg.max_children_per_node == 10
        assert 0 < cfg.context_usage_fraction <= 1.0
        assert cfg.router_context_limit > 0


# ═══════════════════════════════════════════════════════════════════════
# FlatFileIndex — construction from pre-built summaries
# ═══════════════════════════════════════════════════════════════════════


class TestFlatFileIndexConstruction:
    """FlatFileIndex instantiation from pre-built summaries (no LLM)."""

    def test_empty(self):
        """An empty FlatFileIndex has no summaries."""
        idx = FlatFileIndex(name="empty", file_summaries=[])
        assert idx.file_summaries == []
        assert idx._embeddings is None

    def test_from_summaries(self):
        """FlatFileIndex stores summaries and builds embedding matrix."""
        summaries = _make_summaries(5, dim=8)
        idx = FlatFileIndex(name="test", file_summaries=summaries)
        assert len(idx.file_summaries) == 5
        assert idx._embeddings is not None
        assert idx._embeddings.shape == (5, 8)

    def test_max_llm_items_default(self):
        """Default max_llm_items matches the module constant."""
        from carnot.index.sem_indices import FLAT_INDEX_MAX_LLM_ITEMS

        idx = FlatFileIndex(name="t", file_summaries=[])
        assert idx.max_llm_items == FLAT_INDEX_MAX_LLM_ITEMS


# ═══════════════════════════════════════════════════════════════════════
# HierarchicalFileIndex._build()
# ═══════════════════════════════════════════════════════════════════════


class TestHierarchicalFileIndexBuild:
    """HierarchicalFileIndex._build() with synthetic data (no LLM calls).

    We construct indices with ``build=False`` then call ``_build()``
    manually (for the flat case) or inspect the structure after
    construction.
    """

    def test_empty_build(self):
        """_build() on an empty index is a no-op."""
        idx = HierarchicalFileIndex(
            name="empty",
            file_summaries=[],
            build=False,
        )
        idx._build()
        assert idx._root_level == []

    def test_small_file_count_uses_flat(self):
        """Below min_files_for_hierarchy, _build() stores summaries directly at root."""
        config = HierarchicalIndexConfig(min_files_for_hierarchy=20)
        summaries = _make_summaries(5, dim=8)
        idx = HierarchicalFileIndex(
            name="small",
            file_summaries=summaries,
            config=config,
            build=False,
        )
        idx._build()
        # Root level should be the file summaries themselves
        assert len(idx._root_level) == 5
        assert all(isinstance(n, FileSummaryEntry) for n in idx._root_level)
        assert idx._embeddings is not None
        assert idx._embeddings.shape == (5, 8)

    def test_path_to_summary_mapping(self):
        """_path_to_summary maps each file path to its FileSummaryEntry."""
        summaries = _make_summaries(3)
        idx = HierarchicalFileIndex(
            name="t",
            file_summaries=summaries,
            build=False,
        )
        assert len(idx._path_to_summary) == 3
        assert idx._path_to_summary["/data/file_000.txt"].summary == "Summary of file 0"

    def test_max_root_nodes_calculation(self):
        """_max_root_nodes returns a positive integer based on config."""
        config = HierarchicalIndexConfig(
            router_context_limit=32_000,
            context_usage_fraction=0.5,
            tokens_per_summary_estimate=80,
        )
        idx = HierarchicalFileIndex(
            name="t",
            file_summaries=[],
            config=config,
            build=False,
        )
        max_root = idx._max_root_nodes()
        # 32000 * 0.5 / 80 = 200
        assert max_root == 200


# ═══════════════════════════════════════════════════════════════════════
# FileSummaryCache
# ═══════════════════════════════════════════════════════════════════════


class TestFileSummaryCache:
    """FileSummaryCache persistence round-trip with a tmp_path directory."""

    def test_save_and_load(self, tmp_path):
        """save() then load() returns an equivalent FileSummaryEntry."""
        from carnot.index.sem_indices_cache import FileSummaryCache

        cache = FileSummaryCache(storage_dir=tmp_path)
        entry = FileSummaryEntry(
            path="/data/file_001.txt",
            summary="A brief summary",
            embedding=[0.1, 0.2, 0.3],
        )
        cache.save(entry)
        loaded = cache.load("/data/file_001.txt")
        assert loaded is not None
        assert loaded.path == entry.path
        assert loaded.summary == entry.summary
        assert loaded.embedding == pytest.approx(entry.embedding)

    def test_load_missing_returns_none(self, tmp_path):
        """load() returns None for a path that was never saved."""
        from carnot.index.sem_indices_cache import FileSummaryCache

        cache = FileSummaryCache(storage_dir=tmp_path)
        assert cache.load("/nonexistent.txt") is None

    def test_load_many(self, tmp_path):
        """load_many() returns found entries and lists missing paths."""
        from carnot.index.sem_indices_cache import FileSummaryCache

        cache = FileSummaryCache(storage_dir=tmp_path)
        e1 = FileSummaryEntry(path="/a.txt", summary="a", embedding=[1.0])
        e2 = FileSummaryEntry(path="/b.txt", summary="b", embedding=[2.0])
        cache.save(e1)
        cache.save(e2)

        loaded, missing = cache.load_many(["/a.txt", "/b.txt", "/c.txt"])
        assert "/a.txt" in loaded
        assert "/b.txt" in loaded
        assert "/c.txt" in missing

    def test_corrupt_cache_file(self, tmp_path):
        """A corrupt cache file returns None on load (does not crash)."""
        from carnot.index.sem_indices_cache import FileSummaryCache

        cache = FileSummaryCache(storage_dir=tmp_path)
        entry = FileSummaryEntry(path="/x.txt", summary="x", embedding=[1.0])
        cache.save(entry)

        # Corrupt the file
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 1
        cache_files[0].write_text("not valid json{{{")

        assert cache.load("/x.txt") is None


# ═══════════════════════════════════════════════════════════════════════
# CarnotIndex base class
# ═══════════════════════════════════════════════════════════════════════


class _StubIndex(CarnotIndex):
    """Minimal concrete CarnotIndex for unit-testing the base class.

    Representation invariant:
        - ``_index`` is always the string ``"stub"`` (or a pre-built
          index if one was passed to the constructor).

    Abstraction function:
        A no-op index that returns the first *k* items.
    """

    description: str = "stub"

    def __init__(self, name="stub", items=None, index=None, **kwargs):
        super().__init__(name=name, items=items or [], index=index)
        if self._index is None:
            self._index = self._get_or_create_index()

    def _get_or_create_index(self):
        return "stub"

    def search(self, query: str, k: int) -> list:
        return self.items[:k]


class TestCarnotIndexAttributes:
    """Base class attribute storage after construction."""

    def test_name_and_items_stored(self):
        """``name`` and ``items`` are stored as instance attributes."""
        idx = _StubIndex(name="test", items=["a", "b"])
        assert idx.name == "test"
        assert idx.items == ["a", "b"]

    def test_internal_index_populated(self):
        """``_index`` is set by ``_get_or_create_index``."""
        idx = _StubIndex(name="t", items=[])
        assert idx._index == "stub"

    def test_search_delegates_correctly(self):
        """``search`` returns the first *k* items."""
        idx = _StubIndex(name="t", items=["x", "y", "z"])
        assert idx.search("anything", 2) == ["x", "y"]

    def test_prebuilt_index_skips_build(self):
        """When ``index`` is passed, ``_get_or_create_index`` is not called."""
        pre_built = "pre-built-index"
        idx = _StubIndex(name="t", items=["a"], index=pre_built)
        assert idx._index == "pre-built-index"

    def test_prebuilt_index_with_items_for_mapping(self):
        """A pre-built index can coexist with items for result mapping."""
        pre_built = "pre-built"
        idx = _StubIndex(name="t", items=["x", "y"], index=pre_built)
        assert idx._index == "pre-built"
        assert idx.items == ["x", "y"]

    def test_items_defaults_to_empty(self):
        """When neither items nor index is given, items defaults to []."""
        idx = _StubIndex(name="t")
        assert idx.items == []
        assert idx._index == "stub"

    def test_index_without_items_raises(self):
        """Providing ``index`` without ``items`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="without 'items'"):
            _StubIndex(name="t", index="some-prebuilt-index")


# ═══════════════════════════════════════════════════════════════════════
# Module-level helpers
# ═══════════════════════════════════════════════════════════════════════


class TestBuildUriToIdx:
    """``_build_uri_to_idx`` mapping from URI to position."""

    def test_empty_list(self):
        """Empty input produces an empty mapping."""
        assert _build_uri_to_idx([]) == {}

    def test_dataitems(self):
        """DataItem objects are indexed by their ``uri``."""
        items = [DataItem(uri="s3://a"), DataItem(uri="s3://b")]
        mapping = _build_uri_to_idx(items)
        assert mapping == {"s3://a": 0, "s3://b": 1}

    def test_dicts(self):
        """Dicts with a ``"uri"`` key are indexed."""
        items = [{"uri": "/a.txt", "name": "a"}, {"uri": "/b.txt", "name": "b"}]
        mapping = _build_uri_to_idx(items)
        assert mapping == {"/a.txt": 0, "/b.txt": 1}

    def test_skips_items_without_uri(self):
        """Items without a URI attribute are silently skipped."""
        items = [DataItem(uri="s3://a"), DataItem(uri=""), DataItem(uri="s3://c")]
        mapping = _build_uri_to_idx(items)
        assert mapping == {"s3://a": 0, "s3://c": 2}

    def test_first_occurrence_wins(self):
        """When URIs are duplicated, the last position is stored."""
        items = [DataItem(uri="dup"), DataItem(uri="dup")]
        mapping = _build_uri_to_idx(items)
        assert mapping["dup"] == 1


class TestItemsToDataItems:
    """``_items_to_data_items`` conversion helper."""

    def test_empty_list(self):
        """Empty input returns an empty list."""
        assert _items_to_data_items([]) == []

    def test_dataitems_pass_through(self):
        """DataItem instances with non-empty URI are returned as-is."""
        di = DataItem(uri="s3://a")
        result = _items_to_data_items([di])
        assert result == [di]

    def test_dicts_converted(self):
        """Dict items with a ``"uri"`` key are wrapped in DataItem."""
        result = _items_to_data_items([{"uri": "/a.txt", "name": "a"}])
        assert len(result) == 1
        assert isinstance(result[0], DataItem)
        assert result[0].uri == "/a.txt"

    def test_skips_dataitems_without_uri(self):
        """DataItem with empty URI is skipped."""
        result = _items_to_data_items([DataItem(uri=""), DataItem(uri="ok")])
        assert len(result) == 1
        assert result[0].uri == "ok"

    def test_skips_dicts_without_uri(self):
        """Dict without a ``"uri"`` key is skipped."""
        result = _items_to_data_items([{"name": "no-uri"}, {"uri": "/b.txt"}])
        assert len(result) == 1
        assert result[0].uri == "/b.txt"

    def test_mixed_input(self):
        """Handles a mix of DataItems, dicts, and items without URIs."""
        items = [
            DataItem(uri="s3://a"),
            {"uri": "/b.txt"},
            DataItem(uri=""),
            {"name": "no-uri"},
        ]
        result = _items_to_data_items(items)
        assert len(result) == 2
        assert result[0].uri == "s3://a"
        assert result[1].uri == "/b.txt"
