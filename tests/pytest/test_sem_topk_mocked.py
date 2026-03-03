"""Tier 2 mocked tests for :class:`SemTopKOperator`.

SemTopKOperator does **not** use the LLM for generation — it uses an index.
These tests mock the index class so that no embedding API calls are made.
They validate:

* index construction when none pre-exists,
* index reuse when one already exists,
* top-k result slicing,
* dataset threading,
* catalog registration (performed by the operator, not the index),
* unique on-disk naming via ``"ds{dataset_id}_{index_name}"``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from carnot.data.dataset import Dataset
from carnot.operators.sem_topk import SemTopKOperator

# ── Helpers ──────────────────────────────────────────────────────────────────

_ANIMALS = [
    {"animal": "giraffe"},
    {"animal": "anaconda"},
    {"animal": "salmon"},
    {"animal": "elephant"},
    {"animal": "tucan"},
]


class _FakeIndex:
    """A fake index that returns the first *k* items from its stored items."""

    def __init__(self, name, items, model, api_key, index=None):
        self.name = name
        self._items = list(items) if items else []
        self._inner = index
        # If a pre-built inner index is provided, use it for search
        if index is not None and hasattr(index, "search"):
            self._search_delegate = index.search
        else:
            self._search_delegate = None

    def search(self, query: str, k: int) -> list[dict]:
        if self._search_delegate is not None:
            return self._search_delegate(query, k)
        return self._items[:k]


# ── Tests ────────────────────────────────────────────────────────────────────


class TestSemTopKMocked:
    """Mocked-index tests for SemTopKOperator."""

    @patch.dict(
        "carnot.operators.sem_topk.SemTopKOperator.__init__.__globals__",
    )
    def test_returns_top_k_items(self, mock_llm_config):
        """The operator returns exactly *k* items from the index search."""
        with patch.object(SemTopKOperator, "__init__", lambda self, **kw: None):
            op = SemTopKOperator.__new__(SemTopKOperator)
            op.task = "find mammals"
            op.output_dataset_id = "out"
            op.k = 2
            op.model_id = "mock-embedding"
            op.api_key = "fake"
            op.index_name = "flat"
            op.catalog = None
            op.index_cls = _FakeIndex

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        result = op("animals", {"animals": ds})

        assert "out" in result
        assert len(result["out"].items) == 2

    def test_constructs_index_on_the_fly(self, mock_llm_config):
        """When no index exists on the dataset, the operator creates one."""
        with patch.object(SemTopKOperator, "__init__", lambda self, **kw: None):
            op = SemTopKOperator.__new__(SemTopKOperator)
            op.task = "find mammals"
            op.output_dataset_id = "out"
            op.k = 3
            op.model_id = "mock-embedding"
            op.api_key = "fake"
            op.index_name = "test_idx"
            op.catalog = None
            op.index_cls = _FakeIndex

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        assert "test_idx" not in ds.list_indices()

        result = op("animals", {"animals": ds})

        # The index should now be registered on the dataset
        assert "test_idx" in ds.list_indices()
        assert len(result["out"].items) == 3

    def test_reuses_existing_index(self, mock_llm_config):
        """When an index already exists, the operator reuses it (no new construction)."""
        existing_index = MagicMock()
        existing_index.search.return_value = [{"animal": "elephant"}]

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        ds.indices["flat"] = existing_index

        with patch.object(SemTopKOperator, "__init__", lambda self, **kw: None):
            op = SemTopKOperator.__new__(SemTopKOperator)
            op.task = "find mammals"
            op.output_dataset_id = "out"
            op.k = 1
            op.model_id = "mock-embedding"
            op.api_key = "fake"
            op.index_name = "flat"
            op.catalog = None
            op.index_cls = _FakeIndex

        result = op("animals", {"animals": ds})

        # Should have used the existing index
        existing_index.search.assert_called_once_with("find mammals", k=1)
        assert len(result["out"].items) == 1
        assert result["out"].items[0]["animal"] == "elephant"

    def test_input_dataset_passed_through(self, mock_llm_config):
        """The original input dataset is present in the returned dict."""
        with patch.object(SemTopKOperator, "__init__", lambda self, **kw: None):
            op = SemTopKOperator.__new__(SemTopKOperator)
            op.task = "find"
            op.output_dataset_id = "out"
            op.k = 1
            op.model_id = "mock"
            op.api_key = "fake"
            op.index_name = "flat"
            op.catalog = None
            op.index_cls = _FakeIndex

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        result = op("animals", {"animals": ds})

        assert "animals" in result
        assert "out" in result
        assert len(result) == 2

    def test_k_greater_than_items(self, mock_llm_config):
        """When k exceeds the item count, the index returns whatever it has."""
        with patch.object(SemTopKOperator, "__init__", lambda self, **kw: None):
            op = SemTopKOperator.__new__(SemTopKOperator)
            op.task = "find"
            op.output_dataset_id = "out"
            op.k = 100
            op.model_id = "mock"
            op.api_key = "fake"
            op.index_name = "flat"
            op.catalog = None
            op.index_cls = _FakeIndex

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        result = op("animals", {"animals": ds})

        # _FakeIndex returns min(k, len(items))
        assert len(result["out"].items) == len(_ANIMALS)

    def test_catalog_registration_by_operator(self, mock_llm_config):
        """When a catalog is provided, the operator calls ``catalog.register_index``.

        The operator — not the index — is responsible for catalog
        registration.
        """
        catalog = MagicMock()

        with patch.object(SemTopKOperator, "__init__", lambda self, **kw: None):
            op = SemTopKOperator.__new__(SemTopKOperator)
            op.task = "find"
            op.output_dataset_id = "out"
            op.k = 2
            op.model_id = "mock"
            op.api_key = "fake"
            op.index_name = "test_idx"
            op.catalog = catalog
            op.index_cls = _FakeIndex

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        ds.dataset_id = 123
        op("animals", {"animals": ds})

        catalog.register_index.assert_called_once()
        call_kwargs = catalog.register_index.call_args[1]
        assert call_kwargs["dataset_id"] == 123
        assert call_kwargs["name"] == "test_idx"
        assert call_kwargs["index_type"] == "_FakeIndex"
        assert call_kwargs["index_obj"] is ds.indices["test_idx"]

    def test_unique_disk_name(self, mock_llm_config):
        """The on-disk index name includes the dataset_id to prevent collisions.

        Two datasets with different ``dataset_id`` values that both
        request ``"chroma"`` should produce distinct on-disk names.
        """
        with patch.object(SemTopKOperator, "__init__", lambda self, **kw: None):
            op = SemTopKOperator.__new__(SemTopKOperator)
            op.task = "find"
            op.output_dataset_id = "out"
            op.k = 2
            op.model_id = "mock"
            op.api_key = "fake"
            op.index_name = "chroma"
            op.catalog = None
            op.index_cls = _FakeIndex

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        ds.dataset_id = 42
        op("animals", {"animals": ds})

        created_index = ds.indices["chroma"]
        assert created_index.name == "ds42_chroma"

    def test_catalog_lookup_before_build(self, mock_llm_config):
        """When a catalog has a matching non-stale index, the operator reuses
        it instead of building a new one.
        """
        # Create a mock catalog that returns a pre-built index
        catalog = MagicMock()
        catalog_meta = MagicMock()
        catalog_meta.is_stale = False
        catalog_meta.id = 99
        catalog.get_index_by_name.return_value = catalog_meta

        # The loaded index object (inner index)
        loaded_inner = MagicMock()
        catalog.load_index.return_value = loaded_inner

        with patch.object(SemTopKOperator, "__init__", lambda self, **kw: None):
            op = SemTopKOperator.__new__(SemTopKOperator)
            op.task = "find mammals"
            op.output_dataset_id = "out"
            op.k = 2
            op.model_id = "mock-embedding"
            op.api_key = "fake"
            op.index_name = "flat"
            op.catalog = catalog
            op.index_cls = _FakeIndex

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        ds.dataset_id = 7

        result = op("animals", {"animals": ds})

        # Catalog was queried
        catalog.get_index_by_name.assert_called_once_with(7, "flat")
        catalog.load_index.assert_called_once_with(99)

        # No new index registration (reused existing)
        catalog.register_index.assert_not_called()

        # Results still returned
        assert "out" in result

    def test_catalog_stale_index_triggers_rebuild(self, mock_llm_config):
        """When the catalog returns a stale index, the operator builds fresh."""
        catalog = MagicMock()
        stale_meta = MagicMock()
        stale_meta.is_stale = True
        stale_meta.id = 50
        catalog.get_index_by_name.return_value = stale_meta

        with patch.object(SemTopKOperator, "__init__", lambda self, **kw: None):
            op = SemTopKOperator.__new__(SemTopKOperator)
            op.task = "find"
            op.output_dataset_id = "out"
            op.k = 2
            op.model_id = "mock"
            op.api_key = "fake"
            op.index_name = "flat"
            op.catalog = catalog
            op.index_cls = _FakeIndex

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        ds.dataset_id = 8
        op("animals", {"animals": ds})

        # Catalog was checked but stale → should NOT call load_index
        catalog.get_index_by_name.assert_called_once_with(8, "flat")
        catalog.load_index.assert_not_called()

        # A new index was built and registered
        catalog.register_index.assert_called_once()

    def test_catalog_miss_triggers_build(self, mock_llm_config):
        """When the catalog has no matching index, the operator builds one."""
        catalog = MagicMock()
        catalog.get_index_by_name.return_value = None

        with patch.object(SemTopKOperator, "__init__", lambda self, **kw: None):
            op = SemTopKOperator.__new__(SemTopKOperator)
            op.task = "find"
            op.output_dataset_id = "out"
            op.k = 2
            op.model_id = "mock"
            op.api_key = "fake"
            op.index_name = "flat"
            op.catalog = catalog
            op.index_cls = _FakeIndex

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        ds.dataset_id = 9
        op("animals", {"animals": ds})

        # Catalog was checked
        catalog.get_index_by_name.assert_called_once_with(9, "flat")
        # No load_index call since get_index_by_name returned None
        catalog.load_index.assert_not_called()
        # A new index was built and registered
        catalog.register_index.assert_called_once()

    def test_catalog_load_failure_triggers_build(self, mock_llm_config):
        """When catalog.load_index returns None, the operator builds fresh."""
        catalog = MagicMock()
        catalog_meta = MagicMock()
        catalog_meta.is_stale = False
        catalog_meta.id = 77
        catalog.get_index_by_name.return_value = catalog_meta
        catalog.load_index.return_value = None  # Deserialization failed

        with patch.object(SemTopKOperator, "__init__", lambda self, **kw: None):
            op = SemTopKOperator.__new__(SemTopKOperator)
            op.task = "find"
            op.output_dataset_id = "out"
            op.k = 2
            op.model_id = "mock"
            op.api_key = "fake"
            op.index_name = "flat"
            op.catalog = catalog
            op.index_cls = _FakeIndex

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        ds.dataset_id = 10
        op("animals", {"animals": ds})

        # Catalog was queried
        catalog.get_index_by_name.assert_called_once_with(10, "flat")
        catalog.load_index.assert_called_once_with(77)
        # Fell back to building a new index
        catalog.register_index.assert_called_once()
