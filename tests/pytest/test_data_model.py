"""Unit tests for the core data model: Dataset and DataItem.

Tests cover:
1. Dataset construction (from dicts, from DataItems, empty)
2. Lazy materialization semantics (items property, is_materialized flag)
3. Dataset.serialize() round-trip structure
4. Dataset operator chaining (sem_filter, sem_map, sem_join, limit, etc.)
5. DataItem construction, ``from_dict``, and ``materialize()``
6. DataItem backward-compat ``path`` ↔ ``uri`` aliasing
"""

from __future__ import annotations

import warnings

import pytest

from carnot.data.dataset import Dataset
from carnot.data.item import DataItem

# ═══════════════════════════════════════════════════════════════════════
# DataItem
# ═══════════════════════════════════════════════════════════════════════


class TestDataItemConstruction:
    """DataItem creation from various inputs."""

    def test_from_uri(self):
        """DataItem stores the uri passed at construction."""
        item = DataItem(uri="/data/emails/msg001.txt")
        assert item.uri == "/data/emails/msg001.txt"

    def test_from_path_legacy(self):
        """DataItem accepts the legacy ``path`` positional argument."""
        item = DataItem("/data/emails/msg001.txt")
        assert item.uri == "/data/emails/msg001.txt"

    def test_default_uri_is_empty(self):
        """DataItem with no arguments defaults uri to the empty string."""
        item = DataItem()
        assert item.uri == ""

    def test_uri_takes_precedence_over_path(self):
        """When both ``path`` and ``uri`` are given, ``uri`` wins."""
        item = DataItem(path="old.txt", uri="new.txt")
        assert item.uri == "new.txt"


class TestDataItemPathCompat:
    """Backward-compatible ``path`` property emits deprecation warnings."""

    def test_path_getter_warns(self):
        """Accessing ``.path`` emits a DeprecationWarning."""
        item = DataItem(uri="foo.txt")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = item.path
        assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_path_setter_warns(self):
        """Setting ``.path`` emits a DeprecationWarning."""
        item = DataItem(uri="foo.txt")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            item.path = "bar.txt"
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
        assert item.uri == "bar.txt"


class TestDataItemFromDict:
    """DataItem.from_dict() round-trip."""

    def test_from_dict_with_uri(self):
        """from_dict extracts ``uri`` and pre-populates the internal cache."""
        d = {"uri": "emails/msg.txt", "contents": "Hello world"}
        item = DataItem.from_dict(d)
        assert item.uri == "emails/msg.txt"
        # to_dict returns the cached dict directly (no materialization)
        assert item.to_dict() is d

    def test_from_dict_with_path_key(self):
        """from_dict falls back to the ``path`` key when ``uri`` is absent."""
        d = {"path": "emails/msg.txt", "contents": "Hello"}
        item = DataItem.from_dict(d)
        assert item.uri == "emails/msg.txt"

    def test_from_dict_without_uri_or_path(self):
        """from_dict with neither ``uri`` nor ``path`` yields empty uri."""
        d = {"contents": "standalone"}
        item = DataItem.from_dict(d)
        assert item.uri == ""
        assert item.to_dict()["contents"] == "standalone"


class TestDataItemMaterialize:
    """DataItem.materialize() integration with storage layer."""

    def test_materialize_caches_result(self, in_memory_backend, tiered_storage):
        """materialize() stores result in _dict and returns same object on repeat call."""
        in_memory_backend.write("file.txt", b"some content")
        item = DataItem(uri="file.txt")

        result = item.materialize(tiered_storage)
        assert isinstance(result, dict)
        assert result["uri"] == "file.txt"
        assert "some content" in result["contents"]

        # Second call returns cached result (same object)
        assert item.materialize(tiered_storage) is result

    def test_materialize_without_storage_raises(self):
        """materialize() without a storage manager raises ValueError."""
        item = DataItem(uri="file.txt")
        with pytest.raises(ValueError, match="No storage provided"):
            item.materialize()

    def test_from_dict_item_already_materialized(self):
        """A DataItem built via from_dict is already \"materialized\" (has _dict)."""
        d = {"uri": "x.txt", "contents": "cached"}
        item = DataItem.from_dict(d)
        # materialize returns the pre-set dict — no storage needed
        assert item.to_dict() is d
        assert item.materialize() is d


class TestDataItemUpdateDict:
    """DataItem.update_dict() merges into internal cache."""

    def test_update_adds_keys(self):
        """update_dict adds new keys to the cached dict."""
        d = {"uri": "a.txt", "contents": "hello"}
        item = DataItem.from_dict(d)
        item.update_dict({"sentiment": "positive"})
        assert item.to_dict()["sentiment"] == "positive"
        # Original keys still present
        assert item.to_dict()["contents"] == "hello"


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════


class TestDatasetConstruction:
    """Dataset creation from different item types."""

    def test_from_dicts_is_materialized(self):
        """A Dataset constructed with list[dict] is immediately materialized."""
        ds = Dataset(
            name="test",
            items=[{"id": 1}, {"id": 2}],
        )
        assert ds.is_materialized
        assert len(ds.items) == 2
        assert ds.items[0] == {"id": 1}

    def test_from_dataitems_is_not_materialized(self):
        """A Dataset constructed with DataItems is lazy (not materialized)."""
        items = [DataItem(uri="a.txt"), DataItem(uri="b.txt")]
        ds = Dataset(name="test", items=items)
        assert not ds.is_materialized
        assert len(ds.items) == 2
        assert isinstance(ds.items[0], DataItem)

    def test_empty_dataset(self):
        """An empty Dataset has no items and is not materialized."""
        ds = Dataset(name="empty")
        assert not ds.is_materialized
        assert ds.items == []

    def test_metadata_fields(self):
        """name, annotation, and dataset_id are stored correctly."""
        ds = Dataset(name="Movies", annotation="Movie data", dataset_id=42)
        assert ds.name == "Movies"
        assert ds.annotation == "Movie data"
        assert ds.dataset_id == 42


class TestDatasetItemsProperty:
    """Lazy items property getter and setter."""

    def test_setter_with_dicts_materializes(self):
        """Assigning a list[dict] via .items marks the dataset as materialized."""
        ds = Dataset(name="test")
        ds.items = [{"a": 1}]
        assert ds.is_materialized
        assert ds.items == [{"a": 1}]

    def test_setter_with_dataitems_unmaterializes(self):
        """Assigning DataItems via .items marks the dataset as not materialized."""
        ds = Dataset(name="test", items=[{"a": 1}])
        assert ds.is_materialized
        ds.items = [DataItem(uri="x.txt")]
        assert not ds.is_materialized

    def test_setter_with_none_clears(self):
        """Assigning None clears items and sets not-materialized."""
        ds = Dataset(name="test", items=[{"a": 1}])
        ds.items = None
        assert not ds.is_materialized
        assert ds.items == []


class TestDatasetMaterialize:
    """Dataset.materialize() with the storage layer."""

    def test_already_materialized_is_noop(self):
        """Calling materialize on an already-materialized dataset returns self."""
        ds = Dataset(name="t", items=[{"a": 1}])
        assert ds.materialize() is ds

    def test_leaf_dataset_materializes_items(self, in_memory_backend, tiered_storage):
        """A leaf dataset materializes each DataItem through storage."""
        in_memory_backend.write("a.txt", b"alpha content")
        in_memory_backend.write("b.txt", b"beta content")
        items = [DataItem(uri="a.txt"), DataItem(uri="b.txt")]
        ds = Dataset(name="docs", items=items)

        result = ds.materialize(tiered_storage)
        assert result is ds
        assert ds.is_materialized
        assert len(ds.items) == 2
        assert "alpha content" in ds.items[0]["contents"]
        assert "beta content" in ds.items[1]["contents"]

    def test_derived_dataset_raises(self):
        """A derived dataset (has parents) cannot be materialized directly."""
        parent = Dataset(name="parent", items=[{"a": 1}])
        child = parent.sem_filter("some condition")
        with pytest.raises(ValueError, match="Cannot materialize a derived Dataset"):
            child.materialize()


class TestDatasetSerialize:
    """Dataset.serialize() produces the expected plan structure."""

    def test_leaf_serialize(self):
        """A leaf dataset serializes to a dict with empty parents list."""
        ds = Dataset(name="Movies", annotation="films")
        plan = ds.serialize()
        assert plan["name"] == "Movies"
        assert plan["parents"] == []

    def test_single_op_serialize(self):
        """A single operator produces one level of nesting in serialize()."""
        ds = Dataset(name="Movies", annotation="films")
        filtered = ds.sem_filter("rating > 8")
        plan = filtered.serialize()
        assert plan["params"]["operator"] == "SemanticFilter"
        assert len(plan["parents"]) == 1
        assert plan["parents"][0]["name"] == "Movies"

    def test_chained_ops_serialize(self):
        """Chained operators produce the correct nesting depth."""
        ds = Dataset(name="Movies", annotation="films")
        result = ds.sem_filter("rating > 8").sem_map("genre_label", str, "genre name")
        plan = result.serialize()
        assert plan["params"]["operator"] == "SemanticMap"
        parent = plan["parents"][0]
        assert parent["params"]["operator"] == "SemanticFilter"
        assert parent["parents"][0]["name"] == "Movies"

    def test_join_serialize_has_two_parents(self):
        """A sem_join node has two parent entries in serialize()."""
        ds_a = Dataset(name="Movies")
        ds_b = Dataset(name="Reviews")
        joined = ds_a.sem_join(ds_b, "matching movie")
        plan = joined.serialize()
        assert plan["params"]["operator"] == "SemanticJoin"
        assert len(plan["parents"]) == 2
        parent_names = {p["name"] for p in plan["parents"]}
        assert parent_names == {"Movies", "Reviews"}


class TestDatasetOperatorChaining:
    """Each fluent operator returns a new Dataset with correct params."""

    def test_sem_filter(self):
        """sem_filter creates a child with SemanticFilter operator."""
        ds = Dataset(name="Movies")
        child = ds.sem_filter("rating > 8")
        assert child.params["operator"] == "SemanticFilter"
        assert child.params["condition"] == "rating > 8"
        assert child.parents == [ds]

    def test_sem_map(self):
        """sem_map creates a child with SemanticMap operator and field metadata."""
        ds = Dataset(name="Movies")
        child = ds.sem_map("sentiment", str, "overall sentiment")
        assert child.params["operator"] == "SemanticMap"
        assert child.params["field"] == "sentiment"
        assert child.params["type"] == "str"

    def test_sem_flat_map(self):
        """sem_flat_map creates a child with SemanticFlatMap operator."""
        ds = Dataset(name="Movies")
        child = ds.sem_flat_map("keyword", str, "extracted keyword")
        assert child.params["operator"] == "SemanticFlatMap"
        assert child.params["field"] == "keyword"

    def test_sem_topk(self):
        """sem_topk creates a child with SemanticTopK operator."""
        ds = Dataset(name="Movies")
        child = ds.sem_topk("my_index", "action movies", k=3)
        assert child.params["operator"] == "SemanticTopK"
        assert child.params["k"] == 3
        assert child.params["search_str"] == "action movies"
        assert child.params["index_name"] == "my_index"

    def test_limit(self):
        """limit creates a child with Limit operator."""
        ds = Dataset(name="Movies")
        child = ds.limit(5)
        assert child.params["operator"] == "Limit"
        assert child.params["n"] == 5

    def test_write_code(self):
        """write_code creates a child with Code operator."""
        ds = Dataset(name="Movies")
        child = ds.write_code("compute stats")
        assert child.params["operator"] == "Code"
        assert child.params["task"] == "compute stats"

    def test_id_params_increment(self):
        """Successive calls to the same operator type increment the id counter."""
        ds = Dataset(name="Movies")
        f1 = ds.sem_filter("a")
        f2 = ds.sem_filter("b")
        assert f1.name == "FilterOperation1"
        assert f2.name == "FilterOperation2"


class TestDatasetIndices:
    """Index-related accessors on Dataset."""

    def test_no_indices_by_default(self):
        """A fresh Dataset with no indices returns empty."""
        ds = Dataset(name="t")
        assert not ds.has_indices()
        assert ds.list_indices() == []

    def test_explicit_indices(self):
        """Setting indices via constructor makes them available."""

        class FakeIndex:
            description = "fake"

        ds = Dataset(name="t", indices={"fake": FakeIndex()})
        assert ds.has_indices()
        assert ds.list_indices() == ["fake"]
        info = ds.get_indices_info()
        assert info[0]["index_name"] == "fake"
        assert info[0]["index_type"] == "FakeIndex"


class TestDatasetIteration:
    """Dataset is iterable over its items."""

    def test_iter_materialized(self):
        """Iterating a materialized dataset yields dicts."""
        ds = Dataset(name="t", items=[{"a": 1}, {"a": 2}])
        assert list(ds) == [{"a": 1}, {"a": 2}]

    def test_iter_dataitems(self):
        """Iterating an unmaterialized dataset yields DataItem refs."""
        items = [DataItem(uri="a.txt"), DataItem(uri="b.txt")]
        ds = Dataset(name="t", items=items)
        result = list(ds)
        assert len(result) == 2
        assert all(isinstance(r, DataItem) for r in result)
