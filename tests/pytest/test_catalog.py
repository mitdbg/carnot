"""Unit tests for DatasetCatalog and IndexCatalog (Tier 1 — in-memory).

Tests cover the in-memory code paths of both catalogs.  No database or
LLM is required.  These validate the contracts documented in each
catalog class's docstring: CRUD operations, filtering, stale marking,
and ID auto-increment.
"""

from __future__ import annotations

import pytest

from carnot.storage.catalog import DatasetCatalog, DatasetMeta, IndexCatalog, IndexMeta  # noqa: I001

# ═══════════════════════════════════════════════════════════════════════
# DatasetCatalog — in-memory backend
# ═══════════════════════════════════════════════════════════════════════


class TestDatasetCatalogMemory:
    """DatasetCatalog with no DB (in-memory fallback)."""

    @pytest.fixture()
    def catalog(self) -> DatasetCatalog:
        """A fresh in-memory DatasetCatalog."""
        return DatasetCatalog()

    # ── create / get ────────────────────────────────────────────────────

    def test_create_returns_meta(self, catalog: DatasetCatalog):
        """create_dataset returns a DatasetMeta with the given name."""
        meta = catalog.create_dataset(name="movies", annotation="Film reviews")
        assert isinstance(meta, DatasetMeta)
        assert meta.name == "movies"
        assert meta.annotation == "Film reviews"
        assert meta.id == 1

    def test_create_auto_increments_id(self, catalog: DatasetCatalog):
        """Successive creates yield incrementing IDs."""
        m1 = catalog.create_dataset(name="a")
        m2 = catalog.create_dataset(name="b")
        assert m2.id == m1.id + 1

    def test_get_existing(self, catalog: DatasetCatalog):
        """get_dataset returns the correct DatasetMeta by ID."""
        meta = catalog.create_dataset(name="ds")
        got = catalog.get_dataset(meta.id)
        assert got is not None
        assert got.name == "ds"

    def test_get_missing_returns_none(self, catalog: DatasetCatalog):
        """get_dataset returns None for a non-existent ID."""
        assert catalog.get_dataset(999) is None

    # ── list ────────────────────────────────────────────────────────────

    def test_list_empty(self, catalog: DatasetCatalog):
        """An empty catalog returns an empty list."""
        assert catalog.list_datasets() == []

    def test_list_all(self, catalog: DatasetCatalog):
        """list_datasets with no filter returns all datasets."""
        catalog.create_dataset(name="a")
        catalog.create_dataset(name="b")
        assert len(catalog.list_datasets()) == 2

    def test_list_filtered_by_user(self, catalog: DatasetCatalog):
        """list_datasets(user_id=…) returns only that user's datasets."""
        catalog.create_dataset(name="a", user_id="alice")
        catalog.create_dataset(name="b", user_id="bob")
        alice_datasets = catalog.list_datasets(user_id="alice")
        assert len(alice_datasets) == 1
        assert alice_datasets[0].name == "a"

    # ── delete ──────────────────────────────────────────────────────────

    def test_delete_removes(self, catalog: DatasetCatalog):
        """delete_dataset removes the dataset from the catalog."""
        meta = catalog.create_dataset(name="temp")
        catalog.delete_dataset(meta.id)
        assert catalog.get_dataset(meta.id) is None

    def test_delete_nonexistent_is_noop(self, catalog: DatasetCatalog):
        """Deleting a non-existent ID does not raise."""
        catalog.delete_dataset(999)  # should not raise

    # ── timestamps ──────────────────────────────────────────────────────

    def test_created_at_populated(self, catalog: DatasetCatalog):
        """created_at is set on creation."""
        meta = catalog.create_dataset(name="ts")
        assert meta.created_at is not None
        assert meta.updated_at is not None


# ═══════════════════════════════════════════════════════════════════════
# IndexCatalog — in-memory backend
# ═══════════════════════════════════════════════════════════════════════


class TestIndexCatalogMemory:
    """IndexCatalog with no DB (in-memory fallback)."""

    @pytest.fixture()
    def catalog(self) -> IndexCatalog:
        """A fresh in-memory IndexCatalog (no storage backend)."""
        return IndexCatalog()

    # ── register / get ──────────────────────────────────────────────────

    def test_register_returns_meta(self, catalog: IndexCatalog):
        """register_index returns an IndexMeta with the given fields."""
        meta = catalog.register_index(
            dataset_id=1,
            index_type="flat",
            name="flat_idx",
            config={"dim": 128},
            item_count=100,
        )
        assert isinstance(meta, IndexMeta)
        assert meta.name == "flat_idx"
        assert meta.index_type == "flat"
        assert meta.dataset_id == 1
        assert meta.config == {"dim": 128}
        assert meta.item_count == 100

    def test_register_auto_increments(self, catalog: IndexCatalog):
        """Successive registers yield incrementing IDs."""
        m1 = catalog.register_index(dataset_id=1, index_type="flat", name="a")
        m2 = catalog.register_index(dataset_id=1, index_type="flat", name="b")
        assert m2.id == m1.id + 1

    def test_register_updates_existing(self, catalog: IndexCatalog):
        """Registering the same (dataset_id, name) updates existing entry."""
        m1 = catalog.register_index(dataset_id=1, index_type="flat", name="idx", item_count=10)
        m2 = catalog.register_index(dataset_id=1, index_type="hierarchical", name="idx", item_count=20)
        # Same ID, updated fields
        assert m2.id == m1.id
        assert m2.index_type == "hierarchical"
        assert m2.item_count == 20

    def test_get_index_existing(self, catalog: IndexCatalog):
        """get_index returns the correct IndexMeta by ID."""
        meta = catalog.register_index(dataset_id=1, index_type="flat", name="x")
        got = catalog.get_index(meta.id)
        assert got is not None
        assert got.name == "x"

    def test_get_index_missing(self, catalog: IndexCatalog):
        """get_index returns None for unknown ID."""
        assert catalog.get_index(999) is None

    def test_get_index_by_name(self, catalog: IndexCatalog):
        """get_index_by_name matches on (dataset_id, name)."""
        catalog.register_index(dataset_id=1, index_type="flat", name="alpha")
        catalog.register_index(dataset_id=2, index_type="flat", name="alpha")
        result = catalog.get_index_by_name(dataset_id=2, name="alpha")
        assert result is not None
        assert result.dataset_id == 2

    def test_get_index_by_name_miss(self, catalog: IndexCatalog):
        """get_index_by_name returns None when not found."""
        assert catalog.get_index_by_name(dataset_id=1, name="nope") is None

    # ── list ────────────────────────────────────────────────────────────

    def test_list_empty(self, catalog: IndexCatalog):
        """An empty catalog returns an empty list."""
        assert catalog.list_indices() == []

    def test_list_all(self, catalog: IndexCatalog):
        """list_indices with no filter returns all."""
        catalog.register_index(dataset_id=1, index_type="flat", name="a")
        catalog.register_index(dataset_id=2, index_type="flat", name="b")
        assert len(catalog.list_indices()) == 2

    def test_list_filtered_by_dataset(self, catalog: IndexCatalog):
        """list_indices(dataset_id=…) returns only that dataset's indices."""
        catalog.register_index(dataset_id=1, index_type="flat", name="a")
        catalog.register_index(dataset_id=2, index_type="flat", name="b")
        results = catalog.list_indices(dataset_id=1)
        assert len(results) == 1
        assert results[0].name == "a"

    # ── mark stale ──────────────────────────────────────────────────────

    def test_mark_stale(self, catalog: IndexCatalog):
        """mark_stale sets is_stale=True for all indices of a dataset."""
        catalog.register_index(dataset_id=1, index_type="flat", name="a")
        catalog.register_index(dataset_id=1, index_type="flat", name="b")
        catalog.register_index(dataset_id=2, index_type="flat", name="c")
        catalog.mark_stale(dataset_id=1)

        for m in catalog.list_indices(dataset_id=1):
            assert m.is_stale is True
        for m in catalog.list_indices(dataset_id=2):
            assert m.is_stale is False

    # ── delete ──────────────────────────────────────────────────────────

    def test_delete_removes(self, catalog: IndexCatalog):
        """delete_index removes the entry."""
        meta = catalog.register_index(dataset_id=1, index_type="flat", name="tmp")
        catalog.delete_index(meta.id)
        assert catalog.get_index(meta.id) is None

    def test_delete_nonexistent_is_noop(self, catalog: IndexCatalog):
        """Deleting a non-existent ID does not raise."""
        catalog.delete_index(999)

    # ── load_index without storage ──────────────────────────────────────

    def test_load_index_no_storage_returns_none(self, catalog: IndexCatalog):
        """load_index returns None when no storage is configured."""
        meta = catalog.register_index(
            dataset_id=1, index_type="flat", name="x", storage_uri="some/uri"
        )
        assert catalog.load_index(meta.id) is None

    def test_load_index_with_cached_object(self, catalog: IndexCatalog):
        """load_index returns the cached object if one was registered."""
        sentinel = object()
        meta = catalog.register_index(
            dataset_id=1, index_type="flat", name="x", index_obj=sentinel
        )
        loaded = catalog.load_index(meta.id)
        assert loaded is sentinel

    # ── timestamps ──────────────────────────────────────────────────────

    def test_register_sets_timestamps(self, catalog: IndexCatalog):
        """register_index sets created_at and updated_at."""
        meta = catalog.register_index(dataset_id=1, index_type="flat", name="ts")
        assert meta.created_at is not None
        assert meta.updated_at is not None

    def test_update_changes_updated_at(self, catalog: IndexCatalog):
        """Re-registering the same index updates updated_at."""
        meta = catalog.register_index(dataset_id=1, index_type="flat", name="ts")
        original_updated = meta.updated_at
        # Re-register to trigger update
        meta2 = catalog.register_index(dataset_id=1, index_type="hierarchical", name="ts")
        assert meta2.updated_at >= original_updated


# ═══════════════════════════════════════════════════════════════════════
# StorageConfig — catalog backend settings
# ═══════════════════════════════════════════════════════════════════════


class TestStorageConfigCatalog:
    """StorageConfig catalog-backend configuration."""

    def test_default_is_memory(self):
        """Default catalog_backend is 'memory'."""
        from carnot.storage.config import StorageConfig

        config = StorageConfig()
        assert config.catalog_backend == "memory"
        assert not config.has_postgres

    def test_postgres_backend(self):
        """catalog_backend='postgres' enables has_postgres."""
        from carnot.storage.config import StorageConfig

        config = StorageConfig(
            catalog_backend="postgres",
            db_user="user",
            db_password="pass",
            db_name="testdb",
        )
        assert config.has_postgres
        assert "user" in config.database_url
        assert "testdb" in config.database_url

    def test_invalid_backend_raises(self):
        """Invalid catalog_backend raises ValueError."""
        from carnot.storage.config import StorageConfig

        with pytest.raises(ValueError, match="catalog_backend"):
            StorageConfig(catalog_backend="redis")

    def test_database_url_requires_postgres(self):
        """database_url raises ValueError if backend is not postgres."""
        from carnot.storage.config import StorageConfig

        config = StorageConfig()
        with pytest.raises(ValueError, match="database_url"):
            _ = config.database_url

    def test_get_db_session_factory_requires_postgres(self):
        """get_db_session_factory raises ValueError if backend is memory."""
        from carnot.storage.config import StorageConfig

        config = StorageConfig()
        with pytest.raises(ValueError, match="get_db_session_factory"):
            config.get_db_session_factory()

    def test_explicit_session_factory_passthrough(self):
        """An explicit db_session_factory is returned as-is."""
        from carnot.storage.config import StorageConfig

        sentinel = object()
        config = StorageConfig(
            catalog_backend="postgres",
            db_session_factory=sentinel,
        )
        assert config.get_db_session_factory() is sentinel

    def test_db_defaults_from_env(self, monkeypatch):
        """DB parameters fall back to environment variables."""
        from carnot.storage.config import StorageConfig

        monkeypatch.setenv("CARNOT_DB_USER", "envuser")
        monkeypatch.setenv("CARNOT_DB_PASSWORD", "envpass")
        monkeypatch.setenv("CARNOT_DB_NAME", "envdb")
        monkeypatch.setenv("CARNOT_DB_HOST", "envhost")
        monkeypatch.setenv("CARNOT_DB_PORT", "5433")

        config = StorageConfig(catalog_backend="postgres")
        assert config.db_user == "envuser"
        assert config.db_password == "envpass"
        assert config.db_name == "envdb"
        assert config.db_host == "envhost"
        assert config.db_port == 5433

    def test_repr_includes_catalog_backend(self, tmp_path):
        """repr includes catalog_backend."""
        from carnot.storage.config import StorageConfig

        config = StorageConfig(base_dir=tmp_path, catalog_backend="memory")
        r = repr(config)
        assert "memory" in r

    def test_s3_config(self):
        """S3 bucket and prefix are stored."""
        from carnot.storage.config import StorageConfig

        config = StorageConfig(s3_bucket="my-bucket", s3_prefix="data/")
        assert config.s3_bucket == "my-bucket"
        assert config.s3_prefix == "data/"
