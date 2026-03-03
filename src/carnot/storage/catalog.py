"""DatasetCatalog and IndexCatalog — persistent registries.

Both catalogs expose **sync** public interfaces so the execution engine
and operators can call them without ``await``.

When a ``db_session_factory`` (an ``async_sessionmaker``) is provided,
the DB-backed private methods use ``asyncio`` internally.  Otherwise
the catalogs fall back to simple in-memory dicts (tests / scripts).

ORM models are defined in :mod:`carnot.storage.models` so that the
library never imports from the web-application layer (``app``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import faiss
from sqlalchemy import or_

if TYPE_CHECKING:
    from carnot.index.index import CarnotIndex

from carnot.storage.tiered import TieredStorageManager

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from sync code.

    Works whether or not there is already a running event loop
    (e.g. inside FastAPI / Jupyter).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We're inside an existing event loop (FastAPI, Jupyter, etc.).
        # Create a new thread to run the coroutine.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ── Metadata dataclasses ────────────────────────────────────────────────

@dataclass
class DatasetMeta:
    """Metadata about a registered dataset (mirrors the ``datasets`` DB table)."""

    id: int
    name: str
    annotation: str
    user_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class IndexMeta:
    """Metadata about a registered index (mirrors the ``indices`` DB table)."""

    id: int
    dataset_id: int
    name: str
    index_type: str  # "flat", "hierarchical", "chroma", "faiss"
    config: dict = field(default_factory=dict)
    storage_uri: str = ""
    item_count: int | None = None
    is_stale: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


# ── DatasetCatalog ──────────────────────────────────────────────────────

class DatasetCatalog:
    """Catalog of datasets backed by PostgreSQL or an in-memory store.

    When *db_session_factory* is provided it should be an
    :class:`~sqlalchemy.ext.asyncio.async_sessionmaker` that yields
    ``AsyncSession`` instances.  Otherwise the catalog falls back to a
    simple in-memory dict (suitable for tests and standalone scripts).

    Public methods are **sync**.  DB operations run async internally
    via :func:`_run_async`.
    """

    def __init__(self, db_session_factory=None):
        self._db_factory = db_session_factory

        # in-memory fallback for environments without a DB (tests, scripts)
        self._memory_store: dict[int, DatasetMeta] = {}
        self._next_id = 1

    # ── Public API (sync) ──

    def list_datasets(self, user_id: str | None = None) -> list[DatasetMeta]:
        """List all datasets, optionally filtered by user."""
        if self._db_factory is not None:
            return _run_async(self._list_datasets_db(user_id))

        results = list(self._memory_store.values())
        if user_id:
            results = [d for d in results if d.user_id == user_id]
        return results

    def get_dataset(self, dataset_id: int) -> DatasetMeta | None:
        if self._db_factory is not None:
            return _run_async(self._get_dataset_db(dataset_id))
        return self._memory_store.get(dataset_id)

    def create_dataset(
        self,
        name: str,
        annotation: str = "",
        user_id: str | None = None,
    ) -> DatasetMeta:
        if self._db_factory is not None:
            return _run_async(self._create_dataset_db(name, annotation, user_id))

        meta = DatasetMeta(
            id=self._next_id,
            name=name,
            annotation=annotation,
            user_id=user_id,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        self._memory_store[meta.id] = meta
        self._next_id += 1
        return meta

    def delete_dataset(self, dataset_id: int) -> None:
        if self._db_factory is not None:
            return _run_async(self._delete_dataset_db(dataset_id))
        self._memory_store.pop(dataset_id, None)

    # ── DB implementations ──────────────────────────────────────────────

    async def _list_datasets_db(self, user_id: str | None) -> list[DatasetMeta]:
        from sqlalchemy import select

        from carnot.storage.models import DatasetModel

        async with self._db_factory() as session:
            stmt = select(DatasetModel)
            if user_id is not None:
                stmt = stmt.where(or_(DatasetModel.user_id == user_id, DatasetModel.shared))
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                DatasetMeta(
                    id=r.id,
                    name=r.name,
                    annotation=r.annotation or "",
                    user_id=r.user_id,
                    created_at=r.created_at,
                    updated_at=r.updated_at,
                )
                for r in rows
            ]

    async def _get_dataset_db(self, dataset_id: int) -> DatasetMeta | None:
        from sqlalchemy import select

        from carnot.storage.models import DatasetModel

        async with self._db_factory() as session:
            stmt = select(DatasetModel).where(DatasetModel.id == dataset_id)
            result = await session.execute(stmt)
            r = result.scalar_one_or_none()
            if r is None:
                return None
            return DatasetMeta(
                id=r.id,
                name=r.name,
                annotation=r.annotation or "",
                user_id=r.user_id,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )

    async def _create_dataset_db(
        self, name: str, annotation: str, user_id: str | None
    ) -> DatasetMeta:
        from carnot.storage.models import DatasetModel

        async with self._db_factory() as session:
            row = DatasetModel(
                user_id=user_id or "",
                name=name,
                annotation=annotation,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return DatasetMeta(
                id=row.id,
                name=row.name,
                annotation=row.annotation or "",
                user_id=row.user_id,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )

    async def _delete_dataset_db(self, dataset_id: int) -> None:
        from sqlalchemy import delete

        from carnot.storage.models import DatasetModel

        async with self._db_factory() as session:
            stmt = delete(DatasetModel).where(DatasetModel.id == dataset_id)
            await session.execute(stmt)
            await session.commit()


# ── IndexCatalog ────────────────────────────────────────────────────────

class IndexCatalog:
    """Catalog of indices backed by PostgreSQL or an in-memory store.

    Bridges DB records with :class:`TieredStorageManager` for loading/persisting index data.

    Public methods are **sync**.  DB operations run async internally
    via :func:`_run_async`.
    """

    def __init__(
        self,
        storage: TieredStorageManager | None = None,
        db_session_factory=None,
    ):
        self._storage = storage
        self._db_factory = db_session_factory

        # in-memory fallback
        self._memory_store: dict[int, IndexMeta] = {}
        self._index_objects: dict[int, CarnotIndex] = {}
        self._next_id = 1

    # ── Public API (sync) ──

    def list_indices(self, dataset_id: int | None = None) -> list[IndexMeta]:
        if self._db_factory is not None:
            return _run_async(self._list_indices_db(dataset_id))
        results = list(self._memory_store.values())
        if dataset_id is not None:
            results = [m for m in results if m.dataset_id == dataset_id]
        return results

    def get_index(self, index_id: int) -> IndexMeta | None:
        if self._db_factory is not None:
            return _run_async(self._get_index_db(index_id))
        return self._memory_store.get(index_id)

    def get_index_by_name(self, dataset_id: int, name: str) -> IndexMeta | None:
        if self._db_factory is not None:
            return _run_async(self._get_index_by_name_db(dataset_id, name))
        for meta in self._memory_store.values():
            if meta.dataset_id == dataset_id and meta.name == name:
                return meta
        return None

    def register_index(
        self,
        dataset_id: int,
        index_type: str,
        name: str,
        config: dict | None = None,
        storage_uri: str = "",
        item_count: int | None = None,
        is_stale: bool = False,
        index_obj: CarnotIndex | None = None,
    ) -> IndexMeta:
        """Register (or update) an index in the catalog.

        Parameters
        ----------
        index_obj:
            The live index object.  Kept in-memory so ``load_index``
            can return it without deserializing from storage.
        """
        existing = self.get_index_by_name(dataset_id, name)
        if existing is not None:
            existing.index_type = index_type
            existing.config = config or {}
            existing.storage_uri = storage_uri
            existing.item_count = item_count
            existing.is_stale = is_stale
            existing.updated_at = datetime.now(UTC)
            if index_obj is not None:
                self._index_objects[existing.id] = index_obj
            if self._db_factory is not None:
                _run_async(self._update_index_db(existing))
            else:
                self._memory_store[existing.id] = existing
            return existing

        meta = IndexMeta(
            id=self._next_id,
            dataset_id=dataset_id,
            name=name,
            index_type=index_type,
            config=config or {},
            storage_uri=storage_uri,
            item_count=item_count,
            is_stale=is_stale,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        if index_obj is not None:
            self._index_objects[meta.id] = index_obj
        if self._db_factory is not None:
            _run_async(self._create_index_db(meta))
        else:
            self._memory_store[meta.id] = meta
            self._next_id += 1
        return meta

    def load_index(self, index_id: int) -> CarnotIndex | None:
        """Load an index from memory or storage using catalog metadata.

        Checks the in-memory object cache first, then falls back to deserialization from storage.
        """
        # check in-memory object cache first
        if index_id in self._index_objects:
            return self._index_objects[index_id]

        meta = self.get_index(index_id)
        if meta is None:
            return None

        if self._storage is None:
            logger.warning(f"No storage backend configured for IndexCatalog; cannot load index {index_id}")
            return None

        try:
            data = self._storage.read(meta.storage_uri)
            index = self._deserialize_index(meta.index_type, data, meta.config)
            if index is not None:
                self._index_objects[index_id] = index
            return index
        except Exception as e:
            logger.error("Failed to load index %d (%s): %s", index_id, meta.name, e)
            return None

    def mark_stale(self, dataset_id: int) -> None:
        """Mark all indices for a dataset as stale."""
        if self._db_factory is not None:
            _run_async(self._mark_stale_db(dataset_id))
            return

        for meta in self._memory_store.values():
            if meta.dataset_id == dataset_id:
                meta.is_stale = True
                meta.updated_at = datetime.now(UTC)

    def delete_index(self, index_id: int) -> None:
        """Delete index from catalog and storage."""
        self._index_objects.pop(index_id, None)

        meta = self.get_index(index_id)
        if self._db_factory is not None and meta is not None:
            _run_async(self._delete_index_db(meta.id))
            return

        meta = self._memory_store.pop(index_id, None)
        if meta and self._storage and meta.storage_uri:
            try:
                self._storage.delete(meta.storage_uri)
            except Exception as e:
                logger.warning(
                    "Failed to delete index storage at %s: %s",
                    meta.storage_uri,
                    e,
                )

    # ── DB implementations ──────────────────────────────────────────────

    async def _list_indices_db(self, dataset_id: int | None) -> list[IndexMeta]:
        from sqlalchemy import select

        from carnot.storage.models import IndexEntryModel

        async with self._db_factory() as session:
            stmt = select(IndexEntryModel)
            if dataset_id is not None:
                stmt = stmt.where(IndexEntryModel.dataset_id == dataset_id)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._row_to_meta(r) for r in rows]

    async def _get_index_db(self, index_id: int) -> IndexMeta | None:
        from sqlalchemy import select

        from carnot.storage.models import IndexEntryModel

        async with self._db_factory() as session:
            stmt = select(IndexEntryModel).where(IndexEntryModel.id == index_id)
            result = await session.execute(stmt)
            r = result.scalar_one_or_none()
            return self._row_to_meta(r) if r else None

    async def _get_index_by_name_db(
        self, dataset_id: int, name: str
    ) -> IndexMeta | None:
        from sqlalchemy import select

        from carnot.storage.models import IndexEntryModel

        async with self._db_factory() as session:
            stmt = (
                select(IndexEntryModel)
                .where(IndexEntryModel.dataset_id == dataset_id)
                .where(IndexEntryModel.name == name)
            )
            result = await session.execute(stmt)
            r = result.scalar_one_or_none()
            return self._row_to_meta(r) if r else None

    async def _create_index_db(self, meta: IndexMeta) -> None:
        from carnot.storage.models import IndexEntryModel

        async with self._db_factory() as session:
            row = IndexEntryModel(
                dataset_id=meta.dataset_id,
                name=meta.name,
                index_type=meta.index_type,
                config_json=json.dumps(meta.config),
                storage_uri=meta.storage_uri,
                item_count=meta.item_count,
                is_stale=meta.is_stale,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            meta.id = row.id

    async def _update_index_db(self, meta: IndexMeta) -> None:
        from sqlalchemy import update

        from carnot.storage.models import IndexEntryModel

        async with self._db_factory() as session:
            stmt = (
                update(IndexEntryModel)
                .where(IndexEntryModel.id == meta.id)
                .values(
                    index_type=meta.index_type,
                    config_json=json.dumps(meta.config),
                    storage_uri=meta.storage_uri,
                    item_count=meta.item_count,
                    is_stale=meta.is_stale,
                )
            )
            await session.execute(stmt)
            await session.commit()

    async def _mark_stale_db(self, dataset_id: int) -> None:
        from sqlalchemy import update

        from carnot.storage.models import IndexEntryModel

        async with self._db_factory() as session:
            stmt = (
                update(IndexEntryModel)
                .where(IndexEntryModel.dataset_id == dataset_id)
                .values(is_stale=True)
            )
            await session.execute(stmt)
            await session.commit()

    async def _delete_index_db(self, index_id: int) -> None:
        from sqlalchemy import delete

        from carnot.storage.models import IndexEntryModel

        async with self._db_factory() as session:
            stmt = delete(IndexEntryModel).where(IndexEntryModel.id == index_id)
            await session.execute(stmt)
            await session.commit()

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_meta(row) -> IndexMeta:
        config = {}
        if row.config_json:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                config = json.loads(row.config_json)
        return IndexMeta(
            id=row.id,
            dataset_id=row.dataset_id,
            name=row.name,
            index_type=row.index_type,
            config=config,
            storage_uri=row.storage_uri or "",
            item_count=row.item_count,
            is_stale=row.is_stale,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    # ── Deserialization ─────────────────────────────────────────────────

    @staticmethod
    def _deserialize_index(
        index_type: str, data: bytes, config: dict
    ) -> CarnotIndex | None:
        """Deserialize raw bytes into the appropriate CarnotIndex subclass."""
        from carnot.index.sem_indices_cache import HierarchicalIndexCache

        if index_type in ("flat", "hierarchical"):
            try:
                cache_data = json.loads(data)
                cache = HierarchicalIndexCache()
                index = cache._deserialize(cache_data)
                return index
            except Exception as e:
                logger.error(
                    "Failed to deserialize %s index: %s", index_type, e
                )
                return None

        elif index_type == "faiss":
            # FAISS uses raw binary serialization
            with tempfile.NamedTemporaryFile(suffix=".index", delete=True) as tmp:
                tmp.write(data)
                tmp.flush()
                faiss_index = faiss.read_index(tmp.name)
                return faiss_index

        elif index_type == "chroma":
            # ChromaDB self-manages; we just store the collection_name + chroma_dir
            # The caller should use ChromaDB's PersistentClient directly
            logger.info("ChromaDB indices are self-managed; returning None for deserialization")
            return None

        else:
            logger.warning("Unknown index type: %s", index_type)
            return None
