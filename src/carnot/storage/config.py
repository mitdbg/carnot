"""StorageConfig — single source of truth for Carnot storage configuration.

``StorageConfig`` describes:

1. **Local filesystem paths** — the directory tree for data, indices,
   caches, etc.  Even when the durable backend is S3 the paths here
   point to the local machine.

2. **S3 backend** — optional bucket name and prefix for remote durable
   storage.

3. **Catalog backend** — whether the :class:`DatasetCatalog` and
   :class:`IndexCatalog` use a PostgreSQL database or fall back to
   simple in-memory dictionaries.  When Postgres is enabled the config
   provides connection parameters (with sensible environment-variable
   defaults that match the ``app/`` web deployment).

4. **Memory budget** — maximum megabytes for the L1 in-memory cache.

Remote (L3) storage is configured separately via ``S3StorageBackend``
and is transparent to consumers — the ``TieredStorageManager`` bridges
the two.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class StorageConfig:
    """Centralized configuration for all Carnot storage and catalog settings.

    Covers three concerns:

    * **Local paths** — every sub-directory is a child of ``base_dir``.
    * **Catalog backend** — ``catalog_backend`` selects ``"memory"``
      (default, no external DB required) or ``"postgres"`` (persistent
      catalog backed by PostgreSQL).  When ``"postgres"`` is chosen the
      connection is built from the ``db_*`` parameters, which default to
      environment variables matching the ``app/`` web deployment.
    * **S3 backend** — ``s3_bucket`` enables remote durable storage.

    Resolution order for ``base_dir``:

    1. Explicit *base_dir* argument (resolved and expanded).
    2. ``CARNOT_HOME`` environment variable (resolved and expanded).
    3. ``~/.carnot`` (not resolved — returned as-is).

    When *base_dir* is given explicitly or via ``CARNOT_HOME``, the
    resulting path is ``Path(value).expanduser().resolve()``.

    Usage::

        # Simplest — pure in-memory catalogs, local filesystem storage
        config = StorageConfig()

        # Explicit local root
        config = StorageConfig(base_dir="/tmp/carnot")

        # Postgres-backed catalogs (e.g. web deployment)
        config = StorageConfig(
            catalog_backend="postgres",
            db_user="carnotuser",
            db_password="secret",
            db_name="carnotdb",
        )

        # Override just the memory budget
        config = StorageConfig(memory_cache_max_mb=1024)

    Directory layout::

        <base_dir>/                  (always a local path)
        ├── data/                    # raw uploaded data files
        ├── routing/
        │   ├── summaries/           # per-file summary JSON caches
        │   └── indices/             # hierarchical index JSON caches
        ├── chroma/                  # ChromaDB PersistentClient data
        ├── faiss/                   # FAISS .index files
        └── cache/                   # L2 local disk cache for remote backends

    Representation invariant:
        - ``base_dir`` is always a :class:`~pathlib.Path`.
        - All sub-directory properties (``data_dir``, etc.) are children
          of ``base_dir``.
        - ``catalog_backend`` is one of ``"memory"`` or ``"postgres"``.
        - When ``catalog_backend == "postgres"``, ``db_user``,
          ``db_password``, and ``db_name`` are non-empty strings.

    Abstraction function:
        Represents the complete storage and catalog configuration for a
        single Carnot deployment.  Local paths describe where on-disk
        artifacts live; ``catalog_backend`` and ``db_*`` fields describe
        how catalog metadata is persisted; ``s3_bucket`` describes the
        optional remote durable backend; ``memory_cache_max_mb``
        controls the L1 cache budget.
    """

    # ── Catalog backend constants ───────────────────────────────────────

    CATALOG_MEMORY = "memory"
    CATALOG_POSTGRES = "postgres"
    _VALID_CATALOG_BACKENDS = {CATALOG_MEMORY, CATALOG_POSTGRES}

    # ── Default environment-variable names (match app/ deployment) ──────

    _DEFAULT_DB_USER_ENV = "CARNOT_DB_USER"
    _DEFAULT_DB_PASSWORD_ENV = "CARNOT_DB_PASSWORD"
    _DEFAULT_DB_NAME_ENV = "CARNOT_DB_NAME"
    _DEFAULT_DB_HOST_ENV = "CARNOT_DB_HOST"
    _DEFAULT_DB_PORT_ENV = "CARNOT_DB_PORT"

    def __init__(
        self,
        base_dir: str | Path | None = None,
        memory_cache_max_mb: int = 512,
        local_cache_dir: str | Path | None = None,
        # ── Catalog backend ─────────────────────────────────────────
        catalog_backend: str = CATALOG_MEMORY,
        db_user: str | None = None,
        db_password: str | None = None,
        db_name: str | None = None,
        db_host: str | None = None,
        db_port: int | str | None = None,
        db_session_factory: object | None = None,
        # ── S3 backend ──────────────────────────────────────────────
        s3_bucket: str | None = None,
        s3_prefix: str = "",
    ):
        # ── Resolve base directory ──────────────────────────────────
        if base_dir is not None:
            self._base_dir = Path(base_dir).expanduser().resolve()
        elif os.getenv("CARNOT_HOME"):
            self._base_dir = Path(os.getenv("CARNOT_HOME")).expanduser().resolve()
        else:
            self._base_dir = Path.home() / ".carnot"

        self.memory_cache_max_mb = memory_cache_max_mb

        # L2 cache dir
        if local_cache_dir is not None:
            self._local_cache_dir = Path(local_cache_dir).expanduser().resolve()
        else:
            self._local_cache_dir = self._base_dir / "cache"

        # ── Catalog backend ─────────────────────────────────────────
        if catalog_backend not in self._VALID_CATALOG_BACKENDS:
            raise ValueError(
                f"catalog_backend must be one of {self._VALID_CATALOG_BACKENDS!r}, "
                f"got {catalog_backend!r}"
            )
        self.catalog_backend = catalog_backend

        # When the caller provides an already-constructed session
        # factory (e.g. the web app passes ``AsyncSessionLocal``), use
        # it directly and skip connection-string assembly.
        self._db_session_factory = db_session_factory

        # Resolve DB connection parameters from explicit args or env
        self.db_user = db_user or os.getenv(self._DEFAULT_DB_USER_ENV, "carnotuser")
        self.db_password = db_password or os.getenv(self._DEFAULT_DB_PASSWORD_ENV, "")
        self.db_name = db_name or os.getenv(self._DEFAULT_DB_NAME_ENV, "carnotdb")
        self.db_host = db_host or os.getenv(self._DEFAULT_DB_HOST_ENV, "localhost")
        self.db_port = int(db_port or os.getenv(self._DEFAULT_DB_PORT_ENV, "5432"))

        # ── S3 backend ──────────────────────────────────────────────
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix

    # ── Path accessors ──────────────────────────────────────────────────

    @property
    def base_dir(self) -> Path:
        """Root directory for all Carnot-managed storage."""
        return self._base_dir

    @property
    def data_dir(self) -> Path:
        """Directory for raw data files."""
        return self._base_dir / "data"

    @property
    def summaries_dir(self) -> Path:
        """Directory for flat/hierarchical file summary caches."""
        return self._base_dir / "routing" / "summaries"

    @property
    def hierarchical_dir(self) -> Path:
        """Directory for hierarchical index caches."""
        return self._base_dir / "routing" / "indices"

    @property
    def chroma_dir(self) -> Path:
        """Directory for ChromaDB persistence."""
        return self._base_dir / "chroma"

    @property
    def faiss_dir(self) -> Path:
        """Directory for FAISS index files."""
        return self._base_dir / "faiss"

    @property
    def local_cache_dir(self) -> Path:
        """Directory for L2 local disk cache (relevant for remote backends)."""
        return self._local_cache_dir

    def ensure_dirs(self) -> None:
        """Create all storage directories if they don't exist.

        Creates: ``data_dir``, ``summaries_dir``, ``hierarchical_dir``,
         ``chroma_dir``, ``faiss_dir``, and ``local_cache_dir``.
        """
        for d in (
            self.data_dir,
            self.summaries_dir,
            self.hierarchical_dir,
            self.chroma_dir,
            self.faiss_dir,
            self.local_cache_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # ── Catalog helpers ─────────────────────────────────────────────────

    @property
    def has_postgres(self) -> bool:
        """True when the catalog backend is PostgreSQL."""
        return self.catalog_backend == self.CATALOG_POSTGRES

    @property
    def database_url(self) -> str:
        """Async PostgreSQL connection URL for ``asyncpg``.

        Requires:
            ``catalog_backend == "postgres"``.

        Returns:
            A ``postgresql+asyncpg://…`` URL.

        Raises:
            ValueError: if ``catalog_backend`` is not ``"postgres"``.
        """
        if not self.has_postgres:
            raise ValueError(
                "database_url is only available when catalog_backend='postgres'"
            )
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    def get_db_session_factory(self):
        """Return an ``async_sessionmaker`` for the configured database.

        If the caller supplied a ``db_session_factory`` at construction
        time, it is returned directly.  Otherwise a new async engine and
        session factory are created from the ``db_*`` parameters.

        Requires:
            ``catalog_backend == "postgres"``.

        Returns:
            An ``async_sessionmaker[AsyncSession]``.

        Raises:
            ValueError: if ``catalog_backend`` is not ``"postgres"``.
            ImportError: if ``sqlalchemy`` async extensions are not
            installed.
        """
        if self._db_session_factory is not None:
            return self._db_session_factory

        if not self.has_postgres:
            raise ValueError(
                "get_db_session_factory() requires catalog_backend='postgres'"
            )

        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        engine = create_async_engine(self.database_url, echo=False)
        factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        # Cache so subsequent calls return the same factory / engine
        self._db_session_factory = factory
        return factory

    def __repr__(self) -> str:
        """Return a string containing the ``base_dir`` and catalog backend."""
        return (
            f"StorageConfig(base_dir={self._base_dir!r}, "
            f"catalog_backend={self.catalog_backend!r})"
        )
