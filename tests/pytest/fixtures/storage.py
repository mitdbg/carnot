"""Storage-layer fixtures for the Carnot test suite.

Provides:
    - ``InMemoryStorageBackend``: a fast, in-memory implementation of
      :class:`StorageBackend` for tests that don't need real I/O.
    - ``local_backend``: a ``LocalStorageBackend`` rooted at ``tmp_path``
      for tests that exercise real filesystem operations.
    - ``in_memory_backend``: a fresh ``InMemoryStorageBackend`` per test.
    - ``tiered_storage``: a ``TieredStorageManager`` backed by an
      in-memory backend (no disk I/O).
"""

from __future__ import annotations

import io
from typing import BinaryIO

import pytest

from carnot.storage.backend import LocalStorageBackend, StorageBackend
from carnot.storage.tiered import TieredStorageManager

# ── In-memory backend ───────────────────────────────────────────────────────


class InMemoryStorageBackend(StorageBackend):
    """Fully in-memory :class:`StorageBackend` for fast, isolated tests.

    All data lives in a plain ``dict[str, bytes]``.  No filesystem or
    network access is performed.

    Representation invariant:
        - ``_store`` maps URI strings to ``bytes`` objects.
        - Every URI returned by :meth:`list` exists as a key in ``_store``.

    Abstraction function:
        Represents a flat key-value store where each URI is a unique
        storage object whose contents are the corresponding ``bytes``.
    """

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def read(self, uri: str) -> bytes:
        """Read the contents stored at *uri*.

        Requires:
            - *uri* exists in the store.

        Raises:
            KeyError: if *uri* has not been written.
        """
        if uri not in self._store:
            raise KeyError(f"InMemoryStorageBackend: URI not found: {uri}")
        return self._store[uri]

    def read_stream(self, uri: str) -> BinaryIO:
        """Return a readable binary stream for *uri*.

        Requires:
            - *uri* exists in the store.

        Raises:
            KeyError: if *uri* has not been written.
        """
        return io.BytesIO(self.read(uri))

    def write(self, uri: str, data: bytes) -> None:
        """Write *data* to *uri*, overwriting if it already exists."""
        self._store[uri] = data

    def write_stream(self, uri: str, stream: BinaryIO) -> None:
        """Write the contents of *stream* to *uri*."""
        self._store[uri] = stream.read()

    def exists(self, uri: str) -> bool:
        """Return ``True`` if *uri* has been written."""
        return uri in self._store

    def delete(self, uri: str) -> None:
        """Delete *uri*. No-op if it doesn't exist."""
        self._store.pop(uri, None)

    def list(self, prefix: str) -> list[str]:
        """Return all URIs whose key starts with *prefix*, sorted."""
        return sorted(k for k in self._store if k.startswith(prefix))

    def get_uri(self, *path_parts: str) -> str:
        """Join *path_parts* with ``/`` to form a URI."""
        return "/".join(path_parts)

    # ── Test helpers ────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of objects in the store."""
        return len(self._store)

    def clear(self) -> None:
        """Remove all stored objects."""
        self._store.clear()


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def in_memory_backend() -> InMemoryStorageBackend:
    """A fresh in-memory storage backend for each test.

    Returns:
        An empty ``InMemoryStorageBackend``.
    """
    return InMemoryStorageBackend()


@pytest.fixture
def local_backend(tmp_path) -> LocalStorageBackend:
    """A ``LocalStorageBackend`` rooted at a temporary directory.

    The directory is cleaned up automatically by pytest after the test
    session.

    Returns:
        A ``LocalStorageBackend`` whose ``base_dir`` is *tmp_path*.
    """
    return LocalStorageBackend(base_dir=tmp_path)


@pytest.fixture
def tiered_storage(in_memory_backend, tmp_path) -> TieredStorageManager:
    """A ``TieredStorageManager`` backed by in-memory storage.

    L1 is a small memory cache (1 MB); L2 uses a temp directory.
    Suitable for tests that exercise the tiered-read path without
    real I/O latency.

    Returns:
        A ``TieredStorageManager`` ready for use.
    """
    return TieredStorageManager(
        backend=in_memory_backend,
        memory_cache_max_mb=1,
        local_cache_dir=str(tmp_path / "l2_cache"),
    )
