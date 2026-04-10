"""TieredStorageManager — transparent L1/L2/L3 caching over a StorageBackend.

L1: In-memory LRU cache (parsed dictionaries for hot data during query execution)
L2: Local disk cache (for warm data when the backend is remote, e.g. S3)
L3: The durable StorageBackend itself (local FS or S3)
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, BinaryIO

from carnot.storage.backend import StorageBackend
from carnot.storage.parsers import parse_file_contents

logger = logging.getLogger(__name__)


class LRUCache:
    """Simple size-bounded LRU cache storing arbitrary Python objects.

    ``max_size_bytes`` is *advisory* — we estimate the size of cached values
    by the byte-length of the raw data that produced them.  This is good
    enough for controlling memory pressure without requiring deep object-size
    introspection.

    Public interface:

    - ``get(key)`` → value or ``None``.
    - ``put(key, value, size_hint)`` → inserts, evicting oldest entries
      when ``current_size`` exceeds ``max_size_bytes``.
    - ``evict(key)`` → removes a specific key and adjusts ``current_size``.
    - ``clear()`` → removes all entries and resets ``current_size`` to 0.
    - ``current_size`` (property) → current total advisory size in bytes.
    - ``len(cache)`` → number of entries.
    - ``key in cache`` → membership test.

    Representation invariant:
        - ``current_size`` == sum of ``size_hint`` for all stored entries.
        - ``current_size >= 0``.
        - After ``put(k, v, s)``, ``get(k) == v``.
        - After ``evict(k)`` or ``clear()``, ``get(k) is None``.
    """

    def __init__(self, max_size_bytes: int = 512 * 1024 * 1024):
        self._max_size = max_size_bytes
        self._cache: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._current_size = 0

    @property
    def current_size(self) -> int:
        return self._current_size

    def get(self, key: str) -> Any | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key][0]

    def put(self, key: str, value: Any, size_hint: int = 0) -> None:
        """Insert *value* with an estimated byte size of *size_hint*."""
        if key in self._cache:
            # Update existing entry
            old_value, old_size = self._cache.pop(key)
            self._current_size -= old_size

        self._cache[key] = (value, size_hint)
        self._current_size += size_hint
        self._cache.move_to_end(key)
        self._evict_if_needed()

    def evict(self, key: str) -> None:
        if key in self._cache:
            _, size = self._cache.pop(key)
            self._current_size -= size

    def clear(self) -> None:
        self._cache.clear()
        self._current_size = 0

    def _evict_if_needed(self) -> None:
        while self._current_size > self._max_size and self._cache:
            _, (_, size) = self._cache.popitem(last=False)
            self._current_size -= size

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache


class TieredStorageManager:
    """Transparent tiered caching over a :class:`StorageBackend`.

    The query executor calls ``storage.read(uri)`` and doesn't know which
    tier served the data.

    **L1 caches parsed dictionaries** (not raw bytes) per the resolved design
    decisions.  Use ``read_parsed(uri)`` to get a parsed dict for a data file,
    or ``read(uri)`` for raw bytes with L1 caching.

    Tiered caching behavior:

    - ``read(uri)`` checks L1 (in-memory), then L2 (local disk, if
      configured), then L3 (the durable backend).  Successful reads
      populate all skipped tiers.
    - ``write(uri, data)`` writes to the backend and updates L1 and L2.
      Any existing parsed cache for the URI is invalidated.
    - ``invalidate(uri)`` evicts a URI from L1 and L2 (both raw and
      parsed entries).
    - ``delete(uri)`` removes from the backend and then calls
      ``invalidate(uri)``.
    - ``exists(uri)``, ``list(prefix)``, and ``get_uri(*parts)``
      delegate directly to the backend.
    - ``read_parsed(uri)`` returns a parsed ``dict`` with at least a
      ``"contents"`` key.  Parsed results are cached in L1 separately
      from raw bytes.

    When ``local_cache_dir`` is not supplied, the L2 tier is disabled
    and reads fall through directly from L1 to L3.
    """

    def __init__(
        self,
        backend: StorageBackend,
        memory_cache_max_mb: int = 512,
        local_cache_dir: str | Path | None = None,
    ):
        self._backend = backend
        self._l1 = LRUCache(max_size_bytes=memory_cache_max_mb * 1024 * 1024)
        self._l2_dir = Path(local_cache_dir) if local_cache_dir else None
        if self._l2_dir:
            self._l2_dir.mkdir(parents=True, exist_ok=True)

    @property
    def backend(self) -> StorageBackend:
        return self._backend

    # ── Raw byte access (L1 caches raw bytes) ──────────────────────────

    def read(self, uri: str) -> bytes:
        """Read raw bytes, using L1 and L2 caches."""
        cache_key = f"raw:{uri}"

        # L1 check
        cached = self._l1.get(cache_key)
        if cached is not None:
            return cached

        # L2 check
        if self._l2_dir:
            data = self._l2_read(uri)
            if data is not None:
                self._l1.put(cache_key, data, size_hint=len(data))
                return data

        # L3: durable backend
        data = self._backend.read(uri)
        self._l1.put(cache_key, data, size_hint=len(data))
        if self._l2_dir:
            self._l2_write(uri, data)
        return data

    def read_parsed(self, uri: str) -> dict:
        """Read a data file and return a parsed dict (cached in L1).

        This is the primary method used by ``DataItem.materialize()`` and
        ``Dataset.materialize()`` during execution.
        """
        cache_key = f"parsed:{uri}"

        cached = self._l1.get(cache_key)
        if cached is not None:
            return cached

        raw = self.read(uri)
        contents = parse_file_contents(uri, raw)
        parsed = {"contents": contents, "path": uri}
        # Estimate size as the raw byte length
        self._l1.put(cache_key, parsed, size_hint=len(raw))
        return parsed

    def read_stream(self, uri: str) -> BinaryIO:
        """Return a readable binary stream (no L1 caching)."""
        return self._backend.read_stream(uri)

    def write(self, uri: str, data: bytes) -> None:
        """Write raw bytes to the backend, updating caches."""
        self._backend.write(uri, data)
        cache_key = f"raw:{uri}"
        self._l1.put(cache_key, data, size_hint=len(data))
        # Invalidate any parsed cache for this URI
        self._l1.evict(f"parsed:{uri}")
        if self._l2_dir:
            self._l2_write(uri, data)

    def write_stream(self, uri: str, stream: BinaryIO) -> None:
        """Write a stream to the backend."""
        self._backend.write_stream(uri, stream)
        # Invalidate caches since we can't easily cache the stream
        self._l1.evict(f"raw:{uri}")
        self._l1.evict(f"parsed:{uri}")

    def exists(self, uri: str) -> bool:
        return self._backend.exists(uri)

    def delete(self, uri: str) -> None:
        self._backend.delete(uri)
        self.invalidate(uri)

    def list(self, prefix: str) -> list[str]:
        return self._backend.list(prefix)

    def get_uri(self, *path_parts: str) -> str:
        return self._backend.get_uri(*path_parts)

    def invalidate(self, uri: str) -> None:
        """Remove all cached entries for a given URI."""
        self._l1.evict(f"raw:{uri}")
        self._l1.evict(f"parsed:{uri}")
        if self._l2_dir:
            self._l2_delete(uri)

    # ── L2 local disk cache ─────────────────────────────────────────────

    def _l2_key(self, uri: str) -> Path:
        """Map a URI to a local cache file path."""
        h = hashlib.sha256(uri.encode()).hexdigest()
        return self._l2_dir / h[:2] / h[2:]

    def _l2_read(self, uri: str) -> bytes | None:
        path = self._l2_key(uri)
        if path.exists():
            try:
                return path.read_bytes()
            except OSError:
                return None
        return None

    def _l2_write(self, uri: str, data: bytes) -> None:
        path = self._l2_key(uri)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        except OSError as e:
            logger.warning("L2 cache write failed for %s: %s", uri, e)

    def _l2_delete(self, uri: str) -> None:
        path = self._l2_key(uri)
        if path.exists():
            with contextlib.suppress(OSError):
                path.unlink()
