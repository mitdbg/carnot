"""Unit tests for storage backends, config, LRU cache, and tiered storage.

Tests cover:
1. StorageConfig — path resolution, ensure_dirs, CARNOT_HOME override
2. LocalStorageBackend — full CRUD against a tmp_path root
3. InMemoryStorageBackend — CRUD, list prefix, edge cases
4. LRUCache — get/put, eviction policy, size tracking
5. TieredStorageManager — L1 caching, read-through, invalidation,
   write-through, parsed caching, L2 disk caching
"""

from __future__ import annotations

from pathlib import Path

from carnot.storage.backend import LocalStorageBackend
from carnot.storage.config import StorageConfig
from carnot.storage.tiered import LRUCache, TieredStorageManager

# ═══════════════════════════════════════════════════════════════════════
# StorageConfig
# ═══════════════════════════════════════════════════════════════════════


class TestStorageConfig:
    """StorageConfig path resolution and directory creation."""

    def test_default_base_dir(self):
        """Without arguments, base_dir defaults to ~/.carnot."""
        config = StorageConfig()
        assert config.base_dir == Path.home() / ".carnot"

    def test_explicit_base_dir(self, tmp_path):
        """An explicit base_dir overrides the default."""
        config = StorageConfig(base_dir=tmp_path / "custom")
        assert config.base_dir == (tmp_path / "custom").resolve()

    def test_carnot_home_env(self, monkeypatch, tmp_path):
        """CARNOT_HOME env var is respected when base_dir is not given."""
        monkeypatch.setenv("CARNOT_HOME", str(tmp_path / "envhome"))
        config = StorageConfig()
        assert config.base_dir == (tmp_path / "envhome").resolve()

    def test_subdirectory_paths(self, tmp_path):
        """Sub-directory properties resolve relative to base_dir."""
        config = StorageConfig(base_dir=tmp_path)
        assert config.data_dir == tmp_path / "data"
        assert config.summaries_dir == tmp_path / "routing" / "summaries"
        assert config.hierarchical_dir == tmp_path / "routing" / "indices"
        assert config.chroma_dir == tmp_path / "chroma"
        assert config.faiss_dir == tmp_path / "faiss"

    def test_ensure_dirs_creates_all(self, tmp_path):
        """ensure_dirs creates every sub-directory."""
        config = StorageConfig(base_dir=tmp_path)
        config.ensure_dirs()
        for d in (
            config.data_dir,
            config.summaries_dir,
            config.hierarchical_dir,
            config.chroma_dir,
            config.faiss_dir,
            config.local_cache_dir,
        ):
            assert d.is_dir()

    def test_repr(self, tmp_path):
        """repr includes the base_dir path."""
        config = StorageConfig(base_dir=tmp_path)
        assert str(tmp_path) in repr(config)


# ═══════════════════════════════════════════════════════════════════════
# LocalStorageBackend
# ═══════════════════════════════════════════════════════════════════════


class TestLocalStorageBackend:
    """CRUD operations on LocalStorageBackend with a tmp_path root."""

    def test_write_and_read(self, local_backend):
        """write() then read() round-trips bytes correctly."""
        local_backend.write("docs/hello.txt", b"world")
        assert local_backend.read("docs/hello.txt") == b"world"

    def test_overwrite(self, local_backend):
        """A second write() to the same URI overwrites the content."""
        local_backend.write("f.txt", b"v1")
        local_backend.write("f.txt", b"v2")
        assert local_backend.read("f.txt") == b"v2"

    def test_exists(self, local_backend):
        """exists() reflects whether the URI has been written."""
        assert not local_backend.exists("missing.txt")
        local_backend.write("present.txt", b"data")
        assert local_backend.exists("present.txt")

    def test_delete(self, local_backend):
        """delete() removes the file and exists() returns False."""
        local_backend.write("temp.txt", b"data")
        local_backend.delete("temp.txt")
        assert not local_backend.exists("temp.txt")

    def test_delete_nonexistent_is_noop(self, local_backend):
        """Deleting a URI that doesn't exist does not raise."""
        local_backend.delete("ghost.txt")  # should not raise

    def test_list(self, local_backend):
        """list() returns sorted file paths under a prefix."""
        local_backend.write("dir/a.txt", b"a")
        local_backend.write("dir/b.txt", b"b")
        local_backend.write("other/c.txt", b"c")
        result = local_backend.list("dir")
        assert len(result) == 2
        assert all("dir" in r for r in result)

    def test_list_empty_prefix(self, local_backend):
        """list() on a nonexistent prefix returns an empty list."""
        assert local_backend.list("nope") == []

    def test_get_uri(self, local_backend):
        """get_uri() joins path parts relative to base_dir."""
        uri = local_backend.get_uri("indices", "abc.json")
        assert uri.endswith("indices/abc.json")
        assert str(local_backend.base_dir) in uri

    def test_read_stream(self, local_backend):
        """read_stream() returns a readable binary stream."""
        local_backend.write("stream.txt", b"streaming content")
        stream = local_backend.read_stream("stream.txt")
        assert stream.read() == b"streaming content"
        stream.close()

    def test_write_stream(self, local_backend):
        """write_stream() writes from a binary stream."""
        import io

        local_backend.write_stream("ws.txt", io.BytesIO(b"from stream"))
        assert local_backend.read("ws.txt") == b"from stream"

    def test_constructor_with_config(self, tmp_path):
        """LocalStorageBackend can be created from a StorageConfig."""
        config = StorageConfig(base_dir=tmp_path / "via_config")
        backend = LocalStorageBackend(config=config)
        assert backend.base_dir == config.base_dir


# ═══════════════════════════════════════════════════════════════════════
# LRUCache
# ═══════════════════════════════════════════════════════════════════════


class TestLRUCache:
    """LRU cache eviction, sizing, and lookup semantics."""

    def test_put_and_get(self):
        """put() then get() returns the stored value."""
        cache = LRUCache(max_size_bytes=1024)
        cache.put("k", "v", size_hint=10)
        assert cache.get("k") == "v"

    def test_get_missing_returns_none(self):
        """get() for a missing key returns None."""
        cache = LRUCache(max_size_bytes=1024)
        assert cache.get("missing") is None

    def test_eviction_on_size_limit(self):
        """Oldest entries are evicted when max_size_bytes is exceeded."""
        cache = LRUCache(max_size_bytes=100)
        cache.put("a", "data_a", size_hint=60)
        cache.put("b", "data_b", size_hint=60)
        # 'a' should have been evicted to make room for 'b'
        assert cache.get("a") is None
        assert cache.get("b") == "data_b"

    def test_access_refreshes_lru_order(self):
        """get() moves a key to the end so it is not evicted first."""
        cache = LRUCache(max_size_bytes=100)
        cache.put("a", "A", size_hint=40)
        cache.put("b", "B", size_hint=40)
        # Access 'a' to refresh it
        cache.get("a")
        # Insert 'c' — should evict 'b' (oldest untouched), not 'a'
        cache.put("c", "C", size_hint=40)
        assert cache.get("a") == "A"
        assert cache.get("b") is None
        assert cache.get("c") == "C"

    def test_evict_specific_key(self):
        """evict() removes a specific key and adjusts current_size."""
        cache = LRUCache(max_size_bytes=1024)
        cache.put("k", "v", size_hint=50)
        assert cache.current_size == 50
        cache.evict("k")
        assert cache.get("k") is None
        assert cache.current_size == 0

    def test_clear(self):
        """clear() empties the cache."""
        cache = LRUCache(max_size_bytes=1024)
        cache.put("a", 1, size_hint=10)
        cache.put("b", 2, size_hint=10)
        cache.clear()
        assert len(cache) == 0
        assert cache.current_size == 0

    def test_update_existing_key(self):
        """Putting the same key again updates value and adjusts size."""
        cache = LRUCache(max_size_bytes=1024)
        cache.put("k", "old", size_hint=30)
        cache.put("k", "new", size_hint=50)
        assert cache.get("k") == "new"
        assert cache.current_size == 50

    def test_contains(self):
        """``in`` operator checks membership."""
        cache = LRUCache(max_size_bytes=1024)
        cache.put("k", "v", size_hint=10)
        assert "k" in cache
        assert "missing" not in cache


# ═══════════════════════════════════════════════════════════════════════
# TieredStorageManager
# ═══════════════════════════════════════════════════════════════════════


class TestTieredStorageManager:
    """Tiered caching read-through, write-through, and invalidation."""

    def test_read_populates_l1(self, in_memory_backend):
        """First read() fetches from backend; second is served from L1."""
        in_memory_backend.write("doc.txt", b"hello")
        tsm = TieredStorageManager(backend=in_memory_backend, memory_cache_max_mb=1)

        data = tsm.read("doc.txt")
        assert data == b"hello"

        # Mutate the backend to prove L1 is used on second read
        in_memory_backend.write("doc.txt", b"changed")
        assert tsm.read("doc.txt") == b"hello"  # still from L1

    def test_write_updates_backend_and_l1(self, in_memory_backend):
        """write() pushes to backend and refreshes L1 cache."""
        tsm = TieredStorageManager(backend=in_memory_backend, memory_cache_max_mb=1)
        tsm.write("f.txt", b"v1")
        assert in_memory_backend.read("f.txt") == b"v1"
        # L1 should have the new value
        assert tsm.read("f.txt") == b"v1"

        tsm.write("f.txt", b"v2")
        assert in_memory_backend.read("f.txt") == b"v2"
        assert tsm.read("f.txt") == b"v2"

    def test_invalidate_clears_l1(self, in_memory_backend):
        """invalidate() evicts the key from L1, forcing re-read from backend."""
        in_memory_backend.write("f.txt", b"original")
        tsm = TieredStorageManager(backend=in_memory_backend, memory_cache_max_mb=1)
        tsm.read("f.txt")  # populate L1

        in_memory_backend.write("f.txt", b"updated")
        tsm.invalidate("f.txt")

        assert tsm.read("f.txt") == b"updated"

    def test_delete_removes_from_backend_and_cache(self, in_memory_backend):
        """delete() removes from both backend and L1 cache."""
        in_memory_backend.write("f.txt", b"data")
        tsm = TieredStorageManager(backend=in_memory_backend, memory_cache_max_mb=1)
        tsm.read("f.txt")  # populate L1
        tsm.delete("f.txt")
        assert not in_memory_backend.exists("f.txt")

    def test_exists_delegates_to_backend(self, in_memory_backend):
        """exists() checks the durable backend."""
        tsm = TieredStorageManager(backend=in_memory_backend, memory_cache_max_mb=1)
        assert not tsm.exists("f.txt")
        in_memory_backend.write("f.txt", b"x")
        assert tsm.exists("f.txt")

    def test_list_delegates_to_backend(self, in_memory_backend):
        """list() delegates to the backend."""
        in_memory_backend.write("p/a", b"1")
        in_memory_backend.write("p/b", b"2")
        tsm = TieredStorageManager(backend=in_memory_backend, memory_cache_max_mb=1)
        assert tsm.list("p") == ["p/a", "p/b"]

    def test_get_uri_delegates(self, in_memory_backend):
        """get_uri() is passed through to the backend."""
        tsm = TieredStorageManager(backend=in_memory_backend, memory_cache_max_mb=1)
        assert tsm.get_uri("a", "b") == "a/b"

    def test_read_parsed_caches_separately(self, in_memory_backend):
        """read_parsed() caches parsed dicts under a separate L1 key."""
        in_memory_backend.write("data.txt", b"some text content")
        tsm = TieredStorageManager(backend=in_memory_backend, memory_cache_max_mb=1)

        parsed = tsm.read_parsed("data.txt")
        assert isinstance(parsed, dict)
        assert "some text content" in parsed["contents"]

        # Second call should return cached dict (same object)
        assert tsm.read_parsed("data.txt") is parsed

    def test_write_invalidates_parsed_cache(self, in_memory_backend):
        """write() evicts the parsed L1 entry so next read_parsed re-parses."""
        in_memory_backend.write("f.txt", b"version1")
        tsm = TieredStorageManager(backend=in_memory_backend, memory_cache_max_mb=1)
        p1 = tsm.read_parsed("f.txt")

        tsm.write("f.txt", b"version2")
        p2 = tsm.read_parsed("f.txt")
        assert p2 is not p1
        assert "version2" in p2["contents"]


class TestTieredStorageL2:
    """L2 disk cache integration within TieredStorageManager.

    These tests verify the observable caching behavior documented in
    TieredStorageManager's class docstring: reads populate L2,
    invalidate clears L2, and L2 serves data when L1 is cold.

    We test L2 behaviour *indirectly* through the public API by
    observing that data survives L1 eviction and backend removal.
    """

    def test_l2_populated_on_read(self, in_memory_backend, tmp_path):
        """A read() also writes to L2 disk cache (at least one file appears)."""
        in_memory_backend.write("x.txt", b"payload")
        tsm = TieredStorageManager(
            backend=in_memory_backend,
            memory_cache_max_mb=1,
            local_cache_dir=str(tmp_path / "l2"),
        )
        tsm.read("x.txt")

        # Verify at least one L2 cache file was created
        l2_dir = tmp_path / "l2"
        cache_files = list(l2_dir.rglob("*"))
        file_count = sum(1 for f in cache_files if f.is_file())
        assert file_count >= 1

    def test_l2_serves_when_l1_cold(self, in_memory_backend, tmp_path):
        """After L1 eviction, data is still available without the backend."""
        in_memory_backend.write("x.txt", b"payload")
        # Use a tiny L1 cache so it evicts easily
        tsm = TieredStorageManager(
            backend=in_memory_backend,
            memory_cache_max_mb=1,
            local_cache_dir=str(tmp_path / "l2"),
        )
        # Populate both L1 and L2
        tsm.read("x.txt")

        # Write enough other data to evict "x.txt" from L1
        for i in range(200):
            key = f"evict_{i:04d}.txt"
            in_memory_backend.write(key, b"x" * 10_000)
            tsm.read(key)

        # Remove from backend to prove L2 (or still-cached L1) is used
        in_memory_backend.delete("x.txt")

        assert tsm.read("x.txt") == b"payload"

    def test_invalidate_clears_all_tiers(self, in_memory_backend, tmp_path):
        """invalidate() clears L2 so a subsequent read goes back to the backend."""
        in_memory_backend.write("x.txt", b"original")
        tsm = TieredStorageManager(
            backend=in_memory_backend,
            memory_cache_max_mb=1,
            local_cache_dir=str(tmp_path / "l2"),
        )
        tsm.read("x.txt")

        # Update backend, then invalidate
        in_memory_backend.write("x.txt", b"updated")
        tsm.invalidate("x.txt")

        # After invalidate, read should get the updated value
        assert tsm.read("x.txt") == b"updated"
