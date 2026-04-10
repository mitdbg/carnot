"""StorageBackend — abstract interface for raw byte-level storage operations.

Provides ``LocalStorageBackend`` (local filesystem) and ``S3StorageBackend``
(Amazon S3) implementations.  Both are synchronous — the execution engine is
sync; FastAPI routes can wrap calls in ``run_in_threadpool()``.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

import s3fs

from carnot.storage.config import StorageConfig


class StorageBackend(ABC):
    """Abstract interface for raw byte-level storage operations.

    URIs are plain strings whose format depends on the backend:

    * ``LocalStorageBackend``: absolute filesystem paths (``/home/user/.carnot/indices/abc.json``)
    * ``S3StorageBackend``: S3 keys (``s3://bucket/prefix/indices/abc.json``)

    Representation invariant:
        - Every URI returned by ``get_uri`` can be round-tripped through
          ``write`` → ``read`` without data loss.

    Abstraction function:
        Represents a key→bytes store where each key is a URI string and
        the value is a raw byte sequence.
    """

    @abstractmethod
    def read(self, uri: str) -> bytes:
        """Read the entire contents of *uri*.

        Requires:
            - *uri* refers to an existing object in this backend.

        Returns:
            The raw bytes stored at *uri*.

        Raises:
            ``FileNotFoundError`` (or equivalent) if the object does not exist.
        """
        ...

    @abstractmethod
    def read_stream(self, uri: str) -> BinaryIO:
        """Return a readable binary stream for *uri*.

        Requires:
            - *uri* refers to an existing object in this backend.

        Returns:
            An open ``BinaryIO`` stream positioned at the start.
            The caller is responsible for closing the stream.

        Raises:
            ``FileNotFoundError`` (or equivalent) if the object does not exist.
        """
        ...

    @abstractmethod
    def write(self, uri: str, data: bytes) -> None:
        """Write *data* to *uri*, creating or overwriting as needed.

        Requires:
            - *data* is a ``bytes`` object.

        Returns:
            ``None``.  After the call, ``read(uri)`` returns *data*.

        Raises:
            ``OSError`` (or equivalent) if the write cannot be completed
            (e.g., permissions, disk full).
        """
        ...

    @abstractmethod
    def write_stream(self, uri: str, stream: BinaryIO) -> None:
        """Write the contents of *stream* to *uri*.

        Requires:
            - *stream* is a readable ``BinaryIO`` positioned at the
              desired start offset.

        Returns:
            ``None``.  After the call, ``read(uri)`` returns the bytes
            that were in *stream*.

        Raises:
            ``OSError`` (or equivalent) if the write cannot be completed.
        """
        ...

    @abstractmethod
    def exists(self, uri: str) -> bool:
        """Check whether an object exists at *uri*.

        Returns:
            ``True`` iff an object currently exists at *uri*.

        Raises:
            None.
        """
        ...

    @abstractmethod
    def delete(self, uri: str) -> None:
        """Delete the object at *uri*.  No-op if it doesn't exist.

        Returns:
            ``None``.  After the call, ``exists(uri)`` returns ``False``.

        Raises:
            None.
        """
        ...

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        """List all object URIs under *prefix*.

        Requires:
            - *prefix* is a string (possibly empty).

        Returns:
            A (possibly empty) list of URI strings for objects whose
            URI starts with *prefix*.

        Raises:
            None.  Returns ``[]`` if the prefix does not exist.
        """
        ...

    @abstractmethod
    def get_uri(self, *path_parts: str) -> str:
        """Construct a canonical URI from path components.

        Requires:
            - Each element of *path_parts* is a non-empty path segment.

        Returns:
            A single URI string produced by joining the backend's base
            location with *path_parts*.

        Raises:
            None.
        """
        ...


# ─── Local Filesystem ────────────────────────────────────────────────────────

class LocalStorageBackend(StorageBackend):
    """Storage backed by the local filesystem.

    All URIs are resolved relative to *base_dir*.  ``get_uri("indices", "abc.json")``
    produces ``"<base_dir>/indices/abc.json"``.

    Representation invariant:
        - ``_base`` is an absolute, resolved ``Path`` that exists on disk
          (created in ``__init__`` via ``mkdir``).

    Abstraction function:
        Maps URI strings to files under ``_base``.  Absolute URIs are
        used as-is; relative URIs are joined with ``_base``.

    Args:
        base_dir: Root directory for all storage operations.  Defaults to
            the value from ``StorageConfig``.
        config: Alternative way to supply the base directory via a
            ``StorageConfig`` instance (takes precedence over *base_dir*
            if both are provided).
    """

    def __init__(self, base_dir: str | Path | None = None, config: StorageConfig | None = None):
        if config is not None:
            self._base = config.base_dir
        elif base_dir is not None:
            self._base = Path(base_dir).expanduser().resolve()
        else:
            self._base = StorageConfig().base_dir
        self._base.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base

    # ── helpers ──

    def _resolve(self, uri: str) -> Path:
        """Resolve a URI to an absolute filesystem path.

        If *uri* is already absolute it is returned as-is.  Otherwise it is
        treated as relative to ``base_dir``.
        """
        p = Path(uri)
        if p.is_absolute():
            return p
        return self._base / uri

    # ── StorageBackend API ──

    def read(self, uri: str) -> bytes:
        return self._resolve(uri).read_bytes()

    def read_stream(self, uri: str) -> BinaryIO:
        return open(self._resolve(uri), "rb")  # noqa: SIM115

    def write(self, uri: str, data: bytes) -> None:
        path = self._resolve(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def write_stream(self, uri: str, stream: BinaryIO) -> None:
        path = self._resolve(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            shutil.copyfileobj(stream, f)

    def exists(self, uri: str) -> bool:
        return self._resolve(uri).exists()

    def delete(self, uri: str) -> None:
        path = self._resolve(uri)
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    def list(self, prefix: str) -> list[str]:
        root = self._resolve(prefix)
        if not root.exists():
            return []
        if root.is_file():
            return [str(root)]
        return sorted(str(p) for p in root.rglob("*") if p.is_file())

    def get_uri(self, *path_parts: str) -> str:
        return str(self._base.joinpath(*path_parts))


# ─── Amazon S3 ───────────────────────────────────────────────────────────────

class S3StorageBackend(StorageBackend):
    """Storage backed by Amazon S3.

    Uses ``s3fs`` internally for all S3 operations.  ``get_uri("indices", "abc.json")``
    produces ``"s3://<bucket>/<prefix>/indices/abc.json"``.

    The S3 filesystem handle (``_fs``) is created lazily on first use so
    that the constructor never makes network calls.

    Representation invariant:
        - ``_bucket`` is a non-empty string (no ``s3://`` scheme prefix).
        - ``_prefix`` has no leading or trailing slashes.
        - ``_fs`` is ``None`` until the first I/O operation, after which it
          is an ``s3fs.S3FileSystem`` instance.

    Abstraction function:
        Maps URI strings to S3 objects under ``s3://<_bucket>/<_prefix>/``.
        URIs already starting with ``s3://`` are used as-is.

    Args:
        bucket: S3 bucket name (no ``s3://`` prefix).
        prefix: Optional key prefix within the bucket (no leading/trailing slashes).
        **s3_kwargs: Extra keyword arguments forwarded to
            ``s3fs.S3FileSystem`` (e.g., ``anon``, ``key``, ``secret``).
    """

    def __init__(self, bucket: str, prefix: str = "", **s3_kwargs):
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._s3_kwargs = s3_kwargs

        # lazy-import so S3 deps are optional for local-only users
        self._fs = None

    def _get_fs(self):
        """Return the cached ``s3fs.S3FileSystem``, creating it on first call."""
        if self._fs is None:
            self._fs = s3fs.S3FileSystem(**self._s3_kwargs)
        return self._fs

    def _full_key(self, uri: str) -> str:
        """Resolve *uri* to a full S3 key (``bucket/prefix/uri``).

        If *uri* already starts with ``s3://``, the scheme is stripped and the
        remainder is returned as-is.
        """
        # if already an s3:// path, use as-is
        if uri.startswith("s3://"):
            return uri.replace("s3://", "", 1)
        parts = [self._bucket]
        if self._prefix:
            parts.append(self._prefix)
        parts.append(uri.lstrip("/"))
        return "/".join(parts)

    def _s3_path(self, uri: str) -> str:
        """Return the full ``s3://`` path for *uri*."""
        return f"s3://{self._full_key(uri)}"

    # ── StorageBackend API ──

    def read(self, uri: str) -> bytes:
        """Read the entire contents of an S3 object.

        Requires:
            - *uri* identifies an existing S3 object (relative to the
              configured bucket/prefix, or an absolute ``s3://`` path).

        Returns:
            The raw bytes of the object.

        Raises:
            FileNotFoundError: If the object does not exist.
        """
        fs = self._get_fs()
        with fs.open(self._full_key(uri), "rb") as f:
            return f.read()

    def read_stream(self, uri: str) -> BinaryIO:
        """Open an S3 object as a readable binary stream.

        Requires:
            - *uri* identifies an existing S3 object.

        Returns:
            A binary file-like object.  Caller is responsible for closing it.

        Raises:
            FileNotFoundError: If the object does not exist.
        """
        fs = self._get_fs()
        return fs.open(self._full_key(uri), "rb")

    def write(self, uri: str, data: bytes) -> None:
        """Write *data* to an S3 object, creating or overwriting it.

        Requires:
            - *data* is a ``bytes`` instance.

        Returns:
            None.

        Raises:
            PermissionError: If the credentials lack write access.
        """
        fs = self._get_fs()
        with fs.open(self._full_key(uri), "wb") as f:
            f.write(data)

    def write_stream(self, uri: str, stream: BinaryIO) -> None:
        """Copy *stream* into an S3 object.

        Requires:
            - *stream* is a readable binary file-like object.

        Returns:
            None.

        Raises:
            PermissionError: If the credentials lack write access.
        """
        fs = self._get_fs()
        with fs.open(self._full_key(uri), "wb") as f:
            shutil.copyfileobj(stream, f)

    def exists(self, uri: str) -> bool:
        """Check whether an S3 object exists.

        Returns:
            ``True`` if the object exists, ``False`` otherwise.

        Raises:
            None.
        """
        fs = self._get_fs()
        return fs.exists(self._full_key(uri))

    def delete(self, uri: str) -> None:
        """Delete an S3 object or prefix recursively.

        If the object does not exist, this is a no-op.

        Returns:
            None.

        Raises:
            PermissionError: If the credentials lack delete access.
        """
        fs = self._get_fs()
        key = self._full_key(uri)
        if fs.exists(key):
            fs.rm(key, recursive=True)

    def list(self, prefix: str) -> list[str]:
        """List S3 objects under *prefix*.

        Requires:
            - *prefix* is a relative key segment or an ``s3://`` path.

        Returns:
            A sorted list of ``s3://``-prefixed object URIs.  Returns an
            empty list when the prefix does not exist.

        Raises:
            None (``FileNotFoundError`` is caught internally).
        """
        fs = self._get_fs()
        full_prefix = self._full_key(prefix)
        try:
            return [f"s3://{p}" for p in fs.ls(full_prefix, detail=False)]
        except FileNotFoundError:
            return []

    def get_uri(self, *path_parts: str) -> str:
        """Build an ``s3://`` URI from *path_parts*.

        Returns:
            ``"s3://<bucket>/<prefix>/<path_parts[0]>/…/<path_parts[n]>"``.

        Raises:
            None.
        """
        parts = [self._bucket]
        if self._prefix:
            parts.append(self._prefix)
        parts.extend(path_parts)
        return "s3://" + "/".join(parts)
