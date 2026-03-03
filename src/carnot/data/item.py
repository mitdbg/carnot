from __future__ import annotations

import warnings
from typing import Any

from carnot.storage.parsers import parse_file_contents
from carnot.storage.tiered import TieredStorageManager


class DataItem:
    """Lightweight reference to a data item in Carnot.

    The canonical identifier is :pyattr:`uri`.  The ``path`` constructor
    parameter is accepted for backward compatibility (backend routes,
    tests) but all new code should use ``uri``.

    When both ``path`` and ``uri`` are supplied to the constructor,
    ``uri`` takes precedence.  When neither is supplied, ``uri``
    defaults to the empty string ``""``.

    Materialization (reading + parsing file content) happens through
    the storage layer via :meth:`materialize`.  Once materialized the
    result is cached in an internal dict accessible via :meth:`to_dict`.

    A DataItem created via :meth:`from_dict` is considered already
    materialized — :meth:`to_dict` returns the original dict without
    requiring a storage manager.

    Representation invariant:
        - ``uri`` is always a ``str`` (never ``None``).
        - After a successful :meth:`materialize` call, :meth:`to_dict`
          returns the same ``dict`` object without re-materializing.

    Abstraction function:
        Represents a single data record identified by ``uri``.  Before
        materialization it is an opaque reference; after materialization
        it wraps a ``dict`` with at least ``"uri"`` and ``"contents"`` keys.
    """

    def __init__(self, path: str | None = None, *, uri: str | None = None):
        # Accept both `path` (legacy) and `uri` — they are aliases.
        self.uri: str = uri or path or ""
        self._dict: dict | None = None

    # ── Backward-compat alias ───────────────────────────────────────────

    @property
    def path(self) -> str:
        """Deprecated alias for :pyattr:`uri`. Prefer ``item.uri``."""
        warnings.warn(
            "DataItem.path is deprecated; use DataItem.uri instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.uri

    @path.setter
    def path(self, value: str) -> None:
        warnings.warn(
            "DataItem.path is deprecated; use DataItem.uri instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.uri = value

    # ── Materialization ─────────────────────────────────────────────────

    def materialize(self, storage: TieredStorageManager | None = None) -> dict:
        """Read and parse the item's content, returning a dict.

        Uses the storage layer for transparent caching and
        backend-agnostic reads.  The result is cached internally so
        subsequent calls return the same ``dict`` object.

        Requires:
            - Either this item was created via :meth:`from_dict` (already
              cached) **or** *storage* is a non-``None``
              :class:`TieredStorageManager` and ``self.uri`` is non-empty.

        Returns:
            A ``dict`` with at least ``"uri"`` and ``"contents"`` keys.
            If the item was created via :meth:`from_dict`, returns the
            original dict unchanged.

        Raises:
            ValueError: if no cached dict exists and *storage* is
            ``None`` or ``self.uri`` is empty.
        """
        if self._dict is not None:
            return self._dict

        if storage and self.uri:
            raw = storage.read(self.uri)
            contents = parse_file_contents(self.uri, raw)
        else:
            raise ValueError("No storage provided for materialization and direct file reading is not implemented.")

        self._dict = {
            "contents": contents,
            "uri": self.uri,
        }
        return self._dict

    def update_dict(self, data: dict[str, Any]) -> None:
        """Merge *data* into the internal cache dictionary in place.

        If the internal cache has not been populated yet (no prior
        :meth:`materialize` or :meth:`from_dict`), triggers
        :meth:`to_dict` first to populate it.

        Does **not** sync back to instance attributes (e.g. ``uri``).

        Requires:
            - *data* is a ``dict``.

        Returns:
            None.  The internal cache dict is mutated in place.
        """
        if self._dict is None:
            self.to_dict()
        self._dict.update(data)

    def to_dict(self) -> dict:
        """Return the internal cache dictionary.

        Uses cached result from ``materialize()`` if available, otherwise
        triggers materialization (which may raise if no storage is provided).

        Returns:
            The same ``dict`` object returned by the most recent
            :meth:`materialize` or :meth:`from_dict` call.

        Raises:
            ValueError: (via :meth:`materialize`) if no cached dict
            exists and no storage is available.
        """
        if self._dict is not None:
            return self._dict
        return self.materialize()

    @staticmethod
    def from_dict(item_dict: dict) -> DataItem:
        """Create a DataItem whose state is held entirely in *item_dict*.

        The ``uri`` is extracted from the dict: first ``"uri"`` key, then
        ``"path"`` key, defaulting to ``""`` if neither is present.

        The returned item is considered already materialized —
        :meth:`to_dict` returns *item_dict* directly (same object).

        Requires:
            - *item_dict* is a ``dict``.

        Returns:
            A new :class:`DataItem` with ``to_dict() is item_dict``.
        """
        instance = DataItem(uri=item_dict.get("uri") or item_dict.get("path"))
        instance._dict = item_dict
        return instance
