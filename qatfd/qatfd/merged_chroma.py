"""Sharded ChromaDB behind one client-shaped facade.

BioGen's 26.8M abstracts are split into one collection per embedding rank (``f"{base}_r{i}"``)
because chromadb 1.5.x's metadata-segment compaction fails on a single collection that large;
each ~6.7M-row shard compacts fine, and in production each shard is served by its own warm
``chroma run`` server on its own port. Two duck-typed stand-ins hide that from the rest of qatfd:

* ``MergedCollection`` — looks like a chromadb ``Collection``. Fans ``query``/``get``/``count``/
  ``update``/``delete``/``modify`` out across the N shard collections (concurrently) and merges the
  results, so the base corpus reads as one collection.
* ``MergedClient`` — looks like a chromadb ``ClientAPI``. Presents the N shard servers (exactly one
  server per shard) as one database: the base corpus is exposed under its logical name as a
  ``MergedCollection``, while every other collection (working sets, Bootstrap/Enrich-managed
  collections, ...) lives on exactly one shard and is routed there transparently. New collections
  are placed on the least-loaded shard.

Routing is *discovered*, not replicated: a collection lives wherever ``get_collection`` finds it, so
the name -> shard map can never drift from what is actually on the servers (a replicated map could,
if a create succeeded on one shard but the map write failed on another). A per-process cache avoids
re-probing on the hot path, and every collection the client creates is additionally stamped with
``SHARD_KEY`` (its shard index) in collection metadata for provenance/debugging.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from chromadb import Collection
from chromadb.api import ClientAPI
from chromadb.errors import NotFoundError, UniqueConstraintError

# collection-metadata key stamped on every collection MergedClient creates: the index (into
# MergedClient.clients) of the shard it was placed on. Routing does not depend on it.
SHARD_KEY = "qatfd_merged_shard"

# chromadb caps list_collections() at 100 rows per call; page each shard at that size.
_LIST_PAGE = 100

# per-row columns chroma may return (besides "ids"), for query (nested per query) and get (flat).
_QUERY_COLS = ("distances", "documents", "metadatas", "embeddings", "uris", "data")
_GET_COLS = ("documents", "metadatas", "embeddings", "uris", "data")


def _fanout(fn: Callable[[Any], Any], items: Sequence[Any]) -> list:
    """``[fn(x) for x in items]`` run concurrently (one thread per item), results in input order.
    Parallel fan-out is a latency win and is safe when the shards are separate chroma SERVERS
    (each owns its own concurrency); the embedded PersistentClient serializes/deadlocks under
    in-process concurrency, which is why sharded corpora must be served over HTTP."""
    if len(items) == 1:
        return [fn(items[0])]
    with ThreadPoolExecutor(max_workers=len(items)) as ex:
        return list(ex.map(fn, items))


def _sub(seq: Sequence | None, positions: list[int]) -> list | None:
    return None if seq is None else [seq[i] for i in positions]


class MergedCollection:
    """Duck-typed stand-in for a chromadb ``Collection`` over N shard collections.

    Reads (``query``/``get``/``count``/``peek``) fan out and merge. Row-level writes that name ids
    (``update``/``delete``) are routed to the shard that owns each id; a row lives in exactly one
    shard (a doc's chunks all come from one source file -> one rank). ``modify(metadata=...)`` is
    replicated to every shard so the logical collection's metadata stays uniform. ``add``/``upsert``
    are not supported: the merged base corpus is append-only through the build pipeline.
    """

    def __init__(self, collections: Sequence[Collection], name: str | None = None) -> None:
        assert collections, "MergedCollection needs at least one shard collection"
        self._collections = list(collections)
        self._name = name if name is not None else self._collections[0].name

    def __repr__(self) -> str:
        return f"MergedCollection(name={self._name}, shards={[c.name for c in self._collections]})"

    # ---- Collection-shaped attributes ------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self):
        return self._collections[0].id

    @property
    def metadata(self) -> dict | None:
        """The first shard's collection metadata (``modify`` keeps all shards identical)."""
        return self._collections[0].metadata

    @property
    def shards(self) -> list[Collection]:
        return list(self._collections)

    def count(self) -> int:
        return sum(self._fanout(lambda c: c.count()))

    def _fanout(self, fn: Callable[[Collection], Any]) -> list:
        return _fanout(fn, self._collections)

    # ---- reads -----------------------------------------------------------------------------

    def query(self, **kwargs) -> dict:
        """Run the same query on every shard (concurrently), then keep the globally closest
        ``n_results`` per query by distance (chroma's default L2 space: smaller = closer). Every
        included column (documents/metadatas/embeddings/...) is carried through; distances are
        always fetched from the shards for ranking and dropped again if the caller did not ask."""
        n_results = kwargs.get("n_results", 10)
        include = list(kwargs.get("include") or ["metadatas", "documents", "distances"])
        shard_kwargs = dict(kwargs)
        shard_kwargs["include"] = include if "distances" in include else [*include, "distances"]
        results = self._fanout(lambda c: c.query(**shard_kwargs))

        cols = [k for k in _QUERY_COLS if any(r.get(k) is not None for r in results)]
        n_queries = max(len(r["ids"]) for r in results)
        out: dict = {"ids": [], **{k: [] for k in cols}}
        for q in range(n_queries):
            rows: list[dict] = []
            for r in results:
                for i, _id in enumerate(r["ids"][q]):
                    row = {"ids": _id}
                    for k in cols:
                        col = r.get(k)
                        row[k] = col[q][i] if col is not None else None
                    rows.append(row)
            # None distances (only possible for malformed shard replies) sort last, deterministically.
            rows.sort(key=lambda t: (t["distances"] is None, t["distances"] if t["distances"] is not None else 0.0))
            rows = rows[:n_results]
            out["ids"].append([t["ids"] for t in rows])
            for k in cols:
                out[k].append([t[k] for t in rows])
        if "distances" not in include:
            out.pop("distances", None)
        out["included"] = include
        return out

    def _merge_get(self, results: Iterable[dict], limit: int | None) -> dict:
        results = list(results)
        cols = [k for k in _GET_COLS if any(r.get(k) is not None for r in results)]
        out: dict = {"ids": [], **{k: [] for k in cols}}
        for r in results:
            out["ids"].extend(r.get("ids") or [])
            for k in cols:
                col = r.get(k)
                n = len(r.get("ids") or [])
                out[k].extend(list(col) if col is not None else [None] * n)
        if limit is not None:
            for k in out:
                out[k] = out[k][:limit]
        return out

    def get(self, **kwargs) -> dict:
        """Concatenate ``get`` across shards (each id is in one shard). ``limit`` is a global cap.
        A non-zero ``offset`` is not supported (it has no well-defined global meaning across shards)."""
        if kwargs.get("offset"):
            raise NotImplementedError("MergedCollection.get does not support a non-zero offset")
        limit = kwargs.get("limit")
        out = self._merge_get(self._fanout(lambda c: c.get(**kwargs)), limit)
        out["included"] = list(kwargs.get("include") or ["metadatas", "documents"])
        return out

    def peek(self, limit: int = 10) -> dict:
        return self._merge_get(self._fanout(lambda c: c.peek(limit=limit)), limit)

    # ---- writes ----------------------------------------------------------------------------

    def _owners(self, ids: Sequence[str]) -> list[list[int]]:
        """For each shard, the positions (into ``ids``) of the ids it owns. Ids present in no shard
        are dropped (matching chroma, which ignores update/delete of unknown ids)."""
        owned = self._fanout(lambda c: set(c.get(ids=list(ids), include=[])["ids"]))
        seen: set[str] = set()
        parts: list[list[int]] = []
        for shard_ids in owned:
            positions = [i for i, _id in enumerate(ids) if _id in shard_ids and _id not in seen]
            seen.update(ids[i] for i in positions)
            parts.append(positions)
        return parts

    def update(self, ids: Sequence[str], embeddings=None, metadatas=None, documents=None, **kwargs) -> None:
        """Route each id's update to the shard that owns it."""
        ids = list(ids)
        parts = self._owners(ids)

        def run(i: int) -> None:
            pos = parts[i]
            if pos:
                self._collections[i].update(
                    ids=_sub(ids, pos),  # type: ignore[arg-type]
                    embeddings=_sub(embeddings, pos),
                    metadatas=_sub(metadatas, pos),
                    documents=_sub(documents, pos),
                    **{k: _sub(v, pos) for k, v in kwargs.items()},
                )

        _fanout(run, list(range(len(self._collections))))

    def delete(self, ids: Sequence[str] | None = None, where=None, where_document=None) -> None:
        """Delete by id (routed to the owning shard) and/or by filter (applied on every shard)."""
        if ids is not None:
            ids = list(ids)
            parts = self._owners(ids)
            _fanout(
                lambda i: self._collections[i].delete(ids=_sub(ids, parts[i]), where=where, where_document=where_document)
                if parts[i]
                else None,
                list(range(len(self._collections))),
            )
        elif where is not None or where_document is not None:
            self._fanout(lambda c: c.delete(where=where, where_document=where_document))
        else:
            raise ValueError("delete needs ids, where, or where_document")

    def modify(self, name: str | None = None, metadata: dict | None = None, **kwargs) -> None:
        """Replicate a metadata/configuration change to every shard. Renaming is not supported
        (the logical name is not a real chroma collection)."""
        if name is not None:
            raise NotImplementedError("MergedCollection cannot be renamed")
        self._fanout(lambda c: c.modify(metadata=metadata, **kwargs))

    def add(self, *_, **__) -> None:
        raise NotImplementedError("MergedCollection is read-only for new rows (add); write to a shard directly")

    def upsert(self, *_, **__) -> None:
        raise NotImplementedError("MergedCollection is read-only for new rows (upsert); write to a shard directly")


@dataclass(frozen=True)
class Shard:
    """One shard of the base corpus: the client (server) that serves it and the shard collection's
    name. ``label`` (e.g. ``"127.0.0.1:8001"``) is only used in error messages."""

    client: ClientAPI
    collection_name: str
    label: str = ""


class MergedClient:
    """Duck-typed stand-in for a chromadb ``ClientAPI`` over the shard servers of a sharded corpus.

    * ``get_collection(base_name)`` -> a ``MergedCollection`` over the shard collections. The raw
      shard collections are hidden from ``list_collections`` (collapsed into that one entry) but can
      still be fetched by their own names.
    * Any other name is looked up on every shard (concurrently); the first shard that has it wins.
      Hits are cached per process so the hot path (tools calling ``get_collection`` per call) costs
      one round-trip, exactly like a plain client.
    * ``create_collection`` places the new collection on the shard with the fewest non-base
      collections (ties -> lowest index) and stamps ``SHARD_KEY`` into its metadata.
    * ``list_collections(limit, offset)`` is a stable, name-sorted, globally paginated view across
      shards, so callers can page with ``limit``/``offset`` and stop on a short page.

    Only the ``ClientAPI`` surface qatfd uses is implemented; anything else raises AttributeError.
    Creation is serialized within the process; two *processes* racing to create the same new name
    could still place it on two shards (first shard then wins on lookup), which is acceptable for
    agent-owned working sets whose names embed the agent id.
    """

    def __init__(self, base_name: str, shards: Sequence[Shard]) -> None:
        """``shards[i]`` is served by its own server: shard index i == client index i. Precondition:
        every shard has a distinct client and a distinct collection name."""
        assert shards, "MergedClient needs at least one shard"
        self._base_name = base_name
        self._shards = list(shards)
        self._clients: list[ClientAPI] = [s.client for s in self._shards]
        self._labels: list[str] = [s.label or f"shard{i}" for i, s in enumerate(self._shards)]
        if len({id(c) for c in self._clients}) != len(self._clients):
            raise ValueError("each shard must be served by its own client (one server per shard)")
        self._shard_names = {s.collection_name for s in self._shards}
        if len(self._shard_names) != len(self._shards):
            raise ValueError("shard collection names must be distinct")
        if base_name in self._shard_names:
            raise ValueError("base_name must differ from the shard collection names")
        self._lock = threading.RLock()
        self._route: dict[str, int] = {}  # collection name -> index into self._clients
        # fail fast if a shard collection is missing (mirrors a plain client's get_collection).
        self._open_base()

    def __repr__(self) -> str:
        return f"MergedClient(base={self._base_name!r}, shards={[s.collection_name for s in self._shards]})"

    # ---- introspection ---------------------------------------------------------------------

    @property
    def base_name(self) -> str:
        return self._base_name

    @property
    def shards(self) -> list[Shard]:
        return list(self._shards)

    @property
    def clients(self) -> list[ClientAPI]:
        """The per-shard clients; shard indices reported by ``shard_of`` index this list."""
        return list(self._clients)

    def shard_of(self, name: str) -> int:
        """Index (into ``clients``) of the shard that holds ``name``. Raises NotFoundError."""
        if name == self._base_name:
            raise ValueError(f"{name!r} is the merged base collection; it spans every shard")
        self.get_collection(name)
        return self._route[name]

    # ---- helpers ---------------------------------------------------------------------------

    def _client_fanout(self, fn: Callable[[int], Any]) -> list:
        return _fanout(fn, list(range(len(self._clients))))

    def _open_base(self, found: dict[str, Collection] | None = None) -> MergedCollection:
        """The base corpus as a MergedCollection, fetched fresh per call (like a plain client's
        ``get_collection``, so ``.metadata`` is current). ``found`` lets ``list_collections`` reuse
        the Collection objects it already listed instead of re-fetching."""

        def open_shard(i: int) -> Collection:
            s = self._shards[i]
            if found is not None and s.collection_name in found:
                return found[s.collection_name]
            try:
                return s.client.get_collection(name=s.collection_name)
            except NotFoundError as e:
                raise NotFoundError(
                    f"shard collection {s.collection_name!r} not found ({self._labels[i]})"
                ) from e

        return MergedCollection(_fanout(open_shard, list(range(len(self._shards)))), name=self._base_name)

    def _list_all(self) -> list[tuple[int, Collection]]:
        """Every collection on every distinct client as (client_idx, Collection)."""

        def list_one(ci: int) -> list[tuple[int, Collection]]:
            out: list[tuple[int, Collection]] = []
            offset = 0
            while True:
                page = list(self._clients[ci].list_collections(limit=_LIST_PAGE, offset=offset))
                out.extend((ci, c) for c in page)
                if len(page) < _LIST_PAGE:
                    return out
                offset += _LIST_PAGE

        return [x for xs in self._client_fanout(list_one) for x in xs]

    def _try_get(self, ci: int, name: str, **kwargs) -> Collection | None:
        try:
            return self._clients[ci].get_collection(name=name, **kwargs)
        except NotFoundError:
            return None

    def _choose_shard(self, counts: list[int]) -> int:
        """Placement policy for a new collection: fewest non-base collections, ties -> lowest index."""
        return min(range(len(counts)), key=lambda i: (counts[i], i))

    # ---- ClientAPI surface -----------------------------------------------------------------

    def heartbeat(self) -> int:
        """Round-trips every shard (fails if any is down); returns the first shard's heartbeat."""
        return self._client_fanout(lambda ci: self._clients[ci].heartbeat())[0]

    def list_collections(self, limit: int | None = None, offset: int | None = None) -> list[Collection | MergedCollection]:
        """All collections across shards, name-sorted, with the shard collections collapsed into one
        ``MergedCollection`` entry under ``base_name``. Globally paginated by ``limit``/``offset``."""
        by_name: dict[str, tuple[int, Collection]] = {}
        shard_cols: dict[str, Collection] = {}
        for ci, c in self._list_all():
            if c.name in self._shard_names:
                shard_cols.setdefault(c.name, c)
            elif c.name not in by_name:
                by_name[c.name] = (ci, c)
        with self._lock:
            self._route.update({n: ci for n, (ci, _) in by_name.items()})
        entries: list[Collection | MergedCollection] = [c for _, c in by_name.values()]
        if shard_cols:
            entries.append(self._open_base(found=shard_cols))
        entries.sort(key=lambda c: c.name)
        start = offset or 0
        return entries[start:] if limit is None else entries[start : start + limit]

    def count_collections(self) -> int:
        return len(self.list_collections())

    def get_collection(self, name: str, **kwargs) -> Collection | MergedCollection:
        if name == self._base_name:
            return self._open_base()
        with self._lock:
            ci = self._route.get(name)
        if ci is not None:
            c = self._try_get(ci, name, **kwargs)
            if c is not None:
                return c
            with self._lock:  # deleted out from under us; fall through to a fresh probe
                self._route.pop(name, None)
        found = self._client_fanout(lambda ci: self._try_get(ci, name, **kwargs))
        for ci, c in enumerate(found):
            if c is not None:
                with self._lock:
                    self._route[name] = ci
                return c
        raise NotFoundError(f"Collection [{name}] does not exist on any shard ({', '.join(self._labels)})")

    def create_collection(
        self, name: str, metadata: dict | None = None, get_or_create: bool = False, **kwargs
    ) -> Collection | MergedCollection:
        """Create ``name`` on the least-loaded shard. With ``get_or_create`` an existing collection
        (on any shard) is returned instead of raising."""
        if name == self._base_name or name in self._shard_names:
            if get_or_create:
                return self.get_collection(name)
            raise UniqueConstraintError(f"Collection [{name}] already exists (it is part of the base corpus)")
        with self._lock:
            counts = [0] * len(self._clients)
            for ci, c in self._list_all():
                if c.name == name:
                    self._route[name] = ci
                    if get_or_create:
                        return c
                    raise UniqueConstraintError(f"Collection [{name}] already exists ({self._labels[ci]})")
                if c.name not in self._shard_names:
                    counts[ci] += 1
            ci = self._choose_shard(counts)
            c = self._clients[ci].create_collection(name=name, metadata={**(metadata or {}), SHARD_KEY: ci}, **kwargs)
            self._route[name] = ci
            return c

    def get_or_create_collection(self, name: str, metadata: dict | None = None, **kwargs) -> Collection | MergedCollection:
        return self.create_collection(name, metadata=metadata, get_or_create=True, **kwargs)

    def delete_collection(self, name: str) -> None:
        """Delete a routed collection. The base corpus and its shards are protected."""
        if name == self._base_name or name in self._shard_names:
            raise ValueError(f"refusing to delete {name!r}: it is part of the base corpus")
        self.get_collection(name)  # resolves + caches the owning shard, or raises NotFoundError
        with self._lock:
            ci = self._route.pop(name)
        self._clients[ci].delete_collection(name)
