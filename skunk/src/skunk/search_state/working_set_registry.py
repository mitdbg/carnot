import chromadb
from chromadb.api import ClientAPI
from pathlib import Path
from threading import Lock
from typing import Generator

from skunk.chroma_client import make_chroma_client
from skunk.search_state.working_set import WorkingSet

WS_PREFIX = "ws_"
DEFAULT_CHROMA_PATH = str(Path.home() / ".skunk" / "chromadb")

class WorkingSetRegistry:
    """
    A registry for WorkingSets created by agents and stored in the system.

    Every WorkingSet is a physical ChromaDB collection with metadata. All WorkingSets are
    identified by having a collection name with the prefix `ws_`. Each collection also mantains
    metadata including:
    - summary: a summary of the collection
    - actions: a ledger of the actions taken to modify the WorkingSet
    - id states: sets of fetched, pruned, read, and redacted document and chunk ids

    The registry exposes methods for registering collections as well as querying metadata to
    identify collections which are potentially relevant to a given search string.
    """

    def __init__(self, chroma_host: str | None = None, chroma_port: int | None = None, chroma_path: str | None = None):
        # user must provide host and port or a path to a chromadb database on disk
        self.client: ClientAPI
        if chroma_host is not None and chroma_port is not None:
            self.client = make_chroma_client(chroma_host, chroma_port)
        elif chroma_path is not None:
            Path(chroma_path).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(chroma_path)
        else:
            raise ValueError("Must provide host+port or path to chroma database")

        # lock for writing to the registry
        self._write_lock = Lock()

        # rehydrate WorkingSets
        self._registry: dict[str, WorkingSet] = self._rehydrate_working_sets()

    def __iter__(self) -> Generator[WorkingSet]:
        """Yield the every working set in the registry."""
        for ws in self._registry.values():
            yield ws

    def _rehydrate_working_sets(self) -> dict[str, WorkingSet]:
        """Rehydrates the registry by filtering for all collections with the `ws-` prefix.
        Returns a dictionary mapping from WorkingSet name (same as collection name) to itself.
        """
        registry: dict[str, WorkingSet] = {}
        for c in self.client.list_collections():
            if not c.name.startswith(WS_PREFIX):
                continue
            # a ws collection with metadata=None is mid-creation by another agent or an
            # orphan of a run that crashed between create and modify — unusable, skip it
            if c.metadata is None:
                continue
            registry[c.name] = WorkingSet.from_collection(c)
        return registry

    def ancestors_of(self, ws: WorkingSet) -> list[WorkingSet]:
        """Returns all of `ws`'s ancestors using a cycle-safe, breadth-first walk of its parent graph."""
        # NOTE: right now, all experiments run in a single process with async threads, thus, we don't
        # need to worry about the registry having stale parents due to changes from another process.
        # However, in the future we will want to support multiple agents in different processes performing
        # work simultaneously. At this point, we will need to have the parent working sets rehydrated based
        # on the current state of the collection on disk.
        seen, out, queue = {ws.id}, [], list(ws.parent_working_set_ids)
        while queue:
            # if we haven't seen this working set, fetch it from the registry
            wid = queue.pop(0)
            if wid in seen:
                continue
            seen.add(wid)
            parent = self._registry.get(wid)

            # guard against WorkingSet having been deleted
            if parent is None:
                continue

            # add working set to parent and add its parents' ids to queue
            out.append(parent)
            queue.extend(parent.parent_working_set_ids)

        return out

    def contains(self, name: str) -> bool:
        """Returns True if the name is in the registry and False otherwise."""
        return name in self._registry

    def get(self, name: str) -> WorkingSet:
        """Returns the WorkingSet with `name` if it exists and throws an Exception otherwise."""
        if name not in self._registry:
            raise Exception(f"No WorkingSet with name: {name}")

        return self._registry[name]

    def get_or_create(self, name: str) -> WorkingSet:
        """Returns the WorkingSet with `name` if it exists. Otherwise, a new WorkingSet
        is created, registered, and returned.
        """
        if name in self._registry:
            return self._registry[name]

        # create the collection with its metadata ATOMICALLY: a separate modify() after
        # creation leaves a window where a concurrent agent's rehydration sees the
        # collection with metadata=None (fatal under parallel workers)
        ws: WorkingSet
        with self._write_lock:
            collection = self.client.create_collection(name=name, metadata=WorkingSet.default_metadata())
            ws = WorkingSet(collection=collection)
            self._registry[name] = ws

        return ws

    def update(self, name: str, working_set: WorkingSet) -> None:
        """Update the WorkingSet's state in the registry."""
        with self._write_lock:
            self._registry[name] = working_set
