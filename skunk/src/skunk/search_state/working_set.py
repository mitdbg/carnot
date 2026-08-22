from __future__ import annotations

from chromadb import Collection
from jinja2 import Environment, StrictUndefined
from threading import Lock
from typing import Sequence

from skunk.prompts import load_prompts

_ENV = Environment(
    autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined
)

_PROMPTS = load_prompts("working_set")


class WorkingSet:
    """A collection of documents and chunks which are cached in their own collection.
    Agents use a WorkingSet to manage query or topic-specific documents. WorkingSets
    are persisted and may be re-used across agent sessions in order to eliminate
    redundant searching and tool calling. For large document corpora, the use of a
    smaller WorkingSet can also speed up tool-calling, enabling faster end-to-end latency.
    Finally, WorkingSets enable data enrichment at an affordable scale. New metadata
    columns may be added to the WorkingSet (via SemanticMap) without needing to label
    an entire corpus.

    To avoid unnecessary data copying, WorkingSets only store the incremental set of documents
    and chunks that do not exist in their ancestors' collections. However, this tree structure
    is hidden from agents which use the WorkingSet. They are presented with a virtual collection
    containing all of the fetched chunk and doc ids for this WorkingSet and all of its ancestors.
    However, pruned, redacted, and read chunks / documents are unique to each WorkingSet. This
    is done so that an agent may reuse a WorkingSet without being prevented from fetching or
    reading documents which we're pruned, redacted, or read in a previous agent session.

    The class maintains the following members:
    - name: the name of the working set; must be globally unique across all WorkingSets
    - collection: a ChromaDB collection which stores all retrieved documents
    - fetched_(chunk|doc)_ids: the chunk / doc ids which have been inserted into the collection.
    - read_(chunk|doc)_ids: the chunk / doc ids which an agent has read into its context through this collection.
    - pruned_(chunk|doc)_ids: chunks / doc ids which an agent has deemed irrelevant and have been removed from the collection.
    - redacted_(chunk|doc)_ids: chunk / doc ids which have been removed from an agent's context window due to trimming (but are still in the collection).
    - summary: a string summarizing the contents of the working set (more descriptive than the name)
    - actions: a history of the tool calls taken on the working set; a tuple with the exact tool call and a boolean indicating whether it was a fetch
    - parent_working_set_ids: ids of the working sets which this one builds on

    The class exposes the following functions:
    - from_collection: creates a WorkingSet from a compatible collection (it must have the right metadata fields).
    - get_metadata: returns the (serialized) metadata for this WorkingSet
    - id: property which returns the WorkingSet's id (its collection name)
    - empty: returns boolean indicating whether the WorkingSet is empty or not
    - add_parents: updates the WorkingSet's parent_working_set_ids
    - to_message: serializes the WorkingSet to a summary that can be presented to a SearchAgent
    - insert: insert new data into the WorkingSet
    - query: queries this WorkingSet's collection (and its parents' collections) and returns the combined result(s)
    - get: executes .get() on this WorkingSet's collection (and its parents' collections) and returns the combined result(s)
    - enrich: apply a semantic map to enrich the WorkingSet by adding one-or-more metadata columns
    - persist: store the state of the WorkingSet on disk
    """

    # TODO: store actions as a JSON string
    METADATA_LIST_DELIMITER = "||"
    METADATA_TUPLE_DELIMITER = ";;"

    def __init__(
        self,
        collection: Collection,
        pruned_chunk_ids: set[str] | None = None,
        pruned_doc_ids: set[str] | None = None,
        read_chunk_ids: set[str] | None = None,
        read_doc_ids: set[str] | None = None,
        fetched_chunk_ids: set[str] | None = None,
        fetched_doc_ids: set[str] | None = None,
        redacted_chunk_ids: set[str] | None = None,
        redacted_doc_ids: set[str] | None = None,
        summary: str | None = None,
        actions: list[tuple[str, bool]] | None = None,
        parent_working_set_ids: set[str] | None = None,
    ):
        self.collection = collection
        self.pruned_chunk_ids: set[str] = pruned_chunk_ids or set()
        self.pruned_doc_ids: set[str] = pruned_doc_ids or set()
        self.read_chunk_ids: set[str] = read_chunk_ids or set()
        self.read_doc_ids: set[str] = read_doc_ids or set()
        self.fetched_chunk_ids: set[str] = fetched_chunk_ids or set()
        self.fetched_doc_ids: set[str] = fetched_doc_ids or set()
        self.redacted_chunk_ids: set[str] = redacted_chunk_ids or set()
        self.redacted_doc_ids: set[str] = redacted_doc_ids or set()
        self.summary = summary or "(Summary not yet computed.)"
        self.actions: list[tuple[str, bool]] = actions or []
        self.parent_working_set_ids: set[str] = parent_working_set_ids or set()
        self.lock = Lock()

    @staticmethod
    def from_collection(collection: Collection) -> WorkingSet:
        """Rehydrate a WorkingSet from a stored Collection."""
        metadata = collection.metadata
        assert metadata is not None, f"Empty metadata on working set: {collection.name}"

        # get summary directly
        summary = metadata["summary"]

        # NOTE: allegedly, chromadb accepts a non-empty string list for metadata,
        # but the current version is throwing an exception
        pruned_chunk_ids = set(filter(None, metadata["pruned_chunk_ids"].split(WorkingSet.METADATA_LIST_DELIMITER)))
        pruned_doc_ids = set(filter(None, metadata["pruned_doc_ids"].split(WorkingSet.METADATA_LIST_DELIMITER)))
        read_chunk_ids = set(filter(None, metadata["read_chunk_ids"].split(WorkingSet.METADATA_LIST_DELIMITER)))
        read_doc_ids = set(filter(None, metadata["read_doc_ids"].split(WorkingSet.METADATA_LIST_DELIMITER)))
        fetched_chunk_ids = set(filter(None, metadata["fetched_chunk_ids"].split(WorkingSet.METADATA_LIST_DELIMITER)))
        fetched_doc_ids = set(filter(None, metadata["fetched_doc_ids"].split(WorkingSet.METADATA_LIST_DELIMITER)))
        redacted_chunk_ids = set(filter(None, metadata["redacted_chunk_ids"].split(WorkingSet.METADATA_LIST_DELIMITER)))
        redacted_doc_ids = set(filter(None, metadata["redacted_doc_ids"].split(WorkingSet.METADATA_LIST_DELIMITER)))
        parent_working_set_ids = set(filter(None, metadata["parent_working_set_ids"].split(WorkingSet.METADATA_LIST_DELIMITER)))

        # parse list of tuples for actions
        actions = []
        if metadata["actions"]:
            action_tuples = metadata["actions"].split(WorkingSet.METADATA_LIST_DELIMITER)
            for tup in action_tuples:
                action, is_fetch = tup.split(WorkingSet.METADATA_TUPLE_DELIMITER)
                actions.append((action, is_fetch.lower() == "true"))

        return WorkingSet(
            collection,
            pruned_chunk_ids=pruned_chunk_ids,
            pruned_doc_ids=pruned_doc_ids,
            fetched_chunk_ids=fetched_chunk_ids,
            fetched_doc_ids=fetched_doc_ids,
            read_chunk_ids=read_chunk_ids,
            read_doc_ids=read_doc_ids,
            redacted_chunk_ids=redacted_chunk_ids,
            redacted_doc_ids=redacted_doc_ids,
            summary=summary,
            actions=actions,
            parent_working_set_ids=parent_working_set_ids,
        )

    @classmethod
    def default_metadata(cls) -> dict:
        """Serialized metadata of a brand-new (empty) WorkingSet. Used so `create_collection`
        can be given its metadata ATOMICALLY — a separate modify() after creation leaves a
        window in which a sibling agent's registry rehydration sees the collection with
        metadata=None and fails."""
        keys = (
            "pruned_chunk_ids", "pruned_doc_ids", "read_chunk_ids", "read_doc_ids",
            "fetched_chunk_ids", "fetched_doc_ids", "redacted_chunk_ids", "redacted_doc_ids",
            "actions", "parent_working_set_ids",
        )
        metadata = {key: "" for key in keys}
        metadata["summary"] = "(Summary not yet computed.)"
        return metadata

    def get_metadata(self) -> dict:
        """Return the metadata associated with the WorkingSet."""
        actions = [f"{action}{self.METADATA_TUPLE_DELIMITER}{is_fetch}" for action, is_fetch in self.actions]
        return {
            "pruned_chunk_ids": self.METADATA_LIST_DELIMITER.join(self.pruned_chunk_ids),
            "pruned_doc_ids": self.METADATA_LIST_DELIMITER.join(self.pruned_doc_ids),
            "read_chunk_ids": self.METADATA_LIST_DELIMITER.join(self.read_chunk_ids),
            "read_doc_ids": self.METADATA_LIST_DELIMITER.join(self.read_doc_ids),
            "fetched_chunk_ids": self.METADATA_LIST_DELIMITER.join(self.fetched_chunk_ids),
            "fetched_doc_ids": self.METADATA_LIST_DELIMITER.join(self.fetched_doc_ids),
            "redacted_chunk_ids": self.METADATA_LIST_DELIMITER.join(self.redacted_chunk_ids),
            "redacted_doc_ids": self.METADATA_LIST_DELIMITER.join(self.redacted_doc_ids),
            "summary": self.summary,
            "actions": self.METADATA_LIST_DELIMITER.join(actions),
            "parent_working_set_ids": self.METADATA_LIST_DELIMITER.join(self.parent_working_set_ids),
        }

    @property
    def id(self):
        """Returns a unique identifier for this WorkingSet; currently the name of its collection."""
        return self.collection.name

    def empty(self, use_id_state: bool = False):
        """Returns True if the WorkingSet is empty and False otherwise. Normally, this is determined
        by self.collection.count(). For ablation studies where the WorkingSet only performs id tracking
        we instead use the state of the fetched ids."""
        if use_id_state:
            return not (self.fetched_chunk_ids or self.fetched_doc_ids)

        return not self.collection.count()

    def add_parents(self, working_sets: Sequence[WorkingSet]) -> None:
        """Add the the working sets' ids to this working set's parents."""
        with self.lock:
            # update the set of ids
            self.parent_working_set_ids.update([ws.id for ws in working_sets])

            # persist the change by updating the collection's metadata 
            metadata = self.get_metadata()
            metadata["parent_working_set_ids"] = self.METADATA_LIST_DELIMITER.join(self.parent_working_set_ids)
            self.collection.modify(metadata=metadata)

    def to_message(self, ancestors: Sequence[WorkingSet]) -> str:
        """Returns a string summarizing this working set (and its ancestors') contents for an agent."""
        WORKING_SET_TEMPLATE = _PROMPTS["working_set_summary"]
        fetched_chunk_ids = set(self.fetched_chunk_ids)
        fetched_doc_ids = set(self.fetched_doc_ids)
        fetch_actions = [action for action, is_fetch in self.actions if is_fetch]
        for ancestor in ancestors:
            fetched_chunk_ids.update(ancestor.fetched_chunk_ids)
            fetched_doc_ids.update(ancestor.fetched_doc_ids)
            ancestor_fetch_actions = [action for action, is_fetch in ancestor.actions if is_fetch]
            fetch_actions = ancestor_fetch_actions + fetch_actions

        working_set_prompt = _ENV.from_string(WORKING_SET_TEMPLATE).render(
            id=self.id,
            summary=self.summary,
            fetch_actions=fetch_actions,
        )
        return working_set_prompt

    def compute_summary(self) -> None:
        """Compute a summary for the WorkingSet based on the parents and actions taken."""
        # TODO
        self.summary = "(Summary not yet computed.)"

    def add_action(self, tool: str, tool_kwargs: dict) -> None:
        """Adds an action to the working set's list of actions."""
        # reconstruct string for the tool call
        kwargs_str = ""
        for key, arg in tool_kwargs.items():
            kwargs_str += f"{key}='{arg}', " if isinstance(arg, str) else f"{key}={arg}, "
        kwargs_str = kwargs_str[:-2]
        tool_call = f"{tool}({kwargs_str})"

        # append tool call to actions
        is_fetch = tool_kwargs.get("fetch", False)
        self.actions.append((tool_call, is_fetch))

    def prune(self, chunk_ids: list[str] | None = None, doc_ids: list[str] | None = None) -> tuple[set[str], set[str]]:
        """Prunes the chunk_ids and doc_ids from this working set by storing the ids such
        that they can be used in future exclusion filters.
        
        Returns the set of newly pruned chunks and documents.
        """
        # compute the set of newly pruned chunks / docs (omit ones that have previously been pruned)
        new_pruned_chunks = set(chunk_ids or ()) - self.pruned_chunk_ids
        new_pruned_docs = set(doc_ids or ()) - self.pruned_doc_ids

        # add the newly pruned chunks / docs to their respective sets
        self.pruned_chunk_ids.update(new_pruned_chunks)
        self.pruned_doc_ids.update(new_pruned_docs)

        return new_pruned_chunks, new_pruned_docs

    def insert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """Insert the data items into the working set's collection."""
        with self.lock:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,  # type: ignore
                documents=documents,
                metadatas=metadatas,  # type: ignore
            )

    def query(self, query_kwargs: dict) -> dict:
        """Query this collection and its ancestors' collections and return the aggregated result. Top-k values are respected."""
        # TODO
        return {}

    def get(self, query_kwargs: dict) -> dict:
        """Call .get() on this collection and its ancestors' collections and return the aggregated result. Limit values are respected."""
        # TODO
        return {}

    # TODO
    def enrich(self):
        pass

    def persist(self):
        with self.lock:
            # persist the state by updating the collection's metadata 
            metadata = self.get_metadata()
            metadata["parent_working_set_ids"] = self.METADATA_LIST_DELIMITER.join(self.parent_working_set_ids)
            self.collection.modify(metadata=metadata)
