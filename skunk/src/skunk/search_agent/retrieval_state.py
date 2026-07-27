from dataclasses import dataclass, field

@dataclass
class RetrievalState:
    """Per-question retrieval state, shared by reference between a `SearchAgent`
    and its tools (the one mutable contract in the retrieval loop):

    - `pruned_*`: material the agent ruled out. `PruneTool` is the SINGLE writer;
      the search/grep tools read them on every call (server-side `$nin`), so a
      prune takes effect immediately, and `SearchAgent._block_is_visible` reads
      them to redact already-emitted chunks at render time.
    - `seen_*`: material already fetched — chunks returned by search/grep, docs
      opened by `read_document` — auto-excluded from subsequent search/grep so
      each call surfaces NEW material. Unlike pruned chunks, fetched chunks stay
      VISIBLE (prune is reserved for ruling out irrelevant material). Search/grep
      both write (on return) and read (unioned with the pruned sets);
      `read_document` writes `seen_doc_ids`.

    Extra tools (constructed by the caller before the agent's state exists) opt in
    via a duck-typed `bind_retrieval_state(state)` that `SearchAgent.__init__` calls — e.g.
    `SemanticFilterTool`, which always reads the pruned sets and, unless called with
    `exclude=False`, the seen sets too (staying comprehensive over seen-but-not-pruned
    material only when asked), and writes `seen_chunk_ids`.
    """

    pruned_chunk_ids: set[str] = field(default_factory=set)
    pruned_doc_ids: set[str] = field(default_factory=set)
    seen_chunk_ids: set[str] = field(default_factory=set)
    seen_doc_ids: set[str] = field(default_factory=set)
