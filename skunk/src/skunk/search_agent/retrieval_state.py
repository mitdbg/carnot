from dataclasses import dataclass, field

@dataclass
class RetrievalState:
    """Per-question retrieval state, shared by reference between a `SearchAgent`
    and its tools:

    - `pruned_*`: material the agent deems irrelevant to its search. Once a chunk
      or doc is pruned, it is removed from the agent's context window and from the
      RetrievalState. `PruneTool` is the only tool which writes to these sets; the
      search/grep tools use them with an exclusion filter to avoid returning
      already-pruned chunks. `SearchAgent._block_is_visible` reads them to redact
      already-read chunks at render time.
    - `read_*`: material already rendered into the agent's context window. This
      includes chunks returned by search/grep and docs opened by `read_document`.
      These chunks and docs are automatically excluded from subsequent search/grep
      tool calls so that each call surfaces new material. Search/grep write to
      these sets, and `read_document` writes to `read_doc_ids`.
    - `fetched_*`: material that has been fetched from the vector store or document
      store into the RetrievalState, but not yet pruned or read by the agent. This
      is used to create a "working set" of material for the agent to consider, and
      is written to by the search/grep tools.

    Every tool takes the state at construction (`SearchAgent.__init__` builds its tools
    around the shared instance) — e.g. `SemanticFilterTool` always reads the pruned sets
    and the read sets too (staying comprehensive over read-but-not-pruned material only
    when asked), and writes `read_chunk_ids`.
    """

    pruned_chunk_ids: set[str] = field(default_factory=set)
    pruned_doc_ids: set[str] = field(default_factory=set)
    read_chunk_ids: set[str] = field(default_factory=set)
    read_doc_ids: set[str] = field(default_factory=set)
    fetched_chunk_ids: set[str] = field(default_factory=set)
    fetched_doc_ids: set[str] = field(default_factory=set)
    redacted_chunk_ids: set[str] = field(default_factory=set)
    redacted_doc_ids: set[str] = field(default_factory=set)

    def empty(self):
        return not (self.fetched_chunk_ids or self.fetched_doc_ids)
