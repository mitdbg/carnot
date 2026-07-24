from typing import Any, Protocol

class DocumentMap(Protocol):
    """The `doc_id -> full text` lookup the systems need (read_document / answer context). A real
    dict (OfficeQA / BrowseComp-Plus / FinanceBench) or a lazy chroma-backed mapping (TREC-BioGen,
    to avoid holding 26.8M abstracts in RAM) — the systems only do keyed lookups, never iterate it.

    Params are positional-only (`/`) so a plain `dict[str, str]` — whose `get`/`__getitem__` are
    positional-only in typeshed — structurally satisfies the protocol, same as the lazy mapping."""

    def get(self, doc_id: str, default: Any = None, /) -> Any: ...
    def __getitem__(self, doc_id: str, /) -> str: ...
    def __contains__(self, doc_id: object, /) -> bool: ...
