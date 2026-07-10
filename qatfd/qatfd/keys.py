"""Shared corpus-key helpers — the single source of truth for the ID formats that tie the
offline artifacts (embedding metadata + Chroma collections, built by engaging-scripts/) to the
benchmark loaders. See CORPUS_MODEL.md for the taxonomy these keys implement
(chunk -> retrieval unit / `doc_id` -> source group).

The GPU embedding scripts under engaging-scripts/ run standalone on the cluster (no qatfd
install), so they carry their own copies of these helpers with a MUST-match comment pointing
here; everything importable (the loaders, eval scripts, migration scripts) uses this module.
"""

from __future__ import annotations

# --- FinanceBench: page keys "{doc_name}::p{page_num}" ------------------------------------------

# Separator joining a document name and its (zero-indexed) page number into a page key. Chosen so
# it cannot collide with FinanceBench doc_names (which use only [A-Za-z0-9_-]); the embedding job
# (compute_financebench_element_embeddings.py) and the chroma adapter (create_vector_db.py) MUST
# use the identical key, or retrieved doc_ids won't line up with gold for recall.
FINANCEBENCH_PAGE_SEP = "::p"


def financebench_page_key(doc_name: str, page_num: int) -> str:
    """Page-level retrieval-unit key "{doc_name}::p{page_num}" (page_num zero-indexed, as in
    FinanceBench's evidence annotations)."""
    return f"{doc_name}{FINANCEBENCH_PAGE_SEP}{int(page_num)}"


def financebench_doc(page_key: str) -> str:
    """Collapse a page key to its source-group key — the filing's doc_name (drop the trailing ::pN)."""
    return page_key.split(FINANCEBENCH_PAGE_SEP)[0]


# --- OfficeQA: page keys "YYYY_MM_page" ----------------------------------------------------------


def officeqa_doc(page_key: str) -> str:
    """Collapse an OfficeQA page key "YYYY_MM_page" to its source-group key — the monthly
    Treasury Bulletin "YYYY_MM" (drop the trailing page component)."""
    return "_".join(page_key.split("_")[:2])


# --- FreshStack: corpus `_id`s "{repo}/{path}_{start_byte}_{end_byte}" ---------------------------


def freshstack_file_id(chunk_id: str) -> str:
    """The source FILE a chunk belongs to: the corpus `_id` minus its trailing "_{start}_{end}"
    byte range, e.g. "azure-openai/LICENSE.md_0_1140" -> "azure-openai/LICENSE.md". (File paths
    can contain underscores, so we only strip when the last two underscore-separated fields are
    both integers.) MUST match `_file_id` in compute_freshstack_embeddings.py."""
    parts = chunk_id.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0]
    return chunk_id


def freshstack_byte_range(chunk_id: str) -> tuple[int, int] | None:
    """The (start_byte, end_byte) slice a corpus `_id` covers within its source file, or None when
    the `_id` carries no byte-range suffix. `start_byte` is monotonic within a file, so it doubles
    as the chunk's `element_id` (its 0-based-order stand-in) in the Chroma metadata."""
    parts = chunk_id.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return int(parts[1]), int(parts[2])
    return None
