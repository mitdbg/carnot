"""Build a ChromaDB collection from precomputed element embeddings — the qatfd CLI.

The generic resumable loader (npz layout, rank-sharding, resume manifest,
batched upserts) lives in the skunk library
(`skunk.search_agent.prep.create_vector_db`); this script owns the
benchmark-specific bit: one `ElementAdapter` per benchmark mapping that
benchmark's per-element metadata dict to `(doc_id, document_text,
extra_metadata)`. To add a benchmark, write an adapter and register it in
`BENCHMARK_ADAPTERS`.

Usage (was `python -m skunk.search_agent.prep.create_vector_db` before 2026-07-07):

    python3 engaging-scripts/create_vector_db.py \
        --embeddings-dir <dir> --collection-name <name> \
        --chroma-path <dir> --benchmark <officeqa|browsecomp_plus|...>
"""

import argparse

from skunk.search_agent.prep.create_vector_db import ElementAdapter, run_build


def _officeqa_adapter(elt_metadata: dict) -> tuple[str, str, dict]:
    """Adapter for `compute_officeqa_element_embeddings.py` outputs."""
    return (
        elt_metadata["page_key"],
        elt_metadata["cleaned"],
        {
            "file_id": elt_metadata["file_id"],
            "year": elt_metadata["year"],
            "month": elt_metadata["month"],
            "page_id": elt_metadata["page_id"],
            "element_id": elt_metadata["element_id"],
            "type": elt_metadata["type"],
        },
    )


def _browsecomp_plus_adapter(elt_metadata: dict) -> tuple[str, str, dict]:
    """Adapter for `compute_browsecomp_plus_element_embeddings.py` outputs."""
    return (
        elt_metadata["docid"],
        elt_metadata["cleaned"],
        {
            "url": elt_metadata["url"],
            "element_id": elt_metadata["element_id"],
        },
    )


def _biogen_adapter(elt_metadata: dict) -> tuple[str, str, dict]:
    """Adapter for `compute_biogen_embeddings.py` outputs (one element per PubMed abstract)."""
    return (
        elt_metadata["docid"],  # PMID
        elt_metadata["cleaned"],
        {"element_id": elt_metadata["element_id"]},
    )


def _financebench_adapter(elt_metadata: dict) -> tuple[str, str, dict]:
    """Adapter for `compute_financebench_element_embeddings.py` outputs (text/table/figure elements).

    The `page_key` ("{doc_name}::p{page_num}") is the doc_id, so retrieved doc_ids line up with the
    page-level gold the FinanceBench benchmark derives from the evidence page numbers; `type`
    (text/table/figure) is surfaced to the SearchAgent like OfficeQA's element type.
    """
    return (
        elt_metadata["page_key"],
        elt_metadata["cleaned"],
        {
            "doc_name": elt_metadata["doc_name"],
            "page_num": elt_metadata["page_num"],
            "element_id": elt_metadata["element_id"],
            "type": elt_metadata.get("type", "text"),
        },
    )


def _qampari_adapter(elt_metadata: dict) -> tuple[str, str, dict]:
    """Adapter for `compute_qampari_embeddings.py` outputs (one element per ~100-token Wikipedia chunk).

    The chunk_id ("{page_id}__{n}") is the doc_id, so the QAMPARI benchmark's chunk-text document_map
    (keyed by chunk_id) lines up with retrieved doc_ids; `title` is surfaced so retrieved chunks can
    be collapsed to their Wikipedia article (the unit of QAMPARI's gold) for doc-recall.
    """
    return (
        elt_metadata["chunk_id"],
        elt_metadata["cleaned"],
        {
            "title": elt_metadata.get("title", ""),
            "page_id": elt_metadata.get("page_id", ""),
            "url": elt_metadata.get("url", ""),
            "element_id": elt_metadata.get("element_id", 0),
        },
    )


def _freshstack_adapter(elt_metadata: dict) -> tuple[str, str, dict]:
    """Adapter for `compute_freshstack_embeddings.py` outputs (one element per corpus document).

    The corpus `_id` (e.g. "azure-openai/LICENSE.md_0_1140") is the doc_id, so retrieved doc_ids line
    up directly with the FreshStack benchmark's gold (a nugget's relevant_corpus_ids are corpus `_id`s);
    `file_id` is the source file the chunk belongs to (the `_id` minus its byte-range suffix), surfaced
    for file-level grouping/recall; `url` is the GitHub source carried through for reference.
    """
    return (
        elt_metadata["doc_id"],
        elt_metadata["cleaned"],
        {
            "file_id": elt_metadata.get("file_id", ""),
            "url": elt_metadata.get("url", ""),
            "element_id": elt_metadata.get("element_id", 0),
        },
    )


BENCHMARK_ADAPTERS: dict[str, ElementAdapter] = {
    "officeqa": _officeqa_adapter,
    "browsecomp_plus": _browsecomp_plus_adapter,
    "trec_biogen": _biogen_adapter,
    "finance_bench": _financebench_adapter,
    "qampari": _qampari_adapter,
    "freshstack": _freshstack_adapter,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a ChromaDB vector database from precomputed embeddings.")
    parser.add_argument("--embeddings-dir", type=str, required=True)
    parser.add_argument("--collection-name", type=str, required=True)
    parser.add_argument("--chroma-path", type=str, default=".chromadb",
                        help="Directory to store ChromaDB data (default: .chromadb).")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=sorted(BENCHMARK_ADAPTERS.keys()),
                        help="Which embedding-script output format to expect.")
    parser.add_argument("--only-rank", type=int, default=None,
                        help="If set, build ONLY this embedding rank's shard (embeddings_{rank}_*.npz + "
                             "metadata_rank{rank}.json) into --collection-name. Run once per rank with "
                             "distinct --collection-name (e.g. NAME_r0..NAME_r3) to get one Chroma "
                             "collection per rank, keeping each metadata segment small enough to compact.")
    args = parser.parse_args()

    run_build(
        embeddings_dir=args.embeddings_dir,
        collection_name=args.collection_name,
        chroma_path=args.chroma_path,
        adapter=BENCHMARK_ADAPTERS[args.benchmark],
        only_rank=args.only_rank,
    )
