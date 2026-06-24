"""Build a ChromaDB collection from precomputed element embeddings.

The on-disk layout produced by every `compute_*_element_embeddings.py`
script is the same:

  * one or more `embeddings_{...}.npz` files, each containing arrays
    `embeddings` (float32, [n, d]) and `unique_element_ids` (str, [n])
  * one or more `metadata{_rank{r}}.json` files containing a single dict
    mapping `unique_element_id -> per-element metadata dict`

What differs per benchmark is the *shape* of those per-element metadata
dicts and how they map onto the columns we store in ChromaDB:

  * the row id (= the SearchAgent's `chunk_id`)
  * the `documents` column (= the chunk's text)
  * the `metadatas` column, which must always carry `doc_id` + `chunk_id`
    so the SearchAgent's prune / filter logic works, plus any
    benchmark-specific filterable fields.

The benchmark-specific bit is isolated in `_BENCHMARK_ADAPTERS` below: to
add a new benchmark, write an adapter that returns
`(doc_id, document_text, extra_metadata)` for a single per-element
metadata dict and register it.
"""

import argparse
import json
import os
from collections.abc import Callable

import chromadb
import numpy as np

CHROMA_MAX_BATCH_SIZE = 5461


# An adapter maps a single per-element metadata dict (as produced by an embedding script) to:
#   (doc_id, document_text, extra_metadata)
# where `extra_metadata` is a dict of benchmark-specific filterable fields
# that will be stored alongside the common `doc_id` / `chunk_id` keys.
ElementAdapter = Callable[[dict], tuple[str, str, dict]]


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


_BENCHMARK_ADAPTERS: dict[str, ElementAdapter] = {
    "officeqa": _officeqa_adapter,
    "browsecomp_plus": _browsecomp_plus_adapter,
    "trec_biogen": _biogen_adapter,
    "finance_bench": _financebench_adapter,
}


def _load_metadata(embeddings_dir: str) -> dict[str, dict]:
    """Load and merge every `metadata*.json` file in `embeddings_dir`."""
    metadata: dict[str, dict] = {}
    for file in os.listdir(embeddings_dir):
        if file.startswith("metadata") and file.endswith(".json"):
            with open(os.path.join(embeddings_dir, file)) as f:
                metadata.update(json.load(f))
    return metadata


def _add_partition(
    collection: chromadb.Collection,
    npz_path: str,
    metadata: dict[str, dict],
    adapter: ElementAdapter,
) -> None:
    """Add the embeddings stored in one .npz file to the chroma collection."""
    data = np.load(npz_path)
    embeddings = data["embeddings"]
    # `unique_element_ids` are the per-element ids assigned at embedding
    # time. We use them directly as ChromaDB row ids (= chunk_ids exposed
    # to the SearchAgent).
    chunk_ids = [str(cid) for cid in data["unique_element_ids"]]

    doc_ids: list[str] = []
    documents: list[str] = []
    extra_metas: list[dict] = []
    for cid in chunk_ids:
        elt_metadata = metadata[cid]
        doc_id, document_text, extra = adapter(elt_metadata)
        doc_ids.append(doc_id)
        documents.append(document_text)
        extra_metas.append(extra)

    for i in range(0, len(embeddings), CHROMA_MAX_BATCH_SIZE):
        end = i + CHROMA_MAX_BATCH_SIZE
        batch_embeddings = embeddings[i:end]
        batch_chunk_ids = chunk_ids[i:end]
        batch_doc_ids = doc_ids[i:end]
        batch_documents = documents[i:end]
        batch_extras = extra_metas[i:end]

        metadata_list = [
            {"doc_id": batch_doc_ids[j], "chunk_id": batch_chunk_ids[j], **batch_extras[j]}
            for j in range(len(batch_embeddings))
        ]

        collection.add(
            ids=batch_chunk_ids,
            embeddings=batch_embeddings.tolist(),
            documents=batch_documents,
            metadatas=metadata_list,  # type: ignore
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a ChromaDB vector database from precomputed embeddings.")
    parser.add_argument("--embeddings-dir", type=str, required=True)
    parser.add_argument("--collection-name", type=str, required=True)
    parser.add_argument("--chroma-path", type=str, default=".chromadb",
                        help="Directory to store ChromaDB data (default: .chromadb).")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=sorted(_BENCHMARK_ADAPTERS.keys()),
                        help="Which embedding-script output format to expect.")
    args = parser.parse_args()

    # get the adapter for this benchmark
    adapter = _BENCHMARK_ADAPTERS[args.benchmark]

    # create the chroma client and collection
    client = chromadb.PersistentClient(path=args.chroma_path)
    collection = client.get_or_create_collection(name=args.collection_name)
    print(f"Writing to collection {args.collection_name!r} at {args.chroma_path} (benchmark={args.benchmark}).")

    # load the metadata files (mapping from unique_element_id to per-element metadata dict)
    metadata = _load_metadata(args.embeddings_dir)

    # for each embeddings_{...}.npz file, add its contents to the chroma collection
    for file in os.listdir(args.embeddings_dir):
        if not (file.startswith("embeddings") and file.endswith(".npz")):
            continue
        print(f"Processing file {file}...")
        _add_partition(
            collection,
            os.path.join(args.embeddings_dir, file),
            metadata,
            adapter,
        )
