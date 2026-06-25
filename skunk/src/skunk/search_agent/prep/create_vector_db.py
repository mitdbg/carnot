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
import re
from collections.abc import Callable

import chromadb
import numpy as np
from tqdm import tqdm

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


_BENCHMARK_ADAPTERS: dict[str, ElementAdapter] = {
    "officeqa": _officeqa_adapter,
    "browsecomp_plus": _browsecomp_plus_adapter,
    "trec_biogen": _biogen_adapter,
    "finance_bench": _financebench_adapter,
    "qampari": _qampari_adapter,
}


def _load_metadata(embeddings_dir: str) -> dict[str, dict]:
    """Load and merge every `metadata*.json` file in `embeddings_dir`."""
    metadata: dict[str, dict] = {}
    for file in os.listdir(embeddings_dir):
        if file.startswith("metadata") and file.endswith(".json"):
            with open(os.path.join(embeddings_dir, file)) as f:
                metadata.update(json.load(f))
    return metadata


_RANK_META_RE = re.compile(r"^metadata_rank(\d+)\.json$")
_RANK_NPZ_RE = re.compile(r"^embeddings_(\d+)_")


def _rank_metadata_shards(embeddings_dir: str) -> dict[int, str] | None:
    """If metadata is sharded per embedding-rank (`metadata_rank{r}.json`), return {rank: filename};
    else None (a single `metadata.json`).

    Rank-sharding lets us build one rank at a time — load only that rank's metadata, add its
    `embeddings_{r}_*.npz`, then free it — so peak RAM is a single shard rather than the whole
    corpus held at once."""
    shards: dict[int, str] = {}
    for file in os.listdir(embeddings_dir):
        m = _RANK_META_RE.match(file)
        if m:
            shards[int(m.group(1))] = file
    return shards or None


def _manifest_path(chroma_path: str, collection_name: str) -> str:
    """Sidecar listing the .npz partitions already added to this collection, so a killed build can
    resume and skip them. Lives under the chroma dir, so deleting the DB also resets the manifest."""
    return os.path.join(chroma_path, f".{collection_name}.built_npz.txt")


def _load_manifest(manifest_path: str) -> set[str]:
    if not os.path.exists(manifest_path):
        return set()
    with open(manifest_path) as f:
        return {line.strip() for line in f if line.strip()}


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

        # upsert (not add) so resuming a partition that was interrupted mid-way — some of its
        # sub-batches already in the collection — overwrites idempotently instead of raising on
        # duplicate ids. Fully-added partitions are skipped earlier via the build manifest.
        collection.upsert(
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

    npz_files = sorted(f for f in os.listdir(args.embeddings_dir) if f.startswith("embeddings") and f.endswith(".npz"))
    shards = _rank_metadata_shards(args.embeddings_dir)

    # Resume support: skip .npz partitions already recorded in the manifest, and append each one as
    # it finishes (flushed immediately) so a killed run picks up exactly where it left off.
    manifest_path = _manifest_path(args.chroma_path, args.collection_name)
    built = _load_manifest(manifest_path)
    todo = [f for f in npz_files if f not in built]
    if built:
        print(f"Resuming: {len(built)}/{len(npz_files)} partitions already added; {len(todo)} remaining.")

    manifest = open(manifest_path, "a")
    pbar = tqdm(total=len(todo), unit="part", desc="Adding partitions", smoothing=0.05)

    def _add_and_record(file: str, metadata: dict[str, dict]) -> None:
        _add_partition(collection, os.path.join(args.embeddings_dir, file), metadata, adapter)
        manifest.write(file + "\n")
        manifest.flush()
        pbar.update(1)

    if shards is not None:
        # Rank-sharded metadata (e.g. biogen): process one rank at a time so only a single shard's
        # metadata is resident. Each `embeddings_{r}_*.npz` references only rank r's ids.
        npz_by_rank: dict[int, list[str]] = {}
        for file in npz_files:
            m = _RANK_NPZ_RE.match(file)
            if m is None:
                raise ValueError(f"rank-sharded metadata present but {file!r} lacks an embeddings_{{rank}}_ prefix")
            npz_by_rank.setdefault(int(m.group(1)), []).append(file)
        for rank in sorted(shards):
            rank_todo = [f for f in sorted(npz_by_rank.get(rank, [])) if f not in built]
            if not rank_todo:
                continue  # whole rank already added; skip loading its (multi-GB) metadata shard
            pbar.set_description(f"rank {rank} (loading metadata)")
            with open(os.path.join(args.embeddings_dir, shards[rank])) as f:
                metadata = json.load(f)
            pbar.set_description(f"rank {rank}")
            for file in rank_todo:
                _add_and_record(file, metadata)
            del metadata
    else:
        # Single metadata.json (smaller corpora): load once, add every remaining partition.
        metadata = _load_metadata(args.embeddings_dir)
        for file in todo:
            _add_and_record(file, metadata)

    pbar.close()
    manifest.close()
