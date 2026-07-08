"""Generic, resumable loader: precomputed element embeddings → a ChromaDB collection.

The on-disk layout produced by every `compute_*_embeddings.py` script is the same:

  * one or more `embeddings_{...}.npz` files, each containing arrays
    `embeddings` (float32, [n, d]) and `unique_element_ids` (str, [n])
  * one or more `metadata{_rank{r}}.json` files containing a single dict
    mapping `unique_element_id -> per-element metadata dict`

What differs per corpus is the *shape* of those per-element metadata dicts and
how they map onto the columns stored in ChromaDB:

  * the row id (= the SearchAgent's `chunk_id`)
  * the `documents` column (= the chunk's text)
  * the `metadatas` column, which must always carry `doc_id` + `chunk_id`
    so the SearchAgent's prune / filter logic works, plus any
    corpus-specific filterable fields.

That corpus-specific bit is an `ElementAdapter` the CALLER supplies — the
benchmark adapters live with the benchmarks (qatfd's
`engaging-scripts/create_vector_db.py` registers one per benchmark and wraps
`run_build` in a CLI). This module owns only the generic machinery: metadata
loading, rank-sharding, the resume manifest, and batched upserts.
"""

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
# where `extra_metadata` is a dict of corpus-specific filterable fields
# that will be stored alongside the common `doc_id` / `chunk_id` keys.
ElementAdapter = Callable[[dict], tuple[str, str, dict]]


def load_metadata(embeddings_dir: str) -> dict[str, dict]:
    """Load and merge every `metadata*.json` file in `embeddings_dir`."""
    metadata: dict[str, dict] = {}
    for file in os.listdir(embeddings_dir):
        if file.startswith("metadata") and file.endswith(".json"):
            with open(os.path.join(embeddings_dir, file)) as f:
                metadata.update(json.load(f))
    return metadata


_RANK_META_RE = re.compile(r"^metadata_rank(\d+)\.json$")
_RANK_NPZ_RE = re.compile(r"^embeddings_(\d+)_")


def rank_metadata_shards(embeddings_dir: str) -> dict[int, str] | None:
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


def add_partition(
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
            embeddings=batch_embeddings,  # ndarray directly; avoids a 5461×1024 list conversion
            documents=batch_documents,
            metadatas=metadata_list,  # type: ignore
        )


def run_build(
    *,
    embeddings_dir: str,
    collection_name: str,
    chroma_path: str,
    adapter: ElementAdapter,
    only_rank: int | None = None,
) -> None:
    """Build (or resume building) `collection_name` at `chroma_path` from every
    `embeddings*.npz` in `embeddings_dir`, mapping per-element metadata through
    `adapter`. `only_rank` restricts the build to a single embedding rank's shard
    (`embeddings_{rank}_*.npz` + `metadata_rank{rank}.json`) — run once per rank
    with distinct collection names to keep each metadata segment small enough to
    compact."""
    client = chromadb.PersistentClient(path=chroma_path)
    # hnsw:num_threads parallelizes index insertion (the dominant build cost) across all cores.
    # NOTE: HNSW params are baked at collection-creation, so this only takes effect on a *fresh*
    # collection — delete the chroma dir to re-create with it. (If your chromadb version rejects
    # the `hnsw:` metadata key, the configuration= form is the alternative.)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:num_threads": os.cpu_count() or 8},
    )
    print(f"Writing to collection {collection_name!r} at {chroma_path}.")

    npz_files = sorted(f for f in os.listdir(embeddings_dir) if f.startswith("embeddings") and f.endswith(".npz"))
    shards = rank_metadata_shards(embeddings_dir)

    if only_rank is not None:
        # Restrict the build to a single embedding rank -> one collection per rank.
        if not shards or only_rank not in shards:
            raise ValueError(f"--only-rank {only_rank}: no metadata_rank{only_rank}.json in {embeddings_dir}")
        shards = {only_rank: shards[only_rank]}
        npz_files = [f for f in npz_files
                     if (_RANK_NPZ_RE.match(f) and int(_RANK_NPZ_RE.match(f).group(1)) == only_rank)]
        if not npz_files:
            raise ValueError(f"--only-rank {only_rank}: no embeddings_{only_rank}_*.npz in {embeddings_dir}")

    # Resume support: skip .npz partitions already recorded in the manifest, and append each one as
    # it finishes (flushed immediately) so a killed run picks up exactly where it left off.
    manifest_path = _manifest_path(chroma_path, collection_name)
    built = _load_manifest(manifest_path)
    todo = [f for f in npz_files if f not in built]
    if built:
        print(f"Resuming: {len(built)}/{len(npz_files)} partitions already added; {len(todo)} remaining.")

    manifest = open(manifest_path, "a")
    pbar = tqdm(total=len(todo), unit="part", desc="Adding partitions", smoothing=0.05)

    def _add_and_record(file: str, metadata: dict[str, dict]) -> None:
        add_partition(collection, os.path.join(embeddings_dir, file), metadata, adapter)
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
            with open(os.path.join(embeddings_dir, shards[rank])) as f:
                metadata = json.load(f)
            pbar.set_description(f"rank {rank}")
            for file in rank_todo:
                _add_and_record(file, metadata)
            del metadata
    else:
        # Single metadata.json (smaller corpora): load once, add every remaining partition.
        metadata = load_metadata(embeddings_dir)
        for file in todo:
            _add_and_record(file, metadata)

    pbar.close()
    manifest.close()
