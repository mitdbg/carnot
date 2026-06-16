"""Build the DAIS slim ChromaDB collection from precomputed Qwen embeddings.

A dais-local copy of ``search_agent/prep/create_vector_db.py`` (kept separate so core
code is untouched) with a single generic adapter. Consumes the partitioned
``embeddings_{p}.npz`` + ``metadata.json`` produced by ``compute_qwen_embeddings.py`` and
writes a collection whose per-element metadata is
``{doc_id, chunk_id, file_id, page_id, element_id, type, source, year, chroma:document}`` —
the corpus is annual, so ``year`` (int) + ``source`` (era family) replace OfficeQA's
year/month; ``doc_id`` + ``chunk_id`` are what the SearchAgent prune/filter logic needs.

Build OFFLINE: this uses a ``PersistentClient`` and must not run while the ChromaDB server
(the HttpClient read path) is up over the same directory.

Usage:
    python3 -m skunk.dais.create_vector_db \\
        --embeddings-dir dais_embeddings --collection-name dais-slim \\
        --chroma-path .chromadb-dais-slim
"""

from __future__ import annotations

import argparse
import json
import os

import chromadb
import numpy as np

CHROMA_MAX_BATCH_SIZE = 5461


def _dais_adapter(m: dict) -> tuple[str, str, dict]:
    """Map a per-element metadata dict -> (doc_id, document_text, extra_metadata)."""
    extra = {
        "file_id": m["file_id"],
        "page_id": m["page_id"],
        "element_id": m["element_id"],
        "type": m["type"],
        "source": m["source"],
    }
    # Chroma rejects None metadata values; only include `year` when it parsed.
    if m.get("year") is not None:
        extra["year"] = m["year"]
    return (m["page_key"], m["cleaned"], extra)


def _load_metadata(embeddings_dir: str) -> dict[str, dict]:
    """Load and merge every ``metadata*.json`` file in ``embeddings_dir``."""
    metadata: dict[str, dict] = {}
    for file in os.listdir(embeddings_dir):
        if file.startswith("metadata") and file.endswith(".json"):
            with open(os.path.join(embeddings_dir, file)) as f:
                metadata.update(json.load(f))
    return metadata


def _add_partition(collection, npz_path: str, metadata: dict[str, dict]) -> int:
    data = np.load(npz_path)
    embeddings = data["embeddings"]
    chunk_ids = [str(cid) for cid in data["unique_element_ids"]]

    doc_ids, documents, extras = [], [], []
    for cid in chunk_ids:
        doc_id, text, extra = _dais_adapter(metadata[cid])
        doc_ids.append(doc_id)
        documents.append(text)
        extras.append(extra)

    for i in range(0, len(embeddings), CHROMA_MAX_BATCH_SIZE):
        end = i + CHROMA_MAX_BATCH_SIZE
        metas = [
            {"doc_id": doc_ids[j], "chunk_id": chunk_ids[j], **extras[j]}
            for j in range(i, min(end, len(chunk_ids)))
        ]
        collection.add(
            ids=chunk_ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            documents=documents[i:end],
            metadatas=metas,  # type: ignore[arg-type]
        )
    return len(chunk_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the DAIS slim ChromaDB collection.")
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--chroma-path", default=".chromadb-dais-slim",
                        help="ChromaDB PersistentClient directory (default: .chromadb-dais-slim).")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.chroma_path)
    collection = client.get_or_create_collection(name=args.collection_name)
    print(f"Writing to collection {args.collection_name!r} at {args.chroma_path}.")

    metadata = _load_metadata(args.embeddings_dir)
    total = 0
    for file in sorted(os.listdir(args.embeddings_dir)):
        if file.startswith("embeddings") and file.endswith(".npz"):
            print(f"Processing {file}...")
            total += _add_partition(collection, os.path.join(args.embeddings_dir, file), metadata)
    print(f"Done. Added {total} elements to {args.collection_name!r}.")


if __name__ == "__main__":
    main()
