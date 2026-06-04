"""Build a ChromaDB collection from precomputed QA-pair embeddings.

Consumes the output of ``compute_qa_embeddings.py``:

  <embeddings_dir>/embeddings_{rank}_{partition}.npz
    embeddings:          float32  [n, d]
    unique_element_ids:  str      [n]   (e.g. "officeqa_0:q")

  <embeddings_dir>/metadata_rank{rank}.json
    { unique_element_id: { kind, qa_id, is_synthetic, benchmark, text } }

Each embedding row is upserted into the ChromaDB collection with:
  id        = unique_element_id  (e.g. "officeqa_0:q")
  document  = the question or answer text
  metadata  = { kind, qa_id, is_synthetic, benchmark }

The metadata schema matches the constants used by skunk/datagen/dedup.py so
the collection can be passed directly as the ``--qa-collection-name``
argument to harness.py.

Usage:
    python create_qa_vector_db.py \\
        --embeddings-dir ./officeqa-qa-embeddings \\
        --collection-name officeqa-qa-qwen3 \\
        --chroma-path .chromadb
"""

import argparse
import json
import os

import chromadb
import numpy as np

# ChromaDB's SQLite backend limits the batch size for upsert operations.
CHROMA_MAX_BATCH_SIZE = 5461


def _load_metadata(embeddings_dir: str) -> dict[str, dict]:
    """Load and merge every ``metadata_rank*.json`` file in ``embeddings_dir``."""
    metadata: dict[str, dict] = {}
    for fname in sorted(os.listdir(embeddings_dir)):
        if fname.startswith("metadata") and fname.endswith(".json"):
            with open(os.path.join(embeddings_dir, fname)) as f:
                metadata.update(json.load(f))
    if not metadata:
        raise FileNotFoundError(
            f"No metadata_rank*.json files found in {embeddings_dir!r}. "
            "Run compute_qa_embeddings.py first."
        )
    return metadata


def _add_partition(
    collection: chromadb.Collection,
    npz_path: str,
    metadata: dict[str, dict],
) -> None:
    """Upsert the embeddings stored in one .npz file into the collection."""
    data = np.load(npz_path)
    embeddings = data["embeddings"]
    element_ids = [str(eid) for eid in data["unique_element_ids"]]

    texts: list[str] = []
    metas: list[dict] = []
    for eid in element_ids:
        elt = metadata[eid]
        texts.append(elt["text"])
        metas.append({
            "kind": elt["kind"],
            "qa_id": elt["qa_id"],
            "is_synthetic": elt["is_synthetic"],
            "benchmark": elt["benchmark"],
        })

    for i in range(0, len(embeddings), CHROMA_MAX_BATCH_SIZE):
        end = i + CHROMA_MAX_BATCH_SIZE
        collection.upsert(
            ids=element_ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            documents=texts[i:end],
            metadatas=metas[i:end],  # type: ignore
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a ChromaDB collection from precomputed QA-pair embeddings."
    )
    parser.add_argument(
        "--embeddings-dir", type=str, required=True,
        help="Directory produced by compute_qa_embeddings.py.",
    )
    parser.add_argument(
        "--collection-name", type=str, required=True,
        help="Name of the ChromaDB collection to create or extend.",
    )
    parser.add_argument(
        "--chroma-path", type=str, default=".chromadb",
        help="Directory to store ChromaDB data (default: .chromadb).",
    )
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.chroma_path)
    collection = client.get_or_create_collection(name=args.collection_name)
    print(
        f"Writing to collection {args.collection_name!r} at {args.chroma_path}."
    )

    metadata = _load_metadata(args.embeddings_dir)
    print(f"Loaded metadata for {len(metadata)} elements.")

    npz_files = sorted(
        f for f in os.listdir(args.embeddings_dir)
        if f.startswith("embeddings") and f.endswith(".npz")
    )
    if not npz_files:
        raise FileNotFoundError(
            f"No embeddings_*.npz files found in {args.embeddings_dir!r}."
        )

    for fname in npz_files:
        print(f"  Processing {fname}...")
        _add_partition(
            collection,
            os.path.join(args.embeddings_dir, fname),
            metadata,
        )

    print(f"Done. Collection {args.collection_name!r} now has {collection.count()} entries.")
