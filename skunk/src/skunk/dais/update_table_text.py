"""Replace each corrected table's stored document text in the DAIS collection.

By default the Chroma ``chroma:document`` for every element is the qwen-v2 *preprocess*
text (``preprocess_text(raw, ...)``) — for tables that's HTML-stripped cells, not the nice
LLM-corrected Markdown that ``clean_corpus.py`` produced. This script swaps the document
text of every corrected table (keys of ``table_corrections_map.json``) for its corrected
Markdown.

**Embeddings are preserved exactly.** The collection has no real embedding function (it was
built from precomputed qwen vectors), so a bare ``collection.update(documents=...)`` would
re-embed with the default MiniLM model and corrupt the 4096-dim vectors. We therefore read
each element's existing embedding and pass it back into ``update`` alongside the new
document — the vector is rewritten with itself (a no-op), so it stays the qwen-v2 vector and
no re-embedding occurs.

Run **OFFLINE** (PersistentClient — the Chroma server must be down over the same dir).

Usage:
    python3 -m skunk.dais.update_table_text \\
        --cleaned-dir ~/dais/dais_cleaned \\
        --chroma-path ~/dais/.chromadb-dais-slim --collection-name dais-slim
        [--also-metadata ~/dais/dais_embeddings/metadata.json] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os

import chromadb

BATCH = 1000  # well under Chroma's per-call max


def _read_md(md_path: str, cleaned_dir: str) -> str:
    """Read a corrected-table .md, tolerating relative/moved paths (mirror datagen's fallback)."""
    if not os.path.isabs(md_path) and not os.path.exists(md_path):
        md_path = os.path.join(cleaned_dir, os.path.basename(md_path))
    elif not os.path.exists(md_path):
        alt = os.path.join(cleaned_dir, os.path.basename(md_path))
        if os.path.exists(alt):
            md_path = alt
    with open(md_path) as f:
        return f.read()


def main() -> None:
    ap = argparse.ArgumentParser(description="Update DAIS table chunks with corrected Markdown.")
    ap.add_argument("--cleaned-dir", required=True,
                    help="clean_corpus.py output dir (holds table_corrections_map.json + .md).")
    ap.add_argument("--chroma-path", default=".chromadb-dais-slim",
                    help="PersistentClient dir (server must be DOWN).")
    ap.add_argument("--collection-name", default="dais-slim")
    ap.add_argument("--also-metadata",
                    help="Optional path to embeddings metadata.json; if given, also overwrite "
                         "each corrected table's 'cleaned' field so a future rebuild stays "
                         "consistent (embeddings are recomputed from raw, so this is safe).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tcm_path = os.path.join(args.cleaned_dir, "table_corrections_map.json")
    with open(tcm_path) as f:
        tcm = json.load(f)
    print(f"{len(tcm)} corrected tables in {tcm_path}.")

    # chunk_id -> corrected markdown
    new_docs: dict[str, str] = {cid: _read_md(entry[0], args.cleaned_dir) for cid, entry in tcm.items()}
    ids_all = list(new_docs.keys())

    client = chromadb.PersistentClient(path=args.chroma_path)
    collection = client.get_collection(name=args.collection_name)
    print(f"Collection {args.collection_name!r} has {collection.count()} elements.")

    updated, missing = 0, 0
    for i in range(0, len(ids_all), BATCH):
        batch = ids_all[i:i + BATCH]
        # fetch existing embeddings so we can rewrite the doc WITHOUT re-embedding (vector kept as-is)
        got = collection.get(ids=batch, include=["embeddings"])
        present = got["ids"]
        missing += len(batch) - len(present)
        if not present:
            continue
        embs = got["embeddings"]
        docs = [new_docs[cid] for cid in present]
        if not args.dry_run:
            collection.update(
                ids=present,
                documents=docs,
                embeddings=[list(e) for e in embs],  # pass existing vectors back -> no re-embed
            )
        updated += len(present)
        print(f"  {updated}/{len(ids_all)} updated...", end="\r")

    print(f"\nDone. Updated {updated} table documents; {missing} chunk_ids not in collection.")

    if args.also_metadata and not args.dry_run:
        with open(args.also_metadata) as f:
            meta = json.load(f)
        n = 0
        for cid, doc in new_docs.items():
            if cid in meta:
                meta[cid]["cleaned"] = doc
                n += 1
        bak = args.also_metadata + ".pre_table_md.bak"
        if not os.path.exists(bak):
            os.replace(args.also_metadata, bak)
            print(f"Backed up metadata -> {bak}")
        else:
            print(f"Backup {bak} already exists; not overwriting it.")
        with open(args.also_metadata, "w") as f:
            json.dump(meta, f)
        print(f"Updated 'cleaned' for {n} table entries in {args.also_metadata}.")


if __name__ == "__main__":
    main()
