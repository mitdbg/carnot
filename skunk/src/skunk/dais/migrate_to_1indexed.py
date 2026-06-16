"""One-off migration: shift DAIS ``page_id`` from 0-indexed to 1-indexed.

The DAIS pipeline originally emitted **0-indexed** ``page_id`` (parser-native
``bbox[0].page_id``), so ``…_c40_7`` was actually the PDF's 8th page. This script shifts
every artifact by **+1** so ``page_id`` matches the PDF viewer's page number.

**No re-embedding is needed**: embedding vectors are a pure function of element *text*, not
``page_id``. We only rewrite id strings + metadata; the float arrays are copied verbatim.

What it migrates (each step is independently idempotent — guarded by a ``*.0indexed.bak``
backup; if the backup already exists that artifact is treated as already-migrated):

1. **Source parsed JSONs** (mutate in place): every ``el["bbox"][k]["page_id"] += 1``.
   The whole dir is copied once to ``<dir>_0indexed_backup/`` before any write.
2. **``metadata.json``**: re-key by the new ``chunk_id``; bump ``page_id`` + ``page_key``.
3. **``embeddings_{p}.npz``**: remap ``unique_element_ids`` (vectors untouched).
4. **Clean artifacts** (optional dir): ``clean_page_map.json`` (re-key ``doc_id``, rename
   ``{doc_id}.txt``) and ``table_corrections_map.json`` (re-key ``chunk_id``, bump the stored
   ``page_id``, rename ``{doc_id}_table_{eid}.md``).

It does **not** touch ChromaDB. After running, rebuild the collection OFFLINE (server down):

    rm -rf .chromadb-dais-slim
    python3 -m skunk.dais.create_vector_db \\
        --embeddings-dir dais_embeddings --collection-name dais-slim \\
        --chroma-path .chromadb-dais-slim

Usage:
    python3 -m skunk.dais.migrate_to_1indexed \\
        --parsed-json-dir "officeqa_gdrive/Parsed JSON" \\
        --embeddings-dir dais_embeddings \\
        --cleaned-dir dais_cleaned          # optional
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

from skunk.dais.dais_common import chunk_id_of, doc_id_of

BAK = ".0indexed.bak"
_MIG = ".mig1idx"  # transient suffix for collision-free file renames


def _shift_doc_id(doc_id: str) -> str:
    """``{file_id}_{page_id}`` -> ``{file_id}_{page_id+1}`` (page is the trailing int)."""
    file_id, page = doc_id.rsplit("_", 1)
    return doc_id_of(file_id, int(page) + 1)


def _shift_chunk_id(chunk_id: str) -> str:
    """``{file_id}_{page_id}_{element_id}`` -> page+1 (page/elt are trailing ints)."""
    file_id, page, elt = chunk_id.rsplit("_", 2)
    return chunk_id_of(file_id, int(page) + 1, elt)


# --------------------------------------------------------------------- 1. JSONs
def migrate_json_dir(parsed_json_dir: str, dry_run: bool) -> None:
    backup = parsed_json_dir.rstrip("/") + "_0indexed_backup"
    if os.path.exists(backup):
        print(f"[json]   backup {backup!r} exists -> already migrated, skipping.")
        return
    files = [f for f in os.listdir(parsed_json_dir) if f.endswith(".json")]
    print(f"[json]   {len(files)} JSON files; backup -> {backup!r}")
    if dry_run:
        print("[json]   (dry-run) would copy dir + bump every bbox page_id by 1.")
        return
    shutil.copytree(parsed_json_dir, backup)
    n_elts = 0
    for fn in files:
        path = os.path.join(parsed_json_dir, fn)
        with open(path) as f:
            doc = json.load(f)
        for el in doc["document"]["elements"]:
            for bb in el.get("bbox", []):
                if "page_id" in bb:
                    bb["page_id"] += 1
                    n_elts += 1
        with open(path, "w") as f:
            json.dump(doc, f)
    print(f"[json]   bumped page_id on {n_elts} bboxes across {len(files)} files.")


# ----------------------------------------------------------------- 2. metadata
def _load_original_metadata(metadata_path: str) -> dict:
    """Load the 0-indexed metadata (backup if the migration already ran, else the live file)."""
    bak = metadata_path + BAK
    with open(bak if os.path.exists(bak) else metadata_path) as f:
        return json.load(f)


def migrate_metadata(metadata_path: str, dry_run: bool) -> dict[str, str]:
    """Re-key metadata to 1-indexed; return the old_chunk_id -> new_chunk_id map (for the npz)."""
    original = _load_original_metadata(metadata_path)
    id_map: dict[str, str] = {}
    new_meta: dict[str, dict] = {}
    for old_cid, entry in original.items():
        new_pid = entry["page_id"] + 1
        new_cid = chunk_id_of(entry["file_id"], new_pid, entry["element_id"])
        id_map[old_cid] = new_cid
        new_entry = dict(entry)
        new_entry["page_id"] = new_pid
        new_entry["page_key"] = doc_id_of(entry["file_id"], new_pid)
        new_meta[new_cid] = new_entry
    assert len(new_meta) == len(original), "chunk_id collision after shift; aborting."

    if os.path.exists(metadata_path + BAK):
        print(f"[meta]   {metadata_path + BAK!r} exists -> already migrated; map rebuilt only.")
        return id_map
    print(f"[meta]   re-keying {len(new_meta)} entries (page_id += 1).")
    if not dry_run:
        shutil.copy2(metadata_path, metadata_path + BAK)
        with open(metadata_path, "w") as f:
            json.dump(new_meta, f)
    return id_map


# ---------------------------------------------------------------------- 3. npz
def migrate_npz(embeddings_dir: str, id_map: dict[str, str], dry_run: bool) -> None:
    parts = sorted(f for f in os.listdir(embeddings_dir)
                   if f.startswith("embeddings") and f.endswith(".npz"))
    for fn in parts:
        path = os.path.join(embeddings_dir, fn)
        if os.path.exists(path + BAK):
            print(f"[npz]    {fn}{BAK} exists -> skipping.")
            continue
        data = np.load(path)
        old_ids = [str(c) for c in data["unique_element_ids"]]
        new_ids = [id_map.get(c) or _shift_chunk_id(c) for c in old_ids]
        print(f"[npz]    {fn}: remapping {len(new_ids)} ids (vectors unchanged).")
        if dry_run:
            continue
        shutil.copy2(path, path + BAK)
        np.savez_compressed(path, embeddings=data["embeddings"],
                            unique_element_ids=np.array(new_ids))


# ------------------------------------------------------------- 4. clean artifacts
def _two_phase_rename(pairs: list[tuple[str, str]], dry_run: bool) -> None:
    """Rename old->new without collisions (e.g. page 0->1 while old page 1 still exists)."""
    if dry_run:
        return
    for old, _ in pairs:
        if old != _ and os.path.exists(old):
            os.rename(old, old + _MIG)
    for old, new in pairs:
        src = old + _MIG if os.path.exists(old + _MIG) else old
        if src != new and os.path.exists(src):
            os.rename(src, new)


def migrate_clean_page_map(cleaned_dir: str, dry_run: bool) -> None:
    path = os.path.join(cleaned_dir, "clean_page_map.json")
    if not os.path.exists(path):
        print(f"[clean]  {path!r} not found -> skipping clean_page_map.")
        return
    if os.path.exists(path + BAK):
        print(f"[clean]  clean_page_map.json{BAK} exists -> skipping.")
        return
    with open(path) as f:
        cpm = json.load(f)
    new_cpm: dict[str, list] = {}
    renames: list[tuple[str, str]] = []
    for old_doc_id, entry in cpm.items():
        new_doc_id = _shift_doc_id(old_doc_id)
        txt_path = entry[0]
        new_txt = os.path.join(os.path.dirname(txt_path), f"{new_doc_id}.txt")
        renames.append((txt_path, new_txt))
        new_cpm[new_doc_id] = [new_txt, entry[1]]
    print(f"[clean]  clean_page_map.json: {len(new_cpm)} doc_ids + .txt files.")
    if not dry_run:
        shutil.copy2(path, path + BAK)
        _two_phase_rename(renames, dry_run)
        with open(path, "w") as f:
            json.dump(new_cpm, f)


def migrate_table_corrections_map(cleaned_dir: str, dry_run: bool) -> None:
    path = os.path.join(cleaned_dir, "table_corrections_map.json")
    if not os.path.exists(path):
        print(f"[clean]  {path!r} not found -> skipping table_corrections_map.")
        return
    if os.path.exists(path + BAK):
        print(f"[clean]  table_corrections_map.json{BAK} exists -> skipping.")
        return
    with open(path) as f:
        tcm = json.load(f)
    new_tcm: dict[str, list] = {}
    renames: list[tuple[str, str]] = []
    for old_cid, entry in tcm.items():
        new_cid = _shift_chunk_id(old_cid)
        md_path, page_id = entry[0], entry[1]
        # md filename is f"{doc_id}_table_{eid}.md"; rebuild it from the shifted chunk id.
        cfile, cpage, celt = old_cid.rsplit("_", 2)
        new_md = os.path.join(os.path.dirname(md_path),
                              f"{doc_id_of(cfile, int(cpage) + 1)}_table_{celt}.md")
        renames.append((md_path, new_md))
        new_tcm[new_cid] = [new_md, page_id + 1]
    print(f"[clean]  table_corrections_map.json: {len(new_tcm)} chunk_ids + .md files.")
    if not dry_run:
        shutil.copy2(path, path + BAK)
        _two_phase_rename(renames, dry_run)
        with open(path, "w") as f:
            json.dump(new_tcm, f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Migrate DAIS page_ids 0-indexed -> 1-indexed.")
    ap.add_argument("--parsed-json-dir", help="Dir of parsed element JSONs to mutate in place.")
    ap.add_argument("--embeddings-dir", help="Dir with metadata.json + embeddings_*.npz.")
    ap.add_argument("--cleaned-dir", help="Optional dir with clean_page_map.json + .txt/.md.")
    ap.add_argument("--dry-run", action="store_true", help="Report actions without writing.")
    args = ap.parse_args()

    if args.parsed_json_dir:
        migrate_json_dir(args.parsed_json_dir, args.dry_run)
    if args.embeddings_dir:
        metadata_path = os.path.join(args.embeddings_dir, "metadata.json")
        id_map = migrate_metadata(metadata_path, args.dry_run)
        migrate_npz(args.embeddings_dir, id_map, args.dry_run)
    if args.cleaned_dir:
        migrate_clean_page_map(args.cleaned_dir, args.dry_run)
        migrate_table_corrections_map(args.cleaned_dir, args.dry_run)

    print("\nDone. Now rebuild the vector DB OFFLINE (Chroma server must be down):")
    print("    rm -rf .chromadb-dais-slim")
    print("    python3 -m skunk.dais.create_vector_db \\")
    print("        --embeddings-dir dais_embeddings --collection-name dais-slim \\")
    print("        --chroma-path .chromadb-dais-slim")


if __name__ == "__main__":
    main()
