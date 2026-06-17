"""Rebuild ``clean_page_map.json`` + ``table_corrections_map.json`` from on-disk outputs.

``clean_corpus.py`` writes each page's ``.txt`` and each corrected table's ``.md`` to disk
immediately, but only *flushes* the two map files periodically. If the run dies between
flushes (e.g. an SSH drop), every page/table can be fully cleaned on disk while the maps
still reflect an old checkpoint. This script reconstructs both maps from the parsed JSONs
(authoritative page/element structure) + whatever ``.txt``/``.md`` files exist — **no LLM
calls**, so it never redoes table corrections that are already done.

It reproduces ``clean_corpus._write_page``'s exact format:
  - ``clean_page_map[doc_id] = [abs_txt_path, [element_ids in parser order]]`` (only if the
    ``.txt`` exists).
  - ``table_corrections_map[chunk_id] = [abs_md_path, page_id]`` (only if the ``.md`` exists).

Before writing, it **verifies** that every entry in the existing (possibly-stale) map is
reproduced identically — a guard that the reconstruction logic matches what the cleaner
actually wrote. Existing maps are backed up to ``*.prerebuild.bak``.

Usage:
    python3 -m skunk.dais.rebuild_maps_from_disk \\
        --input-json-dir ~/dais/parsed_json --cleaned-dir ~/dais/cleaned [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from skunk.dais.dais_common import chunk_id_of, doc_id_of, file_id_of


def _group_by_page(doc: dict) -> dict[int, list[dict]]:
    """All elements grouped by page_id, parser order preserved (mirrors clean_corpus)."""
    by_page: dict[int, list[dict]] = {}
    for elt in doc["document"]["elements"]:
        by_page.setdefault(elt["bbox"][0]["page_id"], []).append(elt)
    return by_page


def rebuild(input_json_dir: str, cleaned_dir: str) -> tuple[dict, dict, int]:
    cleaned_dir = os.path.abspath(cleaned_dir)
    clean_page_map: dict[str, list] = {}
    table_map: dict[str, list] = {}
    missing_txt = 0
    for jp in sorted(glob.glob(os.path.join(input_json_dir, "*.json"))):
        file_id = file_id_of(jp)
        with open(jp) as f:
            doc = json.load(f)
        for page_id, elements in _group_by_page(doc).items():
            doc_id = doc_id_of(file_id, page_id)
            txt_path = os.path.join(cleaned_dir, f"{doc_id}.txt")
            if not os.path.exists(txt_path):
                missing_txt += 1
                continue
            clean_page_map[doc_id] = [txt_path, [e["id"] for e in elements]]
            for elt in elements:
                if elt.get("type") == "table" and elt.get("content"):
                    eid = elt["id"]
                    md_path = os.path.join(cleaned_dir, f"{doc_id}_table_{eid}.md")
                    if os.path.exists(md_path):
                        table_map[chunk_id_of(file_id, page_id, eid)] = [md_path, page_id]
    return clean_page_map, table_map, missing_txt


def _verify_superset(name: str, existing_path: str, rebuilt: dict) -> bool:
    """Every existing entry must be reproduced identically by the rebuild."""
    if not os.path.exists(existing_path):
        print(f"[verify] {name}: no existing file to check against.")
        return True
    with open(existing_path) as f:
        existing = json.load(f)
    mismatches = [k for k, v in existing.items() if rebuilt.get(k) != v]
    missing = [k for k in existing if k not in rebuilt]
    print(f"[verify] {name}: existing={len(existing)} rebuilt={len(rebuilt)} "
          f"| not-reproduced={len(missing)} | value-mismatch={len(mismatches)}")
    for k in (missing + mismatches)[:5]:
        print(f"         e.g. {k}: existing={existing.get(k)} rebuilt={rebuilt.get(k)}")
    return not missing and not mismatches


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild DAIS clean/table maps from on-disk files.")
    ap.add_argument("--input-json-dir", required=True)
    ap.add_argument("--cleaned-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cpm, tcm, missing_txt = rebuild(args.input_json_dir, args.cleaned_dir)
    print(f"Rebuilt clean_page_map: {len(cpm)} pages | table_corrections_map: {len(tcm)} tables "
          f"| pages missing a .txt: {missing_txt}")

    cpm_path = os.path.join(args.cleaned_dir, "clean_page_map.json")
    tcm_path = os.path.join(args.cleaned_dir, "table_corrections_map.json")
    ok_cpm = _verify_superset("clean_page_map", cpm_path, cpm)
    ok_tcm = _verify_superset("table_corrections_map", tcm_path, tcm)
    if not (ok_cpm and ok_tcm):
        raise SystemExit("Verification FAILED: rebuild does not reproduce existing entries; aborting.")

    if args.dry_run:
        print("Dry-run: verification passed; not writing.")
        return
    for path, data in ((cpm_path, cpm), (tcm_path, tcm)):
        if os.path.exists(path) and not os.path.exists(path + ".prerebuild.bak"):
            os.replace(path, path + ".prerebuild.bak")
            print(f"Backed up -> {path}.prerebuild.bak")
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Wrote {len(data)} entries -> {path}")


if __name__ == "__main__":
    main()
