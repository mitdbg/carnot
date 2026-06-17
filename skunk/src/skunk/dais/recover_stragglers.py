"""Queue uncorrected ``table`` stragglers for reprocessing by ``clean_corpus.py``.

After a clean run, a few tables can have non-empty ``content`` but no corrected ``.md``
(``_correct_table`` returned ``None`` — usually a transient LLM error). Their page ``.txt``
still holds the raw-HTML table, so they're not lost, just un-cleaned.

This finds every such straggler, then **drops the affected pages from
``clean_page_map.json``** (backup -> ``*.prestragglers.bak``). A subsequent
``clean_corpus.py`` run (which resumes by skipping ``clean_page_map`` keys) will reprocess
exactly those pages — re-correcting their tables and re-assembling the page ``.txt`` — and
nothing else. No LLM calls happen here.

Usage:
    python3 -m skunk.dais.recover_stragglers \\
        --input-json-dir ~/dais/parsed_json --cleaned-dir ~/dais/cleaned [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from skunk.dais.dais_common import chunk_id_of, doc_id_of, file_id_of


def find_stragglers(input_json_dir: str, cleaned_dir: str) -> tuple[list[str], set[str]]:
    """Return (straggler_chunk_ids, affected_page_doc_ids)."""
    cleaned_dir = os.path.abspath(cleaned_dir)
    stragglers: list[str] = []
    pages: set[str] = set()
    for jp in sorted(glob.glob(os.path.join(input_json_dir, "*.json"))):
        fid = file_id_of(jp)
        with open(jp) as f:
            doc = json.load(f)
        by_page: dict[int, list[dict]] = {}
        for el in doc["document"]["elements"]:
            by_page.setdefault(el["bbox"][0]["page_id"], []).append(el)
        for pid, els in by_page.items():
            for e in els:
                if e.get("type") == "table" and e.get("content"):
                    md = os.path.join(cleaned_dir, f"{doc_id_of(fid, pid)}_table_{e['id']}.md")
                    if not os.path.exists(md):
                        stragglers.append(chunk_id_of(fid, pid, e["id"]))
                        pages.add(doc_id_of(fid, pid))
    return stragglers, pages


def main() -> None:
    ap = argparse.ArgumentParser(description="Drop uncorrected-table pages from clean_page_map for re-run.")
    ap.add_argument("--input-json-dir", required=True)
    ap.add_argument("--cleaned-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stragglers, pages = find_stragglers(args.input_json_dir, args.cleaned_dir)
    print(f"Straggler tables (content, no .md): {len(stragglers)} across {len(pages)} pages.")
    for cid in stragglers[:40]:
        print("   ", cid)

    if not pages:
        print("Nothing to recover.")
        return

    cpm_path = os.path.join(args.cleaned_dir, "clean_page_map.json")
    with open(cpm_path) as f:
        cpm = json.load(f)
    present = [p for p in pages if p in cpm]
    print(f"\nOf those, {len(present)} pages are currently in clean_page_map "
          f"({len(pages) - len(present)} already absent).")

    if args.dry_run:
        print("Dry-run: not modifying clean_page_map.")
    else:
        bak = cpm_path + ".prestragglers.bak"
        if not os.path.exists(bak):  # full snapshot of the pre-drop map (don't clobber an earlier one)
            with open(bak, "w") as f:
                json.dump(cpm, f)
            print(f"Backed up full clean_page_map -> {bak}")
        for p in present:
            del cpm[p]
        with open(cpm_path, "w") as f:
            json.dump(cpm, f)
        print(f"Dropped {len(present)} pages; clean_page_map now has {len(cpm)} entries.")

    print("\nNext: re-run clean_corpus (resume reprocesses ONLY the dropped pages). Use tmux/nohup:")
    print("    cd ~/carnot/skunk && source venv/bin/activate")
    print("    python3 -m skunk.dais.clean_corpus \\")
    print("        --input-json-dir ~/dais/parsed_json --pdfs-dir ~/dais/pdfs \\")
    print("        --output-dir ~/dais/cleaned --doc-workers 16 --page-workers 8")
    print("Then re-run skunk.dais.recover_stragglers --dry-run to confirm 0 remain.")


if __name__ == "__main__":
    main()
