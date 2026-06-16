"""Table-correction pass for the DAIS slim corpus.

Adapts ``search_agent/prep/table_corrector.py``. For each document we open the PDF once,
render each page once (the full-page image is the shared context for that page's tables),
and issue **one LLM call per table element** to re-emit it as corrected GitHub-flavored
Markdown. The cleaned page text is the document's elements in **parser order**, with each
table's raw HTML replaced by its corrected Markdown — this is what ``read_document`` serves.

Page-element *reordering* is intentionally **not** done: this corpus (the annual U.S.
Combined Statement of Receipts & Expenditures) is table-heavy with little free text, so
reordering isn't worth the extra LLM call per page. Pages with no tables make no LLM call.

Dataset specifics (differ from the treasury OfficeQA data this was derived from):
  * ``page_id`` is **1-indexed** (matches the PDF viewer's page number) → render
    ``fdoc[page_id - 1]`` (PyMuPDF is 0-indexed).
  * bbox coords are in **300-DPI pixel space** → coord->point scale is fixed at
    ``--bbox-dpi / 72`` (≈4.17), not estimated.

Identifiers are generic: ``doc_id = f"{file_id}_{page_id}"``.

Usage:
    python3 -m skunk.dais.clean_corpus \\
        --input-json-dir "officeqa_gdrive/Parsed JSON" \\
        --pdfs-dir officeqa_gdrive/PDFs --output-dir dais_cleaned
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from skunk.common import B64Image
from skunk.corpus import _HTML_TAG_RE
from skunk.dais.dais_common import DaisLLM, chunk_id_of, doc_id_of, file_id_of, pdf_path_for
from skunk.dais.render import (
    group_by_page,
    load_page_image,
    load_table_image,
    png_b64image,
    render_page_png_bytes,
    render_table_png_bytes,
)
# Reuse the (corpus-agnostic) prompt + bbox logic from the OfficeQA table corrector.
from skunk.search_agent.prep.table_corrector import (
    SYSTEM_PROMPT,
    TABLE_CORRECTOR_PROMPT,
    _strip_code_fence,
    _union_bbox,
)

MAX_RETRIES = 3
SERIALIZE_EVERY_N_PAGES = 100
# Parser bbox coords are pixels at this DPI (see render_officeqa_json_simple.py: "DPI must
# match the dpi the JSON bboxes were computed at"). coord -> PDF point = coord / (DPI/72).
DEFAULT_BBOX_DPI = 300


def _element_text(elt: dict) -> str:
    """Plain text for a non-corrected element: figure placeholder, table HTML stripped to
    text, everything else verbatim."""
    content = elt.get("content")
    if content is None:
        return f"<figure id={elt['id']}>"
    if elt.get("type") == "table":
        return _HTML_TAG_RE.sub(" ", str(content))
    return str(content)


def _correct_table(full_img: B64Image, crop_img: B64Image | None, table_html: str, llm: DaisLLM) -> str | None:
    """Re-emit one table as corrected Markdown; None on persistent failure."""
    images = [full_img] + ([crop_img] if crop_img is not None else [])
    user_text = TABLE_CORRECTOR_PROMPT.format(table_html=table_html)
    for _ in range(MAX_RETRIES):
        md = _strip_code_fence(llm.call(system=SYSTEM_PROMPT, user=user_text, images=images).text)
        if md:
            return md
    return None


def _assemble_page(elements: list[dict], corrections: dict[int, str]) -> str:
    """Cleaned page text: elements in parser order, tables replaced by corrected Markdown."""
    parts = []
    for elt in elements:
        if elt.get("type") == "table" and elt["id"] in corrections:
            parts.append(corrections[elt["id"]])
        else:
            parts.append(_element_text(elt))
    return "\n".join(parts)


def _write_page(output_dir: str, file_id: str, page_id: int, elements: list[dict],
                corrections: dict[int, str], clean_page_map: dict, table_map: dict) -> None:
    """Write one page's cleaned txt + any corrected-table .md files; update the maps."""
    doc_id = doc_id_of(file_id, page_id)
    txt_path = os.path.join(output_dir, f"{doc_id}.txt")
    with open(txt_path, "w") as f:
        f.write(_assemble_page(elements, corrections))
    clean_page_map[doc_id] = [txt_path, [e["id"] for e in elements]]  # parser order
    for eid, md in corrections.items():
        md_path = os.path.join(output_dir, f"{doc_id}_table_{eid}.md")
        with open(md_path, "w") as f:
            f.write(md)
        table_map[chunk_id_of(file_id, page_id, eid)] = [md_path, page_id]


def _process_document(
    json_path: str,
    pdfs_dir: str,
    output_dir: str,
    llm: DaisLLM,
    done_doc_ids: set[str],
    page_workers: int,
    bbox_dpi: int,
    text_fallback: bool,
    page_renders_dir: str,
    table_renders_dir: str,
) -> dict:
    """Clean one document. Returns ``{"clean_page_map", "table_map", "errors", "skipped"}``."""
    file_id = file_id_of(json_path)
    with open(json_path) as f:
        doc = json.load(f)

    by_page = group_by_page(doc)
    todo_pages = {p: els for p, els in by_page.items() if doc_id_of(file_id, p) not in done_doc_ids}
    clean_page_map: dict[str, list] = {}
    table_map: dict[str, list] = {}
    if not todo_pages:
        return {"clean_page_map": clean_page_map, "table_map": table_map, "errors": 0, "skipped": 0}

    pdf_path = pdf_path_for(file_id, pdfs_dir)
    if not pdf_path.exists():
        # No PDF: can't render for table correction. Default = skip (resume later);
        # --text-fallback emits LLM-free parser-order text so the doc stays readable.
        if not text_fallback:
            return {"clean_page_map": clean_page_map, "table_map": table_map,
                    "errors": 0, "skipped": len(todo_pages)}
        for page_id, elements in todo_pages.items():
            _write_page(output_dir, file_id, page_id, elements, {}, clean_page_map, table_map)
        return {"clean_page_map": clean_page_map, "table_map": table_map, "errors": 0, "skipped": 0}

    scale = bbox_dpi / 72.0

    # --- gather page images + table crops: load the pre-rendered PNGs (skunk.dais.render_corpus),
    # rendering any that are missing once via a single PDF open so this still runs standalone ---
    full_imgs: dict[int, B64Image] = {}
    crop_imgs: dict[tuple[int, int], B64Image | None] = {}
    fdoc = None
    try:
        for page_id, elements in todo_pages.items():
            tables = [e for e in elements if e.get("type") == "table" and e.get("content")]
            full = load_page_image(page_renders_dir, file_id, page_id)
            crops: dict[int, B64Image | None] = {}
            missing: list[dict] = []
            for elt in tables:
                img = load_table_image(table_renders_dir, file_id, page_id, elt["id"])
                if img is not None:
                    crops[elt["id"]] = img
                else:
                    missing.append(elt)

            if full is None or missing:  # fall back to an inline render for whatever's absent
                import fitz

                if fdoc is None:
                    fdoc = fitz.open(pdf_path)
                if page_id < 1 or page_id > fdoc.page_count:  # page_id is 1-indexed
                    continue
                pg = fdoc[page_id - 1]  # PyMuPDF is 0-indexed
                if full is None:
                    full = png_b64image(render_page_png_bytes(pg))
                for elt in missing:
                    data = render_table_png_bytes(pg, _union_bbox(elt), scale)
                    crops[elt["id"]] = png_b64image(data) if data is not None else None

            full_imgs[page_id] = full
            for elt in tables:
                crop_imgs[(page_id, elt["id"])] = crops.get(elt["id"])
    finally:
        if fdoc is not None:
            fdoc.close()

    # --- correct tables (parallel within the document); pages w/o tables skip the LLM ---
    errors = 0

    def _do_page(page_id: int, elements: list[dict]) -> tuple[int, dict[int, str]]:
        full = full_imgs.get(page_id)
        corrections: dict[int, str] = {}
        if full is None:
            return page_id, corrections
        for elt in elements:
            if elt.get("type") == "table" and elt.get("content"):
                md = _correct_table(full, crop_imgs.get((page_id, elt["id"])), elt["content"], llm)
                if md is not None:
                    corrections[elt["id"]] = md
        return page_id, corrections

    with ThreadPoolExecutor(max_workers=max(1, page_workers)) as pool:
        futures = [pool.submit(_do_page, p, els) for p, els in todo_pages.items()]
        for fut in as_completed(futures):
            page_id, corrections = fut.result()
            try:
                _write_page(output_dir, file_id, page_id, todo_pages[page_id],
                            corrections, clean_page_map, table_map)
            except Exception as e:  # noqa: BLE001
                print(f"  [{file_id}] error writing page {page_id}: {e}")
                errors += 1

    return {"clean_page_map": clean_page_map, "table_map": table_map, "errors": errors, "skipped": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Correct DAIS corpus tables; assemble parser-order pages.")
    parser.add_argument("--input-json-dir", required=True, help="Directory with parsed document JSONs.")
    parser.add_argument("--pdfs-dir", required=True, help="Directory with source PDFs (<file_id>.pdf).")
    parser.add_argument("--output-dir", default="dais_cleaned", help="Output dir for cleaned txt + maps.")
    parser.add_argument("--model-id", default="gemini-3.5-flash", help="LLM model (genai id).")
    parser.add_argument("--doc-workers", type=int, default=8, help="Documents processed concurrently.")
    parser.add_argument("--page-workers", type=int, default=8, help="Table corrections concurrent within a doc.")
    parser.add_argument("--bbox-dpi", type=int, default=DEFAULT_BBOX_DPI,
                        help="DPI the parser bbox pixel coords were computed at (default 300).")
    parser.add_argument("--page-renders-dir", default=str(Path.home() / "dais" / "page_renders"),
                        help="Dir of pre-rendered page PNGs from skunk.dais.render_corpus (default ~/dais/page_renders).")
    parser.add_argument("--table-renders-dir", default=str(Path.home() / "dais" / "table_renders"),
                        help="Dir of pre-rendered table PNGs from skunk.dais.render_corpus (default ~/dais/table_renders).")
    parser.add_argument("--text-fallback", action="store_true",
                        help="For JSONs with no PDF, emit LLM-free parser-order text (tables HTML-stripped).")
    parser.add_argument("--no-fallback", action="store_true", help="Disable Gemini->OpenRouter LLM fallback.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_json_dir):
        raise SystemExit(f"Input directory {args.input_json_dir} does not exist.")
    os.makedirs(args.output_dir, exist_ok=True)

    clean_page_map_path = os.path.join(args.output_dir, "clean_page_map.json")
    table_map_path = os.path.join(args.output_dir, "table_corrections_map.json")
    clean_page_map: dict[str, list] = {}
    table_map: dict[str, list] = {}
    if os.path.isfile(clean_page_map_path):
        with open(clean_page_map_path) as f:
            clean_page_map = json.load(f)
    if os.path.isfile(table_map_path):
        with open(table_map_path) as f:
            table_map = json.load(f)

    json_files = sorted(glob.glob(os.path.join(args.input_json_dir, "*.json")))
    print(f"Processing {len(json_files)} documents ({len(clean_page_map)} pages already done).")

    llm = DaisLLM(model=args.model_id, enable_fallback=not args.no_fallback)
    done_doc_ids = set(clean_page_map.keys())

    completed_pages, errored, skipped = 0, 0, 0
    pages_since_flush = 0
    with ThreadPoolExecutor(max_workers=max(1, args.doc_workers)) as pool:
        futures = {
            pool.submit(_process_document, jp, args.pdfs_dir, args.output_dir, llm, done_doc_ids,
                        args.page_workers, args.bbox_dpi, args.text_fallback,
                        args.page_renders_dir, args.table_renders_dir): jp
            for jp in json_files
        }
        with tqdm(total=len(json_files), desc="Cleaning documents", unit="doc") as pbar:
            for fut in as_completed(futures):
                frag = fut.result()
                clean_page_map.update(frag["clean_page_map"])
                table_map.update(frag["table_map"])
                completed_pages += len(frag["clean_page_map"])
                errored += frag["errors"]
                skipped += frag["skipped"]
                pages_since_flush += len(frag["clean_page_map"])
                if pages_since_flush >= SERIALIZE_EVERY_N_PAGES:
                    with open(clean_page_map_path, "w") as f:
                        json.dump(clean_page_map, f)
                    with open(table_map_path, "w") as f:
                        json.dump(table_map, f)
                    pages_since_flush = 0
                pbar.update(1)
                pbar.set_postfix(pages=completed_pages, tables=len(table_map), skipped=skipped, errors=errored)

    with open(clean_page_map_path, "w") as f:
        json.dump(clean_page_map, f)
    with open(table_map_path, "w") as f:
        json.dump(table_map, f)
    print(f"Done. Cleaned {completed_pages} pages ({len(table_map)} tables corrected); "
          f"skipped {skipped} pages (no PDF), errors {errored}.")


if __name__ == "__main__":
    main()
