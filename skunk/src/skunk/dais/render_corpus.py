"""Pre-render the ENTIRE DAIS corpus to PNG on disk: one PNG per PDF page and one per
table element, in parallel across OS processes.

Why a standalone tool: rasterizing every page is a big, embarrassingly parallel, LLM-free
job, and PyMuPDF holds the GIL while it rasterizes — so threads buy nothing and only
separate processes give real parallelism. Unit of work is **one document**: a child process
opens each PDF once, renders every page, and renders a tight crop for every table element
in the matching parsed JSON. Run it once before the competition to warm the renders that
``clean_corpus`` (offline table correction) and the harness UI (page display) both read,
so neither rasterizes on the hot path.

Output (keyed by the SearchAgent ``doc_id`` / ``chunk_id`` — see ``render.py``):
  * page  -> ``<page-renders-dir>/<file_id>_<page_id>.png``     (every PDF page)
  * table -> ``<table-renders-dir>/<file_id>_<page_id>_<eid>.png`` (every table element)

Cache-first and resumable: a PNG that already exists is skipped, so a re-run only fills
gaps; writes are atomic so a killed run leaves no partial files.

Usage:
    python3 -m skunk.dais.render_corpus \\
        --input-json-dir "officeqa_gdrive/Parsed JSON" \\
        --pdfs-dir officeqa_gdrive/PDFs
    python3 -m skunk.dais.render_corpus ... --only combined_statement__modern__2001__c01
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from skunk.dais.dais_common import file_id_of, pdf_path_for
from skunk.dais.render import (
    atomic_write_png,
    group_by_page,
    page_render_path,
    render_page_png_bytes,
    render_table_png_bytes,
    table_render_path,
)
# Reuse the bbox union geometry from the OfficeQA table corrector.
from skunk.search_agent.prep.table_corrector import _union_bbox

# Parser bbox coords are pixels at this DPI (mirrors clean_corpus.py). coord -> PDF point
# = coord / (DPI / 72).
DEFAULT_BBOX_DPI = 300

_COUNT_KEYS = ("pages_rendered", "pages_cached", "tables_rendered", "tables_cached", "skipped")


def _json_path_for(file_id: str, input_json_dir: str) -> Path | None:
    """The parsed JSON for ``file_id`` (``<input_json_dir>/<file_id>.json``), or None."""
    p = Path(input_json_dir) / f"{file_id}.json"
    return p if p.exists() else None


def _render_document(
    file_id: str,
    pdfs_dir: str,
    input_json_dir: str,
    page_renders_dir: str,
    table_renders_dir: str,
    bbox_dpi: int,
) -> dict[str, int]:
    """Render every page + every table of one document. Returns per-document counts.

    Top-level and picklable so the process pool can fan it out. Page rasterization is
    CPU-bound and GIL-serialized, so this runs in a child process, not a thread."""
    import fitz

    counts = {k: 0 for k in _COUNT_KEYS}
    pdf_path = pdf_path_for(file_id, pdfs_dir)
    if not pdf_path.exists():  # no PDF: nothing to rasterize
        counts["skipped"] += 1
        return counts

    # Tables need parser bboxes; pages need only the PDF. A JSON-less PDF still gets pages.
    by_page: dict[int, list[dict]] = {}
    json_path = _json_path_for(file_id, input_json_dir)
    if json_path is not None:
        with open(json_path) as f:
            by_page = group_by_page(json.load(f))
    scale = bbox_dpi / 72.0

    with fitz.open(pdf_path) as fdoc:
        n_pages = fdoc.page_count
        for page_id in range(1, n_pages + 1):  # page_id is 1-indexed
            pg = None
            page_path = page_render_path(page_renders_dir, file_id, page_id)
            if page_path.exists():
                counts["pages_cached"] += 1
            else:
                pg = fdoc[page_id - 1]  # PyMuPDF is 0-indexed
                atomic_write_png(page_path, render_page_png_bytes(pg))
                counts["pages_rendered"] += 1

            for elt in by_page.get(page_id, []):
                if elt.get("type") != "table" or not elt.get("content"):
                    continue
                tbl_path = table_render_path(table_renders_dir, file_id, page_id, elt["id"])
                if tbl_path.exists():
                    counts["tables_cached"] += 1
                    continue
                if pg is None:
                    pg = fdoc[page_id - 1]
                data = render_table_png_bytes(pg, _union_bbox(elt), scale)
                if data is None:  # degenerate bbox: no crop (clean_corpus uses full page alone)
                    continue
                atomic_write_png(tbl_path, data)
                counts["tables_rendered"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-render DAIS pages + tables to PNG on disk.")
    parser.add_argument("--input-json-dir", required=True, help="Directory with parsed document JSONs.")
    parser.add_argument("--pdfs-dir", required=True, help="Directory with source PDFs (<file_id>.pdf).")
    parser.add_argument("--page-renders-dir", default=str(Path.home() / "dais" / "page_renders"),
                        help="Output dir for per-page PNGs (default ~/dais/page_renders).")
    parser.add_argument("--table-renders-dir", default=str(Path.home() / "dais" / "table_renders"),
                        help="Output dir for per-table PNGs (default ~/dais/table_renders).")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8,
                        help="Parallel render processes (one document per task).")
    parser.add_argument("--bbox-dpi", type=int, default=DEFAULT_BBOX_DPI,
                        help="DPI the parser bbox pixel coords were computed at (default 300).")
    parser.add_argument("--only", default=None,
                        help="Comma-separated file_ids to render (smoke test); default all PDFs found.")
    args = parser.parse_args()

    if not os.path.isdir(args.pdfs_dir):
        raise SystemExit(f"PDFs directory {args.pdfs_dir} does not exist.")
    os.makedirs(args.page_renders_dir, exist_ok=True)
    os.makedirs(args.table_renders_dir, exist_ok=True)

    file_ids = sorted({file_id_of(p) for p in glob.glob(os.path.join(args.pdfs_dir, "*.pdf"))})
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        file_ids = [fid for fid in file_ids if fid in wanted]
    if not file_ids:
        raise SystemExit(f"No matching PDFs found under {args.pdfs_dir}.")

    workers = max(1, min(args.workers, len(file_ids)))
    print(f"[render_corpus] {len(file_ids)} documents -> pages={args.page_renders_dir} "
          f"tables={args.table_renders_dir} ({workers} worker process(es))")

    agg = {k: 0 for k in _COUNT_KEYS}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_render_document, fid, args.pdfs_dir, args.input_json_dir,
                        args.page_renders_dir, args.table_renders_dir, args.bbox_dpi): fid
            for fid in file_ids
        }
        with tqdm(total=len(file_ids), desc="Rendering documents", unit="doc") as pbar:
            for fut in as_completed(futures):
                fid = futures[fut]
                try:
                    counts = fut.result()
                except Exception as e:  # noqa: BLE001 — one bad PDF shouldn't abort the batch
                    print(f"  [{fid}] render error: {e}")
                    agg["skipped"] += 1
                    pbar.update(1)
                    continue
                for k in _COUNT_KEYS:
                    agg[k] += counts[k]
                pbar.update(1)
                pbar.set_postfix(pages=agg["pages_rendered"], tables=agg["tables_rendered"],
                                 cached=agg["pages_cached"] + agg["tables_cached"])

    print(f"[render_corpus] done | {agg}")


if __name__ == "__main__":
    main()
