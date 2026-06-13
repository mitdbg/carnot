#!/usr/bin/env python3
"""Export parsed Treasury Bulletin tables and matching review PNGs."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

import fitz

from tools.pdf_file import parse_pdf_pages


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CLEAN_JSON_DIR = "cache/table_review_cleaned_json"

DATA_DIR = "../data/officeqa"
DATA_DIR = "/orcd/home/002/gerarvit/orcd/scratch/officeqa/"

DEFAULT_RAW_JSON_DIR = f"{DATA_DIR}/treasury_bulletins_parsed/jsons"
DEFAULT_PDF_DIR = f"{DATA_DIR}/treasury_bulletin_pdfs_no_ocr"
DEFAULT_OUTPUT_DIR = "extracted_tables"
DEFAULT_OUTPUT_DIR = f"{DATA_DIR}/extracted_tables"

PAGE_STORE_NAME_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})\.json$")
RAW_JSON_NAME_RE = re.compile(r"^treasury_bulletin_(?P<year>\d{4})_(?P<month>\d{2})\.json$")
BLOCK_RE = re.compile(r"\[(?P<type>[^\]]+)]\s*(?P<content>.*?)(?=\n\n\[[^\]]+]\s|\Z)", re.S)


def issue_from_filename(filename: str) -> tuple[str, str, str] | None:
    match = PAGE_STORE_NAME_RE.match(filename) or RAW_JSON_NAME_RE.match(filename)
    if not match:
        return None
    year = match.group("year")
    month = match.group("month")
    return year, month, f"{year}-{month}"


def tables_from_page_store_json(data: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    tables_by_page: dict[int, list[dict[str, Any]]] = {}
    for page_key, page_text in data.items():
        try:
            page = int(page_key)
        except ValueError:
            continue

        tables: list[dict[str, Any]] = []
        last_heading = ""
        fallback_text = ""
        for match in BLOCK_RE.finditer(str(page_text)):
            element_type = match.group("type").strip()
            content = match.group("content").strip()
            if not content or element_type == "page_number":
                continue
            if element_type in ("section_header", "page_header", "title"):
                last_heading = content[:200]
            elif element_type == "text" and not fallback_text:
                fallback_text = content[:200]
            elif element_type == "table" and "<table" in content.lower():
                label = last_heading or fallback_text or f"Table {len(tables) + 1}"
                tables.append({"content": content, "label": label, "bbox": None})

        if tables:
            tables_by_page[page] = tables
    return tables_by_page


def tables_from_raw_json(data: dict[str, Any]) -> tuple[dict[int, list[dict[str, Any]]], tuple[float, float] | None]:
    tables_by_page: dict[int, list[dict[str, Any]]] = {}
    headings_by_page: dict[int, str] = {}
    fallback_by_page: dict[int, str] = {}
    max_x = 0.0
    max_y = 0.0

    for element in data.get("document", {}).get("elements", []):
        bbox_items = element.get("bbox") or []
        if not bbox_items:
            continue

        coord = bbox_items[0].get("coord") or []
        if len(coord) == 4:
            max_x = max(max_x, float(coord[0]), float(coord[2]))
            max_y = max(max_y, float(coord[1]), float(coord[3]))

        page_id = bbox_items[0].get("page_id")
        if page_id is None:
            continue

        page = int(page_id)
        element_type = str(element.get("type") or "text")
        content = str(element.get("content") or "").strip()
        if not content or element_type == "page_number":
            continue

        if element_type in ("section_header", "page_header", "title"):
            headings_by_page[page] = content[:200]
            continue
        if element_type == "text" and page not in fallback_by_page:
            fallback_by_page[page] = content[:200]
            continue
        if element_type != "table" or "<table" not in content.lower():
            continue

        bbox = [float(value) for value in coord] if len(coord) == 4 else None
        tables = tables_by_page.setdefault(page, [])
        label = headings_by_page.get(page) or fallback_by_page.get(page) or f"Table {len(tables) + 1}"
        tables.append({"content": content, "label": label, "bbox": bbox})

    source_size = (max_x, max_y) if max_x > 0 and max_y > 0 else None
    return tables_by_page, source_size


def write_table_png(
    page_pngs: dict[int, bytes],
    page: int,
    output_path: str,
    bbox: list[float] | None,
    source_size: tuple[float, float] | None,
    margin_ratio: float,
    min_margin: float,
    right_margin_multiplier: float,
) -> str:
    page_png = page_pngs[page - 1]
    page_doc = fitz.open(stream=page_png, filetype="png")
    image_page = page_doc[0]
    clip = None
    image_scope = "page"

    if bbox and source_size:
        x0, y0, x1, y1 = bbox
        left = min(x0, x1)
        right = max(x0, x1)
        top = min(y0, y1)
        bottom = max(y0, y1)
        margin_x = max((right - left) * margin_ratio, min_margin)
        margin_y = max((bottom - top) * margin_ratio, min_margin)

        source_width, source_height = source_size
        left = max(0.0, left - margin_x)
        top = max(0.0, top - margin_y)
        right = min(source_width, right + margin_x * right_margin_multiplier)
        bottom = min(source_height, bottom + margin_y)

        scale_x = image_page.rect.width / source_width
        scale_y = image_page.rect.height / source_height
        clip = fitz.Rect(
            image_page.rect.x0 + left * scale_x,
            image_page.rect.y0 + top * scale_y,
            image_page.rect.x0 + right * scale_x,
            image_page.rect.y0 + bottom * scale_y,
        )
        if clip.width > 1 and clip.height > 1:
            image_scope = "table"
        else:
            clip = None
            image_scope = "page"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if clip is None:
        with open(output_path, "wb") as f:
            f.write(page_png)
    else:
        pix = image_page.get_pixmap(clip=clip, alpha=False)
        pix.save(output_path)
    page_doc.close()
    return image_scope


parser = argparse.ArgumentParser(description="Export all parsed tables as PNGs plus JSON metadata.")
parser.add_argument(
    "--json-dir",
    default=os.environ.get("TABLE_REVIEW_CLEAN_JSON_DIR") or DEFAULT_CLEAN_JSON_DIR,
    help="Directory containing cleaned page-store JSON files.",
)
parser.add_argument(
    "--original-json-dir",
    default=os.environ.get("TABLE_REVIEW_ORIGINAL_JSON_DIR")
    or os.environ.get("OFFICEQA_PARSED_JSON_DIR")
    or DEFAULT_RAW_JSON_DIR,
    help="Directory containing raw parsed JSON files with table bounding boxes.",
)
parser.add_argument(
    "--pdf-dir",
    default=os.environ.get("OFFICEQA_PDF_DIR") or DEFAULT_PDF_DIR,
    help="Directory containing treasury_bulletin_YYYY_MM.pdf files.",
)
parser.add_argument(
    "--output-dir",
    default=DEFAULT_OUTPUT_DIR,
    help="Directory where table PNGs and tables.json will be written.",
)
parser.add_argument(
    "--margin-ratio",
    type=float,
    default=0.2,
    help="Fraction of the bbox width/height added around each cropped table.",
)
parser.add_argument(
    "--min-margin",
    type=float,
    default=140.0,
    help="Minimum margin in source bbox coordinates added around each cropped table.",
)
parser.add_argument(
    "--right-margin-multiplier",
    type=float,
    default=2.0,
    help="Multiplier applied to the right-side horizontal margin.",
)
parser.add_argument(
    "--dpi",
    type=int,
    default=144,
    help="DPI used when rendering PDF pages in parallel.",
)
parser.add_argument(
    "--pdf-workers",
    type=int,
    default=None,
    help="Worker count for parallel PDF page rendering. Defaults to CPU count.",
)
args = parser.parse_args()

json_dir = os.path.abspath(os.path.join(REPO_ROOT, args.json_dir))
raw_json_dir = os.path.abspath(os.path.join(REPO_ROOT, args.original_json_dir)) if args.original_json_dir else None
pdf_dir = os.path.abspath(os.path.join(REPO_ROOT, args.pdf_dir))
output_dir = os.path.abspath(os.path.join(REPO_ROOT, args.output_dir))
os.makedirs(output_dir, exist_ok=True)

metadata_rows: list[dict[str, Any]] = []
json_filenames = sorted(name for name in os.listdir(json_dir) if issue_from_filename(name))

for clean_filename in json_filenames:
    year, month, bulletin = issue_from_filename(clean_filename) or ("", "", "")
    clean_json_path = os.path.join(json_dir, clean_filename)
    with open(clean_json_path) as f:
        clean_data = json.load(f)

    if isinstance(clean_data, dict) and "document" in clean_data:
        clean_tables_by_page, _ = tables_from_raw_json(clean_data)
    elif isinstance(clean_data, dict):
        clean_tables_by_page = tables_from_page_store_json(clean_data)
    else:
        continue

    raw_json_path = None
    raw_tables_by_page: dict[int, list[dict[str, Any]]] = {}
    raw_source_size = None
    if raw_json_dir:
        candidate_paths = [
            os.path.join(raw_json_dir, f"{bulletin}.json"),
            os.path.join(raw_json_dir, f"treasury_bulletin_{year}_{month}.json"),
        ]
        raw_json_path = next((path for path in candidate_paths if os.path.exists(path)), None)
        if raw_json_path:
            with open(raw_json_path) as f:
                raw_data = json.load(f)
            if isinstance(raw_data, dict):
                raw_tables_by_page, raw_source_size = tables_from_raw_json(raw_data)

    pdf_path = os.path.join(pdf_dir, f"treasury_bulletin_{year}_{month}.pdf")
    if not os.path.exists(pdf_path):
        print(f"Skipping {bulletin}: PDF not found at {pdf_path}")
        continue

    with fitz.open(pdf_path) as pdf_doc:
        page_count = len(pdf_doc)

    table_pages = []
    for page in sorted(clean_tables_by_page):
        if page < 1 or page > page_count:
            print(f"Skipping {bulletin} page {page}: page is outside PDF bounds")
            continue
        table_pages.append(page)
    if not table_pages:
        continue

    page_pngs = parse_pdf_pages(
        pdf_path,
        n_workers=args.pdf_workers,
        dpi=args.dpi,
        selected_page_indices=[page - 1 for page in table_pages],
    )

    for page in table_pages:
        clean_tables = clean_tables_by_page[page]
        raw_tables = raw_tables_by_page.get(page, [])
        for table_index, clean_table in enumerate(clean_tables):
            raw_table = raw_tables[table_index] if table_index < len(raw_tables) else {}
            bbox = raw_table.get("bbox") or clean_table.get("bbox")
            table_id = f"{year}_{month}_{page}_{table_index}"
            image_path = os.path.join(output_dir, f"images/{table_id}.png")
            image_scope = write_table_png(
                page_pngs,
                page,
                image_path,
                bbox,
                raw_source_size,
                args.margin_ratio,
                args.min_margin,
                args.right_margin_multiplier,
            )
            metadata_rows.append(
                {
                    "id": table_id,
                    "bulletin": bulletin,
                    "year": year,
                    "month": month,
                    "page": page,
                    "table_index": table_index,
                    "image_path": image_path,
                    "image_scope": image_scope,
                    "bbox": bbox,
                    "label": clean_table.get("label"),
                    "source_json_path": clean_json_path,
                    "original_json_path": raw_json_path,
                    "parsed_ocr": clean_table.get("content", ""),
                }
            )

manifest_path = os.path.join(output_dir, "tables.json")
manifest = {
    "json_dir": json_dir,
    "original_json_dir": raw_json_dir,
    "pdf_dir": pdf_dir,
    "output_dir": output_dir,
    "table_count": len(metadata_rows),
    "tables": metadata_rows,
}
with open(manifest_path, "w") as f:
    json.dump(manifest, f, ensure_ascii=False)
    f.write("\n")

print(f"Exported {len(metadata_rows)} table(s) to {output_dir}")
print(f"Wrote manifest to {manifest_path}")
