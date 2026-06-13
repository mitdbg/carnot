"""Build the 27-bulletin OCR-cleaning dev sample.

The sample is fixed by design:
  - decades: 1940s, 1960s, 1980s
  - years per decade: 1940-1942, 1960-1962, 1980-1982
  - issues per year: June, July, August

The script writes manifests and symlinks source PDFs / parsed JSON files into
dev_samples/ocr_cleaning by default.
"""

from __future__ import annotations

import csv
import json
import os


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = "/orcd/home/002/gerarvit/orcd/scratch/officeqa/"
SAMPLE_ROOT = f"{DATA_DIR}/ocr_cleaning/dev_samples"

PDF_DIR = f"{DATA_DIR}/treasury_bulletin_pdfs_no_ocr"
PARSED_JSON_DIR = f"{DATA_DIR}/treasury_bulletins_parsed/jsons"
EXTRACTED_TABLES_DIR = f"{DATA_DIR}/extracted_tables"
SAMPLE_GRID = [
    ("1940s", [1940, 1941, 1942]),
    ("1960s", [1960, 1961, 1962]),
    ("1980s", [1980, 1981, 1982]),
]
ISSUE_MONTHS = [6, 7, 8]


rows = []
missing = []
sample_bulletins = set()

for decade, years in SAMPLE_GRID:
    for year in years:
        for month in ISSUE_MONTHS:
            bulletin = f"{year}-{month:02d}"
            sample_bulletins.add(bulletin)
            pdf_name = f"treasury_bulletin_{year}_{month:02d}.pdf"
            json_name = f"treasury_bulletin_{year}_{month:02d}.json"
            pdf_source = os.path.join(PDF_DIR, pdf_name)
            json_source = os.path.join(PARSED_JSON_DIR, json_name)

            if not os.path.isfile(pdf_source):
                missing.append(pdf_source)
            if not os.path.isfile(json_source):
                missing.append(json_source)

            rows.append(
                {
                    "bulletin": bulletin,
                    "decade": decade,
                    "year": str(year),
                    "month": f"{month:02d}",
                    "pdf_source": pdf_source,
                    "json_source": json_source,
                    "pdf_link": os.path.join("pdfs", pdf_name),
                    "json_link": os.path.join("parsed_json", json_name),
                }
            )

if missing:
    print("Missing source files:")
    for path in missing:
        print(f"  {path}")
    raise SystemExit(1)

pdf_target_dir = os.path.join(SAMPLE_ROOT, "pdfs")
json_target_dir = os.path.join(SAMPLE_ROOT, "parsed_json")
tables_target_dir = os.path.join(SAMPLE_ROOT, "extracted_tables")
table_images_target_dir = os.path.join(tables_target_dir, "images")
os.makedirs(pdf_target_dir, exist_ok=True)
os.makedirs(json_target_dir, exist_ok=True)
os.makedirs(table_images_target_dir, exist_ok=True)

for row in rows:
    for source_key, link_key in [("pdf_source", "pdf_link"), ("json_source", "json_link")]:
        target = os.path.join(SAMPLE_ROOT, row[link_key])
        if os.path.lexists(target):
            if not os.path.islink(target):
                raise SystemExit(f"Refusing to replace non-symlink target: {target}")
            os.unlink(target)
        os.symlink(row[source_key], target)

tables_manifest_source = os.path.join(EXTRACTED_TABLES_DIR, "tables.json")
if not os.path.isfile(tables_manifest_source):
    raise SystemExit(f"Extracted tables manifest not found: {tables_manifest_source}")

with open(tables_manifest_source) as f:
    tables_manifest = json.load(f)

sample_tables = []
missing_table_images = []
for table in tables_manifest.get("tables", []):
    if table.get("bulletin") not in sample_bulletins:
        continue

    image_source = table.get("image_path") or ""
    if not os.path.isfile(image_source):
        missing_table_images.append(image_source)
        continue

    image_name = os.path.basename(image_source)
    image_link = os.path.join("extracted_tables", "images", image_name)
    image_target = os.path.join(SAMPLE_ROOT, image_link)
    if os.path.lexists(image_target):
        if not os.path.islink(image_target):
            raise SystemExit(f"Refusing to replace non-symlink target: {image_target}")
        os.unlink(image_target)
    os.symlink(image_source, image_target)

    table_copy = dict(table)
    table_copy["image_link"] = image_link
    sample_tables.append(table_copy)

if missing_table_images:
    print("Missing extracted table image files:")
    for path in missing_table_images:
        print(f"  {path}")
    raise SystemExit(1)

sample_tables_manifest = {
    "source_tables_manifest": tables_manifest_source,
    "source_output_dir": tables_manifest.get("output_dir"),
    "sample_root": SAMPLE_ROOT,
    "table_count": len(sample_tables),
    "tables": sample_tables,
}

sample_tables_manifest_path = os.path.join(tables_target_dir, "tables.json")
with open(sample_tables_manifest_path, "w") as f:
    json.dump(sample_tables_manifest, f, ensure_ascii=False)
    f.write("\n")

manifest_csv = os.path.join(SAMPLE_ROOT, "manifest.csv")
manifest_json = os.path.join(SAMPLE_ROOT, "manifest.json")
fieldnames = [
    "bulletin",
    "decade",
    "year",
    "month",
    "pdf_source",
    "json_source",
    "pdf_link",
    "json_link",
]

with open(manifest_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with open(manifest_json, "w") as f:
    json.dump(
        {
            "purpose": "OCR-cleaning dev sample for Treasury Bulletin PDF-to-JSON review",
            "sample_root": SAMPLE_ROOT,
            "pdf_dir": PDF_DIR,
            "parsed_json_dir": PARSED_JSON_DIR,
            "extracted_tables_dir": EXTRACTED_TABLES_DIR,
            "selection": {
                "decades": ["1940s", "1960s", "1980s"],
                "years": {
                    "1940s": [1940, 1941, 1942],
                    "1960s": [1960, 1961, 1962],
                    "1980s": [1980, 1981, 1982],
                },
                "months": ["06", "07", "08"],
            },
            "bulletin_count": len(rows),
            "table_count": len(sample_tables),
            "rows": rows,
        },
        f,
        indent=2,
    )
    f.write("\n")

print(f"Wrote {len(rows)} bulletin rows to {manifest_csv}")
print(f"Wrote {manifest_json}")
print(f"Symlinked PDFs in {pdf_target_dir}")
print(f"Symlinked parsed JSON in {json_target_dir}")
print(f"Wrote {len(sample_tables)} sampled table rows to {sample_tables_manifest_path}")
print(f"Symlinked extracted table PNGs in {table_images_target_dir}")
