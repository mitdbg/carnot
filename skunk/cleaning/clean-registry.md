# OCR Cleaning Sample Registry

## Purpose

This registry documents the representative Treasury Bulletin sample used as a
development set for finding and correcting OCR extraction mistakes introduced
between the source PDFs and parsed JSON files.

## Selection

The sample contains 27 bulletins:

- 3 decades: 1940s, 1960s, 1980s
- 3 adjacent years per decade:
  - 1940s: 1940, 1941, 1942
  - 1960s: 1960, 1961, 1962
  - 1980s: 1980, 1981, 1982
- 3 adjacent monthly issues per year: June, July, August

This yields:

```text
3 decades x 3 years x 3 issues = 27 bulletins
```

## Files

The generated sample lives at:

```text
/orcd/home/002/gerarvit/orcd/scratch/officeqa/ocr_cleaning/dev_samples/
```

Contents:

- `manifest.csv`: row-oriented manifest for scripts and spreadsheet review.
- `manifest.json`: structured manifest with sample metadata and rows.
- `pdfs/`: symlinks to the selected source PDFs.
- `parsed_json/`: symlinks to the selected parsed JSON files.
- `extracted_tables/tables.json`: filtered copy of the extracted-table manifest
  containing only tables from the sampled bulletin issues.
- `extracted_tables/images/`: symlinks to extracted table PNGs referenced by the
  filtered table manifest.

Current generated counts:

- 27 manifest rows
- 27 PDF symlinks
- 27 parsed JSON symlinks
- Table count depends on the current extracted-table dump.

## Script

The repeatable builder is:

```text
scripts/build_ocr_cleaning_sample.py
```

Run it from the repository root:

```bash
python scripts/build_ocr_cleaning_sample.py
```

The script refreshes symlinks and rewrites both manifests. It refuses to replace
non-symlink files in the generated `pdfs/`, `parsed_json/`, or
`extracted_tables/images/` folders.

## Source Paths

The current server-oriented script uses fixed ORCD paths:

- `/orcd/home/002/gerarvit/orcd/scratch/officeqa/treasury_bulletin_pdfs_no_ocr`
- `/orcd/home/002/gerarvit/orcd/scratch/officeqa/treasury_bulletins_parsed/jsons`
- `/orcd/home/002/gerarvit/orcd/scratch/officeqa/extracted_tables`
- `/orcd/home/002/gerarvit/orcd/scratch/officeqa/ocr_cleaning/dev_samples`

## Validation

Before writing bulletin symlinks or manifests, the script checks that every
selected PDF and parsed JSON file exists. Missing sources are printed and the
script exits without creating the sample. It also requires
`extracted_tables/tables.json`, filters rows to the sampled bulletins, verifies
each referenced PNG exists, and writes a sample-local filtered table manifest.

Each manifest row includes:

- `bulletin`
- `decade`
- `year`
- `month`
- `pdf_source`
- `json_source`
- `pdf_link`
- `json_link`
