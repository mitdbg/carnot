"""Tier 0 — corpus manifest builder.

Walk a directory of Treasury bulletin PDFs + optional OCR text files and
produce manifest.csv with columns:
  year, month, pdf_path, ocr_txt_path, n_pages, render_dir

Usage:
    python -m skunk.prep.manifest \
        --bulletin-dir /path/to/pdfs \
        --output cache/manifest.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Patterns for bulletin filename to (year, month) resolution.
# Expected formats: 1940-01.pdf, treasury_1940_01.pdf, tb_1940_01.pdf, etc.
_FILENAME_PATTERNS = [
    re.compile(r"(\d{4})[-_](\d{2})"),     # 1940-01 or 1940_01
    re.compile(r"(\d{4})(\d{2})\.pdf$"),   # 194001.pdf
]


def _parse_year_month(stem: str) -> tuple[int, str] | None:
    """Return (year, 'YYYY-MM') from a filename stem, or None."""
    for pat in _FILENAME_PATTERNS:
        m = pat.search(stem)
        if m:
            year, month = int(m.group(1)), int(m.group(2))
            if 1900 <= year <= 2100 and 1 <= month <= 12:
                return year, f"{year:04d}-{month:02d}"
    return None


def build_manifest(bulletin_dir: str, output_path: str) -> None:
    import pandas as pd

    bulletin_dir_path = Path(bulletin_dir)
    if not bulletin_dir_path.exists():
        raise FileNotFoundError(f"Bulletin directory not found: {bulletin_dir}")

    records = []
    pdf_files = sorted(bulletin_dir_path.rglob("*.pdf"))

    for pdf in pdf_files:
        parsed = _parse_year_month(pdf.stem)
        if parsed is None:
            print(f"[manifest] Skipping (cannot parse year/month): {pdf.name}")
            continue
        year, month_str = parsed

        # OCR text file — try same name with .txt extension
        ocr_candidates = [
            pdf.with_suffix(".txt"),
            pdf.parent / (pdf.stem + "_ocr.txt"),
            pdf.parent / (pdf.stem + ".txt"),
        ]
        ocr_txt_path = next((str(p) for p in ocr_candidates if p.exists()), None)

        # Page count (fast — just open the PDF)
        n_pages = _count_pages(pdf)

        # Render dir (will be populated lazily by pages.py)
        render_dir = str(Path(output_path).parent / "pages" / month_str)

        records.append({
            "year": year,
            "month": month_str,
            "pdf_path": str(pdf),
            "ocr_txt_path": ocr_txt_path or "",
            "n_pages": n_pages,
            "render_dir": render_dir,
        })

    if not records:
        print(f"[manifest] WARNING: no parseable PDFs found in {bulletin_dir}")

    df = pd.DataFrame(records, columns=["year", "month", "pdf_path", "ocr_txt_path", "n_pages", "render_dir"])
    df = df.sort_values(["year", "month"]).reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[manifest] Wrote {len(df)} rows to {output_path}")
    if len(df) > 0:
        years = sorted(df["year"].unique())
        print(f"[manifest] Years covered: {min(years)}–{max(years)}")


def _count_pages(pdf_path: Path) -> int:
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bulletin-dir", required=True, help="Directory containing bulletin PDFs")
    parser.add_argument("--output", default="cache/manifest.csv")
    args = parser.parse_args()
    build_manifest(args.bulletin_dir, args.output)
