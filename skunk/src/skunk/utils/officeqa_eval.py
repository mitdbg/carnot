"""Helpers for parsing OfficeQA ground-truth annotations and ChromaDB metadata.

Ground-truth columns in officeqa_pro.csv:
  - source_docs  — newline-separated URLs like ".../january-1941-6529?page=15"
  - source_files — newline-separated filenames like "treasury_bulletin_1941_01.txt"

Page keys have the form "yyyy_mm_page_id" (e.g. "1941_01_15").
Doc  keys have the form "yyyy_mm"          (e.g. "1941_01").
"""
from __future__ import annotations

import re

_MONTH_NAMES: dict[str, str] = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}


def source_docs_to_page_keys(source_docs: str) -> list[str]:
    """Parse source_docs URLs -> page keys like '1941_01_15'."""
    keys: list[str] = []
    for url in str(source_docs).splitlines():
        url = url.strip()
        m = re.search(r"/([a-zA-Z]+)-(\d{4})-\d+\?page=(\d+)", url)
        if m:
            month_num = _MONTH_NAMES.get(m.group(1).lower())
            if month_num:
                keys.append(f"{m.group(2)}_{month_num}_{m.group(3)}")
    return keys


def source_files_to_doc_keys(source_files: str) -> list[str]:
    """Parse source_files like 'treasury_bulletin_1941_01.txt' -> doc keys like '1941_01'."""
    keys: list[str] = []
    for fname in str(source_files).splitlines():
        fname = fname.strip()
        m = re.search(r"_(\d{4})_(\d{2})", fname)
        if m:
            keys.append(f"{m.group(1)}_{m.group(2)}")
    return keys


def source_files_to_year_months(source_files: str) -> list[tuple[str, str]]:
    """Parse source_files like 'treasury_bulletin_1941_01.txt' -> [(year, month)] tuples."""
    return [tuple(k.split("_")) for k in source_files_to_doc_keys(source_files)]  # type: ignore[return-value]


def metadata_to_page_key(meta: dict) -> str:
    """Convert a ChromaDB element metadata dict to a page key like '1941_01_15'."""
    page_key = meta.get("page_key")
    if page_key:
        return str(page_key)
    return f"{meta.get('year')}_{meta.get('month')}_{meta.get('page_id')}"


def metadata_to_doc_key(meta: dict) -> str:
    """Convert a ChromaDB element metadata dict to a doc key like '1941_01'."""
    file_id = meta.get("file_id")
    if file_id:
        m = re.search(r"_(\d{4})_(\d{2})", str(file_id))
        if m:
            return f"{m.group(1)}_{m.group(2)}"
    return f"{meta.get('year')}_{meta.get('month')}"
