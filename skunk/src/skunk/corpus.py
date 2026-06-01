"""Unified corpus access for the Treasury Bulletin corpus: path resolution, parsed-JSON
reading (LRU-cached, indexed by 1-based PDF page), PyMuPDF page rendering, and text
cleaning. Deliberately free of `HarnessContext`/orchestrator coupling so the offline
prep scripts can import it.

Text accessors read parsed JSON only (cleaner than PyMuPDF text on scanned 1940s-50s
pages); PyMuPDF is used solely for rendering page images."""

from __future__ import annotations

import base64
import html as _html
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import fitz

from skunk.errors import StepFailed

_FILENAME_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})\.pdf$")

_PARSED_JSON_DEFAULT_DIR = Path.home() / "Desktop/officeqa/treasury_bulletins_parsed/jsons"
_PDF_DEFAULT_DIR = Path.home() / "Desktop/officeqa/treasury_bulletin_pdfs"


def parse_bulletin_filename(path: str | Path) -> str:
    """treasury_bulletin_1953_06.pdf  ->  '1953-06'."""
    m = _FILENAME_RE.search(str(path))
    if not m:
        raise ValueError(f"not a treasury bulletin filename: {path}")
    return f"{m.group(1)}-{m.group(2)}"


def parsed_json_dir() -> Path:
    """Resolve the parsed-JSON directory (env override, then default)."""
    d = os.environ.get("OFFICEQA_PARSED_JSON_DIR")
    return Path(d) if d else _PARSED_JSON_DEFAULT_DIR


def pdf_dir_from_env() -> Path:
    """Resolve the PDF corpus directory (env override, then default)."""
    d = os.environ.get("OFFICEQA_PDF_DIR")
    return Path(d) if d else _PDF_DEFAULT_DIR


def pdf_path_for(bulletin: str, pdf_dir: Path | str | None = None) -> Path:
    """Inverse of `parse_bulletin_filename`: '1953-06' -> <dir>/treasury_bulletin_1953_06.pdf."""
    year, mon = bulletin.split("-")
    base = Path(pdf_dir) if pdf_dir is not None else pdf_dir_from_env()
    return base / f"treasury_bulletin_{int(year):04d}_{int(mon):02d}.pdf"


@lru_cache(maxsize=64)
def load_parsed_doc(month: str, base_dir: str | None = None) -> dict:
    """Load a bulletin's parsed-JSON document (LRU-cached). Raises FileNotFoundError
    if missing; json errors propagate."""
    year, mon = month.split("-")
    base = Path(base_dir) if base_dir is not None else parsed_json_dir()
    p = base / f"treasury_bulletin_{year}_{mon}.json"
    if not p.exists():
        raise FileNotFoundError(f"parsed-JSON not found: {p}")
    return json.loads(p.read_text())


def page_elements(
    month: str,
    *,
    base_dir: str | Path | None = None,
    fill_gaps: bool = False,
) -> dict[int, list[dict]]:
    """`{1-based PDF page index → list of parsed-JSON element dicts}`. `fill_gaps=False`
    (default) includes only pages carrying elements; `fill_gaps=True` includes every
    page `1..max_page` (empty list for blanks) — the dense view the build pipeline wants."""
    base = str(base_dir) if base_dir is not None else None
    doc = load_parsed_doc(month, base)

    by_page: dict[int, list[dict]] = {}
    max_page = 0
    for el in doc.get("document", {}).get("elements", []):
        bbox = el.get("bbox") or []
        if not bbox:
            continue
        pid = bbox[0].get("page_id")
        if pid is None:
            continue
        pid = int(pid)
        by_page.setdefault(pid, []).append(el)
        if pid > max_page:
            max_page = pid

    if fill_gaps:
        return {p: by_page.get(p, []) for p in range(1, max_page + 1)}
    return by_page


def get_page_text(month: str | None, page: int | None) -> str | None:
    """Concatenated parsed-JSON `content` for a PDF page (extract Tier 1; HTML tables
    verbatim). Raises `StepFailed` if the source is missing/corrupt; returns None when
    the source is healthy but this page has no content."""
    if month is None or page is None:
        return None
    try:
        idx = page_elements(month)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
        raise StepFailed("extract", f"parsed-JSON unavailable for {month}: {e}") from e
    elements = idx.get(int(page))
    if not elements:
        return None
    parts = [el["content"] for el in elements if el.get("content") is not None]
    return "\n\n".join(parts) if parts else None


def page_text_tagged(month: str, *, base_dir: str | Path | None = None) -> dict[int, str]:
    """`{1-based PDF page index: concatenated page text}` for the build pipeline. Each
    element is prefixed with its parsed `[type]` for structural signal; `page_number`
    elements are dropped, blank pages map to "". Raises FileNotFoundError if missing."""
    out: dict[int, str] = {}
    for pdf_page, elements in page_elements(month, base_dir=base_dir, fill_gaps=True).items():
        parts: list[str] = []
        for el in elements:
            content = el.get("content")
            if content is None:
                continue
            t = el.get("type") or "text"
            if t == "page_number":
                continue
            parts.append(f"[{t}] {content}")
        out[pdf_page] = "\n\n".join(parts) if parts else ""
    return out


def render_page_b64(
    month: str | None,
    page: int | None,
    *,
    dpi: int = 300,
    fmt: str = "png",
    jpg_quality: int | None = None,
    pdf_dir: Path | str | None = None,
) -> tuple[str, str] | None:
    """Render a PDF page to in-memory image bytes → (mime, base64). `fmt` is "png"
    (lossless) or "jpg" (smaller; honors `jpg_quality`). Returns None when the PDF
    doesn't exist; PyMuPDF errors propagate. No disk cache."""
    if month is None or page is None or int(page) <= 0:
        return None
    pdf_path = pdf_path_for(month, pdf_dir)
    if not pdf_path.exists():
        return None
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    with fitz.open(pdf_path) as doc:
        pix = doc[int(page) - 1].get_pixmap(matrix=mat)
    if fmt == "jpg":
        data = pix.tobytes("jpg", jpg_quality=jpg_quality if jpg_quality is not None else 95)
        mime = "image/jpeg"
    else:
        data = pix.tobytes("png")
        mime = "image/png"
    return mime, base64.standard_b64encode(data).decode()


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TD_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL)
_MULTI_NL_RE = re.compile(r"\n{3,}")
_LONG_DOTS_RE = re.compile(r"\.{4,}")
# Four-digit years 1776–2026 as whole tokens.
_YEAR_RE = re.compile(r"\b(177[6-9]|17[89]\d|1[89]\d\d|200\d|201\d|202[0-6])\b")
# Cells with no semantic content (pure numbers, money/percent, ranges, placeholders).
_PLACEHOLDER_CELL_RE = re.compile(
    r"[+\-]?\$\s*[\d,]+(\.\d+)?%?"   # $1,234.56 / $ 194.3
    r"|[+\-]?[\d,]+(\.\d+)?%?"        # 1,234 / 3.5 / 50%
    r"|\d[\d,]*[\/\-]\d[\d,]*"         # 4-5 / 283/444
    r"|\$\s*-+"                         # $ -- / $ -
    r"|\$\s*\.\d+"                      # $ .6 / $.1
    r"|\.\d+"                           # .6 / .2 (bare decimals)
    r"|\*+"                             # * / ** (footnote markers)
    r"|-{2,}"                           # -- / --- (dash placeholders)
)


def page_plain_text(elements: list[dict]) -> str:
    """Reduce one page's parsed-JSON elements to plain text: drops `page_number`
    elements, strips HTML tags from `table` content, joins the rest with newlines."""
    parts: list[str] = []
    for el in elements:
        t = (el.get("type") or "").lower()
        if t == "page_number":
            continue
        content = el.get("content")
        if not content:
            continue
        if t == "table":
            content = _HTML_TAG_RE.sub(" ", str(content))  # strip HTML tags from table markup
        parts.append(str(content))
    return "\n".join(parts)


def preprocess_text(text: str, elt_type: str, strip_years: bool = False) -> str:
    """Preprocess element text for embedding: tables → tag-stripped non-placeholder
    cells joined by spaces; others → collapsed newline runs. Always drops dot-leader
    runs; `strip_years` also removes 1776–2026 year tokens."""
    if elt_type == "table":
        # inner text of each <td>/<th> cell (HTML-unescaped + tag-stripped),
        # dropping pure-number/placeholder cells (page numbers, dot leaders, …)
        cells = []
        for m in _TD_RE.finditer(text):
            cell = _HTML_TAG_RE.sub("", _html.unescape(m.group(1))).strip()
            if not _PLACEHOLDER_CELL_RE.fullmatch(cell):
                cells.append(cell)
        text = " ".join(cells)
    else:
        text = _MULTI_NL_RE.sub("\n\n", text)

    text = _LONG_DOTS_RE.sub("", text)

    if strip_years:
        text = _YEAR_RE.sub("", text)

    return text.strip()
