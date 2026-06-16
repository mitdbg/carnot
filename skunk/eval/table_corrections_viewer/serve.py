#!/usr/bin/env python3
"""
Table Corrections Viewer — local HTTP server for inspecting the output of
`src/skunk/search_agent/prep/table_corrector.py`.

That prep script repairs each parsed `table` element into clean Markdown and writes a
`table_corrections_map.json` (key `{year}_{month}_{element_id}` → `[md_path, page_id]`)
plus one `.md` per table. This viewer lets you eyeball each correction against ground
truth: per table it shows, side by side, the PDF crop the table was extracted from, the
parser's original HTML grid, and the corrected Markdown — so structural fixes (and any new
mistakes) are obvious. Page renders reuse the prep script's own helpers.

Run from the skunk/ directory:

    venv/bin/python eval/table_corrections_viewer/serve.py

Or point it at a specific corrections dir / corpus / port:

    venv/bin/python eval/table_corrections_viewer/serve.py \
        --corrections-dir treasury_bulletins_tables_corrected \
        --input-json-dir treasury_bulletins_parsed/jsons \
        --pdfs-dir treasury_bulletin_pdfs --port 7072
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from skunk.search_agent.prep.table_corrector import (
    _page_heights,
    _union_bbox,
    estimate_coord_scale,
    render_page_and_crop,
)

_APP_HTML = pathlib.Path(__file__).parent / "app.html"


# ── Config (set in main) ─────────────────────────────────────────────────────

class _Cfg:
    corrections_dir: pathlib.Path
    input_json_dir: pathlib.Path
    pdfs_dir: pathlib.Path


def _split_key(key: str) -> tuple[str, str, int, int]:
    """`{year}_{month}_{page_id}_{element_id}` → (year, month, page_id, element_id).

    A table's identity is (page_id, element_id): element `id` only unique within a page."""
    year, month, page_id, eid = key.split("_", 3)
    return year, month, int(page_id), int(eid)


# ── Corpus access (cached) ───────────────────────────────────────────────────

@lru_cache(maxsize=64)
def _doc(year: str, month: str) -> dict:
    """Parsed document JSON for one month, with a (page_id, id) element index and the coord
    scale pre-computed. Cached so flipping through tables on the same bulletin is cheap.
    Keyed on (page_id, id) because element `id` repeats across pages."""
    path = _Cfg.input_json_dir / f"treasury_bulletin_{year}_{month}.json"
    doc = json.loads(path.read_text())
    elements = doc["document"]["elements"]
    heights = _page_heights(f"{year}-{month}", str(_Cfg.pdfs_dir))
    return {
        "by_page_id": {(e["bbox"][0]["page_id"], e["id"]): e for e in elements if e.get("bbox")},
        "scale": estimate_coord_scale(elements, heights),
    }


def _load_map() -> dict[str, list]:
    path = _Cfg.corrections_dir / "table_corrections_map.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _list_corrections() -> list[dict]:
    """Summary row per entry in the corrections map, sorted by (year, month, element_id)."""
    rows = []
    for key in _load_map():
        try:
            year, month, page_id, eid = _split_key(key)
        except ValueError:
            continue
        rows.append({"key": key, "year": year, "month": month, "page_id": page_id, "element_id": eid})
    rows.sort(key=lambda r: (r["year"], r["month"], r["page_id"], r["element_id"]))
    return rows


def _read_markdown(md_path: str) -> str:
    """Corrected Markdown for an entry. The stored path is relative to the cwd of the prep
    run; fall back to resolving the basename inside the corrections dir so the viewer works
    regardless of where it's launched from."""
    p = pathlib.Path(md_path)
    if not p.exists():
        p = _Cfg.corrections_dir / pathlib.Path(md_path).name
    return p.read_text() if p.exists() else ""


def _correction_detail(key: str) -> dict | None:
    """Everything the frontend needs for one table: original HTML, corrected Markdown, and
    the full-page + crop renders (as data-URL-ready {mime,data})."""
    entry = _load_map().get(key)
    if entry is None:
        return None
    md_path, page_id = entry[0], entry[1]
    year, month, page_id, eid = _split_key(key)

    doc = _doc(year, month)
    elt = doc["by_page_id"].get((page_id, eid), {})
    html = elt.get("content", "") or ""
    coord = _union_bbox(elt) if elt.get("bbox") else (0, 0, 0, 0)

    full, crop = render_page_and_crop(f"{year}-{month}", int(page_id), coord, doc["scale"], str(_Cfg.pdfs_dir))
    as_img = lambda b: {"mime": b.mime, "data": b.data} if b else None

    return {
        "key": key, "year": year, "month": month, "element_id": eid, "page_id": page_id,
        "markdown": _read_markdown(md_path), "html": html,
        "full_image": as_img(full), "crop_image": as_img(crop),
    }


# ── HTTP handler ─────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:  # noqa: N802 — silence per-request stderr noise
        pass

    def _send_json(self, data: object, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            content = _APP_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if parsed.path == "/api/corrections":
            return self._send_json({"dir": str(_Cfg.corrections_dir), "corrections": _list_corrections()})

        if parsed.path == "/api/correction":
            key = qs.get("key", [""])[0]
            if not key:
                return self._send_json({"error": "missing key"}, 400)
            try:
                detail = _correction_detail(key)
            except Exception as e:  # surface render/parse errors to the UI instead of 500-ing
                return self._send_json({"error": str(e)}, 500)
            if detail is None:
                return self._send_json({"error": "unknown key"}, 404)
            return self._send_json(detail)

        self.send_response(404)
        self.end_headers()


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Table Corrections Viewer")
    parser.add_argument("--port", type=int, default=7072, help="Port to listen on (default: 7072)")
    parser.add_argument("--corrections-dir", default="treasury_bulletins_tables_corrected",
                        help="Dir with table_corrections_map.json + the .md files.")
    parser.add_argument("--input-json-dir", default="treasury_bulletins_parsed/jsons",
                        help="Parsed-document JSONs (source of the original HTML).")
    parser.add_argument("--pdfs-dir", default="treasury_bulletin_pdfs",
                        help="Source PDFs (for page/crop renders).")
    args = parser.parse_args()

    _Cfg.corrections_dir = pathlib.Path(args.corrections_dir).resolve()
    _Cfg.input_json_dir = pathlib.Path(args.input_json_dir).resolve()
    _Cfg.pdfs_dir = pathlib.Path(args.pdfs_dir).resolve()

    server = HTTPServer(("localhost", args.port), _Handler)
    url = f"http://localhost:{args.port}"
    print(f"Table Corrections Viewer → {url}")
    print(f"Corrections dir          → {_Cfg.corrections_dir}")
    if not (_Cfg.corrections_dir / "table_corrections_map.json").exists():
        print("  (no table_corrections_map.json yet — run table_corrector.py first)")
    print("Press Ctrl-C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
