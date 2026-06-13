#!/usr/bin/env python3
"""Local table extraction review UI."""

from __future__ import annotations

import argparse
import html
import json
import os
import pathlib
import re
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

APP_HTML = pathlib.Path(__file__).parent / "app.html"
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DEFAULT_PAGE_INDEX_DIR = REPO_ROOT / "cache/build_v3"
DEFAULT_CLEAN_JSON_DIR = REPO_ROOT / "cache/table_review_cleaned_json"
DEFAULT_RAW_PARSED_JSON_DIR = (
    REPO_ROOT.parent / "data/officeqa/treasury_bulletins_parsed/jsons"
)

PAGE_STORE_NAME_RE = re.compile(r"^\d{4}-\d{2}\.json$")
RAW_JSON_NAME_RE = re.compile(r"^treasury_bulletin_(\d{4})_(\d{2})\.json$")
BLOCK_RE = re.compile(r"\[(?P<type>[^\]]+)]\s*(?P<content>.*?)(?=\n\n\[[^\]]+]\s|\Z)", re.S)


def _json_path_for_bulletin(json_dir: pathlib.Path, bulletin: str) -> pathlib.Path | None:
    page_store_path = json_dir / f"{bulletin}.json"
    if page_store_path.exists():
        return page_store_path
    year, month = bulletin.split("-")
    raw_path = json_dir / f"treasury_bulletin_{year}_{month}.json"
    if raw_path.exists():
        return raw_path
    return None


def _copy_original_jsons(original_json_dir: pathlib.Path, clean_json_dir: pathlib.Path) -> int:
    clean_json_dir.mkdir(parents=True, exist_ok=True)
    if any(clean_json_dir.glob("*.json")):
        return 0
    copied = 0
    for path in sorted(original_json_dir.glob("*.json")):
        if PAGE_STORE_NAME_RE.match(path.name) or RAW_JSON_NAME_RE.match(path.name):
            shutil.copy2(path, clean_json_dir / path.name)
            copied += 1
    return copied


def _safe_json_load(path: pathlib.Path) -> object | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _fragments_from_page_store_text(text: str) -> list[dict]:
    fragments: list[dict] = []
    for match in BLOCK_RE.finditer(text):
        element_type = match.group("type").strip()
        if element_type == "page_number":
            continue
        content = match.group("content").strip()
        if not content:
            continue
        kind = "table" if element_type == "table" and "<table" in content.lower() else "text"
        fragments.append({"kind": kind, "type": element_type, "content": content})
    if not fragments and text.strip():
        fragments.append({"kind": "text", "type": "text", "content": text.strip()})
    return fragments


def _fragments_from_raw_doc(doc: dict, page: int) -> list[dict]:
    fragments: list[dict] = []
    elements = doc.get("document", {}).get("elements", [])
    for element in elements:
        bbox = element.get("bbox") or []
        if not bbox:
            continue
        page_id = bbox[0].get("page_id")
        if page_id is None or int(page_id) != page:
            continue
        element_type = str(element.get("type") or "text")
        if element_type == "page_number":
            continue
        content = element.get("content")
        if content is None or content == "":
            continue
        kind = "table" if element_type == "table" and "<table" in str(content).lower() else "text"
        fragment = {"kind": kind, "type": element_type, "content": str(content).strip()}
        coord = bbox[0].get("coord")
        if kind == "table" and coord and len(coord) == 4:
            fragment["bbox"] = [float(value) for value in coord]
        fragments.append(fragment)
    return fragments


def _page_fragments(json_dir: pathlib.Path, bulletin: str, page: int) -> dict:
    path = _json_path_for_bulletin(json_dir, bulletin)
    if path is None:
        return {
            "source": str(json_dir),
            "source_file": None,
            "fragments": [],
            "error": "JSON file not found",
        }

    data = _safe_json_load(path)
    if data is None:
        return {
            "source": str(json_dir),
            "source_file": str(path),
            "fragments": [],
            "error": "JSON file could not be read",
        }

    if isinstance(data, dict) and "document" in data:
        fragments = _fragments_from_raw_doc(data, page)
        mode = "raw_parsed_json"
    elif isinstance(data, dict):
        page_text = data.get(str(page), "")
        fragments = _fragments_from_page_store_text(str(page_text))
        mode = "page_store_json"
    else:
        fragments = []
        mode = "unknown"

    return {
        "source": str(json_dir),
        "source_file": str(path),
        "mode": mode,
        "fragments": fragments,
        "error": None,
    }


def _table_label(fragments: list[dict], table_position: int) -> str:
    table_seen = -1
    last_heading = ""
    fallback_text = ""
    for fragment in fragments:
        if fragment["kind"] == "text" and fragment["type"] in ("section_header", "page_header"):
            last_heading = fragment["content"][:120]
        if fragment["kind"] == "text" and fragment["type"] == "text" and not fallback_text:
            fallback_text = fragment["content"][:120]
        if fragment["kind"] != "table":
            continue
        table_seen += 1
        if table_seen == table_position:
            return last_heading or fallback_text or f"Table {table_position + 1}"
    return f"Table {table_position + 1}"


def _page_tables(json_dir: pathlib.Path, bulletin: str, page: int) -> dict:
    payload = _page_fragments(json_dir, bulletin, page)
    tables: list[dict] = []
    table_position = 0
    for fragment in payload.get("fragments", []):
        if fragment.get("kind") != "table":
            continue
        bbox = None
        if fragment.get("bbox"):
            bbox = fragment["bbox"]
        tables.append(
            {
                "index": table_position,
                "label": _table_label(payload.get("fragments", []), table_position),
                "content": fragment.get("content", ""),
                "bbox": bbox,
            }
        )
        table_position += 1
    return {
        "source": payload.get("source"),
        "source_file": payload.get("source_file"),
        "mode": payload.get("mode"),
        "tables": tables,
        "error": payload.get("error"),
    }


def _table_blocks(row: dict) -> list[dict]:
    return [
        block
        for block in row.get("content_blocks", [])
        if isinstance(block, dict) and block.get("kind") == "table"
    ]


def _item_summary(blocks: list[dict]) -> str:
    parts: list[str] = []
    for block in blocks[:2]:
        label = block.get("title") or "(untitled table)"
        details: list[str] = []
        if block.get("column_headers"):
            details.append("cols: " + ", ".join(block["column_headers"][:4]))
        if block.get("row_headers"):
            details.append("rows: " + ", ".join(block["row_headers"][:3]))
        if block.get("summary"):
            details.append(str(block["summary"]))
        parts.append(f"{label} - {'; '.join(details)}" if details else label)
    if len(blocks) > 2:
        parts.append(f"{len(blocks) - 2} more table(s)")
    return " | ".join(parts)


def _catalog_items(
    catalog_dir: pathlib.Path,
    renders_dir: pathlib.Path,
    include_missing_images: bool,
) -> list[dict]:
    items: list[dict] = []
    if not catalog_dir.exists():
        return items

    for path in sorted(catalog_dir.glob("*.jsonl")):
        bulletin = path.stem
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            table_blocks = _table_blocks(row)
            if not table_blocks:
                continue
            page = int(row["page"])
            image_path = renders_dir / bulletin / f"{page}.png"
            image_available = image_path.exists()
            if not include_missing_images and not image_available:
                continue
            items.append(
                {
                    "id": f"{bulletin}:{page}",
                    "bulletin": bulletin,
                    "page": page,
                    "image_available": image_available,
                    "table_count": len(table_blocks),
                    "summary": _item_summary(table_blocks),
                    "date_interval": row.get("date_interval"),
                }
            )
    return items


def _render_image_if_needed(
    renders_dir: pathlib.Path,
    pdf_dir: pathlib.Path | None,
    bulletin: str,
    page: int,
) -> pathlib.Path | None:
    image_path = renders_dir / bulletin / f"{page}.png"
    if image_path.exists():
        return image_path
    if pdf_dir is None:
        return None

    year, month = bulletin.split("-")
    pdf_path = pdf_dir / f"treasury_bulletin_{year}_{month}.pdf"
    if not pdf_path.exists():
        return None

    try:
        import fitz

        with fitz.open(pdf_path) as doc:
            if page < 1 or page > len(doc):
                return None
            pix = doc[page - 1].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            pix.save(image_path)
    except Exception:
        return None
    return image_path if image_path.exists() else None


class _Handler(BaseHTTPRequestHandler):
    page_index_dir: pathlib.Path
    original_json_dir: pathlib.Path
    clean_json_dir: pathlib.Path
    pdf_dir: pathlib.Path | None
    items_cache: dict[bool, list[dict]] = {}

    def log_message(self, fmt, *args):  # noqa: ARG002
        pass

    def _send_json(self, data: object, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, content: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_png(self, path: pathlib.Path) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            return self._send_html(APP_HTML.read_bytes())

        if path == "/api/items":
            include_missing = qs.get("include_missing_images", ["0"])[0] == "1"
            items = self.items_cache.get(include_missing)
            if items is None:
                items = _catalog_items(
                    self.page_index_dir / "catalog",
                    self.page_index_dir / "renders",
                    include_missing,
                )
                self.items_cache[include_missing] = items
            return self._send_json(
                {
                    "items": items,
                    "paths": {
                        "original_json_dir": str(self.original_json_dir),
                        "clean_json_dir": str(self.clean_json_dir),
                        "page_index_dir": str(self.page_index_dir),
                    },
                }
            )

        if path == "/api/page":
            bulletin = qs.get("bulletin", [""])[0]
            page_str = qs.get("page", [""])[0]
            if not re.fullmatch(r"\d{4}-\d{2}", bulletin):
                return self._send_json({"error": "invalid bulletin"}, 400)
            try:
                page = int(page_str)
            except ValueError:
                return self._send_json({"error": "invalid page"}, 400)

            original = _page_fragments(self.original_json_dir, bulletin, page)
            cleaned = _page_fragments(self.clean_json_dir, bulletin, page)
            original_tables = _page_tables(self.original_json_dir, bulletin, page)
            cleaned_tables = _page_tables(self.clean_json_dir, bulletin, page)
            table_count = max(
                len(original_tables.get("tables", [])),
                len(cleaned_tables.get("tables", [])),
            )
            tables = []
            for table_index in range(table_count):
                original_table = next(
                    (
                        table
                        for table in original_tables.get("tables", [])
                        if table.get("index") == table_index
                    ),
                    None,
                )
                cleaned_table = next(
                    (
                        table
                        for table in cleaned_tables.get("tables", [])
                        if table.get("index") == table_index
                    ),
                    None,
                )
                label = (
                    (original_table or {}).get("label")
                    or (cleaned_table or {}).get("label")
                    or f"Table {table_index + 1}"
                )
                tables.append(
                    {
                        "index": table_index,
                        "label": label,
                        "bbox": (original_table or {}).get("bbox"),
                        "original": original_table,
                        "cleaned": cleaned_table,
                    }
                )
            image_path = _render_image_if_needed(
                self.page_index_dir / "renders",
                self.pdf_dir,
                bulletin,
                page,
            )
            return self._send_json(
                {
                    "bulletin": bulletin,
                    "page": page,
                    "image_url": f"/api/image?bulletin={html.escape(bulletin)}&page={page}"
                    if image_path
                    else None,
                    "original": original,
                    "cleaned": cleaned,
                    "tables": tables,
                }
            )

        if path == "/api/image":
            bulletin = qs.get("bulletin", [""])[0]
            page_str = qs.get("page", [""])[0]
            if not re.fullmatch(r"\d{4}-\d{2}", bulletin):
                self.send_response(400)
                self.end_headers()
                return
            try:
                page = int(page_str)
            except ValueError:
                self.send_response(400)
                self.end_headers()
                return
            image_path = _render_image_if_needed(
                self.page_index_dir / "renders",
                self.pdf_dir,
                bulletin,
                page,
            )
            if image_path is None:
                self.send_response(404)
                self.end_headers()
                return
            return self._send_png(image_path)

        self.send_response(404)
        self.end_headers()

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path != "/api/image":
            self.send_response(404)
            self.end_headers()
            return

        bulletin = qs.get("bulletin", [""])[0]
        page_str = qs.get("page", [""])[0]
        if not re.fullmatch(r"\d{4}-\d{2}", bulletin):
            self.send_response(400)
            self.end_headers()
            return
        try:
            page = int(page_str)
        except ValueError:
            self.send_response(400)
            self.end_headers()
            return

        image_path = _render_image_if_needed(
            self.page_index_dir / "renders",
            self.pdf_dir,
            bulletin,
            page,
        )
        if image_path is None:
            self.send_response(404)
            self.end_headers()
            return

        try:
            size = image_path.stat().st_size
        except OSError:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(size))
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local table extraction review UI")
    parser.add_argument("--port", type=int, default=7080)
    parser.add_argument(
        "--page-index-dir",
        default=os.environ.get("SKUNK_PAGE_INDEX_DIR") or str(DEFAULT_PAGE_INDEX_DIR),
    )
    parser.add_argument(
        "--original-json-dir",
        default=os.environ.get("TABLE_REVIEW_ORIGINAL_JSON_DIR")
        or os.environ.get("OFFICEQA_PARSED_JSON_DIR")
        or str(DEFAULT_RAW_PARSED_JSON_DIR if DEFAULT_RAW_PARSED_JSON_DIR.exists() else DEFAULT_PAGE_INDEX_DIR / "pages"),
    )
    parser.add_argument(
        "--clean-json-dir",
        default=os.environ.get("TABLE_REVIEW_CLEAN_JSON_DIR") or str(DEFAULT_CLEAN_JSON_DIR),
    )
    parser.add_argument("--pdf-dir", default=os.environ.get("OFFICEQA_PDF_DIR"))
    args = parser.parse_args()

    _Handler.page_index_dir = pathlib.Path(args.page_index_dir).resolve()
    _Handler.original_json_dir = pathlib.Path(args.original_json_dir).resolve()
    _Handler.clean_json_dir = pathlib.Path(args.clean_json_dir).resolve()
    _Handler.pdf_dir = pathlib.Path(args.pdf_dir).resolve() if args.pdf_dir else None
    _Handler.items_cache = {}

    copied = _copy_original_jsons(_Handler.original_json_dir, _Handler.clean_json_dir)

    server = ThreadingHTTPServer(("localhost", args.port), _Handler)
    url = f"http://localhost:{args.port}"
    print(f"Table Review UI -> {url}")
    print(f"Page index      -> {_Handler.page_index_dir}")
    print(f"Original JSON   -> {_Handler.original_json_dir}")
    print(f"Cleaned JSON    -> {_Handler.clean_json_dir}")
    if copied:
        print(f"Copied {copied} original JSON file(s) into the cleaned folder.")
    print("Press Ctrl-C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
