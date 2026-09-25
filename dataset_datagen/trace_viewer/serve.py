#!/usr/bin/env python3
"""
Datagen Trajectory Viewer — local HTTP server for `follow_up_questions.py` run directories.

Visualises, per seed question, the full generation trajectory:

    (1) seed question
     -> (2) line of inquiry rollout  -> (3) unique determination   (loop back to 2 on duplicate)
     -> (4) follow-up rollout (docs sampled + question generated)
     -> (5) solvable determination   -> (6) qa-pair unique determination
                                        (loop back to 4 on unsolvable / duplicate)

Run from the repo root (defaults to `dataset_datagen/output` as the runs root):

    python3 dataset_datagen/trace_viewer/serve.py
    python3 dataset_datagen/trace_viewer/serve.py --root /path/to/output --port 7072

Endpoints (all JSON except `/`):
    /api/runs                         run directories under the root, newest first
    /api/run?run=<rel>                per-question overview + funnel counts (stats only, fast)
    /api/run_verdicts?run=<rel>       judge-verdict counts per question (parses every trace; cached)
    /api/question?run=<rel>&qid=<id>  the reconstructed trajectory (see trajectory.py)
    /api/doc?run=<rel>&doc_id=<id>    a corpus document's text (officeqa: cleaned bulletin page) and,
                                      when the source PDF is on disk, which PDF page it came from
    /api/doc_page.png?run=&doc_id=&dpi=  that PDF page rendered to PNG (officeqa only; pdftoppm/PyMuPDF)
    /api/doc_pdf?run=<rel>&doc_id=<id>   the whole source PDF (open in a new tab with #page=N)
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import trajectory  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
_APP_HTML = _HERE / "app.html"
_REPO_ROOT = _HERE.parents[1]
_OFFICEQA_CLEAN_DIR = _REPO_ROOT / "qatfd/benchmarks/officeqa/treasury_bulletins_cleaned"
_OFFICEQA_PDF_DIR = _REPO_ROOT / "qatfd/benchmarks/officeqa/treasury_bulletin_pdfs"
# officeqa doc ids are `<year>_<month>_<page>` with a 1-based page number into treasury_bulletin_<year>_<month>.pdf
_OFFICEQA_DOC_ID_RE = re.compile(r"^(\d{4}_\d{2})_(\d+)$")


# ── discovery ────────────────────────────────────────────────────────────────


def _is_run_dir(d: pathlib.Path) -> bool:
    return (
        (d / "generation_stats.jsonl").exists() or (d / "records.jsonl").exists() or any((d / "traces").glob("*.jsonl"))
    )


def find_runs(root: pathlib.Path) -> list[dict]:
    """Run dirs under `root` (recursive, a run is a leaf), newest first by mtime."""
    runs: list[dict] = []

    def _walk(d: pathlib.Path, depth: int) -> None:
        if not d.is_dir() or depth > 4:
            return
        if _is_run_dir(d):
            cfg = trajectory.load_json(d / "run_config.json") or {}
            records = trajectory.load_records(d)
            runs.append(
                {
                    "run": d.relative_to(root).as_posix(),
                    "mtime": max([d.stat().st_mtime] + [p.stat().st_mtime for p in d.glob("*.jsonl")]),
                    "n_records": len(records),
                    "n_failed": sum(1 for r in records.values() if r.get("failed")),
                    "n_traces": len(trajectory.trace_qids(d)),
                    "benchmark": cfg.get("benchmark"),
                    "split": cfg.get("split"),
                    "model_id": cfg.get("model_id"),
                    "k": cfg.get("k"),
                }
            )
            return
        for child in sorted(d.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                _walk(child, depth + 1)

    with contextlib.suppress(FileNotFoundError):
        _walk(root, 0)
    return sorted(runs, key=lambda r: -r["mtime"])


# ── corpus documents ─────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def _officeqa_page_map() -> dict[str, str]:
    """doc_id -> absolute path of its cleaned page text (empty when the corpus isn't on disk)."""
    m = trajectory.load_json(_OFFICEQA_CLEAN_DIR / "clean_page_map.json")
    if not isinstance(m, dict):
        return {}
    return {doc_id: str(_OFFICEQA_CLEAN_DIR / pathlib.Path(entry[0]).name) for doc_id, entry in m.items() if entry}


def load_doc(benchmark: str | None, doc_id: str) -> str | None:
    if benchmark in (None, "officeqa"):
        path = _officeqa_page_map().get(doc_id)
        if path:
            try:
                return pathlib.Path(path).read_text(encoding="utf-8")
            except OSError:
                return None
    return None


def doc_pdf_page(benchmark: str | None, doc_id: str) -> tuple[pathlib.Path, int] | None:
    """(pdf path, 1-based page number) the document was extracted from, or None when unknown / not on disk."""
    if benchmark not in (None, "officeqa"):
        return None
    m = _OFFICEQA_DOC_ID_RE.match(doc_id)
    if not m:
        return None
    pdf = _OFFICEQA_PDF_DIR / f"treasury_bulletin_{m.group(1)}.pdf"
    return (pdf, int(m.group(2))) if pdf.is_file() else None


_PNG_CACHE: collections.OrderedDict[tuple[str, int, int], bytes] = collections.OrderedDict()
_PNG_CACHE_MAX = 48
_PNG_LOCK = threading.Lock()


def _render_pdftoppm(pdf: pathlib.Path, page: int, dpi: int) -> bytes:
    with tempfile.TemporaryDirectory(prefix="trace_viewer_") as tmp:
        out = pathlib.Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(dpi), "-png", "-singlefile", str(pdf), str(out)],
            check=True,
            capture_output=True,
            timeout=60,
        )
        return out.with_suffix(".png").read_bytes()


def _render_pymupdf(pdf: pathlib.Path, page: int, dpi: int) -> bytes:
    import fitz  # PyMuPDF (optional; only used when poppler's pdftoppm is missing)

    with fitz.open(str(pdf)) as doc:
        if not 1 <= page <= doc.page_count:
            raise IndexError(f"page {page} out of range (pdf has {doc.page_count} pages)")
        return doc[page - 1].get_pixmap(dpi=dpi).tobytes("png")


def render_pdf_page_png(pdf: pathlib.Path, page: int, dpi: int) -> bytes:
    """Render one PDF page to PNG bytes (LRU-cached). Uses pdftoppm (poppler) or PyMuPDF, whichever is available."""
    key = (str(pdf), page, dpi)
    with _PNG_LOCK:
        if key in _PNG_CACHE:
            _PNG_CACHE.move_to_end(key)
            return _PNG_CACHE[key]
    if shutil.which("pdftoppm"):
        png = _render_pdftoppm(pdf, page, dpi)
    else:
        try:
            png = _render_pymupdf(pdf, page, dpi)
        except ImportError as e:
            raise RuntimeError(
                "no PDF renderer available: install poppler-utils (pdftoppm) or `pip install pymupdf`"
            ) from e
    with _PNG_LOCK:
        _PNG_CACHE[key] = png
        while len(_PNG_CACHE) > _PNG_CACHE_MAX:
            _PNG_CACHE.popitem(last=False)
    return png


# ── HTTP handler ─────────────────────────────────────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    root: pathlib.Path

    def log_message(self, *_args) -> None:  # silence per-request stderr noise
        pass

    def _safe_run_dir(self, run: str) -> pathlib.Path | None:
        if not run:
            return None
        try:
            run_dir = (self.root / run).resolve()
            run_dir.relative_to(self.root.resolve())
        except ValueError:
            return None
        return run_dir if run_dir.is_dir() else None

    def _send(
        self, body: bytes, ctype: str, status: int = 200, cache: bool = False, filename: str | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=86400" if cache else "no-store, must-revalidate")
        if filename:
            self.send_header("Content-Disposition", f'inline; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: object, status: int = 200) -> None:
        self._send(json.dumps(data, default=str).encode("utf-8"), "application/json", status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        path = parsed.path
        q = lambda k: qs.get(k, [""])[0]  # noqa: E731

        try:
            if path in ("/", "/index.html"):
                return self._send(_APP_HTML.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/runs":
                return self._json({"runs": find_runs(self.root)})

            run_dir = self._safe_run_dir(q("run"))
            if path.startswith("/api/") and run_dir is None:
                return self._json({"error": "unknown run"}, 404)

            if path == "/api/run":
                return self._json(trajectory.run_overview(run_dir))
            if path == "/api/run_verdicts":
                return self._json({"verdicts": trajectory.run_verdicts(run_dir)})
            if path == "/api/question":
                qid = q("qid")
                if not qid or "/" in qid or "\\" in qid or qid in (".", ".."):
                    return self._json({"error": "bad qid"}, 400)
                return self._json(trajectory.build_trajectory(run_dir, qid))
            if path in ("/api/doc", "/api/doc_page.png", "/api/doc_pdf"):
                doc_id = q("doc_id")
                cfg = trajectory.load_json(run_dir / "run_config.json") or {}
                benchmark = cfg.get("benchmark")
                pdf_page = doc_pdf_page(benchmark, doc_id) if doc_id else None
                if path == "/api/doc":
                    text = load_doc(benchmark, doc_id) if doc_id else None
                    if text is None and pdf_page is None:
                        return self._json({"error": f"document {doc_id!r} not found on disk"}, 404)
                    pdf = {"file": pdf_page[0].name, "page": pdf_page[1]} if pdf_page else None
                    return self._json({"doc_id": doc_id, "text": text, "pdf": pdf})
                if pdf_page is None:
                    return self._json({"error": f"no source PDF on disk for document {doc_id!r}"}, 404)
                pdf, page = pdf_page
                if path == "/api/doc_pdf":
                    return self._send(pdf.read_bytes(), "application/pdf", cache=True, filename=pdf.name)
                dpi = max(72, min(int(q("dpi") or 170), 450))
                try:
                    png = render_pdf_page_png(pdf, page, dpi)
                except (
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                    IndexError,
                    OSError,
                    RuntimeError,
                ) as e:
                    detail = e.stderr.decode(errors="replace").strip() if getattr(e, "stderr", None) else str(e)
                    return self._json({"error": f"could not render {pdf.name} page {page}: {detail}"}, 500)
                return self._send(png, "image/png", cache=True)
        except Exception as e:  # surface parser bugs to the UI instead of a silent 500
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        self.send_response(404)
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Datagen Trajectory Viewer")
    parser.add_argument("--port", type=int, default=7072, help="port to listen on (default: 7072)")
    parser.add_argument("--host", default="localhost", help="interface to bind (default: localhost)")
    parser.add_argument(
        "--root", default=None, help="directory holding run dirs (default: <repo>/dataset_datagen/output)"
    )
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve() if args.root else (_HERE.parent / "output").resolve()
    _Handler.root = root
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"Datagen Trajectory Viewer → http://{args.host}:{args.port}")
    print(f"Runs root                 → {root}")
    print("Press Ctrl-C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
