#!/usr/bin/env python3
"""
E2E Trace Viewer — local HTTP server for skunk-format trace runs (qatfd
`results/` reports and grc-officeqa `eval/traces/` runs).

Each run directory contains `report.csv` (one row per question) and per-question
event streams at `traces/<uid>.jsonl` (one JSON object per line, tagged with
`step_idx` / `op` / `kind` / optional `data`; the uid is the filename).
This server discovers those run dirs and serves, per question, the ordered
event list so the frontend (`app.html`) can render the Orchestrator's plan and
each operator node's color-coded agent trace.

Run from the skunk/ directory:

    python3 scripts/trace_viewer/serve.py

Or point it at a specific traces root / port:

    python3 scripts/trace_viewer/serve.py --traces-root ../grc-officeqa/eval/traces --port 7071
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

# A report.csv field (e.g. a verbose judge_rationale, or a long retrieved_docs JSON) can exceed
# Python's default 131072-char CSV field limit and abort the whole read; raise it so one big field
# never blocks a run from loading.
csv.field_size_limit(10**9)

_APP_HTML = pathlib.Path(__file__).parent / "app.html"


# ── Discovery + parsing helpers ─────────────────────────────────────────────

def _find_runs(root: pathlib.Path) -> list[str]:
    """Run directories under `root` that hold per-question traces or a report.csv,
    returned as paths relative to `root`, newest first (dir names are
    timestamp-suffixed, so reverse-sorted == newest).

    Discovers both layouts: the flat `eval/traces/<run>/` and the qatfd
    `results/<benchmark>/<system>/<run>/` hierarchy. A directory qualifies as a run when
    it directly contains `traces/*.jsonl` or `report.csv`; we never descend into a
    run (so a run's own `traces/` subdir is not mistaken for another run)."""
    runs: list[str] = []
    seen: set[pathlib.Path] = set()

    def _walk(d: pathlib.Path) -> None:
        if not d.is_dir():
            return
        if any((d / "traces").glob("*.jsonl")) or (d / "report.csv").exists():
            rel = d.relative_to(root)
            if rel not in seen:
                seen.add(rel)
                runs.append(rel.as_posix())
            return  # a run is a leaf — don't descend further
        for child in sorted(d.iterdir()):
            if child.is_dir():
                _walk(child)

    try:
        _walk(root)
    except FileNotFoundError:
        pass
    return sorted(runs, reverse=True)


def _read_report(run_dir: pathlib.Path) -> list[dict]:
    """Question rows from `report.csv` (uid, question, predicted, gold_answer,
    failed, reason). Empty list when the file is absent.

    Normalizes the qatfd runner's column names onto the viewer's canonical ones:
    `qid`->`uid`, `gold`->`gold_answer`, and the legacy `correct` column ->`score`
    (skunk's eval_e2e.py still writes `correct`; the qatfd runner writes `score`).
    eval_e2e.py rows otherwise pass through unchanged."""
    path = run_dir / "report.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if not r.get("uid") and r.get("qid"):
            r["uid"] = r["qid"]
        if not r.get("gold_answer") and r.get("gold"):
            r["gold_answer"] = r["gold"]
        if r.get("score") in (None, "") and r.get("correct") not in (None, ""):
            r["score"] = r["correct"]
    return rows


def _load_events(path: pathlib.Path) -> list[dict]:
    """Parsed event dicts from one per-question .jsonl file (skipping bad lines)."""
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _trace_uids(run_dir: pathlib.Path) -> list[str]:
    """Uids with a per-question trace file, sorted lexicographically."""
    return sorted(p.stem for p in (run_dir / "traces").glob("*.jsonl"))


def _run_questions(run_dir: pathlib.Path) -> list[dict]:
    """The question list for a run: report.csv rows when present, else synthesized
    from the per-question trace files."""
    rows = _read_report(run_dir)
    if rows:
        return rows
    return [{"uid": uid, "question": "", "predicted": "", "gold_answer": "",
             "failed": "", "reason": ""} for uid in _trace_uids(run_dir)]


def _question_events(run_dir: pathlib.Path, uid: str) -> list[dict]:
    """All events for one uid, in file (== chronological per-question) order. Every
    field — message / step_idx / op / level / kind / data — is preserved for the
    frontend to structure. The uid names a file under the run's traces/, so reject
    anything that could escape it."""
    if not uid or "/" in uid or "\\" in uid or uid in (".", ".."):
        return []
    return _load_events(run_dir / "traces" / f"{uid}.jsonl")


# ── HTTP handler ────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    root: pathlib.Path

    def log_message(self, *_args) -> None:  # noqa: N802 — silence per-request stderr noise
        pass

    def _safe_run_dir(self, run: str) -> pathlib.Path | None:
        """Resolve a run name to a directory under `root`, rejecting traversal."""
        if not run:
            return None
        try:
            run_dir = (self.root / run).resolve()
            run_dir.relative_to(self.root.resolve())
        except ValueError:
            return None
        return run_dir if run_dir.is_dir() else None

    def _send_json(self, data: object, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, content: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        # app.html is edited live and re-read on each request; forbid browser caching so a
        # reload always picks up the latest markup/JS (else a stale cached copy hides fixes).
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            return self._send_html(_APP_HTML.read_bytes())

        if path == "/api/runs":
            return self._send_json({"runs": _find_runs(self.root)})

        if path == "/api/run":
            run_dir = self._safe_run_dir(qs.get("run", [""])[0])
            if run_dir is None:
                return self._send_json({"error": "unknown run"}, 404)
            return self._send_json({"questions": _run_questions(run_dir)})

        if path == "/api/question":
            run_dir = self._safe_run_dir(qs.get("run", [""])[0])
            uid = qs.get("uid", [""])[0]
            if run_dir is None or not uid:
                return self._send_json({"error": "missing run or uid"}, 400)
            return self._send_json({"uid": uid, "events": _question_events(run_dir, uid)})

        self.send_response(404)
        self.end_headers()


# ── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="E2E Trace Viewer")
    parser.add_argument("--port", type=int, default=7071, help="Port to listen on (default: 7071)")
    parser.add_argument(
        "--traces-root",
        default=None,
        help=(
            "Directory holding run dirs (discovered recursively, so it may be a flat "
            "grc-officeqa eval/traces or the nested qatfd results/<benchmark>/<system>/<run>). "
            "Defaults to <repo>/qatfd/results when present, else grc-officeqa/eval/traces."
        ),
    )
    args = parser.parse_args()

    if args.traces_root:
        root = pathlib.Path(args.traces_root).resolve()
    else:
        here = pathlib.Path(__file__).resolve()
        # Script lives at <repo>/skunk/scripts/trace_viewer/serve.py.
        qatfd_results = here.parents[3] / "qatfd" / "results"
        eval_traces = here.parents[3] / "grc-officeqa" / "eval" / "traces"
        root = (qatfd_results if qatfd_results.is_dir() else eval_traces).resolve()

    _Handler.root = root
    server = HTTPServer(("localhost", args.port), _Handler)
    url = f"http://localhost:{args.port}"
    print(f"E2E Trace Viewer → {url}")
    print(f"Traces root      → {root}")
    print("Press Ctrl-C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
