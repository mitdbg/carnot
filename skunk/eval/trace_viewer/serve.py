#!/usr/bin/env python3
"""
E2E Trace Viewer — local HTTP server for `eval/eval_e2e.py` runs.

Each eval run writes a directory `eval/traces/<run>/` containing `report.csv`
(one row per question) and `traces/events.jsonl` (the run-wide structured event
stream — one JSON object per line, tagged with `uid` / `step_idx` / `op` / `kind`
/ optional `data`). This server discovers those run dirs and serves, per question,
the ordered event list so the frontend (`app.html`) can render the Orchestrator's
plan and each operator node's color-coded agent trace.

Run from the skunk/ directory:

    python3 eval/trace_viewer/serve.py

Or point it at a specific traces root / port:

    python3 eval/trace_viewer/serve.py --traces-root eval/traces --port 7071
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

_APP_HTML = pathlib.Path(__file__).parent / "app.html"


# ── Discovery + parsing helpers ─────────────────────────────────────────────

def _events_path(run_dir: pathlib.Path) -> pathlib.Path:
    return run_dir / "traces" / "events.jsonl"


def _find_runs(root: pathlib.Path) -> list[str]:
    """Run directories under `root` that hold an events.jsonl or report.csv, returned
    as paths relative to `root`, newest first (dir names are timestamp-suffixed, so
    reverse-sorted == newest).

    Discovers both layouts: the flat `eval/traces/<run>/` and the qatfd
    `results/<benchmark>/<system>/<run>/` hierarchy. A directory qualifies as a run when
    it directly contains `traces/events.jsonl` or `report.csv`; we never descend into a
    run (so a run's own `traces/` subdir is not mistaken for another run)."""
    runs: list[str] = []
    seen: set[pathlib.Path] = set()

    def _walk(d: pathlib.Path) -> None:
        if not d.is_dir():
            return
        if _events_path(d).exists() or (d / "report.csv").exists():
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


def _iter_events(run_dir: pathlib.Path):
    """Yield parsed event dicts from the run's events.jsonl (skipping bad lines)."""
    path = _events_path(run_dir)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _uids_in_events(run_dir: pathlib.Path) -> list[str]:
    """Distinct uids present in the event stream, in first-seen order."""
    seen: dict[str, None] = {}
    for evt in _iter_events(run_dir):
        uid = evt.get("uid")
        if uid is not None and uid not in seen:
            seen[uid] = None
    return list(seen)


def _run_questions(run_dir: pathlib.Path) -> list[dict]:
    """The question list for a run: report.csv rows when present, else synthesized
    from the uids found in the event stream."""
    rows = _read_report(run_dir)
    if rows:
        return rows
    return [{"uid": uid, "question": "", "predicted": "", "gold_answer": "",
             "failed": "", "reason": ""} for uid in _uids_in_events(run_dir)]


def _question_events(run_dir: pathlib.Path, uid: str) -> list[dict]:
    """All events for one uid, in file (== chronological per-question) order. The
    `uid` key is dropped (implied) but every other field — message / step_idx / op /
    level / kind / data — is preserved for the frontend to structure."""
    out: list[dict] = []
    for evt in _iter_events(run_dir):
        if evt.get("uid") != uid:
            continue
        out.append({k: v for k, v in evt.items() if k != "uid"})
    return out


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
            "eval/traces or the nested qatfd results/<benchmark>/<system>/<run>). "
            "Defaults to <repo>/qatfd/results when present, else eval/traces."
        ),
    )
    args = parser.parse_args()

    if args.traces_root:
        root = pathlib.Path(args.traces_root).resolve()
    else:
        here = pathlib.Path(__file__).resolve()
        # Script lives at <repo>/skunk/eval/trace_viewer/serve.py.
        qatfd_results = here.parents[3] / "qatfd" / "results"
        eval_traces = here.parent.parent / "traces"
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
