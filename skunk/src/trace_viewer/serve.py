#!/usr/bin/env python3
"""
QA Synthesis Trace Viewer — local HTTP server.

Run from the skunk/ directory (default traces root):

    python src/trace_viewer/serve.py

Or pass an explicit traces root:

    python src/trace_viewer/serve.py --traces-root /path/to/skunk --port 7070
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

_APP_HTML = pathlib.Path(__file__).parent / "app.html"


# ── Discovery helpers ──────────────────────────────────────────────────────

def _find_trace_dirs(root: pathlib.Path) -> list[str]:
    """Return relative paths like 'old_qa_synthesis_traces/officeqa'.

    Matches any directory whose name ends with ``_qa_synthesis_traces`` OR is
    exactly ``qa_synthesis_traces`` (the un-prefixed default output dir).
    """
    dirs: list[str] = []
    for parent in sorted(root.iterdir()):
        if not parent.is_dir():
            continue
        n = parent.name
        if n != "qa_synthesis_traces" and not n.endswith("_qa_synthesis_traces"):
            continue
        for bench in sorted(parent.iterdir()):
            if bench.is_dir():
                dirs.append(f"{parent.name}/{bench.name}")
    return dirs


def _list_seeds(root: pathlib.Path, trace_dir: str) -> list[int]:
    """Return sorted seed integers present in the directory."""
    dir_path = root / trace_dir
    seeds: set[int] = set()
    try:
        for f in dir_path.iterdir():
            if f.is_file() and f.name.endswith("_qa_pairs.json"):
                stem = f.name[: -len("_qa_pairs.json")]
                if stem.lstrip("-").isdigit():
                    seeds.add(int(stem))
    except FileNotFoundError:
        pass
    return sorted(seeds)


def _load_json(path: pathlib.Path) -> object | None:
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            return json.load(fh)
    except (json.JSONDecodeError, ValueError, OSError):
        return None


def _seed_data(root: pathlib.Path, trace_dir: str, seed: int) -> dict:
    d = root / trace_dir
    p = str(seed)
    # Prefer the phase-2 filtered rollouts file: it carries the authoritative
    # ``binarized_score`` / ``kept`` / ``discard_reason`` produced by the global
    # mean-threshold pass. ``{seed}_rollouts.json`` is the phase-1 file where
    # those fields are still null, so fall back to it only when no filtered
    # file exists yet (pipeline still in flight).
    filtered = _load_json(d / f"{p}_rollouts_filtered.json")
    rollouts = filtered.get("rollouts") if isinstance(filtered, dict) else None
    if rollouts is None:
        rollouts = _load_json(d / f"{p}_rollouts.json")
    return {
        "seed": seed,
        "qa_pairs":  _load_json(d / f"{p}_qa_pairs.json") or [],
        "dedup":     _load_json(d / f"{p}_qa_pairs_dedup.json"),
        "rollouts":  rollouts,
        "rollout_stats": filtered.get("global_stats") if isinstance(filtered, dict) else None,
        "quality":   _load_json(d / f"{p}_qa_pairs_quality.json"),
    }


def _funnel_data(root: pathlib.Path, trace_dir: str) -> dict:
    """Aggregate funnel counts (synthesized → dedup → rollout → quality) across all seeds."""
    dir_path = root / trace_dir
    seeds: list[int] = []
    try:
        seeds = sorted(
            int(f.stem.split("_")[0])
            for f in dir_path.glob("*_qa_pairs.json")
            if f.stem.split("_")[0].lstrip("-").isdigit()
        )
    except OSError:
        pass

    total_synth = 0
    total_dedup = 0
    total_rollout = 0
    total_quality = 0
    has_dedup = has_rollout = has_quality = False
    seed_rows: list[dict] = []

    dedup_breakdown: dict[str, int] = {
        k: 0 for k in ("not_run", "exact_q_match", "exact_a_match",
                        "intra_batch", "similar_q", "similar_a", "exec_error")
    }
    rollout_breakdown: dict[str, int] = {
        k: 0 for k in ("skipped", "all_pass", "all_fail")
    }
    quality_breakdown: dict[str, int] = {
        k: 0 for k in ("qf_error", "qf_rejected", "qf_missing")
    }
    total_rollout_passed = 0

    for seed in seeds:
        qa_pairs = _load_json(dir_path / f"{seed}_qa_pairs.json")
        n_synth = len(qa_pairs) if isinstance(qa_pairs, list) else 0
        total_synth += n_synth

        dedup = _load_json(dir_path / f"{seed}_qa_pairs_dedup.json")
        n_dedup: int | None = None
        if isinstance(dedup, dict):
            n_dedup = dedup.get("n_kept")
            if n_dedup is not None:
                has_dedup = True
                total_dedup += n_dedup

        n_rollout: int | None = None
        filtered = _load_json(dir_path / f"{seed}_rollouts_filtered.json")
        if isinstance(filtered, dict):
            pairs = filtered.get("pairs", [])
            n_rollout = sum(1 for p in pairs if p.get("kept") is True)
            has_rollout = True
            total_rollout += n_rollout
        else:
            rollouts = _load_json(dir_path / f"{seed}_rollouts.json")
            if isinstance(rollouts, list):
                kept_qa = {r["qa_id"] for r in rollouts if r.get("kept") is True}
                n_rollout = len(kept_qa)
                has_rollout = True
                total_rollout += n_rollout

        n_quality: int | None = None
        quality = _load_json(dir_path / f"{seed}_qa_pairs_quality.json")
        if quality is not None:
            results = quality.get("results", []) if isinstance(quality, dict) else quality
            if isinstance(results, list):
                n_quality = sum(1 for r in results if r.get("valid") is True)
                has_quality = True
                total_quality += n_quality

        # ── dedup drop breakdown ──────────────────────────────────────────
        if not isinstance(dedup, dict):
            # Synthesis completed but dedup was never run (early stop)
            dedup_breakdown["not_run"] += n_synth
        else:
            for entry in dedup.get("funnel", []):
                if not entry.get("filtered"):
                    continue
                if entry.get("exact_question_match_id"):
                    dedup_breakdown["exact_q_match"] += 1
                elif entry.get("exact_answer_match_id"):
                    dedup_breakdown["exact_a_match"] += 1
                elif entry.get("intra_batch_duplicate_of"):
                    dedup_breakdown["intra_batch"] += 1
                elif entry.get("question_neighbor_duplicate_of"):
                    dedup_breakdown["similar_q"] += 1
                elif entry.get("answer_neighbor_duplicate_of"):
                    dedup_breakdown["similar_a"] += 1
                else:
                    dedup_breakdown["exec_error"] += 1

        # ── rollout drop breakdown ────────────────────────────────────────
        # "rollout drops"  = eliminated before quality filter (all_fail/all_pass/skipped)
        # "quality drops"  = passed rollout threshold but rejected by quality filter
        filtered_pair_ids: set[str] = set()
        n_qf_drops_seed = 0
        if isinstance(filtered, dict):
            filtered_pair_ids = {p["qa_id"] for p in filtered.get("pairs", [])}
            for pair_rec in filtered.get("pairs", []):
                if pair_rec.get("kept"):
                    continue
                reason = pair_rec.get("discard_reason") or ""
                if reason in ("all_pass", "all_fail"):
                    rollout_breakdown[reason] += 1
                elif reason in ("qf_error", "qf_rejected", "qf_missing"):
                    quality_breakdown[reason] += 1
                    n_qf_drops_seed += 1
                else:
                    rollout_breakdown["skipped"] += 1
        # Deduped pairs with no filtered rollout output were skipped (early stop
        # or incomplete pipeline).
        if isinstance(dedup, dict):
            deduped_ids: set[str] = set(dedup.get("kept_qa_ids", []))
            rollout_breakdown["skipped"] += len(deduped_ids - filtered_pair_ids)
        # total_rollout_passed = kept (survived quality filter) + quality-filter drops
        if has_rollout:
            total_rollout_passed += (n_rollout or 0) + n_qf_drops_seed

        seed_rows.append({
            "seed": seed,
            "n_synthesized": n_synth,
            "n_dedup": n_dedup,
            "n_rollout_kept": n_rollout,
            "n_quality": n_quality,
        })

    return {
        "total_synthesized":    total_synth,
        "total_dedup":          total_dedup    if has_dedup    else None,
        "total_rollout_passed":  total_rollout_passed if has_rollout else None,
        "total_rollout_kept":    total_rollout  if has_rollout  else None,
        "total_quality":         total_quality  if has_quality  else None,
        "dedup_drop_breakdown":   dedup_breakdown  if has_dedup   else None,
        "rollout_drop_breakdown": rollout_breakdown if has_rollout else None,
        "quality_drop_breakdown": quality_breakdown if has_rollout else None,
        "seeds": seed_rows,
    }


def _qa_text_lookup(root: pathlib.Path, trace_dir: str, ids: list[str]) -> dict:
    """Return {qa_id: {question, answer}} for the requested IDs.

    Synthesized IDs (``N-M`` format) are found in ``{trace_dir}/*_qa_pairs.json``.
    Benchmark IDs (``UID*`` format) are found in ``{root}/{benchmark}_full.csv``
    where ``benchmark`` is the last path component of ``trace_dir``.
    """
    id_set = set(ids)
    result: dict[str, dict] = {}

    uid_ids   = {i for i in id_set if i.upper().startswith("UID")}
    synth_ids = id_set - uid_ids

    # ── synthesized pairs ──────────────────────────────────────────────────
    if synth_ids:
        dir_path = root / trace_dir
        try:
            for f in dir_path.glob("*_qa_pairs.json"):
                if not synth_ids:
                    break
                try:
                    pairs = json.loads(f.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                for p in pairs:
                    qa_id = p.get("qa_id", "")
                    if qa_id not in synth_ids:
                        continue
                    ans = p.get("answer", "")
                    if isinstance(ans, list):
                        ans = ans[0] if ans else ""
                    result[qa_id] = {"question": p.get("question", ""), "answer": str(ans)}
                    synth_ids.discard(qa_id)
        except OSError:
            pass

    # ── benchmark pairs ────────────────────────────────────────────────────
    if uid_ids:
        benchmark = trace_dir.split("/")[-1]
        csv_path = root / f"{benchmark}_full.csv"
        if csv_path.exists():
            try:
                with csv_path.open(newline="") as fh:
                    for row in csv.DictReader(fh):
                        qa_id = row.get("uid", "")
                        if qa_id in uid_ids:
                            result[qa_id] = {
                                "question": row.get("question", ""),
                                "answer":   row.get("answer", ""),
                            }
                            uid_ids.discard(qa_id)
                            if not uid_ids:
                                break
            except OSError:
                pass

    return result


# ── Request handler ────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    root: pathlib.Path  # injected at class level before server starts

    # silence default access log
    def log_message(self, fmt, *args):  # noqa: ARG002
        pass

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
            self._send_html(_APP_HTML.read_bytes())

        elif path == "/api/dirs":
            self._send_json({"dirs": _find_trace_dirs(self.root)})

        elif path == "/api/seeds":
            trace_dir = qs.get("dir", [""])[0]
            if not trace_dir:
                return self._send_json({"error": "missing dir"}, 400)
            self._send_json({"seeds": _list_seeds(self.root, trace_dir)})

        elif path == "/api/seed":
            trace_dir = qs.get("dir", [""])[0]
            seed_str = qs.get("seed", [""])[0]
            if not trace_dir or not seed_str:
                return self._send_json({"error": "missing dir or seed"}, 400)
            try:
                seed = int(seed_str)
            except ValueError:
                return self._send_json({"error": "invalid seed"}, 400)
            self._send_json(_seed_data(self.root, trace_dir, seed))

        elif path == "/api/funnel":
            trace_dir = qs.get("dir", [""])[0]
            if not trace_dir:
                return self._send_json({"error": "missing dir"}, 400)
            self._send_json(_funnel_data(self.root, trace_dir))

        elif path == "/api/qa_text":
            trace_dir = qs.get("dir", [""])[0]
            ids_str   = qs.get("ids", [""])[0]
            if not trace_dir or not ids_str:
                return self._send_json({"error": "missing dir or ids"}, 400)
            ids = [i.strip() for i in ids_str.split(",") if i.strip()]
            self._send_json(_qa_text_lookup(self.root, trace_dir, ids))

        elif path == "/api/messages":
            trace_dir = qs.get("dir", [""])[0]
            file = qs.get("file", [""])[0]
            if not trace_dir or not file:
                return self._send_json({"error": "missing dir or file"}, 400)
            # Security: reject path traversal
            try:
                file_path = (self.root / trace_dir / file).resolve()
                dir_path = (self.root / trace_dir).resolve()
                file_path.relative_to(dir_path)  # raises ValueError if outside
            except ValueError:
                return self._send_json({"error": "forbidden"}, 403)
            data = _load_json(file_path)
            if data is None:
                return self._send_json({"error": "not found"}, 404)
            self._send_json(data)

        else:
            self.send_response(404)
            self.end_headers()


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="QA Synthesis Trace Viewer")
    parser.add_argument("--port", type=int, default=7070, help="Port to listen on (default: 7070)")
    parser.add_argument(
        "--traces-root",
        default=None,
        help=(
            "Root directory containing *_qa_synthesis_traces/ dirs. "
            "Defaults to 3 levels above this script (the skunk/ dir)."
        ),
    )
    args = parser.parse_args()

    if args.traces_root:
        root = pathlib.Path(args.traces_root).resolve()
    else:
        # Script lives at skunk/src/trace_viewer/serve.py  →  3 levels up = skunk/
        root = pathlib.Path(__file__).resolve().parent.parent.parent

    _Handler.root = root

    server = HTTPServer(("localhost", args.port), _Handler)
    url = f"http://localhost:{args.port}"
    print(f"QA Trace Viewer → {url}")
    print(f"Traces root      → {root}")
    print("Press Ctrl-C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
