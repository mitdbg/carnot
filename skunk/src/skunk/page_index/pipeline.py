"""Corpus-agnostic page-index build driver.

Composes a `CorpusProfile` (`--profile <name>`) of five stage
implementations into the end-to-end build:

    catalog → extract_l1 → place_pages → merge_chapters → manifest

Outputs land under `--output-dir`:

    <output-dir>/
      catalog/{bulletin}.jsonl × N    per-bulletin catalog rows
      l1/{bulletin}.json × N          per-bulletin L1 chapter spans
      concept_tree.json               flat global chapter tree
      manifest.json                   build metadata

Every stage is idempotent and re-runnable. Re-running picks up from the
last persisted output; `--start-from <stage>` skips earlier stages and
validates that their on-disk prerequisites are present.

Cost / wall-clock estimates are profile-specific; consult the profile's
docs.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from skunk.common import LLMClient, LLMResponse, load_env_file
from skunk.config import SkunkConfig

from .corpora import PROFILES, load_profile
from skunk.corpus import (
    page_elements, page_text_tagged, parsed_json_dir, pdf_dir_from_env,
)
from .profile import CorpusProfile, StageError
from .schema import PageCatalogRow
from .stages.l1_harvest import SectionSpan


_REPO_ROOT = Path(__file__).resolve().parents[3]
load_env_file(_REPO_ROOT / ".env")


STAGES = ("build_catalog", "extract_l1", "place_pages",
          "merge_chapters", "manifest")


# ---------------------------------------------------------------------------
# Build-time LLM instrumentation
#
# Stages call `llm.call(...)` directly (no `ctx` is propagated), so token
# usage doesn't reach any sink today. `StageStats` accumulates per-stage
# call counts, tokens, and wall-clock; `StageLLMWrapper` is a drop-in for
# `LLMClient` that records each successful response under the active stage.
# ---------------------------------------------------------------------------

@dataclass
class StageStats:
    """Per-stage LLM call accumulator. Thread-safe via the lock below."""
    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_latency_s: float = 0.0   # summed across calls (parallel-overlapped)
    n_errors: int = 0
    wall_s: float = 0.0          # set by stage driver, not by per-call
    extra: dict = field(default_factory=dict)  # stage-specific counters

    def record(
        self, *, input_tokens: int | None, output_tokens: int | None,
        latency_s: float,
    ) -> None:
        self.n_calls += 1
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)
        self.llm_latency_s += float(latency_s or 0.0)


class StageLLMWrapper:
    """Wraps an `LLMClient` so every `.call(...)` records into a `StageStats`
    bucket. Same signature as `LLMClient.call`; passes everything through.

    Errors are not recorded as token usage (they raise before usage is
    populated); the wrapper increments `n_errors` and re-raises so the
    underlying retry/error handling stays unchanged.
    """

    def __init__(self, inner: LLMClient, stats: StageStats,
                 lock: threading.Lock) -> None:
        self._inner = inner
        self._stats = stats
        self._lock = lock

    def call(self, *args, **kwargs) -> LLMResponse:
        try:
            resp = self._inner.call(*args, **kwargs)
        except Exception:
            with self._lock:
                self._stats.n_errors += 1
            raise
        with self._lock:
            self._stats.record(
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                latency_s=resp.latency_s,
            )
        return resp

    # Defer anything we don't know about to the underlying client. Stages
    # only ever call `.call` today; this keeps the wrapper forward-compat.
    def __getattr__(self, name):  # pragma: no cover - thin proxy
        return getattr(self._inner, name)


# Process-wide stage stats, keyed by stage name. Populated by the stage
# drivers below and consumed by `stage_manifest` when it writes
# `build_stats.json`.
_BUILD_STATS: dict[str, StageStats] = {s: StageStats() for s in STAGES}
_BUILD_STATS_LOCK = threading.Lock()


def _approx_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Indicative Gemini Flash rate (matches the deleted retrieve-bench
    harness): $0.30 / M input, $2.50 / M output. Build stages and
    retrieval both run on the same model today."""
    return input_tokens * 0.30e-6 + output_tokens * 2.50e-6


# ---------------------------------------------------------------------------
# Helpers — corpus-agnostic FS layout + bulletin discovery
# ---------------------------------------------------------------------------

def _bulletin_id(path: Path, profile: CorpusProfile) -> str | None:
    """Use the profile's filename pattern to derive `YYYY-MM` for `path`.
    Returns None when the filename doesn't match."""
    m = profile.bulletin_filename_re.search(path.name)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def _discover_bulletins(
    pdf_dir: Path, profile: CorpusProfile, window: tuple[int, int],
    only: set[str] | None = None,
) -> list[tuple[str, Path]]:
    """Walk `pdf_dir`, filter by `window` (and optional `only` allowlist).
    Returns `[(bulletin_id, path), ...]` sorted by id."""
    y_lo, y_hi = window
    out: list[tuple[str, Path]] = []
    for p in sorted(pdf_dir.iterdir()):
        b = _bulletin_id(p, profile)
        if b is None:
            continue
        year = int(b[:4])
        if not (y_lo <= year <= y_hi):
            continue
        if only and b not in only:
            continue
        out.append((b, p))
    return out


def _parse_window(s: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{4})-(\d{4})", s)
    if not m:
        raise argparse.ArgumentTypeError(f"--window must be YYYY-YYYY, got {s!r}")
    lo, hi = int(m.group(1)), int(m.group(2))
    if hi < lo:
        raise argparse.ArgumentTypeError(f"--window end < start: {s}")
    return lo, hi


def _config_with_model(model: str | None) -> SkunkConfig:
    cfg = SkunkConfig.from_env()
    if model and model != cfg.llm_model:
        cfg = dataclasses.replace(cfg, llm_model=model)
    return cfg


def _persist_catalog(rows: list[PageCatalogRow], out_dir: Path,
                     *, drop_banner_self: bool = False) -> Path:
    if not rows:
        raise ValueError("no rows to persist")
    bulletin = rows[0].bulletin
    path = out_dir / f"{bulletin}.jsonl"
    with path.open("w") as f:
        for r in rows:
            f.write(r.to_json(drop_banner_self=drop_banner_self))
            f.write("\n")
    return path


def _load_catalog_by_bulletin(catalog_dir: Path) -> dict[str, list[PageCatalogRow]]:
    out: dict[str, list[PageCatalogRow]] = {}
    for p in sorted(catalog_dir.glob("*.jsonl")):
        rows: list[PageCatalogRow] = []
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            rows.append(PageCatalogRow.from_json(line))
        if rows:
            out[rows[0].bulletin] = rows
    return out


def _save_l1(path: Path, spans: list[SectionSpan]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [dataclasses.asdict(sp) for sp in spans]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _load_l1(path: Path) -> list[SectionSpan]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [SectionSpan(**item) for item in raw]


def _require(condition: bool, hint: str) -> None:
    if not condition:
        raise StageError(hint)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_build_catalog(args: argparse.Namespace, profile: CorpusProfile) -> None:
    print("\n=== Stage 1: build_catalog ===", flush=True)
    _require(args.pdf_dir.is_dir(), f"PDF dir does not exist: {args.pdf_dir}")
    parsed_dir = args.parsed_json_dir or parsed_json_dir()
    _require(parsed_dir.is_dir(),
             f"Parsed-JSON dir does not exist: {parsed_dir}")

    only = ({b.strip() for b in args.bulletins.split(",") if b.strip()}
            if args.bulletins else None)
    chosen = _discover_bulletins(args.pdf_dir, profile, args.window, only=only)
    if only:
        missing = only - {b for b, _ in chosen}
        _require(not missing,
                 f"Requested bulletins not in window: {sorted(missing)}")
    _require(bool(chosen),
             f"No bulletins matched window {args.window} / --bulletins")

    out_dir = args.output_dir / "catalog"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {len(chosen)} bulletins → {out_dir}")
    print(f"  parsed-JSON: {parsed_dir}")
    print(f"  workers: {args.workers}    (no LLM at this stage)\n", flush=True)

    builder = profile.catalog_builder
    t0 = time.monotonic()
    n_ok = n_err = 0

    def _do_one(bulletin: str, pdf_path: Path):
        try:
            pages = page_elements(bulletin, base_dir=parsed_dir, fill_gaps=True)
            rows = builder.parse_bulletin(bulletin, pages)
            out_path = _persist_catalog(rows, out_dir)
            return bulletin, out_path, None
        except Exception as e:  # noqa: BLE001
            return bulletin, None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_do_one, b, p) for b, p in chosen]
        for f in as_completed(futs):
            bulletin, out_path, err = f.result()
            if err:
                n_err += 1
                print(f"  [err] {bulletin}: {err}", file=sys.stderr, flush=True)
            else:
                n_ok += 1
                if args.verbose:
                    print(f"  [ok]  {bulletin} → {out_path}", flush=True)
    wall_s = time.monotonic() - t0
    stats = _BUILD_STATS["build_catalog"]
    stats.wall_s = wall_s
    stats.extra.update({"n_bulletins": len(chosen),
                        "n_ok": n_ok, "n_err": n_err})
    print(f"  built {n_ok} bulletins, {n_err} errors "
          f"in {wall_s:.1f}s", flush=True)


def stage_extract_l1(args: argparse.Namespace, profile: CorpusProfile) -> None:
    print("\n=== Stage 2: extract_l1 ===", flush=True)
    catalog_dir = args.output_dir / "catalog"
    _require(catalog_dir.is_dir(),
             f"catalog/ missing at {catalog_dir} — run build_catalog first.")

    parsed_dir = args.parsed_json_dir or parsed_json_dir()
    _require(parsed_dir.is_dir(),
             f"Parsed-JSON dir does not exist: {parsed_dir}")

    l1_dir = args.output_dir / "l1"
    l1_dir.mkdir(parents=True, exist_ok=True)

    catalog_by_bulletin = _load_catalog_by_bulletin(catalog_dir)
    bulletins = sorted(catalog_by_bulletin)
    print(f"  {len(bulletins)} bulletins → {l1_dir}", flush=True)

    cfg = _config_with_model(args.l1_model)
    inner_llm = LLMClient(cfg)
    stats = _BUILD_STATS["extract_l1"]
    llm = StageLLMWrapper(inner_llm, stats, _BUILD_STATS_LOCK)
    print(f"  model: {cfg.llm_model}\n", flush=True)
    harvester = profile.l1_harvester
    t0 = time.monotonic()

    def _do_one(bulletin: str):
        try:
            pages_text = page_text_tagged(bulletin, base_dir=parsed_dir)
            rows = catalog_by_bulletin[bulletin]
            spans = harvester.harvest_bulletin(
                bulletin=bulletin, rows=rows,
                pages_text=pages_text, llm=llm,
            )
            _save_l1(l1_dir / f"{bulletin}.json", spans)
            return bulletin, len(spans), None
        except Exception as e:  # noqa: BLE001
            return bulletin, 0, f"{type(e).__name__}: {e}"

    n_ok = n_err = n_empty = 0
    span_dist: list[int] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_do_one, b) for b in bulletins]
        for f in as_completed(futs):
            bulletin, n_spans, err = f.result()
            if err:
                n_err += 1
                print(f"  [err] {bulletin}: {err}", file=sys.stderr, flush=True)
            else:
                n_ok += 1
                span_dist.append(n_spans)
                if n_spans == 0:
                    n_empty += 1
                if args.verbose:
                    print(f"  [ok]  {bulletin}: {n_spans} L1 spans", flush=True)

    wall_s = time.monotonic() - t0
    stats.wall_s = wall_s
    stats.extra.update({"n_bulletins": len(bulletins), "n_ok": n_ok,
                        "n_err": n_err, "n_empty_spans": n_empty,
                        "median_spans": (span_dist[len(span_dist) // 2]
                                         if span_dist else 0)})
    if span_dist:
        span_dist.sort()
        mid = span_dist[len(span_dist) // 2]
        print(f"  done {n_ok}/{len(bulletins)} bulletins, {n_err} errors, "
              f"{n_empty} with 0 spans, median {mid} spans "
              f"in {wall_s:.1f}s", flush=True)
    else:
        print(f"  done {n_ok}/{len(bulletins)}, {n_err} errors "
              f"in {wall_s:.1f}s", flush=True)
    print(f"  LLM: calls={stats.n_calls}  "
          f"in={stats.input_tokens:,}  out={stats.output_tokens:,}  "
          f"~${_approx_cost_usd(stats.input_tokens, stats.output_tokens):.3f}",
          flush=True)


def stage_place_pages(args: argparse.Namespace, profile: CorpusProfile) -> None:
    print("\n=== Stage 3: place_pages ===", flush=True)
    catalog_dir = args.output_dir / "catalog"
    l1_dir = args.output_dir / "l1"
    _require(catalog_dir.is_dir(),
             f"catalog/ missing at {catalog_dir} — run build_catalog first.")
    _require(l1_dir.is_dir(),
             f"l1/ missing at {l1_dir} — run extract_l1 first.")

    catalog_by_bulletin = _load_catalog_by_bulletin(catalog_dir)
    l1_by_bulletin: dict[str, list[SectionSpan]] = {}
    for bulletin in catalog_by_bulletin:
        l1_by_bulletin[bulletin] = _load_l1(l1_dir / f"{bulletin}.json")

    print(f"  {len(catalog_by_bulletin)} bulletins; "
          f"sum L1 spans = {sum(len(v) for v in l1_by_bulletin.values())}",
          flush=True)
    cfg = _config_with_model(args.place_model)
    inner_llm = LLMClient(cfg)
    stats = _BUILD_STATS["place_pages"]
    llm = StageLLMWrapper(inner_llm, stats, _BUILD_STATS_LOCK)
    print(f"  model: {cfg.llm_model}", flush=True)

    placer = profile.page_placer
    items = sorted(catalog_by_bulletin.items())
    started = time.monotonic()

    def _run(bulletin: str) -> tuple[str, dict[str, int]]:
        rows = catalog_by_bulletin[bulletin]
        spans = l1_by_bulletin.get(bulletin, [])
        return bulletin, placer.place_bulletin(rows, spans, llm)

    totals: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run, b): b for b, _ in items}
        done = 0
        for f in as_completed(futs):
            _, per_bulletin = f.result()
            for k, n in per_bulletin.items():
                totals[k] += n
            done += 1
            if done % 50 == 0 or done == len(items):
                print(f"  placed {done}/{len(items)} bulletins "
                      f"({time.monotonic() - started:.1f}s)", flush=True)

    print("  placement totals:", flush=True)
    for k in sorted(totals.keys()):
        print(f"    {totals[k]:>7}  {k}", flush=True)

    # Persist with internal banner_self stripped.
    for _, rows in catalog_by_bulletin.items():
        _persist_catalog(rows, catalog_dir, drop_banner_self=True)
    print(f"  persisted updated catalog → {catalog_dir}", flush=True)

    stats.wall_s = time.monotonic() - started
    stats.extra.update({"placement_totals": dict(totals)})
    print(f"  LLM: calls={stats.n_calls}  "
          f"in={stats.input_tokens:,}  out={stats.output_tokens:,}  "
          f"~${_approx_cost_usd(stats.input_tokens, stats.output_tokens):.3f}",
          flush=True)


def stage_merge_chapters(args: argparse.Namespace, profile: CorpusProfile) -> None:
    print("\n=== Stage 4: merge_chapters ===", flush=True)
    catalog_dir = args.output_dir / "catalog"
    _require(catalog_dir.is_dir(),
             f"catalog/ missing at {catalog_dir} — run build_catalog first.")

    catalog: list[PageCatalogRow] = []
    for p in sorted(catalog_dir.glob("*.jsonl")):
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            catalog.append(PageCatalogRow.from_json(line))
    print(f"  {len(catalog)} catalog rows loaded", flush=True)

    cfg = _config_with_model(args.merge_model)
    inner_llm = LLMClient(cfg)
    stats = _BUILD_STATS["merge_chapters"]
    llm = StageLLMWrapper(inner_llm, stats, _BUILD_STATS_LOCK)
    print(f"  model: {cfg.llm_model}", flush=True)

    t0 = time.monotonic()
    tree = profile.chapter_merger.build_tree(
        catalog, llm, drop_unfiled=True, verbose=True,
    )
    stats.wall_s = time.monotonic() - t0
    out_path = args.output_dir / "concept_tree.json"
    out_path.write_text(json.dumps(tree, ensure_ascii=False, indent=2))
    print(f"  wrote tree → {out_path}", flush=True)
    print(f"  LLM: calls={stats.n_calls}  "
          f"in={stats.input_tokens:,}  out={stats.output_tokens:,}  "
          f"~${_approx_cost_usd(stats.input_tokens, stats.output_tokens):.3f}",
          flush=True)


def stage_manifest(args: argparse.Namespace, profile: CorpusProfile) -> None:
    print("\n=== Stage 5: manifest ===", flush=True)
    out = args.output_dir
    tree_path = out / "concept_tree.json"
    _require(tree_path.exists(),
             "concept_tree.json missing — run merge_chapters first.")

    tree = json.loads(tree_path.read_text())
    chapter_pages = {ch: data["n_pages"]
                     for ch, data in tree["chapters"].items()}
    total_pages = sum(chapter_pages.values())

    catalog_rows = 0
    for p in (out / "catalog").glob("*.jsonl"):
        catalog_rows += sum(1 for _ in p.open())
    n_l1_files = len(list((out / "l1").glob("*.json")))

    tree_hash = hashlib.sha256(tree_path.read_bytes()).hexdigest()[:16]
    git_sha = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    cfg = SkunkConfig.from_env()
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": git_sha[:12] if git_sha else None,
        "profile": profile.name,
        "pdf_dir": str(args.pdf_dir),
        "corpus": {
            "pdf_dir": str(args.pdf_dir),
            "bulletins": len(list((out / "catalog").glob("*.jsonl"))),
            "window": list(args.window),
        },
        "tree": {
            "path": "concept_tree.json",
            "sha256_first16": tree_hash,
            "n_chapters": len(chapter_pages),
            "n_pages_indexed": total_pages,
            "chapter_page_counts": chapter_pages,
        },
        "catalog": {
            "path": "catalog/",
            "n_files": len(list((out / "catalog").glob("*.jsonl"))),
            "n_rows": catalog_rows,
        },
        "l1": {"path": "l1/", "n_files": n_l1_files},
        "models": {
            "extract_l1": args.l1_model or cfg.llm_model,
            "place_pages": args.place_model or cfg.llm_model,
            "merge_chapters": args.merge_model or cfg.llm_model,
            "retriever_runtime": cfg.llm_model,
        },
    }
    out_path = out / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  wrote manifest → {out_path}", flush=True)
    print(f"  chapters: {len(chapter_pages)}    pages: {total_pages}    "
          f"catalog rows: {catalog_rows}    L1 files: {n_l1_files}",
          flush=True)

    # ---- Build statistics dump --------------------------------------------
    # Per-stage timing + LLM token usage collected via `StageLLMWrapper`.
    # Plus index-structure stats derived from the freshly-written tree
    # and catalog. Lives next to `manifest.json` so anyone can replay the
    # report without re-running the build.
    stages_out: dict[str, dict] = {}
    total_in = total_out = total_calls = 0
    total_wall = 0.0
    for name in STAGES:
        s = _BUILD_STATS[name]
        stages_out[name] = {
            "wall_s": round(s.wall_s, 2),
            "n_llm_calls": s.n_calls,
            "input_tokens": s.input_tokens,
            "output_tokens": s.output_tokens,
            "sum_llm_latency_s": round(s.llm_latency_s, 2),
            "n_llm_errors": s.n_errors,
            "approx_cost_usd": round(
                _approx_cost_usd(s.input_tokens, s.output_tokens), 4),
            "extra": s.extra,
        }
        total_in += s.input_tokens
        total_out += s.output_tokens
        total_calls += s.n_calls
        total_wall += s.wall_s

    # Index-structure stats: chapter page distribution, examples coverage,
    # year + bulletin coverage, content-tier mix from per-row metadata.
    chapter_examples = {ch: len(data.get("examples", []))
                        for ch, data in tree["chapters"].items()}
    chapter_pages_sorted = sorted(chapter_pages.items(), key=lambda x: -x[1])
    page_counts = list(chapter_pages.values())
    page_counts.sort()
    n_ch = max(1, len(page_counts))
    median_idx = n_ch // 2
    # Catalog year coverage from one pass over catalog rows. (Placement
    # tier counts come from the place_pages stage's `extra` payload — no
    # need to re-derive them here.)
    bulletins_seen: set[str] = set()
    year_min = None
    year_max = None
    n_pages_with_dates = 0
    n_pages_total = 0
    n_pages_with_content = 0
    for p in (out / "catalog").glob("*.jsonl"):
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            row = PageCatalogRow.from_json(line)
            bulletins_seen.add(row.bulletin)
            n_pages_total += 1
            if row.content_blocks:
                n_pages_with_content += 1
            try:
                yr = int(row.bulletin[:4])
                year_min = yr if year_min is None else min(year_min, yr)
                year_max = yr if year_max is None else max(year_max, yr)
            except ValueError:
                pass
            if row.dates:
                n_pages_with_dates += 1

    build_stats = {
        "pipeline_wall_s": round(total_wall, 2),
        "totals": {
            "n_llm_calls": total_calls,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "approx_cost_usd": round(
                _approx_cost_usd(total_in, total_out), 4),
        },
        "stages": stages_out,
        "index_structure": {
            "n_chapters": len(chapter_pages),
            "n_pages_indexed": total_pages,
            "n_bulletins": len(bulletins_seen),
            "year_range": [year_min, year_max],
            "chapter_page_counts": dict(chapter_pages_sorted),
            "chapter_examples_count": chapter_examples,
            "chapter_pages_p50": page_counts[median_idx] if page_counts else 0,
            "chapter_pages_min": page_counts[0] if page_counts else 0,
            "chapter_pages_max": page_counts[-1] if page_counts else 0,
            "chapter_pages_mean": round(sum(page_counts) / n_ch, 1),
            "catalog_rows": n_pages_total,
            "catalog_rows_with_content": n_pages_with_content,
            "catalog_rows_with_dates": n_pages_with_dates,
            "catalog_dates_coverage": (
                round(n_pages_with_dates / max(1, n_pages_total), 4)),
            "placement_tier_counts": (
                _BUILD_STATS["place_pages"].extra.get("placement_totals", {})),
        },
        "models": manifest["models"],
        "built_at": manifest["built_at"],
        "git_sha": manifest["git_sha"],
    }
    stats_path = out / "build_stats.json"
    stats_path.write_text(json.dumps(build_stats, indent=2, ensure_ascii=False))
    print(f"  wrote build stats → {stats_path}", flush=True)
    print(f"  totals: {total_calls} LLM calls, "
          f"in={total_in:,} out={total_out:,}  "
          f"~${_approx_cost_usd(total_in, total_out):.3f}", flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_STAGE_FUNCS = {
    "build_catalog": stage_build_catalog,
    "extract_l1": stage_extract_l1,
    "place_pages": stage_place_pages,
    "merge_chapters": stage_merge_chapters,
    "manifest": stage_manifest,
}


def _default_pdf_dir() -> Path:
    env = os.environ.get("OFFICEQA_PDF_DIR")
    return Path(env) if env else pdf_dir_from_env()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Corpus-agnostic page-index build pipeline.")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="treasury",
                    help="Corpus profile (default: treasury).")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("cache/page_index"),
                    help="Target directory for all stage outputs.")
    ap.add_argument("--window", type=_parse_window, default=(1939, 2025),
                    help="Inclusive year window (default 1939-2025).")
    ap.add_argument("--bulletins", type=str, default=None,
                    help="Comma-separated YYYY-MM to override (debug).")
    ap.add_argument("--pdf-dir", type=Path, default=_default_pdf_dir(),
                    help="Directory holding corpus PDF files.")
    ap.add_argument("--parsed-json-dir", type=Path, default=None,
                    help="Parsed-JSON corpus dir; defaults to env/parsed_json_dir().")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--l1-model", type=str, default=None,
                    help="LLM model override for extract_l1 (else SkunkConfig default).")
    ap.add_argument("--place-model", type=str, default=None,
                    help="LLM model override for place_pages.")
    ap.add_argument("--merge-model", type=str, default=None,
                    help="LLM model override for merge_chapters.")
    ap.add_argument("--start-from", choices=STAGES, default=None,
                    help="Skip stages prior to this one (their outputs must exist).")
    for s in STAGES:
        ap.add_argument(f"--skip-{s.replace('_', '-')}", action="store_true",
                        dest=f"skip_{s}")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    profile = load_profile(args.profile)
    print(f"Pipeline → {args.output_dir}", flush=True)
    print(f"  profile: {profile.name}", flush=True)
    print(f"  window: {args.window}    workers: {args.workers}", flush=True)

    if args.start_from:
        start_idx = STAGES.index(args.start_from)
    else:
        start_idx = 0
    to_run = [s for s in STAGES[start_idx:]
              if not getattr(args, f"skip_{s}", False)]
    print(f"  stages: {to_run}", flush=True)

    t0 = time.monotonic()
    try:
        for stage in to_run:
            _STAGE_FUNCS[stage](args, profile)
    except StageError as e:
        print(f"\n[stage prerequisite error] {e}", file=sys.stderr, flush=True)
        return 2
    print(f"\nPipeline complete in {time.monotonic() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
