"""Treasury Bulletin page-index build driver.

Runs the five stages end-to-end, calling the concrete stage
implementations (in sibling modules) directly:

    catalog → extract_l1 → place_pages → merge_chapters → manifest

Outputs land under `--output-dir`:

    <output-dir>/
      build/{bulletin}.jsonl × N      rich build-only rows (BuildPage); resume state
      catalog/{bulletin}.jsonl × N    slim shipped rows (PageCatalogRow); query-facing
      l1/{bulletin}.json × N          per-bulletin L1 chapter spans
      concept_tree.json               flat global chapter tree
      manifest.json                   build metadata

The build threads its full per-page state (placement, banners, diagnostics)
through `build/`; the final `merge_chapters` stage projects that to the slim,
content-only `catalog/` the query path reads. Only `catalog/` ships.

Every stage is idempotent and re-runnable. Re-running picks up from the
last persisted output; `--start-from <stage>` skips earlier stages and
validates that their on-disk prerequisites are present.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import re
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
from pathlib import Path

from skunk.common import ExecutionContext, LLMResponse, load_env_file
from skunk.config import SkunkConfig
from skunk.trace import configure_obs

from skunk.corpus import (
    page_elements, page_text_tagged, parse_bulletin_filename, parsed_json_dir,
    pdf_dir_from_env,
)
from .data_model import (
    BUILD_STATS_FILE, BUILD_SUBDIR, CATALOG_SUBDIR, L1_SUBDIR, MANIFEST_FILE,
    TREE_FILE, BuildPage, PageCatalogRow,
)
from .catalog import TreasuryCatalogBuilder
from .summarize import PageSummarizer
from .l1_harvest import SectionSpan, TreasuryL1Harvester
from .placer import TreasuryPagePlacer
from .merger import TreasuryChapterMerger

log = logging.getLogger(__name__)


# Treasury Bulletin corpus knobs (the only corpus this build targets). PDF
# filename → (YYYY, MM); used by bulletin discovery and the manifest.
CORPUS_NAME = "treasury"


class StageError(RuntimeError):
    """Raised when a stage's prerequisites are missing on disk (e.g.
    running `place_pages` before `extract_l1`). Carries a remediation hint."""


_REPO_ROOT = Path(__file__).resolve().parents[3]
load_env_file(_REPO_ROOT / ".env")


STAGES = ("build_catalog", "summarize", "extract_l1", "place_pages",
          "merge_chapters", "manifest")


# ---------------------------------------------------------------------------
# Build-time LLM instrumentation
#
# Each LLM stage runs under its own `ExecutionContext`; `_StageClient` binds it
# into every `.call(...)` so the call-envelope telemetry (latency/tokens) flows
# through skunk's unified obs — the same path the query operators use. Per-stage
# `StageStats` are then derived from the ctx event stream by
# `_stage_stats_from_events` (no bespoke per-call accumulator).
# ---------------------------------------------------------------------------

class StageStats(BaseModel):
    """Per-stage LLM totals, derived from a stage ctx's event stream."""
    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_latency_s: float = 0.0   # summed across calls (parallel-overlapped)
    n_errors: int = 0
    wall_s: float = 0.0          # set by stage driver
    extra: dict = Field(default_factory=dict)  # stage-specific counters

    def record(
        self, *, input_tokens: int | None, output_tokens: int | None,
        latency_s: float,
    ) -> None:
        self.n_calls += 1
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)
        self.llm_latency_s += float(latency_s or 0.0)


class _StageClient:
    """Drop-in for `LLMClient` that binds a build stage's `ExecutionContext` into
    every `.call(...)`, so the call-envelope telemetry (latency/tokens, attributed
    by `call_site`) flows through skunk's unified obs. Stages use it exactly like an
    `LLMClient`; per-stage totals come from `ctx.events` afterwards. Shared across a
    stage's worker threads — only `.call` runs concurrently, and the work it does on
    the ctx (`emit` → `events.append`) is GIL-atomic."""

    def __init__(self, ctx: ExecutionContext) -> None:
        self._ctx = ctx
        self.n_errors = 0

    def call(self, *args, **kwargs) -> LLMResponse:
        try:
            return self._ctx.llm_client.call(*args, ctx=self._ctx, **kwargs)
        except Exception:
            self.n_errors += 1
            raise

    def __getattr__(self, name):  # pragma: no cover - thin proxy
        return getattr(self._ctx.llm_client, name)


# Per-stage LLM totals, keyed by stage name. LLM stages assign their derived
# `StageStats` here after running; `stage_manifest` writes them to `build_stats.json`.
_BUILD_STATS: dict[str, StageStats] = {s: StageStats() for s in STAGES}


def _envelope_field(msg: str, key: str) -> str | None:
    """Pull `key=<value>` from an `LLMClient` call-envelope message, or None when
    absent / literally "None"."""
    m = re.search(rf"\b{key}=(\S+)", msg)
    return None if (m is None or m.group(1) == "None") else m.group(1)


def _stage_stats_from_events(events: list[dict], *, n_errors: int) -> StageStats:
    """Derive a stage's LLM totals from its ctx event stream — one `record` per
    `LLMClient` call-envelope (`call call_site=... latency_s=L in_tok=N out_tok=M`)."""
    stats = StageStats(n_errors=n_errors)
    for evt in events:
        msg = evt.get("message", "")
        if not msg.startswith("call call_site="):
            continue
        in_tok = _envelope_field(msg, "in_tok")
        out_tok = _envelope_field(msg, "out_tok")
        lat = _envelope_field(msg, "latency_s")
        stats.record(
            input_tokens=int(in_tok) if in_tok else None,
            output_tokens=int(out_tok) if out_tok else None,
            latency_s=float(lat) if lat else 0.0,
        )
    return stats


def _approx_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Indicative Gemini Flash rate (matches the deleted retrieve-bench
    harness): $0.30 / M input, $2.50 / M output. Build stages and
    retrieval both run on the same model today."""
    return input_tokens * 0.30e-6 + output_tokens * 2.50e-6


# ---------------------------------------------------------------------------
# Helpers — corpus-agnostic FS layout + bulletin discovery
# ---------------------------------------------------------------------------

def _discover_bulletins(
    pdf_dir: Path, window: tuple[int, int],
    only: set[str] | None = None,
) -> list[tuple[str, Path]]:
    """Walk `pdf_dir`, filter by `window` (and optional `only` allowlist).
    Returns `[(bulletin_id, path), ...]` sorted by id."""
    y_lo, y_hi = window
    out: list[tuple[str, Path]] = []
    for p in sorted(pdf_dir.iterdir()):
        try:
            b = parse_bulletin_filename(p)
        except ValueError:
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


def _persist_rows(rows: list[PageCatalogRow], out_dir: Path) -> Path:
    """Write one bulletin's rows to `{out_dir}/{bulletin}.jsonl`. Works for both
    rich `BuildPage` (the `build/` intermediate) and slim `PageCatalogRow` (the
    shipped `catalog/`) — both expose `.bulletin` and `.to_json()`."""
    if not rows:
        raise ValueError("no rows to persist")
    bulletin = rows[0].bulletin
    path = out_dir / f"{bulletin}.jsonl"
    with path.open("w") as f:
        for r in rows:
            f.write(r.to_json())
            f.write("\n")
    return path


def _load_build_by_bulletin(build_dir: Path) -> dict[str, list[BuildPage]]:
    """Load the rich build intermediate, grouped by bulletin."""
    out: dict[str, list[BuildPage]] = {}
    for p in sorted(build_dir.glob("*.jsonl")):
        rows: list[BuildPage] = []
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            rows.append(BuildPage.from_json(line))
        if rows:
            out[rows[0].bulletin] = rows
    return out


def _finalize_catalog(build_by_bulletin: dict[str, list[BuildPage]],
                      catalog_dir: Path) -> int:
    """Project the rich build rows to slim, content-only `PageCatalogRow`s and
    write them to `catalog/` — the only artifact the query path reads. Drops
    non-content pages (blank / ToC / front-matter / masthead). Returns the row
    count written."""
    catalog_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for bulletin in sorted(build_by_bulletin):
        slim = [bp.to_row() for bp in build_by_bulletin[bulletin] if bp.is_content]
        if not slim:
            continue
        _persist_rows(slim, catalog_dir)
        n += len(slim)
    return n


def _save_l1(path: Path, spans: list[SectionSpan]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [sp.model_dump() for sp in spans]
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

def stage_build_catalog(args: argparse.Namespace) -> None:
    log.info("=== Stage 1: build_catalog ===")
    _require(args.pdf_dir.is_dir(), f"PDF dir does not exist: {args.pdf_dir}")
    parsed_dir = args.parsed_json_dir or parsed_json_dir()
    _require(parsed_dir.is_dir(),
             f"Parsed-JSON dir does not exist: {parsed_dir}")

    only = ({b.strip() for b in args.bulletins.split(",") if b.strip()}
            if args.bulletins else None)
    chosen = _discover_bulletins(args.pdf_dir, args.window, only=only)
    if only:
        missing = only - {b for b, _ in chosen}
        _require(not missing,
                 f"Requested bulletins not in window: {sorted(missing)}")
    _require(bool(chosen),
             f"No bulletins matched window {args.window} / --bulletins")

    out_dir = args.output_dir / BUILD_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"{len(chosen)} bulletins → {out_dir}")
    log.info(f"parsed-JSON: {parsed_dir}")
    log.info(f"workers: {args.workers}    (no LLM at this stage)")

    builder = TreasuryCatalogBuilder()
    t0 = time.monotonic()
    n_ok = n_err = 0

    def _do_one(bulletin: str, pdf_path: Path):
        try:
            pages = page_elements(bulletin, base_dir=parsed_dir, fill_gaps=True)
            rows = builder.parse_bulletin(bulletin, pages)
            out_path = _persist_rows(rows, out_dir)
            return bulletin, out_path, None
        except Exception as e:  # noqa: BLE001
            return bulletin, None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_do_one, b, p) for b, p in chosen]
        for f in as_completed(futs):
            bulletin, out_path, err = f.result()
            if err:
                n_err += 1
                log.error(f"[err] {bulletin}: {err}")
            else:
                n_ok += 1
                if args.verbose:
                    log.info(f"[ok]  {bulletin} → {out_path}")
    wall_s = time.monotonic() - t0
    stats = _BUILD_STATS["build_catalog"]
    stats.wall_s = wall_s
    stats.extra.update({"n_bulletins": len(chosen),
                        "n_ok": n_ok, "n_err": n_err})
    log.info(f"built {n_ok} bulletins, {n_err} errors in {wall_s:.1f}s")


def stage_summarize(args: argparse.Namespace) -> None:
    log.info("=== Stage 2: summarize ===")
    build_dir = args.output_dir / BUILD_SUBDIR
    _require(build_dir.is_dir(),
             f"build/ missing at {build_dir} — run build_catalog first.")
    parsed_dir = args.parsed_json_dir or parsed_json_dir()
    _require(parsed_dir.is_dir(),
             f"Parsed-JSON dir does not exist: {parsed_dir}")

    build_by_bulletin = _load_build_by_bulletin(build_dir)
    bulletins = sorted(build_by_bulletin)
    log.info(f"{len(bulletins)} bulletins → {build_dir}")

    cfg = _config_with_model(args.summarize_model)
    ctx = ExecutionContext(question="build:summarize", config=cfg)
    llm = _StageClient(ctx)
    log.info(f"model: {cfg.llm_model}")
    summarizer = PageSummarizer()
    t0 = time.monotonic()

    def _do_one(bulletin: str):
        try:
            pages = page_elements(bulletin, base_dir=parsed_dir, fill_gaps=True)
            rows = build_by_bulletin[bulletin]
            n_upd = summarizer.summarize_bulletin(
                bulletin=bulletin, rows=rows, pages=pages, llm=llm,
            )
            # Re-persist enriched rows back to the build intermediate (carries
            # banner_self / section etc. for the later placement + merge stages).
            _persist_rows(rows, build_dir)
            return bulletin, n_upd, None
        except Exception as e:  # noqa: BLE001
            return bulletin, 0, f"{type(e).__name__}: {e}"

    n_ok = n_err = n_updated_total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_do_one, b) for b in bulletins]
        for f in as_completed(futs):
            bulletin, n_upd, err = f.result()
            if err:
                n_err += 1
                log.error(f"[err] {bulletin}: {err}")
            else:
                n_ok += 1
                n_updated_total += n_upd
                if args.verbose:
                    log.info(f"[ok]  {bulletin}: {n_upd} pages summarized")

    wall_s = time.monotonic() - t0
    stats = _stage_stats_from_events(ctx.events, n_errors=llm.n_errors)
    stats.wall_s = wall_s
    stats.extra.update({"n_bulletins": len(bulletins), "n_ok": n_ok,
                        "n_err": n_err, "n_pages_summarized": n_updated_total})
    _BUILD_STATS["summarize"] = stats
    log.info(f"done {n_ok}/{len(bulletins)} bulletins, {n_err} errors, "
             f"{n_updated_total} pages summarized in {wall_s:.1f}s")
    log.info(f"LLM: calls={stats.n_calls}  "
             f"in={stats.input_tokens:,}  out={stats.output_tokens:,}  "
             f"~${_approx_cost_usd(stats.input_tokens, stats.output_tokens):.3f}")


def stage_extract_l1(args: argparse.Namespace) -> None:
    log.info("=== Stage 3: extract_l1 ===")
    build_dir = args.output_dir / BUILD_SUBDIR
    _require(build_dir.is_dir(),
             f"build/ missing at {build_dir} — run build_catalog first.")

    parsed_dir = args.parsed_json_dir or parsed_json_dir()
    _require(parsed_dir.is_dir(),
             f"Parsed-JSON dir does not exist: {parsed_dir}")

    l1_dir = args.output_dir / L1_SUBDIR
    l1_dir.mkdir(parents=True, exist_ok=True)

    build_by_bulletin = _load_build_by_bulletin(build_dir)
    bulletins = sorted(build_by_bulletin)
    log.info(f"{len(bulletins)} bulletins → {l1_dir}")

    cfg = _config_with_model(args.l1_model)
    ctx = ExecutionContext(question="build:extract_l1", config=cfg)
    llm = _StageClient(ctx)
    log.info(f"model: {cfg.llm_model}")
    harvester = TreasuryL1Harvester()
    t0 = time.monotonic()

    def _do_one(bulletin: str):
        try:
            pages_text = page_text_tagged(bulletin, base_dir=parsed_dir)
            rows = build_by_bulletin[bulletin]
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
                log.error(f"[err] {bulletin}: {err}")
            else:
                n_ok += 1
                span_dist.append(n_spans)
                if n_spans == 0:
                    n_empty += 1
                if args.verbose:
                    log.info(f"[ok]  {bulletin}: {n_spans} L1 spans")

    wall_s = time.monotonic() - t0
    span_dist.sort()
    median_spans = span_dist[len(span_dist) // 2] if span_dist else 0
    stats = _stage_stats_from_events(ctx.events, n_errors=llm.n_errors)
    stats.wall_s = wall_s
    stats.extra.update({"n_bulletins": len(bulletins), "n_ok": n_ok,
                        "n_err": n_err, "n_empty_spans": n_empty,
                        "median_spans": median_spans})
    _BUILD_STATS["extract_l1"] = stats
    if span_dist:
        log.info(f"done {n_ok}/{len(bulletins)} bulletins, {n_err} errors, "
                 f"{n_empty} with 0 spans, median {median_spans} spans in {wall_s:.1f}s")
    else:
        log.info(f"done {n_ok}/{len(bulletins)}, {n_err} errors in {wall_s:.1f}s")
    log.info(f"LLM: calls={stats.n_calls}  "
             f"in={stats.input_tokens:,}  out={stats.output_tokens:,}  "
             f"~${_approx_cost_usd(stats.input_tokens, stats.output_tokens):.3f}")


def stage_place_pages(args: argparse.Namespace) -> None:
    log.info("=== Stage 4: place_pages ===")
    build_dir = args.output_dir / BUILD_SUBDIR
    l1_dir = args.output_dir / L1_SUBDIR
    _require(build_dir.is_dir(),
             f"build/ missing at {build_dir} — run build_catalog first.")
    _require(l1_dir.is_dir(),
             f"l1/ missing at {l1_dir} — run extract_l1 first.")

    build_by_bulletin = _load_build_by_bulletin(build_dir)
    l1_by_bulletin: dict[str, list[SectionSpan]] = {}
    for bulletin in build_by_bulletin:
        l1_by_bulletin[bulletin] = _load_l1(l1_dir / f"{bulletin}.json")

    log.info(f"{len(build_by_bulletin)} bulletins; "
             f"sum L1 spans = {sum(len(v) for v in l1_by_bulletin.values())}")
    cfg = _config_with_model(args.place_model)
    ctx = ExecutionContext(question="build:place_pages", config=cfg)
    llm = _StageClient(ctx)
    log.info(f"model: {cfg.llm_model}")

    placer = TreasuryPagePlacer()
    items = sorted(build_by_bulletin.items())
    started = time.monotonic()

    def _run(bulletin: str) -> tuple[str, dict[str, int]]:
        rows = build_by_bulletin[bulletin]
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
                log.info(f"placed {done}/{len(items)} bulletins "
                         f"({time.monotonic() - started:.1f}s)")

    log.info("placement totals:")
    for k in sorted(totals.keys()):
        log.info(f"  {totals[k]:>7}  {k}")

    # Persist updated section placement back to the build intermediate.
    for _, rows in build_by_bulletin.items():
        _persist_rows(rows, build_dir)
    log.info(f"persisted updated build rows → {build_dir}")

    stats = _stage_stats_from_events(ctx.events, n_errors=llm.n_errors)
    stats.wall_s = time.monotonic() - started
    stats.extra.update({"placement_totals": dict(totals)})
    _BUILD_STATS["place_pages"] = stats
    log.info(f"LLM: calls={stats.n_calls}  "
             f"in={stats.input_tokens:,}  out={stats.output_tokens:,}  "
             f"~${_approx_cost_usd(stats.input_tokens, stats.output_tokens):.3f}")


def stage_merge_chapters(args: argparse.Namespace) -> None:
    log.info("=== Stage 5: merge_chapters ===")
    build_dir = args.output_dir / BUILD_SUBDIR
    _require(build_dir.is_dir(),
             f"build/ missing at {build_dir} — run build_catalog first.")

    build_by_bulletin = _load_build_by_bulletin(build_dir)
    catalog: list[BuildPage] = [r for rows in build_by_bulletin.values() for r in rows]
    log.info(f"{len(catalog)} build rows loaded")

    cfg = _config_with_model(args.merge_model)
    ctx = ExecutionContext(question="build:merge_chapters", config=cfg)
    llm = _StageClient(ctx)
    log.info(f"model: {cfg.llm_model}")

    t0 = time.monotonic()
    tree = TreasuryChapterMerger().build_tree(
        catalog, llm, drop_unfiled=True, verbose=True,
    )
    stats = _stage_stats_from_events(ctx.events, n_errors=llm.n_errors)
    stats.wall_s = time.monotonic() - t0
    _BUILD_STATS["merge_chapters"] = stats
    out_path = args.output_dir / TREE_FILE
    out_path.write_text(json.dumps(tree, ensure_ascii=False, indent=2))
    log.info(f"wrote tree → {out_path}")

    # Finalize: project the rich build rows to the slim, content-only catalog/
    # the query path reads. This is the only stage that writes catalog/.
    catalog_dir = args.output_dir / CATALOG_SUBDIR
    n_shipped = _finalize_catalog(build_by_bulletin, catalog_dir)
    log.info(f"finalized {n_shipped} content rows → {catalog_dir}")
    log.info(f"LLM: calls={stats.n_calls}  "
             f"in={stats.input_tokens:,}  out={stats.output_tokens:,}  "
             f"~${_approx_cost_usd(stats.input_tokens, stats.output_tokens):.3f}")


def stage_manifest(args: argparse.Namespace) -> None:
    log.info("=== Stage 6: manifest ===")
    out = args.output_dir
    tree_path = out / TREE_FILE
    catalog_dir = out / CATALOG_SUBDIR
    l1_dir = out / L1_SUBDIR
    _require(tree_path.exists(),
             "concept_tree.json missing — run merge_chapters first.")

    tree_raw = tree_path.read_bytes()
    tree = json.loads(tree_raw)
    tree_hash = hashlib.sha256(tree_raw).hexdigest()[:16]
    chapter_pages = {ch: data["n_pages"]
                     for ch, data in tree["chapters"].items()}
    total_pages = sum(chapter_pages.values())

    catalog_rows = 0
    for p in catalog_dir.glob("*.jsonl"):
        catalog_rows += sum(1 for line in p.open() if line.strip())
    n_l1_files = len(list(l1_dir.glob("*.json")))
    git_sha = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    cfg = SkunkConfig.from_env()
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": git_sha[:12] if git_sha else None,
        "profile": CORPUS_NAME,
        "pdf_dir": str(args.pdf_dir),
        "corpus": {
            "pdf_dir": str(args.pdf_dir),
            "bulletins": len(list(catalog_dir.glob("*.jsonl"))),
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
            "n_files": len(list(catalog_dir.glob("*.jsonl"))),
            "n_rows": catalog_rows,
        },
        "l1": {"path": "l1/", "n_files": n_l1_files},
        "models": {
            "summarize": args.summarize_model or cfg.llm_model,
            "extract_l1": args.l1_model or cfg.llm_model,
            "place_pages": args.place_model or cfg.llm_model,
            "merge_chapters": args.merge_model or cfg.llm_model,
            "retriever_runtime": cfg.llm_model,
        },
    }
    out_path = out / MANIFEST_FILE
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log.info(f"wrote manifest → {out_path}")
    log.info(f"chapters: {len(chapter_pages)}    pages: {total_pages}    "
             f"catalog rows: {catalog_rows}    L1 files: {n_l1_files}")

    # ---- Build statistics dump --------------------------------------------
    # Per-stage timing + LLM token usage (derived from each stage ctx's event
    # stream), plus index-structure stats from the freshly-written tree and
    # catalog. Lives next to `manifest.json` so anyone can replay the report
    # without re-running the build.
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
    for p in catalog_dir.glob("*.jsonl"):
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
            if row.date_interval:
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
    stats_path = out / BUILD_STATS_FILE
    stats_path.write_text(json.dumps(build_stats, indent=2, ensure_ascii=False))
    log.info(f"wrote build stats → {stats_path}")
    log.info(f"totals: {total_calls} LLM calls, "
             f"in={total_in:,} out={total_out:,}  "
             f"~${_approx_cost_usd(total_in, total_out):.3f}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_STAGE_FUNCS = {
    "build_catalog": stage_build_catalog,
    "summarize": stage_summarize,
    "extract_l1": stage_extract_l1,
    "place_pages": stage_place_pages,
    "merge_chapters": stage_merge_chapters,
    "manifest": stage_manifest,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Treasury Bulletin page-index build pipeline.")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("artifact/page_index_old"),
                    help="Target directory for all stage outputs (the shipped, in-repo index).")
    ap.add_argument("--window", type=_parse_window, default=(1939, 2025),
                    help="Inclusive year window (default 1939-2025).")
    ap.add_argument("--bulletins", type=str, default=None,
                    help="Comma-separated YYYY-MM to override (debug).")
    ap.add_argument("--pdf-dir", type=Path, default=pdf_dir_from_env(),
                    help="Directory holding corpus PDF files.")
    ap.add_argument("--parsed-json-dir", type=Path, default=None,
                    help="Parsed-JSON corpus dir; defaults to env/parsed_json_dir().")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--summarize-model", type=str, default=None,
                    help="LLM model override for summarize (else SkunkConfig default).")
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
    configure_obs()

    log.info(f"Pipeline → {args.output_dir}")
    log.info(f"profile: {CORPUS_NAME}")
    log.info(f"window: {args.window}    workers: {args.workers}")

    if args.start_from:
        start_idx = STAGES.index(args.start_from)
    else:
        start_idx = 0
    to_run = [s for s in STAGES[start_idx:]
              if not getattr(args, f"skip_{s}", False)]
    log.info(f"stages: {to_run}")

    t0 = time.monotonic()
    try:
        for stage in to_run:
            _STAGE_FUNCS[stage](args)
    except StageError as e:
        log.error(f"[stage prerequisite error] {e}")
        return 2
    log.info(f"Pipeline complete in {time.monotonic() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
