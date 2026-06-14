"""End-to-end page-index build pipeline.

One entry point (`main` / `python -m skunk.page_index.pipeline`) runs every stage in
order over the Treasury Bulletin corpus, writing each stage's artifact into a single
build folder. There are deliberately NO flags to run a stage in isolation — the whole
pipeline runs start to finish; ad-hoc single-stage runs are patched in as needed.

The pipeline is a list of `Stage`s (`PIPELINE`) that `run_build` calls in order. Most are
per-bulletin (`BulletinStage`); two are whole-corpus reductions:

  1. `scan`        (`build/scans/<b>.json`)    — one `scan_page` LLM call per non-empty page
     → its `PageScan` (role, blocks, dates) + per-page errors.
  2. `vision_rescan` (reduction)               — re-scan each page the text scan flagged
     (`has_unparsed_graphics` / `parse_broken`) from its rendered PDF IMAGE, overwriting that
     page's record in the scan file in place (one `vision_scan_page` LLM call per flagged page).
     Pages render through the shared `renders/` image cache; warm it ahead of time with the
     standalone `skunk.page_index.prep.render_corpus` tool (cold pages are rendered on demand).
  3. `table_merge` (reduction)                 — link each label-less table fragment (a block
     with NO row/column labels: a continued table's bare data rows, or a footnotes spillover)
     to its parent block via one flash call per candidate, annotating the scan file in place
     (`extra_pages` on the parent / `merged_into` on the fragment; see `table_merge.py`).
  4. `toc`         (`build/toc/<b>.json`)      — coalesce the scan's `toc` pages, then one
     `outline_issue` call extracts the chapter outline AND flags non-ToCs (`is_toc`).
  5. `reconstruct_toc` (reduction)             — fill each ToC-less issue's outline from its
     section-divider pages, using neighbors' real ToCs as reference (overwrites its toc file).
  6. `place`       (`build/place/<b>.json`)    — file each content page under its chapter.
  7. `catalog`     (`build/catalog/<b>.jsonl`) — the slim query-facing per-page rows.
  8. `page_store`  (`build/pages/<b>.json`)    — per catalog row, its page's JSON text
     (reusing `chop_bulletin`) + figure descriptors; the content source the query path reads.
  9. `era_merge`   (`concept_tree.json`, reduction) — segment the timeline into eras, then
     build each era's canonical table of contents.

There is deliberately NO page-level continuation merge: the scan's `is_continuation` flag is
kept as metadata only. An earlier `merge_continuations` stage folded flagged pages into their
preceding anchor, but the flag fires on "(Continued)" captions for pages that are fully
self-contained (in this corpus continued tables restate their headers), so merging glued
distinct self-readable pages into multi-page anchors — wrecking per-page `date_interval`s
and ballooning what retrieval hands extract. Every page stands as its own catalog row; the
rare GENUINE continuations (~50 corpus-wide) are linked at TABLE granularity by `table_merge`.

Shared machinery — a single `BuildContext` owns the LLM client, the corpus list, the
process-wide token budget, and the concurrency semaphore. The `Stage` protocol is just
`name` + `run`; `BulletinStage.run` gives per-bulletin stages uniform resume (skip bulletins
already built), persistence, and progress, while the reductions carry their own resume
(skip when their output already exists).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from skunk.common import ExecutionContext, load_env_file
from skunk.errors import ParseError
from skunk.config import SkunkConfig
from skunk.corpus import page_elements, parse_bulletin_filename
from skunk.llm_client import LLMClient
from skunk.prompted_call import load_prompt_overrides
from skunk.trace import configure_obs

from .data_model import PAGES_SUBDIR, RENDERS_SUBDIR, TREE_FILE, PageCatalogRow
from .eras import build_concept_tree
from .scan import PageScan, scan_page, vision_scan_page
from .table_merge import apply_merge, find_candidates, resolve_parent
from .store import read_cached_image, render_to_cache
from .toc_index import (
    TocHierarchy,
    coalesce_toc_ranges,
    outline_issue,
    place_pages,
    reconstruct_outline,
)

# Fixed name under the `skunk.*` tree (so `python -m`'s `__main__` still logs at INFO).
log = logging.getLogger("skunk.page_index.pipeline")

CORPUS_NAME = "treasury"

_REPO_ROOT = Path(__file__).resolve().parents[3]
load_env_file(_REPO_ROOT / ".env")

# Gemini Flash token pricing (USD/token), matching eval/eval_retrieve.py's
# `_FLASH_IN`/`_FLASH_OUT`: $0.30 / 1M input, $2.50 / 1M output.
_FLASH_IN, _FLASH_OUT = 0.30e-6, 2.50e-6

# Visual element types whose mere presence is content (a figure-only page is not blank).
_VISUAL_TYPES = frozenset({"figure", "image", "chart", "plot", "diagram"})

# Pull the token counts the LLM client stamps onto each `call` event off a unit ctx.
_CALL_RE = re.compile(r"in_tok=(\d+) out_tok=(\d+)")

# How often (in completed bulletins) a stage writes progress + re-checks guards.
_PROGRESS_EVERY = 24


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


def _tokens_from_ctx(ctx: ExecutionContext) -> tuple[int, int]:
    """Sum input/output tokens across a unit ctx's `call` events (None tok → skipped)."""
    tin = tout = 0
    for e in ctx.events:
        m = _CALL_RE.search(e.get("message", ""))
        if m:
            tin += int(m.group(1))
            tout += int(m.group(2))
    return tin, tout


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` via a tmp file + rename (no partial files)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    tmp.rename(path)


def _atomic_write_json(path: Path, obj: object) -> None:
    """Write `obj` as JSON to `path` via a tmp file + rename (no partial files)."""
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Pass 1 — chop a bulletin's parsed JSON into per-page strings (deterministic, no LLM)
# ---------------------------------------------------------------------------


def elements_to_text(elements: list[dict]) -> str:
    """One page's parsed-JSON elements → a tagged string (`[type] content`) in document
    order. A visual element with no text still emits a bare `[type]` marker so
    figure-only pages aren't lost; other text-less elements are skipped."""
    parts: list[str] = []
    for el in elements:
        t = el.get("type") or "text"
        content = el.get("content")
        if content is None or content == "":
            if t in _VISUAL_TYPES:
                parts.append(f"[{t}]")
            continue
        parts.append(f"[{t}] {content}")
    return "\n\n".join(parts)


def chop_bulletin(bulletin: str, *, parsed_json_dir: Path | str) -> dict[int, str]:
    """`{1-based PDF page: page_string}` for every page of one bulletin (1..max in
    order; blank pages map to "")."""
    pages = page_elements(bulletin, base_dir=parsed_json_dir, fill_gaps=True)
    return {idx: elements_to_text(elements) for idx, elements in sorted(pages.items())}


def discover_bulletins(pdf_dir: Path, only: set[str] | None = None) -> list[str]:
    """Corpus bulletin ids (sorted YYYY-MM), optionally restricted to the `only` set."""
    out: list[str] = []
    for p in sorted(pdf_dir.iterdir()):
        try:
            b = parse_bulletin_filename(p)
        except ValueError:
            continue
        if only and b not in only:
            continue
        out.append(b)
    return out


# ---------------------------------------------------------------------------
# Build folder layout + shared run context
# ---------------------------------------------------------------------------

_SCANS_SUBDIR = "scans"
_TOC_SUBDIR = "toc"
_PLACE_SUBDIR = "place"
_CATALOG_SUBDIR = "catalog"


@dataclass(frozen=True)
class BuildPaths:
    """Every build artifact lives under one `root` (the `--build-dir`)."""

    root: Path

    def stage_dir(self, subdir: str) -> Path:
        return self.root / subdir

    def bulletin_file(self, subdir: str, bulletin: str, ext: str = "json") -> Path:
        """A stage's per-bulletin artifact path — the one place the `<root>/<subdir>/
        <bulletin>.<ext>` layout is spelled (the driver writes it; a later stage reads
        an earlier stage's via the same accessor). Most stages use `.json`; the catalog
        stage ships `.jsonl`."""
        return self.root / subdir / f"{bulletin}.{ext}"

    @property
    def progress_file(self) -> Path:
        return self.root / "_progress.json"


@dataclass
class BuildContext:
    """Per-run state shared by every stage: corpus list, config + prompt overrides, the
    one LLM client and its concurrency semaphore, and cumulative token totals for the
    cost report. One instance per `run_build`, used on one event loop.

    The model and its RPM/TPM rate limits are NOT owned here — they come from
    `SkunkConfig` and the `llm_client` rate limiter (env SKUNK_LLM_MODEL,
    SKUNK_MODEL_RPM/TPM, SKUNK_LLM_RPM), the same as every other entry point."""

    config: SkunkConfig
    paths: BuildPaths
    bulletins: list[str]
    overrides: tuple = ()
    llm_concurrency: int = 128

    client: LLMClient = field(init=False)
    llm_sem: asyncio.Semaphore = field(init=False)
    in_tok: int = field(default=0, init=False)
    out_tok: int = field(default=0, init=False)
    _t0: float = field(init=False)

    def __post_init__(self) -> None:
        self.client = LLMClient(self.config)
        # Created here (before the loop) but only ever used inside the single
        # `asyncio.run(run_build(...))` loop, so it binds to that loop on first use.
        self.llm_sem = asyncio.Semaphore(self.llm_concurrency)
        self._t0 = time.perf_counter()

    @property
    def cost(self) -> float:
        return self.in_tok * _FLASH_IN + self.out_tok * _FLASH_OUT

    @asynccontextmanager
    async def unit_ctx(self):
        """One LLM unit's throwaway ctx (shared client; its small event list is GC'd
        after the unit). On exit it folds the unit's token usage into the run totals
        and closes — so every stage's cost is counted by construction, the stage just
        does `async with bctx.unit_ctx() as ctx:`."""
        ctx = ExecutionContext(
            question="build",
            llm_client=self.client,
            config=self.config,
            prompt_overrides=self.overrides,
        )
        try:
            yield ctx
        finally:
            tin, tout = _tokens_from_ctx(ctx)
            self.in_tok += tin
            self.out_tok += tout
            ctx.close()

    def write_progress(self, stage: str, done: int, total: int, agg: dict) -> None:
        _atomic_write_json(
            self.paths.progress_file,
            {
                "stage": stage,
                "stage_done": done,
                "stage_total": total,
                "in_tok": self.in_tok,
                "out_tok": self.out_tok,
                "cost_usd": round(self.cost, 2),
                "wall_s": round(time.perf_counter() - self._t0, 1),
                **{f"stage_{k}": v for k, v in agg.items()},
            },
        )


# ---------------------------------------------------------------------------
# Stage protocol + driver
# ---------------------------------------------------------------------------


class Stage(ABC):
    """One pipeline stage. `run` does the stage's whole job and returns its aggregated
    stats; `run_build` just calls each stage's `run` in order. Per-bulletin stages subclass
    `BulletinStage`; the whole-corpus reductions (`ReconstructTocStage`, `EraMergeStage`)
    implement `run` directly."""

    name: str

    @abstractmethod
    async def run(self, bctx: BuildContext) -> dict: ...


class BulletinStage(Stage):
    """A per-bulletin stage: produces `<build>/<out_subdir>/<bulletin>.<ext>` for each
    bulletin. `run` owns iteration, resume (skip already-built bulletins), concurrency,
    persistence, and progress; a subclass owns only its per-bulletin `process`."""

    out_subdir: str
    out_ext: str = (
        "json"  # per-bulletin file extension; the catalog stage ships "jsonl"
    )
    workers: int = 16  # max bulletins open at once (memory); LLM calls are additionally
    #                     bounded by the shared `BuildContext.llm_sem`.

    @abstractmethod
    async def process(
        self, bctx: BuildContext, bulletin: str
    ) -> tuple[dict | None, dict]:
        """Build one bulletin. Returns `(payload | None, stats_delta)`; `payload=None`
        means "write nothing" (e.g. a prerequisite artifact is missing)."""

    def serialize(self, payload: dict) -> str:
        """Render a `process` payload to the bytes written on disk. Default is JSON;
        the catalog stage overrides this to emit JSONL."""
        return json.dumps(payload, ensure_ascii=False)

    async def run(self, bctx: BuildContext) -> dict:
        """Run `process` over every not-yet-built bulletin: `workers` at a time, each
        persisted as it finishes, with periodic progress writes."""
        out_dir = bctx.paths.stage_dir(self.out_subdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pending = [
            b
            for b in bctx.bulletins
            if not bctx.paths.bulletin_file(self.out_subdir, b, self.out_ext).exists()
        ]
        n_skip = len(bctx.bulletins) - len(pending)
        log.info(
            f"[{self.name}] {len(bctx.bulletins)} bulletins: {n_skip} done, "
            f"{len(pending)} to build -> {out_dir}"
        )
        if not pending:
            return {}

        sem = asyncio.Semaphore(self.workers)
        agg: dict[str, int] = defaultdict(int)
        done = 0

        async def one(bulletin: str) -> None:
            nonlocal done
            async with sem:
                payload, delta = await self.process(bctx, bulletin)
                if payload is not None:
                    _atomic_write_text(
                        bctx.paths.bulletin_file(
                            self.out_subdir, bulletin, self.out_ext
                        ),
                        self.serialize(payload),
                    )
                for k, v in delta.items():
                    agg[k] += v
                done += 1
                if done % _PROGRESS_EVERY == 0 or done == len(pending):
                    bctx.write_progress(self.name, done, len(pending), agg)
                    log.info(
                        f"[{self.name}] {done}/{len(pending)} | ${bctx.cost:.2f} | {dict(agg)}"
                    )

        await asyncio.gather(*[one(b) for b in pending])
        return dict(agg)


# ---------------------------------------------------------------------------
# Stage: scan (pass 2)
# ---------------------------------------------------------------------------


class ScanStage(BulletinStage):
    name = "scan"
    out_subdir = _SCANS_SUBDIR
    workers = (
        16  # ~16 bulletins' page text in memory; ~128 page calls in flight (llm_sem)
    )

    async def process(
        self, bctx: BuildContext, bulletin: str
    ) -> tuple[dict | None, dict]:
        pages = chop_bulletin(bulletin, parsed_json_dir=bctx.config.parsed_json_dir)
        nonempty = {p: t for p, t in pages.items() if t.strip()}
        scans: dict[int, dict] = {}
        errors: dict[int, str] = {}
        deferred: dict[int, str] = {}

        async def scan_one(page: int, text: str) -> None:
            async with bctx.llm_sem, bctx.unit_ctx() as ctx:
                try:
                    scans[page] = (
                        await scan_page(ctx, text, bulletin=bulletin)
                    ).model_dump(mode="json")
                except ParseError as e:
                    # The text scan couldn't yield valid output even after its temperature-escalating
                    # retries — a handful of dense numeric tables (e.g. the Foreign Series Securities
                    # schedule, packed with "M/D/YY" tokens) reliably derail it into degenerate or
                    # invalid JSON. Don't drop the page: keep a `parse_broken` placeholder so the
                    # `vision_rescan` tier re-reads it from the rendered image (a different modality
                    # that doesn't choke on the raw HTML), then overwrites this record.
                    scans[page] = PageScan(
                        page_role="content", parse_broken=True
                    ).model_dump(mode="json")
                    deferred[page] = e.detail[:120]
                except Exception as e:  # noqa: BLE001 — record, never abort the batch
                    errors[page] = f"{type(e).__name__}: {e}"

        await asyncio.gather(*[scan_one(p, t) for p, t in nonempty.items()])

        payload = {
            "bulletin": bulletin,
            "n_pages": len(nonempty),
            "scans": {str(p): scans[p] for p in sorted(scans)},
            "errors": {str(p): errors[p] for p in sorted(errors)},
            "deferred": {str(p): deferred[p] for p in sorted(deferred)},
        }
        return payload, {
            "pages": len(nonempty),
            "failed": len(errors),
            "deferred_to_vision": len(deferred),
        }


# ---------------------------------------------------------------------------
# Stage: vision_rescan — redo flagged pages from rendered PDF images (LLM, vision)
# ---------------------------------------------------------------------------


def _is_flagged(scan: dict, *, include_charts: bool) -> bool:
    """A page the text scan flagged for a vision re-read. Two cases:

    - `parse_broken`: the parsed elements were too mangled to trust — ALWAYS re-read.
    - `has_unparsed_graphics`: a chart/figure whose data isn't in the page's own text/tables.
      Re-read only when `include_charts` is on AND the page has NO table block — i.e. the chart
      is the page's sole data source (chart-only / figure-only), so its data is lost without
      vision. A chart beside a table is redundant (the table carries the numbers). Chart re-read
      is an opt-in phase (`config.vision_rescan_charts`, default off): it's the bulk of the
      vision working set (~6k pages) and low-value for table-centric queries.
    """
    if scan.get("parse_broken"):
        return True
    if include_charts and scan.get("has_unparsed_graphics"):
        return not any(b.get("kind") == "table" for b in scan.get("blocks", []))
    return False


def _patch_catalog_inline(
    bctx: BuildContext, bulletin: str, scans: dict[str, dict], revised: list[int]
) -> None:
    """Replace each `revised` page's row in the bulletin's existing catalog file, in place.
    No-op when the catalog file doesn't exist yet (a fresh forward build — the catalog stage
    builds it later from the updated scans). Untouched rows keep their original bytes; a revised
    page that is no longer `content` is dropped, one that became `content` is added. Rows are
    rewritten in ascending page order, matching `CatalogStage`."""
    if not revised:
        return
    cat_path = bctx.paths.bulletin_file(_CATALOG_SUBDIR, bulletin, "jsonl")
    if not cat_path.exists():
        return
    by_page: dict[int, str] = {}
    for line in cat_path.read_text().splitlines():
        if line.strip():
            by_page[PageCatalogRow.from_json(line).page] = line
    for page in revised:
        row = _catalog_row(bulletin, page, scans[str(page)])
        if row is None:
            by_page.pop(page, None)
        else:
            by_page[page] = row.to_json()
    _atomic_write_text(cat_path, "\n".join(by_page[p] for p in sorted(by_page)))


class VisionRescanStage(Stage):
    """Reduction (after `scan`, before `toc`): re-scan each page the text
    scan flagged — `has_unparsed_graphics` (a chart/figure the parser dropped) or
    `parse_broken` (cells too scrambled to trust) — from its rendered PDF IMAGE, overwriting
    that page's record in the scan artifact in place. The vision LLM can read the graphic or
    the garbled table the parsed text lost, emitting the same `PageScan` object.

    Resume is PER PAGE: a page is re-read iff it is flagged AND its own `vision_rescanned`
    marker is unset, and the marker is set only on a successful re-read. So a transient (e.g.
    network) failure leaves just that page unmarked — it retries next run on its own, with no
    bulletin-wide reset and no redundant re-read of pages that already succeeded. A legacy
    bulletin-level `vision_rescanned` flag (from an earlier coarser pass) is migrated to
    per-page markers on first encounter.

    When a bulletin's catalog file already exists (a patch over an already-built corpus, not a
    fresh forward build where the catalog stage runs later), each revised page's catalog row is
    replaced inline from the new scan via `_catalog_row` — so the shipped catalog reflects the
    vision results without a full catalog rebuild. Untouched rows are left byte-for-byte."""

    name = "vision_rescan"
    workers = (
        16  # bulletins open at once; page calls are additionally bounded by `llm_sem`
    )

    async def run(self, bctx: BuildContext) -> dict:
        def work(data: dict) -> tuple[bool, list[int]]:
            """`(needs_migration, pages_to_read)` for a loaded scan file: a legacy bulletin-level
            flag means migrate; otherwise the flagged pages whose per-page marker is unset."""
            if data.get("vision_rescanned"):
                return True, []
            todo = [
                int(p)
                for p, s in data["scans"].items()
                if _is_flagged(s, include_charts=bctx.config.vision_rescan_charts)
                and not s.get("vision_rescanned")
            ]
            return False, todo

        pending: list[str] = []
        for b in bctx.bulletins:
            path = bctx.paths.bulletin_file(_SCANS_SUBDIR, b)
            if not path.exists():
                continue
            migrate, todo = work(json.loads(path.read_text()))
            if migrate or todo:
                pending.append(b)
        log.info(
            f"[{self.name}] {len(bctx.bulletins)} bulletins: "
            f"{len(bctx.bulletins) - len(pending)} done, {len(pending)} with pages to rescan"
        )
        if not pending:
            return {}

        sem = asyncio.Semaphore(self.workers)
        agg: dict[str, int] = defaultdict(int)
        done = 0
        renders_dir = bctx.paths.root / RENDERS_SUBDIR

        async def one(bulletin: str) -> None:
            nonlocal done
            async with sem:
                path = bctx.paths.bulletin_file(_SCANS_SUBDIR, bulletin)
                data = json.loads(path.read_text())
                scans: dict[str, dict] = data["scans"]
                migrate, todo = work(data)

                if migrate:
                    # Legacy bulletin-level flag → per-page markers on its flagged pages (all of
                    # which were re-read under the old pass), then drop the bulletin-level key.
                    data.pop("vision_rescanned")
                    for s in scans.values():
                        # Legacy bulletin-level flag: the old pass re-read everything flagged
                        # (charts included), so migrate those same pages' per-page markers.
                        if _is_flagged(s, include_charts=True):
                            s["vision_rescanned"] = True
                    _atomic_write_json(path, data)
                    agg["migrated"] += 1
                elif todo:
                    revised: list[int] = []

                    async def rescan_one(page: int) -> None:
                        # Render through the shared `renders/` cache, cache-first: a page the
                        # `render_corpus` prep tool already warmed is just read back; a cold one is
                        # rendered+cached here. Offloaded to a thread so the event loop keeps
                        # driving in-flight LLM calls. Same PNG the query-time vision tier reads
                        # via `PageStore`.
                        status = await asyncio.to_thread(
                            render_to_cache,
                            bulletin,
                            page,
                            str(bctx.config.pdf_dir),
                            str(renders_dir),
                        )
                        img = read_cached_image(renders_dir, bulletin, page)
                        if (
                            status == "skipped" or img is None
                        ):  # PDF missing → leave the page
                            agg["skipped"] += 1
                            return
                        async with bctx.llm_sem, bctx.unit_ctx() as ctx:
                            try:
                                res = await vision_scan_page(
                                    ctx, img, bulletin=bulletin
                                )
                                res.vision_rescanned = (
                                    True  # per-page marker → resume skips it
                                )
                                scans[str(page)] = res.model_dump(mode="json")
                                revised.append(page)
                                agg["revised"] += 1
                            except Exception as e:  # noqa: BLE001 — record, never abort the batch
                                agg["failed"] += 1
                                log.warning(
                                    f"[{self.name}] {bulletin} p{page} failed: "
                                    f"{type(e).__name__}: {e}"
                                )

                    # Only successfully re-read pages get their marker set, so a failed page (e.g.
                    # a transient network error) stays unmarked and retries on the next run.
                    await asyncio.gather(*[rescan_one(p) for p in todo])
                    _atomic_write_json(path, data)
                    _patch_catalog_inline(bctx, bulletin, scans, revised)
                    agg["bulletins"] += 1

                done += 1
                if done % _PROGRESS_EVERY == 0 or done == len(pending):
                    bctx.write_progress(self.name, done, len(pending), agg)
                    log.info(
                        f"[{self.name}] {done}/{len(pending)} | ${bctx.cost:.2f} | {dict(agg)}"
                    )

        await asyncio.gather(*[one(b) for b in pending])
        return dict(agg)


# ---------------------------------------------------------------------------
# Stage: table_merge — link label-less table fragments to their parent block (LLM)
# ---------------------------------------------------------------------------


class TableMergeStage(Stage):
    """Reduction (after `vision_rescan`, before `toc`): for each table block with NO row and
    NO column labels — the validated genuine-continuation signal (~50 corpus-wide; every
    self-contained "(Continued)" reprint restates its labels and is skipped) — one flash call
    picks the parent block it belongs to, and the scan file is annotated in place
    (`table_merge.apply_merge`: `extra_pages` on the parent, `merged_into` on the fragment;
    lossless, nothing deleted). Candidates within a bulletin are judged sequentially in page
    order so a chained fragment links to the already-resolved root, never to a sibling
    fragment; bulletins run in parallel.

    Skips bulletins still carrying legacy page-level `continuation_pages` (un-merge them
    first — `scripts/unmerge_rescan.py`). Resume: a successfully processed bulletin is marked
    `"table_merged"`; a bulletin with any failed call stays unmarked and retries next run
    (already-applied merges are skipped via their `merged_into`/`extra_pages` annotations).
    When the bulletin's catalog file already exists (a retrofit over a built corpus), each
    revised page's catalog row is patched inline, same as `vision_rescan`."""

    name = "table_merge"
    workers = 16

    async def run(self, bctx: BuildContext) -> dict:
        pending: list[str] = []
        legacy: list[str] = []
        for b in bctx.bulletins:
            path = bctx.paths.bulletin_file(_SCANS_SUBDIR, b)
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            if data.get("table_merged"):
                continue
            scans = data["scans"]
            if any(s.get("continuation_pages") for s in scans.values()):
                legacy.append(b)
                continue
            if find_candidates(scans):
                pending.append(b)
        if legacy:
            log.warning(
                f"[{self.name}] {len(legacy)} bulletin(s) still page-merged — skipped; "
                f"run scripts/unmerge_rescan.py first (e.g. {legacy[:5]})"
            )
        log.info(f"[{self.name}] {len(pending)} bulletin(s) with merge candidates")
        if not pending:
            return {"legacy_skipped": len(legacy)} if legacy else {}

        sem = asyncio.Semaphore(self.workers)
        agg: dict[str, int] = defaultdict(int)
        done = 0

        async def one(bulletin: str) -> None:
            nonlocal done
            async with sem:
                path = bctx.paths.bulletin_file(_SCANS_SUBDIR, bulletin)
                data = json.loads(path.read_text())
                scans: dict[str, dict] = data["scans"]
                texts = chop_bulletin(
                    bulletin, parsed_json_dir=bctx.config.parsed_json_dir
                )
                revised: set[int] = set()
                failed = False
                # Sequential within the bulletin: each applied merge updates `scans`, so the
                # next candidate's parent listing excludes already-merged fragments.
                for page, bi in find_candidates(scans):
                    async with bctx.llm_sem, bctx.unit_ctx() as ctx:
                        try:
                            parent = await resolve_parent(
                                ctx, bulletin, scans, page, bi, texts.get(page, "")
                            )
                        except Exception as e:  # noqa: BLE001 — record, never abort the batch
                            failed = True
                            agg["failed"] += 1
                            log.warning(
                                f"[{self.name}] {bulletin} p{page}#{bi} failed: "
                                f"{type(e).__name__}: {e}"
                            )
                            continue
                    if parent is None:
                        agg["standalone"] += 1
                        continue
                    apply_merge(scans, (page, bi), parent)
                    revised.update({page, parent[0]})
                    agg["merged"] += 1
                if not failed:
                    data["table_merged"] = True
                _atomic_write_json(path, data)
                _patch_catalog_inline(bctx, bulletin, scans, sorted(revised))
                agg["bulletins"] += 1
                done += 1
                if done % _PROGRESS_EVERY == 0 or done == len(pending):
                    bctx.write_progress(self.name, done, len(pending), agg)
                    log.info(
                        f"[{self.name}] {done}/{len(pending)} | ${bctx.cost:.2f} | {dict(agg)}"
                    )

        await asyncio.gather(*[one(b) for b in pending])
        if legacy:
            agg["legacy_skipped"] = len(legacy)
        return dict(agg)


# ---------------------------------------------------------------------------
# Stage: toc (pass 3a) — build (coalesce) + extraction (LLM, with is_toc filter)
# ---------------------------------------------------------------------------


class TocStage(BulletinStage):
    name = "toc"
    out_subdir = _TOC_SUBDIR
    workers = (
        64  # one quick call per bulletin → bulletin-level concurrency is the throttle
    )

    async def process(
        self, bctx: BuildContext, bulletin: str
    ) -> tuple[dict | None, dict]:
        scan_path = bctx.paths.bulletin_file(_SCANS_SUBDIR, bulletin)
        if not scan_path.exists():
            return (
                None,
                {},
            )  # scan stage hasn't produced this bulletin (e.g. stopped early)
        data = json.loads(scan_path.read_text())

        # ToC build: coalesce the scan's `toc` tags into contiguous ranges, then read
        # ONLY those pages' text (incl. a single bridged interstitial). No heuristics —
        # the extractor's `is_toc` flag does the filtering.
        ranges = coalesce_toc_ranges(
            sorted(
                int(p) for p, s in data["scans"].items() if s.get("page_role") == "toc"
            )
        )
        cand = sorted({p for a, z in ranges for p in range(a, z + 1)})
        toc_texts: dict[int, str] = {}
        if cand:  # only touch the parsed JSON when there's a candidate ToC to read
            elems = page_elements(
                bulletin, base_dir=bctx.config.parsed_json_dir, fill_gaps=True
            )
            toc_texts = {
                p: txt
                for p in cand
                if (txt := elements_to_text(elems.get(p, []))).strip()
            }

        # A ToC-less issue (no genuine ToC page) is left `is_toc=False` here; the
        # `reconstruct_toc` stage fills it from its section-divider pages once every real ToC
        # is built (it needs neighbors' ToCs as reference). See `ReconstructTocStage`.
        async with bctx.unit_ctx() as ctx:
            try:
                hierarchy = await outline_issue(
                    ctx, bulletin=bulletin, toc_texts=toc_texts
                )
            except Exception as e:  # noqa: BLE001 — record, never abort the batch
                log.warning(f"[toc] {bulletin} failed: {type(e).__name__}: {e}")
                return None, {"failed": 1}

        payload = {
            "bulletin": bulletin,
            "toc_ranges": [list(r) for r in ranges],
            "n_toc_pages": len(toc_texts),
            "toc_pages": list(toc_texts.keys()),
            **hierarchy.model_dump(),
        }
        return payload, {
            "with_toc": int(hierarchy.is_toc),
            "flagged_bogus": int(bool(toc_texts) and not hierarchy.is_toc),
            "chapters": len(hierarchy.chapters),
        }


# ---------------------------------------------------------------------------
# Stage: place (pass 3b) — file content pages under chapters (plain Python, no LLM)
# ---------------------------------------------------------------------------


class PlaceStage(BulletinStage):
    name = "place"
    out_subdir = _PLACE_SUBDIR
    workers = (
        64  # pure Python (no LLM) → bulletin-level concurrency is the only throttle
    )

    async def process(
        self, bctx: BuildContext, bulletin: str
    ) -> tuple[dict | None, dict]:
        toc_path = bctx.paths.bulletin_file(_TOC_SUBDIR, bulletin)
        scan_path = bctx.paths.bulletin_file(_SCANS_SUBDIR, bulletin)
        if not (toc_path.exists() and scan_path.exists()):
            return None, {}  # an upstream stage hasn't produced this bulletin yet

        hierarchy = TocHierarchy.model_validate_json(toc_path.read_text())
        scans = {
            int(p): PageScan.model_validate(s)
            for p, s in json.loads(scan_path.read_text())["scans"].items()
        }
        assign, stats = place_pages(hierarchy, scans)

        # Gather row in the exact shape `merge_chapters` consumes — chapters derived
        # from the placement so names + page counts always agree with `assign`.
        counts = Counter(assign.values())
        children = {c.name: c.children for c in hierarchy.chapters}
        chapters = [
            {"name": name, "children": children.get(name, []), "n_pages": n}
            for name, n in counts.items()
        ]
        payload = {
            "bulletin": bulletin,
            "chapters": chapters,
            "assign": {str(p): name for p, name in sorted(assign.items())},
            "stats": stats,
        }
        return payload, {
            "n_filed": stats["n_filed"],
            "n_unfiled": stats["n_unfiled"],
            "no_chapters": int(not chapters),
        }


# ---------------------------------------------------------------------------
# Stage: catalog — the slim query-facing per-page rows (plain Python, no LLM)
# ---------------------------------------------------------------------------


def _catalog_row(bulletin: str, page: int, scan: dict) -> PageCatalogRow | None:
    """Project one page's scan dict into its shipped catalog row, or None when the page
    is not `content` (only content pages ship). The single source of truth for the
    scan → catalog projection, shared by `CatalogStage` and `vision_rescan`'s inline patch."""
    if scan.get("page_role") != "content":
        return None
    return PageCatalogRow(
        bulletin=bulletin,
        page=page,
        # Fragments the table_merge pass linked to a parent block don't ship — the parent's
        # row covers them (its block carries the fragment's page in `extra_pages`).
        content_blocks=[
            b for b in scan.get("blocks", []) if b.get("merged_into") is None
        ],
        date_interval=scan.get("date_interval"),
        continuation_pages=scan.get("continuation_pages", []),
    )


class CatalogStage(BulletinStage):
    name = "catalog"
    out_subdir = _CATALOG_SUBDIR
    out_ext = "jsonl"  # the query path globs `catalog/*.jsonl`, one row per line
    workers = 64

    async def process(
        self, bctx: BuildContext, bulletin: str
    ) -> tuple[dict | None, dict]:
        scan_path = bctx.paths.bulletin_file(_SCANS_SUBDIR, bulletin)
        if not scan_path.exists():
            return None, {}
        scans = json.loads(scan_path.read_text())["scans"]
        rows = [
            r
            for p, s in sorted(scans.items(), key=lambda kv: int(kv[0]))
            if (r := _catalog_row(bulletin, int(p), s)) is not None
        ]
        return {"lines": [r.to_json() for r in rows]}, {"rows": len(rows)}

    def serialize(self, payload: dict) -> str:
        return "\n".join(payload["lines"])


# ---------------------------------------------------------------------------
# Stage: page_store — per-row member texts + figure descriptors (plain Python, no LLM)
# ---------------------------------------------------------------------------


def _figure_note(elems: dict[int, list[dict]], pages: list[int]) -> str:
    """A figure heads-up across `pages`, or "" when none carry a figure. Figures' plotted
    data is absent from the parsed text, so the text tier appends this so it can defer to the
    vision tier instead of scraping a value from prose."""
    n = 0
    headings: list[str] = []
    for p in pages:
        els = elems.get(p, [])
        n += sum(1 for e in els if e.get("type") == "figure")
        headings += [
            e["content"].strip()
            for e in els
            if e.get("type") in ("title", "section_header") and e.get("content")
        ]
    if n == 0:
        return ""
    headers = "; ".join(dict.fromkeys(headings)) or "(untitled)"
    return (
        f"[This page has {n} figure(s)/chart(s) (headings: {headers}) whose plotted data is NOT "
        f"in the text above. If the value you need appears only in a chart, return [] so the "
        f"vision tier can read it.]"
    )


class PageStoreStage(BulletinStage):
    """Build the page store (`pages/<bulletin>.json` = `{page: text}`): per catalog row, the
    text the store serves — its member pages' JSON text (via `elements_to_text`, the same
    chopping the scan uses) joined, with a figure note appended. `row.member_pages` is just
    the row's own page now that nothing folds continuations, so the store is a dumb
    page→text map. No LLM, no rendering — images render on demand in `PageStore`."""

    name = "page_store"
    out_subdir = PAGES_SUBDIR
    workers = 32  # pure Python; reads parsed JSON (LRU-cached doc load) per bulletin

    async def process(
        self, bctx: BuildContext, bulletin: str
    ) -> tuple[dict | None, dict]:
        catalog_path = bctx.paths.bulletin_file(_CATALOG_SUBDIR, bulletin, "jsonl")
        if not catalog_path.exists():
            return None, {}  # catalog stage hasn't produced this bulletin yet
        rows = [
            PageCatalogRow.from_json(line)
            for line in catalog_path.read_text().splitlines()
            if line
        ]
        # One parsed-JSON load; derive both the per-page text and figure notes from it
        # (`elements_to_text` is exactly what `chop_bulletin` applies).
        elems = page_elements(
            bulletin, base_dir=bctx.config.parsed_json_dir, fill_gaps=True
        )
        texts = {p: elements_to_text(e) for p, e in elems.items()}
        entries: dict[str, str] = {}
        for row in rows:
            body = "\n\n".join(t for p in row.member_pages if (t := texts.get(p, "")))
            note = _figure_note(elems, row.member_pages)
            entries[str(row.page)] = "\n\n".join(filter(None, [body, note]))
        return {"entries": entries}, {"rows": len(rows)}

    def serialize(self, payload: dict) -> str:
        return json.dumps(payload["entries"], ensure_ascii=False)


# ---------------------------------------------------------------------------
# Driver + entry point
# ---------------------------------------------------------------------------


def _page_views(scans: dict[str, dict]) -> list[dict]:
    """Per-page projection for ToC reconstruction — every page's scan signals (role, printed
    label, block titles/summaries), in physical order. All roles are kept: a section-divider
    page (a chapter name on an otherwise empty page) is tagged `non_content` carrying one prose
    block whose title is the chapter name."""
    out: list[dict] = []
    for p in sorted(scans, key=int):
        s = scans[p]
        out.append(
            {
                "page": int(p),
                "role": s["page_role"],
                "printed_page": s.get("printed_page"),
                "blocks": [
                    {"title": b.get("title"), "summary": b.get("summary")}
                    for b in s.get("blocks", [])
                ],
            }
        )
    return out


def _reference_chapters(
    real_tocs: dict[str, list[str]], target: str, *, k: int = 1
) -> list[str]:
    """Top-level chapter names from the `k` nearest real-ToC issues before and after `target`
    (deduped, order preserved) — the reconstruction's naming + coverage reference."""
    before = sorted(b for b in real_tocs if b < target)[-k:]
    after = sorted(b for b in real_tocs if b > target)[:k]
    return list(dict.fromkeys(name for b in before + after for name in real_tocs[b]))


class ReconstructTocStage(Stage):
    """Reduction (after `toc`, before `place`): fill each ToC-less issue's outline from its
    section-divider pages, using neighboring issues' real ToCs as the naming + coverage
    reference. Not per-bulletin — it needs every real ToC built first (for reference), and it
    overwrites the `is_toc=False` toc artifacts in place. Resume skips any already
    reconstructed (`source == "reconstructed"`)."""

    name = "reconstruct_toc"

    async def run(self, bctx: BuildContext) -> dict:
        arts: dict[str, dict] = {}
        for b in bctx.bulletins:
            path = bctx.paths.bulletin_file(_TOC_SUBDIR, b)
            if path.exists():
                arts[b] = json.loads(path.read_text())
        # Reference pool: top-level chapters of genuine (non-reconstructed) ToCs only.
        real_tocs = {
            b: [c["name"] for c in a.get("chapters", [])]
            for b, a in arts.items()
            if a.get("is_toc") and a.get("source", "toc") == "toc"
        }
        pending = [
            b
            for b, a in arts.items()
            if not a.get("is_toc") and a.get("source") != "reconstructed"
        ]
        if not pending:
            return {}

        agg: dict[str, int] = defaultdict(int)

        async def one(bulletin: str) -> None:
            scan_path = bctx.paths.bulletin_file(_SCANS_SUBDIR, bulletin)
            if not scan_path.exists():
                return
            views = _page_views(json.loads(scan_path.read_text())["scans"])
            reference = _reference_chapters(real_tocs, bulletin)
            async with bctx.llm_sem, bctx.unit_ctx() as ctx:
                try:
                    hierarchy = await reconstruct_outline(
                        ctx, bulletin=bulletin, page_views=views, reference=reference
                    )
                except Exception as e:  # noqa: BLE001 — record, never abort the batch
                    log.warning(
                        f"[{self.name}] {bulletin} failed: {type(e).__name__}: {e}"
                    )
                    agg["failed"] += 1
                    return
            # Overwrite the toc artifact in place, keeping its non-hierarchy fields.
            _atomic_write_json(
                bctx.paths.bulletin_file(_TOC_SUBDIR, bulletin),
                {**arts[bulletin], **hierarchy.model_dump()},
            )
            agg["reconstructed"] += int(hierarchy.is_toc)
            agg["empty"] += int(not hierarchy.is_toc)
            agg["chapters"] += len(hierarchy.chapters)

        await asyncio.gather(*[one(b) for b in pending])
        return dict(agg)


def _gather_all(bctx: BuildContext) -> dict:
    """Load every placed bulletin (`build/place/<b>.json`) into the gather dict the era
    merge consumes: `{bulletin: {"chapters": [...], "assign": {...}}}`."""
    gather: dict[str, dict] = {}
    for b in bctx.bulletins:
        p = bctx.paths.bulletin_file(_PLACE_SUBDIR, b)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        gather[b] = {"chapters": d["chapters"], "assign": d["assign"]}
    return gather


def _gather_titles(bctx: BuildContext) -> dict[tuple[str, int], list[str]]:
    """`{(bulletin, page): [table titles]}` from every catalog row — the raw material the era
    merge's describe pass samples to summarize each chapter's description/examples."""
    titles: dict[tuple[str, int], list[str]] = {}
    for b in bctx.bulletins:
        p = bctx.paths.bulletin_file(_CATALOG_SUBDIR, b, "jsonl")
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line:
                continue
            row = PageCatalogRow.from_json(line)
            if ts := [blk.title for blk in row.content_blocks if blk.title]:
                titles[(b, row.page)] = ts
    return titles


class EraMergeStage(Stage):
    """Reduction (last): gather all placements → segment the timeline into eras → build each
    era's canonical table of contents → write `concept_tree.json`. Resume = skip when the file
    already exists; delete it to rebuild after adding or re-placing bulletins."""

    name = "era_merge"

    async def run(self, bctx: BuildContext) -> dict:
        out = bctx.paths.root / TREE_FILE
        if out.exists():
            log.info(f"[{self.name}] {out.name} exists, skipping")
            return {}
        gather = _gather_all(bctx)
        if not gather:
            log.warning(
                f"[{self.name}] no placement rows found; run the place stage first"
            )
            return {}
        titles = _gather_titles(bctx)
        async with bctx.unit_ctx() as ctx:
            tree = await build_concept_tree(ctx, gather, titles, sem=bctx.llm_sem)
        _atomic_write_json(out, tree)
        return {"eras": len(tree["eras"]), "bulletins": len(gather)}


# The pipeline, in order. `run_build` calls each stage's `run` and logs its stats; per-bulletin
# stages (`BulletinStage`) and the reductions share the same `Stage.run` interface.
PIPELINE: tuple[Stage, ...] = (
    ScanStage(),
    VisionRescanStage(),
    TableMergeStage(),
    TocStage(),
    ReconstructTocStage(),
    PlaceStage(),
    CatalogStage(),
    PageStoreStage(),
    EraMergeStage(),
)


async def run_build(bctx: BuildContext) -> None:
    """Run every stage in `PIPELINE`, in order, on one event loop."""
    for stage in PIPELINE:
        agg = await stage.run(bctx)
        log.info(
            f"[{stage.name}] complete {dict(agg)} | cumulative ${bctx.cost:.2f} "
            f"(in={bctx.in_tok:,} out={bctx.out_tok:,})"
        )


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    ap = argparse.ArgumentParser(
        description="Treasury Bulletin page-index build (end-to-end, all stages)."
    )
    ap.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
        help="Build folder for all intermediate artifacts (default: build/).",
    )
    ap.add_argument(
        "--bulletins",
        type=str,
        default=None,
        help="Comma-separated YYYY-MM allowlist for dev (default: whole corpus).",
    )
    ap.add_argument(
        "--concurrency", type=int, default=128, help="Max concurrent LLM calls."
    )
    args = ap.parse_args()

    # Model + per-model RPM/TPM are read from SkunkConfig and the llm_client rate
    # limiter (env SKUNK_LLM_MODEL, SKUNK_MODEL_RPM/TPM, SKUNK_LLM_RPM) — the build
    # doesn't override them.
    configure_obs()
    config = SkunkConfig.from_env()
    only = (
        {b.strip() for b in args.bulletins.split(",") if b.strip()}
        if args.bulletins
        else None
    )
    bulletins = discover_bulletins(config.pdf_dir, only=only)
    if not bulletins:
        print("no bulletins discovered", file=sys.stderr)
        return 1

    overrides_path = Path(config.prompt_overrides_path)
    overrides = load_prompt_overrides(overrides_path) if overrides_path.exists() else ()

    bctx = BuildContext(
        config=config,
        paths=BuildPaths(args.build_dir),
        bulletins=bulletins,
        overrides=overrides,
        llm_concurrency=args.concurrency,
    )

    log.info(
        f"build → {args.build_dir}  corpus={CORPUS_NAME}  model={config.llm_model}  "
        f"concurrency={args.concurrency}"
    )
    log.info(
        f"{len(bulletins)} bulletins; pipeline: {' → '.join(s.name for s in PIPELINE)}"
    )

    asyncio.run(run_build(bctx))

    wall = time.perf_counter() - bctx._t0
    print("\n=== BUILD COMPLETE ===")
    print(f"tokens:  in={bctx.in_tok:,}  out={bctx.out_tok:,}")
    print(
        f"cost:    ${bctx.cost:.2f}  (flash @ ${_FLASH_IN * 1e6:.2f}/${_FLASH_OUT * 1e6:.2f} per 1M in/out)"
    )
    print(f"wall:    {wall / 60:.1f} min ({wall:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
