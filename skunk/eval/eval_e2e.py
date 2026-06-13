"""End-to-end eval. Runs the full pipeline (planner → orchestrator → 4 operators)
per question and writes predicted vs gold answers to a CSV report.

Scoring is intentionally out of scope here — the report is the artifact a
downstream scorer consumes.

Usage
-----
  # All UIDs in the CSV (default) — run dir auto-named under eval/traces/
  python -m eval.eval_e2e --csv data/officeqa_pro.csv

  # Give the run a human-readable label
  python -m eval.eval_e2e --csv data/officeqa_pro.csv --run-name golden_sweep

  # Sample 10 random UIDs
  python -m eval.eval_e2e --csv data/officeqa_pro.csv --sample 10

  # Run the canonical dev set (first 70 non-test UIDs in eval/dev_set_uids.json)
  python -m eval.eval_e2e --csv data/officeqa_pro.csv --dev-set

  # Run only specific UIDs, bypassing retrieve with golden pages
  python -m eval.eval_e2e --csv data/officeqa_pro.csv --uids UID0001,UID0030 --golden

  # Replay a prior run's cached retrieval: skip retrieve, tune extract/compute free
  python -m eval.eval_e2e --csv data/officeqa_pro.csv \
      --retrieval-cache eval/traces/<run>/retrieval_cache.json

All outputs (per-question traces, run.log, report.csv, events.jsonl) land in a
single run directory: eval/traces/<run-name>_<timestamp>/  (gitignored).

`--golden` parses `source_docs?page=N` URLs from --csv and injects them as
PageRefs, so extract/compute run on exactly the pages the benchmark deems
relevant. Use it to measure the extract+compute ceiling without retrieval cost.

Retrieval cache: every real run auto-writes its retrieve output (the deduped page
union per UID) to <run-dir>/retrieval_cache.json. Point `--retrieval-cache <path>`
at one to replay it — the cached pages are injected golden-style (bypassing
retrieve) so you can tune extract/compute without re-paying retrieval. Cached pages
are the retrieve phase's output — i.e. exactly the pages handed to selection/extract
on the original run. `--golden` and `--retrieval-cache` are mutually exclusive
(two page sources).
"""

from __future__ import annotations

import argparse
import csv
import asyncio
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


# .env must be loaded before importing skunk so LLMClient sees the API keys.
_load_env(REPO_ROOT / ".env")

from skunk import (  # noqa: E402
    MissingData,
    Orchestrator,
    PageRef,
    SkunkConfig,
    StepFailed,
    load_prompt_overrides,
)

from skunk.trace import configure_obs  # noqa: E402
from skunk.common import SemPoolEntry  # noqa: E402
from skunk.page_index.query import BlockRef  # noqa: E402

from eval.util import dump_trace  # noqa: E402
from skunk.eval.scoring import SCORER_VERSION, score_correct  # noqa: E402

# ---------------------------------------------------------------------------
# Golden page parsing (source_docs URLs → PageRefs)
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}

_URL_RE = re.compile(
    r"/(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)"
    r"-(?P<year>\d{4})[^?]*\?page=(?P<page>\d+)",
    re.IGNORECASE,
)


def _parse_source_docs(source_docs: str) -> list[PageRef]:
    """Extract every (year, month, page) tuple from a source_docs cell."""
    out: list[PageRef] = []
    if not isinstance(source_docs, str):
        return out
    for m in _URL_RE.finditer(source_docs):
        month_mm = _MONTH_MAP[m.group("month").lower()]
        out.append(
            PageRef(month=f"{m.group('year')}-{month_mm}", page=int(m.group("page")))
        )
    return out


def load_golden(csv_path: str | Path) -> dict[str, list[PageRef]]:
    """Map uid → list[PageRef] for every question in the benchmark CSV."""
    df = pd.read_csv(csv_path)
    return {
        row["uid"]: _parse_source_docs(str(row.get("source_docs", "")))
        for _, row in df.iterrows()
    }


# ---------------------------------------------------------------------------
# Retrieval cache (real retrieve output → JSON → golden-style replay)
# ---------------------------------------------------------------------------
#
# Every real run auto-writes its actual retrieve output to <run-dir>/retrieval_cache.json:
# per UID the sem-filter survivor pool (`sem_pool`) plus the deduped block union (`blocks`).
# A later run with `--retrieval-cache <path>` bypasses the expensive retrieve sweep:
#   - page-index cache (has `sem_pool`): the survivor pool is injected and the selection
#     agent runs over it at runtime — so a replay can iterate on SELECTION, extract, and
#     compute, only retrieve is fixed.
#   - pool-less cache (search-agent / pre-pool): the `blocks` are replayed as final
#     (iterate on extract/compute only).
# Cache JSON per UID:
#   {"blocks": [{"month","page","block_index","members":[...]}...], "sem_pool": [...]}


@dataclass(frozen=True)
class CachedRetrieval:
    """One UID's cached retrieve output: the deduped block union for that UID, plus the
    sem-filter candidate pool (empty for caches written before pools existed) so coverage
    repair can re-select on replay."""

    blocks: list[BlockRef]
    sem_pool: list[SemPoolEntry] = dc_field(default_factory=list)


def _block_to_json(b: BlockRef) -> dict:
    return {
        "month": b.page.month,
        "page": b.page.page,
        "block_index": b.block_index,
        "members": [{"month": m.month, "page": m.page} for m in b.member_refs],
    }


def _block_from_json(e: dict) -> BlockRef:
    """Rebuild a `BlockRef` from cache JSON. `block` (the `ContentBlock`) is left None — extract
    re-resolves it from the catalog by `(page, block_index)`, so it isn't serialized."""
    return BlockRef(
        page=PageRef(month=e["month"], page=e["page"]),
        block_index=e["block_index"],
        member_refs=tuple(
            PageRef(month=m["month"], page=m["page"]) for m in e["members"]
        ),
        block=None,
    )


def _pool_to_json(p: SemPoolEntry) -> dict:
    return {
        **_block_to_json(p.ref),
        "interval": list(p.interval) if p.interval else None,
        "kind": p.kind,
        "title": p.title,
        "summary": p.summary,
        "cols": list(p.cols),
        "rows_tail": list(p.rows_tail),
        "rows": list(p.rows),
    }


def _pool_from_json(e: dict) -> SemPoolEntry:
    return SemPoolEntry(
        ref=_block_from_json(e),
        interval=tuple(e["interval"]) if e.get("interval") else None,  # type: ignore[arg-type]
        kind=e.get("kind", "table"),
        title=e.get("title"),
        summary=e.get("summary"),
        cols=tuple(e.get("cols") or ()),
        rows_tail=tuple(e.get("rows_tail") or ()),
        rows=tuple(e.get("rows") or ()),
    )


def write_retrieval_cache(path: Path, by_uid: dict[str, CachedRetrieval]) -> None:
    """Serialize uid → blocks (+ sem pool when present) to JSON."""
    import json

    payload: dict[str, dict] = {}
    for uid, cr in by_uid.items():
        entry: dict = {"blocks": [_block_to_json(b) for b in cr.blocks]}
        if cr.sem_pool:
            entry["sem_pool"] = [_pool_to_json(p) for p in cr.sem_pool]
        payload[uid] = entry
    path.write_text(json.dumps(payload, indent=2))


def load_retrieval_cache(path: str | Path) -> dict[str, CachedRetrieval]:
    import json

    with open(path) as f:
        payload = json.load(f)
    return {
        uid: CachedRetrieval(
            blocks=[_block_from_json(b) for b in entry["blocks"]],
            sem_pool=[_pool_from_json(p) for p in entry.get("sem_pool", [])],
        )
        for uid, entry in payload.items()
    }


# ---------------------------------------------------------------------------
# Per-question runner
# ---------------------------------------------------------------------------


async def _run_one_question(
    question: str,
    verbose: bool,
    uid: str | None = None,
    golden_pages: list[PageRef] | None = None,
    cached_blocks: list[BlockRef] | None = None,
    cached_sem_pool: list[SemPoolEntry] | None = None,
    trace_path: str | None = None,
    log_path: str | None = None,
) -> dict:
    """Run one question through the Orchestrator and collect a result dict."""
    if golden_pages:
        # Normalize to fresh PageRef instances regardless of input shape.
        golden_pages = [PageRef(month=g.month, page=g.page) for g in golden_pages]
        if verbose:
            print(
                f"[e2e] Golden pages: {len(golden_pages)} → {[str(r) for r in golden_pages]}"
            )

    config = SkunkConfig.from_env()
    config.golden_pages = golden_pages
    # Block-aware replay: inject the cached blocks so extract block-scopes as the live run did,
    # and the cached sem pool so coverage repair can re-select on replay.
    config.cached_blocks = cached_blocks
    config.cached_sem_pool = cached_sem_pool

    overrides_path = Path(config.prompt_overrides_path)
    prompt_overrides = (
        load_prompt_overrides(overrides_path) if overrides_path.exists() else ()
    )

    orch = Orchestrator(
        question,
        uid=uid,
        verbose=verbose,
        log_path=log_path,
        config=config,
        prompt_overrides=prompt_overrides,
    )
    ctx = orch.ctx

    try:
        if verbose:
            print(f"\n[e2e] Planning: {question[:80]}...")

        t0 = time.perf_counter()
        failure: MissingData | StepFailed | None = None
        try:
            await orch.execute()
        except (MissingData, StepFailed) as e:
            # A terminal failure — compute ran out of recovery budget (MissingData)
            # or an operator gave up (StepFailed, e.g. every branch failed). Record
            # it on the result and as a failed row, and keep the batch going.
            # Anything else is an unexpected crash and propagates.
            failure = e
            orch.result.failed = True
            orch.result.failure_reason = (
                f"MissingData: {e.reason}" if isinstance(e, MissingData) else str(e)
            )
        latency_s = time.perf_counter() - t0
        cost_usd = _events_cost(ctx.events)
        result = orch.result
        plan_obj = orch.current_plan

        if verbose:
            if plan_obj is not None:
                print(f"[e2e] Plan: {plan_obj.model_dump_json()}")
            print(f"[e2e] latency_s={latency_s:.3f} cost_usd={cost_usd or '—'}")

        if trace_path is not None:
            plan_dump = plan_obj.model_dump_json() if plan_obj else "(unavailable)"
            dump_trace(
                trace_path,
                uid=uid,
                question=question,
                plan_text=plan_dump,
                golden_pages=golden_pages,
                result=result,
                events=ctx.events,
                model=config.llm_model,
            )

        # The retrieve sweep's deduped blocks + sem pool — empty under golden/replay bypass
        # (retrieve never ran). Surfaced so the harness can cache them for later replay.
        retrieved_blocks = list(orch.retrieved_blocks)
        sem_pool = list(orch.sem_pool)

        if failure is not None:
            return {
                "question": question,
                "answer": None,
                "failed": True,
                "reason": result.failure_reason,
                "latency_s": round(latency_s, 3),
                "cost_usd": cost_usd,
                "retrieved_blocks": retrieved_blocks,
                "sem_pool": sem_pool,
            }
        return {
            "question": question,
            "answer": result.answer,
            "failed": result.failed,
            "reason": result.failure_reason,
            "latency_s": round(latency_s, 3),
            "cost_usd": cost_usd,
            "retrieved_blocks": retrieved_blocks,
            "sem_pool": sem_pool,
        }
    finally:
        ctx.close()


REPORT_FIELDS = [
    "uid",
    # correct / wrong / fail — `correct` per the official cup scorer; `fail` = no answer
    # (uncaught error or MissingData); `wrong` = produced an answer that scored 0. The
    # finer failure-mode triage (compute / retrieval / lookup) the merged report carries
    # is a manual pass over `wrong`/`fail`.
    "category",
    # Free-text annotation. Auto-seeded with the failure reason for `fail` rows; blank
    # otherwise (a human refines it during triage).
    "note",
    # The question text + `gold_answer`/`golden_pages`, mirrored under the trace viewer's
    # expected column names so `eval/trace_viewer/` renders without per-schema patches:
    # `golden_pages` is space-separated `month:page` tokens, and `failed`/`reason` are
    # derived from `category`/`note` for its pass/fail badge.
    "question",
    "predicted",
    "gold_answer",
    "golden_pages",
    "failed",
    "reason",
    # Wall-clock seconds to plan + execute the query, measured in-process (see _run_one_question).
    "latency_s",
    # n_hit/n_gold — retrieved pages (post block-select) that intersect the gold pages.
    "retrieval_recall",
    # USD billed for this UID's generation calls (see PRICES; thinking billed at output).
    "cost_usd",
]

# USD per token (input, output); thinking tokens billed at the output rate. Copied from
# scripts/analyze_trace.py — keep the two in sync if Gemini pricing changes.
PRICES = {
    "gemini-3.5-flash": (1.50e-6, 9.00e-6),  # list price (was 0.30/2.50 — stale gemini-2.5-flash rate)
    "gemini-3-flash-preview": (0.50e-6, 3.00e-6),  # list price (verified ai.google.dev + OpenRouter, 2026-06)
    "gemini-3.1-flash-lite": (0.10e-6, 0.40e-6),
    "gemini-3.1-pro-preview": (2.00e-6, 12.00e-6),
}
_CALL_RE = re.compile(
    r"call call_site=\S+ model=(\S+) .*?in_tok=(\d+) out_tok=(\d+) think_tok=(\S+)"
)


def _events_cost(events: list[dict]) -> str:
    """Sum a UID's generation cost (USD) from its in-memory `call` events (per PRICES;
    thinking billed at the output rate). Reads `ctx.events`, which is always captured
    regardless of `--no-traces`, so cost is available even when no log is written.
    Returns '' when no priced `call` events were seen (e.g. golden/replay bypass, or a
    model absent from PRICES)."""
    cost = 0.0
    saw_call = False
    for evt in events:
        c = _CALL_RE.search(evt.get("message", ""))
        if not c:
            continue
        model, i, o, t = c.group(1), int(c.group(2)), int(c.group(3)), c.group(4)
        t = 0 if t == "None" else int(t)
        price = PRICES.get(model)
        if price is None:
            continue
        pin, pout = price
        cost += i * pin + (o + t) * pout
        saw_call = True
    return f"{cost:.4f}" if saw_call else ""


def _retrieval_recall(retrieved_blocks: list, gold_pages: list[PageRef] | None) -> str:
    """n_hit/n_gold — how many gold pages the retrieve phase actually surfaced (matched on
    month:page across the blocks' member pages). '' when no gold pages are recorded."""
    if not gold_pages:
        return ""
    gold = {(p.month, p.page) for p in gold_pages}
    got = {
        (r.month, r.page)
        for b in retrieved_blocks
        for r in getattr(b, "member_refs", ())
    }
    return f"{len(gold & got)}/{len(gold)}"


# Held-out test set — see CLAUDE.md. Loaded lazily so the file is optional.
_TEST_SET_PATH = REPO_ROOT / "eval" / "test_set_uids.json"

# Dev set — the canonical first-70 non-test UIDs used by --dev-set.
_DEV_SET_PATH = REPO_ROOT / "eval" / "dev_set_uids.json"


def _load_test_set() -> set[str]:
    if not _TEST_SET_PATH.exists():
        return set()
    import json

    with _TEST_SET_PATH.open() as f:
        return set(json.load(f).get("uids", []))


def _load_dev_set() -> list[str]:
    if not _DEV_SET_PATH.exists():
        return []
    import json

    with _DEV_SET_PATH.open() as f:
        return [str(u) for u in json.load(f).get("uids", [])]


def _pick_uids(
    df: pd.DataFrame,
    sample: int | None,
    uids_arg: str | None,
    exclude: set[str] | None = None,
) -> list[str]:
    if uids_arg:
        return [u.strip() for u in uids_arg.split(",") if u.strip()]
    all_uids = [str(u) for u in df["uid"].tolist()]
    if exclude:
        all_uids = [u for u in all_uids if u not in exclude]
    if sample is None:
        return all_uids
    return random.sample(all_uids, min(sample, len(all_uids)))


@dataclass(frozen=True)
class EvalConfig:
    """Settings shared by every UID in one run.

    Built once in `main` after argument parsing, then passed (read-only) to every worker.
    """

    csv_path: str
    df_by_uid: pd.DataFrame
    golden_lookup: dict[str, list[PageRef]] | None
    # Cached real-retrieval output (uid → pages + selected blocks) injected to bypass retrieve
    # on replay (`--retrieval-cache`): pages golden-style, blocks via `config.cached_blocks` so
    # extract block-scopes as the live run did. Mutually exclusive with golden_lookup.
    cached_lookup: dict[str, CachedRetrieval] | None
    # Always-populated golden map for the report's `golden_pages` column (the trace
    # viewer's groundtruth chips), independent of `--golden` injection above.
    golden_report: dict[str, list[PageRef]]
    trace_dir: Path | None
    verbose: bool


async def process_uid(uid: str, cfg: EvalConfig) -> dict | None:
    """Run one UID end-to-end. Returns a row dict, or None if the UID
    is missing from the CSV (skip-with-warning, not fatal). Catches
    and records uncaught exceptions so one runaway UID doesn't kill
    the batch."""
    if uid not in cfg.df_by_uid.index:
        print(f"[e2e] WARNING: {uid!r} not found in {cfg.csv_path}", file=sys.stderr)
        return None

    row = cfg.df_by_uid.loc[uid]
    question = str(row["question"])
    gold_answer = row.get("answer")
    print(f"\n{'=' * 60}\nUID: {uid}\nQ: {question}")

    golden_pages = None
    cached_blocks = None
    cached_sem_pool = None
    if cfg.golden_lookup is not None:
        golden_pages = cfg.golden_lookup.get(uid, [])
        if not golden_pages:
            print(f"[e2e] WARNING: no golden pages for {uid!r}")
    elif cfg.cached_lookup is not None:
        # Replay: inject the cached retrieve output for this UID, bypassing the expensive
        # retrieve sweep. A UID absent from the cache can't be replayed without paying for
        # retrieval, which the cache exists to avoid — skip it loudly.
        cached = cfg.cached_lookup.get(uid)
        if cached is None:
            print(
                f"[e2e] WARNING: {uid!r} not in retrieval cache — skipping "
                f"(replay can't run retrieve)",
                file=sys.stderr,
            )
            return None
        if cached.sem_pool:
            # Survivor cache: inject the sem-filter survivor pool and let the selection
            # agent run over it at runtime — so a cached run can iterate on selection,
            # not just extract/compute. The pool's
            # presence triggers the retrieve bypass (no golden_pages / cached_blocks needed).
            cached_sem_pool = cached.sem_pool
        else:
            # Pool-less cache (search-agent / pre-pool): replay the selected blocks as final,
            # bypassing block selection too. golden_pages (the deduped member-ref union)
            # triggers the bypass.
            cached_blocks = cached.blocks
            seen: set[PageRef] = set()
            golden_pages = []
            for b in cached_blocks:
                for r in b.member_refs:
                    if r not in seen:
                        seen.add(r)
                        golden_pages.append(r)

    trace_path = log_path = None
    if cfg.trace_dir:
        trace_path = str(cfg.trace_dir / f"{uid}.txt")
        log_path = str(cfg.trace_dir / f"{uid}.log")

    try:
        result = await _run_one_question(
            question=question,
            verbose=cfg.verbose,
            golden_pages=golden_pages,
            cached_blocks=cached_blocks,
            cached_sem_pool=cached_sem_pool,
            uid=uid,
            trace_path=trace_path,
            log_path=log_path,
        )
    except Exception as e:
        import traceback as _tb

        tb_str = _tb.format_exc()
        print(f"[e2e] ABORTED UID {uid}: {type(e).__name__}: {e}", file=sys.stderr)
        if trace_path is not None:
            try:
                Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
                Path(trace_path + ".failed").write_text(
                    f"UID: {uid}\nQ: {question}\n\nUncaught exception:\n{tb_str}\n"
                )
            except Exception:
                pass
        result = {
            "question": question,
            "answer": None,
            "failed": True,
            "reason": f"Uncaught: {type(e).__name__}: {e}",
            # The exception escaped before _run_one_question measured the run.
            "latency_s": None,
            "cost_usd": "",
            "retrieved_blocks": [],
        }

    predicted = result["answer"] if not result["failed"] else ""

    # Grade against gold with the official cup scorer at 0.0% rel-err (1/0).
    correct = score_correct(gold_answer, predicted)

    if result["failed"]:
        print(f"[e2e] {uid} FAILED: {result['reason']}")
    else:
        mark = "✓" if correct else "✗"
        print(
            f"[e2e] {uid} Answer: {result['answer']}  [{mark} vs gold: {gold_answer!r}]"
        )

    retrieved_blocks = result.get("retrieved_blocks", [])
    category = "correct" if correct else ("fail" if result["failed"] else "wrong")
    gold_report_pages = cfg.golden_report.get(uid) or []
    return {
        "uid": uid,
        "category": category,
        # Seed `note` with the failure reason so `fail` rows are self-describing; correct/
        # wrong rows start blank for manual triage.
        "note": (result["reason"] or "") if result["failed"] else "",
        # Trace-viewer mirror columns (see REPORT_FIELDS). `golden_pages` uses the viewer's
        # `month:page` token form; `failed`/`reason` feed its pass/fail badge.
        "question": question,
        "predicted": predicted,
        "gold_answer": gold_answer,
        "golden_pages": " ".join(f"{p.month}:{p.page}" for p in gold_report_pages),
        "reason": result.get("reason") or "",
        "retrieval_recall": _retrieval_recall(
            retrieved_blocks, cfg.golden_report.get(uid)
        ),
        # Wall-clock seconds to plan + execute the query, and USD billed for its generation
        # calls — both measured in-process by _run_one_question (no trace log needed, so they
        # are populated under --no-traces too). latency_s is None if the run aborted before
        # timing; cost_usd is '' when no priced calls ran.
        "latency_s": result.get("latency_s"),
        "cost_usd": result.get("cost_usd") or "",
        # Internal helper keys — dropped from the CSV by extrasaction="ignore"; consumed by
        # main() for the accuracy tally and retrieval cache.
        "correct": correct,
        "failed": result["failed"],
        "retrieved_blocks": retrieved_blocks,
        "sem_pool": result.get("sem_pool") or [],
    }


# We parallelize the execution of questions using a thread pool; each worker thread
# runs an event loop (via `asyncio.run`) which allows network I/O (LLM calls) to be
# concurrent within the question. Since each question executes an orchestrator, which
# may execute multiple operators in parallel, this enables us to run questions (and
# to some extent their operators) concurrently.
def _run_uid(uid: str, cfg: EvalConfig) -> dict | None:
    return asyncio.run(process_uid(uid, cfg))


def main() -> None:
    # Force line-buffered stdout/stderr so live operator/LLM events stream to
    # logs and `tail -f` in real time. Without this, Python block-buffers when
    # stdout is redirected to a file (~4–8KB chunks), making the harness look
    # stalled mid-question even though it's working.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore

    parser = argparse.ArgumentParser(
        description="End-to-end OfficeQA eval (all UIDs by default)"
    )
    parser.add_argument("--csv", required=True, help="Path to officeqa_pro.csv")
    parser.add_argument(
        "--run-name",
        default="",
        help="Human-readable label prepended to the auto-timestamped run directory "
        "under eval/traces/ (e.g. 'golden_sweep' → eval/traces/golden_sweep_20260601_153000/). "
        "Defaults to the empty string, giving eval/traces/20260601_153000/.",
    )
    parser.add_argument("--sample", type=int, help="Run a random subset of N UIDs")
    parser.add_argument("--uids", help="Comma-separated UIDs (overrides --sample)")
    parser.add_argument(
        "--dev-set",
        action="store_true",
        help="Run on the canonical dev set in eval/dev_set_uids.json (the first 70 "
        "non-test UIDs). Mutually exclusive with --uids; combine with --sample to "
        "run a random subset of the dev set.",
    )
    parser.add_argument(
        "--golden",
        action="store_true",
        help="Inject golden pages from --csv instead of running retrieve",
    )
    parser.add_argument(
        "--retrieval-cache",
        metavar="PATH",
        help="Replay a prior run's cached retrieve output: inject each UID's cached "
        "pages (golden-style) instead of running retrieve, to iterate on "
        "extract/compute for free. Every real run auto-writes its cache to "
        "<run-dir>/retrieval_cache.json — point this at one. Without an explicit UID "
        "selection, the run pool is restricted to the cache's UIDs. Incompatible "
        "with --golden.",
    )
    parser.add_argument(
        "--no-traces",
        action="store_true",
        help="Disable per-question trace files (run dir still created for report.csv).",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Also echo the live (interleaved) event firehose to stdout. Off by "
        "default — per-question events stream to <run-dir>/traces/<uid>.log instead.",
    )
    parser.add_argument(
        "--include-test-set",
        action="store_true",
        help="Include the held-out test-set UIDs (see CLAUDE.md). Default is to exclude "
        "them — only opt in for a deliberate final-number measurement.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="UID-level concurrency (default: %(default)s). Each worker runs one "
        "_run_one_question independently; the process-wide LLM rate limiter "
        "throttles cross-worker traffic (env: SKUNK_LLM_RPM).",
    )
    args = parser.parse_args()

    # --golden and --retrieval-cache both inject pages (bypassing retrieve); two
    # page sources can't both win.
    if args.golden and args.retrieval_cache:
        print(
            "[e2e] ABORT: --golden and --retrieval-cache are mutually exclusive.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Build the run directory: eval/traces/[<run-name>_]<YYYYMMDD_HHMMSS>/
    import datetime

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = f"{args.run_name}_{ts}" if args.run_name else ts
    run_dir = Path("eval/traces") / run_label
    trace_dir = run_dir / "traces" if not args.no_traces else None
    report_path = run_dir / "report.csv"

    run_dir.mkdir(parents=True, exist_ok=True)
    if trace_dir:
        trace_dir.mkdir(parents=True, exist_ok=True)

    # Configure the unified logging pipeline once for the whole process. When a
    # trace dir is set, also open a run-wide JSONL sink (one structured line per
    # event, tagged with uid/step_idx) alongside the per-question text traces.
    jsonl_path = str(trace_dir / "events.jsonl") if trace_dir else None
    configure_obs(jsonl_path=jsonl_path)

    # Persist stdlib-logging warnings (LLM retry/timeout warnings from
    # `llm_client._retry_call`, third-party libs) to the run dir. These have no
    # per-question ctx so they bypass the event stream; without this sink they
    # exist only on the console and vanish with the terminal.
    import logging

    from skunk.trace import _LineFormatter

    warn_handler = logging.FileHandler(run_dir / "warnings.log")
    warn_handler.setLevel(logging.WARNING)
    warn_handler.setFormatter(_LineFormatter())
    logging.getLogger().addHandler(warn_handler)

    print(f"[e2e] Run directory: {run_dir}")
    print(f"[e2e] Scoring with official cup scorer ({SCORER_VERSION}) at 0.0% rel-err.")

    df = pd.read_csv(args.csv)
    df_by_uid = df.set_index("uid")

    test_set = _load_test_set()

    cached_lookup = (
        load_retrieval_cache(args.retrieval_cache) if args.retrieval_cache else None
    )
    if cached_lookup is not None:
        print(f"[e2e] --retrieval-cache: loaded {len(cached_lookup)} cached UID(s).")

    if args.dev_set and args.uids:
        print(
            "[e2e] ABORT: --dev-set and --uids are mutually exclusive.",
            file=sys.stderr,
        )
        sys.exit(2)

    # ABORT (not warn) when --uids names test UIDs without --include-test-set.
    # Per CLAUDE.md: there is no legitimate reason for an explicit UID list
    # from the command line to hit the test set unless --include-test-set is
    # set. Random samples are filtered silently below (the user didn't ask
    # for those specific UIDs).
    if args.uids and test_set and not args.include_test_set:
        requested = {u.strip() for u in args.uids.split(",") if u.strip()}
        collision = sorted(requested & test_set)
        if collision:
            print(
                f"[e2e] ABORT: --uids names {len(collision)} held-out test-set UID(s): "
                f"{', '.join(collision)}. "
                f"Pass --include-test-set to override (see CLAUDE.md).",
                file=sys.stderr,
            )
            sys.exit(2)

    # Exclude test UIDs from the pool BEFORE sampling so --sample N returns N
    # dev UIDs (rather than N minus however many test UIDs happened to be drawn).
    exclude = test_set if (test_set and not args.include_test_set) else None
    if exclude and not args.uids and not args.dev_set:
        print(
            f"[e2e] Excluded {len(exclude)} held-out test-set UID(s) from the pool. "
            f"Pass --include-test-set to override (see CLAUDE.md).",
            file=sys.stderr,
        )

    if args.dev_set:
        dev_uids = _load_dev_set()
        if not dev_uids:
            print(
                f"[e2e] ABORT: --dev-set given but {_DEV_SET_PATH} is missing or empty.",
                file=sys.stderr,
            )
            sys.exit(2)
        # The dev set is test-set-free by construction; guard anyway so a stale
        # file can never smuggle a held-out UID into a dev run.
        if test_set and not args.include_test_set:
            leaked = sorted(set(dev_uids) & test_set)
            if leaked:
                print(
                    f"[e2e] ABORT: {_DEV_SET_PATH} contains {len(leaked)} held-out "
                    f"test-set UID(s): {', '.join(leaked)}.",
                    file=sys.stderr,
                )
                sys.exit(2)
        uids = dev_uids
        if args.sample:
            uids = random.sample(uids, min(args.sample, len(uids)))
        print(f"[e2e] --dev-set: {len(uids)} of {len(dev_uids)} dev UID(s).")
    else:
        uids = _pick_uids(df, args.sample, args.uids, exclude=exclude)

    # On replay without an explicit UID list, the cache defines the run pool —
    # restrict to its UIDs so we don't skip-warn through every uncached UID. An
    # explicit --uids list is honored as-is (uncached ones skip-warn in process_uid).
    if cached_lookup is not None and not args.uids:
        before = len(uids)
        uids = [u for u in uids if u in cached_lookup]
        print(
            f"[e2e] --retrieval-cache: restricted pool to {len(uids)} cached UID(s) "
            f"(of {before})."
        )

    if test_set and args.include_test_set:
        print(
            f"[e2e] WARNING: --include-test-set is on; the held-out {len(test_set)}-UID test set "
            f"is in play. Only use this for final-number measurement, not iterative tuning.",
            file=sys.stderr,
        )

    sample_note = f" (sample: {args.sample})" if args.sample and not args.uids else ""
    print(
        f"[e2e] Running {len(uids)} UID(s){sample_note} with --workers {args.workers}"
    )

    # `golden_report` is always loaded (for the report's groundtruth column);
    # `golden_lookup` (the injection path) stays gated on the `--golden` ablation.
    golden_report = load_golden(args.csv)
    golden_lookup = golden_report if args.golden else None

    cfg = EvalConfig(
        csv_path=args.csv,
        df_by_uid=df_by_uid,
        golden_lookup=golden_lookup,
        cached_lookup=cached_lookup,
        golden_report=golden_report,
        trace_dir=trace_dir,
        verbose=args.console,
    )

    # `results` is pre-allocated so the output CSV preserves input UID order
    # regardless of completion order.
    results: list[dict | None] = [None] * len(uids)
    if args.workers <= 1:
        for i, uid in enumerate(uids):
            results[i] = _run_uid(uid, cfg)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_uid, uid, cfg): i for i, uid in enumerate(uids)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()

    rows = [r for r in results if r is not None]

    out = report_path
    with out.open("w", newline="", encoding="utf-8") as f:
        # extrasaction="ignore" drops the non-column `retrieved_blocks` key (kept on
        # each row only to build the retrieval cache below).
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Every run auto-caches its real retrieve output for later replay. Golden / replay
    # runs bypass retrieve so they produce no blocks — nothing to write.
    cache_by_uid = {
        r["uid"]: CachedRetrieval(
            blocks=r["retrieved_blocks"], sem_pool=r.get("sem_pool") or []
        )
        for r in rows
        if r.get("retrieved_blocks")
    }
    if cache_by_uid:
        cache_path = run_dir / "retrieval_cache.json"
        write_retrieval_cache(cache_path, cache_by_uid)
        print(
            f"[e2e] Wrote retrieval cache {cache_path} ({len(cache_by_uid)} UID(s))  "
            f"— replay with --retrieval-cache {cache_path}"
        )

    n_total = len(rows)
    n_failed = sum(1 for r in rows if r["failed"])
    print(f"\n[e2e] Wrote {out} ({n_total} rows)")
    print(f"[e2e] {n_total - n_failed}/{n_total} produced an answer")
    if n_total:
        n_correct = sum(1 for r in rows if r["correct"] == 1)
        pct = 100.0 * n_correct / n_total
        print(
            f"[e2e] Accuracy: {n_correct}/{n_total} correct ({pct:.1f}%) "
            f"at 0.0% absolute relative error ({SCORER_VERSION})"
        )


if __name__ == "__main__":
    main()
