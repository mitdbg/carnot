"""Palimpzest-style two-stage cascade semantic filter.

For each sampled dev UID and each `RetrieveBranch` in its plan:

  Stage 0 — production retrieve:
      L1 chapter pick → date filter → branch survivors.

  Stage A — COARSE filter (cheap, aggressive prune):
      Inputs: page METADATA only (titles, column_headers, dates,
      keywords). No full text.
      Output: bool only.
      Batch size: 20 pages per LLM call.

  Stage B — FINE filter (precise, on stage-A survivors only):
      Inputs: full page plain text + structural anchors.
      Output: one-sentence justification BEFORE the bool, so the
      model commits its reasoning on paper before deciding.
      Batch size: 20 pages per LLM call.

  UID-level kept set is the UNION of per-branch post-stage-B kept
  sets (mirrors production retrieve, which unions branches).

Per-stage metrics (n_calls, in/out tokens, latency, page counts) are
recorded separately so we can attribute cost to each stage.

Held-out test UIDs are excluded by default (see CLAUDE.md).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_load_env(REPO_ROOT / ".env")

from skunk.common import LLMClient  # noqa: E402
from skunk.config import SkunkConfig  # noqa: E402
from skunk.page_index import default_profile  # noqa: E402
from skunk.page_index.corpora.treasury._extract import _page_plain_text  # noqa: E402
from skunk.page_index.pdf import (  # noqa: E402
    parsed_json_dir, pdf_dir_from_env, pdf_path_for, read_page_elements,
)
from skunk.page_index.retrieve_probe import (  # noqa: E402
    load_catalog, load_concept_tree, one_shot_parent_chapter_retrieve,
)
from skunk.page_index.schema import PageCatalogRow  # noqa: E402
from skunk.plan import Plan, RetrieveBranch  # noqa: E402


_PERIOD = default_profile().period_parser
_TEST_SET_PATH = REPO_ROOT / "eval" / "test_set_uids.json"


def _load_test_set_uids() -> set[str]:
    if not _TEST_SET_PATH.exists():
        return set()
    return set(json.loads(_TEST_SET_PATH.read_text()).get("uids", []))


# ---------------------------------------------------------------------------
# Benchmark loader
# ---------------------------------------------------------------------------

_MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
           "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
           "november": 11, "december": 12}
_URL_RE = re.compile(
    r"/(?P<month>january|february|march|april|may|june|july|august|"
    r"september|october|november|december)"
    r"-(?P<year>\d{4})[^?]*\?page=(?P<page>\d+)", re.IGNORECASE,
)


def _parse_source_docs(text: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    if not isinstance(text, str):
        return out
    for m in _URL_RE.finditer(text):
        mn = _MONTHS[m.group("month").lower()]
        out.append((f"{m.group('year')}-{mn:02d}", int(m.group("page"))))
    return out


def _load_benchmark(csv_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            out[r["uid"]] = {
                "question": r["question"],
                "goldens": _parse_source_docs(r.get("source_docs", "")),
            }
    return out


# ---------------------------------------------------------------------------
# Page-text loading (lazy, lru-cached)
# ---------------------------------------------------------------------------

_PARSED_DIR = parsed_json_dir()
_PDF_DIR = pdf_dir_from_env()


@lru_cache(maxsize=1024)
def _page_text_cached(bulletin: str, page: int, max_chars: int) -> str:
    try:
        elements = read_page_elements(
            pdf_path_for(bulletin, _PDF_DIR), parsed_dir=_PARSED_DIR,
        )
    except FileNotFoundError:
        return ""
    text = _page_plain_text(elements.get(page, []))
    if len(text) > max_chars:
        return text[:max_chars] + " …[TRUNCATED]"
    return text


def _page_meta_block(row: PageCatalogRow) -> dict:
    """Coarse-stage input: metadata only, no full text."""
    titles: list[str] = []
    cols: list[str] = []
    rows: list[str] = []
    for b in row.content_blocks:
        if b.title:
            titles.append(b.title)
        cols.extend(b.column_headers)
        rows.extend(b.row_headers_sample)
    return {
        "bulletin": row.bulletin,
        "page": row.page,
        "titles": titles[:4],
        "column_headers": cols[:24],
        "row_labels_sample": rows[:24],
        "dates": row.dates[:12],
        "keywords": row.keywords[:10],
    }


def _page_full_block(row: PageCatalogRow, *, max_chars: int) -> dict:
    """Fine-stage input: full page plain text plus a thin metadata
    shim (titles + headers + dates) for structural anchors."""
    titles: list[str] = []
    cols: list[str] = []
    for b in row.content_blocks:
        if b.title:
            titles.append(b.title)
        cols.extend(b.column_headers)
    return {
        "bulletin": row.bulletin,
        "page": row.page,
        "titles": titles[:4],
        "column_headers": cols[:24],
        "dates": row.dates[:12],
        "text": _page_text_cached(row.bulletin, row.page, max_chars),
    }


# ---------------------------------------------------------------------------
# Stage A — COARSE prompt (metadata only, bool output)
# ---------------------------------------------------------------------------

_COARSE_SYSTEM_PROMPT = """\
You decide, for EACH page in a batch, whether the page reports
values for a SPECIFIC retrieve target: a `key` (the data concept) and
a `period` (the time window). Each batch is up to ~hundreds of
candidate pages from the U.S. Treasury Bulletin.

Input shape (the user message):
  - `key`: short NL phrase describing the data to find
    (e.g. "national defense expenditures",
     "weekly average discount rate for new 91-day bills").
  - `period`: time window — `CY1940`, `FY1991..FY1995`, `2007-01-01`,
    etc. May also be a comma-enumerated list of points.
  - `pages`: a JSON array, each entry
    {"id": <int>, "bulletin": "<YYYY-MM>", "page": <int>,
     "titles": [...], "column_headers": [...],
     "row_labels_sample": [...], "dates": [...], "keywords": [...]}.

Output: a SINGLE JSON object (no prose, no fences, no justification):

  {"decisions": [{"id": <int>, "relevant": <true|false>}, ...]}

One entry per input page, in the same order. Use the input `id`.
Do NOT include a `reason` field — just the boolean.

## Decision rule — be PRECISE

Default is **relevant=false**. Mark `true` only when the page's
column headers, row labels, title, AND time signal together indicate
that the page reports the named `key` for a time inside (or that is
a plausible retrospective reporting issue for) `period`.

Check three things per page:

1. **Subject match.** The page's title / column headers / keywords
   must name the SUBJECT in `key`. Topic-adjacent is not enough —
   "Capital movements" is different from "Foreign currency
   positions" even though both involve money flowing between
   countries. If `key` names a specific security, program, or
   account, the page must mention that specific entity.

2. **Time match.** The page's `dates` field (or its bulletin month
   for retrospective tables) must intersect `period`. A page whose
   only dates are years before / after `period` is not relevant
   unless its bulletin month is the conventional issue that reports
   data for `period` (typically the month immediately after the
   period closes).

3. **Granularity match.** A monthly table cannot answer a daily-
   resolution `key`; a country-level aggregate cannot answer a state-
   level `key`; an annual snapshot cannot answer a weekly-series
   question. If the page reports a coarser or finer breakdown than
   `key` implies, reject.

Always reject pages that are tables of contents, chapter covers,
indexes, introductory prose, or pure boilerplate (no numeric tables).
"""


# ---------------------------------------------------------------------------
# Stage B — FINE prompt (full text, justification-then-bool)
# ---------------------------------------------------------------------------

_FINE_SYSTEM_PROMPT = """\
You are STAGE B of a two-stage cascading filter. Stage A pruned
obvious metadata mismatches; you now see each page's full plain text
plus structural anchors. Your job is to emit `relevant=true` ONLY
when you can point at a concrete numeric value or row in the page
text that IS the answer to the question, or IS one of the input
values the answer is built from.

The downstream extractor will read the page next. We want the final
candidate set to be ≤ 2× the size of the actual gold set per query.
If you cannot quote a specific value cell from the page text that
matches the retrieve target, REJECT.

Input (user message):
  - `key`: short NL phrase describing the data to find
    (e.g. "adjusted price for 2-3/8% U.S Treasury Inflation-Protected
    Security").
  - `period`: time window — `CY1940`, `FY1991..FY1995`, `2007-01-01`,
    or a comma-enumerated list.
  - `pages`: JSON array, each entry
    {"id": <int>, "bulletin": "<YYYY-MM>", "page": <int>,
     "titles": [...], "column_headers": [...], "dates": [...],
     "text": "<page plain text, may be truncated>"}.

Output: a SINGLE JSON object (no prose, no fences):

  {"decisions": [
     {"id": <int>,
      "value_quote": "<EXACT substring you copied from the page
                      `text` showing the value cell: the row label
                      + the value, e.g. 'Japanese yen ... 12,345' or
                      '2-3/8% TIPS—01/15/17-A ... 99.342280'. Empty
                      string if no such cell exists.>",
      "justification": "<one sentence, ≤25 words, explaining how
                        `value_quote` ties to `key` + `period`.>",
      "relevant": <true|false>},
     ...
  ]}

ORDER MATTERS inside each decision: emit `value_quote`, then
`justification`, then `relevant`.

## Decision rule — REJECT by default

Mark `true` ONLY when ALL of the following hold:

1. **`value_quote` is non-empty AND is a real substring of the
   page `text`.** You must literally copy a row + cell from the
   text. No paraphrasing. No "the page contains values like ...".
   If you cannot find such a substring, `value_quote=""` and
   `relevant=false`.

2. **The quoted cell is for the SPECIFIC entity in `key`.** Same
   currency, same country, same program, same security, same fund,
   same rate series. Topic-adjacent does not qualify ("Federal
   Disability Insurance" ≠ "Old-Age and Survivors Insurance";
   "Japanese yen" ≠ "Swiss franc"; "Capital movements" ≠ "Foreign
   currency positions" unless the row itself shows the specific
   entity).

3. **The cell's time signal matches `period`.** A date next to the
   value, the column header, the row's reporting date, the
   bulletin month for retrospective tables — any of these must
   land inside `period`. A page that has the right subject but
   only for a different time window must be rejected.

4. **The granularity matches `key`.** Annual roll-ups cannot
   answer monthly questions; country aggregates cannot answer
   line-item questions. If the cell is at a coarser or finer
   resolution than `key` asks for, reject.

If any one of (1)–(4) fails, `relevant=false`.

Pages whose text is a table of contents, chapter cover, section
index, front-matter prose, or boilerplate are ALWAYS rejected with
`value_quote=""`.

## Calibration

You should reject the vast majority of pages. A typical batch of
20 will likely yield 0–2 `relevant=true`. If you find yourself
marking >5 of 20 as true, re-read your `value_quote`s — most
likely you're keeping topic-adjacent pages whose quoted cell isn't
actually for the entity in `key`.
"""


# ---------------------------------------------------------------------------
# Prompt builder + response parser
# ---------------------------------------------------------------------------

def _build_batch_user_prompt(
    key: str, period: str | None, pages: list[dict],
) -> str:
    return (
        f"key: {key}\n"
        f"period: {period or 'null'}\n\n"
        f"pages (JSON, {len(pages)} entries):\n"
        f"{json.dumps(pages, ensure_ascii=False, indent=1)}\n\n"
        f"Return decisions for ALL {len(pages)} ids."
    )


def _strip_fences(t: str) -> str:
    t = t.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", t)
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _parse_batch_decisions(text: str) -> dict[int, bool]:
    """Extract `{id: relevant}`. Tolerates an optional `justification`
    field appearing alongside `relevant` in either order. Missing ids
    are absent from the dict; callers default to `True`."""
    t = _strip_fences(text)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        out: dict[int, bool] = {}
        for m in re.finditer(
            r"\"id\"\s*:\s*(\d+).*?\"relevant\"\s*:\s*(true|false)",
            t, re.IGNORECASE | re.DOTALL,
        ):
            out[int(m.group(1))] = m.group(2).lower() == "true"
        return out
    decisions = obj.get("decisions") or []
    out: dict[int, bool] = {}
    for d in decisions:
        try:
            idx = int(d["id"])
        except (KeyError, TypeError, ValueError):
            continue
        out[idx] = bool(d.get("relevant", True))
    return out


# ---------------------------------------------------------------------------
# Per-branch retrieve survivor set (chapter pick + date filter)
# ---------------------------------------------------------------------------

def _branch_survivors(
    branch: RetrieveBranch, idx: int, question: str, llm: LLMClient,
    tree: dict, catalog_index: dict,
) -> list[tuple[str, int]]:
    chap_top, _trace = one_shot_parent_chapter_retrieve(
        tree, question=question, concept=branch.key, period=branch.period,
        llm=llm, catalog_index=catalog_index, retrieve_idx=idx,
    )
    period_intervals = _PERIOD.intervals(branch.period)
    out: list[tuple[str, int]] = []
    for c in chap_top:
        key = (c["bulletin"], int(c["page"]))
        if period_intervals is None:
            out.append(key); continue
        row = catalog_index.get(key)
        if row is None or not row.dates:
            out.append(key); continue
        if _PERIOD.dates_overlap_period(row.dates, period_intervals):
            out.append(key)
    return out


# ---------------------------------------------------------------------------
# Stage runner (one batch group → kept set)
# ---------------------------------------------------------------------------

def _chunk(seq: list, n: int) -> list[list]:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


@dataclasses.dataclass
class StageStats:
    name: str          # "coarse" | "fine"
    pre_size: int
    post_size: int
    n_calls: int
    in_tokens: int
    out_tokens: int
    sum_latency_s: float


def _run_stage(
    name: str, key: str, period: str | None,
    survivors: list[tuple[str, int]], catalog_index: dict,
    page_block_fn, system_prompt: str, llm: LLMClient,
    batch_size: int, workers: int,
) -> tuple[list[tuple[str, int]], StageStats]:
    """Run one filter stage. `page_block_fn(row)` returns the dict the
    LLM sees for that page. Returns (kept_pages, stats)."""
    if not survivors:
        return [], StageStats(name, 0, 0, 0, 0, 0, 0.0)

    blocks: list[dict] = []
    for j, pk in enumerate(survivors):
        blocks.append({"id": j, **page_block_fn(catalog_index[pk])})
    batches = _chunk(blocks, batch_size)

    kept: set[tuple[str, int]] = set()
    n_calls = 0
    in_tok = 0
    out_tok = 0
    sum_lat = 0.0
    lock = threading.Lock()

    def _one_batch(batch_pages: list[dict]):
        user = _build_batch_user_prompt(key, period, batch_pages)
        resp = llm.call(system=system_prompt, user=user, temperature=0.0)
        return (resp, _parse_batch_decisions(resp.text), batch_pages)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_one_batch, b) for b in batches]
        for f in as_completed(futs):
            resp, parsed, batch_pages = f.result()
            with lock:
                n_calls += 1
                in_tok += resp.input_tokens or 0
                out_tok += resp.output_tokens or 0
                sum_lat += resp.latency_s
            for p in batch_pages:
                pk = (p["bulletin"], p["page"])
                # Default missing ids to True (recall-safe on response
                # truncation; happens occasionally on large batches).
                if parsed.get(p["id"], True):
                    kept.add(pk)

    # Preserve input order in returned list.
    ordered = [pk for pk in survivors if pk in kept]
    stats = StageStats(name, len(survivors), len(ordered),
                       n_calls, in_tok, out_tok, sum_lat)
    return ordered, stats


# ---------------------------------------------------------------------------
# Per-UID driver
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class BranchOutcome:
    branch_idx: int
    key: str
    period: str | None
    pre_size: int
    coarse: StageStats
    fine: StageStats
    post_size: int     # = fine.post_size
    # Per-branch page lists so a follow-up run can replay just the fine
    # stage against the same coarse survivors (avoids re-paying for the
    # coarse stage when iterating on the fine prompt).
    survivors: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    coarse_survivors: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    fine_survivors: list[tuple[str, int]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class FilterResult:
    uid: str
    question: str
    goldens: list[tuple[str, int]]
    goldens_in_index: list[tuple[str, int]]
    n_branches: int
    pre_filter_size: int        # union after date filter
    coarse_size: int            # union after coarse stage
    post_filter_size: int       # union after fine stage
    pre_filter_hits: int
    coarse_hits: int
    post_filter_hits: int
    coarse_calls: int
    fine_calls: int
    coarse_in_tokens: int
    coarse_out_tokens: int
    fine_in_tokens: int
    fine_out_tokens: int
    coarse_latency_s: float
    fine_latency_s: float
    wall_s: float
    branches: list[BranchOutcome] = dataclasses.field(default_factory=list)
    dropped_goldens_coarse: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    dropped_goldens_fine: list[tuple[str, int]] = dataclasses.field(default_factory=list)


def _run_one_uid(
    uid: str, rec: dict, plan_json: str, tree: dict, catalog_index: dict,
    indexed: set[tuple[str, int]], llm_retrieve: LLMClient,
    llm_filter: LLMClient, filter_workers: int, batch_size: int,
    max_page_chars: int, verbose: bool,
    coarse_cache: dict[int, list[tuple[str, int]]] | None = None,
) -> FilterResult:
    t0 = time.monotonic()
    plan = Plan.model_validate_json(plan_json)
    goldens = set(rec["goldens"])
    goldens_in_idx = goldens & indexed
    retrieve_branches = [b for b in plan.branches if isinstance(b, RetrieveBranch)]

    union_pre: set[tuple[str, int]] = set()
    union_coarse: set[tuple[str, int]] = set()
    union_post: set[tuple[str, int]] = set()
    branch_outcomes: list[BranchOutcome] = []

    coarse_calls = fine_calls = 0
    coarse_in = coarse_out = 0
    fine_in = fine_out = 0
    coarse_lat = fine_lat = 0.0

    for bi, br in enumerate(retrieve_branches):
        survivors = _branch_survivors(br, bi, rec["question"], llm_retrieve,
                                      tree, catalog_index)
        survivors_set = set(survivors)
        union_pre.update(survivors_set)
        if verbose:
            print(f"  [{uid}] branch{bi}: key={br.key!r}  "
                  f"period={br.period!r}  → {len(survivors)} survivors",
                  flush=True)

        # Stage A — coarse (metadata only, bool only). When a cache is
        # supplied for this (uid, branch), skip the coarse LLM calls and
        # use the cached survivors directly.
        if coarse_cache is not None and bi in coarse_cache:
            coarse_kept = list(coarse_cache[bi])
            c_stats = StageStats("coarse", len(survivors), len(coarse_kept),
                                 0, 0, 0, 0.0)
            if verbose:
                print(f"    [{uid}] branch{bi} coarse: cached "
                      f"({len(coarse_kept)} survivors)", flush=True)
        else:
            coarse_kept, c_stats = _run_stage(
                "coarse", br.key, br.period, survivors, catalog_index,
                _page_meta_block, _COARSE_SYSTEM_PROMPT,
                llm_filter, batch_size, filter_workers,
            )
        union_coarse.update(coarse_kept)
        coarse_calls += c_stats.n_calls
        coarse_in += c_stats.in_tokens; coarse_out += c_stats.out_tokens
        coarse_lat += c_stats.sum_latency_s
        if verbose:
            print(f"    [{uid}] branch{bi} coarse: {len(survivors)} → "
                  f"{len(coarse_kept)}  ({100*len(coarse_kept)/max(1,len(survivors)):.1f}%)"
                  f"  calls={c_stats.n_calls}", flush=True)

        # Stage B — fine (full text, justification + bool)
        full_block = lambda row: _page_full_block(row, max_chars=max_page_chars)
        fine_kept, f_stats = _run_stage(
            "fine", br.key, br.period, coarse_kept, catalog_index,
            full_block, _FINE_SYSTEM_PROMPT,
            llm_filter, batch_size, filter_workers,
        )
        union_post.update(fine_kept)
        fine_calls += f_stats.n_calls
        fine_in += f_stats.in_tokens; fine_out += f_stats.out_tokens
        fine_lat += f_stats.sum_latency_s

        branch_outcomes.append(BranchOutcome(
            branch_idx=bi, key=br.key, period=br.period,
            pre_size=len(survivors_set),
            coarse=c_stats, fine=f_stats,
            post_size=len(fine_kept),
            survivors=list(survivors), coarse_survivors=list(coarse_kept),
            fine_survivors=list(fine_kept),
        ))
        if verbose:
            kept_gold = sorted(goldens & set(fine_kept))
            print(f"    [{uid}] branch{bi} fine: {len(coarse_kept)} → "
                  f"{len(fine_kept)}  ({100*len(fine_kept)/max(1,len(coarse_kept)):.1f}%)"
                  f"  calls={f_stats.n_calls}  gold-in-kept={kept_gold}",
                  flush=True)

    pre_size = len(union_pre)
    coarse_size = len(union_coarse)
    post_size = len(union_post)
    pre_hits = len(goldens & union_pre)
    coarse_hits = len(goldens & union_coarse)
    post_hits = len(goldens & union_post)
    dropped_coarse = sorted((goldens & union_pre) - union_coarse)
    dropped_fine = sorted((goldens & union_coarse) - union_post)

    return FilterResult(
        uid=uid, question=rec["question"],
        goldens=sorted(goldens), goldens_in_index=sorted(goldens_in_idx),
        n_branches=len(retrieve_branches),
        pre_filter_size=pre_size, coarse_size=coarse_size,
        post_filter_size=post_size,
        pre_filter_hits=pre_hits, coarse_hits=coarse_hits,
        post_filter_hits=post_hits,
        coarse_calls=coarse_calls, fine_calls=fine_calls,
        coarse_in_tokens=coarse_in, coarse_out_tokens=coarse_out,
        fine_in_tokens=fine_in, fine_out_tokens=fine_out,
        coarse_latency_s=coarse_lat, fine_latency_s=fine_lat,
        wall_s=time.monotonic() - t0,
        branches=branch_outcomes,
        dropped_goldens_coarse=dropped_coarse,
        dropped_goldens_fine=dropped_fine,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(
        description="Two-stage cascade semantic filter (coarse → fine).")
    ap.add_argument("--page-index-dir", type=Path,
                    default=REPO_ROOT / "cache" / "page_index_final")
    ap.add_argument("--benchmark", type=Path,
                    default=REPO_ROOT / "data" / "officeqa_pro.csv")
    ap.add_argument("--plan-cache", type=Path,
                    default=REPO_ROOT / "cache" / "retrieve_bench_plans.jsonl")
    ap.add_argument("--filter-model", default="gemini-2.5-flash")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--uids", type=str, default=None)
    ap.add_argument("--filter-workers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--max-page-chars", type=int, default=12_000)
    ap.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "eval" / "reports" / "sem_filter")
    ap.add_argument("--coarse-cache-from", type=Path, default=None,
                    help="Path to a prior results.jsonl. For each UID + "
                         "branch present in the cache, skip the coarse "
                         "stage and use the cached coarse_survivors. "
                         "Cache misses fall back to running coarse.")
    ap.add_argument("--include-test-set", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    tree = load_concept_tree(args.page_index_dir / "concept_tree.json")
    indexed = {(p["bulletin"], p["page"])
               for cdata in tree["chapters"].values()
               for p in cdata.get("pages", [])}
    catalog = {(r.bulletin, r.page): r
               for r in load_catalog(args.page_index_dir / "catalog")}
    print(f"Tree: {len(tree['chapters'])} chapters, {len(indexed):,} pages")
    print(f"Catalog: {len(catalog):,} rows")

    benchmark = _load_benchmark(args.benchmark)
    test_uids = _load_test_set_uids()
    plans = {json.loads(l)["uid"]: json.loads(l)["plan_json"]
             for l in open(args.plan_cache)}

    if args.uids:
        wanted = [u.strip() for u in args.uids.split(",") if u.strip()]
        if not args.include_test_set:
            collide = sorted(set(wanted) & test_uids)
            if collide:
                print(f"[ABORT] --uids names held-out test UIDs: "
                      f"{collide}.  --include-test-set to override.",
                      file=sys.stderr)
                return 2
        sample = wanted
    else:
        eligible = [u for u in benchmark
                    if u in plans
                    and (args.include_test_set or u not in test_uids)]
        rng = random.Random(args.seed)
        sample = sorted(rng.sample(eligible, min(args.n, len(eligible))))
    if not sample:
        print("No UIDs to run.", file=sys.stderr)
        return 2
    print(f"UIDs: {sample}\n")

    cfg_retrieve = SkunkConfig.from_env()
    cfg_filter = dataclasses.replace(cfg_retrieve, llm_model=args.filter_model)
    llm_retrieve = LLMClient(cfg_retrieve)
    llm_filter = LLMClient(cfg_filter)
    print(f"Retrieve model: {cfg_retrieve.llm_model}")
    print(f"Filter model:   {cfg_filter.llm_model}")
    print(f"Stages: coarse (metadata, bool, batch={args.batch_size}) → "
          f"fine (full text ≤{args.max_page_chars}ch, "
          f"justification+bool, batch={args.batch_size})")

    # Load the coarse-survivor cache if provided.
    coarse_cache_by_uid: dict[str, dict[int, list[tuple[str, int]]]] = {}
    if args.coarse_cache_from is not None and args.coarse_cache_from.exists():
        for line in args.coarse_cache_from.open():
            d = json.loads(line)
            uid = d.get("uid")
            if uid is None:
                continue
            by_branch: dict[int, list[tuple[str, int]]] = {}
            for b in d.get("branches", []):
                pages = [tuple(x) for x in b.get("coarse_survivors", [])]
                by_branch[int(b["branch_idx"])] = pages
            coarse_cache_by_uid[uid] = by_branch
        print(f"Loaded coarse-survivor cache for {len(coarse_cache_by_uid)} "
              f"UIDs from {args.coarse_cache_from}\n")
    else:
        print()

    results: list[FilterResult] = []
    for uid in sample:
        if uid not in benchmark or uid not in plans:
            print(f"  [skip] {uid}")
            continue
        rec = benchmark[uid]
        if verbose:
            print(f"=== {uid} ===\n  Q: {rec['question'][:160]}")
        res = _run_one_uid(
            uid, rec, plans[uid], tree, catalog, indexed,
            llm_retrieve, llm_filter,
            args.filter_workers, args.batch_size, args.max_page_chars,
            verbose,
            coarse_cache=coarse_cache_by_uid.get(uid),
        )
        results.append(res)
        coarse_cost = res.coarse_in_tokens*0.30e-6 + res.coarse_out_tokens*2.5e-6
        fine_cost = res.fine_in_tokens*0.30e-6 + res.fine_out_tokens*2.5e-6
        print(f"  → pre={res.pre_filter_size}  "
              f"coarse={res.coarse_size} (gold {res.coarse_hits}/{res.pre_filter_hits})  "
              f"post={res.post_filter_size} (gold {res.post_filter_hits}/{res.coarse_hits})  "
              f"cost=${coarse_cost+fine_cost:.4f} (C${coarse_cost:.3f}+F${fine_cost:.3f})  "
              f"wall={res.wall_s:.1f}s", flush=True)

    rows_path = args.output_dir / "results.jsonl"
    with rows_path.open("w") as f:
        for r in results:
            f.write(json.dumps(dataclasses.asdict(r), default=str))
            f.write("\n")

    print("\n" + "=" * 116)
    hdr = (f"{'UID':10s}  {'br':>3s}  {'pre':>5s}  {'coarse':>6s}  "
           f"{'post':>5s}  {'preH':>4s}  {'coH':>4s}  {'poH':>4s}  "
           f"{'C-calls':>7s}  {'F-calls':>7s}  "
           f"{'$coarse':>8s}  {'$fine':>7s}  {'wall':>7s}")
    print(hdr)
    tot = {"pre": 0, "coarse": 0, "post": 0, "preH": 0, "coH": 0, "poH": 0,
           "Cc": 0, "Fc": 0, "Cin": 0, "Cout": 0, "Fin": 0, "Fout": 0,
           "wall": 0.0}
    for r in results:
        cc = r.coarse_in_tokens*0.30e-6 + r.coarse_out_tokens*2.5e-6
        fc = r.fine_in_tokens*0.30e-6 + r.fine_out_tokens*2.5e-6
        print(f"{r.uid:10s}  {r.n_branches:>3d}  {r.pre_filter_size:>5d}  "
              f"{r.coarse_size:>6d}  {r.post_filter_size:>5d}  "
              f"{r.pre_filter_hits:>4d}  {r.coarse_hits:>4d}  "
              f"{r.post_filter_hits:>4d}  {r.coarse_calls:>7d}  "
              f"{r.fine_calls:>7d}  ${cc:>6.4f}  ${fc:>5.4f}  "
              f"{r.wall_s:>6.1f}s")
        tot["pre"] += r.pre_filter_size; tot["coarse"] += r.coarse_size
        tot["post"] += r.post_filter_size
        tot["preH"] += r.pre_filter_hits; tot["coH"] += r.coarse_hits
        tot["poH"] += r.post_filter_hits
        tot["Cc"] += r.coarse_calls; tot["Fc"] += r.fine_calls
        tot["Cin"] += r.coarse_in_tokens; tot["Cout"] += r.coarse_out_tokens
        tot["Fin"] += r.fine_in_tokens; tot["Fout"] += r.fine_out_tokens
        tot["wall"] += r.wall_s
    tot_cost_C = tot["Cin"]*0.30e-6 + tot["Cout"]*2.5e-6
    tot_cost_F = tot["Fin"]*0.30e-6 + tot["Fout"]*2.5e-6
    print(f"{'TOTAL':10s}  {'-':>3s}  {tot['pre']:>5d}  {tot['coarse']:>6d}  "
          f"{tot['post']:>5d}  {tot['preH']:>4d}  {tot['coH']:>4d}  "
          f"{tot['poH']:>4d}  {tot['Cc']:>7d}  {tot['Fc']:>7d}  "
          f"${tot_cost_C:>6.4f}  ${tot_cost_F:>5.4f}  "
          f"{tot['wall']:>6.1f}s")
    print(f"Combined cost: ${tot_cost_C + tot_cost_F:.4f}")
    print(f"\nRaw per-UID rows → {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
