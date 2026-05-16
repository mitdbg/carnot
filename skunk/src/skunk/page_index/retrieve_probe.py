"""Retrieval probe over the flat concept tree.

Three entry points are exposed:

* `one_shot_parent_chapter_retrieve` — L1 only. One LLM call picks a
  canonical chapter; every page under it is returned as the prediction.
  Useful as a recall ceiling for the chapter-pick decision.

* `l1_vector_retrieve` — L1 chapter pick + vector cosine top-K within the
  picked chapter. Optionally:
    - `use_period_mask=True` to AND with the symbolic bulletin/year mask
      before the cosine pass,
    - `pre_date_intersect=True` to AND with the strict date-envelope
      intersect (uses `_row_date_envelope` per row),
    - `date_filter_mode="soft"` to post-filter top-K survivors by date
      envelope (with bulletin-window fallback for unparseable rows),
    - `use_llm_dates=True` to have an LLM propose query-side dates from
      the question text instead of using the planner's structured period,
    - `top_n=N` to cap the returned candidate count.

* `leaf_rank_postings` — single-LLM-call ranker over a candidate list.
  Used by `vector_retrieve.retrieve_vector` to rerank vector hits with a
  question-aware LLM pass.

Tree loaded via `load_concept_tree`; catalog via `load_catalog`.
Telemetry: every level appends a `LevelTrace` to its `RetrieveTrace`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from skunk.common import LLMClient

from .period import (
    intervals_overlap, period_to_intervals, verbatim_date_to_intervals,
)
from .schema import PageCatalogRow


# ---------------------------------------------------------------------------
# Telemetry — one LevelTrace per LLM call; one RetrieveTrace per retrieve op
# ---------------------------------------------------------------------------

@dataclass
class LevelTrace:
    """Telemetry for one LLM call inside the walker. When the hierarchical
    walker lands, each level (section_pick, cluster_pick, leaf_rank) emits
    one of these in the order it ran.
    """
    level: str                            # "leaf_rank" today; more later
    input_count: int                      # number of candidate items shown to the LLM
    input_chars: int                      # length of the user message
    input_tokens: int | None = None       # from Gemini usage metadata, if available
    output_chars: int = 0                 # length of the raw model response
    output_tokens: int | None = None
    latency_s: float = 0.0                # wall-clock for the LLM call (incl. retries)
    output_count: int = 0                 # number of items the LLM returned
    prompt_excerpt: str = ""              # first 800 chars of the user message
    response_excerpt: str = ""            # first 800 chars of the raw model response
    steps: list[dict[str, Any]] = field(default_factory=list)  # per-round actions (agentic loops)


@dataclass
class RetrieveTrace:
    uid: str | None
    retrieve_idx: int
    concept: str
    period: str
    catalog_size: int
    prefilter_s: float = 0.0
    candidate_count: int = 0
    levels: list[LevelTrace] = field(default_factory=list)
    top_k: list[dict[str, Any]] = field(default_factory=list)
    total_walk_s: float = 0.0


# ---------------------------------------------------------------------------
# Catalog + prefilter
# ---------------------------------------------------------------------------

def load_catalog(catalog_dir: Path) -> list[PageCatalogRow]:
    rows: list[PageCatalogRow] = []
    for f in sorted(catalog_dir.glob("*.jsonl")):
        for line in f.open():
            rows.append(PageCatalogRow.from_json(line))
    return rows


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", s)
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


# ---------------------------------------------------------------------------
# Hierarchical walker (v0.2): section → cluster → term → leaf
# ---------------------------------------------------------------------------

def load_concept_tree(path: Path) -> dict[str, Any]:
    """Load `concept_tree.json` produced by `skunk.page_index.concept_tree`."""
    return json.loads(Path(path).read_text())


_LEAF_RANK_HIER_SYSTEM = """You are given a list of candidate Treasury Bulletin
pages reached via the concept walk. Each candidate is a TEXT BLOB formatted as:

  === YYYY-MM p<N> ===
  [title]    <verbatim table caption, or "(no title)">
  [columns]  <up to 50 column-header strings, joined by " | ">
  [rows]     <up to 12 sample row-header strings, joined by " | ">
  [dates]    <verbatim date strings on the page, joined by "; ">

Prose pages substitute `[content]` for `[columns]`/`[rows]`, listing the page's
dateless concept phrases.

Pick the top-K pages that best answer the user's question. The blob shows
exactly what each candidate page contains. Use the signals:
  - `[title]` is the AUTHORITATIVE caption — e.g.
      "Table 1.- Status under Limitation, December 31, 1949"
      "Maturity Schedule of Interest-Bearing Public Marketable Securities ...
       Outstanding January 31, 1950"
      "Summary by Months and Calendar Years"
    Any date inside the title is the page's SNAPSHOT date (what the table is
    reporting on). Title wording distinguishes summary vs. detail tables
    ("Summary by Months and Calendar Years" vs. "Detail of Expenditures by
    Months and Years") — pages with identical concepts often differ ONLY in
    their title.
  - `[columns]` and `[rows]` show the table's actual schema — use them to
    confirm the page has the right axes/breakdown for the question. A page
    titled "Public Debt" with columns "Bills | Notes | Bonds | Total" is
    very different from one with columns "December 31 | March 31 | June 30 |
    September 30".
  - `[dates]` is supplementary — ALL date strings on the page, including
    bulletin month, column-header years, and footnote years. Only weight
    entries that the title doesn't already cover.

Ranking rules:
  1. PERIOD: prefer pages whose `[title]` date matches the question's
     period. If the title has no date, fall back to `[dates]`.
  2. CONCEPT: among period-matching pages, prefer the one whose title most
     specifically names what the question asks about ("Summary by Calendar
     Years" > "Detail of Expenditures" for a calendar-year question).
  3. Day-of-month is fuzzy: 'February 28, 1952' matches 'February 29, 1952'
     or 'February 1952' — same monthly snapshot.
  4. Publication lag is expected: a bulletin published months AFTER the
     question's period is often the canonical retrospective home.
  5. LITERAL TITLE WORDING: when the question uses phrases like "calendar
     years" vs. "fiscal years" vs. "monthly", prefer titles that contain
     those literal phrases. "Summary by Months and Calendar Years" is a
     DISTINCT table from "Summary by Months and Years" or "Summary of
     Budget Results by Months and Years" — they sit side-by-side in the
     bulletin and report different things. Title wording is authoritative;
     do not collapse near-synonyms.
  6. EARLIEST CANONICAL RETROSPECTIVE: when multiple bulletins (e.g. 1950-02,
     1952-02, 1953-02) carry the SAME or near-identical title and all cover
     the question's period, prefer the EARLIEST one. Treasury publishes the
     canonical year-end summary in the first bulletin after the period
     closes; later bulletins republish the same table with extended columns.

Source-bulletin hard pin: when the QUESTION TEXT literally names a bulletin
issue (e.g., "page 5 of the September 1990 Treasury Bulletin", "the March
1948 issue"), return ONLY matches from that bulletin and reject candidates
from other bulletins. Otherwise consider all bulletins — the answer often
lives in a retrospective bulletin published months after the question's
period (e.g. "as of March 31, 2025" is often answered in the 2025-06 issue).

Output a SINGLE JSON object (no prose, no fences):
  {"ranked": [
      {"bulletin": "YYYY-MM", "page": <int>, "reason": "<short rationale>"}
    ]}

Rules:
  - Return AT LEAST 1 page even if no candidate is a perfect match. An
    empty result is never useful — pick the best available candidate so
    downstream extract has something to try. If many pages genuinely match
    (multi-year question spanning multiple snapshots), return all of them
    best-to-worst.
  - Do not pad with obviously irrelevant matches when 1-2 pages clearly fit.
  - Concept match comes first; among concept-matching pages, prefer the one
    whose summary date falls inside (or at the closing month of) the period.
  - Dedupe by (bulletin, page).
"""




def _build_leaf_blob(row: PageCatalogRow) -> str:
    """Per-candidate text blob fed to the leaf-rank LLM.

    Format is fixed and documented in `_LEAF_RANK_HIER_SYSTEM`. For a
    multi-content page (table + chart), table blocks contribute columns
    and row samples; chart blocks contribute their captions. Prose-only
    pages substitute the page-level keyword list as content. Field caps
    keep the per-candidate footprint bounded so a few-hundred-candidate
    prompt stays well under the context window.
    """
    titles = row.all_titles()
    title_line = " | ".join(titles) if titles else "(no title)"
    lines = [f"=== {row.bulletin} p{row.page} ===",
             f"[title]    {title_line}"]

    if row.content_blocks and not row.has_visual_block():
        if row.keywords:
            lines.append(f"[content]  {' | '.join(row.keywords[:20])}")
    else:
        lines.append(f"[columns]  {' | '.join(row.all_column_headers()[:20])}")
        lines.append(f"[rows]     {' | '.join(row.all_row_headers_sample()[:8])}")
    if row.dates:
        lines.append(f"[dates]    {'; '.join(row.dates[:20])}")
    return "\n".join(lines)


def build_vector_blob(row: PageCatalogRow) -> str:
    """Minimal blob for the dense vector index — title(s) + keywords only.

    Drops column_headers, row_headers_sample, and dates: those fields
    are heavy on generic axis labels ("total", month names, country
    names, year stubs) that dilute the pooled embedding without
    discriminating between pages. Symbolic date filtering at L2 handles
    period overlap, so dates don't need to live in the vector either.

    Title and keywords are the discriminative content per page:
    captions name the table / chart / writeup; keywords are dateless
    noun phrases extracted from the caption.
    """
    titles = row.all_titles()
    title_line = " | ".join(titles) if titles else "(no title)"
    kw_line = " | ".join(row.keywords[:10]) if row.keywords else "(no keywords)"
    return (
        f"=== {row.bulletin} p{row.page} ===\n"
        f"[title]    {title_line}\n"
        f"[keywords] {kw_line}"
    )


def leaf_rank_postings(
    question: str,
    concept: str,
    period: str,
    candidates: list[tuple[str, int]],
    catalog_index: dict[tuple[str, int], PageCatalogRow],
    llm: LLMClient,
    *,
    k: int = 50,
) -> tuple[list[dict[str, Any]], LevelTrace]:
    """Rank a flat list of (bulletin, page) candidates using rich text-blob
    payloads built from the catalog. Return however many the LLM judges
    relevant (best-to-worst).

    `k` is a defensive cap on output parsing, NOT a directive to the LLM.
    Source-bulletin hard pins (e.g. "page 5 of the September 1990 bulletin")
    are extracted by the LLM directly from the question text."""
    trace = LevelTrace(level="leaf_rank", input_count=len(candidates), input_chars=0)
    if not candidates:
        return [], trace

    blobs: list[str] = []
    for key in candidates:
        row = catalog_index.get(key)
        if row is None:
            continue
        blobs.append(_build_leaf_blob(row))
    if not blobs:
        return [], trace

    user = (
        f"Question: {question}\n\n"
        f"Concept: {concept}\n"
        f"Period:  {period}\n\n"
        f"Candidate pages ({len(blobs)}):\n\n"
        + "\n\n".join(blobs)
        + "\n"
    )
    trace.input_chars = len(user)
    trace.prompt_excerpt = user[:800]

    t0 = time.monotonic()
    resp = llm.call(system=_LEAF_RANK_HIER_SYSTEM, user=user, temperature=0.0)
    trace.latency_s = time.monotonic() - t0
    trace.output_chars = len(resp.text)
    trace.input_tokens = resp.input_tokens
    trace.output_tokens = resp.output_tokens
    trace.response_excerpt = resp.text[:800]

    out: list[dict[str, Any]] = []
    try:
        obj = json.loads(_strip_code_fence(resp.text))
    except json.JSONDecodeError:
        return out, trace
    seen: set[tuple[str, int]] = set()
    for r in obj.get("ranked", []) or []:
        try:
            bulletin = str(r["bulletin"])
            page = int(r["page"])
        except (KeyError, ValueError, TypeError):
            continue
        key = (bulletin, page)
        if key in seen:
            continue
        seen.add(key)
        out.append({"bulletin": bulletin, "page": page,
                    "reason": str(r.get("reason", ""))})
        if len(out) >= k:
            break
    trace.output_count = len(out)
    return out, trace



def _safe_json(text: str) -> dict[str, Any] | None:
    s = _strip_code_fence(text)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None



# The concept tree is now flat: {chapters: {<canonical>: {n_pages, members, pages}}}.
# Phase 3 (merge.py) derives the canonical chapter set from data; the
# hand-curated parent-chapter rollup is no longer needed.


def chapter_index(tree: dict[str, Any]) -> list[str]:
    """Return the canonical chapter names in the tree, sorted by page count."""
    chapters = tree.get("chapters", {})
    return sorted(chapters.keys(),
                  key=lambda c: -chapters[c].get("n_pages", 0))


# ---------------------------------------------------------------------------
# One-shot section retriever — diagnostic baseline
# ---------------------------------------------------------------------------
#
# Strips everything except `section_pick(k=1)`: one LLM call, exactly one
# canonical section returned, predicted set = every page in that section.
# Used to measure how often the agent can correctly identify the gold's
# canonical section in a single shot.


def _chapter_pages(tree: dict[str, Any], chapter: str) -> list[tuple[str, int]]:
    """Flat-tree: list (bulletin, page) tuples under a canonical chapter."""
    data = tree.get("chapters", {}).get(chapter)
    if not data:
        return []
    seen: set[tuple[str, int]] = set()
    out: list[tuple[str, int]] = []
    for p in data.get("pages", []):
        k = (p["bulletin"], p["page"])
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


_PARENT_PICK_SYSTEM = """You pick which Treasury Bulletin chapter most
likely contains the answer to the user's question.

You will see the question, a concept tag, a period, and a list of
canonical chapters. Each entry has:
  - `chapter`: the canonical chapter name (your output MUST be this)
  - `n_pages`: total pages in the chapter
  - `description`: a scope statement of what the chapter covers
  - `examples`: concrete sub-area / topic names that live in this
    chapter. These are string-match anchors for questions that
    reference specific eras, programs, or table topics by name.

Use BOTH `description` and `examples` to match the question. The
description gives the chapter's abstract framing; the examples surface
specific sub-areas (e.g. era-specific programs, named tables) that the
description may not mention.

Output a SINGLE JSON object (no prose, no fences):
  {"picked": "<exact chapter label>"}

Rules:
  - Return exactly ONE chapter label, the best match.
  - Use the EXACT `chapter` value shown; do NOT emit anything from a
    `description` or `examples` list.
"""


def one_shot_parent_chapter_retrieve(
    tree: dict[str, Any],
    question: str,
    concept: str,
    period: str,
    llm: LLMClient,
    catalog_index: dict[tuple[str, int], PageCatalogRow],
    *,
    uid: str | None = None,
    retrieve_idx: int = 0,
) -> tuple[list[dict[str, Any]], RetrieveTrace]:
    """One LLM call → one canonical chapter → every page under it.

    Operates on the flat tree shape (Phase-3 output): each chapter is a
    top-level node with its own page list. The validator accepts a
    single chapter string or a one-element list; out-of-vocabulary picks
    are dropped.
    """
    chapters_data = tree.get("chapters", {})
    trace = RetrieveTrace(
        uid=uid, retrieve_idx=retrieve_idx,
        concept=concept, period=period,
        catalog_size=sum(c.get("n_pages", 0) for c in chapters_data.values()),
    )
    t_walk = time.monotonic()

    listing: list[dict[str, Any]] = [
        {
            "chapter": chapter,
            "n_pages": data.get("n_pages", 0),
            "description": data.get("description", ""),
            "examples": data.get("examples", []),
        }
        for chapter, data in chapters_data.items()
    ]
    listing.sort(key=lambda x: -x["n_pages"])

    level_trace = LevelTrace(level="parent_pick",
                             input_count=len(listing), input_chars=0)
    user = (
        f"Question: {question}\n\n"
        f"Concept: {concept}\n"
        f"Period:  {period}\n\n"
        f"Chapters ({len(listing)}):\n"
        f"{json.dumps(listing, ensure_ascii=False, indent=1)}\n"
    )
    level_trace.input_chars = len(user)
    level_trace.prompt_excerpt = user[:800]

    t0 = time.monotonic()
    resp = llm.call(system=_PARENT_PICK_SYSTEM, user=user, temperature=0.0)
    level_trace.latency_s = time.monotonic() - t0
    level_trace.output_chars = len(resp.text)
    level_trace.input_tokens = resp.input_tokens
    level_trace.output_tokens = resp.output_tokens
    level_trace.response_excerpt = resp.text[:800]

    obj = _safe_json(resp.text) or {}
    raw_picked = obj.get("picked", "")
    if isinstance(raw_picked, str):
        raw_picks = [raw_picked]
    elif isinstance(raw_picked, list):
        raw_picks = [str(x).strip() for x in raw_picked if str(x).strip()]
    else:
        raw_picks = []
    valid = {c.lower(): c for c in chapters_data}
    picked_chapters: list[str] = []
    seen_picks: set[str] = set()
    for raw in raw_picks[:2]:
        ch = valid.get(raw.strip().lower())
        if ch and ch not in seen_picks:
            picked_chapters.append(ch)
            seen_picks.add(ch)
    trace.levels.append(level_trace)

    if not picked_chapters:
        trace.total_walk_s = time.monotonic() - t_walk
        return [], trace

    seen_pages: set[tuple[str, int]] = set()
    top: list[dict[str, Any]] = []
    for chapter in picked_chapters:
        for (b, p) in _chapter_pages(tree, chapter):
            if (b, p) in seen_pages:
                continue
            seen_pages.add((b, p))
            top.append({"bulletin": b, "page": p,
                        "reason": f"chapter={chapter}"})
    trace.candidate_count = len(top)
    level_trace.output_count = len(picked_chapters)
    trace.top_k = top[:50]
    trace.total_walk_s = time.monotonic() - t_walk
    return top, trace


def one_shot_section_retrieve(
    tree: dict[str, Any],
    question: str,
    concept: str,
    period: str,
    llm: LLMClient,
    catalog_index: dict[tuple[str, int], PageCatalogRow],
    *,
    uid: str | None = None,
    retrieve_idx: int = 0,
) -> tuple[list[dict[str, Any]], RetrieveTrace]:
    """One LLM call → one canonical chapter → all pages under it.

    In the flat-tree world, "section" and "chapter" are the same node;
    this entrypoint exists for back-compat with eval_retrieve.py's
    `--retriever one-shot-section` mode. It delegates to the
    parent-chapter retriever.
    """
    return one_shot_parent_chapter_retrieve(
        tree, question, concept, period, llm, catalog_index,
        uid=uid, retrieve_idx=retrieve_idx,
    )


# ---------------------------------------------------------------------------
# L1 chapter pick + L2 vector top-K within chapter
# ---------------------------------------------------------------------------

def _row_date_envelope(
    row: PageCatalogRow | None,
) -> tuple[str, str] | None:
    """Min/max ISO envelope over all parsed `row.dates` entries.

    Each date string parses to one or more `(start_iso, end_iso)`
    intervals via `verbatim_date_to_intervals`; we take the earliest
    start and latest end across all of them. Returns None when the row
    is missing, has no dates, or none parse — callers treat that as
    "no date signal" and may fall back to other heuristics.

    Storing dates as a single envelope per page (rather than checking
    every interval) is simpler, faster, and slightly more forgiving for
    historical tables that print a gap-year set (e.g. only every fifth
    year): the envelope spans the printed range; any query year inside
    that range counts as a match.
    """
    if row is None or not row.dates:
        return None
    starts: list[str] = []
    ends: list[str] = []
    for s in row.dates:
        for a, b in verbatim_date_to_intervals(s):
            starts.append(a)
            ends.append(b)
    if not starts:
        return None
    return min(starts), max(ends)


def _row_dates_overlap_period(
    row: PageCatalogRow | None,
    period_intervals: list[tuple[str, str]],
) -> tuple[bool, bool]:
    """Wrapper around `_row_date_envelope` that returns
    `(has_signal, overlaps)` against the query period."""
    env = _row_date_envelope(row)
    if env is None:
        return False, False
    e_start, e_end = env
    for p_start, p_end in period_intervals:
        if intervals_overlap(e_start, e_end, p_start, p_end):
            return True, True
    return True, False


_PROPOSE_DATES_SYSTEM = """You read a user question and emit the list of
dates / date ranges that any answer page would have to overlap.

Use the project's period grammar:
  YYYY              bare year  (e.g. "1934", "2025")
  YYYY-MM           year-month (e.g. "1990-09")
  YYYY-MM-DD        specific date
  FYYYYY            fiscal year (e.g. "FY1991")
  CYYYYY            calendar year (e.g. "CY1990")
  Qn-YYYY           quarter (e.g. "Q2-1991")
  pointA..pointB    range (e.g. "FY1990..FY1995", "1934..1940")
  enumeration       comma-list of points (e.g. "CY1991, CY1996")

You will see:
  - The user's full question
  - The plan's concept tag for THIS retrieve branch
  - The plan's period tag for THIS retrieve branch

Your output covers ONLY this branch — not the whole question. If the
branch's period tag already captures the dates cleanly, emit it
verbatim. If the question text reveals additional period framing the
tag may have lost (a fiscal-year context, an end-of-period date, an
adjacent reference year), include those too. Keep the list tight;
every entry will be OR-ed when filtering pages.

Output a SINGLE JSON object (no prose, no fences):
  {"dates": ["<period string>", ...]}

Every entry MUST parse under the grammar above.
"""


def propose_query_dates(
    question: str, concept: str, period: str, llm: LLMClient,
) -> tuple[list[tuple[str, str]], str]:
    """One LLM call → list of (start_iso, end_iso) intervals for this
    retrieve branch. Returns (intervals, response_excerpt). Falls back to
    `period_to_intervals(period)` on parse failure."""
    user = (
        f"Question: {question}\n\n"
        f"Branch concept: {concept}\n"
        f"Branch period:  {period}\n"
    )
    resp = llm.call(system=_PROPOSE_DATES_SYSTEM, user=user,
                    temperature=0.0, thinking_budget=0)
    obj = _safe_json(resp.text) or {}
    raw = obj.get("dates") or []
    intervals: list[tuple[str, str]] = []
    if isinstance(raw, list):
        for s in raw:
            if not isinstance(s, str):
                continue
            try:
                intervals.extend(period_to_intervals(s.strip()))
            except ValueError:
                continue
    if not intervals and period:
        try:
            intervals = period_to_intervals(period)
        except ValueError:
            intervals = []
    return intervals, resp.text[:300]


def _bulletin_in_period_window(
    bulletin: str,
    period_intervals: list[tuple[str, str]],
    publish_lag_months: int = 12,
) -> bool:
    """Bulletin month within `[period_start, period_end + publish_lag]`.
    Same rule as `vector_index._bulletin_to_iso` + window check; lifted
    here so the post-filter doesn't drag in numpy."""
    try:
        y, m = bulletin.split("-")
        b_iso = f"{int(y):04d}-{int(m):02d}-15"
    except (ValueError, AttributeError):
        return False
    for p_start, p_end in period_intervals:
        # End-of-window = end + publish_lag_months.
        try:
            ey, em = int(p_end[:4]), int(p_end[5:7])
        except (ValueError, IndexError):
            continue
        total = ey * 12 + (em - 1) + publish_lag_months
        wy, wm = divmod(total, 12)
        w_end = f"{wy:04d}-{wm + 1:02d}-{p_end[8:10]}"
        if intervals_overlap(b_iso, b_iso, p_start, w_end):
            return True
    return False


def l1_vector_retrieve(
    tree: dict[str, Any],
    question: str,
    concept: str,
    period: str,
    llm: LLMClient,
    catalog_index: dict[tuple[str, int], PageCatalogRow],
    vector_index: Any,  # VectorIndex from vector_index.py
    *,
    top_k_vector: int = 100,
    use_period_mask: bool = False,
    pre_date_intersect: bool = False,
    date_filter_mode: str = "off",  # "off" | "soft"
    use_llm_dates: bool = False,
    skip_l1: bool = False,
    top_n: int | None = None,
    uid: str | None = None,
    retrieve_idx: int = 0,
) -> tuple[list[dict[str, Any]], RetrieveTrace]:
    """L1 chapter pick → masked vector cosine top-K within that chapter.

    Two-stage: the L1 retriever picks one canonical chapter, then we run
    a query embedding against just that chapter's pages and return the
    top-K cosine neighbors. No LLM rerank at L2 — the goal is to measure
    how far raw vector neighborhood gets us as a candidate-reduction step.

    When `use_period_mask=True`, ANDs the chapter mask with the symbolic
    period mask (bulletin month within [period_start, period_end + 12mo]
    plus a date-string year check), pruning era-irrelevant pages before
    the cosine pass.

    When `skip_l1=True`, no chapter is picked: the cosine pass runs over
    every indexed page (subject to `use_period_mask` if also set). For
    ablation studies that isolate the L1 step's contribution.
    """
    import numpy as np
    from .vector_index import period_mask as _period_mask, search

    if skip_l1:
        trace = RetrieveTrace(
            uid=uid, retrieve_idx=retrieve_idx,
            concept=concept, period=period,
            catalog_size=len(vector_index.keys),
        )
        chapter_pages: set[tuple[str, int]] = set(vector_index.keys)
    else:
        top_l1, trace = one_shot_parent_chapter_retrieve(
            tree, question, concept, period, llm, catalog_index,
            uid=uid, retrieve_idx=retrieve_idx,
        )
        chapter_pages = {(r["bulletin"], r["page"]) for r in top_l1}
    if not chapter_pages:
        return [], trace

    t_walk = time.monotonic()

    # Resolve query-side date intervals once. Either let an LLM propose
    # them (use_llm_dates) or parse the planner's structured `period`.
    query_intervals: list[tuple[str, str]] = []
    llm_dates_excerpt = ""
    if use_llm_dates:
        query_intervals, llm_dates_excerpt = propose_query_dates(
            question, concept, period, llm,
        )
    if not query_intervals and period:
        try:
            query_intervals = period_to_intervals(period)
        except ValueError:
            query_intervals = []

    mask = np.fromiter(
        (k in chapter_pages for k in vector_index.keys),
        dtype=bool, count=len(vector_index.keys),
    )
    if use_period_mask and period:
        pmask = _period_mask(vector_index.keys, period,
                             catalog_index=catalog_index)
        mask = mask & pmask

    # Strict date-envelope intersect as a PRE-mask. Shrinks the pool the
    # cosine ranks over, so top-K becomes more selective within a tight
    # era cohort. Soft on empty `dates` (falls back to bulletin window).
    if pre_date_intersect and query_intervals:
        strict = np.zeros(len(vector_index.keys), dtype=bool)
        for i, key in enumerate(vector_index.keys):
            if not mask[i]:
                continue
            row = catalog_index.get(key)
            _, overlaps = _row_dates_overlap_period(row, query_intervals)
            # Same logic as the soft post-filter: keep if dates
            # intersect OR bulletin is within the publish-lag window.
            # The bulletin-window OR is essential for retrospective
            # tables whose printed dates are historical years.
            if overlaps or _bulletin_in_period_window(key[0], query_intervals):
                strict[i] = True
        mask = mask & strict

    # Bare-text query: the preamble experiment compressed the embedding
    # space too much (top hits bunched at cosine 0.86; gold sat at rank
    # 3742 on a known case). Concatenating concept + question as plain
    # text gives the embedding model the discriminating signal without a
    # shared instructional prefix dominating the vector.
    embed_text = f"{concept}\n\n{question}"
    t_embed = time.monotonic()
    vecs = llm.embed([embed_text], task_type="RETRIEVAL_QUERY",
                     dim=vector_index.dim)
    embed_latency = time.monotonic() - t_embed
    if not vecs:
        return [], trace

    import numpy as _np
    query_vec = _np.asarray(vecs[0], dtype=_np.float32)

    t_ann = time.monotonic()
    hits = search(vector_index, query_vec, mask, top_k=top_k_vector)
    ann_latency = time.monotonic() - t_ann

    ann_trace = LevelTrace(
        level="l2_vector",
        input_count=int(mask.sum()),
        input_chars=len(embed_text),
        latency_s=embed_latency + ann_latency,
        output_count=len(hits),
        prompt_excerpt=embed_text[:800],
        response_excerpt="; ".join(
            f"{b}/p{p}:{score:.3f}" for score, (b, p) in hits[:10]
        ),
    )
    trace.levels.append(ann_trace)

    # Post-filter on date overlap. Cosine ordering is preserved among
    # survivors. Uses the same `query_intervals` resolved above (LLM
    # proposal or planner period).
    top: list[dict[str, Any]] = []
    n_kept_date = n_kept_bull = n_dropped = 0
    for rank, (_score, (b, p)) in enumerate(hits):
        if date_filter_mode == "soft" and query_intervals:
            row = catalog_index.get((b, p))
            _, overlaps = _row_dates_overlap_period(row, query_intervals)
            # Soft = "dates intersect period" OR "bulletin within window".
            # Bulletin-window catches retrospective tables whose printed
            # `dates` are historical years that don't include the query
            # period, but whose publication date does.
            if overlaps:
                n_kept_date += 1
            elif _bulletin_in_period_window(b, query_intervals):
                n_kept_bull += 1
            else:
                n_dropped += 1
                continue
        top.append({"bulletin": b, "page": p,
                    "reason": f"vector-rank-{rank + 1}"})
        if top_n is not None and len(top) >= top_n:
            break

    if date_filter_mode == "soft":
        ann_trace.response_excerpt = (
            f"{ann_trace.response_excerpt}  | post-filter: "
            f"kept_date={n_kept_date} kept_bulletin={n_kept_bull} "
            f"dropped={n_dropped}"
        )
    if use_llm_dates and llm_dates_excerpt:
        ann_trace.response_excerpt = (
            f"{ann_trace.response_excerpt}  | llm_dates={llm_dates_excerpt}"
        )

    trace.candidate_count = len(top)
    trace.top_k = top[:50]
    trace.total_walk_s = time.monotonic() - t_walk
    return top, trace


def write_trace_jsonl(trace: RetrieveTrace, path: Path) -> None:
    """Append one RetrieveTrace as a JSON line to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(trace), ensure_ascii=False))
        f.write("\n")


def load_traces(path: Path) -> list[RetrieveTrace]:
    """Read back a trace JSONL for offline analysis."""
    out: list[RetrieveTrace] = []
    if not path.exists():
        return out
    for line in path.open():
        d = json.loads(line)
        levels = [LevelTrace(**lvl) for lvl in d.pop("levels", [])]
        out.append(RetrieveTrace(levels=levels, **d))
    return out
