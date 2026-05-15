"""Retrieval probe — hierarchical walker over the concept tree.

Two walker shapes live in this module:

* `retrieve_for_op` — the legacy flat walker. Period prefilter on the catalog
  then one LLM call over all survivors. Kept for diff/comparison only.

* `retrieve_hierarchical` — the v0.2 walker. Four LLM calls per retrieve op:
  `section_pick → cluster_pick → term_pick → leaf_rank`. Each level sees
  ≤100 small items. No symbolic period filter — period info flows through
  (a) the question text shown to every level and (b) the verbatim date
  strings inside descriptions at L4.

Both walkers populate the same `RetrieveTrace.levels` list, so the trace
schema is identical between them. Use `load_concept_tree(path)` to load
the offline-built `concept_tree.json`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from skunk.common import LLMClient

from .period import intervals_overlap, period_to_intervals
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


def period_prefilter(
    catalog: list[PageCatalogRow],
    period: str,
) -> list[PageCatalogRow]:
    """Keep rows whose periods_covered overlaps the query period.
    Restricts to page_kind in {table, chart}."""
    intervals = period_to_intervals(period)
    out: list[PageCatalogRow] = []
    for row in catalog:
        if row.page_kind not in ("table", "chart"):
            continue
        for q_start, q_end in intervals:
            if any(intervals_overlap(p.start, p.end, q_start, q_end)
                   for p in row.periods_covered):
                out.append(row)
                break
    return out


# ---------------------------------------------------------------------------
# Leaf rank — single-LLM-call walker level
# ---------------------------------------------------------------------------

_LEAF_RANK_SYSTEM = """You are a retrieval ranker over a catalog of U.S. Treasury Bulletin
pages. The catalog has already been pre-filtered by date (every candidate page
reports on a date that overlaps the user's query period). Your job is to pick
which pages most likely contain the data the user is asking about.

You will receive:
  - The user's question (verbatim).
  - The query concept tag (e.g. 'budget_expenditures').
  - The query period.
  - A list of candidate pages, each with: bulletin, page, section, table_title, keywords.

Output a SINGLE JSON object (no prose, no fences):
  {"ranked": [
      {"bulletin": "YYYY-MM", "page": <int>, "reason": "<short rationale>"}
   ]}

Rules:
  - Return AT MOST 5 entries, ordered best-to-worst.
  - A page is a strong match when its `table_title` or `keywords` name the
    concept the user is asking about (use Treasury terminology, not synonyms —
    a page about "national defense" matches "national defense and associated
    activities" but NOT "national security").
  - When two pages both match the concept, prefer ones whose granularity and
    period kind match the question (e.g. "individual calendar months" → monthly
    page over annual).
  - Pages flagged `is_retrospective=true` (period ends before bulletin date)
    are typically the canonical home for backward-looking questions ("CY1940
    totals" → look in 1941 or later issues). Prefer them when the question
    asks about closed periods.
  - It is okay to return fewer than 5 if you only see 1-2 strong matches.
"""


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", s)
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


def _candidate_payload(c: PageCatalogRow) -> dict[str, Any]:
    return {
        "bulletin": c.bulletin,
        "page": c.page,
        "section": c.section,
        "table_title": c.table_title,
        "keywords": c.keywords,
        "granularity": c.granularity,
        "is_retrospective": c.is_retrospective,
    }


def leaf_rank(
    question: str,
    concept: str,
    period: str,
    candidates: list[PageCatalogRow],
    llm: LLMClient,
) -> tuple[list[dict[str, Any]], LevelTrace]:
    """Single LLM call → top-K with rationales. Returns (top_k, trace)."""
    trace = LevelTrace(level="leaf_rank", input_count=len(candidates), input_chars=0)

    if not candidates:
        trace.response_excerpt = "<skipped: 0 candidates>"
        return [], trace

    payload = [_candidate_payload(c) for c in candidates]
    user = (
        f"Question: {question}\n\n"
        f"Concept: {concept}\n"
        f"Period:  {period}\n\n"
        f"Candidates ({len(candidates)} pages):\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=1)}\n"
    )
    trace.input_chars = len(user)
    trace.prompt_excerpt = user[:800]

    t0 = time.monotonic()
    resp = llm.call(system=_LEAF_RANK_SYSTEM, user=user, temperature=0.0)
    trace.latency_s = time.monotonic() - t0
    trace.output_chars = len(resp.text)
    trace.input_tokens = resp.input_tokens
    trace.output_tokens = resp.output_tokens
    trace.response_excerpt = resp.text[:800]

    out: list[dict[str, Any]] = []
    try:
        obj = json.loads(_strip_code_fence(resp.text))
    except json.JSONDecodeError:
        trace.output_count = 0
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
        out.append({
            "bulletin": bulletin,
            "page": page,
            "reason": str(r.get("reason", "")),
        })
        if len(out) >= 5:
            break
    trace.output_count = len(out)
    return out, trace


# Back-compat alias: the original entry point. Same behavior, no telemetry.
def walk(
    question: str,
    concept: str,
    period: str,
    candidates: list[PageCatalogRow],
    llm: LLMClient,
) -> list[dict[str, Any]]:
    """Deprecated: prefer `leaf_rank()` for the (results, trace) pair."""
    out, _ = leaf_rank(question, concept, period, candidates, llm)
    return out


# ---------------------------------------------------------------------------
# Hierarchical walker (v0.2): section → cluster → term → leaf
# ---------------------------------------------------------------------------

def load_concept_tree(path: Path) -> dict[str, Any]:
    """Load `concept_tree.json` produced by `skunk.page_index.concept_tree`."""
    return json.loads(Path(path).read_text())


_SECTION_PICK_SYSTEM = """You pick which U.S. Treasury Bulletin sections likely contain
the data the user asks about. You will see the user's question, a query concept tag,
and a query period, plus a list of section labels (from the bulletin's native table
of contents). Each section is a sub-tree of clusters/terms/pages — by picking a
section you commit to drilling into it.

Output a SINGLE JSON object (no prose, no fences):
  {"picked": ["<section label>", ...]}

Rules:
  - Return AT MOST K labels (default 3), ordered best-to-worst. Stick to the LABELS
    EXACTLY AS SHOWN — do not paraphrase.
  - You may return 1 label if only one section clearly matches the concept.
  - Prefer sections that name the concept directly (e.g., 'Statutory debt limitation'
    for a debt-limit question) over auxiliary sections.
"""


_CLUSTER_PICK_SYSTEM = """You pick which keyword clusters inside selected sections
likely lead to the answer. You will see the user's question, the concept tag, the
period, and a list of clusters (each with its parent section, a label, and a
keyword count). Each cluster groups related Treasury keywords; by picking one
you commit to ranking its keyword postings.

Output a SINGLE JSON object (no prose, no fences):
  {"picked": [
      {"section": "<section label>", "cluster_label": "<cluster label>"},
      ...
    ]}

Rules:
  - Return AT MOST K (default 3) entries, best-to-worst. Use EXACT section
    labels and cluster_labels as shown.
  - Prefer clusters whose label names the concept directly.
"""


_TERM_PICK_SYSTEM = """You pick high-information Treasury terms — table titles,
named series, technical line items — that most likely identify the answer page.
You will see the user's question, the concept tag, the period, and a list of
candidate terms (each with its parent section and cluster). A "good" term is
specific, verbatim Treasury vocabulary (e.g., 'Statutory Debt Limitation',
'Status under Limitation', 'Treasury bonds - bank eligible') — NOT generic words
like 'Marketable' or 'monthly' that match anything.

Output a SINGLE JSON object (no prose, no fences):
  {"picked": [
      {"section": "<section label>", "cluster_id": "<Cnn>", "term": "<verbatim term>"},
      ...
    ]}

Rules:
  - Return AT MOST K (default 4) entries, best-to-worst. Use EXACT strings as shown.
  - Each picked term will be expanded to its full postings list (pages where it
    appears), so prefer specific, distinctive terms over broad ones.
"""


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


def _picked_set(obj: Any, key: str) -> list[Any]:
    if not isinstance(obj, dict):
        return []
    v = obj.get(key, [])
    return v if isinstance(v, list) else []


def section_pick(
    question: str,
    concept: str,
    period: str,
    sections: list[tuple[str, dict[str, Any]]],
    llm: LLMClient,
    *,
    k: int = 3,
) -> tuple[list[str], LevelTrace]:
    """sections = [(section_label, section_subtree_dict)]. Return up to k labels."""
    trace = LevelTrace(level="section_pick", input_count=len(sections), input_chars=0)
    if not sections:
        return [], trace

    listing = [
        {"section": label, "n_pages": data["n_pages_in_section"],
         "n_clusters": data["n_clusters"]}
        for label, data in sections
    ]
    user = (
        f"Question: {question}\n\n"
        f"Concept: {concept}\n"
        f"Period:  {period}\n"
        f"K (max picks): {k}\n\n"
        f"Sections ({len(listing)}):\n"
        f"{json.dumps(listing, ensure_ascii=False, indent=1)}\n"
    )
    trace.input_chars = len(user)
    trace.prompt_excerpt = user[:800]

    t0 = time.monotonic()
    resp = llm.call(system=_SECTION_PICK_SYSTEM, user=user, temperature=0.0)
    trace.latency_s = time.monotonic() - t0
    trace.output_chars = len(resp.text)
    trace.input_tokens = resp.input_tokens
    trace.output_tokens = resp.output_tokens
    trace.response_excerpt = resp.text[:800]

    try:
        obj = json.loads(_strip_code_fence(resp.text))
    except json.JSONDecodeError:
        return [], trace
    # Case-insensitive section-label lookup — the LLM sometimes lower-cases
    # parts of titles, and we don't want to drop the right pick over casing.
    valid = {label.lower(): label for label, _ in sections}
    picked: list[str] = []
    picked_lower: set[str] = set()
    for s in _picked_set(obj, "picked"):
        key = str(s).strip().lower()
        canonical = valid.get(key)
        if canonical and key not in picked_lower:
            picked.append(canonical)
            picked_lower.add(key)
        if len(picked) >= k:
            break
    trace.output_count = len(picked)
    return picked, trace


def cluster_pick(
    question: str,
    concept: str,
    period: str,
    clusters: list[tuple[str, str, dict[str, Any]]],
    llm: LLMClient,
    *,
    k: int = 3,
) -> tuple[list[tuple[str, str]], LevelTrace]:
    """clusters = [(section_label, cluster_label, cluster_data)].
    Return up to k (section, cluster_label) tuples."""
    trace = LevelTrace(level="cluster_pick", input_count=len(clusters), input_chars=0)
    if not clusters:
        return [], trace

    listing = [
        {"section": section, "cluster_label": cluster_label,
         "n_keywords": cdata.get("n_keywords", 0),
         "n_pages": cdata.get("n_pages", 0)}
        for section, cluster_label, cdata in clusters
    ]
    user = (
        f"Question: {question}\n\n"
        f"Concept: {concept}\n"
        f"Period:  {period}\n"
        f"K (max picks): {k}\n\n"
        f"Clusters ({len(listing)}):\n"
        f"{json.dumps(listing, ensure_ascii=False, indent=1)}\n"
    )
    trace.input_chars = len(user)
    trace.prompt_excerpt = user[:800]

    t0 = time.monotonic()
    resp = llm.call(system=_CLUSTER_PICK_SYSTEM, user=user, temperature=0.0)
    trace.latency_s = time.monotonic() - t0
    trace.output_chars = len(resp.text)
    trace.input_tokens = resp.input_tokens
    trace.output_tokens = resp.output_tokens
    trace.response_excerpt = resp.text[:800]

    try:
        obj = json.loads(_strip_code_fence(resp.text))
    except json.JSONDecodeError:
        return [], trace
    valid = {(s.lower(), c.lower()): (s, c) for s, c, _ in clusters}
    picked: list[tuple[str, str]] = []
    picked_lower: set[tuple[str, str]] = set()
    for entry in _picked_set(obj, "picked"):
        if not isinstance(entry, dict):
            continue
        key = (str(entry.get("section", "")).strip().lower(),
               str(entry.get("cluster_label", "")).strip().lower())
        canonical = valid.get(key)
        if canonical and key not in picked_lower:
            picked.append(canonical)
            picked_lower.add(key)
        if len(picked) >= k:
            break
    trace.output_count = len(picked)
    return picked, trace


def term_pick(
    question: str,
    concept: str,
    period: str,
    terms: list[tuple[str, str, str, list[dict[str, Any]]]],
    llm: LLMClient,
    *,
    k: int = 4,
) -> tuple[list[tuple[str, str, str]], LevelTrace]:
    """terms = [(section, cluster_id, term_str, postings_list)]. Return up to k (section, cluster_id, term) tuples."""
    trace = LevelTrace(level="term_pick", input_count=len(terms), input_chars=0)
    if not terms:
        return [], trace

    listing = [
        {"section": section, "cluster_id": cid, "term": term, "n_pages": len(postings)}
        for section, cid, term, postings in terms
    ]
    user = (
        f"Question: {question}\n\n"
        f"Concept: {concept}\n"
        f"Period:  {period}\n"
        f"K (max picks): {k}\n\n"
        f"Terms ({len(listing)}):\n"
        f"{json.dumps(listing, ensure_ascii=False, indent=1)}\n"
    )
    trace.input_chars = len(user)
    trace.prompt_excerpt = user[:800]

    t0 = time.monotonic()
    resp = llm.call(system=_TERM_PICK_SYSTEM, user=user, temperature=0.0)
    trace.latency_s = time.monotonic() - t0
    trace.output_chars = len(resp.text)
    trace.input_tokens = resp.input_tokens
    trace.output_tokens = resp.output_tokens
    trace.response_excerpt = resp.text[:800]

    try:
        obj = json.loads(_strip_code_fence(resp.text))
    except json.JSONDecodeError:
        return [], trace
    valid = {(s, c, t) for s, c, t, _ in terms}
    picked: list[tuple[str, str, str]] = []
    for entry in _picked_set(obj, "picked"):
        if not isinstance(entry, dict):
            continue
        key = (str(entry.get("section", "")),
               str(entry.get("cluster_id", "")),
               str(entry.get("term", "")))
        if key in valid and key not in picked:
            picked.append(key)
        if len(picked) >= k:
            break
    trace.output_count = len(picked)
    return picked, trace


def _build_leaf_blob(row: PageCatalogRow) -> str:
    """Per-candidate text blob fed to the leaf-rank LLM.

    Format is fixed and documented in `_LEAF_RANK_HIER_SYSTEM`. Prose pages
    have no column/row headers, so we substitute the keyword list as content.
    Field caps keep the per-candidate footprint bounded so a few-hundred-
    candidate prompt stays well under the context window.
    """
    title = row.table_title or "(no title)"
    lines = [f"=== {row.bulletin} p{row.page} ===",
             f"[title]    {title}"]
    if row.page_kind == "prose":
        if row.keywords:
            lines.append(f"[content]  {' | '.join(row.keywords[:20])}")
    else:
        cols = row.column_headers[:20]
        rows_sample = row.row_headers_sample[:8]
        lines.append(f"[columns]  {' | '.join(cols)}")
        lines.append(f"[rows]     {' | '.join(rows_sample)}")
    if row.dates:
        lines.append(f"[dates]    {'; '.join(row.dates[:20])}")
    return "\n".join(lines)


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


def leaf_rank_batched(
    question: str,
    concept: str,
    period: str,
    candidates: list[tuple[str, int]],
    catalog_index: dict[tuple[str, int], PageCatalogRow],
    llm: LLMClient,
    *,
    batch_size: int = 100,
    workers: int = 8,
) -> tuple[list[dict[str, Any]], LevelTrace]:
    """Run leaf_rank_postings in parallel batches of `batch_size`. Each batch
    sees a small, focused candidate list (~100); the LLM picks the strong
    matches within that batch. We then pool the per-batch picks, dedupe by
    (bulletin, page), and return the merged list.

    The returned LevelTrace aggregates input/output/latency across batches.
    Per-batch response excerpts are concatenated for offline analysis.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    trace = LevelTrace(level="leaf_rank", input_count=len(candidates), input_chars=0)
    if not candidates:
        return [], trace

    chunks: list[list[tuple[str, int]]] = [
        candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)
    ]

    def _run(idx: int, chunk: list[tuple[str, int]]):
        top, sub = leaf_rank_postings(question, concept, period, chunk,
                                      catalog_index, llm)
        return idx, top, sub

    per_batch: list[tuple[int, list[dict[str, Any]], LevelTrace]] = []
    if workers <= 1 or len(chunks) == 1:
        for i, ch in enumerate(chunks):
            per_batch.append(_run(i, ch))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run, i, ch) for i, ch in enumerate(chunks)]
            for f in as_completed(futs):
                per_batch.append(f.result())

    per_batch.sort(key=lambda t: t[0])
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    excerpts: list[str] = []
    for idx, top, sub in per_batch:
        trace.input_chars += sub.input_chars
        trace.output_chars += sub.output_chars
        trace.input_tokens = (trace.input_tokens or 0) + (sub.input_tokens or 0)
        trace.output_tokens = (trace.output_tokens or 0) + (sub.output_tokens or 0)
        trace.latency_s += sub.latency_s
        if sub.response_excerpt:
            excerpts.append(f"[batch {idx}] {sub.response_excerpt[:200]}")
        for r in top:
            key = (r["bulletin"], r["page"])
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
    trace.response_excerpt = " || ".join(excerpts)[:800]
    trace.prompt_excerpt = (
        f"(batched: {len(chunks)} chunks × ≤{batch_size}) "
        + per_batch[0][2].prompt_excerpt[:700]
    )
    trace.output_count = len(out)
    return out, trace


def retrieve_hierarchical(
    tree: dict[str, Any],
    question: str,
    concept: str,
    period: str,
    llm: LLMClient,
    catalog_index: dict[tuple[str, int], PageCatalogRow],
    *,
    uid: str | None = None,
    retrieve_idx: int = 0,
    k1: int = 8,
    k2: int = 4,
    max_leaf_candidates: int = 10000,
    skip_leaf_rank: bool = False,
) -> tuple[list[dict[str, Any]], RetrieveTrace]:
    """Three-level walker over `tree` (loaded concept_tree.json).
    L1=section_pick, L2=cluster_pick, L3=leaf_rank_postings (term_pick dropped
    in v0.3 — postings under L2-picked clusters flatten directly into the
    leaf rank). Returns (top_k, trace).
    """
    sections = tree.get("sections", {})
    trace = RetrieveTrace(
        uid=uid, retrieve_idx=retrieve_idx,
        concept=concept, period=period,
        catalog_size=sum(s["n_pages_in_section"] for s in sections.values()),
    )

    t_walk = time.monotonic()

    # L1 — section_pick. Show all sections with ≥1 cluster.
    section_listing = [
        (label, data) for label, data in sorted(sections.items())
        if data.get("n_clusters", 0) > 0
    ]
    picked_sections, t1 = section_pick(question, concept, period, section_listing, llm, k=k1)
    trace.levels.append(t1)
    if not picked_sections:
        trace.total_walk_s = time.monotonic() - t_walk
        return [], trace

    # L2 — cluster_pick. Expand picked sections' clusters (keyed by label in v0.4).
    cluster_listing: list[tuple[str, str, dict[str, Any]]] = []
    for sec in picked_sections:
        for label, cdata in sections[sec]["clusters"].items():
            cluster_listing.append((sec, label, cdata))
    picked_clusters, t2 = cluster_pick(question, concept, period, cluster_listing, llm, k=k2)
    trace.levels.append(t2)
    if not picked_clusters:
        trace.total_walk_s = time.monotonic() - t_walk
        return [], trace

    # L3 — leaf_rank. Collect the unique (bulletin, page) keys under the
    # picked clusters. The catalog_index supplies the rich per-candidate
    # blob inside leaf_rank_postings. Source-bulletin hard pins are picked
    # up by the leaf-rank LLM directly from the question text.
    candidate_keys: list[tuple[str, int]] = []
    seen_keys: set[tuple[str, int]] = set()
    for sec, cluster_label in picked_clusters:
        cdata = sections[sec]["clusters"][cluster_label]
        for posts in cdata.get("keywords", {}).values():
            for p in posts:
                key = (p["bulletin"], p["page"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                candidate_keys.append(key)
    # Hard cap on total L3 work. Within the cap, candidates are batched
    # into small chunks so each LLM call sees a focused set (~100) rather
    # than thousands.
    trace.candidate_count = len(candidate_keys)
    if len(candidate_keys) > max_leaf_candidates:
        candidate_keys = candidate_keys[:max_leaf_candidates]
    if skip_leaf_rank:
        # Recall-ceiling mode: return everything that survived L1+L2 as the
        # prediction. No LLM call, no L3 LevelTrace. Precision will be ~0;
        # the metric of interest is recall.
        top = [{"bulletin": b, "page": p, "reason": "L1+L2 only"}
               for (b, p) in candidate_keys]
    else:
        top, t3 = leaf_rank_batched(question, concept, period, candidate_keys,
                                    catalog_index, llm)
        trace.levels.append(t3)
    trace.top_k = top
    trace.total_walk_s = time.monotonic() - t_walk
    return top, trace


# ---------------------------------------------------------------------------
# Single-bucket agentic retriever
# ---------------------------------------------------------------------------
#
# Variant that commits to ONE (section, cluster) leaf instead of running a
# full L3 leaf-rank. The LLM is given access to a `sample_bucket` tool: it
# names a candidate bucket and the tool returns ≤20 random pages from that
# bucket with their keywords/title for the LLM to inspect. The LLM iterates
# until it commits to a single bucket; the prediction is then ALL pages in
# that bucket.
#
# Recall ceiling = "single bucket's coverage of the question's gold pages".

import random as _random


def sample_bucket_pages(
    tree: dict[str, Any],
    section: str,
    cluster: str,
    catalog_index: dict[tuple[str, int], PageCatalogRow],
    *,
    n: int = 20,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Return up to `n` random pages from the (section, cluster) bucket
    with their keywords and table titles, for the LLM to inspect.

    Each entry: {bulletin, page, keywords (≤8), table_title, printed_page}.
    Returns [] if the bucket doesn't exist.
    """
    sd = tree.get("sections", {}).get(section)
    if not sd:
        return []
    cd = sd.get("clusters", {}).get(cluster)
    if not cd:
        return []

    # Collect unique (bulletin, page) under this bucket.
    seen: set[tuple[str, int]] = set()
    keys: list[tuple[str, int]] = []
    for posts in cd.get("keywords", {}).values():
        for p in posts:
            k = (p["bulletin"], p["page"])
            if k in seen:
                continue
            seen.add(k)
            keys.append(k)

    if not keys:
        return []
    rng = _random.Random(seed)
    sample = rng.sample(keys, min(n, len(keys)))

    out: list[dict[str, Any]] = []
    for bulletin, page in sample:
        row = catalog_index.get((bulletin, page))
        entry: dict[str, Any] = {"bulletin": bulletin, "page": page}
        if row is not None:
            entry["printed_page"] = row.printed_page
            entry["keywords"] = list(row.keywords[:8])
            entry["table_title"] = row.table_title
        out.append(entry)
    return out


def _bucket_keys(tree: dict[str, Any], section: str, cluster: str) -> list[tuple[str, int]]:
    """All unique (bulletin, page) pairs under one bucket — used to assemble
    the prediction once the LLM commits."""
    sd = tree.get("sections", {}).get(section)
    if not sd:
        return []
    cd = sd.get("clusters", {}).get(cluster)
    if not cd:
        return []
    seen: set[tuple[str, int]] = set()
    out: list[tuple[str, int]] = []
    for posts in cd.get("keywords", {}).values():
        for p in posts:
            k = (p["bulletin"], p["page"])
            if k in seen:
                continue
            seen.add(k)
            out.append(k)
    return out


_BUCKET_PICK_SYSTEM = """You are choosing ONE (section, cluster) bucket of
U.S. Treasury Bulletin pages that most likely contains the answer to the
user's question. The bucket you commit to determines the entire retrieved
page set — there is no later rerank, so you must verify what's inside a
bucket before committing.

You will see:
  - The user's question (verbatim), the query concept tag, and the period.
  - A list of candidate buckets — (section, cluster) pairs with their
    page counts. The cluster labels are LLM-generated and often
    misleading; treat them as hints only.
  - Any bucket samples you have requested so far (≤20 random pages per
    bucket, with each page's table_title, keywords, and printed_page).

Per turn, output a SINGLE JSON object (no prose, no fences):
  {"action": "open", "section": "<exact>", "cluster": "<exact>"}
    → the next turn will include 20 random pages from that bucket.
  {"action": "commit", "section": "<exact>", "cluster": "<exact>"}
    → finalizes your choice; the entire bucket is returned.

**REQUIRED PROCEDURE:**
  1. You MUST `open` at least 2 candidate buckets BEFORE you `commit`.
     Cluster labels alone are insufficient — always inspect samples.
  2. Open the 2-4 buckets whose label or parent section best matches the
     question's concept. Compare what their sample pages actually contain
     (look at `table_title` first — that's the authoritative caption).
  3. Commit to the bucket whose samples most directly correspond to what
     the question asks about. Period dates that match the question's
     period are a strong positive signal; consistent table titles
     mentioning the question's specific concept beat parent-section name
     matches.
  4. If two buckets look equally plausible after opening, open a third
     to break the tie.
  5. You MUST commit by the final round listed in the prompt.
"""


def _safe_json(text: str) -> dict[str, Any] | None:
    s = _strip_code_fence(text)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def single_bucket_retrieve(
    tree: dict[str, Any],
    question: str,
    concept: str,
    period: str,
    llm: LLMClient,
    catalog_index: dict[tuple[str, int], PageCatalogRow],
    *,
    uid: str | None = None,
    retrieve_idx: int = 0,
    k1: int = 8,
    max_rounds: int = 6,
    sample_n: int = 20,
    sample_seed: int | None = 0,
) -> tuple[list[dict[str, Any]], RetrieveTrace]:
    """L1 section_pick → agentic bucket open/commit loop → return all pages
    under the committed bucket.

    `max_rounds` includes all open + commit turns. If the loop exits without
    a commit (LLM keeps opening or errors out), we commit to the largest
    bucket among those opened so far, or to the first cluster of the first
    picked section as a last resort.
    """
    sections = tree.get("sections", {})
    trace = RetrieveTrace(
        uid=uid, retrieve_idx=retrieve_idx,
        concept=concept, period=period,
        catalog_size=sum(s["n_pages_in_section"] for s in sections.values()),
    )

    t_walk = time.monotonic()

    # L1 — section_pick.
    section_listing = [
        (label, data) for label, data in sorted(sections.items())
        if data.get("n_clusters", 0) > 0
    ]
    picked_sections, t1 = section_pick(question, concept, period,
                                       section_listing, llm, k=k1)
    trace.levels.append(t1)
    if not picked_sections:
        trace.total_walk_s = time.monotonic() - t_walk
        return [], trace

    # Build the candidate-bucket listing the LLM will see each turn.
    candidate_buckets: list[tuple[str, str, int]] = []  # (section, cluster, n_pages)
    for sec in picked_sections:
        for label, cdata in sections[sec].get("clusters", {}).items():
            candidate_buckets.append((sec, label, cdata.get("n_pages", 0)))
    # Sort by page count desc so the LLM sees big buckets first.
    candidate_buckets.sort(key=lambda x: -x[2])

    # Agentic loop.
    opened: dict[tuple[str, str], list[dict[str, Any]]] = {}
    committed: tuple[str, str] | None = None
    loop_trace = LevelTrace(level="bucket_loop", input_count=len(candidate_buckets),
                            input_chars=0)
    t_loop = time.monotonic()
    last_user_excerpt: str = ""
    last_resp_excerpt: str = ""

    min_opens_before_commit = 2
    for round_idx in range(max_rounds):
        bucket_list = [
            {"section": s, "cluster": c, "n_pages": n}
            for s, c, n in candidate_buckets
        ]
        is_final = (round_idx == max_rounds - 1)
        opens_so_far = len(opened)
        opens_left = max(0, min_opens_before_commit - opens_so_far)
        opened_section: list[dict[str, Any]] = []
        for (s, c), pages in opened.items():
            opened_section.append({
                "section": s, "cluster": c, "n_pages_total":
                    next((n for (ss, cc, n) in candidate_buckets
                          if ss == s and cc == c), len(pages)),
                "samples": pages,
            })

        user = (
            f"Question: {question}\n\n"
            f"Concept: {concept}\n"
            f"Period:  {period}\n\n"
            f"Round {round_idx + 1} of {max_rounds}. "
            + ("YOU MUST COMMIT THIS TURN. " if is_final else "")
            + (f"You must `open` at least {opens_left} more bucket(s) "
               f"before you may commit. " if opens_left > 0 and not is_final
               else "")
            + "Pick an action below.\n\n"
            f"Candidate buckets ({len(bucket_list)}):\n"
            f"{json.dumps(bucket_list, ensure_ascii=False, indent=1)}\n\n"
            + (f"Buckets you've opened so far:\n"
               f"{json.dumps(opened_section, ensure_ascii=False, indent=1)}\n"
               if opened else "(no buckets opened yet)\n")
        )
        last_user_excerpt = user[:800]

        t0 = time.monotonic()
        resp = llm.call(system=_BUCKET_PICK_SYSTEM, user=user, temperature=0.0)
        loop_trace.latency_s += time.monotonic() - t0
        loop_trace.input_tokens = (loop_trace.input_tokens or 0) + (resp.input_tokens or 0)
        loop_trace.output_tokens = (loop_trace.output_tokens or 0) + (resp.output_tokens or 0)
        loop_trace.input_chars += len(user)
        loop_trace.output_chars += len(resp.text)
        last_resp_excerpt = resp.text[:200]

        obj = _safe_json(resp.text)
        if not obj:
            continue
        action = str(obj.get("action", "")).strip().lower()
        section = str(obj.get("section", "")).strip()
        cluster = str(obj.get("cluster", "")).strip()

        # Validate against tree.
        if section not in sections or cluster not in sections[section].get("clusters", {}):
            # Case-insensitive rescue.
            sec_lower = {s.lower(): s for s in sections}
            section = sec_lower.get(section.lower(), section)
            if section in sections:
                cl_lower = {c.lower(): c for c in sections[section].get("clusters", {})}
                cluster = cl_lower.get(cluster.lower(), cluster)
        if section not in sections or cluster not in sections[section].get("clusters", {}):
            continue

        if action == "open":
            opened[(section, cluster)] = sample_bucket_pages(
                tree, section, cluster, catalog_index,
                n=sample_n, seed=sample_seed,
            )
            continue
        if action == "commit":
            # Enforce the minimum-opens rule server-side unless this is
            # the final round (where we accept whatever they give us).
            if opens_left > 0 and not is_final:
                continue
            committed = (section, cluster)
            break

    # Fallback: largest opened bucket, or first cluster of first picked section.
    if committed is None:
        if opened:
            committed = max(
                opened.keys(),
                key=lambda k: next((n for (s, c, n) in candidate_buckets
                                    if s == k[0] and c == k[1]), 0),
            )
        elif candidate_buckets:
            s, c, _ = candidate_buckets[0]
            committed = (s, c)

    loop_trace.total_walk_s = time.monotonic() - t_loop
    loop_trace.prompt_excerpt = last_user_excerpt
    loop_trace.response_excerpt = last_resp_excerpt
    if committed:
        loop_trace.output_count = 1
    trace.levels.append(loop_trace)

    # Predicted set = every (bulletin, page) under the committed bucket.
    top: list[dict[str, Any]] = []
    if committed:
        keys = _bucket_keys(tree, committed[0], committed[1])
        top = [{"bulletin": b, "page": p,
                "reason": f"bucket={committed[0]}|{committed[1]}"}
               for (b, p) in keys]
        trace.candidate_count = len(keys)

    trace.top_k = top[:50]  # cap for trace storage
    trace.total_walk_s = time.monotonic() - t_walk
    return top, trace


# ---------------------------------------------------------------------------
# Agentic-traverse retriever — (time-bucket, section) cells
# ---------------------------------------------------------------------------
#
# Flips L1 and L2: time-bucket becomes the outer dimension, canonical section
# becomes the inner. Cells = `(time_bucket, section)` containing all pages
# from that bulletin range under that section. The agent navigates the
# 18 × 32 ≈ 200-400 non-empty cells via the same open/commit semantics as
# single_bucket_retrieve. Predicted = every page in the committed cell.
#
# Smaller cells (mean ~170 pages vs single-bucket's ~1100) mean tighter
# precision, at the cost of needing the agent to pick the right time +
# section combo together.


def _variable_bucket_label(bulletin: str) -> str | None:
    """Floor a `YYYY-MM` bulletin to a variable-width time-bucket label.

    Bucket width depends on era:
      * <1980 → 10-year buckets (sparse, slow-changing data)
      * 1980-2019 → 5-year
      * 2020+ → 2-year (fast-moving recent data)
    """
    try:
        year = int(bulletin.split("-", 1)[0])
    except (ValueError, IndexError):
        return None
    if year < 1980:
        n = 10
    elif year < 2020:
        n = 5
    else:
        n = 2
    start = (year // n) * n
    return f"{start}-{start + n - 1}"


def build_bucket_section_index(
    catalog: list[PageCatalogRow],
    banner_rewrite: dict[str, str],
) -> dict[tuple[str, str], list[tuple[str, int]]]:
    """Index `{(time_bucket, canonical_section): [(bulletin, page), ...]}`.

    Sections come from each row's `section` after `banner_rewrite`
    canonicalization. Rows with no section land under `"Unsectioned"`.
    Only `table/chart/prose` page_kinds are indexed. Bucket widths vary
    by era (see `_variable_bucket_label`).
    """
    idx: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for r in catalog:
        if r.page_kind not in ("table", "chart", "prose"):
            continue
        bucket = _variable_bucket_label(r.bulletin)
        if bucket is None:
            continue
        raw = (r.section or "").strip()
        if raw:
            section = banner_rewrite.get(raw.lower(), raw)
        else:
            section = "Unsectioned"
        idx.setdefault((bucket, section), []).append((r.bulletin, r.page))
    return idx


def sample_cell_pages(
    cell_index: dict[tuple[str, str], list[tuple[str, int]]],
    bucket: str,
    section: str,
    catalog_index: dict[tuple[str, int], PageCatalogRow],
    *,
    n: int = 20,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Return up to `n` random pages from a (bucket, section) cell with
    table_title / keywords[:8] / printed_page for LLM inspection."""
    keys = cell_index.get((bucket, section), [])
    if not keys:
        return []
    rng = _random.Random(seed)
    sample = rng.sample(keys, min(n, len(keys)))
    out: list[dict[str, Any]] = []
    for bulletin, page in sample:
        row = catalog_index.get((bulletin, page))
        entry: dict[str, Any] = {"bulletin": bulletin, "page": page}
        if row is not None:
            entry["printed_page"] = row.printed_page
            entry["keywords"] = list(row.keywords[:8])
            entry["table_title"] = row.table_title
        out.append(entry)
    return out


_TRAVERSE_SYSTEM = """You navigate a U.S. Treasury Bulletin index whose
cells are indexed by (time_bucket, section). Time buckets are
variable-width spans of the bulletin's publication year — 10 years for
pre-1980 buckets ("1940-1949"), 5 years for 1980-2019 ("1985-1989"),
and 2 years for 2020+ ("2024-2025"). Sections are canonical Treasury
chapter names (e.g. "Federal fiscal operations", "Capital movements").
Each cell contains every page from that bulletin range under that
section.

Your job: commit to 1-3 (time_bucket, section) cells. The prediction is
every page in those cells (union); there is no later rerank.

You will see:
  - The user's question (verbatim), the query concept tag, and the period.
  - All non-empty cells with their (time_bucket, section, n_pages) — sorted
    by page count descending.
  - Any cells you have opened so far (≤20 random pages with table_title,
    keywords, printed_page).

Per turn, output a SINGLE JSON object (no prose, no fences):
  {"action": "open", "time_bucket": "<YYYY-YYYY>", "section": "<exact>"}
    → next turn includes 20 random pages from that cell.
  {"action": "commit", "cells": [
      {"time_bucket": "<YYYY-YYYY>", "section": "<exact>"},
      ... (1-3 cells, all under the same section is the usual case)
   ]}
    → finalizes your choice; the union of cells' pages is returned.

  Alternate single-cell commit shape (back-compat):
  {"action": "commit", "time_bucket": "<YYYY-YYYY>", "section": "<exact>"}

**REQUIRED PROCEDURE:**
  1. The query period tells you the LIKELY time_bucket(s) to inspect first.
     If period is "2025-03", explore "2024-2025". If period is
     "FY1990..FY1998", that range crosses bucket boundaries — explore
     "1990-1994" and "1995-1999". Bulletins often retrospectively
     publish data in the following year, so the bucket AFTER the period
     can also be relevant.
  2. You MUST `open` at least 2 candidate cells BEFORE you `commit`.
     Don't rely on section names alone — inspect the sampled pages.
  3. Open cells whose time_bucket aligns with the period AND whose
     section best matches the question's concept. The `table_title` and
     `keywords` of the sampled pages are the authoritative signal.
  4. When the period clearly stays inside ONE time_bucket, commit to 1
     cell. When the period spans MULTIPLE buckets (e.g. "1972-1976",
     "1996-2000"), commit to 2-3 cells under the SAME section covering
     each relevant bucket. Cap at 3 cells.
  5. You MUST commit by the final round listed in the prompt.
"""


def agentic_traverse_retrieve(
    tree: dict[str, Any],
    question: str,
    concept: str,
    period: str,
    llm: LLMClient,
    catalog_index: dict[tuple[str, int], PageCatalogRow],
    cell_index: dict[tuple[str, str], list[tuple[str, int]]],
    *,
    uid: str | None = None,
    retrieve_idx: int = 0,
    max_rounds: int = 8,
    sample_n: int = 20,
    sample_seed: int | None = 0,
) -> tuple[list[dict[str, Any]], RetrieveTrace]:
    """Agentic open/commit traversal over (time_bucket, section) cells.

    No `section_pick` prefix — the agent sees every non-empty cell from
    the start and decides which to inspect. Predicted set = pages in the
    committed cell.
    """
    sections = tree.get("sections", {})
    trace = RetrieveTrace(
        uid=uid, retrieve_idx=retrieve_idx,
        concept=concept, period=period,
        catalog_size=sum(s["n_pages_in_section"] for s in sections.values()),
    )
    t_walk = time.monotonic()

    # Candidate cells sorted by page count desc.
    cell_listing = sorted(
        ((b, s, len(pages)) for (b, s), pages in cell_index.items()),
        key=lambda x: -x[2],
    )

    opened: dict[tuple[str, str], list[dict[str, Any]]] = {}
    committed: list[tuple[str, str]] = []
    loop_trace = LevelTrace(level="traverse_loop",
                            input_count=len(cell_listing), input_chars=0)
    t_loop = time.monotonic()
    last_user_excerpt: str = ""
    last_resp_excerpt: str = ""

    min_opens_before_commit = 2
    steps_log: list[dict[str, Any]] = []
    for round_idx in range(max_rounds):
        is_final = (round_idx == max_rounds - 1)
        opens_so_far = len(opened)
        opens_left = max(0, min_opens_before_commit - opens_so_far)

        cell_list = [
            {"time_bucket": b, "section": s, "n_pages": n}
            for b, s, n in cell_listing
        ]
        opened_payload: list[dict[str, Any]] = []
        for (b, s), pages in opened.items():
            opened_payload.append({
                "time_bucket": b, "section": s,
                "n_pages_total": next((n for (bb, ss, n) in cell_listing
                                       if bb == b and ss == s), len(pages)),
                "samples": pages,
            })

        user = (
            f"Question: {question}\n\n"
            f"Concept: {concept}\n"
            f"Period:  {period}\n\n"
            f"Round {round_idx + 1} of {max_rounds}. "
            + ("YOU MUST COMMIT THIS TURN. " if is_final else "")
            + (f"You must `open` at least {opens_left} more cell(s) "
               f"before you may commit. " if opens_left > 0 and not is_final
               else "")
            + "Pick an action below.\n\n"
            f"All non-empty cells ({len(cell_list)}):\n"
            f"{json.dumps(cell_list, ensure_ascii=False, indent=1)}\n\n"
            + (f"Cells you've opened so far:\n"
               f"{json.dumps(opened_payload, ensure_ascii=False, indent=1)}\n"
               if opened else "(no cells opened yet)\n")
        )
        last_user_excerpt = user[:800]

        t0 = time.monotonic()
        resp = llm.call(system=_TRAVERSE_SYSTEM, user=user, temperature=0.0)
        loop_trace.latency_s += time.monotonic() - t0
        loop_trace.input_tokens = (loop_trace.input_tokens or 0) + (resp.input_tokens or 0)
        loop_trace.output_tokens = (loop_trace.output_tokens or 0) + (resp.output_tokens or 0)
        loop_trace.input_chars += len(user)
        loop_trace.output_chars += len(resp.text)
        last_resp_excerpt = resp.text[:200]

        obj = _safe_json(resp.text)
        if not obj:
            steps_log.append({"round": round_idx + 1, "action": "invalid_json",
                              "opens_so_far": opens_so_far,
                              "rejected_reason": "could not parse JSON"})
            continue
        action = str(obj.get("action", "")).strip().lower()

        def _validate(bucket: str, section: str) -> tuple[str, str] | None:
            """Return (bucket, section) if it's a known cell, with a
            case-insensitive rescue on section. None if not found."""
            if (bucket, section) in cell_index:
                return (bucket, section)
            sec_lower = {s.lower(): s for (b, s) in cell_index if b == bucket}
            section = sec_lower.get(section.lower(), section)
            if (bucket, section) in cell_index:
                return (bucket, section)
            return None

        if action == "open":
            bucket = str(obj.get("time_bucket", "")).strip()
            section = str(obj.get("section", "")).strip()
            key = _validate(bucket, section)
            if key is None:
                steps_log.append({
                    "round": round_idx + 1, "action": "open",
                    "cells": [{"time_bucket": bucket, "section": section}],
                    "opens_so_far": opens_so_far,
                    "rejected_reason": "unknown cell",
                })
                continue
            samples = sample_cell_pages(
                cell_index, key[0], key[1], catalog_index,
                n=sample_n, seed=sample_seed,
            )
            opened[key] = samples
            steps_log.append({
                "round": round_idx + 1, "action": "open",
                "cells": [{"time_bucket": key[0], "section": key[1]}],
                "opens_so_far": opens_so_far + 1,
                "sample_titles": [s.get("table_title") for s in samples[:5]],
                "rejected_reason": None,
            })
            continue

        if action == "commit":
            # Parse the cells array (preferred) or the single-cell shape.
            raw_cells: list[dict[str, Any]] = []
            cells_field = obj.get("cells")
            if isinstance(cells_field, list) and cells_field:
                raw_cells = [c for c in cells_field if isinstance(c, dict)]
            else:
                raw_cells = [{
                    "time_bucket": obj.get("time_bucket"),
                    "section": obj.get("section"),
                }]

            chosen: list[tuple[str, str]] = []
            seen: set[tuple[str, str]] = set()
            requested_cells: list[dict[str, Any]] = []
            for c in raw_cells[:3]:  # hard cap at 3 cells
                bucket = str(c.get("time_bucket", "")).strip()
                section = str(c.get("section", "")).strip()
                requested_cells.append({"time_bucket": bucket, "section": section})
                key = _validate(bucket, section)
                if key is None or key in seen:
                    continue
                seen.add(key)
                chosen.append(key)

            if opens_left > 0 and not is_final:
                steps_log.append({
                    "round": round_idx + 1, "action": "commit",
                    "cells": requested_cells,
                    "opens_so_far": opens_so_far,
                    "rejected_reason": f"need {opens_left} more opens",
                })
                continue

            if not chosen:
                steps_log.append({
                    "round": round_idx + 1, "action": "commit",
                    "cells": requested_cells,
                    "opens_so_far": opens_so_far,
                    "rejected_reason": "no valid cells in commit",
                })
                continue

            steps_log.append({
                "round": round_idx + 1, "action": "commit",
                "cells": [{"time_bucket": b, "section": s} for (b, s) in chosen],
                "opens_so_far": opens_so_far,
                "rejected_reason": None,
            })
            committed = chosen
            break

    if not committed:
        if opened:
            committed = [max(
                opened.keys(),
                key=lambda k: len(cell_index.get(k, [])),
            )]
        elif cell_listing:
            committed = [(cell_listing[0][0], cell_listing[0][1])]

    loop_trace.total_walk_s = time.monotonic() - t_loop
    loop_trace.prompt_excerpt = last_user_excerpt
    loop_trace.response_excerpt = last_resp_excerpt
    loop_trace.output_count = len(committed)
    loop_trace.steps = steps_log
    trace.levels.append(loop_trace)

    top: list[dict[str, Any]] = []
    seen_pages: set[tuple[str, int]] = set()
    candidate_total = 0
    for cell in committed:
        keys = cell_index.get(cell, [])
        candidate_total += len(keys)
        for (b, p) in keys:
            if (b, p) in seen_pages:
                continue
            seen_pages.add((b, p))
            top.append({"bulletin": b, "page": p,
                        "reason": f"cell={cell[0]}|{cell[1]}"})
    trace.candidate_count = candidate_total
    trace.top_k = top[:50]
    trace.total_walk_s = time.monotonic() - t_walk
    return top, trace


# ---------------------------------------------------------------------------
# Parent-chapter rollup of the 32 canonical sections (E13)
# ---------------------------------------------------------------------------
#
# Treasury Bulletin's actual ToC is hierarchical. The 32 canonical sections
# from E7 are a flattened mix of parent chapters and their sub-headings. The
# map below rolls each canonical section back up to its parent chapter so
# section_pick has a tighter, less-ambiguous vocab.

_PARENT_CHAPTER_MAP: dict[str, str] = {
    # Federal Fiscal Operations chapter — budget, receipts, expenditures, gov
    # accounts.
    "Federal fiscal operations": "Federal Fiscal Operations",
    "Budget receipts and expenditures": "Federal Fiscal Operations",
    "Account of the U.S. Treasury": "Federal Fiscal Operations",
    "Cash income and outgo": "Federal Fiscal Operations",
    "Internal revenue collections": "Federal Fiscal Operations",
    "Federal obligations": "Federal Fiscal Operations",

    # Federal Debt chapter — issuance, outstanding, ownership, yields, savings
    # bonds (all about who owns/issues Treasury debt).
    "Federal debt": "Federal Debt",
    "Public debt operations": "Federal Debt",
    "Debt operations": "Federal Debt",
    "Debt outstanding": "Federal Debt",
    "Ownership of Federal securities": "Federal Debt",
    "Treasury survey of ownership": "Federal Debt",
    "Treasury survey of ownership of Federal securities": "Federal Debt",
    "Market quotations on Treasury securities": "Federal Debt",
    "Average yields of long-term bonds": "Federal Debt",
    "United States savings bonds": "Federal Debt",

    # Capital Movements chapter — foreign capital flows.
    "Capital movements": "Capital Movements",
    "CAPITAL MOVEMENTS BETWEEN U.S. AND FOREIGN COUNTRIES": "Capital Movements",

    # Foreign Currency Positions chapter — currency holdings, ESF.
    "Foreign currency positions": "Foreign Currency Positions",
    "Exchange Stabilization Fund": "Foreign Currency Positions",

    # International Financial Statistics chapter — int'l + monetary stats.
    "International financial statistics": "International Financial Statistics",
    "Monetary statistics": "International Financial Statistics",

    # Trust Funds chapter.
    "TRUST FUNDS": "Trust Funds",
    "Trust account and other transactions": "Trust Funds",

    # Government Corporations and Other Business-Type Activities chapter.
    "Financial operations of Government agencies and funds": "Government Corporations and Business-Type Activities",
    "GOVERNMENT CORPORATIONS AND OTHER BUSINESS-TYPE ACTIVITIES": "Government Corporations and Business-Type Activities",
    "Corporations and certain other business-type activities - statements of financial condition": "Government Corporations and Business-Type Activities",
    "Corporations and certain other business-type activities - income and expense, and source and application of funds": "Government Corporations and Business-Type Activities",

    # Profile of the Economy chapter — meta/special articles.
    "PROFILE OF THE ECONOMY": "Profile of the Economy",
    "Special Article": "Profile of the Economy",

    # Other / catch-all.
    "Unsectioned": "Unsectioned",
    "Cumulative Table of Contents": "Unsectioned",
}


def parent_chapter_index(tree: dict[str, Any]) -> dict[str, list[str]]:
    """Group canonical sections by parent chapter.

    Returns `{parent_chapter: [canonical_section, ...]}`. Any canonical
    section not in `_PARENT_CHAPTER_MAP` falls through as its own parent
    (no merge).
    """
    out: dict[str, list[str]] = {}
    for section in tree.get("sections", {}):
        parent = _PARENT_CHAPTER_MAP.get(section, section)
        out.setdefault(parent, []).append(section)
    return out


# ---------------------------------------------------------------------------
# One-shot section retriever — diagnostic baseline
# ---------------------------------------------------------------------------
#
# Strips everything except `section_pick(k=1)`: one LLM call, exactly one
# canonical section returned, predicted set = every page in that section.
# Used to measure how often the agent can correctly identify the gold's
# canonical section in a single shot.


def _section_pages(tree: dict[str, Any], section: str) -> list[tuple[str, int]]:
    sd = tree.get("sections", {}).get(section)
    if not sd:
        return []
    seen: set[tuple[str, int]] = set()
    out: list[tuple[str, int]] = []
    for cd in sd.get("clusters", {}).values():
        for posts in cd.get("keywords", {}).values():
            for p in posts:
                k = (p["bulletin"], p["page"])
                if k in seen:
                    continue
                seen.add(k)
                out.append(k)
    return out


_PARENT_PICK_SYSTEM = """You pick which Treasury Bulletin parent chapter
most likely contains the answer to the user's question.

You will see the question, a concept tag, a period, and a list of parent
chapters with their total page counts and the canonical sub-sections each
chapter contains. Each chapter spans many sub-sections — picking a chapter
returns every page under all of its sub-sections.

Output a SINGLE JSON object (no prose, no fences):
  {"picked": "<exact chapter label>"}

Rules:
  - Return exactly ONE chapter label, the best match.
  - Use the EXACT chapter label as shown.
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
    """One LLM call → one parent chapter → union of pages across all of its
    sub-sections. Tests whether the agent can pick the right parent
    chapter when the sub-section granularity is hidden."""
    sections = tree.get("sections", {})
    trace = RetrieveTrace(
        uid=uid, retrieve_idx=retrieve_idx,
        concept=concept, period=period,
        catalog_size=sum(s["n_pages_in_section"] for s in sections.values()),
    )
    t_walk = time.monotonic()

    chapters = parent_chapter_index(tree)
    # Build the listing: {chapter: {n_pages, sub_sections}}.
    listing: list[dict[str, Any]] = []
    for chapter in sorted(chapters):
        subs = chapters[chapter]
        n_pages = sum(
            sections[s]["n_pages_in_section"] for s in subs if s in sections
        )
        listing.append({
            "chapter": chapter,
            "n_pages": n_pages,
            "sub_sections": subs,
        })
    listing.sort(key=lambda x: -x["n_pages"])

    level_trace = LevelTrace(level="parent_pick",
                             input_count=len(listing), input_chars=0)
    user = (
        f"Question: {question}\n\n"
        f"Concept: {concept}\n"
        f"Period:  {period}\n\n"
        f"Parent chapters ({len(listing)}):\n"
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
    # Accept both shapes: {"picked": "X"} or {"picked": ["X", "Y"]}.
    raw_picked = obj.get("picked", "")
    if isinstance(raw_picked, str):
        raw_picks = [raw_picked]
    elif isinstance(raw_picked, list):
        raw_picks = [str(x).strip() for x in raw_picked if str(x).strip()]
    else:
        raw_picks = []
    # Cap to 2; case-insensitive rescue.
    valid = {c.lower(): c for c in chapters}
    # Also accept sub-section names (LLM sometimes returns the more specific
    # label it saw in the listing); map them back to their parent chapter.
    sub_to_chapter = {s.lower(): ch for ch, subs in chapters.items() for s in subs}
    picked_chapters: list[str] = []
    seen_picks: set[str] = set()
    for raw in raw_picks[:2]:
        key = raw.strip().lower()
        ch = valid.get(key) or sub_to_chapter.get(key)
        if ch and ch not in seen_picks:
            picked_chapters.append(ch)
            seen_picks.add(ch)
    trace.levels.append(level_trace)

    if not picked_chapters:
        trace.total_walk_s = time.monotonic() - t_walk
        return [], trace

    # Union pages across all sub-sections of every picked chapter.
    seen_pages: set[tuple[str, int]] = set()
    top: list[dict[str, Any]] = []
    for chapter in picked_chapters:
        for sub in chapters[chapter]:
            for (b, p) in _section_pages(tree, sub):
                if (b, p) in seen_pages:
                    continue
                seen_pages.add((b, p))
                top.append({"bulletin": b, "page": p,
                            "reason": f"chapter={chapter}|sub={sub}"})
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
    """One LLM call → one canonical section → all pages in that section.

    Pure baseline measurement of section_pick accuracy.
    """
    sections = tree.get("sections", {})
    trace = RetrieveTrace(
        uid=uid, retrieve_idx=retrieve_idx,
        concept=concept, period=period,
        catalog_size=sum(s["n_pages_in_section"] for s in sections.values()),
    )

    t_walk = time.monotonic()
    section_listing = [
        (label, data) for label, data in sorted(sections.items())
        if data.get("n_clusters", 0) > 0
    ]
    picked, t1 = section_pick(question, concept, period, section_listing, llm, k=1)
    trace.levels.append(t1)
    if not picked:
        trace.total_walk_s = time.monotonic() - t_walk
        return [], trace

    section = picked[0]
    keys = _section_pages(tree, section)
    top = [{"bulletin": b, "page": p, "reason": f"section={section}"}
           for (b, p) in keys]
    trace.candidate_count = len(keys)
    trace.top_k = top[:50]
    trace.total_walk_s = time.monotonic() - t_walk
    return top, trace


# ---------------------------------------------------------------------------
# Legacy flat driver — orchestrates prefilter + walker levels, captures RetrieveTrace
# ---------------------------------------------------------------------------

def retrieve_for_op(
    catalog: list[PageCatalogRow],
    question: str,
    concept: str,
    period: str,
    llm: LLMClient,
    *,
    uid: str | None = None,
    retrieve_idx: int = 0,
) -> tuple[list[PageCatalogRow], list[dict[str, Any]], RetrieveTrace]:
    """Run the probe end-to-end: prefilter then walker levels.
    Returns (survivors, top_k, trace).
    """
    trace = RetrieveTrace(
        uid=uid,
        retrieve_idx=retrieve_idx,
        concept=concept,
        period=period,
        catalog_size=len(catalog),
    )

    t0 = time.monotonic()
    survivors = period_prefilter(catalog, period)
    trace.prefilter_s = time.monotonic() - t0
    trace.candidate_count = len(survivors)

    t_walk = time.monotonic()
    top, leaf_trace = leaf_rank(question, concept, period, survivors, llm)
    trace.levels.append(leaf_trace)
    trace.total_walk_s = time.monotonic() - t_walk
    trace.top_k = top
    return survivors, top, trace


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
