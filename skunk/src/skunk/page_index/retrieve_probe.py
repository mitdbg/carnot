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


@dataclass
class RetrieveTrace:
    uid: str | None
    retrieve_idx: int
    concept: str
    period: str
    source_bulletin: str | None
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
    source_bulletin: str | None = None,
) -> list[PageCatalogRow]:
    """Keep rows whose periods_covered overlaps the query period.
    Restricts to page_kind in {table, chart}. Optional source_bulletin pin."""
    intervals = period_to_intervals(period)
    out: list[PageCatalogRow] = []
    for row in catalog:
        if row.page_kind not in ("table", "chart"):
            continue
        if source_bulletin is not None and row.bulletin != source_bulletin:
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
period, and a list of cluster labels (with their parent section). Each cluster
groups related Treasury terms; by picking one you commit to ranking its terms.

Output a SINGLE JSON object (no prose, no fences):
  {"picked": [
      {"section": "<section label>", "cluster_id": "<Cnn>"},
      ...
    ]}

Rules:
  - Return AT MOST K (default 3) entries, best-to-worst. Use EXACT section labels
    and cluster_ids as shown.
  - Prefer clusters whose label or central_terms name the concept directly.
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


_LEAF_RANK_HIER_SYSTEM = """You are given a flat list of candidate Treasury Bulletin
pages reached via the concept walk. Each entry is `(description, bulletin, page)`
where `description` is the page's verbatim table title — often including a date
string like 'December 31, 1949' or 'February 20, 1952'.

Pick the top-K pages that best answer the user's question. The question's period
(e.g. '1949-12', '1952-02-29', '1942-03..1948-10') needs to match against the
date string inside each description. Treat day-of-month as fuzzy: a question
about 'February 28, 1952' matches a description about 'February 20, 1952' —
both denote the same monthly snapshot.

Output a SINGLE JSON object (no prose, no fences):
  {"ranked": [
      {"bulletin": "YYYY-MM", "page": <int>, "reason": "<short rationale>"}
    ]}

Rules:
  - Return AT MOST 5 entries, best-to-worst.
  - Match the description's date string to the question's period. Prefer
    descriptions whose date falls inside (or at the closing month of) the period.
  - A bulletin published months AFTER the question's period is the canonical
    retrospective home — don't reject it for that reason; it's expected.
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
    valid = {label for label, _ in sections}
    picked = []
    for s in _picked_set(obj, "picked"):
        s = str(s).strip()
        if s in valid and s not in picked:
            picked.append(s)
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
    """clusters = [(section_label, cluster_id, cluster_data)]. Return up to k (section, cluster_id) tuples."""
    trace = LevelTrace(level="cluster_pick", input_count=len(clusters), input_chars=0)
    if not clusters:
        return [], trace

    listing = [
        {"section": section, "cluster_id": cid, "label": cdata["label"],
         "central_terms": cdata["central_terms"],
         "n_terms": cdata["n_terms"], "n_pages": cdata["n_pages"]}
        for section, cid, cdata in clusters
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
    valid = {(s, c) for s, c, _ in clusters}
    picked: list[tuple[str, str]] = []
    for entry in _picked_set(obj, "picked"):
        if not isinstance(entry, dict):
            continue
        key = (str(entry.get("section", "")), str(entry.get("cluster_id", "")))
        if key in valid and key not in picked:
            picked.append(key)
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


def leaf_rank_postings(
    question: str,
    concept: str,
    period: str,
    postings: list[dict[str, Any]],
    llm: LLMClient,
    *,
    k: int = 5,
) -> tuple[list[dict[str, Any]], LevelTrace]:
    """postings = [{bulletin, page, description}]. Return top-K (bulletin, page, reason)."""
    trace = LevelTrace(level="leaf_rank", input_count=len(postings), input_chars=0)
    if not postings:
        return [], trace

    user = (
        f"Question: {question}\n\n"
        f"Concept: {concept}\n"
        f"Period:  {period}\n"
        f"K (max picks): {k}\n\n"
        f"Candidate pages ({len(postings)}):\n"
        f"{json.dumps(postings, ensure_ascii=False, indent=1)}\n"
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


def retrieve_hierarchical(
    tree: dict[str, Any],
    question: str,
    concept: str,
    period: str,
    source_bulletin: str | None,
    llm: LLMClient,
    *,
    uid: str | None = None,
    retrieve_idx: int = 0,
    k1: int = 8,
    k2: int = 4,
    k3: int = 5,
    k4: int = 5,
) -> tuple[list[dict[str, Any]], RetrieveTrace]:
    """Four-level walker over `tree` (loaded concept_tree.json).
    Returns (top_k, trace). `source_bulletin` is honored as a post-L4 filter
    when set (we still walk the full tree to learn which clusters matter).
    """
    sections = tree.get("sections", {})
    trace = RetrieveTrace(
        uid=uid, retrieve_idx=retrieve_idx,
        concept=concept, period=period, source_bulletin=source_bulletin,
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

    # L2 — cluster_pick. Expand picked sections' clusters.
    cluster_listing: list[tuple[str, str, dict[str, Any]]] = []
    for sec in picked_sections:
        for cid, cdata in sections[sec]["clusters"].items():
            cluster_listing.append((sec, cid, cdata))
    picked_clusters, t2 = cluster_pick(question, concept, period, cluster_listing, llm, k=k2)
    trace.levels.append(t2)
    if not picked_clusters:
        trace.total_walk_s = time.monotonic() - t_walk
        return [], trace

    # L3 — term_pick. Expand picked clusters' terms.
    term_listing: list[tuple[str, str, str, list[dict[str, Any]]]] = []
    for sec, cid in picked_clusters:
        cdata = sections[sec]["clusters"][cid]
        for term, postings in cdata["terms"].items():
            term_listing.append((sec, cid, term, postings))
    picked_terms, t3 = term_pick(question, concept, period, term_listing, llm, k=k3)
    trace.levels.append(t3)
    if not picked_terms:
        trace.total_walk_s = time.monotonic() - t_walk
        return [], trace

    # L4 — leaf_rank. Collect postings, dedupe, optional bulletin pin, rank.
    postings_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for sec, cid, term in picked_terms:
        cdata = sections[sec]["clusters"][cid]
        for p in cdata["terms"][term]:
            key = (p["bulletin"], p["page"])
            if key in postings_by_key:
                continue  # first description wins (deterministic)
            if source_bulletin is not None and p["bulletin"] != source_bulletin:
                continue
            postings_by_key[key] = {
                "bulletin": p["bulletin"],
                "page": p["page"],
                "description": p["description"],
            }
    flat_postings = list(postings_by_key.values())
    top, t4 = leaf_rank_postings(question, concept, period, flat_postings, llm, k=k4)
    trace.levels.append(t4)
    trace.candidate_count = len(flat_postings)
    trace.top_k = top
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
    source_bulletin: str | None,
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
        source_bulletin=source_bulletin,
        catalog_size=len(catalog),
    )

    t0 = time.monotonic()
    survivors = period_prefilter(catalog, period, source_bulletin)
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
