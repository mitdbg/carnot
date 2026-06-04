"""Query path — two-stage parallel semantic filter.

The third pass of the page-index query path
(`ToC pick → year filter → semantic filter → candidate set`). Given the
year-filtered survivors of one retrieve branch, prune them down to a
tight candidate set the extractor can read.

Two cascading stages, both batched and run in parallel across batches:

  Stage A — COARSE (cheap, aggressive prune):
      Input: page METADATA only (titles, column/row headers, dates,
      keywords) drawn straight from the catalog row — no PDF read.
      Output: a bool per page. Aggressively rejects topic/period/
      granularity mismatches.

  Stage B — FINE (precise, on stage-A survivors only):
      Input: full page plain text + a thin structural shim.
      Output: a value-quote + one-sentence justification BEFORE the
      bool, so the model commits its evidence on paper before deciding.

Missing ids in a batch response default to `relevant=True` (recall-safe
on the occasional truncated large-batch response).

Knobs live on `SkunkConfig` (`semfilter_*`). The stages call
`ctx.llm_client.acall` directly; batch fan-out is `asyncio.gather` (no
hand-rolled retry — the client owns retries).
"""

from __future__ import annotations

import asyncio
import json
import re
from functools import lru_cache

from skunk.common import ExecutionContext

from skunk.corpus import page_elements, page_plain_text, parsed_json_dir

from .schema import PageCatalogRow
from .util import safe_json_loads

_PARSED_DIR = parsed_json_dir()


# ---------------------------------------------------------------------------
# Page-text loading (lazy, lru-cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _bulletin_pages(bulletin: str) -> dict[int, list[dict]]:
    """Per-bulletin `{page → elements}` map, cached so the fine stage scans
    each bulletin's parsed doc once rather than once per surviving page."""
    try:
        return page_elements(bulletin, base_dir=_PARSED_DIR)
    except FileNotFoundError:
        return {}


@lru_cache(maxsize=4096)
def _page_text_cached(bulletin: str, page: int, max_chars: int) -> str:
    text = page_plain_text(_bulletin_pages(bulletin).get(page, []))
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
    """Fine-stage input: full page plain text plus a thin metadata shim
    (titles + headers + dates) for structural anchors."""
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


def _parse_batch_decisions(text: str) -> dict[int, bool]:
    """Extract `{id: relevant}`. Tolerates an optional `justification` /
    `value_quote` field alongside `relevant`. Falls back to a regex
    scan when the JSON is truncated. Missing ids are absent from the
    dict; callers default them to True."""
    obj = safe_json_loads(text, context="semfilter")
    if isinstance(obj, dict):
        out: dict[int, bool] = {}
        for d in obj.get("decisions") or []:
            try:
                idx = int(d["id"])
            except (KeyError, TypeError, ValueError):
                continue
            out[idx] = bool(d.get("relevant", True))
        if out:
            return out
    # Truncated / non-object response — scan for id/relevant pairs.
    out = {}
    for m in re.finditer(
        r"\"id\"\s*:\s*(\d+).*?\"relevant\"\s*:\s*(true|false)",
        text, re.IGNORECASE | re.DOTALL,
    ):
        out[int(m.group(1))] = m.group(2).lower() == "true"
    return out


# ---------------------------------------------------------------------------
# Stage runner (one batch group → kept set)
# ---------------------------------------------------------------------------

def _chunk(seq: list, n: int) -> list[list]:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


async def _run_stage(
    survivors: list[tuple[str, int]],
    catalog_index: dict[tuple[str, int], PageCatalogRow],
    key: str,
    period: str | None,
    page_block_fn,
    system_prompt: str,
    ctx: ExecutionContext,
    *,
    batch_size: int,
    workers: int,
) -> list[tuple[str, int]]:
    """Run one filter stage. `page_block_fn(row)` returns the dict the LLM
    sees for that page. Returns the kept pages in input order."""
    if not survivors:
        return []

    blocks: list[dict] = [
        {"id": j, **page_block_fn(catalog_index[pk])}
        for j, pk in enumerate(survivors)
    ]
    batches = _chunk(blocks, batch_size)

    kept: set[tuple[str, int]] = set()

    async def _one_batch(batch_pages: list[dict]):
        user = _build_batch_user_prompt(key, period, batch_pages)
        resp = await ctx.llm_client.acall(system=system_prompt, user=user, temperature=0.0)
        return _parse_batch_decisions(resp.text), batch_pages

    # `workers` no longer caps threads (all batches are coroutines on one loop);
    # the async LLM rate limiter paces concurrency. The set is mutated only in
    # this single-threaded gather aftermath, so no lock is needed.
    for parsed, batch_pages in await asyncio.gather(*[_one_batch(b) for b in batches]):
        for p in batch_pages:
            # Default missing ids to True (recall-safe on the
            # occasional truncated large-batch response).
            if parsed.get(p["id"], True):
                kept.add((p["bulletin"], p["page"]))

    return [pk for pk in survivors if pk in kept]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def semantic_filter(
    survivors: list[tuple[str, int]],
    catalog_index: dict[tuple[str, int], PageCatalogRow],
    key: str,
    period: str | None,
    ctx: ExecutionContext,
) -> tuple[list[tuple[str, int]], dict]:
    """Coarse → fine cascade over one branch's year-filtered survivors.

    Returns `(kept_pages, meta)` where `kept_pages` preserves input order
    and `meta` carries per-stage sizes for the trace.
    """
    cfg = ctx.config
    coarse_kept = await _run_stage(
        survivors, catalog_index, key, period,
        _page_meta_block, _COARSE_SYSTEM_PROMPT, ctx,
        batch_size=cfg.semfilter_batch_size, workers=cfg.semfilter_workers,
    )
    fine_kept = await _run_stage(
        coarse_kept, catalog_index, key, period,
        lambda row: _page_full_block(row, max_chars=cfg.semfilter_max_page_chars),
        _FINE_SYSTEM_PROMPT, ctx,
        batch_size=cfg.semfilter_batch_size, workers=cfg.semfilter_workers,
    )
    meta = {
        "enabled": True,
        "pre": len(survivors),
        "coarse": len(coarse_kept),
        "fine": len(fine_kept),
    }
    return fine_kept, meta
