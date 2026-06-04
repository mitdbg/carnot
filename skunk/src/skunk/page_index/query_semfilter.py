"""Query path — coarse semantic filter (third pass of the page-index query path).

`ToC pick → year filter → semantic filter → candidate set`. Given the year-filtered
survivors of one retrieve branch, a SINGLE coarse pass judges each page from its metadata
SUMMARY only — titles, column/row labels, dates, keywords drawn straight from the catalog
row (no full text, no PDF read). Output is one positional true/false per page.

Verdicts are cached per (model, question, page) and parity-checked: a response whose length
doesn't match the batch is re-issued from scratch, and any page left undecided defaults to
kept (recall-safe). Knobs live on `SkunkConfig` (`semfilter_{enabled,model,batch_size}`).
The stage calls `ctx.llm_client.acall` directly; batch fan-out is `asyncio.gather` (the
client owns retries + rate limiting).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from skunk.common import ExecutionContext

from .schema import PageCatalogRow
from .util import safe_json_loads

log = logging.getLogger(__name__)

# Max times to re-issue a batch whose response length doesn't match the input
# (a positional-array misalignment that would otherwise flip verdicts).
_PARITY_MAX_RETRIES = 3


def _page_meta_block(row: PageCatalogRow) -> dict:
    """The page's metadata SUMMARY — the only thing the filter sees (no full text, no
    truncation). (`row_headers_sample` is itself a build-time sample stored in the
    catalog, so fuller row labels would need a catalog rebuild, not a change here.)"""
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
        "titles": titles,
        "column_headers": cols,
        "row_labels_sample": rows,
        "dates": row.dates,
        "keywords": row.keywords,
    }


_COARSE_SYSTEM_PROMPT = """\
For each candidate Treasury Bulletin page, decide whether it could plausibly help answer the question below.
You see only a compact SUMMARY of each page — its table titles, column and row labels, dates, and keywords (not the page's actual numbers).
A page plausibly helps if its summary suggests the page reports the kind of data the question needs, for a time period the question could use.
When the summary plausibly matches, keep the page (true). Mark a page false when its summary is clearly unrelated to what the question asks about.

Output a single JSON array, one entry per input page, of true/false on whether the page plausibly helps.
"""


def _chunk(seq: list, n: int) -> list[list]:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def _build_question_prompt(question: str, pages: list[dict]) -> str:
    # Double-attention: repeat the instructions after the data so the model re-reads
    # them right before answering; the output/format line stays LAST.
    return (
        f"question: {question}\n\n"
        f"pages (JSON, {len(pages)} entries):\n"
        f"{json.dumps(pages, ensure_ascii=False, indent=1)}\n\n"
        f"{_COARSE_SYSTEM_PROMPT}\n"
        f"Return a JSON array of exactly {len(pages)} true/false values, "
        f"one per page in the order given."
    )


def _parse_bool_list(text: str) -> list[bool]:
    """Raw positional bools from a model response — NO padding/truncation, so the
    caller can parity-check the length against the batch. Tolerates a single-key
    object wrapper; falls back to scanning true/false tokens in order."""
    obj = safe_json_loads(text, context="semfilter_bool")
    arr: list | None = None
    if isinstance(obj, list):
        arr = obj
    elif isinstance(obj, dict):
        arr = next((v for v in obj.values() if isinstance(v, list)), None)
    if arr is None:
        arr = re.findall(r"\btrue\b|\bfalse\b", text, re.IGNORECASE)

    def _truthy(x) -> bool:
        if isinstance(x, bool):
            return x
        if isinstance(x, str):
            return x.strip().lower() in ("true", "1", "yes")
        return bool(x)

    return [_truthy(x) for x in arr]


# Process-wide decision cache, keyed by (model, question, bulletin, page). The
# relevance target is the verbatim question, so a page's true/false verdict is
# stable across a question's retrieve branches — judge it once, reuse it. Keyed
# by question, not UID, so there's no cross-question bleed (different questions
# legitimately decide the same page differently). Temp=0 makes verdicts
# deterministic, so caching changes cost, not the result.
_DECISION_CACHE: dict[tuple[str, str, str, int], bool] = {}


async def _run_summary_filter(
    survivors: list[tuple[str, int]],
    catalog_index: dict[tuple[str, int], PageCatalogRow],
    question: str,
    ctx: ExecutionContext,
    *,
    batch_size: int,
    model: str | None,
) -> list[tuple[str, int]]:
    """Single coarse pass over each survivor's metadata SUMMARY (`_page_meta_block`
    — titles, column/row labels, dates, keywords; no full text, so it's cheap),
    judged against `question`, on `model`. One positional true/false per page
    (parsed by order, not id). Pages already decided for this (model, question)
    are served from `_DECISION_CACHE`, so each page is LLM-judged at most once per
    question. Returns kept pages in input order."""
    if not survivors:
        return []
    mkey = model or ""

    def ckey(pk: tuple[str, int]) -> tuple[str, str, str, int]:
        return (mkey, question, pk[0], pk[1])

    pending = [pk for pk in survivors if ckey(pk) not in _DECISION_CACHE]
    ctx.emit(
        f"summary_filter pages={len(survivors)} judged={len(pending)} "
        f"cached={len(survivors) - len(pending)} model={model}"
    )
    blocks = [_page_meta_block(catalog_index[pk]) for pk in pending]
    batches = _chunk(blocks, batch_size)

    async def _one_batch(batch_pages: list[dict]) -> list[tuple[tuple[str, int], bool]]:
        n = len(batch_pages)
        base_user = _build_question_prompt(question, batch_pages)
        for attempt in range(1, _PARITY_MAX_RETRIES + 1):
            # Parity check: the response MUST have one verdict per input page.
            # A wrong length means a positional misalignment that would silently
            # flip verdicts — so re-issue the batch from scratch (vary temperature
            # + add a corrective so the resample can actually differ).
            user = base_user if attempt == 1 else (
                base_user
                + f"\n\nYour previous reply had the wrong number of values. Return EXACTLY "
                  f"{n} true/false values, one per page, in order — nothing else."
            )
            resp = await ctx.llm_client.acall(
                system=_COARSE_SYSTEM_PROMPT, user=user,
                temperature=0.0 if attempt == 1 else 0.5, model=model,
                ctx=ctx, call_site="semfilter",
            )
            bools = _parse_bool_list(resp.text)
            if len(bools) == n:
                return [((p["bulletin"], p["page"]), bools[i]) for i, p in enumerate(batch_pages)]
            log.warning(
                "semfilter length mismatch (attempt %d/%d): got %d, expected %d — retrying batch from scratch",
                attempt, _PARITY_MAX_RETRIES, len(bools), n,
            )
            ctx.emit(f"semfilter_length_mismatch attempt={attempt} got={len(bools)} expected={n}")
        # Exhausted retries — keep all pages (recall-safe) rather than risk a
        # misaligned drop.
        log.warning(
            "semfilter parity failed after %d attempts (expected %d) — keeping all pages (recall-safe)",
            _PARITY_MAX_RETRIES, n,
        )
        ctx.emit(f"semfilter_parity_failed expected={n} kept_all=true")
        return [((p["bulletin"], p["page"]), True) for p in batch_pages]

    for batch_res in await asyncio.gather(*[_one_batch(b) for b in batches]):
        for pk, keep in batch_res:
            _DECISION_CACHE[ckey(pk)] = keep

    # Recall-safe: any page left undecided (e.g. dropped from a truncated
    # response) defaults to kept.
    return [pk for pk in survivors if _DECISION_CACHE.get(ckey(pk), True)]


async def semantic_filter(
    survivors: list[tuple[str, int]],
    catalog_index: dict[tuple[str, int], PageCatalogRow],
    ctx: ExecutionContext,
) -> tuple[list[tuple[str, int]], dict]:
    """Prune one branch's year-filtered survivors with a single coarse pass over each
    page's metadata summary, judged against the full question on `semfilter_model`.
    Returns `(kept_pages, meta)`; `kept_pages` preserves input order and `meta` carries
    per-stage sizes for the trace."""
    cfg = ctx.config
    model = cfg.semfilter_model or cfg.llm_model
    kept = await _run_summary_filter(
        survivors, catalog_index, ctx.question, ctx,
        batch_size=cfg.semfilter_batch_size, model=model,
    )
    meta = {
        "enabled": True,
        "mode": "summary_coarse",
        "model": model,
        "pre": len(survivors),
        "kept": len(kept),
    }
    return kept, meta
