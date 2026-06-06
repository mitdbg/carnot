"""Page-index query path: the `PageIndexRetriever` over an offline-built artifact.

Retrieval is three passes: ToC chapter pick → year filter → coarse semantic
(summary) filter → candidate set."""

from __future__ import annotations

import json
import asyncio
import os
import re
from pathlib import Path
from skunk.common import ExecutionContext, PageRef, chunk, parse_json_response
from skunk.errors import StepFailed, ParseError
from skunk.plan import RetrieveBranch
from skunk.prompted_call import PromptedCall
from typing import Any

from .data_model import CATALOG_SUBDIR, TREE_FILE, ConceptTree, PageCatalogRow


def page_index_root() -> Path:
    """Query-side artifact root — `SKUNK_PAGE_INDEX_DIR`, the single source of truth."""
    env = os.environ.get("SKUNK_PAGE_INDEX_DIR")
    if not env:
        raise StepFailed("retrieve", "SKUNK_PAGE_INDEX_DIR is not set; point it at "
                         "a built page-index artifact.")
    return Path(env)


# -- period matching (for the year filter) ----------------------------------
# The planner emits periods as canonical `YYYY-MM` — a single month, an inclusive
# `lo..hi` range, or a comma-separated enumeration. Fiscal-year / quarter expansion
# is the planner's job (see the corpus prompt), so there is no grammar to expand.

_MONTH_RE = re.compile(r"\d{4}-\d{2}")


def _to_intervals(period: str | None) -> list[tuple[str, str]] | None:
    """Parse a `YYYY-MM` period into inclusive `(low, high)` month intervals.
    None when empty or malformed (the year filter then no-ops)."""
    if not period:
        return None
    try:
        out: list[tuple[str, str]] = []
        for part in (p.strip() for p in period.split(",")):
            if not part:
                continue
            lo, _, hi = part.partition("..")
            lo, hi = lo.strip(), (hi.strip() or lo.strip())
            if not (_MONTH_RE.fullmatch(lo) and _MONTH_RE.fullmatch(hi)):
                raise ValueError(f"not YYYY-MM: {part!r}")
            if lo > hi:
                raise ValueError(f"range start > end: {part!r}")
            out.append((lo, hi))
        return out or None
    except ValueError:
        return None


def _overlaps(interval: tuple[str, str], period_intervals: list[tuple[str, str]]) -> bool:
    """True iff a page's `YYYY-MM` `(low, high)` span overlaps any period interval.
    Compared at month granularity (`[:7]`, tolerating a stray day in stored data)."""
    lo, hi = interval[0][:7], interval[1][:7]
    return any(not (hi < p_lo[:7] or lo > p_hi[:7]) for p_lo, p_hi in period_intervals)


class PageIndexRetriever:
    """Orchestrates the retrieve stages: ToC chapter pick → year filter → coarse
    semantic filter. The artifact (concept tree + per-page catalog) is loaded and
    projected once at construction, then read by the (parallel) branch fan-out."""

    _CHAPTER_PICK_PROMPT = """\
You pick the Treasury Bulletin chapter most likely to contain the answer.

## Input

A single JSON object:
  {"question": "<question>", "concept": "<concept tag>", "period": "<period>",
   "chapters": [{"chapter": "<name>", "n_pages": <int>,
                 "description": "<scope>", "examples": ["<sub-area>", ...]}, ...]}

Match the question against each chapter's `description` and `examples`.

## Output

A single bare JSON object — no prose, no markdown fences:
  {"picked": ["<exact chapter name>", ...]}

Return the best chapter, using its EXACT `chapter` value. Add a second only when the
question genuinely straddles two and you can't tell which holds the answer. You may return
at most two chapters.
"""

    _SEMFILTER_SYSTEM_PROMPT = """\
For each candidate Treasury Bulletin page, decide whether it contains information to help answer the question.
You see only a compact SUMMARY per page — table titles, column/row labels, dates, keywords — not the actual numbers.
Keep a page (true) when its summary suggests it reports the kind of data the question needs for a relevant
time period; mark it false when the page is clearly unrelated.

## Input

A question line, then a JSON array of candidate pages. Each page (any field may be absent when empty):
  {"bulletin": "<YYYY-MM>", "page": <int>, "keywords": ["<term>", ...],
   "date_interval": ["<YYYY-MM>", "<YYYY-MM>"],
   "content_blocks": [{"title": "<table/chart title>", "column_headers": ["<col>", ...],
                       "row_headers": ["<row>", ...], "summary": "<what the block is about>"}, ...]}

## Output

A single bare JSON array of booleans — no prose, no markdown fences — one entry per input page, in order:
  [true, false, ...]
"""

    def __init__(self) -> None:
        root = page_index_root()
        try:
            tree = ConceptTree.model_validate_json((root / TREE_FILE).read_bytes())
        except FileNotFoundError as e:
            raise StepFailed("retrieve", f"page index not built ({e}); run the "
                             "page-index build pipeline first.") from e
        rows = [PageCatalogRow.from_json(line)
                for f in sorted((root / CATALOG_SUBDIR).glob("*.jsonl"))
                for line in f.read_text().splitlines() if line]
        self._chapters = tree.chapters
        self._catalog = {r.ref: r for r in rows}
        # Chapter projection for the ToC prompt, largest first (`pages` excluded).
        self._listing = sorted(
            ({"chapter": name,
              **c.model_dump(include={"n_pages", "description", "examples"})}
             for name, c in self._chapters.items()),
            key=lambda x: -x["n_pages"],
        )
        self._valid = {name.lower(): name for name in self._chapters}   # lowercased -> canonical
        self._catalog_size = sum(c.n_pages for c in self._chapters.values())

        # `output_instruction` is re-appended after the data on every call/retry
        # (the double-attention reminder), so the user turn carries only the inputs.
        self._chapter_pick: PromptedCall[list[str]] = PromptedCall(
            name="toc_pick",
            system_prompt=self._CHAPTER_PICK_PROMPT,
            parse=self._parse_chapter_picks,
            output_instruction='Output ONLY the JSON object {"picked": [...]} — exact chapter names, at most two.',
        )
        self._semfilter: PromptedCall[list[bool]] = PromptedCall(
            name="semfilter",
            system_prompt=self._SEMFILTER_SYSTEM_PROMPT,
            parse=self._parse_bool_list,
            output_instruction="Output ONLY a JSON array of true/false — one entry per page, in the order given, no prose.",
        )

    # -- reply parsing ---------------------------------------------------------

    def _parse_chapter_picks(self, text: str, _ctx: ExecutionContext) -> list[str]:
        """Parse `picked` and canonicalize against the loaded chapters
        (case-insensitive; unknowns dropped, deduped, order preserved). Raises
        `ParseError` (→ one reprompt) on a non-array, more than two picks, or no
        recognized chapter."""
        obj = parse_json_response(text)
        raw = obj.get("picked") if isinstance(obj, dict) else None
        if not isinstance(raw, list):
            raise ParseError(text, 'expected a "picked" array of chapter names')
        names = [s for x in raw if (s := str(x).strip())]
        if len(names) > 2:
            raise ParseError(text, f"expected at most 2 chapters, got {len(names)}")
        picked = list(dict.fromkeys(v for s in names if (v := self._valid.get(s.lower()))))
        if not picked:
            raise ParseError(text, "no recognized chapter names in 'picked'")
        return picked

    @staticmethod
    def _parse_bool_list(text: str, _ctx: ExecutionContext) -> list[bool]:
        """Strict: a JSON array of JSON booleans — no coercion, no token-scan
        fallback. Raises `ParseError` (→ one reprompt) on a non-array or any
        non-boolean element. The per-batch count is checked by the caller."""
        obj = parse_json_response(text)
        if not isinstance(obj, list) or not all(isinstance(x, bool) for x in obj):
            raise ParseError(text, "expected a JSON array of booleans (true/false)")
        return obj

    # -- stages ----------------------------------------------------------------

    async def _pick_chapters(
        self,
        *,
        question: str,
        concept: str,
        period: str | None,
        ctx: ExecutionContext,
    ) -> list[PageRef]:
        user = json.dumps(
            {"question": question, "concept": concept, "period": period,
             "chapters": self._listing},
            ensure_ascii=False, indent=1,
        )
        picked = await self._chapter_pick.call(ctx, user, temperature=0.0)
        # Pages partition across chapters, so the union across picks is unique.
        pages = [ref for ch in picked for ref in self._chapters[ch].pages]
        ctx.emit(f"pick_chapters picked={picked!r} pages={len(pages)}")
        return pages

    def _year_filter(
        self,
        candidates: list[PageRef],
        branch: RetrieveBranch,
        ctx: ExecutionContext,
    ) -> list[PageRef]:
        if branch.as_of:
            as_of_intervals = _to_intervals(branch.as_of)
            if as_of_intervals:
                kept = [ref for ref in candidates
                        if ref.month and _overlaps((ref.month, ref.month), as_of_intervals)]
                ctx.emit(f"year_filter as_of={branch.as_of!r} kept={len(kept)}/{len(candidates)}")
                return kept

        period_intervals = _to_intervals(branch.period)
        if not period_intervals:
            return list(candidates)
        kept = []
        for ref in candidates:
            row = self._catalog.get(ref)
            # Keep when the page has no span (can't filter it) or its span overlaps.
            if row is None or row.date_interval is None or _overlaps(row.date_interval, period_intervals):
                kept.append(ref)
        ctx.emit(f"year_filter period={branch.period!r} kept={len(kept)}/{len(candidates)}")
        return kept

    async def _semantic_filter(
        self,
        survivors: list[PageRef],
        ctx: ExecutionContext,
    ) -> list[PageRef]:
        cfg = ctx.config
        model = cfg.model_overrides.get("semfilter", cfg.llm_model)
        # What the filter sees: page-level signals plus content_blocks one level
        # deep — no full text, no numeric grid. `exclude_none` drops null fields.
        filter_view: dict[str, Any] = {
            "bulletin": True, "page": True, "keywords": True, "date_interval": True,
            "content_blocks": {"__all__": {"title", "column_headers", "row_headers", "summary"}},
        }
        blocks = [(pk, self._catalog[pk].model_dump(include=filter_view, exclude_none=True))
                  for pk in survivors]

        async def _one_batch(batch: list[tuple[PageRef, dict]]) -> list[PageRef]:
            pages = [d for _, d in batch]
            n = len(pages)
            user = (
                f"question: {ctx.question}\n\n"
                f"pages (JSON, {n} entries):\n"
                f"{json.dumps(pages, ensure_ascii=False, indent=1)}"
            )
            bools = await self._semfilter.call(ctx, user, temperature=0.0)
            if len(bools) != n:
                raise StepFailed("retrieve",
                                 f"semantic filter returned {len(bools)} verdicts for {n} pages")
            return [pk for (pk, _), keep in zip(batch, bools) if keep]

        batches_kept = await asyncio.gather(
            *[_one_batch(b) for b in chunk(blocks, cfg.semfilter_batch_size)])
        kept = [pk for batch in batches_kept for pk in batch]
        ctx.emit(f"semantic_filter model={model} kept={len(kept)}/{len(survivors)}")
        return kept

    async def retrieve(self, ctx: ExecutionContext, *, branch: RetrieveBranch) -> list[PageRef]:
        chapter_pages = await self._pick_chapters(
            question=ctx.question, concept=branch.key, period=branch.period, ctx=ctx,
        )
        filtered = self._year_filter(chapter_pages, branch, ctx)
        filtered = await self._semantic_filter(filtered, ctx)

        ctx.emit(
            f"page_index_retrieve key={branch.key!r} period={branch.period!r} as_of={branch.as_of!r} "
            f"catalog_size={self._catalog_size} candidate_count={len(filtered)} "
        )
        # Empty is fine to return — extract raises loudly on no refs, which the
        # orchestrator records as a failed branch and replans.
        return filtered
