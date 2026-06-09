"""Page-index query path: the `PageIndexRetriever` over an offline-built artifact.

Retrieval (`retrieve_all`) runs all of a question's branches at once: each branch keeps
its own cheap candidate pass (per-era ToC chapter pick → year filter), then ONE coarse
semantic (summary) filter scans the deduped union of candidates — every unique page at
most once. That filter explodes each page into its CONTENT BLOCKS and judges them as a
single FLAT list (one boolean per block) against the branch RETRIEVAL TARGETS; a page is
kept iff ANY of its blocks fits ANY target. Two choices make the coarse model reliable
here: (1) judging the clean targets rather than the raw, computation-heavy question, and
(2) the flat block list rather than nested page objects, which keeps per-item judgments
stable at larger batches. A question-scoped decision cache keeps a page from being judged
twice across sibling branches (per-branch `run` calls share it via the ctx). Survivors
route back to a branch iff that branch's cheap pass kept the page AND the filter marked
it relevant.

Because the filter judges per block, the retriever's native output is BLOCK-granular
(`BlockRef` — the kept block, its anchor page, and that page's member refs). The
extraction pipeline still consumes `PageRef`s, so the block→page translation lives in
`RetrieveOp` (`retrieve.py`), keeping this module's output at its true granularity."""

from __future__ import annotations

import json
import asyncio
import re
from dataclasses import dataclass, field
from skunk.common import ExecutionContext, PageRef, parse_json_response
from skunk.errors import StepFailed, ParseError
from skunk.plan import RetrieveBranch
from skunk.prompted_call import PromptedCall

from .data_model import (
    CATALOG_SUBDIR,
    CHAPTER_FIELDS,
    CONTENT_BLOCK_FIELDS,
    TREE_FILE,
    ConceptTree,
    ContentBlock,
    EraTree,
    PageCatalogRow,
    page_index_root,
)


@dataclass(frozen=True)
class BlockRef:
    """One retrieved CONTENT BLOCK — the retriever's native output unit now that the semantic
    filter judges per block. `page` is the anchor page the block lives on and `block_index` is
    its position in that page's `content_blocks` (None for a page that carried no blocks but was
    kept wholesale for recall). `member_refs` is the page's anchor + folded continuation run —
    carried here so the block→page translation (in `retrieve.py`) can expand to the physical
    pages extract must read without re-touching the catalog. `block` is the resolved
    `ContentBlock` for downstream consumers; it's excluded from identity (so `BlockRef`s stay
    hashable and de-dupe on `page`/`block_index`/`member_refs`)."""

    page: PageRef
    block_index: int | None
    member_refs: tuple[PageRef, ...]
    block: ContentBlock | None = field(default=None, compare=False)


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


def _overlaps(
    interval: tuple[str, str], period_intervals: list[tuple[str, str]]
) -> bool:
    """True iff a page's `YYYY-MM` `(low, high)` span overlaps any period interval.
    Compared at month granularity (`[:7]`, tolerating a stray day in stored data)."""
    lo, hi = interval[0][:7], interval[1][:7]
    return any(not (hi < p_lo[:7] or lo > p_hi[:7]) for p_lo, p_hi in period_intervals)


class PageIndexRetriever:
    """Orchestrates the retrieve stages: per-era ToC chapter pick → year filter → coarse
    semantic filter. The artifact (concept tree + per-page catalog) is loaded and
    projected once at construction, then read by the (parallel) branch fan-out."""

    _CHAPTER_PICK_PROMPT = (
        """\
For each Treasury Bulletin chapter, decide whether the answer to the question could PLAUSIBLY be
in it. Judge every chapter independently — this is NOT a single best pick; any number may be true.

## Input

A single JSON object:
  {"question": "<question>", "concept": "<concept tag>", "period": "<period>",
   "chapters": [{"chapter": "<name>", "n_pages": <int>, "description": "<scope>",
                 "examples": ["<sub-area>", ...]}, ...]}

"""
        + CHAPTER_FIELDS
        + """

Match the question against each chapter's name, `description`, and `examples`.

## Output

A single bare JSON array of booleans — no prose, no markdown fences — one entry per chapter, in the
SAME ORDER as the input `chapters`:
  [true, false, ...]

Mark a chapter `true` when the answer could plausibly be in it; `false` only when it is clearly
unrelated. ERR TOWARD true: when in doubt, return `true`. When two chapters are similar or cover
overlapping topics, return `true` for BOTH — do not guess which one holds the table. Keeping an
extra chapter is cheap; dropping the one that holds the answer loses it for good.
"""
    )

    _SEMFILTER_SYSTEM_PROMPT = (
        """\
You are given a list of RETRIEVAL TARGETS (data concepts a research question needs) and a flat batch
of CONTENT BLOCKS — each a single table/chart/prose region from a Treasury Bulletin page, tagged
with its bulletin + page. For EACH block, decide whether it contains data relevant to ANY ONE of the
targets. You see only a compact SUMMARY per block — its title, column/row labels, dates — not the
actual numbers. Judge each block INDEPENDENTLY.

A block is relevant to a target when its summary suggests it reports the kind of data that target
needs for a relevant time period. Match on the DATA SERIES, not on whether the block alone could
answer some downstream computation: KEEP blocks that hold the raw data series even when they cover
only PART of the needed time span (the answer is assembled across several issues), and KEEP prose
blocks whose summary names the relevant instrument/series.

## Input

A numbered list of `targets` (each a concept, optionally with the `period` its data covers), then a
JSON array of candidate blocks. Each block (any field may be absent when empty):
  {"bulletin": "<YYYY-MM>", "page": <int>, "date_interval": [...], "kind": ..., "title": ...,
   "column_headers": [...], "row_headers": [...], "summary": ...}

where (block fields):
"""
        + CONTENT_BLOCK_FIELDS
        + """

`bulletin`/`page` identify the source page (several consecutive blocks may share one page);
`date_interval` is the page's `[low, high]` YYYY-MM data span.

## Output

A single bare JSON array of booleans — no prose, no markdown fences. One entry per input block, in
the block order given:
  [true, false, ...]

Mark a block `true` when its summary fits AT LEAST ONE target; `false` only when clearly unrelated
to every target.
"""
    )

    def __init__(self) -> None:
        root = page_index_root()
        try:
            self._tree = ConceptTree.model_validate_json(
                (root / TREE_FILE).read_bytes()
            )
        except FileNotFoundError as e:
            raise StepFailed(
                "retrieve",
                f"page index not built ({e}); run the page-index build pipeline first.",
            ) from e
        rows = [
            PageCatalogRow.from_json(line)
            for f in sorted((root / CATALOG_SUBDIR).glob("*.jsonl"))
            for line in f.read_text().splitlines()
            if line
        ]
        self._catalog = {r.ref: r for r in rows}
        self._catalog_size = sum(
            c.n_pages for era in self._tree.eras for c in era.chapters.values()
        )

    # -- reply parsing ---------------------------------------------------------

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

    def _eras_for_branch(self, branch: RetrieveBranch) -> list[EraTree]:
        """The eras whose ToC this branch needs to scan. Eras are keyed by bulletin PUBLICATION
        month; `branch.period` is the DATA date. An issue cannot report data dated AFTER it was
        published, so an era whose publication span ends before the period's earliest month can't
        hold the branch's data — skip it. Publication LAG the other way is unbounded (old data is
        restated in much later issues / revisions), so every era at or after the period start is
        kept. A branch with no parseable period (nor `as_of`) keeps all eras (can't narrow safely).

        `as_of` pins a specific issue (publication month), so when it's set we scan only the era(s)
        whose publication span contains it — eras are keyed by publication month, so that's exact."""
        as_of_iv = _to_intervals(branch.as_of)
        if as_of_iv:
            lo = min(x[0][:7] for x in as_of_iv)
            hi = max(x[1][:7] for x in as_of_iv)
            kept = [
                e
                for e in self._tree.eras
                if not (e.span[1][:7] < lo or e.span[0][:7] > hi)
            ]
            return kept or self._tree.eras
        intervals = _to_intervals(branch.period)
        if not intervals:
            return self._tree.eras
        earliest = min(lo[:7] for lo, _ in intervals)
        kept = [e for e in self._tree.eras if e.span[1][:7] >= earliest]
        return kept or self._tree.eras

    async def _pick_chapters(
        self,
        *,
        branch: RetrieveBranch,
        ctx: ExecutionContext,
    ) -> list[PageRef]:
        # One ToC pick per era (each era has its own chapter taxonomy), run in parallel. A page
        # belongs to exactly one era, and within an era pages partition across chapters, so the
        # union across eras (and picks) is duplicate-free. Eras predating the period are skipped
        # (`_eras_for_branch`) — an issue can't publish data before it exists — but every era at or
        # after the period start is kept, since revisions restate old data in arbitrarily later
        # issues. Fine page-level temporal narrowing is still the year filter's job.
        eras = self._eras_for_branch(branch)
        ctx.emit(
            f"pick_chapters_eras key={branch.key!r} period={branch.period!r} as_of={branch.as_of!r} "
            f"eras={len(eras)}/{len(self._tree.eras)}"
        )
        per_era = await asyncio.gather(
            *[self._pick_era_chapters(era, branch=branch, ctx=ctx) for era in eras]
        )
        return [ref for era_pages in per_era for ref in era_pages]

    async def _pick_era_chapters(
        self,
        era: EraTree,
        *,
        branch: RetrieveBranch,
        ctx: ExecutionContext,
    ) -> list[PageRef]:
        chapters = era.chapters
        if not chapters:
            return []
        # Listing is largest-first; the boolean reply is POSITIONAL, in this same order.
        listing = sorted(
            (
                {
                    "chapter": name,
                    "n_pages": c.n_pages,
                    "description": c.description,
                    "examples": list(c.examples),
                }
                for name, c in chapters.items()
            ),
            key=lambda x: -x["n_pages"],
        )
        names = [x["chapter"] for x in listing]
        pages = {name: list(c.refs) for name, c in chapters.items()}

        # Built per call: the parser closes over this era's positional `names`, so the shared
        # retriever stays stateless under the parallel branch + era fan-out. Reuses the strict
        # boolean-array parse (as the semantic filter does); a length mismatch reprompts.
        def _parse_keep(text: str, _ctx: ExecutionContext) -> list[str]:
            bools = self._parse_bool_list(text, _ctx)
            if len(bools) != len(names):
                raise ParseError(
                    text,
                    f"expected {len(names)} booleans (one per chapter, in order), got {len(bools)}",
                )
            return [nm for nm, keep in zip(names, bools) if keep]

        pick_call: PromptedCall[list[str]] = PromptedCall(
            name="toc_pick",
            system_prompt=self._CHAPTER_PICK_PROMPT,
            parse=_parse_keep,
            # Reason before answering: the pick is a per-chapter relevance judgment over a whole
            # era's taxonomy, and non-obvious matches (e.g. "unemployment tax receipts" ↔ "Internal
            # Revenue Receipts by State") were missed at effort=off. Picks are ~2 min over the whole
            # dev set (not the latency bottleneck — the semantic filter scan volume is), so keep
            # medium for the recall. (Override: SKUNK_EFFORT_OVERRIDES.)
            default_effort="medium",
            output_instruction="Output ONLY a JSON array of true/false — one per chapter, in the given order, no prose.",
        )
        user = json.dumps(
            {
                "question": ctx.question,
                "concept": branch.key,
                "period": branch.period,
                "chapters": listing,
            },
            ensure_ascii=False,
            indent=1,
        )
        picked = await pick_call.call(ctx, user, temperature=0.0)
        out = [ref for ch in picked for ref in pages[ch]]
        ctx.emit(f"pick_chapters era={era.span} picked={picked!r} pages={len(out)}")
        return out

    def _year_filter(
        self,
        candidates: list[PageRef],
        branch: RetrieveBranch,
        ctx: ExecutionContext,
    ) -> list[PageRef]:
        # Both `as_of` and `period` apply when both are set, as_of FIRST:
        #   1. as_of pins the named issue — keep only pages from that bulletin, matched on
        #      PUBLICATION month (`ref.month`). A page lives in issue X even when its data span is
        #      earlier (a chart plotting prior years, a reprinted vintage table).
        #   2. period then restricts those to pages whose DATA span overlaps the requested window.
        # With only one set, only that filter runs; with neither, all candidates pass.
        kept = list(candidates)
        as_of_intervals = _to_intervals(branch.as_of)
        if as_of_intervals:
            kept = [
                ref
                for ref in kept
                if ref.month and _overlaps((ref.month, ref.month), as_of_intervals)
            ]
        period_intervals = _to_intervals(branch.period)
        if period_intervals:
            # Keep pages whose DATA span overlaps the period; drop undatable pages (no
            # `date_interval`) — front matter / dividers / OCR-broken pages — only when a period
            # is actually in force.
            kept = [
                ref
                for ref in kept
                if (row := self._catalog.get(ref)) is not None
                and row.date_interval is not None
                and _overlaps(row.date_interval, period_intervals)
            ]
        ctx.emit(
            f"year_filter as_of={branch.as_of!r} period={branch.period!r} "
            f"kept={len(kept)}/{len(candidates)}"
        )
        return kept

    async def _candidates_for(
        self,
        branch: RetrieveBranch,
        ctx: ExecutionContext,
    ) -> list[PageRef]:
        """A branch's cheap candidate set: per-era ToC chapter pick → year filter. The
        expensive LLM semantic filter is NOT applied here — it runs once, globally, over
        the deduped union of every branch's candidates (see `retrieve_all`)."""
        chapter_pages = await self._pick_chapters(branch=branch, ctx=ctx)
        return self._year_filter(chapter_pages, branch, ctx)

    @staticmethod
    def _decision_cache(ctx: ExecutionContext) -> dict[PageRef, list[bool]]:
        """The question-scoped page→per-block-verdict cache, lazily created on the ctx. Each
        value is one boolean per the page's `content_blocks`, in order (an EMPTY list marks a page
        that carried no blocks but is kept wholesale for recall). One ctx per question
        (single-threaded — see `ExecutionContext`), so this judges each unique page at most once
        across a question's sibling branches: the union inside one `retrieve_all` dedups within a
        call, and this cache carries decisions across the separate per-branch `run` calls too.
        Freed with the ctx — no cross-question leak."""
        cache = getattr(ctx, "_semfilter_decisions", None)
        if cache is None:
            cache = {}
            ctx._semfilter_decisions = cache  # type: ignore[attr-defined]
        return cache

    @staticmethod
    def _page_kept(verdicts: list[bool]) -> bool:
        """A page is kept iff ANY of its blocks is relevant, OR it carried no blocks at all
        (empty verdict list → kept wholesale for recall)."""
        return not verdicts or any(verdicts)

    async def _semantic_filter(
        self,
        pages: list[PageRef],
        branches: list[RetrieveBranch],
        ctx: ExecutionContext,
    ) -> dict[PageRef, list[bool]]:
        """Judge each unique candidate page against the branch RETRIEVAL TARGETS (concept +
        period), returning one boolean PER CONTENT BLOCK (aligned to the page's `content_blocks`;
        an empty list for a page that carried no blocks — kept wholesale for recall). Pages are
        exploded into their blocks and judged as one FLAT list (each block its own row), not as
        nested page objects: the flat shape keeps the coarse model's per-item judgments stable at
        larger batches (nested page→blocks objects degrade fast as the batch grows). A block is
        true if it fits ANY target. Judging clean targets rather than the raw (computation-heavy)
        question is what lets the coarse model keep on-topic blocks.

        Pages already decided this question (the decision cache) are skipped. Batches are packed
        page-coherently up to `cfg.semfilter_batch_size` BLOCKS per call (a page's blocks never
        split across calls; a page with more blocks than the cap stands alone). A batch whose reply
        is unparsable degrades to keep-all rather than failing the sweep — recall is preserved
        (extract is the precision gate) and sibling branches stay alive."""
        cfg = ctx.config
        model = cfg.model_overrides.get("semfilter", cfg.llm_model)
        cache = self._decision_cache(ctx)
        # The numbered target list shared by every batch — one entry per branch, in order.
        targets = [
            {"target": i + 1, "concept": b.key, "period": b.period}
            for i, b in enumerate(branches)
        ]
        targets_json = json.dumps(targets, ensure_ascii=False, indent=1)

        # Explode each undecided page into a flat list of its content blocks (no full text, no
        # numeric grid). A page with no blocks can't be judged — keep it (recall over precision).
        todo = [pk for pk in pages if pk not in cache]
        groups: list[tuple[PageRef, list[dict]]] = []
        for pk in todo:
            row = self._catalog[pk]
            rows = [
                {
                    "bulletin": pk.month,
                    "page": pk.page,
                    "date_interval": row.date_interval,
                    "kind": b.kind,
                    "title": b.title,
                    "column_headers": b.column_headers,
                    "row_headers": b.row_headers,
                    "summary": b.summary,
                }
                for b in row.content_blocks
            ]
            if rows:
                groups.append((pk, rows))
            else:
                cache[pk] = []  # no blocks → kept wholesale (empty verdict list)

        # Page-coherent adaptive packing: greedily fill a call up to `cap` blocks, never splitting
        # a page across calls (dense pages take fewer per call, sparse pages more).
        cap = max(1, cfg.semfilter_batch_size)
        batches: list[list[tuple[PageRef, list[dict]]]] = []
        cur: list[tuple[PageRef, list[dict]]] = []
        n_cur = 0
        for pk, rows in groups:
            if cur and n_cur + len(rows) > cap:
                batches.append(cur)
                cur, n_cur = [], 0
            cur.append((pk, rows))
            n_cur += len(rows)
        if cur:
            batches.append(cur)

        async def _one_batch(
            batch: list[tuple[PageRef, list[dict]]],
        ) -> dict[PageRef, list[bool]]:
            flat = [(pk, r) for pk, rows in batch for r in rows]
            block_rows = [r for _, r in flat]
            n = len(block_rows)
            user = (
                f"targets (JSON, {len(targets)} entries):\n{targets_json}\n\n"
                f"blocks (JSON, {n} entries):\n"
                f"{json.dumps(block_rows, ensure_ascii=False, indent=1)}"
            )

            # Validate the COUNT inside the parse fn (closing over this batch's n) so a wrong
            # length raises ParseError and rides PromptedCall's built-in reprompt. Only if the
            # reprompt ALSO fails do we degrade to keep-all (recall over precision).
            def _parse_shaped(text: str, _ctx: ExecutionContext) -> list[bool]:
                verdicts = self._parse_bool_list(text, _ctx)
                if len(verdicts) != n:
                    raise ParseError(
                        text,
                        f"expected exactly {n} booleans (one per block, in order); "
                        f"got {len(verdicts)}",
                    )
                return verdicts

            sem_call: PromptedCall[list[bool]] = PromptedCall(
                name="semfilter",
                system_prompt=self._SEMFILTER_SYSTEM_PROMPT,
                parse=_parse_shaped,
                output_instruction=(
                    f"Output ONLY a JSON array of EXACTLY {n} booleans — one per block, in the "
                    "order given. No prose, no markdown fences."
                ),
            )
            try:
                verdicts = await sem_call.call(ctx, user, temperature=0.0)
            except ParseError as e:
                ctx.emit(f"semfilter_batch_degraded n_blocks={n} error={str(e)!r}")
                verdicts = [True] * n
            # Regroup the flat verdicts back per page, in content_blocks order (a page's blocks
            # are contiguous in `flat`, so append preserves alignment).
            out: dict[PageRef, list[bool]] = {pk: [] for pk, _ in batch}
            for (pk, _), v in zip(flat, verdicts):
                out[pk].append(v)
            return out

        per_batch = await asyncio.gather(*[_one_batch(b) for b in batches])
        for d in per_batch:
            cache.update(d)
        n_kept = sum(1 for pk in pages if self._page_kept(cache[pk]))
        n_blocks_kept = sum(sum(cache[pk]) for pk in pages)
        n_blocks = sum(len(rows) for _, rows in groups)
        ctx.emit(
            f"semantic_filter model={model} kept={n_kept}/{len(pages)} "
            f"blocks_kept={n_blocks_kept} judged={len(todo)} "
            f"cached={len(pages) - len(todo)} blocks={n_blocks}"
        )
        return {pk: cache[pk] for pk in pages}

    def _block_refs_for(self, ref: PageRef, verdicts: list[bool]) -> list[BlockRef]:
        """Turn one kept candidate page + its per-block verdicts into `BlockRef`s. Each kept
        block becomes one ref; a page with NO blocks (empty verdicts) but otherwise kept yields a
        single whole-page ref (`block_index=None`). Every ref carries the page's member refs
        (anchor + folded continuation run) so the block→page translation can expand to the
        physical pages extract reads without re-touching the catalog."""
        row = self._catalog.get(ref)
        members = tuple(row.member_refs()) if row is not None else (ref,)
        if not verdicts:
            return [
                BlockRef(page=ref, block_index=None, member_refs=members, block=None)
            ]
        blocks = row.content_blocks if row is not None else []
        return [
            BlockRef(
                page=ref,
                block_index=bi,
                member_refs=members,
                block=blocks[bi] if bi < len(blocks) else None,
            )
            for bi, keep in enumerate(verdicts)
            if keep
        ]

    async def retrieve_all(
        self,
        ctx: ExecutionContext,
        branches: list[RetrieveBranch],
    ) -> list[list[BlockRef]]:
        """Retrieve for every branch in one pass, scanning each unique candidate page with
        the LLM semantic filter AT MOST ONCE. Each branch keeps its own cheap candidate set
        (ToC pick → year filter); the deduped union is then filtered once against the branch
        RETRIEVAL TARGETS, judged per CONTENT BLOCK. A block routes to branch `i` iff `i`'s cheap
        filters kept its page AND the filter marked that block relevant. Output is BLOCK-granular
        (`BlockRef`s), aligned to `branches`; the block→page translation for the extraction
        pipeline lives in `RetrieveOp` (`retrieve.py`).

        NOTE: the filter judges against ALL branch targets together, so callers wanting the
        full "fits any target" behaviour must pass every branch in one call (production routes
        through `RetrieveOp.run_all`); a single-branch call only ever sees that one target."""
        # Phase 1 — per-branch cheap candidates (parallel).
        cand = await asyncio.gather(*[self._candidates_for(b, ctx) for b in branches])

        # Dedupe into page → requesting-branch-indices, ordered deterministically so batches
        # are stable across runs.
        wanted: dict[PageRef, set[int]] = {}
        for i, refs in enumerate(cand):
            for ref in refs:
                wanted.setdefault(ref, set()).add(i)
        unique = sorted(wanted, key=lambda r: (r.month or "", r.page or 0))

        # Phase 2 — one semantic-filter sweep over the unique pages (skippable for the
        # ToC+date recall-ceiling eval mode, which keeps every cheap candidate). Verdicts are
        # per content block (empty list → page kept wholesale).
        if ctx.config.retrieve_skip_semfilter:
            verdict = {
                ref: [True] * len(self._catalog[ref].content_blocks)
                if ref in self._catalog
                else []
                for ref in unique
            }
            ctx.emit(f"semantic_filter SKIPPED kept={len(unique)}/{len(unique)}")
        else:
            verdict = await self._semantic_filter(unique, branches, ctx)

        # Route per branch, preserving each branch's candidate order; emit kept blocks.
        out: list[list[BlockRef]] = []
        for i, refs in enumerate(cand):
            kept_pages = [ref for ref in refs if self._page_kept(verdict[ref])]
            block_refs = [
                br
                for ref in kept_pages
                for br in self._block_refs_for(ref, verdict[ref])
            ]
            b = branches[i]
            ctx.emit(
                f"page_index_retrieve key={b.key!r} period={b.period!r} as_of={b.as_of!r} "
                f"catalog_size={self._catalog_size} anchor_count={len(kept_pages)} block_count={len(block_refs)} "
            )
            # Empty is fine to return — the translation/extract raises loudly on no refs, which
            # the orchestrator records as a failed branch and replans.
            out.append(block_refs)
        return out
