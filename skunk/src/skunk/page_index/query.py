"""Page-index query path: retrieval (`PageIndexRetriever`) and block selection.

`PageIndexRetriever.retrieve_all` runs all branches at once: per-era ToC chapter pick →
year filter → one coarse semantic filter over the deduped union. The filter judges per
CONTENT BLOCK (flat list, one boolean each) against the branch RETRIEVAL TARGETS; a page
is kept iff any block fits any target. Output is BLOCK-granular (`BlockRef`).

`PageIndexRetriever.select_blocks` is the precision stage that follows: a tournament
reduction over the semantic filter's survivors, narrowing each branch to at most `_KEEP`
blocks."""

from __future__ import annotations

import json
import asyncio
import re
from pathlib import Path
from skunk.common import BlockRef, ExecutionContext, PageRef, chunk, parse_json_response, traced_step
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
from .store import get_page_store


_MONTH_RE = re.compile(r"\d{4}-\d{2}")


def _to_intervals(period: str | None) -> list[tuple[str, str]] | None:
    """Parse a `YYYY-MM` period string into inclusive `(lo, hi)` month intervals.
    Returns None when empty or malformed (callers then no-op the filter)."""
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
    """True iff a `(lo, hi)` YYYY-MM span overlaps any interval in `period_intervals`."""
    lo, hi = interval[0][:7], interval[1][:7]
    return any(not (hi < p_lo[:7] or lo > p_hi[:7]) for p_lo, p_hi in period_intervals)


def _batch_by_size(
    groups: list[tuple[PageRef, list[dict]]], cap: int
) -> list[list[tuple[PageRef, list[dict]]]]:
    """Pack `groups` into batches whose total row count does not exceed `cap`."""
    batches: list[list[tuple[PageRef, list[dict]]]] = []
    cur: list[tuple[PageRef, list[dict]]] = []
    n = 0
    for pk, rows in groups:
        if cur and n + len(rows) > cap:
            batches.append(cur)
            cur, n = [], 0
        cur.append((pk, rows))
        n += len(rows)
    if cur:
        batches.append(cur)
    return batches


class PageIndexRetriever:
    """Orchestrates retrieval (ToC chapter pick → year filter → semantic filter) and block
    selection (tournament precision stage). The artifact is loaded once at construction."""

    # -- prompts ---------------------------------------------------------------

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

    _SEMFILTER_PROMPT = (
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

    _BLOCK_SELECT_PROMPT = """\
You select which already-filtered Treasury Bulletin content blocks actually carry the data the given retrieval
target needs, in the context of the question. Candidate blocks are one per line. Each line starts with its
`block_id` `YYYY_MM_page#block` — the leading `YYYY_MM` is the ISSUE (the bulletin's publication month) the block
appears in — then `dates=`, the time span the block's DATA covers, then a
compact summary (title, column/row labels) — NOT the numbers.

Return ONLY the blocks that most directly report the target's data, best-first, and never more
than the cap stated in the request — fewer when fewer are appropriate. When choosing:
  - If the question pins a specific issue ("as reported in the <Month Year> Bulletin", "as of <date>"), select
    blocks whose ISSUE (the `YYYY_MM`) is that bulletin; the explicit wording overrides the generic data period.
  - Otherwise prefer the block whose title / headers / `dates=` span match the target most precisely, and prefer
    a data table over a chart of the same series.
  - Match the asked concept at its EXACT scope. A row/column label that wraps the concept in extra words — "<concept> and
    related activities", "<concept>, including …", etc. — names a BROADER aggregate and will have different values from
    the bare concept. Prefer the block that reports exactly the asked scope.
  - If multiple blocks match the target, pick the best one and do not emit duplicates.
  - When the same figure is restated across many issues, prefer the most recent issue unless the question
    explicitly asks for a version.

## Output

A single JSON object, no prose, no markdown fences, listing the selected ids (best first, no
more than the requested cap) under "block_ids":
  {"block_ids": ["2001_06_41#0"]}
Use each `block_id` exactly as it appears."""

    _GROUP_SIZE = 64  # blocks per tournament call
    _KEEP = 4         # blocks kept per group; hard output cap per branch

    # -- construction ----------------------------------------------------------

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

    # -- shared parse util -----------------------------------------------------

    @staticmethod
    def _parse_bool_list(text: str, _ctx: ExecutionContext, n: int | None = None) -> list[bool]:
        """Parse a JSON boolean array; raises `ParseError` on non-array, non-boolean elements,
        or (when `n` is given) wrong length."""
        obj = parse_json_response(text)
        if not isinstance(obj, list) or not all(isinstance(x, bool) for x in obj):
            raise ParseError(text, "expected a JSON array of booleans (true/false)")
        if n is not None and len(obj) != n:
            raise ParseError(text, f"expected {n} booleans (one per item, in order), got {len(obj)}")
        return obj

    # -- retrieval stages ------------------------------------------------------

    def _eras_for_branch(self, branch: RetrieveBranch) -> list[EraTree]:
        """Eras whose ToC to scan. `as_of` pins a publication month (scan only containing eras);
        `period` prunes eras whose span ends before the period starts; neither keeps all eras."""
        as_of_iv = _to_intervals(branch.as_of)
        if as_of_iv:
            lo = min(x[0][:7] for x in as_of_iv)
            hi = max(x[1][:7] for x in as_of_iv)
            kept = [e for e in self._tree.eras if not (e.span[1][:7] < lo or e.span[0][:7] > hi)]
            return kept or self._tree.eras
        intervals = _to_intervals(branch.period)
        if not intervals:
            return self._tree.eras
        earliest = min(lo[:7] for lo, _ in intervals)
        kept = [e for e in self._tree.eras if e.span[1][:7] >= earliest]
        return kept or self._tree.eras

    async def _pick_chapters(self, *, branch: RetrieveBranch, ctx: ExecutionContext) -> list[PageRef]:
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
        self, era: EraTree, *, branch: RetrieveBranch, ctx: ExecutionContext
    ) -> list[PageRef]:
        chapters = era.chapters
        if not chapters:
            return []
        listing = sorted(
            (
                {"chapter": name, "n_pages": c.n_pages, "description": c.description, "examples": list(c.examples)}
                for name, c in chapters.items()
            ),
            key=lambda x: -x["n_pages"],
        )
        names = [x["chapter"] for x in listing]
        pages = {name: list(c.refs) for name, c in chapters.items()}

        def _parse_keep(text: str, _ctx: ExecutionContext) -> list[str]:
            bools = self._parse_bool_list(text, _ctx, n=len(names))
            return [nm for nm, keep in zip(names, bools) if keep]

        pick_call: PromptedCall[list[str]] = PromptedCall(
            name="toc_pick",
            system_prompt=self._CHAPTER_PICK_PROMPT,
            parse=_parse_keep,
            # Non-obvious matches (e.g. "unemployment tax receipts" ↔ "Internal Revenue Receipts
            # by State") were missed at effort=off; low reasoning helps recall without being the
            # latency bottleneck (the semfilter scan is).
            default_effort="low",
            output_instruction="Output ONLY a JSON array of true/false — one per chapter, in the given order, no prose.",
        )
        user = json.dumps(
            {"question": ctx.question, "concept": branch.key, "period": branch.period, "chapters": listing},
            ensure_ascii=False,
            indent=1,
        )
        picked = await pick_call.call(ctx, user, temperature=0.0)
        out = [ref for ch in picked for ref in pages[ch]]
        ctx.emit(f"pick_chapters era={era.span} picked={picked!r} pages={len(out)}")
        return out

    def _year_filter(
        self, candidates: list[PageRef], branch: RetrieveBranch, ctx: ExecutionContext
    ) -> list[PageRef]:
        # as_of filters by PUBLICATION month; period filters by DATA span. Both apply when set.
        kept = list(candidates)
        as_of_intervals = _to_intervals(branch.as_of)
        if as_of_intervals:
            kept = [ref for ref in kept if ref.month and _overlaps((ref.month, ref.month), as_of_intervals)]
        period_intervals = _to_intervals(branch.period)
        if period_intervals:
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

    async def _candidates_for(self, branch: RetrieveBranch, ctx: ExecutionContext) -> list[PageRef]:
        chapter_pages = await self._pick_chapters(branch=branch, ctx=ctx)
        return self._year_filter(chapter_pages, branch, ctx)

    async def _semantic_filter(
        self,
        pages: list[PageRef],
        branches: list[RetrieveBranch],
        ctx: ExecutionContext,
    ) -> dict[PageRef, list[bool]]:
        """Judge each candidate page against all branch RETRIEVAL TARGETS, returning one boolean
        per content block. A batch whose reply is unparsable degrades to keep-all (recall over
        precision)."""
        cfg = ctx.config
        targets = [{"target": i + 1, "concept": b.key, "period": b.period} for i, b in enumerate(branches)]
        targets_json = json.dumps(targets, ensure_ascii=False, indent=1)

        verdict: dict[PageRef, list[bool]] = {}
        groups: list[tuple[PageRef, list[dict]]] = []
        for pk in pages:
            row = self._catalog[pk]
            rows = [
                {
                    "bulletin": pk.month, "page": pk.page, "date_interval": row.date_interval,
                    "kind": b.kind, "title": b.title, "column_headers": b.column_headers,
                    "row_headers": b.row_headers, "summary": b.summary,
                }
                for b in row.content_blocks
            ]
            if rows:
                groups.append((pk, rows))
            else:
                verdict[pk] = []  # no blocks → kept wholesale

        cap = max(1, cfg.semfilter_batch_size)
        batches = _batch_by_size(groups, cap)

        async def _one_batch(batch: list[tuple[PageRef, list[dict]]]) -> dict[PageRef, list[bool]]:
            flat = [(pk, r) for pk, rows in batch for r in rows]
            block_rows = [r for _, r in flat]
            n = len(block_rows)
            user = (
                f"targets (JSON, {len(targets)} entries):\n{targets_json}\n\n"
                f"blocks (JSON, {n} entries):\n"
                f"{json.dumps(block_rows, ensure_ascii=False, indent=1)}"
            )

            def _parse_shaped(text: str, _ctx: ExecutionContext) -> list[bool]:
                return self._parse_bool_list(text, _ctx, n=n)

            sem_call: PromptedCall[list[bool]] = PromptedCall(
                name="semfilter",
                system_prompt=self._SEMFILTER_PROMPT,
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
            out: dict[PageRef, list[bool]] = {pk: [] for pk, _ in batch}
            for (pk, _), v in zip(flat, verdicts):
                out[pk].append(v)
            return out

        for d in await asyncio.gather(*[_one_batch(b) for b in batches]):
            verdict.update(d)
        n_kept = sum(1 for pk in pages if not verdict[pk] or any(verdict[pk]))
        n_blocks_kept = sum(sum(verdict[pk]) for pk in pages)
        n_blocks = sum(len(rows) for _, rows in groups)
        ctx.emit(
            f"semantic_filter kept={n_kept}/{len(pages)} "
            f"blocks_kept={n_blocks_kept} blocks={n_blocks}"
        )
        return verdict

    def _block_refs_for(self, ref: PageRef, verdicts: list[bool]) -> list[BlockRef]:
        """Turn a kept page + its per-block verdicts into `BlockRef`s. Empty verdicts → one
        whole-page ref (`block_index=None`)."""
        row = self._catalog.get(ref)
        members = tuple(row.member_refs()) if row is not None else (ref,)
        if not verdicts:
            return [BlockRef(page=ref, block_index=None, member_refs=members, block=None)]
        blocks = row.content_blocks if row is not None else []
        return [
            BlockRef(
                page=ref,
                block_index=bi,
                member_refs=(
                    tuple(row.block_refs(blocks[bi]))
                    if row is not None and bi < len(blocks)
                    else members
                ),
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
        """Retrieve for every branch in one pass and narrow with block selection. Phases:
        (1) per-branch cheap candidates (ToC pick → year filter) run in parallel;
        (2) one semantic-filter sweep over the deduped union; (3) block selection per
        branch in parallel — a selection failure falls back to the semantic-filter output
        for that branch rather than failing it. Output is BLOCK-granular, aligned to `branches`."""
        # Phase 1 — cheap candidates
        cand = await asyncio.gather(*[self._candidates_for(b, ctx) for b in branches])

        wanted: dict[PageRef, set[int]] = {}
        for i, refs in enumerate(cand):
            for ref in refs:
                wanted.setdefault(ref, set()).add(i)
        unique = sorted(wanted, key=lambda r: (r.month or "", r.page or 0))

        # Phase 2 — semantic filter
        if ctx.config.retrieve_skip_semfilter:
            verdict = {
                ref: [True] * len(self._catalog[ref].content_blocks) if ref in self._catalog else []
                for ref in unique
            }
            ctx.emit(f"semantic_filter SKIPPED kept={len(unique)}/{len(unique)}")
        else:
            verdict = await self._semantic_filter(unique, branches, ctx)

        # Route per branch
        branch_blocks: list[list[BlockRef]] = []
        for i, refs in enumerate(cand):
            kept_pages = [ref for ref in refs if not verdict[ref] or any(verdict[ref])]
            block_refs = [br for ref in kept_pages for br in self._block_refs_for(ref, verdict[ref])]
            b = branches[i]
            ctx.emit(
                f"page_index_retrieve key={b.key!r} period={b.period!r} as_of={b.as_of!r} "
                f"catalog_size={self._catalog_size} anchor_count={len(kept_pages)} block_count={len(block_refs)} "
            )
            branch_blocks.append(block_refs)

        # Phase 3 — block selection per branch in parallel
        pdf_dir = str(ctx.config.pdf_dir)

        async def _select(block_refs: list[BlockRef], branch: RetrieveBranch) -> list[BlockRef]:
            if not block_refs:
                return block_refs
            member_refs = list(dict.fromkeys(r for b in block_refs for r in b.member_refs))
            selected = await traced_step(
                ctx, "block_select",
                lambda: self.select_blocks(member_refs, pdf_dir, ctx, ctx.question, branch),
            )
            if not selected:
                ctx.emit(f"block_select_empty key={branch.key!r} falling back to semfilter output")
                return block_refs
            return selected

        settled = await asyncio.gather(
            *(_select(brs, b) for brs, b in zip(branch_blocks, branches)),
            return_exceptions=True,
        )
        out: list[list[BlockRef]] = []
        for block_refs, r in zip(branch_blocks, settled):
            if isinstance(r, BaseException):
                ctx.emit(f"block_select_failed error={str(r)!r}")
                out.append(block_refs)
            else:
                out.append(r)
        return out

    async def _pick_blocks(
        self,
        ctx: ExecutionContext,
        items: list[tuple[str, tuple[PageCatalogRow, int, ContentBlock]]],
        question: str,
        branch: RetrieveBranch,
        *,
        keep: int,
    ) -> list[str]:
        """One tournament group call. Returns at most `keep` block ids, best-first."""
        lines = []
        for bid, (row, _bi, block) in items:
            dates = f"{row.date_interval[0]}..{row.date_interval[1]}" if row.date_interval else "none"
            line = f"[{bid}] dates={dates} | {block.kind} with title: {block.title or '(untitled)'}"
            if block.column_headers:
                line += f" [cols: {', '.join(block.column_headers)}]"
            if block.row_headers:
                line += f" [rows: {', '.join(block.row_headers)}]"
            if block.summary:
                line += f" — content summary: {block.summary}"
            lines.append(line)

        parts = [f'Research question: "{question}"', f"Retrieval target: {branch.key}"]
        if branch.period:
            parts.append(f"Data period: {branch.period}")
        if branch.as_of:
            parts.append(f"Reported in / as of (issue pinned by the plan): {branch.as_of}")
        parts.append(f"Candidate blocks — return at most {keep}:\n" + "\n".join(lines))
        user = "\n".join(parts)

        valid_ids = {bid for bid, _ in items}

        def _parse(text: str, _ctx: ExecutionContext) -> list[str]:
            obj = parse_json_response(text)
            ids = obj.get("block_ids") if isinstance(obj, dict) else None
            if isinstance(ids, str):
                ids = [ids]
            if not isinstance(ids, list):
                raise ParseError(text, 'expected {"block_ids": [...]}')
            for i in ids:
                if str(i) not in valid_ids:
                    raise ParseError(text, f"unknown block_id {str(i)!r} — return only ids from the candidate list")
            return list(dict.fromkeys(str(i) for i in ids))[:keep]

        call: PromptedCall[list[str]] = PromptedCall(
            name="block_select",
            system_prompt=self._BLOCK_SELECT_PROMPT,
            parse=_parse,
            # Thinking OFF: validated no recall regression vs medium, ~10x faster per call.
            default_effort="off",
            output_instruction=(
                f'Output ONLY a JSON object {{"block_ids": [...]}} with AT MOST {keep} '
                "ids, best first (or an empty list if none fit) — no prose."
            ),
        )
        return await call.call(ctx, user, temperature=0.0)

    async def select_blocks(
        self,
        refs: list[PageRef],
        pdf_dir: str | Path,
        ctx: ExecutionContext,
        question: str,
        branch: RetrieveBranch,
    ) -> list[BlockRef]:
        """Tournament-reduce `refs` to at most `_KEEP` blocks for this branch. Uses
        `get_page_store` to resolve continuation refs to their anchor rows. Returns `[]` if no
        ref resolves to a catalog block."""
        store = get_page_store(str(pdf_dir))

        rows: list[PageCatalogRow] = []
        seen_anchors: set[PageRef] = set()
        for ref in refs:
            row = store.catalog_row(ref)
            if row is None or row.ref in seen_anchors:
                continue
            seen_anchors.add(row.ref)
            rows.append(row)

        all_blocks: dict[str, tuple[PageCatalogRow, int, ContentBlock]] = {}
        for row in rows:
            for bi, block in enumerate(row.content_blocks):
                bid = f"{row.bulletin.replace('-', '_')}_{row.page}#{bi}"
                all_blocks[bid] = (row, bi, block)

        if not all_blocks:
            return []

        current = list(all_blocks.items())
        n0 = len(current)
        rounds = 0
        while len(current) > self._GROUP_SIZE:
            rounds += 1
            groups = chunk(current, self._GROUP_SIZE)
            results = await asyncio.gather(
                *(self._pick_blocks(ctx, g, question, branch, keep=self._KEEP) for g in groups)
            )
            survivor_ids = list(dict.fromkeys(bid for ids in results for bid in ids))
            ctx.emit(
                f"block_select_round round={rounds} groups={len(groups)} "
                f"in={len(current)} survivors={len(survivor_ids)}"
            )
            current = [(bid, all_blocks[bid]) for bid in survivor_ids]
            if not current:
                break

        chosen = await self._pick_blocks(ctx, current, question, branch, keep=self._KEEP) if current else []
        selected = []
        for bid in chosen:
            row, bi, block = all_blocks[bid]
            selected.append(BlockRef(page=row.ref, block_index=bi, member_refs=tuple(row.block_refs(block)), block=block))
        ctx.emit(
            f"block_select key={branch.key!r} candidates={n0} rounds={rounds} "
            f"selected_blocks={len(selected)} top_k={self._KEEP}"
        )
        return selected
