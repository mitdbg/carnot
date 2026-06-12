"""Page-index query path: retrieval (`PageIndexRetriever`) and block selection.

`PageIndexRetriever.retrieve_all` runs all branches at once: per-era ToC chapter pick →
year filter → one coarse semantic filter over the deduped union. The filter judges per
CONTENT BLOCK (flat list, one boolean each) against the branch RETRIEVAL TARGETS; a page
is kept iff any block fits any target. Output is BLOCK-granular (`BlockRef`).

`PageIndexRetriever.select_blocks` is the precision stage that follows: a tournament
reduction over the semantic filter's survivors, narrowing each branch to at most `_KEEP`
blocks."""

import json
import asyncio
import logging
import re
from pathlib import Path
from typing import NamedTuple
from skunk.common import (
    BlockRef,
    ExecutionContext,
    PageRef,
    SemPoolEntry,
    chunk,
    parse_json_response,
)
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


class _PeriodEntry(NamedTuple):
    lo: str  # inclusive YYYY-MM data-span start
    hi: str  # inclusive YYYY-MM data-span end
    pin: str | None  # YYYY-MM ISSUE this span's value is pinned to, or None

    @property
    def span(self) -> str:
        return self.lo if self.lo == self.hi else f"{self.lo}..{self.hi}"

    @property
    def label(self) -> str:
        return (
            f"{self.span} (as reported in the {self.pin} issue)"
            if self.pin
            else self.span
        )


def _to_entries(period: str | None) -> list[_PeriodEntry] | None:
    """Parse a period string into entries: a comma-list of `YYYY-MM` months /
    `YYYY-MM..YYYY-MM` ranges. Returns None when empty or malformed (callers then
    no-op the filter). Pins arrive separately on `branch.as_of` — see
    `_branch_entries`."""
    if not period:
        return None
    try:
        out: list[_PeriodEntry] = []
        for part in (p.strip() for p in period.split(",")):
            if not part:
                continue
            lo, _, hi = part.partition("..")
            lo, hi = lo.strip(), (hi.strip() or lo.strip())
            if not (_MONTH_RE.fullmatch(lo) and _MONTH_RE.fullmatch(hi)):
                raise ValueError(f"not YYYY-MM: {part!r}")
            if lo > hi:
                raise ValueError(f"range start > end: {part!r}")
            out.append(_PeriodEntry(lo, hi, None))
        return out or None
    except ValueError:
        return None


def _branch_entries(branch: RetrieveBranch) -> list[_PeriodEntry] | None:
    """The branch's period entries with per-entry issue pins attached from a list
    `as_of` (aligned 1:1 with the entries, None slots = unpinned; parity is enforced
    at plan parse time, so a mismatch here just leaves entries unpinned). A scalar
    `as_of` is a whole-branch publication filter, applied separately."""
    entries = _to_entries(branch.period)
    if entries and isinstance(branch.as_of, list) and len(branch.as_of) == len(entries):
        entries = [
            e._replace(pin=pin) if pin and _MONTH_RE.fullmatch(pin) else e
            for e, pin in zip(entries, branch.as_of)
        ]
    return entries


def _scalar_as_of_intervals(
    as_of: str | list[str | None] | None,
) -> list[tuple[str, str]] | None:
    """Publication-month intervals of a WHOLE-BRANCH (scalar) `as_of`. A list `as_of`
    is per-entry pins — handled via `_branch_entries`, never as a branch-wide filter
    (it would exclude the pages serving the unpinned entries)."""
    return None if isinstance(as_of, list) else _to_intervals(as_of)


def _to_intervals(period: str | None) -> list[tuple[str, str]] | None:
    """The period's `(lo, hi)` data spans — for callers that only care about
    data-time coverage (era pruning)."""
    entries = _to_entries(period)
    return [(e.lo, e.hi) for e in entries] if entries else None


def _overlaps(
    interval: tuple[str, str], period_intervals: list[tuple[str, str]]
) -> bool:
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
in it. Judge every chapter independently and return true if you cannot confidently rule it out.

## Input

A single JSON object:
  {"question": "<question>",
   "chapters": [{"chapter": "<name>", "n_pages": <int>, "description": "<scope>",
                 "examples": ["<sub-area>", ...]}, ...]}

"""
        + CHAPTER_FIELDS
        + """

Each chapter's `description`, and `examples` should be used for reference, but may not contain the verbatim wording of the question even though
it contains relevant information. Use semantic understanding to make the judgement.

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
`block_id`, then `dates=`, the time span the block's data covers, then a compact summary (title, column/row labels) — NOT the numbers.

Judge EACH block and mark `true` only the ones that most directly report the target's data and `false` for every other block.
Respect the cap given, and return fewer blocks when fewer fit. Apply the rules below when deciding:
  - If the question pins a specific source ("as reported in the <Month Year> Bulletin", "as of <date>"), select
    blocks that best match that source.
  - Otherwise prefer the block whose title / headers / date range  match the target most precisely. 
  - When the same figure is restated across many issues, prefer the most recent issue that covers all of the required
    period unless the question explicitly asks for a version.

## Output

A single bare JSON array of booleans — no prose, no markdown fences — one entry per candidate block, in the
SAME ORDER given: `true` to SELECT the block, `false` to drop it. No more than the requested cap may be `true`.
  [true, false, false, ...]"""

    _GROUP_SIZE = 32  # blocks per selection call
    _KEEP = 4  # uniform cap: blocks kept per group AND final blocks kept per period

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
        self._warn_if_tree_stale()

    def _warn_if_tree_stale(self) -> None:
        """Warn when a meaningful share of catalog anchors fall outside every chapter's
        page ranges — such pages can never surface through the ToC pick, so retrieval
        silently loses them. Happens when the tree is rebuilt from stale place files (or
        not refiled after a placement patch; see scripts/refile_concept_tree.py). ~1.5%
        front-matter pages are legitimately unplaced, hence the threshold."""
        covered: dict[str, list[tuple[int, int]]] = {}
        for era in self._tree.eras:
            for c in era.chapters.values():
                for pr in c.pages:
                    covered.setdefault(pr.bulletin, []).append((pr.start, pr.end))
        unreachable = 0
        for ref in self._catalog:
            page = ref.page
            if page is None:
                continue
            if not any(s <= page <= e for s, e in covered.get(ref.month or "", ())):
                unreachable += 1
        frac = unreachable / len(self._catalog) if self._catalog else 0.0
        if frac > 0.05:
            logging.getLogger(__name__).warning(
                "page-index tree is stale: %d/%d catalog anchors (%.1f%%) are in no "
                "chapter page range and are unreachable by retrieval — refile the tree "
                "(python3 -m scripts.refile_concept_tree).",
                unreachable,
                len(self._catalog),
                100 * frac,
            )

    # -- shared parse util -----------------------------------------------------

    @staticmethod
    def _parse_bool_list(
        text: str, _ctx: ExecutionContext, n: int | None = None
    ) -> list[bool]:
        """Parse a JSON boolean array; raises `ParseError` on non-array, non-boolean elements,
        or (when `n` is given) wrong length."""
        obj = parse_json_response(text)
        if not isinstance(obj, list) or not all(isinstance(x, bool) for x in obj):
            raise ParseError(text, "expected a JSON array of booleans (true/false)")
        if n is not None and len(obj) != n:
            raise ParseError(
                text, f"expected {n} booleans (one per item, in order), got {len(obj)}"
            )
        return obj

    # -- retrieval stages ------------------------------------------------------

    def _eras_for_branch(self, branch: RetrieveBranch) -> list[EraTree]:
        """Eras whose ToC to scan. `as_of` pins a publication month (scan only containing eras);
        `period` prunes eras whose span ends before the period starts; neither keeps all eras."""
        as_of_iv = _scalar_as_of_intervals(branch.as_of)
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

    def _eras_for_branches(self, branches: list[RetrieveBranch]) -> list[EraTree]:
        """Union (in tree order) of the eras relevant to ANY branch — the eras the single
        question-driven ToC pick scans. Per-branch date narrowing is reapplied later in the
        per-branch `_year_filter`, so this only needs to be a superset."""
        wanted = {id(e) for b in branches for e in self._eras_for_branch(b)}
        return [e for e in self._tree.eras if id(e) in wanted]

    async def _pick_chapters(
        self, *, branches: list[RetrieveBranch], ctx: ExecutionContext
    ) -> list[PageRef]:
        """One unified ToC pick for the whole question, shared by every branch. Picks chapters
        from the branch-spanning eras conditioned ONLY on the original question text (not any
        branch's concept/period); returns the deduped union of candidate pages, which each
        branch then narrows by its own period/as_of in `_year_filter`."""
        eras = self._eras_for_branches(branches)
        ctx.emit(
            f"pick_chapters_eras eras={len(eras)}/{len(self._tree.eras)} branches={len(branches)}"
        )
        per_era = await asyncio.gather(
            *[self._pick_era_chapters(era, ctx=ctx) for era in eras]
        )
        seen: set[PageRef] = set()
        out: list[PageRef] = []
        for era_pages in per_era:
            for ref in era_pages:
                if ref not in seen:
                    seen.add(ref)
                    out.append(ref)
        # Survivor identities (not just counts) so per-stage recall/elimination is
        # computable post-hoc from events.jsonl (eval/stage_report.py).
        ctx.emit(
            f"toc_pick_pages n={len(out)}",
            data={"pages": [f"{r.month}:{r.page}" for r in out]},
        )
        return out

    async def _pick_era_chapters(
        self, era: EraTree, *, ctx: ExecutionContext
    ) -> list[PageRef]:
        chapters = era.chapters
        if not chapters:
            return []
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
            {"question": ctx.question, "chapters": listing},
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
        as_of_intervals = _scalar_as_of_intervals(branch.as_of)
        if as_of_intervals:
            kept = [
                ref
                for ref in kept
                if ref.month and _overlaps((ref.month, ref.month), as_of_intervals)
            ]
        entries = _branch_entries(branch)
        if entries:
            # A candidate passes on ANY entry: an unpinned entry wants data-span
            # overlap; a pinned entry wants the page PUBLISHED in the pinned issue
            # (and a missing date_interval doesn't disqualify pages of the named
            # issue — the pin already narrows to one bulletin).
            def _passes(ref: PageRef) -> bool:
                row = self._catalog.get(ref)
                if row is None:
                    return False
                for e in entries:
                    if e.pin is not None:
                        if ref.month == e.pin and (
                            row.date_interval is None
                            or _overlaps(row.date_interval, [(e.lo, e.hi)])
                        ):
                            return True
                    elif row.date_interval is not None and _overlaps(
                        row.date_interval, [(e.lo, e.hi)]
                    ):
                        return True
                return False

            kept = [ref for ref in kept if _passes(ref)]
        ctx.emit(
            f"year_filter as_of={branch.as_of!r} period={branch.period!r} "
            f"kept={len(kept)}/{len(candidates)}",
            data={"pages": [f"{r.month}:{r.page}" for r in kept]},
        )
        return kept

    async def _semantic_filter(
        self,
        pages: list[PageRef],
        branches: list[RetrieveBranch],
        ctx: ExecutionContext,
    ) -> dict[PageRef, list[bool]]:
        """Judge each candidate page against all branch RETRIEVAL TARGETS, returning one boolean
        per content block. A batch whose reply is unparsable (after parse retries) raises
        StepFailed — no keep-all fallback — so the sweep fails and every branch replans."""
        cfg = ctx.config
        targets = [
            {"target": i + 1, "concept": b.key, "period": b.period}
            for i, b in enumerate(branches)
        ]
        targets_json = json.dumps(targets, ensure_ascii=False, indent=1)

        verdict: dict[PageRef, list[bool]] = {}
        groups: list[tuple[PageRef, list[dict]]] = []
        for pk in pages:
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
                verdict[pk] = []  # no blocks → kept wholesale

        cap = max(1, cfg.semfilter_batch_size)
        batches = _batch_by_size(groups, cap)

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
                # No keep-all fallback: an unparsable verdict (after parse retries) fails
                # the retrieve sweep so every branch replans, rather than silently keeping
                # the whole batch (which re-inflates the downstream block set).
                raise StepFailed(
                    "retrieve",
                    f"semfilter verdict unparsable for {n}-block batch: {e}",
                ) from e
            out: dict[PageRef, list[bool]] = {pk: [] for pk, _ in batch}
            for (pk, _), v in zip(flat, verdicts):
                out[pk].append(v)
            return out

        for d in await asyncio.gather(*[_one_batch(b) for b in batches]):
            verdict.update(d)
        kept_refs = [pk for pk in pages if not verdict[pk] or any(verdict[pk])]
        n_blocks_kept = sum(sum(verdict[pk]) for pk in pages)
        n_blocks = sum(len(rows) for _, rows in groups)
        ctx.emit(
            f"semantic_filter kept={len(kept_refs)}/{len(pages)} "
            f"blocks_kept={n_blocks_kept} blocks={n_blocks}",
            data={"pages": [f"{r.month}:{r.page}" for r in kept_refs]},
        )
        return verdict

    def _block_refs_for(self, ref: PageRef, verdicts: list[bool]) -> list[BlockRef]:
        """Turn a kept page + its per-block verdicts into `BlockRef`s. Empty verdicts → one
        whole-page ref (`block_index=None`)."""
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
        """Phases (1) and (2) only: ToC pick → year filter → semantic filter.
        Returns the sem-filter survivor blocks per branch, aligned to `branches`.
        No block selection — the caller (orchestrator) runs tournament selection or
        selection-agent as a separate step."""
        # Phase 1 — one unified ToC pick (question only), then each branch's own date filter.
        chapter_pages = await self._pick_chapters(branches=branches, ctx=ctx)
        cand = [self._year_filter(chapter_pages, b, ctx) for b in branches]

        wanted: dict[PageRef, set[int]] = {}
        for i, refs in enumerate(cand):
            for ref in refs:
                wanted.setdefault(ref, set()).add(i)
        unique = sorted(wanted, key=lambda r: (r.month or "", r.page or 0))

        # Phase 2 — semantic filter
        verdict = await self._semantic_filter(unique, branches, ctx)

        # Route per branch
        branch_blocks: list[list[BlockRef]] = []
        for i, refs in enumerate(cand):
            kept_pages = [ref for ref in refs if not verdict[ref] or any(verdict[ref])]
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
            branch_blocks.append(block_refs)

        return branch_blocks

    async def _pick_blocks(
        self,
        ctx: ExecutionContext,
        items: list[tuple[str, tuple[PageCatalogRow, int, ContentBlock]]],
        question: str,
        branch: RetrieveBranch,
        *,
        keep: int,
        period_label: str | None = None,
    ) -> list[str]:
        """One tournament group call. Returns at most `keep` block ids, best-first.
        `period_label` overrides the displayed data period (set when selecting for one
        interval of a multi-interval period)."""
        lines = []
        for bid, (row, _bi, block) in items:
            dates = (
                f"{row.date_interval[0]}..{row.date_interval[1]}"
                if row.date_interval
                else "none"
            )
            line = f"[{bid}] dates={dates} | {block.kind} with title: {block.title or '(untitled)'}"
            if block.column_headers:
                line += f" [cols: {', '.join(block.column_headers)}]"
            if block.row_headers:
                line += f" [rows: {', '.join(block.row_headers)}]"
            if block.summary:
                line += f" — content summary: {block.summary}"
            lines.append(line)

        parts = [f'Research question: "{question}"', f"Retrieval target: {branch.key}"]
        if period_label or branch.period:
            parts.append(f"Data period: {period_label or branch.period}")
        n = len(items)
        parts.append(
            f"Candidate blocks ({n}) — mark true the AT MOST {keep} that best report the target, "
            "false for the rest:\n" + "\n".join(lines)
        )
        user = "\n".join(parts)

        def _parse(text: str, _ctx: ExecutionContext) -> list[bool]:
            # Parity-checked boolean array (one true/false per block); a wrong length
            # ParseErrors → PromptedCall retries.
            return self._parse_bool_list(text, _ctx, n=n)

        call: PromptedCall[list[bool]] = PromptedCall(
            name="block_select",
            system_prompt=self._BLOCK_SELECT_PROMPT,
            parse=_parse,
            default_effort="off",  # per-block boolean keep/drop, thinking off
            output_instruction=(
                f"Output ONLY a JSON array of EXACTLY {n} booleans — one per block, in the order "
                f"given (true=select, false=drop), with AT MOST {keep} true. No prose, no markdown fences."
            ),
        )
        verdicts = await call.call(ctx, user, temperature=0.0)
        # Keep the selected (true) ids in input order, capped at `keep` (defensive — the prompt
        # already asks for ≤keep true; truncation only bites if the model over-selects).
        return [bid for (bid, _), keep_it in zip(items, verdicts) if keep_it][:keep]

    async def _tournament(
        self,
        ctx: ExecutionContext,
        blocks: list[tuple[str, tuple[PageCatalogRow, int, ContentBlock]]],
        question: str,
        branch: RetrieveBranch,
        *,
        final_keep: int,
        period_label: str | None = None,
    ) -> tuple[list[str], int]:
        """Tournament-reduce `blocks` to at most `final_keep` ids, best-first: while the
        field exceeds one group, keep the best `_KEEP` of each `_GROUP_SIZE` group, then one
        precision call over the survivors. Returns `(chosen_ids, rounds)`."""
        by_id = dict(blocks)
        current = blocks
        rounds = 0
        while len(current) > self._GROUP_SIZE:
            rounds += 1
            groups = chunk(current, self._GROUP_SIZE)
            results = await asyncio.gather(
                *(
                    self._pick_blocks(
                        ctx,
                        g,
                        question,
                        branch,
                        keep=self._KEEP,
                        period_label=period_label,
                    )
                    for g in groups
                )
            )
            survivor_ids = list(dict.fromkeys(bid for ids in results for bid in ids))
            ctx.emit(
                f"block_select_round round={rounds} groups={len(groups)} "
                f"in={len(current)} survivors={len(survivor_ids)}"
            )
            current = [(bid, by_id[bid]) for bid in survivor_ids]
            if not current:
                break

        if not current:
            return [], rounds
        chosen = await self._pick_blocks(
            ctx, current, question, branch, keep=final_keep, period_label=period_label
        )
        return chosen, rounds

    @staticmethod
    def pool_for_blocks(
        blocks: list[BlockRef], pdf_dir: str | Path
    ) -> list[SemPoolEntry]:
        """Pool entries for exactly `blocks` (no selection): a specific block maps to its
        own entry; a whole-page block (`block_index=None`) expands to every block on its
        anchor's catalog row. Deduped on (anchor, block_index). Static — needs only the
        page store, so the selection agent can rebuild a pool without loading the tree."""
        store = get_page_store(str(pdf_dir))
        out: list[SemPoolEntry] = []
        seen: set[tuple[PageRef, int]] = set()
        for b in blocks:
            row = store.catalog_row(b.page)
            if row is None:
                continue
            idxs = (
                [b.block_index]
                if b.block_index is not None and b.block_index < len(row.content_blocks)
                else range(len(row.content_blocks))
            )
            for bi in idxs:
                key = (row.ref, bi)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    PageIndexRetriever._pool_entry(row, bi, row.content_blocks[bi])
                )
        return out

    def build_sem_pool(
        self, refs: list[PageRef], pdf_dir: str | Path
    ) -> list[SemPoolEntry]:
        """The repair pool for `refs` without running any selection: anchor-resolve the
        refs (same prologue as `select_blocks`) and emit every content block as a
        `SemPoolEntry`. Used by the eval harness to dump pools at sem-filter cost."""
        store = get_page_store(str(pdf_dir))
        rows: list[PageCatalogRow] = []
        seen: set[PageRef] = set()
        for ref in refs:
            row = store.catalog_row(ref)
            if row is None or row.ref in seen:
                continue
            seen.add(row.ref)
            rows.append(row)
        return [
            self._pool_entry(row, bi, block)
            for row in rows
            for bi, block in enumerate(row.content_blocks)
        ]

    @staticmethod
    def _pool_entry(row: PageCatalogRow, bi: int, block: ContentBlock) -> SemPoolEntry:
        """A self-contained repair-pool entry for one candidate block."""
        return SemPoolEntry(
            ref=BlockRef(
                page=row.ref,
                block_index=bi,
                member_refs=tuple(row.block_refs(block)),
                block=None,
            ),
            interval=(
                (row.date_interval[0], row.date_interval[1])
                if row.date_interval
                else None
            ),
            kind=block.kind,
            title=block.title,
            summary=block.summary,
            cols=tuple(block.column_headers[:12]),
            rows_tail=tuple(block.row_headers[-8:]),
            rows=tuple(block.row_headers),
        )

    async def select_blocks(
        self,
        refs: list[PageRef],
        pdf_dir: str | Path,
        ctx: ExecutionContext,
        question: str,
        branch: RetrieveBranch,
    ) -> tuple[list[BlockRef], list[SemPoolEntry]]:
        """Tournament-reduce `refs` to the branch's blocks. Uses `get_page_store` to resolve
        continuation refs to their anchor rows. Returns `(selected, pool)` where `pool` is
        every candidate block as a `SemPoolEntry` (the repair pool — what extract's coverage
        repair re-selects from). `([], [])` if no ref resolves to a catalog block.

        Block selection runs PER PERIOD: one sub-selection per period entry (over the blocks whose
        data span overlaps that entry), each capped at `_KEEP`, and unions the winners. A branch
        with no parseable period runs a single selection over all candidates. A flat per-branch cap
        would let one period's blocks crowd out another's; there is deliberately no global
        re-narrowing pass over the union."""
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
            return [], []
        pool = [
            self._pool_entry(row, bi, block) for row, bi, block in all_blocks.values()
        ]

        items = list(all_blocks.items())
        n0 = len(items)
        entries = _branch_entries(branch)

        def _entry_part(
            e: _PeriodEntry,
        ) -> list[tuple[str, tuple[PageCatalogRow, int, ContentBlock]]]:
            # A pinned entry's pool is the pinned ISSUE's blocks only; an unpinned
            # entry's pool is everything whose data span overlaps (no-interval pages
            # join every pool).
            return [
                it
                for it in items
                if (e.pin is None or it[1][0].bulletin == e.pin)
                and (
                    it[1][0].date_interval is None
                    or _overlaps(it[1][0].date_interval, [(e.lo, e.hi)])
                )
            ]

        # Block selection runs PER PERIOD: one sub-selection per period entry (restricted to the
        # candidates whose data span overlaps it), each capped at `_KEEP`, then unioned. A branch
        # with no parseable period runs a single selection over all candidates. There is no global
        # re-narrowing pass over the union — each period keeps its own blocks.
        parts: list[
            tuple[
                str | None, list[tuple[str, tuple[PageCatalogRow, int, ContentBlock]]]
            ]
        ]
        if entries:
            parts = [(e.label, p) for e in entries if (p := _entry_part(e))]
            if not parts:  # no candidate overlaps any entry (unlikely post year-filter)
                parts = [(entries[0].label, items)]
        else:
            parts = [(None, items)]

        results = await asyncio.gather(
            *(
                self._tournament(
                    ctx,
                    part,
                    question,
                    branch,
                    final_keep=self._KEEP,
                    period_label=label,
                )
                for label, part in parts
            )
        )
        for (label, part), (ids, _) in zip(parts, results):
            ctx.emit(
                f"block_select_interval interval={label!r} "
                f"candidates={len(part)} selected={len(ids)}"
            )
        rounds = max((r for _, r in results), default=0)
        chosen = list(dict.fromkeys(bid for ids, _ in results for bid in ids))
        cap = self._KEEP * len(parts)

        selected = []
        for bid in chosen:
            row, bi, block = all_blocks[bid]
            selected.append(
                BlockRef(
                    page=row.ref,
                    block_index=bi,
                    member_refs=tuple(row.block_refs(block)),
                    block=block,
                )
            )
        ctx.emit(
            f"block_select key={branch.key!r} candidates={n0} rounds={rounds} "
            f"selected_blocks={len(selected)} top_k={cap}"
        )
        return selected, pool
