"""Block-selection stage — a tournament-reduction selector (PROTOTYPE).

Sits between the recall-over-precision semantic filter and extract, narrowing the filter's
survivors to exactly what a sub-question needs. It reduces the candidates with a TOURNAMENT
of small packed calls, selecting at BLOCK granularity (the unit the semantic filter already
judges), then expands the chosen blocks' pages to extract-ready refs.

The tournament: partition the candidate blocks into groups of at most `_GROUP_SIZE`, ask the
model to keep the best `_FANOUT_KEEP` per group, union the winners, and repeat over the
survivors until the field fits one group; a final precision call over that group narrows to
`_MAX_SELECT`. Each call sees only a few dozen blocks, so it reasons over a manageable slate
side by side (every vintage of a table, every period of a series) instead of grepping, and a
strong block need only win its small group to advance — no candidate is judged against the
whole 1000-block field at once, and nothing relevant is crowded out before the final round.
The output is HARD-CAPPED at `_MAX_SELECT` blocks per retrieval target, so the picker always
narrows and a branch can never dump its candidate set into extract.

Its surface — `__init__(refs, pdf_dir)`, `has_candidates()`,
`async select(ctx, question, branch) -> list[PageRef]` — drops into the orchestrator
seam (`_maybe_select`, behind `config.block_select`). `select_blocks` exposes the underlying
block-granular selection for callers that want it.

PROTOTYPE: wired behind `config.block_select`; under evaluation.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from skunk.common import ExecutionContext, PageRef, parse_json_response
from skunk.errors import ParseError
from skunk.plan import RetrieveBranch
from skunk.prompted_call import PromptedCall
from skunk.page_index.data_model import ContentBlock, PageCatalogRow
from skunk.page_index.query import BlockRef
from skunk.page_index.store import get_page_store

# Blocks per tournament call. Small enough that the model weighs the whole slate carefully;
# a block is only a few hundred tokens. Tune via SKUNK_BLOCK_SELECT_GROUP.
_GROUP_SIZE = int(os.environ.get("SKUNK_BLOCK_SELECT_GROUP", "64"))

# Blocks each group call advances to the next round (its high-recall fan-out). Larger than
# `_MAX_SELECT` so a near-miss survives its group and gets a global comparison before being
# dropped. Tune via SKUNK_BLOCK_SELECT_FANOUT.
_FANOUT_KEEP = int(os.environ.get("SKUNK_BLOCK_SELECT_FANOUT", "4"))

# Hard cap on the picker's OUTPUT per retrieval target: the final call returns at most this
# many blocks, enforced both in the prompt and in code (replies are truncated). Bounds the
# pages handed to extract so a branch can never dump its candidate set downstream. Tune via
# SKUNK_BLOCK_SELECT_TOPK.
_MAX_SELECT = int(os.environ.get("SKUNK_BLOCK_SELECT_TOPK", "3"))

# Max group calls in flight at once within a round — bounds the per-minute token burst (and
# the 429 rate) when a big branch fans out to dozens of groups. Tune via
# SKUNK_BLOCK_SELECT_CONCURRENCY.
_CONCURRENCY = int(os.environ.get("SKUNK_BLOCK_SELECT_CONCURRENCY", "8"))


def _page_key(row: PageCatalogRow) -> str:
    """The model-facing key for a candidate page (`YYYY_MM_pageid`)."""
    return f"{row.bulletin.replace('-', '_')}_{row.page}"


def _block_id(row: PageCatalogRow, block_index: int) -> str:
    """Stable id for one block on a page (`page_key#blockidx`). Round-trips through the model:
    robust to reordering/omission, unlike a bare positional integer."""
    return f"{_page_key(row)}#{block_index}"


def _block_line(bid: str, dates: str, block: ContentBlock) -> str:
    """One packed summary line for a block — its id, the page's data span, then the block's
    kind / title / headers / summary (no numeric values)."""
    part = f"[{bid}] dates={dates} | {block.kind}: {block.title or '(untitled)'}"
    if block.column_headers:
        part += f" [cols: {', '.join(block.column_headers)}]"
    if block.row_headers:
        part += f" [rows: {', '.join(block.row_headers)}]"
    if block.summary:
        part += f" — {block.summary}"
    return part


_SYSTEM_PROMPT = """\
You select which already-filtered Treasury Bulletin CONTENT BLOCKS actually carry the data a
research question needs. You are given the full QUESTION, a specific RETRIEVAL TARGET (the data
concept to find for it), and candidate CONTENT BLOCKS — one block per line. Each block line
starts with its `block_id` of the form `YYYY_MM_page#block`: the leading `YYYY_MM` is the ISSUE
(the bulletin's publication month) the block appears in, then comes `dates=` (the time span the
block's DATA covers — which may be earlier than the issue) and a compact summary (title,
column/row labels) — NOT the numbers.

Return ONLY the blocks that most directly report the target's data, best-first, and never more
than the cap stated in the request (the "return at most N" line) — fewer when fewer are
relevant. These are the blocks an extraction step will read to answer the target. When choosing:
  - READ THE FULL QUESTION for any issue it specifies. If it pins a particular bulletin — e.g.
    "as reported in the <Month Year> Bulletin", "in the <Month Year> issue", "as of <date>" —
    select blocks whose ISSUE (the `YYYY_MM` in the `block_id`) is exactly that bulletin; the
    question's explicit wording overrides the generic data period.
  - otherwise prefer the block whose title / headers / data span match the target most
    precisely; prefer a data table over a chart of the same series (a chart has no readable
    values to extract);
  - MATCH THE EXACT STATISTIC the question asks for — not a broader or narrower aggregate that
    merely shares wording. A series for "X and related activities" (or "total X, including …")
    reports a DIFFERENT number than bare "X": e.g. "national defense" (a single function) is not
    "national defense and related activities" (a broader aggregate). Read the question's wording
    and pick the block whose reported quantity is exactly that scope; do not substitute a wider
    total or a sub-component, even if its title looks similar;
  - if the data is split across pages of ONE issue (a table continued), pick the page that holds
    the relevant rows;
  - among near-duplicate reprints (the SAME table recurs across issues because they restate /
    revise it), pick the right vintage by intent: a point-in-time value, or one pinned to an
    issue → that issue; a figure later revised where the question wants the best / current value
    → the LATEST issue that reports that date — do NOT return every reprint.

Never invent a `block_id`; only return ids that appear in the list.

## Output

A single JSON object, no prose, no markdown fences, listing the selected ids (best first, no
more than the requested cap) under "block_ids":
  {"block_ids": ["2001_06_41#0"]}
Use each `block_id` exactly as it appears."""


class BlockSelectAgent:
    """Tournament-reduction block selector. Resolves a branch's candidate refs to their
    distinct catalog anchor rows, explodes those into content blocks, and (in `select_blocks`)
    reduces them to `_MAX_SELECT` via rounds of small packed group calls. One instance per
    branch."""

    def __init__(
        self,
        refs: list[PageRef],
        pdf_dir: str | Path,
    ) -> None:
        store = get_page_store(str(pdf_dir))

        # Resolve (expanded) candidate refs to distinct anchor rows, first-seen order — a
        # continuation ref resolves to its anchor; dedup so each page appears once.
        rows: list[PageCatalogRow] = []
        seen: set[PageRef] = set()
        for ref in refs:
            row = store.catalog_row(ref)
            if row is None or row.ref in seen:
                continue
            seen.add(row.ref)
            rows.append(row)
        self._rows = rows

        # Flatten to candidate blocks (block_id -> (row, block_index, block)), in row order.
        self._blocks: dict[str, tuple[PageCatalogRow, int, ContentBlock]] = {}
        for row in rows:
            for bi, block in enumerate(row.content_blocks):
                self._blocks[_block_id(row, bi)] = (row, bi, block)

    def has_candidates(self) -> bool:
        """Whether any candidate ref resolved to a catalog block (else nothing to select over —
        the caller should keep the original refs)."""
        return bool(self._blocks)

    @staticmethod
    def _to_block_ref(row: PageCatalogRow, bi: int, block: ContentBlock) -> BlockRef:
        return BlockRef(
            page=row.ref,
            block_index=bi,
            member_refs=tuple(row.member_refs()),
            block=block,
        )

    async def _pick(
        self,
        ctx: ExecutionContext,
        items: list[tuple[str, tuple[PageCatalogRow, int, ContentBlock]]],
        question: str,
        branch: RetrieveBranch,
        *,
        keep: int,
    ) -> list[str]:
        """One packed selection call over `items` (one tournament group). Returns at most `keep`
        block ids, best-first. An empty reply (`{"block_ids": []}`) is a VALID answer — none of
        this group's blocks are right — and returns `[]`. Validation rides PromptedCall's reprompt
        loop: the parser closes over the group's valid ids and raises `ParseError` on a bad shape
        or an invented id, so the model is re-asked to correct it (no local fallback)."""
        listing = "\n".join(
            _block_line(bid, _dates(row), block) for bid, (row, _bi, block) in items
        )
        parts = [
            f'Research question: "{question}"',
            f"Retrieval target (the sub-question / data concept to find): {branch.key}",
        ]
        if branch.period:
            parts.append(f"Data period: {branch.period}")
        if branch.as_of:
            parts.append(
                f"Reported in / as of (issue pinned by the plan): {branch.as_of}"
            )
        parts.append(
            f"Candidate blocks ({len(items)}) — return at most {keep}, or none if none fit:"
            f"\n{listing}"
        )
        user = "\n".join(parts)

        valid_ids = {bid for bid, _ in items}

        def _parse(text: str, _ctx: ExecutionContext) -> list[str]:
            obj = parse_json_response(text)
            ids = obj.get("block_ids") if isinstance(obj, dict) else None
            if isinstance(ids, str):
                ids = [ids]
            if not isinstance(ids, list):
                raise ParseError(text, 'expected {"block_ids": [...]}')
            out: list[str] = []
            seen: set[str] = set()
            for i in ids:
                s = str(i)
                if s not in valid_ids:
                    raise ParseError(
                        text, f"unknown block_id {s!r} — return only ids from the candidate list"
                    )
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            return out[:keep]

        call: PromptedCall[list[str]] = PromptedCall(
            name="block_select",
            system_prompt=_SYSTEM_PROMPT,
            parse=_parse,
            # Thinking OFF: validated no recall regression vs medium, ~10x faster per call, and
            # avoids the runaway-thought-trace calls that blow the per-request timeout under load.
            default_effort="off",
            output_instruction=(
                f'Output ONLY a JSON object {{"block_ids": [...]}} with AT MOST {keep} '
                "ids, best first (or an empty list if none fit) — no prose."
            ),
        )
        return await call.call(ctx, user, temperature=0.0)

    async def select_blocks(
        self,
        ctx: ExecutionContext,
        question: str,
        branch: RetrieveBranch,
    ) -> list[BlockRef]:
        """Pick at most `_MAX_SELECT` blocks for this retrieval target, best-first, by tournament
        reduction: while the field is larger than one group, partition it into `_GROUP_SIZE`
        groups, keep the best `_FANOUT_KEEP` of each (rounds run concurrently, bounded by
        `_CONCURRENCY`), and recurse over the union of winners; then a final precision call over
        the surviving group narrows to `_MAX_SELECT`. A round that yields no survivors means none
        of the candidates fit — the selection is empty. The output is ALWAYS ≤ `_MAX_SELECT` —
        the picker never returns the whole branch (no keep-all), so a branch can't flood extract."""
        current = list(self._blocks.items())
        if not current:
            return []
        n0 = len(current)

        sem = asyncio.Semaphore(_CONCURRENCY)

        async def pick_group(
            group: list[tuple[str, tuple[PageCatalogRow, int, ContentBlock]]],
        ) -> list[str]:
            async with sem:
                return await self._pick(ctx, group, question, branch, keep=_FANOUT_KEEP)

        # Reduction rounds: shrink the field until it fits one group.
        rounds = 0
        while len(current) > _GROUP_SIZE:
            rounds += 1
            groups = [
                current[i : i + _GROUP_SIZE]
                for i in range(0, len(current), _GROUP_SIZE)
            ]
            results = await asyncio.gather(*(pick_group(g) for g in groups))
            survivor_ids: list[str] = []
            seen: set[str] = set()
            for ids in results:
                for bid in ids:
                    if bid not in seen:
                        seen.add(bid)
                        survivor_ids.append(bid)
            ctx.emit(
                f"block_select_round round={rounds} groups={len(groups)} "
                f"in={len(current)} survivors={len(survivor_ids)}"
            )
            current = [(bid, self._blocks[bid]) for bid in survivor_ids]
            if not current:
                break

        # Final precision pick over the ≤ `_GROUP_SIZE` survivors (empty if none survived).
        chosen = (
            await self._pick(ctx, current, question, branch, keep=_MAX_SELECT)
            if current
            else []
        )
        selected = [self._to_block_ref(*self._blocks[bid]) for bid in chosen]
        ctx.emit(
            f"block_select key={branch.key!r} candidates={n0} rounds={rounds} "
            f"selected_blocks={len(selected)} top_k={_MAX_SELECT}"
        )
        return selected

    async def select(
        self,
        ctx: ExecutionContext,
        question: str,
        branch: RetrieveBranch,
    ) -> list[PageRef]:
        """Run the block selection, then collapse the
        chosen blocks to extract-ready page refs (their anchors expanded to member pages,
        deduped). Several selected blocks on one page collapse to that page once."""
        blocks = await self.select_blocks(ctx, question, branch)
        out: list[PageRef] = []
        seen: set[PageRef] = set()
        for b in blocks:
            for ref in b.member_refs:
                if ref not in seen:
                    seen.add(ref)
                    out.append(ref)
        return out


def _dates(row: PageCatalogRow) -> str:
    """The page's `YYYY-MM..YYYY-MM` data span, or "none" when undatable."""
    return (
        f"{row.date_interval[0]}..{row.date_interval[1]}"
        if row.date_interval
        else "none"
    )
