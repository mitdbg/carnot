"""Organized extraction over selected blocks — the stage after `block_select`.

`run_extract` takes each retrieve branch's SELECTED blocks (`block_select.run_select`'s
output) and reads them into `AnnotatedValue`s. Its one job beyond calling the extractor
tiers is a cross-branch invariant: a block selected by several branches is **read once**
and its entries are **attributed once** — to the first branch that selected it — so compute
sees each datum exactly once. The unit of work ("unique selected block across all branches")
only exists once every branch's selection is known, which is why this is a sweep over all
branches rather than a per-branch step.

Each unique block is read text-tier-first with a pure-vision fallback (`_extract_block`).
This module owns no selection logic; it shares only block identity (`_block_id`) with
`block_select`, so the two agree on what "the same block" means."""

from __future__ import annotations

import asyncio

from skunk.block_select import _block_id
from skunk.common import (
    AnnotatedValue,
    BlockRef,
    ExecutionContext,
    SemPoolEntry,
    traced_step,
)
from skunk.errors import StepFailed
from skunk.extract import (
    TextExtractor,
    VisionExtractor,
    _blocks_to_pagerefs,
    _render_pages_b64,
)
from skunk.plan import RetrieveBranch

_TEXT = TextExtractor()
_VISION = VisionExtractor()


def _page_keys(e: SemPoolEntry) -> set[str]:
    """Every `month:page` key the entry stands for (anchor + merged member pages)."""
    return {f"{e.ref.page.month}:{e.ref.page.page}"} | {
        f"{m.month}:{m.page}" for m in e.ref.member_refs
    }


def _synth_branch(branches: list[RetrieveBranch]) -> RetrieveBranch:
    """One stamp-bearing branch for a multi-branch page read. Branch identity is
    irrelevant at extraction, so the call-level provenance fields carry the union of
    the requesting branches."""
    keys = list(dict.fromkeys(b.key for b in branches))
    periods = list(dict.fromkeys(p for b in branches if (p := b.period)))
    return RetrieveBranch(
        key="; ".join(keys),
        period=", ".join(periods) or None,
        visual_only=any(b.visual_only for b in branches),
    )


async def _extract_block(
    ctx: ExecutionContext,
    ref: BlockRef,
    branches: list[RetrieveBranch],
) -> list[AnnotatedValue]:
    """One block's read serving EVERY branch that selected it: the call's opening line
    lists all their targets, so a single page read extracts for each. Text tier first,
    pure vision as the fallback — for visual_only branches, the `extract_vision_only`
    override, or a text pass that found nothing."""
    branch = branches[0] if len(branches) == 1 else _synth_branch(branches)
    looking = None
    if len(branches) > 1:
        lines = []
        for b in branches:
            line = f"- {b.key}"
            if b.period:
                line += f" (for the period {b.period})"
            lines.append(line)
        looking = "You are looking for ALL of the following:\n" + "\n".join(lines)
    if not branch.visual_only and not ctx.config.extract_vision_only:
        entries = await _TEXT.run(
            ctx.question, branch, [ref], ctx, looking_for=looking
        )
        if entries:
            return entries
    images, rendered_refs = _render_pages_b64(_blocks_to_pagerefs([ref]), ctx)
    if not images:
        return []
    return await _VISION.run(
        ctx.question, branch, images, rendered_refs, ctx, looking_for=looking
    )


async def run_extract(
    ctx: ExecutionContext,
    branches: list[RetrieveBranch],
    selections: list[list[SemPoolEntry] | StepFailed],
    branch_ids: list[int],
) -> list[list[AnnotatedValue] | StepFailed]:
    """Read every branch's selected blocks: one organized sweep where each unique block is
    read ONCE, mapped to every branch that selected it, its entries owned by the FIRST such
    branch (compute must see each datum exactly once). `selections` is `run_select`'s output,
    one slot per branch — the selected entries, or a `StepFailed` to carry through. Returns
    one result per branch (its entries, or the `StepFailed` to attribute to it)."""
    results: list[list[AnnotatedValue] | StepFailed | None] = [None] * len(branches)
    sel_by_pos: dict[int, list[SemPoolEntry]] = {}
    for pos, sel in enumerate(selections):
        if isinstance(sel, StepFailed):
            results[pos] = sel
        else:
            sel_by_pos[pos] = sel

    # Organized extraction: each unique block read ONCE, mapped to EVERY branch that
    # selected it; entries attributed to the first selecting branch (`owner`).
    want: dict[str, tuple[SemPoolEntry, list[int]]] = {}
    for pos in sorted(sel_by_pos):
        for e in sel_by_pos[pos]:
            bid = _block_id(e)
            if bid in want:
                want[bid][1].append(pos)
            else:
                want[bid] = (e, [pos])
    n_req = sum(len(v) for v in sel_by_pos.values())
    ctx.emit(
        f"select_extract n_requested={n_req} n_reads={len(want)} "
        f"n_already_read={n_req - len(want)}"
    )

    extracted: dict[str, list[AnnotatedValue]] = {}

    async def _extract_phase() -> None:
        reads = await asyncio.gather(
            *(
                _extract_block(ctx, e.ref, [branches[p] for p in poss])
                for e, poss in want.values()
            ),
            return_exceptions=True,
        )
        for bid, res in zip(want, reads):
            if isinstance(res, BaseException):
                ctx.emit(f"select_extract_failed block={bid} error={str(res)!r}")
                extracted[bid] = []
            else:
                extracted[bid] = res

    if want:
        await traced_step(ctx, "extract", _extract_phase)

    for pos in sorted(sel_by_pos):
        sel = sel_by_pos[pos]
        bids = [_block_id(e) for e in sel]
        owned = [
            v
            for bid in bids
            if want[bid][1][0] == pos
            for v in extracted.get(bid, [])
        ]
        covered = any(extracted.get(bid) for bid in bids)
        if covered or owned:
            results[pos] = owned
        else:
            results[pos] = StepFailed(
                "extract",
                f"selected blocks yielded no data for {branches[pos].key!r}",
            )
        sel_pages = sorted(set().union(*(_page_keys(e) for e in sel)) if sel else set())
        ctx.emit(
            f"select_pipeline_branch branch_id={branch_ids[pos]} "
            f"n_blocks={len(bids)} n_entries={len(owned)} covered={covered}",
            data={"branch_id": branch_ids[pos], "pages": sel_pages},
        )
    return [
        r if r is not None else StepFailed("extract", "branch produced no result")
        for r in results
    ]
