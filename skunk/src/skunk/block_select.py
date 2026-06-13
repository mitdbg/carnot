"""Tournament-style block selection over semantic-filter survivor blocks.

`run_select` is the precision stage between `retrieve` and `compute`: for each retrieve
branch it reduces the branch's sem-filter survivor pool to the blocks an extraction step
should read. Its selected blocks are then read by `block_extract.run_extract` — selection
hands off SELECTED blocks and knows nothing about how they are extracted.

Selection runs PER PERIOD ENTRY (one sub-selection per entry of the branch's period,
restricted to the candidates whose data span overlaps it). Every LLM call is the SAME
keep/drop bracket (`_SELECT_PROMPT`) over title-grouped candidates with row/column labels
in view — at most `_BATCH` blocks in, at most `_KEEP` kept out. Parallel narrowing brackets
run until the field fits one bracket; that bracket's keeps ARE the selection (no distinct
"final" call — every bracket is identical).

A branch whose survivors are empty, or where the final call keeps nothing, surfaces as
a per-branch `StepFailed` (→ replanner re-routes) rather than dumping the uncapped set
into extract."""

from __future__ import annotations

import asyncio
import re
from collections import Counter

from skunk.common import (
    BranchRetrieval,
    ExecutionContext,
    SemPoolEntry,
    traced_step,
)
from skunk.errors import StepFailed
from skunk.page_index.query import PageIndexRetriever, _branch_entries, _overlaps
from skunk.plan import RetrieveBranch
from skunk.prompted_call import PromptedCall


# Selection shape: every call is the SAME keep/drop bracket — at most `_BATCH` candidate
# blocks in, at most `_KEEP` kept out. Parallel narrowing brackets run until the field fits
# ONE bracket, whose keeps ARE the selection. There is no separate "final" call: every
# bracket is identical (same input cap, same output cap).
_BATCH = 64
_KEEP = 8


# ---------------------------------------------------------------------------
# Pool helpers
# ---------------------------------------------------------------------------


def _block_id(e: SemPoolEntry) -> str:
    bi = e.ref.block_index
    return f"{e.ref.page.month}:{e.ref.page.page}#{'w' if bi is None else bi}"


def _ident(e: SemPoolEntry) -> str:
    return f"{e.ref.page.month} p.{e.ref.page.page}"


def _canon_title(raw: str | None) -> str:
    """Canonical grouping key for a title: whitespace collapsed, trailing punctuation
    stripped, case folded — so reprints whose captions differ only by OCR noise
    ("STATUTORY DEBT LIMITATION" vs "Statutory Debt Limitation.") form ONE group."""
    return re.sub(r"\s+", " ", raw or "(untitled)").strip().rstrip(" .,;:").casefold()


def _title_groups(pool: list[SemPoolEntry], idxs: list[int]) -> list[tuple[str, list[int]]]:
    """Group pool indices by CANONICAL title into `(display_title, [idx, ...])`, members
    in pool (issue) order; the display title is the group's most common cleaned variant.
    Groups ordered by descending size then title, so reprint runs lead and same-title
    blocks share a drop call (the coverage rule only applies within a title)."""
    groups: dict[str, list[int]] = {}
    variants: dict[str, Counter] = {}
    for i in idxs:
        raw = pool[i].title or f"(untitled {pool[i].kind})"
        key = _canon_title(raw)
        groups.setdefault(key, []).append(i)
        cleaned = re.sub(r"\s+", " ", raw).strip().rstrip(" .,;:")
        variants.setdefault(key, Counter())[cleaned or raw] += 1
    out = [(variants[k].most_common(1)[0][0], v) for k, v in groups.items()]
    return sorted(out, key=lambda kv: (-len(kv[1]), kv[0]))


def _title_batches(
    pool: list[SemPoolEntry], idxs: list[int], cap: int
) -> list[list[tuple[str, list[int]]]]:
    """Pack title groups into batches of at most `cap` blocks, splitting oversized
    groups into consecutive (issue-ordered) chunks so coverage neighbours stay in
    one call."""
    batches: list[list[tuple[str, list[int]]]] = []
    cur: list[tuple[str, list[int]]] = []
    n = 0
    for title, members in _title_groups(pool, idxs):
        for k in range(0, len(members), cap):
            chunk = members[k : k + cap]
            if cur and n + len(chunk) > cap:
                batches.append(cur)
                cur, n = [], 0
            cur.append((title, chunk))
            n += len(chunk)
    if cur:
        batches.append(cur)
    return batches


def _block_line(num: int, e: SemPoolEntry) -> str:
    """One candidate line: issue/page, data span, then axis labels. The series the question
    names sits on the row OR column axis (varies by table), so show BOTH when their combined
    count is < 128; once at/over that, show only the SHORTER axis — it bounds input size and
    is usually the series breakdown (the disambiguating one), since the date axis is long."""
    dates = f"{e.interval[0]}..{e.interval[1]}" if e.interval else "none"
    line = f"[{num}] {_ident(e)}  dates={dates}"
    rows = list(e.rows or e.rows_tail)
    cols = list(e.cols)
    if len(rows) + len(cols) >= 128:
        if len(rows) <= len(cols):
            cols = []
        else:
            rows = []
    if cols:
        line += f"  [cols: {', '.join(cols)}]"
    if rows:
        line += f"  [rows: {', '.join(rows)}]"
    if e.summary:
        line += f"\n    summary: {e.summary}"
    return line


# ---------------------------------------------------------------------------
# Selection — ONE prompt for every call; narrowing rounds until the survivors
# fit one final call, whose keeps are the selection
# ---------------------------------------------------------------------------

_SELECT_PROMPT = """\
You select the content blocks an extraction step should read to fulfill one retrieval
goal of a research question. Each candidate is one table/chart/prose block from a
Treasury Bulletin page, described by catalog metadata (issue, page, data-date span,
column and row labels, summary), grouped by table title. You may be shown the whole
candidate set or one slice of it; kept blocks re-compete in later rounds, and the
final round's keeps are read by extraction.

Drop a block ONLY on one of these grounds:
1. Different series: nothing in the block — title, column labels, row labels, or
   summary — carries the target series.
2. Covered: another KEPT print of the SAME table already contains all the information
   this block contributes to the question, so it adds nothing.

Rules for what you keep:
- Match the original question's exact wording — the precise series with every
  qualifier, total vs subtotal, unit, and time basis. The retrieval target is a
  paraphrase; the question governs.
- Confirm in the row labels that the requested dates exist as rows at the needed
  granularity (monthly rows vs an annual/fiscal-year roll-up).
- Recurring series: the same table recurs across consecutive issues with a shifting
  data window, sometimes with revisions. Keep the FEWEST prints whose windows together
  cover the requested period at the needed granularity — one print when one suffices,
  else a tiling of the same table reaching the period's first and last months.
- If the question pins a specific source ("as reported in the <Month Year> Bulletin", "as
  of <date>"), select the blocks that best match that source.
- When the same figure is restated across issues, prefer the most recent issue covering the
  required period unless the question asks for a specific version.
- When the question compares or combines several periods, keep all of them on the
  SAME time basis."""


def _render_candidates(
    groups: list[tuple[str, list[int]]],
    pool: list[SemPoolEntry],
) -> tuple[str, list[int]]:
    """Render title-grouped candidates as numbered lines ([1]..[n], call-local
    numbering). Returns `(text, flat_idxs)` with `flat_idxs[k]` the pool index of
    line `[k+1]`."""
    flat: list[int] = []
    parts: list[str] = []
    for title, members in groups:
        lines = []
        for i in members:
            flat.append(i)
            lines.append(_block_line(len(flat), pool[i]))
        parts.append(f'Title "{title}" ({len(members)} block(s)):\n' + "\n".join(lines))
    return "\n\n".join(parts), flat


def _selection_header(
    ctx: ExecutionContext, branch: RetrieveBranch, period_label: str | None
) -> list[str]:
    parts = [
        f'Research question: "{ctx.question}"',
        f"Retrieval target: {branch.key}",
    ]
    if period_label or branch.period:
        parts.append(f"Data period: {period_label or branch.period}")
    return parts


async def _select_call(
    ctx: ExecutionContext,
    pool: list[SemPoolEntry],
    groups: list[tuple[str, list[int]]],
    branch: RetrieveBranch,
    period_label: str | None,
    *,
    round_label: str,
    batch_no: int,
) -> list[int]:
    """One keep/drop selection call over one title-grouped batch. Returns the kept
    pool indices, in input order. Every call — narrowing round or final — is
    configured identically; the PromptedCall is rebuilt per call only because the
    parser is parity-checked against the batch's block count. Emits one structured
    `block_select_call` event (inputs + verdicts + latency/tokens) — the unit the
    trace viewer's tournament visualization renders."""
    listing, flat = _render_candidates(groups, pool)
    n = len(flat)
    parts = _selection_header(ctx, branch, period_label)
    parts.append(
        f"Candidate blocks ({n}), grouped by title — mark each true (KEEP) or false "
        f"(DROP):\n{listing}"
    )

    def _parse(text: str, _ctx: ExecutionContext) -> list[bool]:
        return PageIndexRetriever._parse_bool_list(text, _ctx, n=n)

    call: PromptedCall[list[bool]] = PromptedCall(
        name="block_select",
        system_prompt=_SELECT_PROMPT,
        parse=_parse,
        default_effort="low",
        output_instruction=(
            f"Output ONLY a JSON array of EXACTLY {n} booleans — one per block, in the "
            f"order given (true=keep, false=drop). Mark AT MOST {_KEEP} true — the {_KEEP} "
            "most relevant blocks; drop the rest. No prose, no markdown fences."
        ),
    )
    # Per-call stats summed over attempts (parse retries included) — the cost the
    # tournament viz attributes to this box.
    stats = {"latency_s": 0.0, "in_tok": 0, "out_tok": 0, "think_tok": 0}

    def _on_resp(r) -> None:
        stats["latency_s"] += r.latency_s
        stats["in_tok"] += r.input_tokens or 0
        stats["out_tok"] += r.output_tokens or 0
        stats["think_tok"] += r.thinking_tokens or 0

    verdicts = await call.call(
        ctx, "\n".join(parts), temperature=0.0, on_response=_on_resp
    )
    # Enforce the output bracket size: keep at most `_KEEP`, in input (relevance) order.
    kept = [i for i, keep in zip(flat, verdicts) if keep][:_KEEP]
    kept_set = set(kept)
    ctx.emit(
        f"block_select_call interval={period_label!r} round={round_label} "
        f"batch={batch_no} in={n} kept={len(kept)} "
        f"latency_s={round(stats['latency_s'], 3)}",
        data={
            "interval": period_label,
            "round": round_label,
            "batch": batch_no,
            "model": ctx.config.model_overrides.get(
                "block_select", ctx.config.llm_model
            ),
            "latency_s": round(stats["latency_s"], 3),
            "in_tok": stats["in_tok"],
            "out_tok": stats["out_tok"],
            "think_tok": stats["think_tok"],
            "blocks": [
                {
                    "id": _block_id(pool[i]),
                    "page": f"{pool[i].ref.page.month}:{pool[i].ref.page.page}",
                    "title": pool[i].title or f"(untitled {pool[i].kind})",
                    "kept": i in kept_set,
                }
                for i in flat
            ],
        },
    )
    return kept


async def _tournament(
    ctx: ExecutionContext,
    pool: list[SemPoolEntry],
    idxs: list[int],
    branch: RetrieveBranch,
    period_label: str | None,
) -> tuple[list[int], int]:
    """Run parallel keep/drop brackets (≤`_BATCH` blocks in, ≤`_KEEP` out each) until the
    field fits ONE bracket; that bracket's keeps ARE the selection. Every bracket is the
    same call — no distinct final round. Returns `(selected_idxs, rounds)`."""
    current = list(idxs)
    rounds = 0
    while True:
        rounds += 1
        batches = _title_batches(pool, current, _BATCH)
        results = await asyncio.gather(
            *(
                _select_call(
                    ctx, pool, g, branch, period_label,
                    round_label=str(rounds), batch_no=bi,
                )
                for bi, g in enumerate(batches, 1)
            )
        )
        survivors = [i for kept in results for i in kept]
        ctx.emit(
            f"block_select_round round={rounds} batches={len(batches)} "
            f"in={len(current)} survivors={len(survivors)}"
        )
        # ONE bracket → its keeps are the selection. Empty, or a round that stops shrinking
        # the set (the keep cap makes this rare) → stop with what stands.
        if len(batches) == 1 or not survivors or len(survivors) >= len(current):
            return survivors, rounds
        current = survivors


async def _select_for_branch(
    ctx: ExecutionContext,
    pool: list[SemPoolEntry],
    branch: RetrieveBranch,
) -> list[int]:
    """Selection PER PERIOD ENTRY (over the candidates whose data span overlaps that
    entry), winners unioned. A branch with no parseable period runs a single selection
    over all candidates. There is deliberately no global re-narrowing pass over the
    union — each period keeps its own selection."""
    entries = _branch_entries(branch)
    all_idxs = list(range(len(pool)))

    def _entry_part(e) -> list[int]:
        return [
            i
            for i in all_idxs
            if (iv := pool[i].interval) is None or _overlaps(iv, [(e.lo, e.hi)])
        ]

    parts: list[tuple[str | None, list[int]]]
    if entries:
        parts = [(e.label, p) for e in entries if (p := _entry_part(e))]
        if not parts:  # no candidate overlaps any entry (unlikely post year-filter)
            parts = [(entries[0].label, all_idxs)]
    else:
        parts = [(None, all_idxs)]

    results = await asyncio.gather(
        *(_tournament(ctx, pool, part, branch, label) for label, part in parts)
    )
    for (label, part), (ids, _) in zip(parts, results):
        ctx.emit(
            f"block_select_interval interval={label!r} "
            f"candidates={len(part)} selected={len(ids)}",
            data={"interval": label, "candidates": len(part), "selected": len(ids)},
        )
    rounds = max((r for _, r in results), default=0)
    chosen = list(dict.fromkeys(i for ids, _ in results for i in ids))
    ctx.emit(
        f"block_select key={branch.key!r} candidates={len(pool)} rounds={rounds} "
        f"selected_blocks={len(chosen)}"
    )
    return chosen


# ---------------------------------------------------------------------------
# The selection pipeline
# ---------------------------------------------------------------------------


async def run_select(
    ctx: ExecutionContext,
    branches: list[RetrieveBranch],
    retrievals: list[BranchRetrieval | StepFailed],
    branch_ids: list[int],
) -> list[list[SemPoolEntry] | StepFailed]:
    """Select the blocks an extraction step should read, for every retrieve branch of one
    question: build each branch's candidate pool, then run the per-period narrowing
    tournament over it (or, for `pre_selected` retrievals, keep the whole pool). Returns one
    slot per branch aligned to `branches` — the selected `SemPoolEntry`s, or the `StepFailed`
    to attribute to it. `block_extract.run_extract` reads the selected blocks."""
    results: list[list[SemPoolEntry] | StepFailed | None] = [None] * len(branches)

    # The candidate pool is built HERE, once, for every backend — the page store is the
    # canonical blocks→pool builder. `pre_selected` retrievals (golden / search-agent) skip
    # the tournament below and extract every block; live page-index blocks run selection.
    branch_pools: dict[int, list[SemPoolEntry]] = {}
    for pos, (branch, retrieval) in enumerate(zip(branches, retrievals)):
        if isinstance(retrieval, StepFailed):
            results[pos] = retrieval
            continue
        pool_entries = PageIndexRetriever.pool_for_blocks(
            list(retrieval.blocks), str(ctx.config.pdf_dir)
        )
        if not pool_entries:
            results[pos] = StepFailed(
                "retrieve",
                f"semantic filter kept no blocks for branch {branch.key!r}",
            )
            continue
        branch_pools[pos] = pool_entries

    # Golden / search-agent blocks are ALREADY final — bypass the tournament entirely and
    # keep every block. Golden's whole point is PERFECT retrieval straight into extract;
    # running selection there would contaminate the extract+compute ceiling and (under the
    # keep cap) could drop gold pages. Selection only applies on the live page-index path.
    if any(isinstance(r, BranchRetrieval) and r.pre_selected for r in retrievals):
        for pos in branch_pools:
            results[pos] = branch_pools[pos]
    elif branch_pools:
        async def _select(pos: int) -> list[SemPoolEntry]:
            pool = branch_pools[pos]

            async def _run() -> list[SemPoolEntry]:
                idxs = await _select_for_branch(ctx, pool, branches[pos])
                if not idxs:
                    # No fallback. The tournament rejecting every candidate means the target
                    # isn't in this branch's pages (e.g. an external series mis-routed to a
                    # retrieve branch). Fail the branch so compute reports NeedsMore and the
                    # replanner re-routes — never dump the full uncapped semfilter set into
                    # extract (that path produced a ~1000-page vision call → 1 GiB 400).
                    raise StepFailed(
                        "retrieve",
                        f"block_select selected no blocks for branch {branches[pos].key!r}",
                    )
                return [pool[i] for i in idxs]

            return await traced_step(ctx, "block_select", _run, branch_id=branch_ids[pos])

        positions = sorted(branch_pools)
        settled = await asyncio.gather(
            *(_select(pos) for pos in positions), return_exceptions=True
        )
        for pos, r in zip(positions, settled):
            if isinstance(r, StepFailed):
                results[pos] = r
            elif isinstance(r, BaseException):
                ctx.emit(
                    f"block_select_failed key={branches[pos].key!r} error={str(r)!r}"
                )
                results[pos] = StepFailed(
                    "retrieve", f"block_select error for {branches[pos].key!r}: {r}"
                )
            else:
                results[pos] = r

    return [
        r if r is not None else StepFailed("block_select", "no candidate pool")
        for r in results
    ]
