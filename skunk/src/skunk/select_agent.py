"""Selection pipeline — the experimental replacement for the linear
block_select → extract → review pipeline (config.selection_agent).

Three phases per question:

1. SELECT (ONE agent for the whole question). The agent gets a flat numbered
   list of retrieval GOALS (one per retrieve branch: concept, period, optional
   pinned issue) and ONE deduped candidate pool — the union of every branch's
   semantic-filter survivors. It is summaries-only: it browses the unique-title
   index and `list_blocks` (issue dates, data spans, column labels) and COMMITS
   a few candidate block numbers per goal. It cannot call extract — choosing
   which print/issue/table to read is its whole job, and it must do so from the
   catalog summaries alone.
2. EXTRACT (downstream). The committed blocks are organized FIRST: each unique
   block is read ONCE — text tier plus a vision confirmation round (OCR digit
   correction only), falling back to pure vision — and when several goals
   committed the same block, that one call's opening line lists ALL of them —
   one page read serves every goal. No per-goal entry copies are made: each block's
   entries are attributed to the first goal that committed it (`owner`), other
   committers are merely marked covered, so compute sees each datum exactly
   once. Branch identity is irrelevant at this point — goals are re-drafted on
   retry anyway.
3. CHECK (one LLM call). With every goal's extracted entries in schema view, a
   checker audits completeness (every goal covered by a conforming entry) and
   duplicates (the same datum under several entry ids). It may drop duplicate
   entries, and for goals whose coverage is missing or wrong it DRAFTS AN
   UPDATED TARGET; those targets — plus the blocks already selected — are
   appended onto the SAME selection agent's history (`call(resume=True)`), so
   the re-selection continues from everything the agent already saw instead of
   re-deriving it. One retry round; re-extraction skips already-read blocks.

The selection — which print/column/vintage answers the question — is the step
the linear pipeline fumbled (see notes.md clusters C1–C3); here it is an
explicit, observable agent decision, and the checker gives the question one
cross-goal round of error correction before compute.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass, field

from skunk.common import (
    AnnotatedValue,
    BlockRef,
    ExecutionContext,
    SemPoolEntry,
    _describe_entry,
    traced_step,
)
from skunk.errors import ParseError, StepFailed
from skunk.extract import (
    TextExtractor,
    VisionExtractor,
    VisualValidator,
    _blocks_to_pagerefs,
    _render_pages_b64,
)
from skunk.multi_turn_agent import Block, MultiTurnAgent, TextBlock, Tool
from skunk.page_index.query import _to_intervals
from skunk.plan import RetrieveBranch
from skunk.prompted_call import PromptedCall


# ---------------------------------------------------------------------------
# Pool helpers
# ---------------------------------------------------------------------------


def _block_id(e: SemPoolEntry) -> str:
    bi = e.ref.block_index
    return f"{e.ref.page.month}:{e.ref.page.page}#{'w' if bi is None else bi}"


def _ident(e: SemPoolEntry) -> str:
    return f"{e.ref.page.month} p.{e.ref.page.page}"


def _untitled(e: SemPoolEntry) -> str:
    return e.title or f"(untitled {e.kind})"


def _canon_title(raw: str) -> str:
    """Canonical grouping key for a title: whitespace collapsed, trailing punctuation
    stripped, case folded — so reprints whose captions differ only by OCR noise
    ("STATUTORY DEBT LIMITATION" vs "Statutory Debt Limitation.") form ONE group."""
    return re.sub(r"\s+", " ", raw).strip().rstrip(" .,;:").casefold()


def _group_titles(pool: list[SemPoolEntry]) -> list[tuple[str, list[int]]]:
    """Group the (globally numbered) pool by CANONICAL title into
    `(display_title, [block_number, ...])`, block numbers 1-based into `pool`, members
    in pool (issue) order. The display title is the group's most common cleaned variant.
    Titles ordered by descending reprint count then alphabetically, so the most-reprinted
    core tables lead."""
    groups: dict[str, list[int]] = {}
    variants: dict[str, Counter] = {}
    for num, e in enumerate(pool, start=1):
        raw = _untitled(e)
        key = _canon_title(raw) or raw
        groups.setdefault(key, []).append(num)
        cleaned = re.sub(r"\s+", " ", raw).strip().rstrip(" .,;:")
        variants.setdefault(key, Counter())[cleaned or raw] += 1
    out = [
        (variants[key].most_common(1)[0][0], nums) for key, nums in groups.items()
    ]
    return sorted(out, key=lambda kv: (-len(kv[1]), kv[0]))


def _span(pool: list[SemPoolEntry], nums: list[int]) -> str:
    """Union data-date span over a title group's blocks ("YYYY-MM..YYYY-MM"), or "n/a"."""
    ivs = [iv for n in nums if (iv := pool[n - 1].interval)]
    if not ivs:
        return "n/a"
    return f"{min(lo for lo, _ in ivs)}..{max(hi for _, hi in ivs)}"


def _title_index_line(ti: int, title: str, pool: list[SemPoolEntry], nums: list[int]) -> str:
    return f"  T{ti}  ×{len(nums)}  spans {_span(pool, nums)}  {title}"


def _fmt_rows(rows: tuple[str, ...]) -> str:
    """Full row headers, elided in the middle for very long tables (e.g. per-security
    quote tables) — the head shows the leading annual rows, the tail the data window."""
    if len(rows) > 40:
        rows = (*rows[:12], f"…(+{len(rows) - 36} rows)…", *rows[-24:])
    return ", ".join(rows)


def _block_line(pool: list[SemPoolEntry], num: int, show_cols: bool = True) -> str:
    e = pool[num - 1]
    dates = f"{e.interval[0]}..{e.interval[1]}" if e.interval else "none"
    line = f"  block {num} — {_ident(e)}  dates={dates}"
    if show_cols and e.cols:
        line += f"  [cols: {', '.join(e.cols)}]"
    rows = e.rows or e.rows_tail
    if rows:
        line += f"  [rows tail: {', '.join(rows[-3:])}]"
    return line


def _summary_entry(eid: str, e: AnnotatedValue) -> str:
    """Schema view of one extracted entry — IDENTICAL to what the compute agent sees
    (`common._describe_entry`): description, qualifiers, provenance, kind, shape, and
    axis labels, but NOT the interior cell values. This is what the checker reads; the
    full values reach compute via selection by id."""
    lines = _describe_entry(0, e)
    lines[0] = lines[0].replace("input_values[0]", eid, 1)
    return "\n".join(lines)


@dataclass
class _SelectState:
    """Question-level agent state: the deduped union candidate pool (block numbers are
    1-based into `pool`), its `titles` grouping (Tn → block numbers, shown at startup),
    and the goal numbers the current agent run must answer for (set per run; the
    validator checks the final answer covers exactly these)."""

    ctx: ExecutionContext
    pool: list[SemPoolEntry]
    titles: list[tuple[str, list[int]]] = field(default_factory=list)
    goal_nums: list[int] = field(default_factory=list)


class ListBlocksTool(Tool):
    name = "list_blocks"
    doc = """\
### list_blocks(titles: int | list[int], period: str = "")
Expand title group(s) into their member blocks, oldest first; batch every title of
interest into one call. `period` ("YYYY-MM" or "YYYY-MM..YYYY-MM") keeps only blocks
whose data span overlaps it. Returns, per title, a content summary, then each block's
global number (the numbers your final answer commits), issue, page, data span, column
labels, and tail row labels; at most 40 blocks per title.

```python
list_blocks([12, 31, 47], period="1962-04")
```"""

    def __init__(self, state: _SelectState) -> None:
        self._state = state

    def __call__(self, titles, period: str = "") -> str:
        s = self._state
        if isinstance(titles, (int, str)):
            titles = [titles]
        try:
            tis = list(dict.fromkeys(int(t) for t in titles))
        except (TypeError, ValueError):
            return f"invalid titles {titles!r} — pass Tn numbers (the integer n) from the candidate list"
        bad = [t for t in tis if not 1 <= t <= len(s.titles)]
        if bad:
            return f"unknown title(s) {bad!r} — the candidate list runs T1..T{len(s.titles)}"
        intervals = _to_intervals(str(period)) if period else None

        def overlaps(num: int) -> bool:
            iv = s.pool[num - 1].interval
            if not intervals or not iv:
                return True
            lo, hi = iv
            return any(not (hi < p_lo or lo > p_hi) for p_lo, p_hi in intervals)

        out: list[str] = []
        for ti in tis:
            name, nums = s.titles[ti - 1]
            kept = [n for n in nums if overlaps(n)]
            head = f'T{ti} "{name}": {len(nums)} block(s)'
            if intervals:
                head += f" (filtered to {period}: {len(kept)})"
            if not kept:
                out.append(head + " — none overlap that period")
                continue
            # Column labels are near-identical across a title's reprints; hoist them
            # to one group-level line when every kept block agrees, instead of
            # repeating a long [cols: ...] tail on all ~40 block lines.
            col_sets = {tuple(s.pool[n - 1].cols or ()) for n in kept}
            shared = len(col_sets) == 1 and next(iter(col_sets))
            if shared:
                head += f"\n  all blocks have cols: {', '.join(shared)}"
            summ = next((s.pool[n - 1].summary for n in kept if s.pool[n - 1].summary), None)
            if summ:
                head += f"\n  summary: {summ if len(summ) <= 240 else summ[:240] + '…'}"
            shown = kept
            if len(shown) > 40:
                head += f"\n  (showing the latest 40 of {len(shown)} — narrow with period=)"
                shown = shown[-40:]
            lines = [_block_line(s.pool, n, show_cols=not shared) for n in shown]
            out.append(head + ":\n" + "\n".join(lines))
        return "\n\n".join(out)


class InspectBlocksTool(Tool):
    name = "inspect_blocks"
    doc = """\
### inspect_blocks(blocks: int | list[int])
Full detail for up to 8 blocks (global numbers from list_blocks): issue, page, data
span, full summary, all column and row labels. The row labels show whether a period
exists as monthly rows or only as an annual/fiscal-year row — check before committing.

```python
inspect_blocks([645, 338])
```"""

    def __init__(self, state: _SelectState) -> None:
        self._state = state

    def __call__(self, blocks) -> str:
        s = self._state
        if isinstance(blocks, (int, str)):
            blocks = [blocks]
        try:
            nums = list(dict.fromkeys(int(b) for b in blocks))
        except (TypeError, ValueError):
            return f"invalid blocks {blocks!r} — pass global block numbers from list_blocks"
        bad = [n for n in nums if not 1 <= n <= len(s.pool)]
        if bad:
            return f"unknown block number(s) {bad!r} — the candidate list runs 1..{len(s.pool)}"
        if len(nums) > 8:
            return f"{len(nums)} blocks — inspect at most 8; shortlist with list_blocks first"
        out: list[str] = []
        for n in nums:
            e = s.pool[n - 1]
            dates = f"{e.interval[0]}..{e.interval[1]}" if e.interval else "none"
            part = f"block {n} — {_ident(e)}  dates={dates}"
            if e.title:
                part += f"\n  title: {e.title}"
            if e.summary:
                part += f"\n  summary: {e.summary}"
            if e.cols:
                part += f"\n  cols: {', '.join(e.cols)}"
            rows = e.rows or e.rows_tail
            if rows:
                part += f"\n  rows: {_fmt_rows(rows)}"
            else:
                part += "\n  rows: (none recorded — chart/prose block)"
            out.append(part)
        return "\n\n".join(out)


# ---------------------------------------------------------------------------
# The selection agent — one per question, summaries-only, commits per goal
# ---------------------------------------------------------------------------

# Hard ceiling on committed blocks per goal — the agent is asked for "a few";
# the validator rejects sweeps so extraction stays bounded.
_MAX_COMMIT = 8


def _goal_key(x) -> int | None:
    """Normalize a goals-dict key ("G2" / "g2" / "2" / 2) to its goal number."""
    s = str(x).strip().lstrip("Gg")
    try:
        return int(s)
    except ValueError:
        return None


class SelectAgent(MultiTurnAgent):
    name = "select_agent"
    default_effort = "medium"

    briefing = """\
You select which content blocks an extraction step should read to fulfill a
set of retrieval goals for one research question, given
metadata for each page (titles, issue dates, data spans, summaries, column/row
labels). The user message gives the question, the numbered goals (retrieval target, period),
and the pool as a list of unique table/chart titles (T1, T2, ...); the same table is reprinted across
consecutive issues, sometimes with revisions.

Pay attention to the original question's wording: match the exact series with every qualifier, total vs subtotal, unit,
and time basis. The goals are paraphrases and should be take less literally. When searching, be efficient with steps 
and emit batched calls. Commit when you are convinced that the goal is met. Report [] when you are certain that no
title carries the requested data. Output only a handful of blocks per goal. Your 8 most recent observations are
visible; never reference block numbers you can no longer see.

Notes:
- A label wrapping the concept in extra words ("... and related activities",
  "..., including ...") is a broader aggregate; prefer the exact asked scope.
- When the question describes the source of the data (published in, reported by, etc.), emit issues that fit the
  description and discard the rest.
- A block's date span unions all rows, annual rows included; confirm with
  inspect_blocks that the needed dates exist as rows at the needed granularity and your selections cover the entire
  requested period.
- When multiple reprints exist, you should prefer the latest that carries all the requested data, but you must inspect
  the row labels to confirm that the needed dates exist as rows at the needed granularity and work your backwards if the
  later issue does not carry the data.
- Favor whole-table coverage: a single block whose rows span the entire requested period beats a patchwork of partial
  windows. Assemble from multiple blocks only when no single print carries the whole period, and then tile reprints of
  the same table — never a mix of different tables.
"""

    final_answer_doc = """\
  {"goals": {"G1": [645, 338], "G2": []}, "why": "<one short line per goal: which title/issue and why>"}
- `goals` has one entry PER GOAL, keyed by its number, listing the GLOBAL block
  numbers (from list_blocks) to read for that goal. Goals may share blocks.
- An empty list means the pool genuinely lacks that goal's data; `why` must then
  say exactly what is missing."""

    # The 8 most recent tool observations stay visible (older ones collapse to a
    # placeholder): enough to shortlist via list_blocks, verify via inspect_blocks,
    # and still see both when committing, without unbounded context growth.
    visible_observations = 8

    # Same contract as the base template, except a python step may issue SEVERAL
    # tool calls at once (print each) — the step budget is tiny, so the agent is
    # told to batch every expansion it wants into one step.
    _SYSTEM_TEMPLATE = """\
{{ briefing }}

## Tools (already imported)

{{ tools_doc }}

You have ≤{{ max_steps }} steps. On each step, output exactly ONE fenced block:
  - a ```python``` block containing one or more tool calls — when issuing several,
    wrap each in print(...); they all execute this step and their output appears
    as your next observation; or
  - a ```json``` block containing your final answer — emit this once, when you are
    ready to finish. It is parsed as data (not executed), so write plain JSON
    literals (no Python, no variables, no trailing commas).

Requirements for the final answer:
{{ final_answer_doc }}"""

    def __init__(self, state: _SelectState, *, max_steps: int) -> None:
        self._state = state
        # Bound each selection-step LLM call the same way SearchAgent does: cap
        # output and impose a hard per-request wall-clock timeout. Unbounded, these
        # turns ran to 400+s on large inputs with no output, and one parse-retry
        # blew out to ~63k thinking tokens. See SkunkConfig for the thinking caveat.
        cfg = state.ctx.config
        self.max_output_tokens = cfg.select_agent_max_output_tokens
        self.request_timeout_s = cfg.select_agent_request_timeout_s
        super().__init__(
            [ListBlocksTool(state), InspectBlocksTool(state)], max_steps=max_steps
        )

    def _blocks_from_output(self, out) -> list[Block]:
        """A step whose code produced nothing was comment-only (no tool call) — the
        observed failure mode is the model re-emitting 'wait, let me check…' thoughts
        forever. Replace the silent '[no output]' with a pointed nudge."""
        blocks = super()._blocks_from_output(out)
        if len(blocks) == 1 and isinstance(blocks[0], TextBlock) and blocks[0].text == "[no output]":
            return [
                TextBlock(
                    "[no output] — your code called no tool. Either call "
                    "list_blocks(...) or emit your final ```json``` answer NOW."
                )
            ]
        return blocks

    def validate_final_answer(self, payload: object, observations: list[str]):
        if not isinstance(payload, dict) or "goals" not in payload:
            return 'final answer must be {"goals": {"G1": [<block numbers>], ...}, "why": "..."}'
        goals = payload["goals"]
        if not isinstance(goals, dict):
            return "'goals' must be an object mapping goal numbers to block-number lists"
        provided: dict[int, list[int]] = {}
        for k, v in goals.items():
            num = _goal_key(k)
            if num is None:
                return f"unrecognized goal key {k!r} — use the goal numbers (e.g. \"G1\")"
            if not isinstance(v, list):
                return f"goal {k!r} must map to a LIST of global block numbers"
            try:
                provided[num] = list(dict.fromkeys(int(x) for x in v))
            except (TypeError, ValueError):
                return f"goal {k!r} contains a non-integer — use global block numbers from list_blocks"
        expected = set(self._state.goal_nums)
        missing = sorted(expected - set(provided))
        if missing:
            return f"missing goal(s) {[f'G{n}' for n in missing]!r} — every goal needs an entry (possibly [])"
        unknown = sorted(set(provided) - expected)
        if unknown:
            return f"unknown goal(s) {[f'G{n}' for n in unknown]!r} — the goals are {[f'G{n}' for n in sorted(expected)]!r}"
        for num, nums in provided.items():
            bad = [n for n in nums if not 1 <= n <= len(self._state.pool)]
            if bad:
                return f"unknown block number(s) {bad!r} in G{num} — the candidate list runs 1..{len(self._state.pool)}"
            if len(nums) > _MAX_COMMIT:
                return (
                    f"G{num} commits {len(nums)} blocks — that is a sweep, not a selection; commit at most "
                    f"{_MAX_COMMIT}: the latest covering print per period plus genuine alternatives"
                )
        if any(not nums for nums in provided.values()) and not str(payload.get("why", "")).strip():
            return "an empty goal requires a 'why' explaining what is missing"
        return None


# ---------------------------------------------------------------------------
# Downstream extraction — text tier + vision confirmation round (OCR digit
# correction only), pure vision as fallback; one read per block serving ALL its goals
# ---------------------------------------------------------------------------

_TEXT = TextExtractor()
_CONFIRM = VisualValidator()
_VISION = VisionExtractor()


@dataclass
class _Goal:
    """One retrieve branch's goal through the pipeline. `num` is the stable 1-based
    goal number shown to the selector and the checker; `pos` indexes the pipeline's
    input branch list. `entries` holds only the entries this goal OWNS (first
    committer of their block); `covered` is true when any of its committed blocks
    yielded entries, wherever they are attributed."""

    num: int
    pos: int
    branch: RetrieveBranch
    bid: int
    blocks: list[int] = field(default_factory=list)  # committed pool numbers (1-based)
    entries: list[AnnotatedValue] = field(default_factory=list)
    covered: bool = False


def _goal_line(g: _Goal) -> str:
    line = f"  G{g.num}: {g.branch.key}"
    if g.branch.period:
        line += f"  [period: {g.branch.period}]"
    if g.branch.as_of:
        line += f"  [pinned source issue: {g.branch.as_of}]"
    return line


def _synth_branch(goals: list[_Goal]) -> RetrieveBranch:
    """One stamp-bearing branch for a multi-goal page read. Branch identity is
    irrelevant at extraction (goals are re-drafted on retry anyway), so the
    call-level provenance fields carry the union of the requesting goals."""
    keys = list(dict.fromkeys(g.branch.key for g in goals))
    periods = list(dict.fromkeys(p for g in goals if (p := g.branch.period)))
    as_ofs = {a for g in goals if isinstance(a := g.branch.as_of, str)}
    return RetrieveBranch(
        key="; ".join(keys),
        period=", ".join(periods) or None,
        as_of=next(iter(as_ofs)) if len(as_ofs) == 1 else None,
        visual_only=any(g.branch.visual_only for g in goals),
    )


async def _extract_block(
    ctx: ExecutionContext, state: _SelectState, num: int, goals: list[_Goal]
) -> list[AnnotatedValue]:
    """One block's read serving EVERY goal that committed it: the call's opening line
    lists all the goals, so a single page read extracts for each. Same tier order as
    `ExtractOp`: text tier first, then the vision confirmation round (OCR digit
    correction only) over its output; pure vision as the fallback — for visual_only
    goals, the `extract_vision_only` override, or a text pass that found nothing.
    Output and timeout caps come from the extract tier itself."""
    ref = state.pool[num - 1].ref
    branch = goals[0].branch if len(goals) == 1 else _synth_branch(goals)
    looking = None
    if len(goals) > 1:
        lines = []
        for g in goals:
            line = f"- {g.branch.key}"
            if g.branch.period:
                line += f" (for the period {g.branch.period})"
            lines.append(line)
        looking = "You are looking for ALL of the following:\n" + "\n".join(lines)
    if not branch.visual_only and not ctx.config.extract_vision_only:
        entries = await _TEXT.run(
            ctx.question, branch, [ref], ctx, looking_for=looking
        )
        if entries:
            return await _CONFIRM.run(ctx.question, branch, [ref], entries, ctx)
    images, rendered_refs = _render_pages_b64(_blocks_to_pagerefs([ref]), ctx)
    if not images:
        return []
    return await _VISION.run(
        ctx.question, branch, images, rendered_refs, ctx, looking_for=looking
    )


# ---------------------------------------------------------------------------
# Cross-goal checker — one completeness/duplicate audit, drafts updated targets
# ---------------------------------------------------------------------------

_CHECK_SYSTEM = """\
You audit the data gathered for one research question before computation. For
each retrieval goal you see its target, the committed blocks, and the entries
extracted from them as schema views (no cell values). An entry listed under
several goals was read once for all of them; that is not duplication.

Pay attention to the original question's wording: an entry covers a goal only
when it matches the exact series with every qualifier, total vs subtotal,
unit, and time basis. The goals are paraphrases and should be taken less
literally. Check that the entries cover the entire requested period — each
month/quarter/year of a goal's period must appear among the entries' index
labels; a missing unit is incomplete coverage even when the series matches.
When the question describes the source of the data (published in, reported
by, etc.), entries from issues that do not fit the description are wrong
coverage. Check duplicates — the same series, scope, and period under several
entry ids; keep the source the question specifies, else the latest print that
carries all the requested data. Different periods of one series are
complements, not duplicates.

Output one JSON object, nothing else:
{"complete": true|false,
 "drop": ["E3", ...],
 "retry": [{"goal": <goal number>, "target": "<revised one-line retrieval target>"}]}
- "drop": only duplicates of a kept entry; never shrink coverage; when unsure,
  keep. An entry that fails the question's wording is a coverage problem —
  "retry" (with complete=false), not "drop".
- "retry": goals whose coverage is missing or wrong; "target" rewrites the
  goal's retrieval target, naming exactly what is missing (the series and the
  specific months). Retry sparingly, only when fixable from the pool, and
  never for a goal already reported absent.
- Nothing to fix: {"complete": true, "drop": [], "retry": []}"""


def _parse_check(text: str, _: ExecutionContext) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m is None:
        raise ParseError(raw=text, detail="no JSON object found — emit exactly one JSON object")
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ParseError(raw=text, detail=f"malformed JSON — {e}") from e
    if not isinstance(out, dict):
        raise ParseError(raw=text, detail="top-level JSON must be an object")
    return out


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def _hydrate_rows(
    pool: list[SemPoolEntry], ctx: ExecutionContext
) -> list[SemPoolEntry]:
    """Backfill full row headers from the page catalog for entries built before
    `rows` existed (e.g. replayed retrieval caches)."""
    from dataclasses import replace

    from skunk.page_index.store import get_page_store

    store = get_page_store(str(ctx.config.pdf_dir))
    out: list[SemPoolEntry] = []
    for e in pool:
        if not e.rows and e.ref.block_index is not None:
            row = store.catalog_row(e.ref.page)
            if row is not None and 0 <= e.ref.block_index < len(row.content_blocks):
                e = replace(
                    e, rows=tuple(row.content_blocks[e.ref.block_index].row_headers)
                )
        out.append(e)
    return out


async def run_select_pipeline(
    ctx: ExecutionContext,
    branches: list[RetrieveBranch],
    docs: list[list[BlockRef] | StepFailed],
    pools: list[list[SemPoolEntry]],
    branch_ids: list[int],
) -> list[list[AnnotatedValue] | StepFailed]:
    """Select → extract → check for all retrieve branches of one question, driven by
    ONE selection agent over the union candidate pool. Returns one result per branch
    (its entries, or the `StepFailed` to attribute to it)."""
    results: list[list[AnnotatedValue] | StepFailed | None] = [None] * len(branches)

    # Union candidate pool across the branches whose retrieve succeeded, deduped by
    # block identity; each surviving branch becomes a numbered goal.
    union: dict[str, SemPoolEntry] = {}
    goals: list[_Goal] = []
    for pos, (branch, doc, pool, bid) in enumerate(
        zip(branches, docs, pools, branch_ids)
    ):
        if isinstance(doc, StepFailed):
            results[pos] = doc
            continue
        pool_entries = pool
        if not pool_entries:
            from skunk.page_index.query import PageIndexRetriever

            pool_entries = PageIndexRetriever.pool_for_blocks(
                doc, str(ctx.config.pdf_dir)
            )
        for e in pool_entries:
            union.setdefault(_block_id(e), e)
        goals.append(_Goal(num=len(goals) + 1, pos=pos, branch=branch, bid=bid))
    if not goals:
        return [r if r is not None else StepFailed("select_agent", "no goals") for r in results]
    if not union:
        e = StepFailed("select_agent", "selection agent has no candidate pool")
        for g in goals:
            results[g.pos] = e
        return [r if r is not None else e for r in results]

    pool_list = sorted(
        union.values(),
        key=lambda e: (
            e.ref.page.month or "",
            e.ref.page.page or 0,
            e.ref.block_index if e.ref.block_index is not None else -1,
        ),
    )
    pool_list = _hydrate_rows(pool_list, ctx)
    state = _SelectState(ctx=ctx, pool=pool_list, titles=_group_titles(pool_list))
    # Step budget scales with the goal count: the configured base (8) covers up to
    # 3 goals, +2 steps per goal beyond the 3rd (4 goals → 10, 5 → 12).
    max_steps = ctx.config.select_agent_max_steps + 2 * max(0, len(goals) - 3)
    agent = SelectAgent(state, max_steps=max_steps)

    # The question's extraction record: each unique block read once, its entries
    # owned by the FIRST goal that committed it (no per-goal copies — compute must
    # see each datum exactly once).
    extracted: dict[int, list[AnnotatedValue]] = {}
    owner: dict[int, int] = {}
    whys: dict[int, str] = {}

    async def _run_agent(
        targets: list[_Goal], message: str, resume: bool
    ) -> None:
        state.goal_nums = [g.num for g in targets]
        payload = await agent.call(ctx, message, resume=resume)
        commits = {
            num: list(dict.fromkeys(int(x) for x in v))
            for k, v in payload["goals"].items()
            if (num := _goal_key(k)) is not None
        }
        why = str(payload.get("why", ""))
        for g in targets:
            g.blocks = commits.get(g.num, [])
            whys[g.num] = why
        shown = {f"G{n}": v for n, v in sorted(commits.items())}
        ctx.emit(f"select_commit goals={shown!r} why={why[:200]!r}")

    async def _extract_phase(targets: list[_Goal]) -> None:
        # Organize first: unique not-yet-read blocks, each mapped to EVERY goal
        # (across the whole question) that committed it — one read serves them all.
        want: dict[int, list[_Goal]] = {}
        for g in goals:
            for n in g.blocks:
                if n in extracted:
                    continue
                want.setdefault(n, []).append(g)
        n_req = sum(len(g.blocks) for g in targets)
        ctx.emit(
            f"select_extract n_requested={n_req} n_reads={len(want)} "
            f"n_already_read={n_req - len(want)}"
        )
        reads = await asyncio.gather(
            *(_extract_block(ctx, state, n, gs) for n, gs in want.items()),
            return_exceptions=True,
        )
        for (n, gs), res in zip(want.items(), reads):
            if isinstance(res, BaseException):
                ctx.emit(f"select_extract_failed block={n} error={str(res)!r}")
                extracted[n] = []
            else:
                extracted[n] = res
            owner[n] = gs[0].num
        # Re-attribute for every goal: own only the blocks first committed by you;
        # covered when ANY committed block yielded entries, wherever attributed.
        for g in goals:
            g.entries = [
                e
                for n in g.blocks
                if owner.get(n) == g.num
                for e in extracted.get(n, [])
            ]
            g.covered = any(extracted.get(n) for n in g.blocks)

    # Phase 1+2: one selection run over all goals, then one organized extraction sweep.
    parts = [
        f'Research question: "{ctx.question}"',
        "Retrieval goals (commit blocks for EVERY goal):\n"
        + "\n".join(_goal_line(g) for g in goals),
    ]
    months = sorted({e.ref.page.month for e in pool_list if e.ref.page.month})
    span = f"{months[0]}..{months[-1]}" if months else "unknown"
    listing = "\n".join(
        _title_index_line(ti, name, pool_list, nums)
        for ti, (name, nums) in enumerate(state.titles, start=1)
    )
    parts.append(
        f"Candidate pool: {len(pool_list)} content blocks from issues {span}, under "
        f"{len(state.titles)} unique titles (×n = issues reprinting it). list_blocks([<Tn>, ...]) "
        f"expands titles into their blocks:\n" + listing
    )
    try:
        await traced_step(
            ctx, "select_agent", lambda: _run_agent(goals, "\n".join(parts), False)
        )
    except StepFailed as e:
        for g in goals:
            results[g.pos] = e
        return [r if r is not None else e for r in results]
    await traced_step(ctx, "extract", lambda: _extract_phase(goals))

    # Phase 3: one cross-goal completeness/duplicate audit. Checker failure is never
    # fatal — the entries stand as extracted.
    try:
        await _check_and_repair(ctx, goals, extracted, owner, _run_agent, _extract_phase)
    except Exception as e:  # noqa: BLE001 — audit must never break the question
        ctx.emit(f"select_check_failed error={e!r}")

    for g in goals:
        if g.covered or g.entries:
            results[g.pos] = g.entries
        else:
            results[g.pos] = StepFailed(
                "extract",
                f"selection agent found no data for {g.branch.key!r}: {whys.get(g.num, '')}",
            )
        ctx.emit(
            f"select_pipeline_branch branch_id={g.bid} goal=G{g.num} "
            f"n_blocks={len(g.blocks)} n_entries={len(g.entries)} covered={g.covered}",
            data={"branch_id": g.bid},
        )
    return [
        r if r is not None else StepFailed("select_agent", "branch produced no result")
        for r in results
    ]


async def _check_and_repair(
    ctx: ExecutionContext,
    goals: list[_Goal],
    extracted: dict[int, list[AnnotatedValue]],
    owner: dict[int, int],
    run_agent,
    extract_phase,
) -> None:
    """One checker call over every goal's entries. Apply its drops; for flagged goals,
    append the checker's UPDATED TARGETS (plus the blocks already selected) onto the
    selection agent's existing history and resume it for one re-selection round, then
    re-extract (already-read blocks are skipped by the extraction record)."""
    eid_of: dict[int, str] = {}  # id(entry) → eid
    flat: list[AnnotatedValue] = []
    lines = [f'Research question: "{ctx.question}"', ""]
    for g in goals:
        lines.append(f"Goal G{g.num}: target={g.branch.key!r}"
                     + (f" period={g.branch.period}" if g.branch.period else "")
                     + (f" as_of={g.branch.as_of}" if g.branch.as_of else ""))
        if not g.blocks:
            lines.append("  selector committed NO blocks (reported absent from the pool)")
        elif not g.covered:
            lines.append("  extracted entries: (none)")
        for n in g.blocks:
            for e in extracted.get(n, []):
                if id(e) not in eid_of:
                    eid_of[id(e)] = f"E{len(flat)}"
                    flat.append(e)
                    lines.append(_summary_entry(eid_of[id(e)], e))
                else:
                    lines.append(f"  {eid_of[id(e)]} (also covers this goal)")
        lines.append("")
    prompt: PromptedCall[dict] = PromptedCall(
        name="select_check",
        system_prompt=_CHECK_SYSTEM,
        default_effort="medium",
        parse=_parse_check,
    )
    cfg = ctx.config
    verdict = await prompt.call(
        ctx,
        "\n".join(lines),
        max_output_tokens=cfg.select_agent_max_output_tokens,
        timeout_s=cfg.select_agent_request_timeout_s,
    )
    drops = {str(x) for x in verdict.get("drop") or []}
    retries = verdict.get("retry") or []
    ctx.emit(
        f"select_check complete={verdict.get('complete')} n_drop={len(drops)} "
        f"n_retry={len(retries)}",
        data={"verdict": verdict},
    )

    by_num = {g.num: g for g in goals}
    retry_goals: list[_Goal] = []
    targets: dict[int, str] = {}
    for r in retries:
        if not isinstance(r, dict):
            continue
        num = _goal_key(r.get("goal"))
        if num is None or num not in by_num or num in targets:
            continue
        # MECHANICAL gate, not just prompt guidance: a goal the selector committed
        # NOTHING for was established absent from the pool — re-selecting cannot fix
        # it, and it must flow out as missing data (StepFailed → replanner → e.g.
        # lookup_external), never spin here. Only non-empty goals are retryable.
        if not by_num[num].blocks:
            ctx.emit(f"select_check_retry_skipped goal=G{num} reason=empty_commit")
            continue
        retry_goals.append(by_num[num])
        targets[num] = str(r.get("target", "")).strip()

    # Drops: remove the flagged entries from the extraction record (so attribution
    # rebuilds reflect them). Entries owned by a retried goal are left alone — that
    # goal's selection is being replaced wholesale.
    retried = {g.num for g in retry_goals}
    if drops:
        drop_ids = {ide for ide, eid in eid_of.items() if eid in drops}
        n_dropped = 0
        for n, entries in extracted.items():
            if owner.get(n) in retried:
                continue
            kept = [e for e in entries if id(e) not in drop_ids]
            n_dropped += len(entries) - len(kept)
            extracted[n] = kept
        if n_dropped:
            ctx.emit(f"select_check_drop n_dropped={n_dropped}")
        for g in goals:
            g.entries = [
                e for n in g.blocks if owner.get(n) == g.num for e in extracted.get(n, [])
            ]
            g.covered = any(extracted.get(n) for n in g.blocks)

    if not retry_goals:
        return
    empty_blocks = sorted(n for n, ents in extracted.items() if not ents)
    feedback = [
        "Checker feedback: the goals below are not yet correctly covered; "
        "re-select for their revised targets:",
        *(f"  G{n}: {t}" for n, t in sorted(targets.items()) if t),
        "Blocks already selected and extracted (their values are kept): "
        + ", ".join(
            f"G{g.num}=[{', '.join(map(str, g.blocks))}]" for g in goals if g.blocks
        ),
        *(
            [
                f"Blocks already read that contained no relevant values: {empty_blocks} "
                "— commit different blocks."
            ]
            if empty_blocks
            else []
        ),
        "Before committing, inspect row labels to confirm the missing dates exist "
        "as rows at the needed granularity; when the reprint you tried does not "
        "carry them, work your way backwards through earlier issues (or the issues "
        "published just after the missing dates) to one that does.",
        "Emit a new final answer covering only the goals above (same JSON format); "
        "the other goals' selections stand. You may call list_blocks first.",
    ]
    try:
        await traced_step(
            ctx,
            "select_agent",
            lambda: run_agent(retry_goals, "\n".join(feedback), True),
        )
    except StepFailed:
        return  # failed retry keeps the goals' original commits/entries
    await traced_step(ctx, "extract", lambda: extract_phase(retry_goals))
