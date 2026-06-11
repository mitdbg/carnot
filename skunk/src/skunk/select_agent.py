"""Iterative selection agent — the experimental replacement for the linear
block_select → extract → review pipeline (config.selection_agent).

One `SelectAgent` (a `MultiTurnAgent`) runs per retrieve branch. Its opening
message carries the branch's complete semantic-filter SURVIVOR pool as a
numbered list of block titles — nothing is held back or truncated. From there
it works iteratively: pull the catalog summaries of title matches (`search`),
extract typed values from chosen blocks (`extract_values` — one extract call
per block, fanned out in parallel), and finally SELECT which extracted entries
flow to compute. Selection — which print/column/vintage answers the question —
is the step the linear pipeline fumbled (see notes.md clusters C1–C3); here it
is an explicit, observable agent decision.

Tool calls run synchronously inside the agent loop (the `MultiTurnAgent`
contract), so the extraction tool drives the LLM through `LLMClient.call` — the
SYNC path — rather than the async pipeline in `extract.py`. The prompts, parse
hooks, and provenance stamping are imported from `extract.py` so the two paths
read pages identically; only the transport differs. Blocking the question's
event loop during a tool call stalls sibling branches of the same question
only. `extract_values` fans its per-block calls out over a thread pool: each
worker runs under its own `contextvars` copy (preserving the ctx step frame for
emits), the JSONL trace sink is lock-guarded, and the per-question log file
receives whole lines per emit, so concurrent emits stay well-formed.
"""

from __future__ import annotations

import contextvars
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from skunk.common import (
    AnnotatedValue,
    BlockRef,
    ExecutionContext,
    SemPoolEntry,
    _describe_entry,
)
from skunk.errors import ParseError, StepFailed
from skunk.extract import (
    _EXTRACT_OUTPUT_INSTRUCTION,
    TextExtractor,
    VisionExtractor,
    VisualValidator,
    _carry_provenance,
    _entry_semantic_dict,
    _make_confirm_parse,
    _make_text_parse,
    _parse_extract_response,
    _render_pages_b64,
    _stamp_provenance,
)
from skunk.multi_turn_agent import Block, MultiTurnAgent, TextBlock, Tool
from skunk.page_index.query import _to_intervals
from skunk.page_index.store import get_page_store
from skunk.plan import RetrieveBranch
from skunk.prompted_call import PromptedCall


# ---------------------------------------------------------------------------
# Sync prompted call — a minimal synchronous twin of `PromptedCall.call` for use
# inside agent tool bodies (which run synchronously on the question's loop
# thread, where the async client cannot be awaited).
# ---------------------------------------------------------------------------


def _sync_prompted(
    ctx: ExecutionContext,
    *,
    name: str,
    system_prompt: str,
    user: str,
    parse,
    images=None,
    default_effort: str = "medium",
    max_parse_retries: int = 1,
):
    """One sync LLM call with the same prompt assembly (overrides tail), model/effort
    resolution, event emits, and ParseError→retry behavior as `PromptedCall.call`."""
    pc: PromptedCall = PromptedCall(
        name=name,
        system_prompt=system_prompt,
        output_instruction=_EXTRACT_OUTPUT_INSTRUCTION,
    )
    system = pc._assemble_system_prompt(ctx)
    effort = ctx.config.effort_overrides.get(name, default_effort)
    model = pc._resolve_model(ctx)
    client = ctx.llm_client
    assert client is not None  # built in ExecutionContext.__post_init__
    ctx.emit(
        f"prompt_system call_site={name} chars={len(system)}",
        kind="system",
        data={"text": system},
    )
    retry: ParseError | None = None
    attempt = 0
    while True:
        user_msg = pc._compose_user(user, retry)
        if attempt == 0:
            ctx.emit(
                f"prompt_user call_site={name} chars={len(user_msg)}",
                kind="user",
                data={"text": user_msg},
            )
        temp = 0.0 if attempt == 0 else min(1.0, 0.4 + 0.2 * (attempt - 1))
        resp = client.call(
            system,
            user_msg,
            images=images,
            temperature=temp,
            effort=effort,  # type: ignore[arg-type]
            ctx=ctx,
            call_site=name,
            model=model,
        )
        ctx.emit(
            f"assistant call_site={name} chars={len(resp.text)}",
            kind="assistant",
            data={"text": resp.text},
        )
        try:
            return parse(resp.text, ctx)
        except ParseError as e:
            if attempt >= max_parse_retries:
                raise
            ctx.emit(
                f"parse_retry call_site={name} attempt={attempt + 1} error={e.detail!r}"
            )
            retry = e
            attempt += 1


# ---------------------------------------------------------------------------
# Sync extraction — text tier + vision confirm + vision fallback, mirroring
# `ExtractOp._extract_once` over the sync call path.
# ---------------------------------------------------------------------------

_text = TextExtractor()


def _extract_group_sync(
    ctx: ExecutionContext,
    branch: RetrieveBranch,
    anchor,
    member_refs,
    block_idxs,
) -> list[AnnotatedValue]:
    """Sync mirror of `TextExtractor._extract_block_group` + `_extract_content`."""
    pages = _text._fetch_page_texts(member_refs, ctx)
    if not pages:
        return []
    content = "\n\n".join(text for _, text in pages)
    prov_refs = [r for r, _ in pages]
    row = get_page_store(str(ctx.config.pdf_dir)).catalog_row(anchor)
    specific = [
        bi
        for bi in block_idxs
        if bi is not None and row is not None and 0 <= bi < len(row.content_blocks)
    ]
    if specific and len(specific) == len(block_idxs):
        metadata = "\n".join(
            _text._block_meta_line(anchor.page, row.content_blocks[bi])  # type: ignore[union-attr]
            for bi in specific
        )
    else:
        metadata = _text._page_metadata(prov_refs, ctx)
    user_msg = "\n\n".join(
        [
            f"You are looking for {branch.key}{f' for the period {branch.period}' if branch.period else ''}.",
            f'For full context, this lookup serves to help answer the question: "{ctx.question}"',
            *(
                [
                    f"Page metadata (context to interpret the layout — not a source of values):\n{metadata}"
                ]
                if metadata
                else []
            ),
            content,
        ]
    )
    try:
        parsed = _sync_prompted(
            ctx,
            name="extract.text",
            system_prompt=TextExtractor._SYSTEM,
            user=user_msg,
            parse=_make_text_parse(content),
        )
    except ParseError as e:
        ctx.emit(f"extract_parse_failed tier=parsed_json error={e.detail!r}")
        return []
    ctx.emit(f"extracted tier=parsed_json n_entries={len(parsed)}")
    return _stamp_provenance(parsed, prov_refs, branch)


def _confirm_sync(
    ctx: ExecutionContext,
    branch: RetrieveBranch,
    blocks: list[BlockRef],
    entries: list[AnnotatedValue],
) -> list[AnnotatedValue]:
    """Sync mirror of `VisualValidator.run` — per page-group digit re-read against the
    rendered images; any failure passes that group's entries through unchanged."""
    groups = VisualValidator._group_by_source(entries, blocks)
    out = list(entries)
    for refs, members in groups:
        originals = [e for _, e in members]
        images, rendered_refs = _render_pages_b64(refs, ctx)
        if not images:
            continue
        period = f" for the period {branch.period}" if branch.period else ""
        image_lines = [
            f"Image {i + 1}: PDF page {ref.page} of the {ref.month} Treasury Bulletin"
            for i, ref in enumerate(rendered_refs)
        ]
        payload = json.dumps(
            [_entry_semantic_dict(e) for e in originals], ensure_ascii=False
        )
        user_msg = "\n\n".join(
            [
                f"You are confirming values for {branch.key}{period}.",
                f'For full context, this lookup serves to help answer the question: "{ctx.question}"',
                "Images attached, in order:\n" + "\n".join(image_lines),
                "Transcribed values to confirm:\n" + payload,
            ]
        )
        try:
            corrected = _sync_prompted(
                ctx,
                name="extract.confirm",
                system_prompt=VisualValidator._SYSTEM,
                user=user_msg,
                parse=_make_confirm_parse(originals),
                images=images,
            )
        except ParseError as e:
            ctx.emit(f"confirm_kept_original reason=parse_failed error={e.detail!r}")
            continue
        if not corrected:
            continue
        corrected = _carry_provenance(corrected, originals)
        for (idx, _), new in zip(members, corrected):
            out[idx] = new
    return out


def _vision_sync(
    ctx: ExecutionContext, branch: RetrieveBranch, blocks: list[BlockRef]
) -> list[AnnotatedValue]:
    """Sync mirror of `VisionExtractor.run` over the blocks' rendered pages."""
    from skunk.extract import _blocks_to_pagerefs

    images, rendered_refs = _render_pages_b64(_blocks_to_pagerefs(blocks), ctx)
    if not images:
        return []
    period = f" for the period {branch.period}" if branch.period else ""
    image_lines = [
        f"Image {i + 1}: PDF page {ref.page} of the {ref.month} Treasury Bulletin"
        for i, ref in enumerate(rendered_refs)
    ]
    user_msg = "\n\n".join(
        [
            f"You are looking for {branch.key}{period}.",
            f'For full context, this lookup serves to help answer the question: "{ctx.question}"',
            "Images attached, in order:\n" + "\n".join(image_lines),
        ]
    )
    try:
        entries = _sync_prompted(
            ctx,
            name="extract.vision",
            system_prompt=VisionExtractor._prompt._system_prompt,
            user=user_msg,
            parse=_parse_extract_response,
            images=images,
        )
    except ParseError as e:
        ctx.emit(f"extract_parse_failed tier=vision error={e.detail!r}")
        return []
    return _stamp_provenance(entries, rendered_refs, branch)


def _extract_blocks_sync(
    ctx: ExecutionContext, branch: RetrieveBranch, blocks: list[BlockRef]
) -> list[AnnotatedValue]:
    """Sync extract over `blocks`: text tier per anchor group, vision confirm round,
    vision fallback when the text tier finds nothing — `ExtractOp._extract_once`'s
    behavior on the sync transport."""
    entries: list[AnnotatedValue] = []
    if not branch.visual_only and not ctx.config.extract_vision_only:
        for anchor, member_refs, block_idxs in _text._block_groups(blocks):
            entries.extend(
                _extract_group_sync(ctx, branch, anchor, member_refs, block_idxs)
            )
        if entries:
            entries = _confirm_sync(ctx, branch, blocks, entries)
    if not entries:
        entries = _vision_sync(ctx, branch, blocks)
    return entries


# ---------------------------------------------------------------------------
# Agent state + tools
# ---------------------------------------------------------------------------

# Concurrency knob only (worker threads per extract_values fan-out) — the agent's
# observations are never truncated or capped.
_EXTRACT_THREADS = 8


def _block_id(e: SemPoolEntry) -> str:
    bi = e.ref.block_index
    return f"{e.ref.page.month}:{e.ref.page.page}#{'w' if bi is None else bi}"


def _ident(e: SemPoolEntry) -> str:
    return f"{e.ref.page.month} p.{e.ref.page.page}"


def _untitled(e: SemPoolEntry) -> str:
    return e.title or f"(untitled {e.kind})"


def _group_titles(pool: list[SemPoolEntry]) -> list[tuple[str, list[int]]]:
    """Group the (globally numbered) pool by title into `(title, [block_number, ...])`,
    block numbers 1-based into `pool`, members in pool (issue) order. Titles ordered by
    descending reprint count then alphabetically, so the most-reprinted core tables lead."""
    groups: dict[str, list[int]] = {}
    for num, e in enumerate(pool, start=1):
        groups.setdefault(_untitled(e), []).append(num)
    return sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))


def _span(pool: list[SemPoolEntry], nums: list[int]) -> str:
    """Union data-date span over a title group's blocks ("YYYY-MM..YYYY-MM"), or "n/a"."""
    ivs = [iv for n in nums if (iv := pool[n - 1].interval)]
    if not ivs:
        return "n/a"
    return f"{min(lo for lo, _ in ivs)}..{max(hi for _, hi in ivs)}"


def _title_index_line(ti: int, title: str, pool: list[SemPoolEntry], nums: list[int]) -> str:
    return f"  T{ti}  ×{len(nums)}  spans {_span(pool, nums)}  {title}"


def _block_line(pool: list[SemPoolEntry], num: int, extracted: bool) -> str:
    e = pool[num - 1]
    dates = f"{e.interval[0]}..{e.interval[1]}" if e.interval else "none"
    line = f"  block {num} — {_ident(e)}  dates={dates}"
    if e.cols:
        line += f"  [cols: {', '.join(e.cols)}]"
    if extracted:
        line += "  (already extracted)"
    return line


def _summary_entry(eid: str, e: AnnotatedValue) -> str:
    """Schema view of one extracted entry — IDENTICAL to what the compute agent sees
    (`common._describe_entry`): description, qualifiers, provenance, kind, shape, and
    axis labels, but NOT the interior cell values. This is the model-facing render; the
    full values are kept in `state.entries` and reach compute via selection by id."""
    lines = _describe_entry(0, e)
    lines[0] = lines[0].replace("input_values[0]", eid, 1)
    return "\n".join(lines)


def _render_entry(eid: str, e: AnnotatedValue) -> str:
    """One extracted entry for an agent observation — full identity + every value
    (never truncated), so the agent can compare competing variants."""
    src = f"bulletin={e.bulletin or '?'} pages={list(e.pages) or '?'}"
    head = f"{eid}: description={e.description!r}"
    if e.qualifiers:
        head += f" qualifiers={e.qualifiers!r}"
    head += f" unit={e.unit!r} kind={e.kind} {src}"
    if e.kind == "scalar":
        return f"{head}\n    value={e.value!r}"
    if e.kind == "vector":
        v = e.value if isinstance(e.value, dict) else {}
        pairs = ", ".join(f"{k}: {c!r}" for k, c in v.items())
        return f"{head}\n    index={e.index_name!r} values={{{pairs}}}"
    v = e.value if isinstance(e.value, dict) else {}
    lines = [head, f"    rows({e.row_name!r}) x cols({e.col_name!r}):"]
    for rk, rowv in v.items():
        cells = ", ".join(f"{ck}: {cv!r}" for ck, cv in rowv.items())
        lines.append(f"    {rk}: {{{cells}}}")
    return "\n".join(lines)


@dataclass
class _SelectState:
    """Per-branch agent state shared by the tools: the candidate pool (block numbers are
    1-based into `pool`), its `titles` grouping (Tn → block numbers, shown at startup),
    and the registry of extracted entries (by entry id) the final answer selects from.
    `pending_blocks` hands the extract tool's dual-view observation (full text persisted,
    summary shown to the model) to `SelectAgent._blocks_from_output`."""

    ctx: ExecutionContext
    branch: RetrieveBranch
    pool: list[SemPoolEntry]
    titles: list[tuple[str, list[int]]] = field(default_factory=list)
    entries: list[AnnotatedValue] = field(default_factory=list)
    entry_ids: dict[str, int] = field(default_factory=dict)
    extracted_blocks: set[int] = field(default_factory=set)
    pending_blocks: list[Block] | None = None

    def register(self, entries: list[AnnotatedValue]) -> list[str]:
        ids = []
        for e in entries:
            eid = f"E{len(self.entries)}"
            self.entry_ids[eid] = len(self.entries)
            self.entries.append(e)
            ids.append(eid)
        return ids


class ListBlocksTool(Tool):
    name = "list_blocks"
    doc = """\
### list_blocks(title: int, period: str = "")
Expand one title group from the candidate list into its member blocks (the same table
reprinted across issues), oldest first. `title` is a Tn number from the candidate list
(pass the integer n). `period` ("YYYY-MM" or "YYYY-MM..YYYY-MM") keeps only blocks whose
data span overlaps it. Returns each block's GLOBAL number — pass those to extract_values
— plus its issue, page, and data span.

```python
list_blocks(12, period="1962-04")
```"""

    def __init__(self, state: _SelectState) -> None:
        self._state = state

    def __call__(self, title, period: str = "") -> str:
        s = self._state
        try:
            ti = int(title)
        except (TypeError, ValueError):
            return f"invalid title {title!r} — pass a Tn number (the integer n) from the candidate list"
        if not 1 <= ti <= len(s.titles):
            return f"unknown title T{title} — the candidate list runs T1..T{len(s.titles)}"
        name, nums = s.titles[ti - 1]
        intervals = _to_intervals(str(period)) if period else None

        def overlaps(num: int) -> bool:
            iv = s.pool[num - 1].interval
            if not intervals or not iv:
                return True
            lo, hi = iv
            return any(not (hi < p_lo or lo > p_hi) for p_lo, p_hi in intervals)

        kept = [n for n in nums if overlaps(n)]
        head = f'T{ti} "{name}": {len(nums)} block(s)'
        if intervals:
            head += f" (filtered to {period}: {len(kept)})"
        if not kept:
            return head + " — none overlap that period"
        lines = [_block_line(s.pool, n, n in s.extracted_blocks) for n in kept]
        return head + ":\n" + "\n".join(lines)


def _extract_one(
    ctx: ExecutionContext, branch: RetrieveBranch, e: SemPoolEntry
) -> list[AnnotatedValue]:
    """One block's full extract cascade, run under its own contextvars copy so
    worker-thread emits keep the caller's step frame."""
    return contextvars.copy_context().run(_extract_blocks_sync, ctx, branch, [e.ref])


class ExtractValuesTool(Tool):
    name = "extract_values"
    doc = """\
### extract_values(blocks: list[int])
Extract typed values from candidate blocks (global block numbers from list_blocks).
Each block is read by its own extraction call over just that block's page(s), and the
calls run in PARALLEL — batch every block you want read into one call rather than
extracting one block at a time. Returns a SCHEMA VIEW of each extracted entry
(description, qualifiers, unit, provenance, shape, axis labels — not the raw cells),
tagged with an entry id (E0, E1, …); your final answer selects among ALL entry ids
accumulated so far.

```python
extract_values([12, 47, 48])
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
            return f"invalid block number in {blocks!r} — use integers from the candidate list"
        bad = [n for n in nums if not 1 <= n <= len(s.pool)]
        if bad:
            return f"unknown block number(s) {bad!r} — the candidate list runs 1..{len(s.pool)}"
        cands = [s.pool[n - 1] for n in nums]
        s.ctx.emit(f"extract_fanout n_blocks={len(cands)}")
        if len(cands) == 1:
            results: list[list[AnnotatedValue] | BaseException] = [
                _extract_one(s.ctx, s.branch, cands[0])
            ]
        else:
            with ThreadPoolExecutor(
                max_workers=min(len(cands), _EXTRACT_THREADS)
            ) as pool:
                futures = [
                    pool.submit(_extract_one, s.ctx, s.branch, e) for e in cands
                ]
                results = []
                for f in futures:
                    try:
                        results.append(f.result())
                    except Exception as exc:  # one bad block must not sink the batch
                        results.append(exc)
        # Two renders per block: `full` (every cell — persisted to the trajectory + trace
        # viewer) and `summary` (the compute-agent schema view — what the model sees).
        # The dual-view TextBlock carries both; see SelectAgent._blocks_from_output.
        full: list[str] = []
        summary: list[str] = []
        for num, e, res in zip(nums, cands, results):
            head = f"block {num} ({_ident(e)}):"
            if isinstance(res, BaseException):
                s.ctx.emit(f"extract_block_failed block={num} error={str(res)!r}")
                line = f"{head} extraction failed — {type(res).__name__}: {res}"
                full.append(line)
                summary.append(line)
                continue
            s.extracted_blocks.add(num)
            if not res:
                full.append(f"{head} no relevant values found")
                summary.append(f"{head} no relevant values found")
                continue
            eids = s.register(res)
            full.append("\n".join([head, *(_render_entry(i, v) for i, v in zip(eids, res))]))
            summary.append("\n".join([head, *(_summary_entry(i, v) for i, v in zip(eids, res))]))
        s.pending_blocks = [
            TextBlock(text="\n\n".join(full), llm_text="\n\n".join(summary))
        ]
        return "\n\n".join(summary)


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class SelectAgent(MultiTurnAgent):
    name = "select_agent"
    default_effort = "medium"

    briefing = """\
You gather and SELECT the data that answers one retrieval request from a pool of
candidate U.S. Treasury Bulletin content blocks (tables/charts, one per bulletin
page region). The user message gives the research question, the retrieval target
(concept, period(s), optional pinned issue), and the candidate pool as a list of
the UNIQUE table/chart titles in the pool (keyed T1, T2, …), each with how many
issues reprinted it and the data-date span it covers. The same table is typically
reprinted across consecutive issues, sometimes with revisions.

Work iteratively:
1. Scan the unique titles for the one(s) carrying the asked concept. list_blocks(Tn)
   expands a title into its per-issue blocks (each with a global block number and its
   data span); pass `period=` to keep only the issues covering a date you need.
2. extract_values on the promising block numbers. Each block is read by its own LLM
   call and the calls run in PARALLEL — batch every block you currently want read
   into ONE extract_values call instead of extracting one at a time. Extract
   COMPETING candidates (different issues reprinting the same table, or different
   tables carrying the same label) in the same batch when it is not obvious which
   print the question wants.

You see each extracted entry as a schema view — its description, qualifiers,
provenance (which issue/pages), shape, and axis labels — NOT the raw cell values.
That is enough to select: the asked scope shows in the description/qualifiers/labels,
and which print to keep follows from the provenance (issue date).

Selection rules:
- Match the question's qualifier words ("subject to limitation", "tenders
  accepted", "issued", "total outstanding", series names) against each entry's
  qualifiers and description; the exact column/row scope must match the asked
  scope — a broader aggregate ("<concept> and related activities") is a
  DIFFERENT series.
- Same datum printed in several issues: select the MOST UP-TO-DATE print — a
  later issue's figure supersedes earlier prints (revisions). Deviate only when
  the question pins a source issue (as_of) or explicitly asks for the original
  / contemporaneous figure.
- Select exactly ONE entry per requested concept+period; competing variants you
  extracted but did not select stay behind.

Finish — emit your final answer — once BOTH hold:
- every requested period is covered by one selected entry whose description and
  qualifiers match the asked scope, and
- for each selected datum you checked the list for later issues reprinting it
  and selected the latest print (or honored the pin).
Do not finish with a period uncovered while plausible titles remain unextracted.
Report an empty selection only after the unique-title list shows no table that
could carry the requested series."""

    final_answer_doc = """\
  {"selected": ["E0", "E3", ...], "why": "<one short line per kept entry: which print/column and why>"}
- `selected` lists entry ids from your extract_values observations — these (and
  only these) entries are handed to the downstream compute step.
- An empty `selected` means the pool genuinely lacks the requested data; `why`
  must then say exactly what is missing."""

    visible_observations = None  # selection needs the full comparison history

    def __init__(self, state: _SelectState, *, max_steps: int) -> None:
        self._state = state
        # Bound each selection-step LLM call the same way SearchAgent does: cap
        # output and impose a hard per-request wall-clock timeout. Unbounded, these
        # turns ran to 400+s on large inputs (the full survivor list + accumulated
        # extraction observations) with no output, and one parse-retry blew out to
        # ~63k thinking tokens. See SkunkConfig for the thinking/output caveat.
        cfg = state.ctx.config
        self.max_output_tokens = cfg.select_agent_max_output_tokens
        self.request_timeout_s = cfg.select_agent_request_timeout_s
        super().__init__(
            [ListBlocksTool(state), ExtractValuesTool(state)],
            max_steps=max_steps,
        )

    def _blocks_from_output(self, out) -> list[Block]:
        """Use the extract tool's stashed dual-view block (full text persisted, summary
        shown to the model) when present; otherwise the default stdout/result render."""
        pending = self._state.pending_blocks
        self._state.pending_blocks = None
        if pending is not None:
            return pending
        return super()._blocks_from_output(out)

    def validate_final_answer(self, payload: object, observations: list[str]):
        if not isinstance(payload, dict) or "selected" not in payload:
            return 'final answer must be {"selected": [<entry ids>], "why": "..."}'
        sel = payload["selected"]
        if not isinstance(sel, list) or not all(isinstance(x, str) for x in sel):
            return "'selected' must be a list of entry id strings (e.g. [\"E0\"])"
        unknown = [x for x in sel if x not in self._state.entry_ids]
        if unknown:
            return (
                f"unknown entry id(s) {unknown!r} — use ids returned by extract_values"
            )
        if not sel and not str(payload.get("why", "")).strip():
            return "an empty selection requires a 'why' explaining what is missing"
        return None


async def run_select_agent(
    ctx: ExecutionContext,
    branch: RetrieveBranch,
    blocks: list[BlockRef],
    sem_pool: list[SemPoolEntry],
) -> list[AnnotatedValue]:
    """Run one branch's selection agent and return the entries it selected.
    `sem_pool` is the branch's survivor pool; when absent (golden replay,
    search-agent retriever) it is rebuilt from `blocks` via the catalog."""
    pool_entries = sem_pool
    if not pool_entries:
        from skunk.page_index.query import PageIndexRetriever

        pool_entries = PageIndexRetriever.pool_for_blocks(
            blocks, str(ctx.config.pdf_dir)
        )
    if not pool_entries:
        raise StepFailed("extract", "selection agent has no candidate pool")
    pool = list({_block_id(e): e for e in pool_entries}.values())
    pool.sort(
        key=lambda e: (
            e.ref.page.month or "",
            e.ref.page.page or 0,
            e.ref.block_index if e.ref.block_index is not None else -1,
        )
    )
    titles = _group_titles(pool)
    state = _SelectState(ctx=ctx, branch=branch, pool=pool, titles=titles)
    agent = SelectAgent(state, max_steps=ctx.config.select_agent_max_steps)

    months = sorted({e.ref.page.month for e in pool if e.ref.page.month})
    span = f"{months[0]}..{months[-1]}" if months else "unknown"
    parts = [
        f'Research question: "{ctx.question}"',
        f"Retrieval target: {branch.key}",
    ]
    if branch.period:
        parts.append(f"Period(s) the data must cover: {branch.period}")
    if branch.as_of:
        parts.append(f"Pinned source issue (as_of): {branch.as_of}")
    listing = "\n".join(
        _title_index_line(ti, name, pool, nums)
        for ti, (name, nums) in enumerate(titles, start=1)
    )
    parts.append(
        f"Candidate pool: {len(pool)} content blocks from issues {span}, under "
        f"{len(titles)} unique titles (×n = issues reprinting it). list_blocks(<Tn>) "
        f"expands a title into its blocks:\n" + listing
    )
    payload = await agent.call(ctx, "\n".join(parts))

    sel = payload["selected"]
    why = str(payload.get("why", ""))
    ctx.emit(
        f"select_agent_done n_extracted={len(state.entries)} n_selected={len(sel)} "
        f"selected={sel!r} why={why[:200]!r}"
    )
    if not sel:
        raise StepFailed("extract", f"selection agent found no data: {why}")
    return [state.entries[state.entry_ids[x]] for x in sel]
