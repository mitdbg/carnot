"""Page-index query path: retrieval (`PageIndexRetriever`).

`PageIndexRetriever.retrieve_all` runs all branches at once: per-era ToC chapter pick →
year filter → one coarse semantic filter over the deduped union. The filter judges per
CONTENT BLOCK (flat list, one boolean each) against the branch RETRIEVAL TARGETS; a page
is kept iff any block fits any target. Output is BLOCK-granular (`BlockRef`); the
block-selection tournament (`skunk.block_select`) narrows the survivors downstream."""

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
    parse_json_response,
)
from skunk.errors import StepFailed, ParseError
from skunk.plan import PagePin, RetrieveBranch
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


def _sample_pages(refs: list[PageRef], limit: int = 5) -> list[dict[str, str | int | None]]:
    return [
        {"bulletin": ref.month, "page": ref.page}
        for ref in refs[:limit]
    ]


class _PeriodEntry(NamedTuple):
    lo: str  # inclusive YYYY-MM data-span start
    hi: str  # inclusive YYYY-MM data-span end

    @property
    def label(self) -> str:
        return self.lo if self.lo == self.hi else f"{self.lo}..{self.hi}"


def _to_entries(period: str | None) -> list[_PeriodEntry] | None:
    """Parse a period string into entries: a comma-list of `YYYY-MM` months /
    `YYYY-MM..YYYY-MM` ranges. Returns None when empty or malformed (callers then
    no-op the filter)."""
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
            out.append(_PeriodEntry(lo, hi))
        return out or None
    except ValueError:
        return None


def _branch_entries(branch: RetrieveBranch) -> list[_PeriodEntry] | None:
    """The branch's parsed period entries."""
    return _to_entries(branch.period)


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


# A requested period can POST-DATE the data it refers to — e.g. a security identified by its
# maturity ("notes maturing July 1984") whose auction happened years earlier. ToC era pruning
# therefore extends the period's lower bound this many years backward, so the earlier era that
# actually holds the data is still scanned. (The per-branch `_year_filter` stays strict.)
_TOC_BACKWARD_SLACK_YEARS = 5


def _shift_years(month: str, delta: int) -> str:
    """Shift a `YYYY-MM` string by `delta` years, clamping the year to a valid 4-digit value."""
    year = min(9999, max(1, int(month[:4]) + delta))
    return f"{year:04d}{month[4:]}"


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
    """Orchestrates retrieval (ToC chapter pick → year filter → semantic filter). The
    artifact is loaded once at construction."""

    # -- prompts ---------------------------------------------------------------

    _CHAPTER_PICK_PROMPT = (
        """\
For each Treasury Bulletin chapter, decide whether the answer to the question could plausibly be
in it. Judge every chapter independently.

## Input

A single JSON object:
  {"question": "<question>",
   "chapters": [{"chapter": "<name>", "n_pages": <int>, "description": "<scope>",
                 "examples": ["<sub-area>", ...]}, ...]}

"""
        + CHAPTER_FIELDS
        + """

`description` and `examples` indicate a chapter's scope but rarely use the question's wording;
judge semantically.

## Output

A single bare JSON array of booleans — no prose, no markdown fences — one entry per chapter, in
the input order:
  [true, false, ...]

Return true when the answer could plausibly be in the chapter; false only when it is clearly
unrelated. When in doubt, or when two chapters overlap, return true for both.
"""
    )

    _SEMFILTER_PROMPT = (
        """\
You are given retrieval targets (data concepts a research question needs) and a flat batch of
content blocks — each a single table/chart/prose region from a Treasury Bulletin page. For each
block, decide independently whether it holds data relevant to any one of the targets. You see only
a compact summary per block (title, column/row labels, dates), not the numbers.

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

True when the summary fits at least one target; false only when clearly unrelated to every target.
"""
    )

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
        self.catalog = self._catalog  # public read-only alias (the SelectAgent queries it)
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
        """Eras whose ToC to scan: `period` prunes eras whose span ends before the
        period start, less a `_TOC_BACKWARD_SLACK_YEARS` slack (a requested period may
        post-date its data, e.g. a security's maturity); no period keeps all eras."""
        intervals = _to_intervals(branch.period)
        if not intervals:
            return self._tree.eras
        earliest = min(lo[:7] for lo, _ in intervals)
        cutoff = _shift_years(earliest, -_TOC_BACKWARD_SLACK_YEARS)
        kept = [e for e in self._tree.eras if e.span[1][:7] >= cutoff]
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
        branch then narrows by its own period in `_year_filter`."""
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
        picked = await pick_call.call(ctx, user, temperature=0.4)
        out = [ref for ch in picked for ref in pages[ch]]
        ctx.emit(
            f"pick_chapters era={era.span} picked={picked!r} pages={len(out)}",
            data={
                "era": list(era.span),
                "picked": picked,
                "n_chapters": len(names),
                "pages": len(out),
            },
        )
        return out

    def _year_filter(
        self, candidates: list[PageRef], branch: RetrieveBranch, ctx: ExecutionContext
    ) -> list[PageRef]:
        # period filters by DATA span; a candidate passes on ANY entry.
        kept = list(candidates)
        entries = _branch_entries(branch)
        if entries:

            def _passes(ref: PageRef) -> bool:
                row = self._catalog.get(ref)
                if row is None or row.date_interval is None:
                    return False
                return any(
                    _overlaps(row.date_interval, [(e.lo, e.hi)]) for e in entries
                )

            kept = [ref for ref in kept if _passes(ref)]
        ctx.emit(
            f"year_filter key={branch.key!r} period={branch.period!r} "
            f"kept={len(kept)}/{len(candidates)}",
            data={
                "key": branch.key,
                "period": branch.period,
                "total": len(candidates),
                "pages": [f"{r.month}:{r.page}" for r in kept],
            },
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
            data={
                "total": len(pages),
                "blocks_kept": n_blocks_kept,
                "blocks": n_blocks,
                "n_calls": len(batches),
                "pages": [f"{r.month}:{r.page}" for r in kept_refs],
            },
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

    def _resolve_page_pin(self, pin: PagePin, ctx: ExecutionContext) -> list[PageRef]:
        """Resolve a `page_pin` to candidate pages, returning BOTH interpretations of its
        (ambiguous) page number so downstream decides which is meant:
          (a) the printed-label page(s) — catalog rows in the pinned issue whose `printed_page`
              equals the stated number (0..n; usually 1, but a corrected reprint can repeat a
              label), and
          (b) the PDF-index page — `PageRef(issue, page)` when it's a catalog row (0..1).
        Deduped (printed first, then a distinct PDF page), in page order. Empty when neither
        resolves — the caller turns that into a `StepFailed` so the branch replans."""
        want = str(pin.page)
        printed = sorted(
            (
                ref
                for ref, row in self._catalog.items()
                if ref.month == pin.bulletin
                and row.printed_page is not None
                and row.printed_page.strip() == want
            ),
            key=lambda r: (r.page or 0),
        )
        pdf_ref = PageRef(month=pin.bulletin, page=pin.page)
        pages = list(printed)
        if pdf_ref in self._catalog and pdf_ref not in pages:
            pages.append(pdf_ref)
        ctx.emit(
            f"page_pin_resolve bulletin={pin.bulletin} page={pin.page} "
            f"printed={[r.page for r in printed]} pdf={pdf_ref.page if pdf_ref in self._catalog else None} "
            f"resolved={len(pages)}",
            data={
                "bulletin": pin.bulletin,
                "page": pin.page,
                "printed_pages": [r.page for r in printed],
                "pdf_page": pdf_ref.page if pdf_ref in self._catalog else None,
            },
        )
        return pages

    async def retrieve_all(
        self,
        ctx: ExecutionContext,
        branches: list[RetrieveBranch],
        *,
        document_scopes: list[list[str] | None] | None = None,
    ) -> list[list[BlockRef]]:
        """Phases (1) and (2) only: ToC pick → year/scope filter → semantic filter.
        Returns the sem-filter survivor blocks per branch, aligned to `branches`.
        No block selection — the caller (orchestrator) runs the selection agent
        as a separate step. `document_scopes` hard-scopes a branch's candidates to a
        set of bulletins (HITL human-required documents); a scoped branch skips the
        ToC/year filter and takes every page of those bulletins as its candidates."""
        # A `page_pin` branch is a pure positional FETCH: it bypasses ToC pick / year / semantic
        # filter entirely and resolves straight to its issue+page (both interpretations). Resolve
        # pins up front so pinned branches drop out of the content-retrieval phases below.
        scopes = document_scopes or [None] * len(branches)
        pinned: list[list[PageRef] | None] = [
            self._resolve_page_pin(b.page_pin, ctx) if b.page_pin is not None else None
            for b in branches
        ]
        # Phase 1 — one unified ToC pick (question only), then each branch's own date filter.
        unscoped = [
            branch
            for branch, scope, pin in zip(branches, scopes, pinned)
            if not scope and pin is None
        ]
        chapter_pages = (
            await self._pick_chapters(branches=unscoped, ctx=ctx)
            if unscoped
            else []
        )
        cand = []
        for branch, scope, pin in zip(branches, scopes, pinned):
            if pin is not None:
                cand.append(pin)
            elif scope:
                refs = sorted(
                    (
                        ref
                        for ref in self._catalog
                        if ref.month in set(scope)
                    ),
                    key=lambda ref: (ref.month or "", ref.page or 0),
                )
                ctx.emit(
                    f"human_document_scope key={branch.key!r} "
                    f"bulletins={scope!r} pages={len(refs)}",
                    data={
                        "key": branch.key,
                        "bulletins": scope,
                        "pages": len(refs),
                        "sample_pages": _sample_pages(refs),
                    },
                )
                cand.append(refs)
            else:
                cand.append(self._year_filter(chapter_pages, branch, ctx))

        # Phase 2 — semantic filter over the content-addressed candidates only (pinned branches
        # are fetched whole, never filtered).
        wanted: dict[PageRef, set[int]] = {}
        for i, refs in enumerate(cand):
            if pinned[i] is not None:
                continue
            for ref in refs:
                wanted.setdefault(ref, set()).add(i)
        unique = sorted(wanted, key=lambda r: (r.month or "", r.page or 0))
        verdict = await self._semantic_filter(unique, branches, ctx)

        # Route per branch
        branch_blocks: list[list[BlockRef]] = []
        for i, refs in enumerate(cand):
            b = branches[i]
            if pinned[i] is not None:
                # Pure fetch: every resolved page in full (whole-page refs), no filtering — both
                # the printed-label and PDF-index page go downstream for the caller to decide.
                block_refs = [br for ref in refs for br in self._block_refs_for(ref, [])]
                pin_label = (
                    f"{b.page_pin.bulletin}:{b.page_pin.page}" if b.page_pin else "?"
                )
                ctx.emit(
                    f"page_pin_retrieve key={b.key!r} pin={pin_label} "
                    f"pages={[f'{r.month}:{r.page}' for r in refs]} block_count={len(block_refs)}"
                )
                branch_blocks.append(block_refs)
                continue
            kept_pages = [ref for ref in refs if not verdict[ref] or any(verdict[ref])]
            block_refs = [
                br
                for ref in kept_pages
                for br in self._block_refs_for(ref, verdict[ref])
            ]
            ctx.emit(
                f"page_index_retrieve key={b.key!r} period={b.period!r} "
                f"catalog_size={self._catalog_size} anchor_count={len(kept_pages)} block_count={len(block_refs)} "
            )
            branch_blocks.append(block_refs)

        return branch_blocks

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
        """The pool for `refs` without running any selection: anchor-resolve the
        refs to their catalog rows and emit every content block as a
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
        """A self-contained pool entry for one candidate block."""
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
