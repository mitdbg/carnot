"""Page-selection agent — the intermediate narrowing stage (PROTOTYPE).

Sits between the semantic filter and extract. The filter is recall-over-precision and
hands a few hundred candidate pages per branch straight to extract; this agent narrows that
to exactly the pages a sub-question needs — which may be one page, a multi-page table, a
whole time series, or the right vintage among near-duplicate monthly reprints. It does this
without any brittle deterministic grouping (which would lean on the `is_continuation` /
table-identity signal we already distrust). Instead it navigates the candidates the way a
human greps, all served by the `PageStore`:

  - candidate SUMMARIES (`store.summary`) are laid out into a few greppable per-era "files";
  - `grep` / `read_file` browse that summary index (cheap, navigable);
  - `read_page` (`store.text`) returns a candidate's full cleaned text — to verify a value is
    actually printed on a given page before committing;
  - the briefing routes selection by what the question actually needs (single page /
    continuation / series / revised-vintage), keeping recall when uncertain.

It is a SELECTION agent: it emits page refs (`{"page_keys": [...]}`), and extract still
pulls the values. It reuses `MultiTurnAgent` (loop + JSON final answer + terminal turn)
exactly as `SearchAgent` does — only the tool set and briefing differ. Everything it needs
(summaries, page text, catalog rows) comes from the shared `PageStore`; eras come from the
concept tree.

PROTOTYPE: wired into the orchestrator behind `config.page_select`, not yet validated.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from skunk.common import ExecutionContext, PageRef
from skunk.multi_turn_agent import MultiTurnAgent, Tool
from skunk.plan import RetrieveBranch
from skunk.page_index.data_model import (
    TREE_FILE,
    ConceptTree,
    EraTree,
    PageCatalogRow,
    page_index_root,
)
from skunk.page_index.store import PageStore, get_page_store, summarize_row

# Caps so a wide grep / read_file can't blow the agent's context window.
_MAX_GREP_LINES = 200
_MAX_FILE_LINES = 400


@lru_cache(maxsize=1)
def _load_eras() -> tuple[EraTree, ...]:
    """The page-index eras, loaded once per process from the concept tree (used only to group
    the candidate summaries into per-era files)."""
    tree = ConceptTree.model_validate_json((page_index_root() / TREE_FILE).read_bytes())
    return tuple(tree.eras)


def _page_key(row: PageCatalogRow) -> str:
    """The agent-facing key for a candidate page (`YYYY_MM_pageid`)."""
    return f"{row.bulletin.replace('-', '_')}_{row.page}"


def _bulletin_in_span(bulletin: str, span: tuple[str, str]) -> bool:
    """True iff `bulletin` ("YYYY-MM") falls within an era `span` (compared at month
    granularity, tolerating a stray day in stored spans)."""
    lo, hi = span[0][:7], span[1][:7]
    return lo <= bulletin[:7] <= hi


def _era_name(era: EraTree) -> str:
    """A stable, greppable file name for an era — its label, falling back to its span."""
    return era.label.strip() or f"{era.span[0]}..{era.span[1]}"


# --------------------------------------------------------------------------- tools


class GrepSummariesTool(Tool):
    name = "grep"

    def __init__(self, files: dict[str, list[str]]) -> None:
        self._files = files

    def __call__(self, pattern: str, file: str | None = None) -> str:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"[error] bad regex {pattern!r}: {e}"
        if file is not None and file not in self._files:
            return f"[error] no such file {file!r}; available: {sorted(self._files)}"
        names = [file] if file is not None else list(self._files)
        hits: list[str] = []
        for nm in names:
            hits.extend(line for line in self._files[nm] if rx.search(line))
        if not hits:
            return "[no matches]"
        shown = hits[:_MAX_GREP_LINES]
        out = "\n".join(shown)
        if len(hits) > len(shown):
            out += f"\n... ({len(hits) - len(shown)} more matches — narrow the pattern or pass file=)"
        return out

    doc = """\
### grep(pattern: str, file: str | None = None)
Regex search (case-insensitive) over the candidate-page SUMMARY index — one line per page,
each starting with its `page_key`. Searches every era file by default, or just `file=` when
given. Use it to find which issues report a given table/series (e.g. the same table reprinted
across many months).

```python
grep("federal debt outstanding")
grep("receipts", file="1996-06..2010-12")
```"""


class ReadFileTool(Tool):
    name = "read_file"

    def __init__(self, files: dict[str, list[str]]) -> None:
        self._files = files

    def __call__(self, file: str) -> str:
        lines = self._files.get(file)
        if lines is None:
            return f"[error] no such file {file!r}; available: {sorted(self._files)}"
        shown = lines[:_MAX_FILE_LINES]
        out = f"# {file} ({len(lines)} pages)\n" + "\n".join(shown)
        if len(lines) > len(shown):
            out += f"\n... ({len(lines) - len(shown)} more — use grep to filter)"
        return out

    doc = """\
### read_file(file: str)
Return a whole per-era summary file: every candidate page's summary line in that era. Use it
to scan an era's pages at once (e.g. see all vintages of one table, or every period of a
series, side by side).

```python
read_file("1996-06..2010-12")
```"""


class ReadPageTool(Tool):
    name = "read_page"

    def __init__(self, store: PageStore, key_to_row: dict[str, PageCatalogRow]) -> None:
        self._store = store
        self._key_to_row = key_to_row

    def __call__(self, page_key: str) -> str:
        row = self._key_to_row.get(page_key)
        if row is None:
            return f"[error] {page_key!r} is not in the candidate set"
        try:
            text = self._store.text(row.ref)
        except Exception as e:  # noqa: BLE001 — surface store problems as an observation, don't crash the loop
            return f"[error] could not read {page_key!r}: {type(e).__name__}: {e}"
        if not text:
            return f"=== {page_key} ===\n[no stored text]"
        return f"=== {page_key} ===\n{text}"

    doc = """\
### read_page(page_key: str)
Return the full cleaned text of one candidate page (the anchor's merged text). Use it to
VERIFY a value is actually printed on a given page before selecting it — the summaries can be
lossy and tables are sometimes mis-merged.

```python
read_page("2001_06_41")
```"""


# --------------------------------------------------------------------------- agent


class PageSelectAgent(MultiTurnAgent):
    """Selects exactly the candidate pages a branch's sub-question needs, by browsing a
    per-era summary index (grep / read_file) and verifying with read_page. The selection may
    be one page, a multi-page table, a whole series, or the right vintage among reprints. One
    agent per branch — it resolves that branch's candidate refs to catalog rows (via the
    `PageStore`) and materialises the summary index in `__init__`."""

    name = "page_select"
    # A bounded selection loop — far cheaper than the cold-corpus search agent.
    visible_observations = 6

    briefing = (
        "You select which already-filtered Treasury Bulletin pages actually carry the data a "
        "single sub-question needs. The candidates are given to you as a few SUMMARY files, one "
        "per era (a span of years with a stable reporting convention). Each summary line starts "
        "with the page's `page_key` and describes the page's tables/charts — NOT their numbers.\n\n"
        "Return exactly the pages the sub-question needs — no more, no fewer. Most targets need "
        "just 1-3 pages: aim for that few, and only exceed it when the target is genuinely a "
        "multi-period time series (then keep one page per period). Depending on the question the "
        "selection may be:\n"
        "  - a single page (a one-off / point-in-time value);\n"
        "  - several pages of ONE issue (a table continued across pages — keep them all);\n"
        "  - many pages across issues (a time series — keep one page per period the question "
        "spans);\n"
        "  - one of several near-duplicate reprints (the SAME table recurs every month because "
        "issues restate/revise it — when only one value is needed, keep the right vintage, not "
        "every reprint).\n\n"
        "For that last, revision case, pick the vintage by the question's intent: point-in-time "
        "/ 'as reported in <issue>' → that issue; a figure later revised, where the question "
        "wants the best / current value → the LATEST issue that reports that date.\n\n"
        "Workflow: `grep` the summary index to find which issues carry the table; `read_file` to "
        "scan an era's pages together; `read_page` to confirm the value is actually printed "
        "before you commit (summaries can be lossy and tables are sometimes mis-merged).\n\n"
        "When you CANNOT confidently decide, KEEP the plausible pages — recall matters more than "
        "precision here; a later extraction step is the precision gate. Never invent a "
        "`page_key`; only return keys that appear in the summaries."
    )

    final_answer_doc = """\
A JSON object listing the `page_key`s you selected, under the key "page_keys":
```json
{"page_keys": ["2001_06_41", "2001_06_42"]}
```
Use each `page_key` exactly as it appears in the summaries."""

    def __init__(
        self,
        refs: list[PageRef],
        pdf_dir: str | Path,
        *,
        max_steps: int = 6,
    ) -> None:
        store = get_page_store(str(pdf_dir))
        eras = _load_eras()

        # Resolve the (expanded) candidate refs to their distinct catalog anchor rows via the
        # store — a continuation ref resolves to its anchor, deduped, first-seen order.
        rows: list[PageCatalogRow] = []
        seen: set[PageRef] = set()
        for ref in refs:
            row = store.catalog_row(ref)
            if row is None or row.ref in seen:
                continue
            seen.add(row.ref)
            rows.append(row)
        self._key_to_row: dict[str, PageCatalogRow] = {_page_key(r): r for r in rows}

        # Materialise the summary index into per-era files (one line per candidate page) —
        # summaries served by the store. Rows outside every span land in "other".
        files: dict[str, list[str]] = {}
        for row in rows:
            era = next((e for e in eras if _bulletin_in_span(row.bulletin, e.span)), None)
            name = _era_name(era) if era is not None else "other"
            dates = f"{row.date_interval[0]}..{row.date_interval[1]}" if row.date_interval else "none"
            # Format from the row already in hand — avoids a second catalog lookup via store.summary.
            body = summarize_row(row)
            files.setdefault(name, []).append(f"{_page_key(row)} | dates={dates} | {body}")
        for lines in files.values():
            lines.sort()  # stable, deterministic order within a file (by page_key)
        self._files = files

        tools = [
            GrepSummariesTool(files),
            ReadFileTool(files),
            ReadPageTool(store, self._key_to_row),
        ]
        super().__init__(tools, max_steps=max_steps)

    def has_candidates(self) -> bool:
        """Whether any candidate ref resolved to a catalog row (else there's nothing to select
        over — the caller should keep the original refs)."""
        return bool(self._key_to_row)

    async def select(
        self,
        ctx: ExecutionContext,
        question: str,
        branch: RetrieveBranch,
    ) -> list[PageRef]:
        """Run the selection loop and return extract-ready page refs (kept anchors expanded to
        their member pages). Falls back to the FULL candidate set on an empty / unusable answer
        or a loop failure, so the stage never silently drops everything (recall guard)."""
        if not self._key_to_row:
            return []

        file_listing = "\n".join(
            f"  - {name} ({len(lines)} pages)" for name, lines in sorted(self._files.items())
        )
        parts = [
            f"Sub-question target: {branch.key}",
            f'Full question (for context): "{question}"',
        ]
        if branch.period:
            parts.append(f"Data period: {branch.period}")
        if branch.as_of:
            parts.append(f"Reported in / as of: {branch.as_of}")
        parts.append(f"Candidate summary files ({len(self._key_to_row)} pages total):\n{file_listing}")
        user = "\n".join(parts)

        try:
            payload = await self.call(ctx, user)
        except Exception as e:  # noqa: BLE001 — max-steps / loop failure: keep recall, don't sink the branch
            ctx.emit(f"page_select_fallback reason={type(e).__name__}: {e}")
            return self._all_member_refs()

        keys = payload.get("page_keys") if isinstance(payload, dict) else None
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            ctx.emit("page_select_fallback reason=empty_answer")
            return self._all_member_refs()

        bad: list[str] = []
        selected_rows: list[PageCatalogRow] = []
        seen: set[str] = set()
        for k in keys:
            k = str(k)
            row = self._key_to_row.get(k)
            if row is None:
                bad.append(k)
                continue
            if k not in seen:
                seen.add(k)
                selected_rows.append(row)
        if bad:
            ctx.emit(f"page_select_bad_keys n_bad={len(bad)} keys={bad[:5]!r}")
        if not selected_rows:
            ctx.emit("page_select_fallback reason=no_valid_keys")
            return self._all_member_refs()

        refs = _expand(selected_rows)
        ctx.emit(
            f"page_select key={branch.key!r} candidates={len(self._key_to_row)} "
            f"selected_anchors={len(selected_rows)} expanded_refs={len(refs)}"
        )
        return refs

    def _all_member_refs(self) -> list[PageRef]:
        """Every candidate, anchors expanded — the recall-preserving fallback."""
        return _expand(self._key_to_row.values())


def _expand(rows: Iterable[PageCatalogRow]) -> list[PageRef]:
    """Expand kept anchors into their member refs (anchor + folded continuation pages),
    deduped, preserving first-seen order — so extract reads full multi-page tables."""
    out: list[PageRef] = []
    seen: set[PageRef] = set()
    for row in rows:
        for ref in row.member_refs():
            if ref not in seen:
                seen.add(ref)
                out.append(ref)
    return out
