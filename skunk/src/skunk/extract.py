from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator
from typing import Any

from skunk.common import (
    AnnotatedValue,
    B64Image,
    BlockRef,
    ExecutionContext,
    PageRef,
    parse_json_response,
)
from skunk.errors import ParseError, StepFailed
from skunk.prompted_call import PromptedCall
from skunk.plan import RetrieveBranch
from skunk.page_index.store import get_page_store


def _blocks_to_pagerefs(blocks: list[BlockRef]) -> list[PageRef]:
    """Deduped union (first-seen order) of every block's member pages — the physical pages the
    vision tier renders for a set of blocks."""
    seen: set[PageRef] = set()
    refs: list[PageRef] = []
    for b in blocks:
        for r in b.member_refs:
            if r not in seen:
                seen.add(r)
                refs.append(r)
    return refs


def _render_pages_b64(
    refs: list[PageRef],
    ctx: ExecutionContext,
    *,
    strict: bool = False,
) -> tuple[list[B64Image], list[PageRef]]:
    """Page images for the vision tier, from the page store (rendered on demand at 200 DPI +
    cached). The returned refs identify each image's source page so the prompt's numbered list
    can't conflate them. When `strict`, a page that fails to render (error or missing PNG) raises
    StepFailed instead of being silently skipped — the vision tier must not drop the answer page.
    The best-effort confirm pass leaves `strict` off (a render miss there just skips correction)."""
    store = get_page_store(str(ctx.config.pdf_dir))
    images: list[B64Image] = []
    rendered_refs: list[PageRef] = []
    for ref in refs:
        try:
            img = store.image(ref)
        except Exception as e:  # noqa: BLE001 — any render error
            ctx.emit(f"render_failed page={str(ref)} error={str(e)!r}")
            if strict:
                raise StepFailed(
                    "extract", f"failed to render page {ref} for vision extract: {e}"
                ) from e
            continue
        if img is None:
            ctx.emit(f"no_png page={str(ref)}")
            if strict:
                raise StepFailed("extract", f"no rendered PNG available for page {ref}")
            continue
        ctx.emit(f"rendered_png page={str(ref)}")
        images.append(img)
        rendered_refs.append(ref)
    return images, rendered_refs


def _parse_extract_response(raw: str, ctx: ExecutionContext) -> list[AnnotatedValue]:
    """Parse an LLM reply into AnnotatedValues, as a `PromptedCall` parse hook.
    Raises `ParseError` (→ `call()` re-prompts, echoing the detail, with escalating
    temperature) on a reply the model should fix: unparseable, non-array, or any entry
    that fails `AnnotatedValue` validation. An empty array is a legitimate "nothing
    relevant found" — it returns `[]` (no retry; the caller falls through to the next
    tier), since re-prompting can't conjure data that isn't on the page."""
    obj = parse_json_response(raw)
    if obj is None:
        raise ParseError(
            raw,
            "reply is not valid JSON — return a JSON array of AnnotatedValue entries",
        )
    if not isinstance(obj, list):
        raise ParseError(
            raw,
            f"expected a JSON array of AnnotatedValue entries, got {type(obj).__name__}",
        )

    entries: list[AnnotatedValue] = []
    bad: list[str] = []
    for i, entry in enumerate(obj):
        try:
            entries.append(AnnotatedValue.model_validate(entry))
        except (ValueError, TypeError) as e:
            bad.append(f"entry[{i}]: {e}")
    if bad:
        raise ParseError(
            raw,
            "some entries are not valid AnnotatedValues — fix their shape/fields:\n"
            + "\n".join(bad),
        )
    return entries


def _make_text_parse(
    content: str,
    allowed_blocks: set[tuple[int, int | None]],
) -> Callable[[str, ExecutionContext], list[AnnotatedValue]]:
    """Build the text-tier parse hook: shape-validate via `_parse_extract_response`,
    then verify every emitted cell appears verbatim in `content` (the source page text).
    A non-verbatim cell is a transcription/hallucination error the model should fix, so
    raise `ParseError` (→ `call()` re-prompts) listing the offenders. `content` is
    captured per call, so this is built fresh for each extract call."""

    def parse(raw: str, ctx: ExecutionContext) -> list[AnnotatedValue]:
        entries = _parse_extract_response(raw, ctx)
        violations: list[str] = []
        for e in entries:
            source_block = (e.source_block_page, e.source_block_index)
            if e.source_block_page is None or source_block not in allowed_blocks:
                violations.append(
                    f"{e.description!r}: source block {source_block!r} is not one of "
                    f"{sorted(allowed_blocks, key=lambda item: (item[0], item[1] is None, item[1] or -1))!r}"
                )
            for path, v in _cells_with_path(e):
                if not _cell_in_text(v, content):
                    loc = "/".join(path) if path else e.description
                    violations.append(f"{e.description!r} [{loc}] = {v!r}")
        if violations:
            raise ParseError(
                raw,
                "these values do NOT appear verbatim in the page text — extract only "
                "printed values, transcribing every digit exactly:\n"
                + "\n".join(violations),
            )
        return entries

    return parse


def _make_block_parse(
    allowed_blocks: set[tuple[int, int | None]],
) -> Callable[[str, ExecutionContext], list[AnnotatedValue]]:
    """Build a parse hook that requires every value to name one selected block."""

    def parse(raw: str, ctx: ExecutionContext) -> list[AnnotatedValue]:
        entries = _parse_extract_response(raw, ctx)
        violations = []
        for e in entries:
            source_block = (e.source_block_page, e.source_block_index)
            if e.source_block_page is None or source_block not in allowed_blocks:
                violations.append(
                    f"{e.description!r}: source block {source_block!r} is not one of "
                    f"{sorted(allowed_blocks, key=lambda item: (item[0], item[1] is None, item[1] or -1))!r}"
                )
        if violations:
            raise ParseError(
                raw,
                "each entry must identify the selected block that supplied it:\n"
                + "\n".join(violations),
            )
        return entries

    return parse


def _stamp_provenance(
    entries: list[AnnotatedValue],
    refs: list[PageRef],
    branch: RetrieveBranch,
) -> list[AnnotatedValue]:
    """Copy machine-fact provenance from the source refs + branch onto each entry —
    never LLM-written. `bulletin`/`pages` are attributable only when every ref in the
    call shares one bulletin month (otherwise we can't tell which issue a value came
    from, so they're left empty). Branch fields (`as_of`/`period`/`key`) are call-level
    and always stamped. The model is frozen, so we rebuild via `model_copy`."""
    months = {r.month for r in refs if r.month}
    bulletin = next(iter(months)) if len(months) == 1 else None
    pages = (
        tuple(sorted({r.page for r in refs if r.page is not None}))
        if bulletin is not None
        else ()
    )
    return [
        e.model_copy(
            update={
                "bulletin": bulletin,
                "pages": pages,
                "as_of": branch.as_of,
                "requested_period": branch.period,
                "retrieve_key": branch.key,
            }
        )
        for e in entries
    ]


_PROVENANCE_FIELDS = (
    "bulletin",
    "pages",
    "as_of",
    "requested_period",
    "retrieve_key",
    "source_block_page",
    "source_block_index",
)


def _carry_provenance(
    corrected: list[AnnotatedValue], originals: list[AnnotatedValue]
) -> list[AnnotatedValue]:
    """Copy machine provenance from each original entry onto its confirm-round replacement.
    The confirm model only re-reads digits and returns semantic fields, so the parsed reply
    carries default (empty) provenance — restore the original's, which confirm never changes.
    `corrected` is position-aligned with `originals` (the parse hook guarantees this)."""
    return [
        c.model_copy(update={f: getattr(o, f) for f in _PROVENANCE_FIELDS})
        for c, o in zip(corrected, originals)
    ]


def _cells_with_path(entry: AnnotatedValue) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield (key_path, primitive_cell) for every cell in `entry.value`."""
    if entry.kind == "scalar":
        yield (), entry.value
    elif entry.kind == "vector":
        for k, v in entry.value.items():
            yield (k,), v
    elif entry.kind == "table":
        for rk, row in entry.value.items():
            for ck, v in row.items():
                yield (rk, ck), v


def _cell_in_text(value: int | float | str, text: str) -> bool:
    """True if primitive `value` appears verbatim in `text`. Integer-valued numerics
    also try the comma-formatted form (2582 → "2,582"); strings are case-insensitive."""
    if isinstance(value, str):
        return value.strip().lower() in text.lower()
    candidates: set[str] = {str(value)}
    is_int_valued = isinstance(value, int) or (
        isinstance(value, float) and value == int(value)
    )
    if is_int_valued:
        iv = int(value)
        candidates.add(str(iv))
        if abs(iv) >= 1000:
            candidates.add(f"{iv:,}")
    return any(c in text for c in candidates)


# Shared envelope spec (shape + field semantics + output rules) appended to each
# extract tier's system prompt (after the tier's own `_PREAMBLE`).
EXTRACT_COMMON_PROMPT = """\
## AnnotatedValue shape

A single JSON ARRAY of entries, one per distinct datum (or [] if nothing
relevant is found). Pick the shape that best preserves the page structure:
- scalar  when the page has a single relevant value for one period
- vector  when the row spans multiple years/periods and several may
          be needed (e.g. a time-series row); key by year/period label
- table   when both rows and columns vary

  scalar: {"description":"...","kind":"scalar",
           "value":<num|str>,"unit":"...",
           "source_block_page":<int>,"source_block_index":<int|null>}
  vector: {"description":"...","kind":"vector",
           "index_name":"<dim>",
           "value":{"<index>":<value>,...},"unit":"...",
           "source_block_page":<int>,"source_block_index":<int|null>}
  table:  {"description":"...","kind":"table",
           "row_name":"<dim>","col_name":"<dim>",
           "value":{"<row>":{"<col>":<value>,...},...},"unit":"...",
           "source_block_page":<int>,"source_block_index":<int|null>}

Cells should be simple number or string — no nested cells.

Transcribe numbers exactly as printed — every digit and decimal place;
never round, truncate, or drop trailing digits.

## Field semantics

description   natural-language label that uniquely identifies the
              datum (series + period + sub-category + any other
              distinguishing context). For the label text, use the
              page's verbatim row text / column header / caption phrase
              so the downstream consumer can map it back to the page.
              
index_name    (vector only) name of the varying dimension.

row_name /col_name      (table only) names of the two varying dimensions.

unit          natural-language label for the printed scale and base,
              e.g. "millions of dollars", "percent", "year". Match
              what the page prints. Leave blank ("") if the value is
              not a measurement (e.g. a name or other string answer).
              Every cell in a vector/table shares one unit — apply any
              conversion once over the whole payload, never cell-by-cell.

source_block_page / source_block_index
              copy the page and index from the single `[block page=... index=...]`
              metadata entry that supplied this value. Use JSON null for a
              whole-page block whose index is shown as null.
"""


# Each tier's system prompt is its own `_PREAMBLE` followed by the shared
# `EXTRACT_COMMON_PROMPT`; both tiers parse with `_parse_extract_response` and share
# the JSON-array output instruction below.
_EXTRACT_OUTPUT_INSTRUCTION = (
    "Output a JSON array of AnnotatedValue entries — no markdown fences, no prose."
)


class TextExtractor:
    _PREAMBLE = """\
You retrieve printed values from page text to fulfill a specific lookup.
Each user message describes the lookup — what to find and (when stated)
the period — followed by the full-context question this lookup supports,
then PAGE METADATA (a structured summary of each table/figure on the
page: its title, column/row labels, and a short description) and finally
the page text to draw values from. The page text is flattened, so its
columns and rows can be hard to read; use the metadata to make sense of
the layout — which column/row a value sits under, which table it belongs
to, what the period and units are. Emit one entry per distinct row that
could plausibly satisfy the lookup — including cases where multiple rows
partially match. Do not compute or transform — extract only what is
printed. Every numeric value emitted MUST appear in the PAGE TEXT
verbatim (the metadata is context, not a source of values). Choose the
AnnotatedValue shape (scalar / vector / table) that fits the data on the
page; pick the smallest shape that captures every relevant value.
A period written `YYYY-MM..YYYY-MM` is an INCLUSIVE range: extract every
month from the first endpoint through the last, both endpoints included."""

    _SYSTEM = _PREAMBLE + "\n\n" + EXTRACT_COMMON_PROMPT

    @staticmethod
    def _fetch_page_texts(
        refs: list[PageRef], ctx: ExecutionContext
    ) -> list[tuple[PageRef, str]]:
        """Fetch each ref's text from the page store (an anchor's text is its merged member
        pages, with any figure note already baked in); skip refs with none (the vision tier can
        still read them)."""
        store = get_page_store(str(ctx.config.pdf_dir))
        pages: list[tuple[PageRef, str]] = []
        for ref in refs:
            text = store.text(ref)
            if not text:
                ctx.emit(f"no_text tier=parsed_json page={str(ref)}")
                continue
            ctx.emit(f"got_text tier=parsed_json page={str(ref)} chars={len(text)}")
            pages.append((ref, text))
        return pages

    @staticmethod
    def _block_meta_line(page: int | None, block_index: int | None, block: Any) -> str:
        """One metadata line for a content block — title, kind, column/row labels, and the
        block summary (NO numeric values) — to help the model read the flattened page text."""
        parts = [f"{block.kind}: {block.title or '(untitled)'}"]
        if block.column_headers:
            parts.append(f"columns: {', '.join(block.column_headers)}")
        if block.row_headers:
            parts.append(f"rows: {', '.join(block.row_headers)}")
        if block.summary:
            parts.append(block.summary)
        index = "null" if block_index is None else str(block_index)
        return f"- [block page={page} index={index}] " + " | ".join(parts)

    @classmethod
    def _page_metadata(cls, refs: list[PageRef], ctx: ExecutionContext) -> str:
        """Structured summary of every content block on the group's pages. Empty string when no
        catalog metadata is available."""
        store = get_page_store(str(ctx.config.pdf_dir))
        lines: list[str] = []
        for ref in refs:
            row = store.catalog_row(ref)
            if row is None:
                continue
            for block in row.content_blocks:
                lines.append(cls._block_meta_line(ref.page, None, block))
        return "\n".join(lines)

    async def _extract_content(
        self,
        content: str,
        prov_refs: list[PageRef],
        metadata: str,
        allowed_blocks: set[tuple[int, int | None]],
        branch: RetrieveBranch,
        question: str,
        ctx: ExecutionContext,
    ) -> list[AnnotatedValue]:
        """Run one extraction call over `content` (whole-page text OR a block slice), verify
        every emitted cell appears in `content`, and stamp provenance from `prov_refs`. `content`
        is kept on its own message line so the verifier checks emitted cells against the source
        text, not the prompt scaffolding."""
        user_msg = "\n\n".join(
            [
                f"You are looking for {branch.key}{f' for the period {branch.period}' if branch.period else ''}.",
                f'For full context, this lookup serves to help answer the question: "{question}"',
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
        # Parse hook validates shape AND verbatim-checks every cell against `content`;
        # a defect raises ParseError, so `call()` re-prompts (escalating temperature)
        # before giving up. On exhaustion the group degrades to empty and the operator
        # falls through to the vision tier.
        prompt: PromptedCall[list[AnnotatedValue]] = PromptedCall(
            name="extract.text",
            system_prompt=self._SYSTEM,
            default_effort="medium",
            parse=_make_text_parse(content, allowed_blocks),
            output_instruction=_EXTRACT_OUTPUT_INSTRUCTION,
        )
        try:
            parsed = await prompt.call(ctx, user_msg, temperature=0.0)
        except ParseError as e:
            ctx.emit(f"extract_parse_failed tier=parsed_json error={e.detail!r}")
            return []
        ctx.emit(f"extracted tier=parsed_json n_entries={len(parsed)}")
        return _stamp_provenance(parsed, prov_refs, branch)

    @staticmethod
    def _block_groups(
        blocks: list[BlockRef],
    ) -> list[tuple[PageRef, list[PageRef], list[int | None]]]:
        """Group blocks by their anchor page (first-seen order), collecting each page's chosen
        `block_index`es. Returns `(anchor, member_refs, block_idxs)` per page — several blocks on
        one page collapse to one group (one extract call). `member_refs` is the UNION of those
        blocks' spans (each block carries its own anchor + table-merge `extra_pages`), so the call
        feeds every page they touch. A whole-page block contributes `block_index=None`."""
        order: list[PageRef] = []
        idxs_by: dict[PageRef, list[int | None]] = {}
        refs_by: dict[PageRef, list[PageRef]] = {}
        for b in blocks:
            if b.page not in idxs_by:
                idxs_by[b.page] = []
                refs_by[b.page] = []
                order.append(b.page)
            idxs_by[b.page].append(b.block_index)
            for r in b.member_refs:
                if r not in refs_by[b.page]:
                    refs_by[b.page].append(r)
        return [(p, refs_by[p], idxs_by[p]) for p in order]

    async def _extract_block_group(
        self,
        anchor: PageRef,
        member_refs: list[PageRef],
        block_idxs: list[int | None],
        branch: RetrieveBranch,
        question: str,
        ctx: ExecutionContext,
    ) -> list[AnnotatedValue]:
        """Extraction for one anchor page's blocks. The updated page index resolves a block to at
        most two physical pages (its anchor + one table-merge continuation), so we feed those pages'
        FULL text — no within-page slicing — annotated with the SELECTED blocks' metadata to point
        the read at the right table(s). Whole-page blocks (`block_index=None`, golden / search-agent)
        carry no specific block, so they fall back to the page's full metadata. `member_refs` is the
        pages (the group's blocks' spans, unioned and deduped)."""
        pages = self._fetch_page_texts(member_refs, ctx)
        if not pages:
            ctx.emit(
                f"group_skipped tier=parsed_json reason=no_text refs={[str(r) for r in member_refs]!r}"
            )
            return []
        content = "\n\n".join(text for _, text in pages)
        prov_refs = [r for r, _ in pages]
        row = get_page_store(str(ctx.config.pdf_dir)).catalog_row(anchor)
        specific = [
            bi
            for bi in block_idxs
            if bi is not None and row is not None and 0 <= bi < len(row.content_blocks)
        ]
        # Focused block metadata when every block is specific; otherwise (a whole-page block in the
        # group) fall back to the pages' full metadata.
        if specific and len(specific) == len(block_idxs):
            metadata = "\n".join(
                self._block_meta_line(anchor.page, bi, row.content_blocks[bi])  # type: ignore[union-attr]
                for bi in specific
            )
            allowed_blocks = {(int(anchor.page), bi) for bi in specific}
        else:
            metadata = self._page_metadata(prov_refs, ctx)
            allowed_blocks = {(int(anchor.page), None)}
        ctx.emit(
            f"block_scoped page={str(anchor)} n_pages={len(pages)} "
            f"n_blocks={len(block_idxs)} chars={len(content)}"
        )
        return await self._extract_content(
            content, prov_refs, metadata, allowed_blocks, branch, question, ctx
        )

    async def run(
        self,
        question: str,
        branch: RetrieveBranch,
        blocks: list[BlockRef],
        ctx: ExecutionContext,
    ) -> list[AnnotatedValue]:
        """Extract from the selected blocks, one extract call per anchor page (its blocks'
        member pages fed whole). Whole-page blocks (`block_index=None`, from golden / search-agent)
        flow through the same path — just with no specific block to focus on."""
        groups = self._block_groups(blocks)
        ctx.emit(
            f"fan_out tier=parsed_json n_groups={len(groups)} n_blocks={len(blocks)} "
            f"group_sizes={[len(idxs) for _, _, idxs in groups]}"
        )
        per_group = await asyncio.gather(
            *[
                self._extract_block_group(a, m, idxs, branch, question, ctx)
                for a, m, idxs in groups
            ]
        )
        entries = [e for kept in per_group for e in kept]
        if not entries:
            ctx.emit("tier_empty tier=parsed_json reason=no_entries")
        return entries


class VisionExtractor:
    _PREAMBLE = """\
You retrieve visible values from rendered page images to fulfill a
specific lookup. Each user message describes the lookup — what to find
and (when stated) the period — followed by the full-context question
this lookup supports, then a numbered list identifying each attached
image. The page images themselves arrive as attachments in the same
order as the list. Emit every visible value that could plausibly satisfy
the lookup. Do not compute or transform — extract only what is visible.
Every printed numeric value emitted MUST be visibly printed on the page.
The only exception is when the question asks for visual understanding
(e.g., count of bars exceeding a threshold). Choose the AnnotatedValue
shape (scalar / vector / table) that fits the data on the page.
A period written `YYYY-MM..YYYY-MM` is an INCLUSIVE range: extract every
month from the first endpoint through the last, both endpoints included."""

    async def run(
        self,
        question: str,
        branch: RetrieveBranch,
        blocks: list[BlockRef],
        images: list[B64Image],
        rendered_refs: list[PageRef],
        ctx: ExecutionContext,
    ) -> list[AnnotatedValue]:
        """One T=0 vision call over the rendered page images. The numbered image list
        maps each attachment back to its source page so the LLM can't conflate them."""
        period = f" for the period {branch.period}" if branch.period else ""
        image_lines = [
            f"Image {i + 1}: PDF page {ref.page} of the {ref.month} Treasury Bulletin"
            for i, ref in enumerate(rendered_refs)
        ]
        block_lines = []
        allowed_blocks: set[tuple[int, int | None]] = set()
        for block_ref in blocks:
            page = int(block_ref.page.page)
            allowed_blocks.add((page, block_ref.block_index))
            if block_ref.block is None:
                block_lines.append(f"- [block page={page} index=null] whole page")
            else:
                block_lines.append(
                    TextExtractor._block_meta_line(
                        page,
                        block_ref.block_index,
                        block_ref.block,
                    )
                )
        user_msg = "\n\n".join(
            [
                f"You are looking for {branch.key}{period}.",
                f'For full context, this lookup serves to help answer the question: "{question}"',
                "Images attached, in order:\n" + "\n".join(image_lines),
                "Selected blocks:\n" + "\n".join(block_lines),
            ]
        )
        ctx.emit(f"vision_call tier=vision n_images={len(images)}")
        prompt: PromptedCall[list[AnnotatedValue]] = PromptedCall(
            name="extract.vision",
            system_prompt=self._PREAMBLE + "\n\n" + EXTRACT_COMMON_PROMPT,
            default_effort="medium",
            parse=_make_block_parse(allowed_blocks),
            output_instruction=_EXTRACT_OUTPUT_INSTRUCTION,
        )
        try:
            entries = await prompt.call(
                ctx, user_msg, images=images, temperature=0.0
            )
        except ParseError as e:
            ctx.emit(f"extract_parse_failed tier=vision error={e.detail!r}")
            entries = []
        ctx.emit(f"vision_result tier=vision n_entries={len(entries)}")
        # Stamp provenance from the rendered refs. A single vision call may span
        # several issues (no per-image attribution on the reply), so bulletin/pages
        # land only when all images share one bulletin — the common as_of/single-issue
        # branch; multi-issue calls keep bulletin empty.
        return _stamp_provenance(entries, rendered_refs, branch)


def _entry_semantic_dict(e: AnnotatedValue) -> dict[str, Any]:
    """The shape/value fields of an entry (no machine provenance) for the confirm prompt —
    what the model needs to locate the cell and re-read its digits."""
    d: dict[str, Any] = {"description": e.description, "kind": e.kind, "value": e.value}
    if e.unit:
        d["unit"] = e.unit
    for name in ("index_name", "row_name", "col_name"):
        v = getattr(e, name)
        if v is not None:
            d[name] = v
    d["source_block_page"] = e.source_block_page
    d["source_block_index"] = e.source_block_index
    return d


def _key_diff(orig_keys: list[str], got_keys: list[str], label: str) -> list[str]:
    """Set-difference between two key lists (order-independent — reordering is harmless)."""
    missing = sorted(set(orig_keys) - set(got_keys))
    added = sorted(set(got_keys) - set(orig_keys))
    msgs: list[str] = []
    if missing:
        msgs.append(f"{label}: dropped {missing!r}")
    if added:
        msgs.append(f"{label}: added {added!r}")
    return msgs


def _confirm_structure_diff(orig: AnnotatedValue, got: AnnotatedValue) -> list[str]:
    """Structural differences between a confirm-round entry and its original — everything
    except the primitive cell VALUES. An empty list means a legitimate digits-only
    correction. `got` is already shape-valid (the caller ran `_parse_extract_response`),
    so its `value` matches its `kind`."""
    diffs: list[str] = []
    if got.description != orig.description:
        diffs.append(f"description changed to {got.description!r}")
    if got.kind != orig.kind:
        # Once kind differs the value shapes aren't comparable; report just that.
        return diffs + [f"kind changed {orig.kind} -> {got.kind}"]
    if got.unit != orig.unit:
        diffs.append(f"unit changed {orig.unit!r} -> {got.unit!r}")
    for name in ("index_name", "row_name", "col_name"):
        if getattr(got, name) != getattr(orig, name):
            diffs.append(
                f"{name} changed {getattr(orig, name)!r} -> {getattr(got, name)!r}"
            )
    for name in ("source_block_page", "source_block_index"):
        if getattr(got, name) != getattr(orig, name):
            diffs.append(
                f"{name} changed {getattr(orig, name)!r} -> {getattr(got, name)!r}"
            )
    ov, gv = orig.value, got.value
    if orig.kind == "scalar":
        o_list, g_list = isinstance(ov, list), isinstance(gv, list)
        if o_list != g_list:
            diffs.append(
                f"scalar shape changed ({'list' if o_list else 'single'} -> "
                f"{'list' if g_list else 'single'})"
            )
        elif o_list and len(ov) != len(gv):
            diffs.append(f"scalar list length changed {len(ov)} -> {len(gv)}")
    elif orig.kind == "vector":
        diffs.extend(_key_diff(list(ov), list(gv), "vector keys"))
    else:  # table
        diffs.extend(_key_diff(list(ov), list(gv), "table rows"))
        for rk in ov:
            if rk in gv:
                diffs.extend(_key_diff(list(ov[rk]), list(gv[rk]), f"row {rk!r} cols"))
    return diffs


def _make_confirm_parse(
    originals: list[AnnotatedValue],
) -> Callable[[str, ExecutionContext], list[AnnotatedValue]]:
    """Build the confirm-round parse hook: shape-validate via `_parse_extract_response`,
    then enforce that the reply is a digits-only correction of `originals` — the same
    number of entries, in the same order, each preserving its description, kind, unit,
    dimension names, and the keys/shape of its value. Only primitive cell values may
    differ. Any structural drift is a spec violation the model should fix, so raise
    `ParseError` (→ `call()` re-prompts) naming the offending entries. `originals` is
    captured per call, so this is built fresh for each confirm round."""

    def parse(raw: str, ctx: ExecutionContext) -> list[AnnotatedValue]:
        entries = _parse_extract_response(raw, ctx)
        if len(entries) != len(originals):
            raise ParseError(
                raw,
                f"returned {len(entries)} entries but exactly {len(originals)} were given "
                "— return the same entries in the same order, correcting only digits",
            )
        violations: list[str] = []
        for i, (orig, got) in enumerate(zip(originals, entries)):
            for msg in _confirm_structure_diff(orig, got):
                violations.append(f"entry[{i}] {orig.description!r}: {msg}")
        if violations:
            raise ParseError(
                raw,
                "these entries changed something other than digits — keep each entry's "
                "description, kind, unit, dimension names, and value keys/shape identical, "
                "fixing only the digits inside cell values:\n" + "\n".join(violations),
            )
        return entries

    return parse


class VisualValidator:
    """Vision confirmation round. The parsed-text (OCR) tier can corrupt digits; this re-reads
    each emitted value off the rendered page images and corrects only what the image plainly
    contradicts. Frames the task as correction (not approval) and forces a fresh read before
    comparison, to blunt the natural confirmation bias of a check-this pass."""

    _PREAMBLE = """\
The values below were transcribed from page TEXT that may contain OCR
errors (misread digits, dropped decimals, misaligned rows/columns). The rendered page images are
the GROUND TRUTH. Correct OCR digit mistakes — and nothing else.

Each user message gives a numbered list identifying each attached image
(and its source page), and a JSON array of the transcribed values. The
images arrive as attachments in that order.

For EACH entry, locate its cell(s) on the image by description / row +
column label / period, read the digits, and overwrite only the digits the
image contradicts. Transcribe numbers exactly as printed — every digit and
decimal place. If a transcribed value already matches the image, return it
unchanged.

You may ONLY change digits inside cell values. Do NOT change any description, kind, unit,
index_name / row_name / col_name, source_block_page / source_block_index, or the keys or
shape of any value."""

    _SYSTEM = _PREAMBLE + "\n\n" + EXTRACT_COMMON_PROMPT

    async def run(
        self,
        question: str,
        branch: RetrieveBranch,
        blocks: list[BlockRef],
        entries: list[AnnotatedValue],
        ctx: ExecutionContext,
    ) -> list[AnnotatedValue]:
        """Re-read `entries` against their source page images and return the corrected set.
        Fans out ONE confirm call per source page(-group): each call shows only the page(s)
        a subset of entries came from and re-reads only those entries — mirroring the text
        tier's per-page extraction, so the model never has to map an entry across unrelated
        page images. Conservative per group: a group whose page won't render, or whose reply
        is empty/unparseable, passes its entries through unchanged."""
        groups = self._group_by_source(entries, blocks)
        ctx.emit(f"confirm_fan_out n_groups={len(groups)} n_entries={len(entries)}")
        results = await asyncio.gather(
            *[
                self._confirm_group(question, branch, refs, members, ctx)
                for refs, members in groups
            ]
        )
        # Scatter each group's corrected entries back to their original positions.
        out = list(entries)
        for (_, members), corrected in zip(groups, results):
            for (idx, _), new in zip(members, corrected):
                out[idx] = new
        return out

    @staticmethod
    def _group_by_source(
        entries: list[AnnotatedValue], blocks: list[BlockRef]
    ) -> list[tuple[list[PageRef], list[tuple[int, AnnotatedValue]]]]:
        """Partition `(index, entry)` pairs by the entry's stamped source page(s) — the unit
        a single confirm call re-reads. Entries lacking page provenance (`pages=()`) share one
        fallback group rendered against all retrieved pages (best-effort, the old behavior)."""
        fallback = tuple(_blocks_to_pagerefs(blocks))
        groups: dict[tuple[PageRef, ...], list[tuple[int, AnnotatedValue]]] = {}
        for i, e in enumerate(entries):
            if e.pages and e.bulletin:
                refs = tuple(PageRef(month=e.bulletin, page=p) for p in e.pages)
            else:
                refs = fallback
            groups.setdefault(refs, []).append((i, e))
        return [(list(refs), members) for refs, members in groups.items()]

    async def _confirm_group(
        self,
        question: str,
        branch: RetrieveBranch,
        refs: list[PageRef],
        members: list[tuple[int, AnnotatedValue]],
        ctx: ExecutionContext,
    ) -> list[AnnotatedValue]:
        """Confirm one page-group: render `refs`, re-read just these entries off them, and
        return the corrected entries (aligned with `members`). Passes the entries through
        unchanged on any render/parse failure."""
        originals = [e for _, e in members]
        images, rendered_refs = _render_pages_b64(refs, ctx)
        if not images:
            ctx.emit("confirm_skipped reason=no_images")
            return originals
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
                f'For full context, this lookup serves to help answer the question: "{question}"',
                "Images attached, in order:\n" + "\n".join(image_lines),
                "Transcribed values to confirm:\n" + payload,
            ]
        )
        ctx.emit(f"confirm_call n_images={len(images)} n_entries={len(originals)}")
        # Per-call prompt: the parse hook enforces a digits-only correction of THESE entries
        # (same count/order/structure), so the reply aligns position-for-position with them.
        prompt: PromptedCall[list[AnnotatedValue]] = PromptedCall(
            name="extract.confirm",
            system_prompt=self._SYSTEM,
            default_effort="medium",
            parse=_make_confirm_parse(originals),
            output_instruction=_EXTRACT_OUTPUT_INSTRUCTION,
        )
        try:
            corrected = await prompt.call(ctx, user_msg, images=images, temperature=0.0)
        except ParseError as e:
            ctx.emit(f"confirm_kept_original reason=parse_failed error={e.detail!r}")
            return originals
        if not corrected:
            ctx.emit("confirm_kept_original reason=empty_reply")
            return originals
        # Confirm never changes provenance — carry the originals' onto the digit-corrected copies.
        corrected = _carry_provenance(corrected, originals)
        n_changed = 0
        for orig, e in zip(originals, corrected):
            if orig.value != e.value:
                n_changed += 1
                ctx.emit(
                    f"confirm_changed description={e.description!r} "
                    f"before={orig.value!r} after={e.value!r}"
                )
        ctx.emit(
            f"confirm_done n_changed={n_changed} n_in={len(originals)} n_out={len(corrected)}"
        )
        return corrected


class ExtractOp:
    """The extract operator — question-driven extraction. Owns one instance of
    each call-site extractor and drives the parsed_json → vision tier fallback, with a
    vision confirmation round over the parsed_json tier's output."""

    def __init__(self) -> None:
        self._text = TextExtractor()
        self._vision = VisionExtractor()
        self._confirm = VisualValidator()

    async def run(
        self,
        blocks: list[BlockRef],
        ctx: ExecutionContext,
        branch: RetrieveBranch,
    ) -> list[AnnotatedValue]:
        """Extract from the branch's retrieved blocks. The text tier feeds each block's pages whole
        (scoped by the selected blocks' metadata); the vision tier renders those same pages."""
        if not blocks:
            raise StepFailed("extract", "No blocks to extract from")

        # parsed_json tier first (skipped for visual_only charts/figures, or entirely when
        # `extract_vision_only` forces straight-to-vision); fall through to the vision tier
        # when it finds nothing.
        if not branch.visual_only and not ctx.config.extract_vision_only:
            entries = await self._text.run(ctx.question, branch, blocks, ctx)
            if entries:
                # Vision confirmation round — correct OCR digit errors against the rendered pages.
                entries = await self._confirm.run(
                    ctx.question, branch, blocks, entries, ctx
                )
                ctx.emit(
                    f"tier_result tier=parsed_json descriptions={[e.description for e in entries]!r}"
                )
                return entries

        # vision tier — render the blocks' pages, then read values off the images. strict: a page
        # that won't render must fail the branch (→ replan), not be silently dropped.
        images, rendered_refs = _render_pages_b64(
            _blocks_to_pagerefs(blocks), ctx, strict=True
        )
        entries = await self._vision.run(
            ctx.question, branch, blocks, images, rendered_refs, ctx
        )
        if not entries:
            raise StepFailed("extract", "no relevant values found across tiers")
        ctx.emit(
            f"tier_result tier=vision descriptions={[e.description for e in entries]!r}"
        )
        return entries
