from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

from skunk.common import (
    AnnotatedValue,
    B64Image,
    BlockRef,
    ExecutionContext,
    PageRef,
    parse_json_response,
)
from skunk.errors import StepFailed
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
) -> tuple[list[B64Image], list[PageRef]]:
    """Page images for the vision tier, from the page store (rendered on demand at 200 DPI +
    cached). The returned refs identify each image's source page so the prompt's numbered list
    can't conflate them."""
    store = get_page_store(str(ctx.config.pdf_dir))
    images: list[B64Image] = []
    rendered_refs: list[PageRef] = []
    for ref in refs:
        try:
            img = store.image(ref)
        except Exception as e:  # noqa: BLE001 — vision tier fallback; any render error → skip page
            ctx.emit(f"render_failed page={str(ref)} error={str(e)!r}")
            continue
        if img is None:
            ctx.emit(f"no_png page={str(ref)}")
            continue
        ctx.emit(f"rendered_png page={str(ref)}")
        images.append(img)
        rendered_refs.append(ref)
    return images, rendered_refs


def _parse_extract_response(
    raw: str, ctx: ExecutionContext
) -> list[AnnotatedValue] | None:
    """Parse an LLM response into AnnotatedValues. Returns None on unparseable
    / non-array / empty (caller falls through to the next tier); drops individual
    entries that fail validation with a diagnostic."""
    obj = parse_json_response(raw)
    if obj is None:
        ctx.emit(f"rejected_unparseable raw={raw!r}")
        return None
    if not isinstance(obj, list):
        ctx.emit(f"rejected_non_array got={type(obj).__name__} raw={raw!r}")
        return None
    if not obj:
        return None

    entries: list[AnnotatedValue] = []
    for i, entry in enumerate(obj):
        try:
            entries.append(AnnotatedValue.model_validate(entry))
        except (ValueError, TypeError) as e:
            ctx.emit(f"rejected_entry entry_idx={i} reason={str(e)!r}")
    return entries


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
           "value":<num|str>,"unit":"..."}
  vector: {"description":"...","kind":"vector",
           "index_name":"<dim>",
           "value":{"<index>":<value>,...},"unit":"..."}
  table:  {"description":"...","kind":"table",
           "row_name":"<dim>","col_name":"<dim>",
           "value":{"<row>":{"<col>":<value>,...},...},"unit":"..."}

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
page; pick the smallest shape that captures every relevant value."""

    _prompt = PromptedCall(
        name="extract.text",
        system_prompt=_PREAMBLE + "\n\n" + EXTRACT_COMMON_PROMPT,
        default_effort="off",
        parse=_parse_extract_response,
        output_instruction=_EXTRACT_OUTPUT_INSTRUCTION,
    )

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
    def _block_meta_line(page: int | None, block: Any) -> str:
        """One metadata line for a content block — title, kind, column/row labels, and the
        block summary (NO numeric values) — to help the model read the flattened page text."""
        parts = [f"{block.kind}: {block.title or '(untitled)'}"]
        if block.column_headers:
            parts.append(f"columns: {', '.join(block.column_headers)}")
        if block.row_headers:
            parts.append(f"rows: {', '.join(block.row_headers)}")
        if block.summary:
            parts.append(block.summary)
        return f"- page {page}: " + " | ".join(parts)

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
                lines.append(cls._block_meta_line(ref.page, block))
        return "\n".join(lines)

    async def _extract_content(
        self,
        content: str,
        prov_refs: list[PageRef],
        metadata: str,
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
        parsed = await self._prompt.call(ctx, user_msg, temperature=0.0) or []
        ctx.emit(f"extracted tier=parsed_json n_entries={len(parsed)}")
        kept = [
            e
            for e in parsed
            if all(_cell_in_text(v, content) for _, v in _cells_with_path(e))
        ]
        if len(kept) < len(parsed):
            ctx.emit(
                f"verifier_dropped tier=parsed_json n_dropped={len(parsed) - len(kept)} n_parsed={len(parsed)}"
            )
        return _stamp_provenance(kept, prov_refs, branch)

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
                self._block_meta_line(anchor.page, row.content_blocks[bi])  # type: ignore[union-attr]
                for bi in specific
            )
        else:
            metadata = self._page_metadata(prov_refs, ctx)
        ctx.emit(
            f"block_scoped page={str(anchor)} n_pages={len(pages)} "
            f"n_blocks={len(block_idxs)} chars={len(content)}"
        )
        return await self._extract_content(
            content, prov_refs, metadata, branch, question, ctx
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
shape (scalar / vector / table) that fits the data on the page."""

    _prompt = PromptedCall(
        name="extract.vision",
        system_prompt=_PREAMBLE + "\n\n" + EXTRACT_COMMON_PROMPT,
        default_effort="off",
        parse=_parse_extract_response,
        output_instruction=_EXTRACT_OUTPUT_INSTRUCTION,
    )

    async def run(
        self,
        question: str,
        branch: RetrieveBranch,
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
        user_msg = "\n\n".join(
            [
                f"You are looking for {branch.key}{period}.",
                f'For full context, this lookup serves to help answer the question: "{question}"',
                "Images attached, in order:\n" + "\n".join(image_lines),
            ]
        )
        ctx.emit(f"vision_call tier=vision n_images={len(images)}")
        entries = await self._prompt.call(ctx, user_msg, images=images, temperature=0.0)
        ctx.emit(
            f"vision_result tier=vision n_entries={0 if entries is None else len(entries)}"
        )
        # Stamp provenance from the rendered refs. A single vision call may span
        # several issues (no per-image attribution on the reply), so bulletin/pages
        # land only when all images share one bulletin — the common as_of/single-issue
        # branch; multi-issue calls keep bulletin empty.
        return _stamp_provenance(entries or [], rendered_refs, branch)


class ExtractOp:
    """The extract operator — question-driven extraction. Owns one instance of
    each call-site extractor and drives the parsed_json → vision tier fallback."""

    def __init__(self) -> None:
        self._text = TextExtractor()
        self._vision = VisionExtractor()

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
                ctx.emit(
                    f"tier_result tier=parsed_json descriptions={[e.description for e in entries]!r}"
                )
                return entries

        # vision tier — render the blocks' pages, then read values off the images.
        images, rendered_refs = _render_pages_b64(_blocks_to_pagerefs(blocks), ctx)
        entries = await self._vision.run(
            ctx.question, branch, images, rendered_refs, ctx
        )
        if not entries:
            raise StepFailed("extract", "no relevant values found across tiers")
        ctx.emit(
            f"tier_result tier=vision descriptions={[e.description for e in entries]!r}"
        )
        return entries
