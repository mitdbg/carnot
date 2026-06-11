from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Iterator
from typing import Any

from skunk.common import (
    AnnotatedValue,
    B64Image,
    BlockRef,
    ExecutionContext,
    PageRef,
    SemPoolEntry,
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

    # Distinguishability: two entries with identical (description, qualifiers) but
    # different values are unusable downstream — the consumer cannot tell which cell
    # is which (the page discriminator was dropped). Re-prompt with the offenders.
    by_label: dict[tuple[str, str], list[int]] = {}
    for i, e in enumerate(entries):
        by_label.setdefault((e.description, e.qualifiers), []).append(i)
    clashes = [
        idxs
        for idxs in by_label.values()
        if len(idxs) > 1
        and len({json.dumps(entries[i].value, sort_keys=True) for i in idxs}) > 1
    ]
    if clashes:
        raise ParseError(
            raw,
            "these entry groups are indistinguishable (same description AND qualifiers) "
            "yet hold different values — add the discriminating column header / year / "
            "footnote / table title to 'qualifiers':\n"
            + "\n".join(
                f"entries {idxs}: {entries[idxs[0]].description!r}" for idxs in clashes
            ),
        )
    return entries


def _make_text_parse(
    content: str,
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


_PROVENANCE_FIELDS = ("bulletin", "pages", "as_of", "requested_period", "retrieve_key")


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

  scalar: {"description":"...","qualifiers":"...","kind":"scalar",
           "value":<num|str>,"unit":"..."}
  vector: {"description":"...","qualifiers":"...","kind":"vector",
           "index_name":"<dim>",
           "value":{"<index>":<value>,...},"unit":"..."}
  table:  {"description":"...","qualifiers":"...","kind":"table",
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

qualifiers    the page's verbatim fragments that LOCATE and DISCRIMINATE
              this datum: the exact column header the value(s) sit under,
              the row label, any footnote markers on the value or its
              row/column (e.g. "2/", "p", "r"), the table title when
              several similar tables share the page. Copy the fragments
              verbatim; separate with " | ". Two entries reading
              different cells MUST differ in description or qualifiers.
              "" only when the page offers no such discriminators.

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
        # Parse hook validates shape AND verbatim-checks every cell against `content`;
        # a defect raises ParseError, so `call()` re-prompts (escalating temperature)
        # before giving up. On exhaustion the group degrades to empty and the operator
        # falls through to the vision tier.
        prompt: PromptedCall[list[AnnotatedValue]] = PromptedCall(
            name="extract.text",
            system_prompt=self._SYSTEM,
            default_effort="medium",
            parse=_make_text_parse(content),
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
shape (scalar / vector / table) that fits the data on the page.
A period written `YYYY-MM..YYYY-MM` is an INCLUSIVE range: extract every
month from the first endpoint through the last, both endpoints included."""

    _prompt = PromptedCall(
        name="extract.vision",
        system_prompt=_PREAMBLE + "\n\n" + EXTRACT_COMMON_PROMPT,
        default_effort="medium",
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
        try:
            entries = await self._prompt.call(
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
    if e.qualifiers:
        d["qualifiers"] = e.qualifiers
    for name in ("index_name", "row_name", "col_name"):
        v = getattr(e, name)
        if v is not None:
            d[name] = v
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
    if orig.qualifiers and got.qualifiers != orig.qualifiers:
        diffs.append(f"qualifiers changed {orig.qualifiers!r} -> {got.qualifiers!r}")
    for name in ("index_name", "row_name", "col_name"):
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
index_name / row_name / col_name, or the keys or shape of any value."""

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


def _truncate_keys(keys: list[str], cap: int = 30) -> str:
    shown = ", ".join(keys[:cap])
    extra = f", …(+{len(keys) - cap} more)" if len(keys) > cap else ""
    return f"[{shown}{extra}]"


def _entry_review_line(i: int, e: AnnotatedValue) -> str:
    """One compact line per entry for the review prompt: identity + provenance + the
    period-bearing structure (keys matter for coverage; values are capped)."""
    src = f"bulletin={e.bulletin or '?'} pages={list(e.pages) or '?'}"
    if e.kind == "scalar":
        body = f"value={e.value!r}"
    elif e.kind == "vector":
        v = e.value if isinstance(e.value, dict) else {}
        pairs = ", ".join(f"{k}: {c!r}" for k, c in list(v.items())[:12])
        extra = f", …(+{len(v) - 12} more)" if len(v) > 12 else ""
        body = f"index={e.index_name!r} value={{{pairs}{extra}}}"
    else:  # table
        v = e.value if isinstance(e.value, dict) else {}
        cols = sorted({c for row in v.values() for c in row}) if v else []
        body = (
            f"rows({e.row_name!r})={_truncate_keys(list(v))} "
            f"cols({e.col_name!r})={_truncate_keys(cols)}"
        )
    unit = f" unit={e.unit!r}" if e.unit else ""
    quals = f" qualifiers={e.qualifiers!r}" if e.qualifiers else ""
    return f"[{i}] description={e.description!r}{unit}{quals} kind={e.kind} {src} {body}"


def _make_review_parse(n_entries: int) -> Callable[[str, ExecutionContext], dict]:
    """Parse hook for the coverage review verdict. Validates shape and internal
    consistency; violations raise `ParseError` (→ `call()` re-prompts)."""

    def parse(raw: str, ctx: ExecutionContext) -> dict:
        obj = parse_json_response(raw)
        if not isinstance(obj, dict):
            raise ParseError(raw, "reply must be a single JSON object")
        problems: list[str] = []
        complete = obj.get("complete")
        missing = obj.get("missing")
        dups = obj.get("duplicates")
        if not isinstance(complete, bool):
            problems.append("'complete' must be a boolean")
        if not isinstance(missing, list) or not all(
            isinstance(m, dict)
            and isinstance(m.get("period"), str)
            and isinstance(m.get("what"), str)
            and m["what"].strip()
            for m in missing
        ):
            problems.append(
                '\'missing\' must be a list of {"period": str, "what": str} objects'
            )
        elif isinstance(complete, bool) and complete != (len(missing) == 0):
            problems.append("'complete' must be true if and only if 'missing' is empty")
        seen: set[int] = set()
        if not isinstance(dups, list):
            problems.append("'duplicates' must be a list")
        else:
            for g, grp in enumerate(dups):
                if not (
                    isinstance(grp, dict)
                    and isinstance(grp.get("keep"), int)
                    and isinstance(grp.get("drop"), list)
                    and all(isinstance(d, int) for d in grp["drop"])
                ):
                    problems.append(
                        f'duplicates[{g}] must be {{"keep": int, "drop": [int, ...], "why": str}}'
                    )
                    continue
                idxs = [grp["keep"], *grp["drop"]]
                bad = [i for i in idxs if not 0 <= i < n_entries]
                if bad:
                    problems.append(
                        f"duplicates[{g}]: entry indices {bad} out of range 0..{n_entries - 1}"
                    )
                if grp["keep"] in grp["drop"]:
                    problems.append(
                        f"duplicates[{g}]: 'keep' index also listed in 'drop'"
                    )
                overlap = seen.intersection(idxs)
                if overlap:
                    problems.append(
                        f"duplicates[{g}]: indices {sorted(overlap)} already in another group"
                    )
                seen.update(idxs)
        if problems:
            raise ParseError(raw, "fix these issues:\n" + "\n".join(problems))
        return {"complete": complete, "missing": missing, "duplicates": dups}

    return parse


class CoverageReview:
    """SHADOW coverage/duplicate audit of a branch's final extracted entries. One text-only
    call per branch: does the entry set cover every requested period at the implied
    granularity, and which entries duplicate one another (same series + period reprinted
    across issues)? The verdict is only emitted to the event stream — `run_shadow` never
    raises and never alters the entries, so it cannot influence the run."""

    _SYSTEM = """\
You audit the output of a data-extraction pass over U.S. Treasury Bulletin pages.

The user message gives one retrieval request: the data sought, the period(s) the data
must cover (comma-separated `YYYY-MM` months or inclusive `YYYY-MM..YYYY-MM` ranges;
absent = unpinned), optionally a pinned source issue (`as_of` = the bulletin the values
must be read from), the full question the request serves, and a numbered list of the
extracted entries (description, unit, shape, source issue/pages, values).

Decide two things:

1. COVERAGE. The entries must contain the requested values for EVERY requested period,
   at the granularity the request implies (monthly / quarterly / annual / single as-of
   date). List every requested period or value no entry provides.
   - A fiscal-year or annual figure does NOT satisfy a request for specific months.
   - A value from any issue counts — unless `as_of` pins the issue, then only entries
     from that issue count.
   - Flag only values the request needs. Do not invent nice-to-haves.
2. DUPLICATES. Group entries reporting the SAME series for the SAME period(s) — e.g. one
   table reprinted in consecutive issues. Per group keep ONE entry: the one from the
   latest bulletin, unless `as_of` pins an issue (then keep that issue's entry).
   Entries covering different periods of the same series are NOT duplicates. Singleton
   entries appear in no group."""

    async def run_shadow(
        self,
        question: str,
        branch: RetrieveBranch,
        entries: list[AnnotatedValue],
        ctx: ExecutionContext,
    ) -> tuple[list[AnnotatedValue], dict | None]:
        """Emit the review verdict for `entries`; swallow every failure. Returns
        `(entries, verdict)` — entries unchanged unless `extract_review_dedup` is on, in
        which case entries in any duplicate group's `drop` list are removed. `verdict` is
        None when the review itself failed (the caller treats that as no-op)."""
        try:
            verdict = await self._review(question, branch, entries, ctx)
        except ParseError as e:
            ctx.emit(f"review_parse_failed error={e.detail!r}")
            return entries, None
        except Exception as e:  # noqa: BLE001 — review must never break the branch
            ctx.emit(f"review_failed error={e!r}")
            return entries, None
        if not ctx.config.extract_review_dedup or not verdict["duplicates"]:
            return entries, verdict
        drop = {i for g in verdict["duplicates"] for i in g["drop"]}
        kept = [e for i, e in enumerate(entries) if i not in drop]
        ctx.emit(
            f"review_dedup_applied dropped={sorted(drop)} n_in={len(entries)} n_out={len(kept)} "
            f"dropped_descriptions={[entries[i].description for i in sorted(drop)]!r}"
        )
        return kept, verdict

    async def _review(
        self,
        question: str,
        branch: RetrieveBranch,
        entries: list[AnnotatedValue],
        ctx: ExecutionContext,
    ) -> dict:
        period = (
            f"\nPeriod(s) the data must cover: {branch.period}" if branch.period else ""
        )
        as_of = f"\nPinned source issue (as_of): {branch.as_of}" if branch.as_of else ""
        lines = "\n".join(_entry_review_line(i, e) for i, e in enumerate(entries))
        user_msg = "\n\n".join(
            [
                f"Retrieval request: {branch.key}{period}{as_of}",
                f'For full context, the request serves to help answer the question: "{question}"',
                f"Extracted entries ({len(entries)}):\n{lines}",
            ]
        )
        prompt: PromptedCall[dict] = PromptedCall(
            name="extract.review",
            system_prompt=self._SYSTEM,
            default_effort="low",  # span-coverage judgment is a reasoning task
            parse=_make_review_parse(len(entries)),
            output_instruction=(
                "Output ONLY a JSON object — no prose, no markdown fences:\n"
                '{"complete": <bool — true iff "missing" is empty>,\n'
                ' "missing": [{"period": "<YYYY-MM | YYYY-MM..YYYY-MM | short label>", '
                '"what": "<the missing value(s)>"}, ...],\n'
                ' "duplicates": [{"keep": <entry index>, "drop": [<entry indices>], '
                '"why": "<short>"}, ...]}'
            ),
        )
        verdict = await prompt.call(ctx, user_msg, temperature=0.0)
        ctx.emit(f"review_verdict {json.dumps(verdict, ensure_ascii=False)}")
        return verdict


# Coverage-repair bounds: one add-only round per branch, ≤_REPAIR_MAX_NEEDS flagged gaps
# consumed, ≤_REPAIR_KEEP_PER_NEED blocks selected per gap from a ≤_REPAIR_GROUP_SIZE
# prefiltered candidate group, and the branch's total block count (original + repair)
# capped at _REPAIR_MAX_TOTAL_BLOCKS (extract fan-out guard).
_REPAIR_MAX_NEEDS = 4
# 4, not 2: a monthly gap often spans several issues' ~3-month windows (e.g. Jan–Aug =
# three consecutive quarterly issues) — the pick must be able to TILE the gap.
_REPAIR_KEEP_PER_NEED = 4
_REPAIR_GROUP_SIZE = 32
_REPAIR_MAX_TOTAL_BLOCKS = 16

_NEED_MONTH_RE = re.compile(r"\b(\d{4})-(\d{2})\b")


def _need_interval(period: str) -> tuple[str, str] | None:
    """The [lo, hi] month span of a review need's `period` text ("YYYY-MM",
    "YYYY-MM..YYYY-MM", or a free label containing such tokens); None if no month parses."""
    months = [f"{y}-{m}" for y, m in _NEED_MONTH_RE.findall(period)]
    return (min(months), max(months)) if months else None


def _month_idx(month: str) -> int:
    return int(month[:4]) * 12 + int(month[5:7])


def _block_key(ref: BlockRef) -> tuple[str | None, int | None, int | None]:
    return (ref.page.month, ref.page.page, ref.block_index)


def _repair_prefilter(
    pool: list[SemPoolEntry],
    period: str,
    exclude: set[tuple[str | None, int | None, int | None]],
    cap: int = _REPAIR_GROUP_SIZE,
) -> list[SemPoolEntry]:
    """Deterministic candidate narrowing for one need: drop already-used blocks, keep
    interval-overlap with the gap (no-interval entries pass — recall side), and rank by
    bulletin-month proximity to the gap, preferring issues at/after the gap start (data for
    period P prints in issues shortly after P — the boundary-vintage rule)."""
    span = _need_interval(period)
    cands = [
        e
        for e in pool
        if _block_key(e.ref) not in exclude
        and not (
            span and e.interval and (e.interval[1] < span[0] or e.interval[0] > span[1])
        )
    ]
    if span:
        lo, hi = _month_idx(span[0]), _month_idx(span[1])

        def rank(e: SemPoolEntry) -> tuple[int, int]:
            m = e.ref.page.month
            mi = _month_idx(m) if m else 1 << 30
            return (0 if mi >= lo else 1, abs(mi - hi))

        cands.sort(key=rank)
    return cands[:cap]


async def _repair_select(
    ctx: ExecutionContext,
    question: str,
    branch: RetrieveBranch,
    period: str,
    what: str,
    cands: list[SemPoolEntry],
) -> list[SemPoolEntry]:
    """One boolean-array selection call over the prefiltered candidates for one need.
    Reuses the block-select system prompt; the user message states what the coverage
    review found missing so the pick targets the gap, not the broad branch request."""
    from skunk.page_index.query import PageIndexRetriever  # lazy, mirrors retrieve.py

    lines = []
    for i, e in enumerate(cands):
        dates = f"{e.interval[0]}..{e.interval[1]}" if e.interval else "none"
        line = f"[{i}] dates={dates} | {e.kind} with title: {e.title or '(untitled)'}"
        if e.cols:
            line += f" [cols: {', '.join(e.cols)}]"
        if e.rows_tail:
            line += f" [last rows: {', '.join(e.rows_tail)}]"
        if e.summary:
            line += f" — content summary: {e.summary}"
        lines.append(line)
    n = len(cands)
    user = "\n".join(
        [
            f'Research question: "{question}"',
            f"Retrieval target: {branch.key}",
            f"REPAIR PASS: a coverage review of the data already extracted found this still "
            f"MISSING: {what} (period: {period or branch.period or 'unspecified'}). Select "
            "only blocks that supply the missing values. If no single block covers the whole "
            "missing period, select the SET of blocks whose data windows JOINTLY tile it — "
            "every missing month must be covered by some selected block.",
            f"Candidate blocks ({n}) — mark true the AT MOST {_REPAIR_KEEP_PER_NEED} that "
            "best supply the MISSING data, false for the rest:\n" + "\n".join(lines),
        ]
    )

    def _parse(text: str, _ctx: ExecutionContext) -> list[bool]:
        return PageIndexRetriever._parse_bool_list(text, _ctx, n=n)

    call: PromptedCall[list[bool]] = PromptedCall(
        name="extract.repair_select",
        system_prompt=PageIndexRetriever._BLOCK_SELECT_PROMPT,
        parse=_parse,
        default_effort="off",
        output_instruction=(
            f"Output ONLY a JSON array of EXACTLY {n} booleans — one per block, in the order "
            f"given (true=select, false=drop), with AT MOST {_REPAIR_KEEP_PER_NEED} true. "
            "No prose, no markdown fences."
        ),
    )
    verdicts = await call.call(ctx, user, temperature=0.0)
    return [e for e, k in zip(cands, verdicts) if k][:_REPAIR_KEEP_PER_NEED]


class ExtractOp:
    """The extract operator — question-driven extraction. Owns one instance of
    each call-site extractor and drives the parsed_json → vision tier fallback, with a
    vision confirmation round over the parsed_json tier's output, then the coverage
    review (shadow / dedup / repair per config)."""

    def __init__(self) -> None:
        self._text = TextExtractor()
        self._vision = VisionExtractor()
        self._confirm = VisualValidator()
        self._review = CoverageReview()

    async def run(
        self,
        blocks: list[BlockRef],
        ctx: ExecutionContext,
        branch: RetrieveBranch,
        sem_pool: list[SemPoolEntry] | None = None,
    ) -> list[AnnotatedValue]:
        """Extract from the branch's retrieved blocks, then review. `sem_pool` is the
        branch's sem-filter candidate pool — when `extract_review_repair` is on and the
        review flags missing coverage, one add-only repair round re-selects from it and
        extracts the additions."""
        entries = await self._extract_once(blocks, ctx, branch)
        if not ctx.config.extract_review_shadow:
            return entries
        entries, verdict = await self._review.run_shadow(
            ctx.question, branch, entries, ctx
        )
        if (
            ctx.config.extract_review_repair
            and verdict is not None
            and not verdict["complete"]
            and sem_pool
        ):
            try:
                entries = await self._repair(
                    blocks, ctx, branch, entries, verdict["missing"], sem_pool
                )
            except Exception as e:  # noqa: BLE001 — repair must never break the branch
                ctx.emit(f"repair_failed error={e!r}")
        return entries

    async def _extract_once(
        self,
        blocks: list[BlockRef],
        ctx: ExecutionContext,
        branch: RetrieveBranch,
    ) -> list[AnnotatedValue]:
        """One tier sweep over `blocks` (no review). The text tier feeds each block's pages
        whole (scoped by the selected blocks' metadata); the vision tier renders those same
        pages."""
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
            ctx.question, branch, images, rendered_refs, ctx
        )
        if not entries:
            raise StepFailed("extract", "no relevant values found across tiers")
        ctx.emit(
            f"tier_result tier=vision descriptions={[e.description for e in entries]!r}"
        )
        return entries

    async def _repair(
        self,
        blocks: list[BlockRef],
        ctx: ExecutionContext,
        branch: RetrieveBranch,
        entries: list[AnnotatedValue],
        missing: list[dict],
        sem_pool: list[SemPoolEntry],
    ) -> list[AnnotatedValue]:
        """One add-only repair round: per flagged need, deterministically prefilter the
        pool, one selection call, then a single extract sweep over all newly picked blocks;
        the new entries are merged onto `entries`. Bounded by the _REPAIR_* constants; any
        sub-step failure degrades to returning `entries` unchanged."""
        budget = _REPAIR_MAX_TOTAL_BLOCKS - len(blocks)
        if budget <= 0:
            ctx.emit(f"repair_skipped reason=block_budget n_blocks={len(blocks)}")
            return entries
        used = {_block_key(b) for b in blocks}
        picked: list[SemPoolEntry] = []
        for need in missing[:_REPAIR_MAX_NEEDS]:
            period = str(need.get("period", ""))
            what = str(need.get("what", ""))
            cands = _repair_prefilter(sem_pool, period, exclude=used)
            if not cands:
                ctx.emit(
                    f"repair_need period={period!r} what={what[:90]!r} candidates=0 picked=0"
                )
                continue
            try:
                sel = await _repair_select(
                    ctx, ctx.question, branch, period, what, cands
                )
            except ParseError as e:
                ctx.emit(f"repair_select_failed error={e.detail!r}")
                continue
            for s in sel:
                key = _block_key(s.ref)
                if key not in used:
                    used.add(key)
                    picked.append(s)
            ctx.emit(
                f"repair_need period={period!r} what={what[:90]!r} "
                f"candidates={len(cands)} picked={[f'{s.ref.page.month}:{s.ref.page.page}#{s.ref.block_index}' for s in sel]!r}"
            )
        picked = picked[:budget]
        if not picked:
            ctx.emit("repair_done added_blocks=0 added_entries=0")
            return entries
        new_blocks = [p.ref for p in picked]
        try:
            extra = await self._extract_once(new_blocks, ctx, branch)
        except StepFailed as e:
            ctx.emit(f"repair_extract_failed error={str(e)!r}")
            return entries
        ctx.emit(
            f"repair_done added_blocks={len(new_blocks)} added_entries={len(extra)} "
            f"descriptions={[x.description for x in extra]!r}"
        )
        return entries + extra
