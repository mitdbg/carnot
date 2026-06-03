from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

from skunk.common import (
    AnnotatedValue,
    B64Image,
    ExecutionContext,
    PageRef,
    parse_json_response,
)
from skunk.errors import StepFailed
from skunk.prompted_call import PromptedCall
from skunk.plan import RetrieveBranch
from skunk.corpus import get_page_text, page_elements, render_page_b64

def _render_pages_b64(
    refs: list[PageRef],
    ctx: ExecutionContext,
    *,
    dpi: int = 300,
    fmt: str = "png",
) -> tuple[list[B64Image], list[PageRef]]:
    images: list[B64Image] = []
    rendered_refs: list[PageRef] = []
    for ref in refs:
        try:
            img = render_page_b64(ref.month, ref.page, dpi=dpi, fmt=fmt)
        except Exception as e:  # noqa: BLE001 — vision tier fallback; any fitz error → skip page
            ctx.emit(f"render_failed page={str(ref)} error={str(e)!r}")
            continue
        if img:
            ctx.emit(f"rendered_png page={str(ref)}")
            images.append(img)
            rendered_refs.append(ref)
        else:
            ctx.emit(f"no_png page={str(ref)}")
    return images, rendered_refs


def _parse_extract_response(raw: str, ctx: ExecutionContext) -> list[AnnotatedValue] | None:
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

## Field semantics

description   natural-language label that uniquely identifies the
              datum (series + period + sub-category + any other
              distinguishing context). For the label text, use the
              page's verbatim row text / column header / caption phrase
              so the downstream consumer can map it back to the page.
              If the value comes from a specific year or period column
              of a multi-year table, you should include that year/period
              in the description.

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
_EXTRACT_OUTPUT_INSTRUCTION = "Output a JSON array of AnnotatedValue entries — no markdown fences, no prose."


class TextExtractor:
    _PREAMBLE = """\
You retrieve printed values from page text to fulfill a specific lookup.
Each user message describes the lookup — what to find and (when stated)
the period — followed by the full-context question this lookup supports,
then the page text to draw values from. Emit one entry per distinct row
that could plausibly satisfy the lookup — including cases where multiple
rows partially match. Do not compute or transform — extract only what
is printed. Every numeric value emitted MUST appear on the page verbatim.
Choose the AnnotatedValue shape (scalar / vector / table) that fits the data on the page;
pick the smallest shape that captures every relevant value."""

    _prompt = PromptedCall(
        name="extract.text",
        system_prompt=_PREAMBLE + "\n\n" + EXTRACT_COMMON_PROMPT,
        default_effort="off",
        parse=_parse_extract_response,
        output_instruction=_EXTRACT_OUTPUT_INSTRUCTION,
    )

    @staticmethod
    def _fetch_page_texts(refs: list[PageRef], ctx: ExecutionContext) -> list[tuple[PageRef, str]]:
        """Fetch parsed text per ref; skip pages with none (the vision tier can still
        read them), appending a figure-note hint when the page carries charts."""
        pages: list[tuple[PageRef, str]] = []
        for ref in refs:
            text = get_page_text(ref.month, ref.page)
            if not text:
                ctx.emit(f"no_text tier=parsed_json page={str(ref)}")
                continue
            # Figures are parsed as `type="figure"` with `content=null`: their plotted
            # data is absent from the text, so without a heads-up the tier reports the
            # value missing or scrapes it from prose. Flag any figures so it can defer
            # to the vision tier instead.
            try:
                els = page_elements(ref.month).get(ref.page) or []  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001 — best-effort hint; never block extraction on a parse miss
                els = []
            n_figs = sum(1 for e in els if e.get("type") == "figure")
            if n_figs:
                headers = list(dict.fromkeys(
                    e["content"].strip() for e in els
                    if e.get("type") in ("title", "section_header") and e.get("content")
                ))
                note = (
                    f"[This page has {n_figs} figure(s)/chart(s) (headings: "
                    f"{'; '.join(headers) or '(untitled)'}) whose plotted data is NOT in the "
                    f"text above. If the value you need appears only in a chart, return [] so "
                    f"the vision tier can read it.]"
                )
                ctx.emit(f"figure_hint tier=parsed_json page={str(ref)} note_chars={len(note)}")
                text = f"{text}\n\n{note}"
            ctx.emit(f"got_text tier=parsed_json page={str(ref)} chars={len(text)}")
            pages.append((ref, text))
        return pages

    @staticmethod
    def _group_refs(refs: list[PageRef], *, single_group: bool) -> list[list[PageRef]]:
        """Bundle consecutive same-bulletin pages so a table spanning pages reads as one
        prompt. `single_group` (golden mode) keeps every ref in one group instead.
        Operates on refs alone; text is fetched per group afterward — so a continuation
        run stays together even if a middle page has no parsed text."""
        if single_group:
            return [list(refs)]
        groups: list[list[PageRef]] = []
        for ref in refs:
            prev = groups[-1][-1] if groups else None
            if (
                prev is not None
                and ref.month is not None and ref.month == prev.month
                and ref.page is not None and prev.page is not None
                and ref.page == prev.page + 1
            ):
                groups[-1].append(ref)
            else:
                groups.append([ref])
        return groups

    async def _extract_group(
        self,
        group_refs: list[PageRef],
        branch: RetrieveBranch,
        question: str,
        ctx: ExecutionContext,
    ) -> list[AnnotatedValue]:
        pages = self._fetch_page_texts(group_refs, ctx)
        if not pages:
            ctx.emit(f"group_skipped tier=parsed_json reason=no_text refs={[str(r) for r in group_refs]!r}")
            return []
        # Build the prompt around the group's joined page text. `content` is kept on its
        # own so the verifier checks emitted cells against the page text, not the prompt
        # scaffolding.
        # TODO: further cleanup / context-management should happen here per group
        content = "\n\n".join(text for _, text in pages)
        user_msg = "\n\n".join([
            f"You are looking for {branch.key}{f' for the period {branch.period}' if branch.period else ''}.",
            f'For full context, this lookup serves to help answer the question: "{question}"',
            content,
        ])
        parsed = await self._prompt.call(ctx, user_msg, temperature=0.0) or []
        ctx.emit(f"extracted tier=parsed_json n_pages={len(pages)} n_entries={len(parsed)}")
        kept = [e for e in parsed if all(_cell_in_text(v, content) for _, v in _cells_with_path(e))]
        if len(kept) < len(parsed):
            ctx.emit(f"verifier_dropped tier=parsed_json n_dropped={len(parsed) - len(kept)} n_parsed={len(parsed)}")
        return kept

    async def run(
        self,
        question: str,
        branch: RetrieveBranch,
        refs: list[PageRef],
        ctx: ExecutionContext,
    ) -> list[AnnotatedValue]:
        groups = self._group_refs(refs, single_group=ctx.config.golden_pages is not None)
        ctx.emit(
            f"fan_out tier=parsed_json n_groups={len(groups)} n_refs={len(refs)} "
            f"group_sizes={[len(g) for g in groups]}"
        )

        per_group = await asyncio.gather(*[self._extract_group(g, branch, question, ctx) for g in groups])
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
        user_msg = "\n\n".join([
            f"You are looking for {branch.key}{period}.",
            f'For full context, this lookup serves to help answer the question: "{question}"',
            "Images attached, in order:\n" + "\n".join(image_lines),
        ])
        ctx.emit(f"vision_call tier=vision n_images={len(images)}")
        entries = await self._prompt.call(ctx, user_msg, images=images, temperature=0.0)
        ctx.emit(f"vision_result tier=vision n_entries={0 if entries is None else len(entries)}")
        return entries or []


class ExtractOp:
    """The extract operator — question-driven extraction. Owns one instance of
    each call-site extractor and drives the parsed_json → vision tier fallback."""

    def __init__(self) -> None:
        self._text = TextExtractor()
        self._vision = VisionExtractor()

    async def run(
        self,
        refs: list[PageRef] | None,
        ctx: ExecutionContext,
        branch: RetrieveBranch,
    ) -> list[AnnotatedValue]:
        if not refs:
            raise StepFailed("extract", "No page refs to extract from")

        # parsed_json tier first (skipped for visual_only charts/figures); fall through
        # to the vision tier when it finds nothing.
        if not branch.visual_only:
            entries = await self._text.run(ctx.question, branch, refs, ctx)
            if entries:
                ctx.emit(f"tier_result tier=parsed_json descriptions={[e.description for e in entries]!r}")
                return entries

        # vision tier — render the pages, then read values off the images.
        images, rendered_refs = _render_pages_b64(refs, ctx)
        entries = await self._vision.run(ctx.question, branch, images, rendered_refs, ctx)
        if not entries:
            raise StepFailed("extract", "no relevant values found across tiers")
        ctx.emit(f"tier_result tier=vision descriptions={[e.description for e in entries]!r}")
        return entries

