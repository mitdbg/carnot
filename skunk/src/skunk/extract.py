from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from skunk.common import (
    AnnotatedValue,
    B64Image,
    BranchRetrieval,
    ExecutionContext,
    PageRef,
    parse_json_response,
    traced_step,
)
from skunk.errors import ParseError, StepFailed
from skunk.page_store import PageContentStore
from skunk.prompted_call import PromptedCall
from skunk.plan import RetrieveBranch


def _require_store(ctx: ExecutionContext) -> PageContentStore:
    """The injected page-content backend, or a clear StepFailed when the app forgot to
    wire one (`Orchestrator(page_store=...)`) — extract has no other content source."""
    if ctx.page_store is None:
        raise StepFailed(
            "extract",
            "no page store configured — pass page_store= to the Orchestrator "
            "(see skunk.page_store.PageContentStore)",
        )
    return ctx.page_store


def _render_pages_b64(
    refs: list[PageRef],
    ctx: ExecutionContext,
    *,
    strict: bool = False,
) -> tuple[list[B64Image], list[PageRef]]:
    """Page images for the vision tier, from the page store (rendered on demand +
    cached). The returned refs identify each image's source page so the prompt's numbered list
    can't conflate them. When `strict`, a page that fails to render (error or missing PNG) raises
    StepFailed instead of being silently skipped — the vision tier must not drop the answer page.
    The vision fallback leaves `strict` off (a render miss there just skips that page)."""
    store = _require_store(ctx)
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


# A string cell that is really a number wearing print decorations: optional
# parens (negative), thousands commas, and a trailing print flag (r/p/e),
# footnote marker (2/), or asterisks. Genuine text cells ("n.a.", labels,
# dates like "2002/06") do not match.
_NUMERIC_CELL_RE = re.compile(
    r"^(?P<neg>\()?\s*(?P<sign>-)?\s*\$?\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(?(neg)\))\s*(?:[rpe]|\d{1,2}/|\*{1,2})?\s*$",
    re.IGNORECASE,
)


def _coerce_cell(v: Any) -> Any:
    """Deterministically repair a numeric cell the model emitted as a string —
    "376 r" → 376, "1,234.5" → 1234.5, "(12)" → -12. Non-matching strings pass
    through untouched."""
    if not isinstance(v, str):
        return v
    m = _NUMERIC_CELL_RE.match(v.strip())
    if m is None:
        return v
    num = float(m.group("num").replace(",", ""))
    if m.group("neg") or m.group("sign"):
        num = -num
    return int(num) if num.is_integer() and "." not in m.group("num") else num


def _coerce_numeric_cells(entry: Any) -> Any:
    """Apply `_coerce_cell` to an entry dict's payload cells, so a stray print flag
    or comma inside a value becomes a clean number BEFORE AnnotatedValue validation.
    Keys/labels are never touched. Pure — returns a new dict (shallow-copied with a
    rebuilt `value`); the caller's parsed-JSON input is never mutated."""
    if not isinstance(entry, dict):
        return entry
    v = entry.get("value")
    kind = entry.get("kind")
    if kind == "vector" and isinstance(v, dict):
        value = {k: _coerce_cell(c) for k, c in v.items()}
    elif kind == "table" and isinstance(v, dict):
        value = {
            r: {c: _coerce_cell(x) for c, x in row.items()}
            if isinstance(row, dict)
            else row
            for r, row in v.items()
        }
    elif kind == "scalar":
        value = [_coerce_cell(c) for c in v] if isinstance(v, list) else _coerce_cell(v)
    else:
        return entry
    return {**entry, "value": value}


def _parse_extract_response(raw: str, ctx: ExecutionContext) -> list[AnnotatedValue]:
    """Parse an LLM reply into AnnotatedValues, as a `PromptedCall` parse hook.
    Numeric cells emitted as decorated strings ("376 r") are deterministically
    coerced to numbers first — print flags belong in `notes`, and a stray flag
    must not poison the frame's dtype. Raises `ParseError` (→ `call()` re-prompts,
    echoing the detail, with escalating temperature) on a reply the model should fix:
    unparseable, non-array, or any entry that fails `AnnotatedValue` validation. An
    empty array is a legitimate "nothing relevant found" — it returns `[]` (no retry;
    the caller falls through to the next tier), since re-prompting can't conjure data
    that isn't on the page."""
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
            entries.append(AnnotatedValue.model_validate(_coerce_numeric_cells(entry)))
        except (ValueError, TypeError) as e:
            bad.append(f"entry[{i}]: {e}")
    if bad:
        raise ParseError(
            raw,
            "some entries are not valid AnnotatedValues — fix their shape/fields:\n"
            + "\n".join(bad),
        )

    # Distinguishability: two entries with the same description but different values
    # are unusable downstream — the consumer cannot tell which cell is which (the page
    # discriminator was dropped). `notes` is shared context, not a discriminator, so
    # description alone must separate them. Re-prompt with the offenders.
    by_label: dict[str, list[int]] = {}
    for i, e in enumerate(entries):
        by_label.setdefault(e.description, []).append(i)
    clashes = [
        idxs
        for idxs in by_label.values()
        if len(idxs) > 1
        and len({json.dumps(entries[i].value, sort_keys=True) for i in idxs}) > 1
    ]
    if clashes:
        raise ParseError(
            raw,
            "these entry groups are indistinguishable (same description) yet hold "
            "different values — make each description unique by folding in the "
            "discriminating column header / year / footnote / table title:\n"
            + "\n".join(
                f"entries {idxs}: {entries[idxs[0]].description!r}" for idxs in clashes
            ),
        )
    return entries


def _stamp_provenance(
    entries: list[AnnotatedValue],
    refs: list[PageRef],
    branch: RetrieveBranch,
    *,
    obtained_visually: bool = False,
) -> list[AnnotatedValue]:
    """Copy machine-fact provenance from the source refs + branch onto each entry —
    never LLM-written. `source_stem`/`pages` are attributable only when every ref in the
    call shares one source document (otherwise we can't tell which document a value came
    from, so they're left empty). `source_stem` is that document's filename stem. Branch
    fields (`period`/`key`) are call-level and always stamped. The model is frozen, so we
    rebuild via `model_copy`."""
    source_docs = {r.stem for r in refs if r.stem}
    source_stem = next(iter(source_docs)) if len(source_docs) == 1 else None
    pages = (
        tuple(sorted({r.page for r in refs if r.page is not None}))
        if source_stem is not None
        else ()
    )
    return [
        e.model_copy(
            update={
                "source_stem": source_stem,
                "pages": pages,
                "requested_period": branch.period,
                "retrieve_key": branch.key,
                "obtained_visually": obtained_visually,
            }
        )
        for e in entries
    ]


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

  scalar: {"description":"...","notes":"...","kind":"scalar",
           "value":<num|str>,"unit":"..."}
  vector: {"description":"...","notes":"...","kind":"vector",
           "index_name":"<dim>",
           "value":{"<index>":<value>,...},"unit":"..."}
  table:  {"description":"...","notes":"...","kind":"table",
           "row_name":"<dim>","col_name":"<dim>",
           "value":{"<row>":{"<col>":<value>,...},...},"unit":"..."}

Cells should be simple number or string — no nested cells. A numeric cell is
a bare number: keep print flags (r, p) and footnote markers (2/) out of the
value; record these instead in `notes`.

Transcribe a number's digits exactly — every digit and decimal place; never
round, truncate, or drop trailing digits. You may normalize only a value's
FORMAT per the corpus notes (e.g. space-separated cents -> a decimal point),
never its digits.

## Field semantics

description   natural-language label uniquely identifying the datum
              (series + period + sub-category), prefer the page's
              verbatim row/column/caption wording.

notes         the page's textual context bearing on the question:
              footnotes (and what their markers "2/" mean), headnotes,
              comments, scope caveats, break-in-series notes, the
              meaning of print flags ("p", "r"), and the table title.
              Summarize as it pertains to the question, preferring the
              page's original wording. One note per payload (shared by a
              vector/table). "" when the page carries no relevant text.
              This is context, not a discriminator — put per-datum
              distinctions in description.

index_name    (vector only) name of the varying dimension.

row_name/col_name      (table only) names of the two varying dimensions.

unit          the printed scale and base, in natural language
              ("millions of dollars", "percent"); "" for
              non-measurements. One unit per vector/table — never
              convert cell-by-cell.
"""


# Each tier's system prompt is its own `_PREAMBLE` followed by the shared
# `EXTRACT_COMMON_PROMPT`; both tiers parse with `_parse_extract_response` and share
# the JSON-array output instruction below.
_EXTRACT_OUTPUT_INSTRUCTION = (
    "Output a JSON array of AnnotatedValue entries — no markdown fences, no prose."
)


class TextExtractor:
    _PREAMBLE = """\
You retrieve printed values from page text to fulfill a specific lookup. The
user message gives the lookup (and period, when stated), the question it
serves, page metadata (each block's title), and the page text. Work out from
the table markup in the text which column/row a value sits under, the
period, and the units. Emit all data that could satisfy the lookup, including partial
matches. Extract only printed values — never compute, derive, or invent one
(metadata is context, not a source of values). A period `YYYY-MM..YYYY-MM` is
an inclusive month range."""

    _SYSTEM = _PREAMBLE + "\n\n" + EXTRACT_COMMON_PROMPT

    # Parse hook validates SHAPE only (valid AnnotatedValue array, distinguishable
    # entries); a shape defect raises a retryable ParseError so `call()` re-prompts
    # (escalating temperature) before degrading the group to empty. Exact transcription
    # is a PROMPT-level instruction (like data_prep/compute), NOT a parse-level check —
    # the old verbatim gate was net-negative, mostly false-rejecting format variants
    # (comma floats, parenthesized negatives, space-cents) and downgrading good pages.
    _prompt: PromptedCall[list[AnnotatedValue]] = PromptedCall(
        name="extract.text",
        system_prompt=_SYSTEM,
        default_effort="low",  # transcription, not reasoning
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
        store = _require_store(ctx)
        pages: list[tuple[PageRef, str]] = []
        for ref in refs:
            text = store.text(ref)
            if not text:
                ctx.emit(f"no_text tier=parsed_json page={str(ref)}")
                continue
            ctx.emit(f"got_text tier=parsed_json page={str(ref)} chars={len(text)}")
            pages.append((ref, text))
        return pages

    async def _extract_content(
        self,
        content: str,
        prov_refs: list[PageRef],
        metadata: str,
        branch: RetrieveBranch,
        question: str,
        ctx: ExecutionContext,
        looking_for: str,
    ) -> list[AnnotatedValue]:
        """Run one extraction call over `content` (whole-page text OR a block slice), verify
        every emitted cell appears in `content`, and stamp provenance from `prov_refs`. `content`
        is kept on its own message line so the verifier checks emitted cells against the source
        text, not the prompt scaffolding. `looking_for` is the opening line naming the read's
        goal(s) — composed once by `_extract_page` (see `_looking_line`)."""
        user_msg = "\n\n".join(
            [
                looking_for,
                f'For full context, this lookup serves to help answer the question: "{question}"',
                *(
                    [
                        f"Page metadata (what each block is — not a source of values):\n{metadata}"
                    ]
                    if metadata
                    else []
                ),
                content,
            ]
        )
        try:
            parsed = await self._prompt.call(ctx, user_msg, temperature=0.0)
        except ParseError as e:
            ctx.emit(f"extract_parse_failed tier=parsed_json error={e.detail!r}")
            return []
        ctx.emit(f"extracted tier=parsed_json n_entries={len(parsed)}")
        return _stamp_provenance(parsed, prov_refs, branch)

    @staticmethod
    def _page_groups(
        pages: list[PageRef], store: PageContentStore
    ) -> list[tuple[PageRef, list[PageRef]]]:
        """Expand each unique retrieved page (first-seen order) to the physical pages an extract
        call must read (`store.read_group`: the page plus any dependent continuation/notes
        pages). Returns `(page, refs)` per page — one extract call each."""
        out: list[tuple[PageRef, list[PageRef]]] = []
        seen: set[PageRef] = set()
        for p in pages:
            if p in seen:
                continue
            seen.add(p)
            out.append((p, store.read_group(p)))
        return out

    async def _extract_page_group(
        self,
        page: PageRef,
        refs: list[PageRef],
        branch: RetrieveBranch,
        question: str,
        ctx: ExecutionContext,
        looking_for: str,
    ) -> list[AnnotatedValue]:
        """Extraction for one retrieved page: feed its `refs` (the page plus its linked notes
        pages) FULL text — no within-page slicing — annotated with the pages' structural
        metadata (`store.page_metadata`), plus any per-page standalone-read context
        (`store.extra_read_context`, e.g. a continuation page's inherited column grammar)."""
        texts = self._fetch_page_texts(refs, ctx)
        if not texts:
            ctx.emit(
                f"group_skipped tier=parsed_json reason=no_text refs={[str(r) for r in refs]!r}"
            )
            return []
        content = "\n\n".join(text for _, text in texts)
        prov_refs = [r for r, _ in texts]
        store = _require_store(ctx)
        metadata = store.page_metadata(prov_refs)
        cont = store.extra_read_context(page)
        if cont:
            metadata = f"{cont}\n{metadata}" if metadata else cont
        ctx.emit(
            f"page_scoped page={str(page)} n_pages={len(texts)} chars={len(content)}"
        )
        return await self._extract_content(
            content, prov_refs, metadata, branch, question, ctx, looking_for
        )

    async def run(
        self,
        question: str,
        branch: RetrieveBranch,
        pages: list[PageRef],
        ctx: ExecutionContext,
        looking_for: str,
    ) -> list[AnnotatedValue]:
        """Extract from the retrieved pages — one extract call per unique page, its dependent
        pages (header predecessors + notes) fed whole. `looking_for` is the composed opening
        line naming the read's goal(s)."""
        groups = self._page_groups(pages, _require_store(ctx))
        ctx.emit(
            f"fan_out tier=parsed_json n_groups={len(groups)} n_pages={len(pages)}"
        )
        per_group = await asyncio.gather(
            *[
                self._extract_page_group(p, refs, branch, question, ctx, looking_for)
                for p, refs in groups
            ]
        )
        entries = [e for kept in per_group for e in kept]
        if not entries:
            ctx.emit("tier_empty tier=parsed_json reason=no_entries")
        return entries


class VisionExtractor:
    _PREAMBLE = """\
You retrieve visible values from rendered page images to fulfill a specific
lookup. The user message gives the lookup (and period, when stated), the
question it serves, and a numbered list of the attached images. Emit every
visible value that could plausibly satisfy the lookup, including partial matches.
When the question asks for visual understanding of a chart (e.g. counting bars above a threshold),
you may directly answer the question. Otherwise, extract only what is visibly printed — never compute,
derive, or invent. A period `YYYY-MM..YYYY-MM` is an inclusive month range."""

    _prompt = PromptedCall(
        name="extract.vision",
        system_prompt=_PREAMBLE + "\n\n" + EXTRACT_COMMON_PROMPT,
        default_effort="low",  # verbatim transcription, not reasoning
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
        looking_for: str,
        extra_context: str | None = None,
    ) -> list[AnnotatedValue]:
        """One vision call over the rendered page images. The numbered image list
        maps each attachment back to its source page so the LLM can't conflate them.
        `looking_for` is the composed opening line naming the read's goal(s).
        `extra_context` carries a header-less continuation page's inherited column
        grammar / own summaries."""
        image_lines = [
            f"Image {i + 1}: PDF page {ref.page} of document {ref.stem}"
            for i, ref in enumerate(rendered_refs)
        ]
        user_msg = "\n\n".join(
            [
                looking_for,
                f'For full context, this lookup serves to help answer the question: "{question}"',
                *([extra_context] if extra_context else []),
                "Images attached, in order:\n" + "\n".join(image_lines),
            ]
        )
        ctx.emit(f"vision_call tier=vision n_images={len(images)}")
        try:
            entries = await self._prompt.call(
                ctx, user_msg, images=images, temperature=0.0,
                max_output_tokens=ctx.config.extract_max_output_tokens,
                timeout_s=ctx.config.extract_request_timeout_s,
            )
        except ParseError as e:
            ctx.emit(f"extract_parse_failed tier=vision error={e.detail!r}")
            entries = []
        ctx.emit(f"vision_result tier=vision n_entries={len(entries)}")
        # Stamp provenance from the rendered refs. A single vision call may span
        # several documents (no per-image attribution on the reply), so source_stem/pages
        # land only when all images share one source document — the common single-doc
        # branch; multi-doc calls keep source_stem empty.
        return _stamp_provenance(entries, rendered_refs, branch, obtained_visually=True)



# ---------------------------------------------------------------------------
# Extraction sweep — read every retrieve branch's pages into AnnotatedValues.
# (Search-agent / golden retrieval is the sole frontend: a branch's result is a
# list of whole PageRefs, read directly here — no block/selection translation.)
# ---------------------------------------------------------------------------

_TEXT = TextExtractor()
_VISION = VisionExtractor()


def _synth_branch(branches: list[RetrieveBranch]) -> RetrieveBranch:
    """One stamp-bearing branch for a multi-branch page read. Branch identity is irrelevant
    at extraction, so the call-level provenance fields carry the union of the requesting
    branches."""
    keys = list(dict.fromkeys(b.key for b in branches))
    periods = list(dict.fromkeys(p for b in branches if (p := b.period)))
    return RetrieveBranch(
        key="; ".join(keys),
        period=", ".join(periods) or None,
        visual_only=any(b.visual_only for b in branches),
    )


def _looking_line(branches: list[RetrieveBranch]) -> str:
    """The read's opening line, naming every retrieval goal this page serves —
    composed ONCE here and passed verbatim through both tiers."""
    if len(branches) == 1:
        b = branches[0]
        period = f" for the period {b.period}" if b.period else ""
        return f"You are looking for {b.key}{period}."
    lines = []
    for b in branches:
        line = f"- {b.key}"
        if b.period:
            line += f" (for the period {b.period})"
        lines.append(line)
    return "You are looking for ALL of the following:\n" + "\n".join(lines)


async def _extract_page(
    ctx: ExecutionContext, page: PageRef, branches: list[RetrieveBranch]
) -> list[AnnotatedValue]:
    """One page's read serving EVERY branch that retrieved it: the call's opening line lists
    all their targets, so a single page read extracts for each. Tier escalation is the
    explicit list below: text first (unless a visual_only branch or the
    `extract_vision_only` override skips it), pure vision as the fallback; the first tier
    that yields entries wins."""
    branch = branches[0] if len(branches) == 1 else _synth_branch(branches)
    looking = _looking_line(branches)

    async def _text_tier() -> list[AnnotatedValue]:
        return await _TEXT.run(ctx.question, branch, [page], ctx, looking_for=looking)

    async def _vision_tier() -> list[AnnotatedValue]:
        images, rendered_refs = _render_pages_b64([page], ctx)
        if not images:
            return []
        return await _VISION.run(
            ctx.question, branch, images, rendered_refs, ctx, looking_for=looking,
            extra_context=_require_store(ctx).extra_read_context(page) or None,
        )

    tiers = (
        [_vision_tier]
        if branch.visual_only or ctx.config.extract_vision_only
        else [_text_tier, _vision_tier]
    )
    for tier in tiers:
        entries = await tier()
        if entries:
            return entries
    return []


async def run_extract(
    ctx: ExecutionContext,
    branches: list[RetrieveBranch],
    retrievals: list[BranchRetrieval | StepFailed],
    branch_ids: list[int],
) -> list[list[AnnotatedValue] | StepFailed]:
    """Read every branch's retrieved pages into `AnnotatedValue`s in ONE organized sweep. Each
    unique page is read exactly ONCE, mapped to every branch that retrieved it, its entries
    attributed to the FIRST such branch — so compute sees each datum once. `retrievals` is one
    slot per branch (its pages, or a `StepFailed` to carry through). Returns one result per
    branch (its entries, or the `StepFailed` to attribute to it)."""
    results: list[list[AnnotatedValue] | StepFailed | None] = [None] * len(branches)
    pages_by_pos: dict[int, list[PageRef]] = {}
    for pos, r in enumerate(retrievals):
        if isinstance(r, StepFailed):
            results[pos] = r
        else:
            pages_by_pos[pos] = list(r.pages)

    # Each unique page read ONCE, mapped to EVERY branch that retrieved it; entries
    # attributed to the first (lowest-pos) branch — its `owner`.
    want: dict[PageRef, list[int]] = {}
    for pos in sorted(pages_by_pos):
        for p in pages_by_pos[pos]:
            want.setdefault(p, []).append(pos)
    n_req = sum(len(v) for v in pages_by_pos.values())
    ctx.emit(
        f"select_extract n_requested={n_req} n_reads={len(want)} "
        f"n_already_read={n_req - len(want)}"
    )

    extracted: dict[PageRef, list[AnnotatedValue]] = {}

    async def _extract_phase() -> list[AnnotatedValue]:
        reads = await asyncio.gather(
            *(
                _extract_page(ctx, p, [branches[i] for i in poss])
                for p, poss in want.items()
            ),
            return_exceptions=True,
        )
        for p, res in zip(want, reads):
            if isinstance(res, BaseException):
                ctx.emit(
                    f"select_extract_failed page={p.stem}:{p.page} error={str(res)!r}"
                )
                extracted[p] = []
            else:
                extracted[p] = res
        # Return the flattened reads so the traced "extract" step boundary summarizes the
        # values it produced; returning [] makes the trace viewer render the step "(none)".
        return [v for vs in extracted.values() for v in vs]

    if want:
        await traced_step(ctx, "extract", _extract_phase)

    for pos in sorted(pages_by_pos):
        pages = pages_by_pos[pos]
        owned = [
            v for p in pages if want[p][0] == pos for v in extracted.get(p, [])
        ]
        covered = any(extracted.get(p) for p in pages)
        if covered or owned:
            results[pos] = owned
        else:
            results[pos] = StepFailed(
                "extract",
                f"retrieved pages yielded no data for {branches[pos].key!r}",
            )
        page_keys = sorted({f"{p.stem}:{p.page}" for p in pages})
        ctx.emit(
            f"select_pipeline_branch branch_id={branch_ids[pos]} "
            f"n_pages={len(pages)} n_entries={len(owned)} covered={covered}",
            data={"branch_id": branch_ids[pos], "pages": page_keys},
        )
    return [
        r if r is not None else StepFailed("extract", "branch produced no result")
        for r in results
    ]
