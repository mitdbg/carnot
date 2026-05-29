"""extract operator — question-driven extraction over page text/images.

Takes the user's question plus a set of retrieved pages, returns a
`list[AnnotatedValue]` for the downstream compute step. Each entry has one
of three shapes:

  scalar  — a single number or string.
  vector  — a 1-D series indexed by one varying dim.
  table   — a 2-D grid indexed by two varying dims (row × col).

Entries carry `description`, `unit`, and shape-specific axis labels
(`index_name` for vectors; `row_name` + `col_name` for tables). Cells are
always primitive scalars — deeper nesting is rejected.

Three call-site executors back the two tiers:
  `ExtractTextPromptedCall`   — parsed-table text (parsed_json tier).
  `ExtractVisionPromptedCall` — rendered page images (vision tier).
  `ExtractDedupPromptedCall`  — consolidates redundant entries from multiple
                            sampling passes; cannot invent values, only
                            select representatives.

`ExtractExecutor` owns one instance of each and drives the tier dispatch.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from skunk.common import parse_json_response
from skunk.errors import StepFailed
from skunk.prompted_call import PromptedCall
from skunk.models import AnnotatedValue, HarnessContext, PageRef
from skunk.plan import RetrieveBranch
from skunk.pdf_prep import (
    get_text_for_pdf_page,
    render_pdf_page_b64,
)


def _cells_with_path(
    entry: AnnotatedValue,
) -> Iterator[tuple[tuple[str, ...], Any]]:
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


def _parse_response_raw(raw: str, ctx: HarnessContext) -> list[AnnotatedValue] | None:
    """Parse an LLM response into AnnotatedValues. Returns None on unparseable
    / non-array / empty (caller falls through to the next tier); drops individual
    entries that fail validation with a diagnostic."""
    obj = parse_json_response(raw)
    if obj is None:
        ctx.emit("extract", "rejected unparseable response", raw=raw)
        return None
    if not isinstance(obj, list):
        ctx.emit(
            "extract", "rejected non-array response", got=type(obj).__name__, raw=raw
        )
        return None
    if not obj:
        return None

    entries: list[AnnotatedValue] = []
    for i, entry in enumerate(obj):
        try:
            entries.append(AnnotatedValue.model_validate(entry))
        except (ValueError, TypeError) as e:
            ctx.emit("extract", "rejected entry", entry_idx=i, reason=str(e))
    return entries


def _cell_in_text(value: int | float | str, text: str) -> bool:
    """True if primitive `value` appears verbatim in `text`. For integer-valued
    numerics, also try the comma-formatted form (2582 → "2,582") so the prompt's
    "no commas" output rule doesn't cause false negatives against comma-formatted
    page text. String check is case-insensitive."""
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


def _cells_equal(a: Any, b: Any) -> bool:
    """Primitive cell equivalence: float-coerce so 4 / 4.0 / "4" all match;
    string fallback is case-insensitive trimmed."""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


def _output_grounded_in_inputs(
    output: AnnotatedValue, inputs: list[AnnotatedValue]
) -> bool:
    """Every cell of `output` must come verbatim from a SINGLE input entry.
    Scalar output is special-cased: the LLM may flatten one cell of any input
    vector/table into a scalar, so any cell of any single input may match."""
    out_cells = list(_cells_with_path(output))
    for s in inputs:
        if output.kind == "scalar":
            if any(_cells_equal(output.value, v) for _, v in _cells_with_path(s)):
                return True
        elif s.kind == output.kind:
            inp = dict(_cells_with_path(s))
            if all(p in inp and _cells_equal(v, inp[p]) for p, v in out_cells):
                return True
    return False


# Shared envelope spec used by all three extract executors. Defines the
# AnnotatedValue shape (scalar / vector / table), field semantics, and the
# universal output rules. Per-modality system prompts append their own role
# description and grounding rule.
EXTRACT_COMMON_PROMPT = """\
## AnnotatedValue shape

A single JSON ARRAY of entries. One entry per distinct datum. Pick the
smallest shape that fits:

  scalar: {"description":"...","kind":"scalar",
           "value":<num|str>,"unit":"..."}
  vector: {"description":"...","kind":"vector",
           "index_name":"<dim>",
           "value":{"<index>":<value>,...},"unit":"..."}
  table:  {"description":"...","kind":"table",
           "row_name":"<dim>","col_name":"<dim>",
           "value":{"<row>":{"<col>":<value>,...},...},"unit":"..."}

Cells MUST be primitive (number or string). For a third axis, emit
multiple separate entries — no nested cells.

## Field semantics

description   natural-language label that uniquely identifies the
              datum (series + period + sub-category + any other
              distinguishing context). For the label text, use the
              page's verbatim row text / column header / caption phrase
              so the downstream consumer can map it back to the page.
              
index_name    (vector only) name of the varying dimension.

row_name /
col_name      (table only) names of the two varying dimensions.

unit          natural-language label for the printed scale and base,
              e.g. "millions of dollars", "percent", "year". Match
              what the page prints. Leave blank ("") if the value is
              not a measurement (e.g. a name or other string answer).

## Other rules

- Every cell in a vector/table shares one unit (apply any conversion
  once over the whole payload, never cell-by-cell).
- Never skip the Total / Balance / standalone variant just because a partial / Net / consolidated variant on the same row already matches the lookup phrase.
- Numbers in `value` are bare — no commas, no $, no %.
- If nothing relevant is found, return [].
- Output ONLY the JSON array — no fences, no prose.
"""


class ExtractTextPromptedCall(PromptedCall):
    name: str = "extract.text"
    default_effort = "off"
    system_prompt: str = """\
You retrieve printed values from page text to fulfill a specific lookup.
Each user message describes the lookup — what to find and (when stated)
the period — followed by the full-context question this lookup supports,
then the page text to draw values from. Emit one entry per distinct row
that could plausibly satisfy the lookup — including cases where multiple
rows partially match. Do not compute or transform — extract only what
is printed. Every numeric value emitted MUST appear on the page verbatim.
Choose the AnnotatedValue shape (scalar / vector / table) that fits the data on the page;
pick the smallest shape that captures every relevant value.

{{ common }}
{{ default_tail }}"""

    def template_vars(self, ctx: HarnessContext) -> dict:
        return {"common": EXTRACT_COMMON_PROMPT}

    def _extract_from_group(
        self,
        question: str,
        branch: RetrieveBranch,
        group: list[tuple[PageRef, str]],
        ctx: HarnessContext,
    ) -> list[AnnotatedValue]:
        """Sample `n_samples` LLM extractions over one page-group at sampling
        temperature, per-cell-verify each sample's entries against the group's
        joined page text, return the flat list of kept entries."""
        # Phrase the lookup in prose, then the surrounding question as
        # context, then the group's page text concatenated as one block.
        # Same-bulletin adjacent pages flow together so a continuation table
        # reads as one contiguous table — the PDF page break is a layout
        # artifact. The bulletin's own date/page headers inside the page
        # text provide whatever provenance the LLM needs when writing
        # `description` fields. Shape (scalar/vector/table) is decided by
        # the LLM from the data; the planner does not pre-declare it.
        target = f"You are looking for {branch.key}"
        if branch.period:
            target += f" for the period {branch.period}"
        target += "."
        context_line = (
            f'For full context, this lookup serves to help answer the question: '
            f'"{question}"'
        )
        content = "\n\n".join(text for _, text in group)
        user_msg = "\n\n".join([target, context_line, content])
        verify_text = content
        n_samples = ctx.config.extract_n_samples
        temperature = ctx.config.extract_sample_temperature

        def _sample(sample_idx: int) -> list[AnnotatedValue]:
            resp = self.call(ctx, user_msg, temperature=temperature)
            parsed = _parse_response_raw(resp.text, ctx) or []
            ctx.emit(
                "extract",
                f"tier=parsed_json sample {sample_idx + 1}/{n_samples} "
                f"(p={len(group)})",
                raw=resp.text,
                n_entries=len(parsed),
            )
            kept = [
                e
                for e in parsed
                if all(_cell_in_text(v, verify_text) for _, v in _cells_with_path(e))
            ]
            n_dropped = len(parsed) - len(kept)
            if n_dropped:
                ctx.emit(
                    "extract",
                    f"tier=parsed_json verifier dropped {n_dropped}/{len(parsed)}",
                    sample_idx=sample_idx + 1,
                )
            return kept

        max_workers = max(1, min(n_samples, ctx.config.max_parallel_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return [e for kept in pool.map(_sample, range(n_samples)) for e in kept]

    def run(
        self,
        question: str,
        branch: RetrieveBranch,
        pages: list[tuple[PageRef, str]],
        ctx: HarnessContext,
    ) -> list[AnnotatedValue]:
        """Group pages into continuation-table bundles, extract each group in
        parallel, return one flat list of verified entries. Golden mode keeps
        all pages in one group so the LLM sees them in a single prompt."""
        groups: list[list[tuple[PageRef, str]]] = []
        if ctx.config.golden_pages is not None:
            groups = [pages]
        else:
            for item in pages:
                ref, _ = item
                if groups:
                    prev_ref = groups[-1][-1][0]
                    same_bulletin = (
                        ref.month is not None and ref.month == prev_ref.month
                    )
                    adjacent = (
                        ref.page is not None
                        and prev_ref.page is not None
                        and ref.page == prev_ref.page + 1
                    )
                    if same_bulletin and adjacent:
                        groups[-1].append(item)
                        continue
                groups.append([item])
        ctx.emit(
            "extract",
            f"tier=parsed_json fan-out {len(groups)}g × {ctx.config.extract_n_samples}s "
            f"@ T={ctx.config.extract_sample_temperature}",
            n_groups=len(groups),
            n_pages=len(pages),
            group_sizes=[len(g) for g in groups],
            total_chars=sum(len(t) for _, t in pages),
            system_prompt=self.assemble_system_prompt(ctx),
        )

        max_workers = max(1, min(len(groups), ctx.config.max_parallel_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            per_group = pool.map(
                lambda g: self._extract_from_group(question, branch, g, ctx),
                groups,
            )
            return [e for kept in per_group for e in kept]


class ExtractVisionPromptedCall(PromptedCall):
    name: str = "extract.vision"
    default_effort = "off"
    system_prompt: str = """\
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

{{ common }}
{{ default_tail }}"""

    def template_vars(self, ctx: HarnessContext) -> dict:
        return {"common": EXTRACT_COMMON_PROMPT}

    def call_once(
        self,
        question: str,
        branch: RetrieveBranch,
        images: list[tuple[str, str]],
        rendered_refs: list[PageRef],
        ctx: HarnessContext,
    ) -> list[AnnotatedValue]:
        """One deterministic (T=0) vision call over rendered page images."""
        # Images carry no inherent (month, page) header the way page text
        # does — the numbered identification list below is what maps each
        # attachment back to its source page.
        target = f"You are looking for {branch.key}"
        if branch.period:
            target += f" for the period {branch.period}"
        target += "."
        context_line = (
            f'For full context, this lookup serves to help answer the question: '
            f'"{question}"'
        )
        image_lines = [
            f"Image {i + 1}: PDF page {ref.page} of the {ref.month} Treasury Bulletin"
            for i, ref in enumerate(rendered_refs)
        ]
        user_msg = "\n\n".join(
            [
                target,
                context_line,
                "Images attached, in order:\n" + "\n".join(image_lines),
            ]
        )

        ctx.emit("extract", "tier=vision single-call (T=0)", n_images=len(images))
        resp = self.call(ctx, user_msg, images=images, temperature=0.0)
        raw = resp.text
        parsed = _parse_response_raw(raw, ctx)
        ctx.emit(
            "extract",
            "tier=vision single-call",
            raw=raw,
            n_entries=0 if parsed is None else len(parsed),
        )
        return parsed if parsed is not None else []


class ExtractDedupPromptedCall(PromptedCall):
    name: str = "extract.dedup"
    default_effort = "medium"
    system_prompt: str = """\
You consolidate redundant extraction entries. Multiple independent
passes over the same pages produced overlapping entries; collapse
wording duplicates into one representative per distinct datum. The user
message contains a JSON array of AnnotatedValue entries inside a ```json
fenced block; output a JSON array in the same envelope.

You are a PICKER, not a calculator. Every value, key, and cell in your
output MUST appear verbatim in some input entry. Do not compute,
aggregate, average, derive, rescale, round, reformat, or invent values
or keys.

{{ common }}

## Dedup rules

- One output entry per distinct datum. Wording duplicates → one
  representative; prefer the clearest, most specific description.
- All cells of an output entry come from a SINGLE input entry — do not
  graft cells across inputs. If two inputs disagree on a value at the
  same key, keep both as separate output entries with disambiguating
  descriptions.
- N genuinely distinct datums → N entries.
{{ default_tail }}"""

    def template_vars(self, ctx: HarnessContext) -> dict:
        return {"common": EXTRACT_COMMON_PROMPT}

    def dedup(
        self,
        merged_entries: list[AnnotatedValue],
        ctx: HarnessContext,
    ) -> list[AnnotatedValue]:
        """LLM-based semantic dedup at T=0. Output is structurally verified
        against `merged_entries`; entries that fail are dropped. Returns `[]`
        on empty/unparseable response or all-rejected output — caller's
        tier-fallback handles it (no silent fallback to un-deduped inputs)."""
        if len(merged_entries) <= 1:
            return merged_entries
        envelope = [e.model_dump(exclude_none=True) for e in merged_entries]
        user_msg = (
            f"Input entries from multiple independent extraction passes:\n"
            f"```json\n{json.dumps(envelope, indent=2, default=str)}\n```\n\n"
            f"Output the consolidated set as a JSON array in the same envelope."
        )
        ctx.emit(
            "extract",
            "tier=parsed_json dedup call (T=0)",
            n_input_entries=len(envelope),
        )
        resp = self.call(ctx, user_msg, temperature=0.0)
        raw = resp.text
        parsed = _parse_response_raw(raw, ctx)
        ctx.emit(
            "extract",
            "tier=parsed_json dedup response",
            raw=raw,
            n_entries=0 if parsed is None else len(parsed),
        )
        if not parsed:
            ctx.emit(
                "extract",
                "tier=parsed_json dedup failed: empty or unparseable response",
                n_input_entries=len(merged_entries),
            )
            return []
        kept = [e for e in parsed if _output_grounded_in_inputs(e, merged_entries)]
        n_dropped = len(parsed) - len(kept)
        if n_dropped:
            ctx.emit(
                "extract",
                f"tier=parsed_json dedup verifier dropped {n_dropped}/{len(parsed)}",
            )
        if not kept:
            ctx.emit(
                "extract",
                "tier=parsed_json dedup failed: every output entry rejected by verifier",
                n_input_entries=len(merged_entries),
                n_output_entries=len(parsed),
            )
            return []
        return kept


class ExtractExecutor:
    """Question-driven extraction. Owns one instance of each call-site
    executor and drives the parsed_json → vision tier fallback."""

    def __init__(self) -> None:
        self._text = ExtractTextPromptedCall()
        self._vision = ExtractVisionPromptedCall()
        self._dedup = ExtractDedupPromptedCall()

    def _parsed_json_tier(
        self,
        refs: list[PageRef],
        ctx: HarnessContext,
        branch: RetrieveBranch,
    ) -> list[AnnotatedValue] | None:
        """Tier 1 — group-aware page fan-out → per-cell text verifier → LLM dedup."""
        # Gather text per ref. Outside golden mode, cap the page count first:
        # every ref produces a text block in the prompt. Refs whose parsed
        # source has no text for this PDF page are dropped — the vision tier
        # can still pick them up on fallback.
        if ctx.config.golden_pages is None:
            refs = refs[: ctx.config.extract_max_pages]
        pages: list[tuple[PageRef, str]] = []
        for ref in refs:
            text = get_text_for_pdf_page(ref, ctx)
            if not text:
                ctx.emit("extract", "tier=parsed_json no text", page=str(ref))
                continue
            ctx.emit(
                "extract",
                "tier=parsed_json got text",
                page=str(ref),
                chars=len(text),
            )
            pages.append((ref, text))
        if not pages:
            ctx.emit("extract", "tier=parsed_json skipped (no text from any ref)")
            return None
        all_entries = self._text.run(ctx.question, branch, pages, ctx)
        if not all_entries:
            ctx.emit("extract", "tier=parsed_json all samples empty after verifier")
            return None
        ctx.emit(
            "extract",
            "tier=parsed_json merged; dedup over full input",
            n_full_input=len(all_entries),
        )
        deduped = self._dedup.dedup(all_entries, ctx)
        return deduped or None

    def _vision_tier(
        self,
        refs: list[PageRef],
        ctx: HarnessContext,
        branch: RetrieveBranch,
    ) -> list[AnnotatedValue] | None:
        """Tier 2 — single T=0 call over rendered page images, with per-image
        PageRef labels in the user message so the LLM can't conflate pages."""
        if ctx.config.golden_pages is None:
            refs = refs[: ctx.config.extract_max_pages]
        images: list[tuple[str, str]] = []
        rendered_refs: list[PageRef] = []
        for ref in refs:
            img = render_pdf_page_b64(ref, ctx)
            if img:
                ctx.emit("extract", "tier=vision rendered png", page=str(ref))
                images.append(img)
                rendered_refs.append(ref)
            else:
                ctx.emit("extract", "tier=vision no png", page=str(ref))

        if not images:
            ctx.emit("extract", "tier=vision skipped (no images)")
            return None

        entries = self._vision.call_once(
            ctx.question, branch, images, rendered_refs, ctx
        )
        return entries or None

    def run(
        self,
        prev: list[PageRef] | None,
        ctx: HarnessContext,
        branch: RetrieveBranch,
    ) -> list[AnnotatedValue]:
        refs = prev or []
        if not refs:
            raise StepFailed("extract", "No page refs to extract from")

        ctx.emit(
            "extract",
            "starting",
            visual_only=branch.visual_only,
            n_refs=len(refs),
            refs=[str(r) for r in refs],
            key=branch.key or None,
            period=branch.period,
        )

        tiers = [] if branch.visual_only else [("parsed_json", self._parsed_json_tier)]
        tiers.append(("vision", self._vision_tier))

        for tier_name, tier_fn in tiers:
            result = tier_fn(refs, ctx, branch)
            if result is None:
                continue
            ctx.emit(
                "extract",
                f"tier={tier_name} produced values",
                descriptions=[e.description for e in result],
            )
            return result

        raise StepFailed("extract", "no relevant values found across tiers")
