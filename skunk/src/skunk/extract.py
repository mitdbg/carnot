"""extract operator — question-driven extraction over page text/images, returning
`list[AnnotatedValue]` (scalar / vector / table shapes; see `skunk.common`).

`ExtractExecutor` drives a parsed_json → vision tier fallback over three call-site
executors: `TextExtractor` (parsed-table text), `VisionExtractor` (rendered images),
`DedupExtractor` (consolidates redundant entries; picks representatives, never invents)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from skunk.common import parse_json_response
from skunk.errors import StepFailed
from skunk.prompted_call import PromptedCall
from skunk.common import AnnotatedValue, HarnessContext, PageRef
from skunk.plan import RetrieveBranch
from skunk.corpus import get_page_text, render_page_b64


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
        ctx.emit("extract", "rejected_unparseable", raw=raw)
        return None
    if not isinstance(obj, list):
        ctx.emit("extract", "rejected_non_array", got=type(obj).__name__, raw=raw)
        return None
    if not obj:
        return None

    entries: list[AnnotatedValue] = []
    for i, entry in enumerate(obj):
        try:
            entries.append(AnnotatedValue.model_validate(entry))
        except (ValueError, TypeError) as e:
            ctx.emit("extract", "rejected_entry", entry_idx=i, reason=str(e))
    return entries


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
    """Every cell of `output` must come verbatim from a SINGLE input entry. Scalar
    output may match any single cell of any input (the LLM may flatten one cell)."""
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


# Shared envelope spec (shape + field semantics + output rules) appended to all
# three extract system prompts via the `{{ common }}` template var.
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


def _make_extract_prompt(name: str, system_prompt: str, default_effort: str) -> PromptedCall:
    """Build an extract-tier `PromptedCall` (all tiers share the common-rules var
    and `_parse_response_raw`; only name / prompt / effort differ)."""
    return PromptedCall(
        name=name,
        system_prompt=system_prompt,
        default_effort=default_effort,
        # inject the shared extraction-rules block into the SYSTEM template
        template_vars=lambda ctx: {"common": EXTRACT_COMMON_PROMPT},
        parse=_parse_response_raw,
    )


class TextExtractor:
    _SYSTEM_PROMPT = """\
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

    def __init__(self) -> None:
        self._prompt = _make_extract_prompt("extract.text", self._SYSTEM_PROMPT, "off")

    def _extract_from_group(
        self,
        question: str,
        branch: RetrieveBranch,
        group: list[tuple[PageRef, str]],
        ctx: HarnessContext,
    ) -> list[AnnotatedValue]:
        """Sample `n_samples` extractions over one page-group at sampling temperature,
        per-cell-verify each against the group's joined page text, return kept entries.
        Adjacent same-bulletin pages are joined so a continuation table reads as one."""
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
            parsed = self._prompt.call(ctx, user_msg, temperature=temperature) or []
            ctx.emit(
                "extract", "sample",
                tier="parsed_json",
                idx=sample_idx + 1,
                n=n_samples,
                n_pages=len(group),
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
                    "extract", "verifier_dropped",
                    tier="parsed_json",
                    n_dropped=n_dropped,
                    n_parsed=len(parsed),
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
            "extract", "fan_out",
            tier="parsed_json",
            n_groups=len(groups),
            n_samples=ctx.config.extract_n_samples,
            temperature=ctx.config.extract_sample_temperature,
            n_pages=len(pages),
            group_sizes=[len(g) for g in groups],
            total_chars=sum(len(t) for _, t in pages),
            system_prompt=self._prompt.assemble_system_prompt(ctx),
        )

        max_workers = max(1, min(len(groups), ctx.config.max_parallel_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            per_group = pool.map(
                lambda g: self._extract_from_group(question, branch, g, ctx),
                groups,
            )
            return [e for kept in per_group for e in kept]


class VisionExtractor:
    _SYSTEM_PROMPT = """\
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

    def __init__(self) -> None:
        self._prompt = _make_extract_prompt("extract.vision", self._SYSTEM_PROMPT, "off")

    def call_once(
        self,
        question: str,
        branch: RetrieveBranch,
        images: list[tuple[str, str]],
        rendered_refs: list[PageRef],
        ctx: HarnessContext,
    ) -> list[AnnotatedValue]:
        """One deterministic (T=0) vision call over rendered page images. The numbered
        identification list maps each attachment back to its source page."""
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

        ctx.emit("extract", "vision_call", tier="vision", n_images=len(images))
        parsed = self._prompt.call(ctx, user_msg, images=images, temperature=0.0)
        ctx.emit(
            "extract", "vision_result",
            tier="vision",
            n_entries=0 if parsed is None else len(parsed),
        )
        return parsed if parsed is not None else []


class DedupExtractor:
    _SYSTEM_PROMPT = """\
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

    def __init__(self) -> None:
        self._prompt = _make_extract_prompt("extract.dedup", self._SYSTEM_PROMPT, "medium")

    def dedup(
        self,
        merged_entries: list[AnnotatedValue],
        ctx: HarnessContext,
    ) -> list[AnnotatedValue]:
        """LLM semantic dedup at T=0, structurally verified against `merged_entries`.
        Returns `[]` on empty/unparseable/all-rejected output — caller's tier-fallback
        handles it (no silent fallback to un-deduped inputs)."""
        if len(merged_entries) <= 1:
            return merged_entries
        envelope = [e.model_dump(exclude_none=True) for e in merged_entries]
        user_msg = (
            f"Input entries from multiple independent extraction passes:\n"
            f"```json\n{json.dumps(envelope, indent=2, default=str)}\n```\n\n"
            f"Output the consolidated set as a JSON array in the same envelope."
        )
        ctx.emit(
            "extract", "dedup_call",
            tier="parsed_json",
            n_input_entries=len(envelope),
        )
        parsed = self._prompt.call(ctx, user_msg, temperature=0.0)
        ctx.emit(
            "extract", "dedup_response",
            tier="parsed_json",
            n_entries=0 if parsed is None else len(parsed),
        )
        if not parsed:
            ctx.emit(
                "extract", "dedup_failed",
                tier="parsed_json",
                reason="empty_or_unparseable",
                n_input_entries=len(merged_entries),
            )
            return []
        kept = [e for e in parsed if _output_grounded_in_inputs(e, merged_entries)]
        n_dropped = len(parsed) - len(kept)
        if n_dropped:
            ctx.emit(
                "extract", "dedup_verifier_dropped",
                tier="parsed_json",
                n_dropped=n_dropped,
                n_parsed=len(parsed),
            )
        if not kept:
            ctx.emit(
                "extract", "dedup_failed",
                tier="parsed_json",
                reason="all_rejected_by_verifier",
                n_input_entries=len(merged_entries),
                n_output_entries=len(parsed),
            )
            return []
        return kept


class ExtractExecutor:
    """Question-driven extraction. Owns one instance of each call-site
    executor and drives the parsed_json → vision tier fallback."""

    def __init__(self) -> None:
        self._text = TextExtractor()
        self._vision = VisionExtractor()
        self._dedup = DedupExtractor()

    def _parsed_json_tier(
        self,
        refs: list[PageRef],
        ctx: HarnessContext,
        branch: RetrieveBranch,
    ) -> list[AnnotatedValue] | None:
        """Tier 1 — group-aware page fan-out → per-cell text verifier → LLM dedup.
        Refs with no parsed text are dropped (the vision tier can still pick them up)."""
        if ctx.config.golden_pages is None:
            refs = refs[: ctx.config.extract_max_pages]
        pages: list[tuple[PageRef, str]] = []
        for ref in refs:
            text = get_page_text(ref.month, ref.page)
            if not text:
                ctx.emit("extract", "no_text", tier="parsed_json", page=str(ref))
                continue
            ctx.emit(
                "extract", "got_text",
                tier="parsed_json", page=str(ref), chars=len(text),
            )
            pages.append((ref, text))
        if not pages:
            ctx.emit(
                "extract", "tier_skipped",
                tier="parsed_json", reason="no_text_from_any_ref",
            )
            return None
        all_entries = self._text.run(ctx.question, branch, pages, ctx)
        if not all_entries:
            ctx.emit(
                "extract", "tier_empty",
                tier="parsed_json", reason="all_samples_empty_after_verifier",
            )
            return None
        return all_entries or None

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
            try:
                img = render_page_b64(ref.month, ref.page, dpi=300, fmt="png")
            except Exception as e:  # noqa: BLE001 — tier-fallback; any fitz error → skip page
                ctx.emit("extract", "render_failed", tier="vision", page=str(ref), error=str(e))
                img = None
            if img:
                ctx.emit("extract", "rendered_png", tier="vision", page=str(ref))
                images.append(img)
                rendered_refs.append(ref)
            else:
                ctx.emit("extract", "no_png", tier="vision", page=str(ref))

        if not images:
            ctx.emit("extract", "tier_skipped", tier="vision", reason="no_images")
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

        # No "starting" boundary emit — the orchestrator's trace records this
        # step's boundary (and the prior retrieve step's output is these refs).

        tiers = [] if branch.visual_only else [("parsed_json", self._parsed_json_tier)]
        tiers.append(("vision", self._vision_tier))

        for tier_name, tier_fn in tiers:
            result = tier_fn(refs, ctx, branch)
            if result is None:
                continue
            ctx.emit(
                "extract", "tier_result",
                tier=tier_name,
                descriptions=[e.description for e in result],
            )
            return result

        raise StepFailed("extract", "no relevant values found across tiers")
