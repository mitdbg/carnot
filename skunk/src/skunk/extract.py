"""extract operator — question-driven extraction over page text/images.

Takes the user's question plus a set of retrieved pages, returns a
`list[AnnotatedValue]` for the downstream compute step. Each entry has one
of three shapes:

  scalar  — a single number or string.
  vector  — a 1-D series indexed by one varying dim.
  table   — a 2-D grid indexed by two varying dims (row × col).

Entries carry `description`, `unit`, `tag`, and shape-specific axis labels
(`index_name` for vectors; `row_name` + `col_name` for tables). Cells are
always primitive scalars — deeper nesting is rejected.

Three call-site executors implement the three input modalities:
  `ExtractTextExecutor`   — parsed-table / OCR text.
  `ExtractVisionExecutor` — rendered page images.
  `ExtractDedupExecutor`  — consolidates redundant entries from multiple
                            sampling passes; cannot invent values, only
                            select representatives.

`ExtractOperator` owns one instance of each and drives the tier dispatch.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from skunk.common import parse_json_response
from skunk.errors import StepFailed
from skunk.executor import SkunkExecutor
from skunk.models import AnnotatedValue, HarnessContext, PageRef
from skunk.pdf_prep import (
    extract_pdf_text,
    get_printed_page,
    get_text_for_pdf_page,
    render_pdf_page_b64,
)


@dataclass(frozen=True)
class _BranchHints:
    """Per-branch planner advisories threaded through the extract pipeline.
    Built once in `ExtractOperator.run` from its kwargs; passed as a single
    object to each tier so adding a new advisory doesn't fan out as another
    kwarg on five functions."""
    key: str = ""
    period: str = ""
    value_kind: str | None = None


# ---------------------------------------------------------------------------
# Response parsing + grounding verifiers (no LLM dependency).
# ---------------------------------------------------------------------------

_DESC_WS_RE = re.compile(r"\s+")


def _parse_response_raw(raw: str, ctx: HarnessContext) -> list[AnnotatedValue] | None:
    """Parse one LLM response into a list of AnnotatedValue.

    Expected envelope: a JSON array of entries validated onto `AnnotatedValue`.
    Returns None when the response failed to parse, wasn't an array, or was empty
    (signal: nothing relevant — caller falls through to the next tier). Drops
    individual entries whose payload doesn't match the kind's flat shape,
    emitting a diagnostic per drop.
    """
    obj = parse_json_response(raw)
    if obj is None:
        ctx.emit("extract", "rejected unparseable response", raw=raw)
        return None
    if not isinstance(obj, list):
        ctx.emit("extract", "rejected non-array response",
                 got=type(obj).__name__, raw=raw)
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
    """Return True if a single primitive `value` appears verbatim in `text`.

    Numeric check tries the bare repr and the comma-formatted integer form
    (e.g. 2582 → "2582" and "2,582") so the prompt's "no commas" rule for
    the value field doesn't cause false negatives against comma-formatted text.
    String check is case-insensitive substring.
    """
    if isinstance(value, str):
        return value.strip().lower() in text.lower()
    # Numeric scalar
    candidates: set[str] = {str(value)}
    # Integer-valued numeric (int, or float with no fractional part): also try
    # the int form (9.0 → "9") and the comma-formatted form for magnitudes ≥ 1000.
    # Non-integer floats (1500.5) skip both — adding "1,500" would falsely match.
    is_int_valued = isinstance(value, int) or (
        isinstance(value, float) and value == int(value)
    )
    if is_int_valued:
        iv = int(value)
        candidates.add(str(iv))
        if abs(iv) >= 1000:
            candidates.add(f"{iv:,}")
    return any(c in text for c in candidates)


def _value_in_text(entry: AnnotatedValue, text: str) -> bool:
    """Verbatim verifier: every primitive cell in `entry.value` must appear in
    `text`. For vector/table entries, all cells must match (all-or-nothing).
    One missing cell rejects the entry."""
    if entry.kind == "scalar":
        return _cell_in_text(entry.value, text)
    if entry.kind == "vector":
        return all(_cell_in_text(c, text) for c in entry.value.values())
    if entry.kind == "table":
        for row in entry.value.values():
            for cell in row.values():
                if not _cell_in_text(cell, text):
                    return False
        return True
    return False


def _values_match(a: Any, b: Any) -> bool:
    """Cell-level equivalence used by the dedup input-grounded verifier.

    Numeric a/b are compared as floats so 4 / 4.0 / "4" / "4.0" all match.
    String/string is case-insensitive trimmed compare. Bool is excluded as a
    cell type by `AnnotatedValue.from_dict`, so we don't see it here.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


def _scalar_grounded(output: AnnotatedValue, inputs: list[AnnotatedValue]) -> bool:
    """A scalar output is grounded if its value matches some input scalar,
    OR some cell of a single input vector/table (the LLM may have flattened on dedup)."""
    for s in inputs:
        if s.kind == "scalar" and _values_match(output.value, s.value):
            return True
        if s.kind == "vector" and any(_values_match(output.value, v) for v in s.value.values()):
            return True
        if s.kind == "table" and any(
            _values_match(output.value, v) for row in s.value.values() for v in row.values()
        ):
            return True
    return False


def _vector_grounded(output: AnnotatedValue, inputs: list[AnnotatedValue]) -> bool:
    """A vector output is grounded if some input vector is a key-wise superset
    (every output key exists in the input with a matching value)."""
    for s in inputs:
        if s.kind != "vector":
            continue
        if all(k in s.value and _values_match(v, s.value[k]) for k, v in output.value.items()):
            return True
    return False


def _table_grounded(output: AnnotatedValue, inputs: list[AnnotatedValue]) -> bool:
    """A table output is grounded if some input table is a (row, col)-wise
    superset (every output cell exists in the input with a matching value)."""
    for s in inputs:
        if s.kind != "table":
            continue
        ok = all(
            row_key in s.value
            and col_key in s.value[row_key]
            and _values_match(v, s.value[row_key][col_key])
            for row_key, row in output.value.items()
            for col_key, v in row.items()
        )
        if ok:
            return True
    return False


def _output_entry_in_inputs(output: AnnotatedValue, inputs: list[AnnotatedValue]) -> bool:
    """Input-grounded verifier: every cell in `output` must come verbatim from
    a SINGLE input entry. No cross-entry grafting — the dedup LLM picks a
    representative wholesale; if it wants to combine inputs, it should emit
    them as separate output entries.
    """
    if output.kind == "scalar":
        return _scalar_grounded(output, inputs)
    if output.kind == "vector":
        return _vector_grounded(output, inputs)
    if output.kind == "table":
        return _table_grounded(output, inputs)
    return False


def _deep_values_match(a: Any, b: Any) -> bool:
    """Recursive cell-level match for scalar / vector dict / table dict-of-dicts."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_values_match(a[k], b[k]) for k in a)
    return _values_match(a, b)


def _norm_desc(s: str) -> str:
    return _DESC_WS_RE.sub(" ", s.strip().lower())


def _entries_structurally_equal(a: AnnotatedValue, b: AnnotatedValue) -> bool:
    """True if two entries carry the same data: description (normalized), kind,
    unit, axis names, and (deep) value. The description is normalized
    (case-insensitive, whitespace-collapsed) before comparison since
    independent LLM runs may word-vary the same intent."""
    if _norm_desc(a.description) != _norm_desc(b.description):
        return False
    if a.kind != b.kind or a.unit != b.unit:
        return False
    if a.index_name != b.index_name or a.row_name != b.row_name or a.col_name != b.col_name:
        return False
    return _deep_values_match(a.value, b.value)


def _run_quorum_split(
    runs: list[list[AnnotatedValue]],
    quorum: int = 2,
) -> tuple[list[AnnotatedValue], list[AnnotatedValue]]:
    """Bucket entries across runs by structural identity. Buckets contributed
    to by `quorum`+ distinct runs are accepted (one representative each);
    buckets unique to a single run land in the leftover.

    Returns (accepted, leftover).
    """
    buckets: list[tuple[AnnotatedValue, set[int]]] = []
    for run_idx, run in enumerate(runs):
        for s in run:
            matched = False
            for rep, runs_seen in buckets:
                if _entries_structurally_equal(rep, s):
                    runs_seen.add(run_idx)
                    matched = True
                    break
            if not matched:
                buckets.append((s, {run_idx}))
    accepted: list[AnnotatedValue] = []
    leftover: list[AnnotatedValue] = []
    for rep, runs_seen in buckets:
        if len(runs_seen) >= quorum:
            accepted.append(rep)
        else:
            leftover.append(rep)
    return accepted, leftover


def _reconcile_periods(entries: list[AnnotatedValue], ctx: HarnessContext) -> list[AnnotatedValue]:
    """Merge same-series scalars across periods into one vector.

    Groups kind=scalar entries by (tag_prefix, unit) where tag_prefix is the
    series part of `tag` (everything before ':'). When a group has 2+ scalars
    with distinct period suffixes, replace them with a single vector keyed by
    period suffix. Entries without a tag or with non-scalar kind pass through
    unchanged.

    Reduces noise for long-vector questions where extract emitted per-bulletin
    scalars (e.g., national defense FY1940..FY1951 as 12 separate scalars).
    Downstream compute then sees one merged vector to iterate, not 12 selects.
    """
    if not entries:
        return entries
    by_group: dict[tuple[str, str], list[tuple[str, AnnotatedValue]]] = {}
    pass_through: list[AnnotatedValue] = []
    for e in entries:
        if e.kind != "scalar" or not e.tag or ":" not in e.tag:
            pass_through.append(e)
            continue
        prefix, _, period = e.tag.partition(":")
        if not prefix or not period:
            pass_through.append(e)
            continue
        by_group.setdefault((prefix, e.unit), []).append((period, e))

    merged: list[AnnotatedValue] = []
    for (prefix, unit), items in by_group.items():
        if len(items) < 2:
            for _, e in items:
                pass_through.append(e)
            continue
        # Dedup by period suffix (last writer wins; quorum has already converged).
        # Sort lexicographically — ISO-shaped period strings sort meaningfully
        # (cy1940 < cy1941, 1942-03 < 1942-04, fy1939 < fy1940).
        seen: dict[str, AnnotatedValue] = {}
        for period, e in items:
            seen[period] = e
        sorted_periods = sorted(seen.keys())
        value_dict = {p: seen[p].value for p in sorted_periods}
        first_p, last_p = sorted_periods[0], sorted_periods[-1]
        new_tag = f"{prefix}:{first_p}-{last_p}"
        # Build a description that names the series and the range
        sample_desc = seen[first_p].description or prefix
        new_desc = f"{sample_desc} (reconciled vector across {len(seen)} periods: {first_p}..{last_p})"
        merged.append(AnnotatedValue(
            description=new_desc,
            value=value_dict,
            unit=unit,
            kind="vector",
            index_name="period",
            tag=new_tag,
            expected_index_range=f"{first_p}..{last_p}",
        ))
        ctx.emit("extract", "reconciled period-scalars into vector",
                 tag_prefix=prefix, n_merged=len(seen),
                 range=f"{first_p}..{last_p}")

    return pass_through + merged


def _finalize_entries(
    entries: list[AnnotatedValue],
    ctx: HarnessContext,
    tier_name: str,
) -> list[AnnotatedValue] | None:
    """Emit-and-log helper. Returns None when `entries` is empty so the caller
    can fall through to the next tier. Applies period-reconciliation before
    returning so downstream compute sees merged vectors when possible.

    Note: `None` here intentionally conflates "tier produced nothing" with
    "tier had no relevant content" — both signal the dispatcher in
    `ExtractOperator.run` to try the next tier. Don't add an error branch
    that distinguishes them without rethinking the fallback chain."""
    if not entries:
        return None
    entries = _reconcile_periods(entries, ctx)
    ctx.emit("extract", f"tier={tier_name} built entries", n_entries=len(entries))
    return entries


def _refs_for_tier(refs: list[PageRef], ctx: HarnessContext) -> list[PageRef]:
    # Golden mode: every requested page is gold; don't apply the page cap.
    if ctx.config.golden_pages is not None:
        return refs
    return refs[:ctx.config.extract_max_pages]


def _page_header(ref: PageRef, printed_page: str | None) -> str:
    """`--- PDF page N (bulletin printed page "X") ---` separator above each
    raw page text block in the user message."""
    header = f"--- PDF page {ref.page}"
    if printed_page:
        header += f' (bulletin printed page "{printed_page}")'
    return header + " ---"


def _request_envelope(
    question: str,
    hints: _BranchHints,
    *,
    pages: list[tuple[PageRef, str | None, str]] | None = None,
    rendered_refs: list[tuple[PageRef, str | None]] | None = None,
) -> str:
    """Build the JSON-fenced request envelope at the top of each extract
    user message. Wraps structured request fields (question, optional
    expected_shape advisory, optional focus key/period, per-page or
    per-image metadata). Page text bodies are NOT included — they go as
    raw blocks after the envelope, since escaping page text inside a JSON
    string would degrade the LLM's ability to read the page.
    """
    req: dict[str, Any] = {"question": question}
    if hints.value_kind:
        req["expected_shape"] = {"value_kind": hints.value_kind}
    if hints.key or hints.period:
        focus: dict[str, str] = {}
        if hints.key:
            focus["key"] = hints.key
        if hints.period:
            focus["period"] = hints.period
        req["focus"] = focus
    if pages is not None:
        req["pages"] = [
            {"pdf_page": ref.page, "month": ref.month,
             **({"printed_page": pp} if pp else {})}
            for ref, pp, _ in pages
        ]
    if rendered_refs is not None:
        req["images"] = [
            {"image_idx": i + 1, "pdf_page": ref.page, "month": ref.month,
             **({"printed_page": pp} if pp else {})}
            for i, (ref, pp) in enumerate(rendered_refs)
        ]
    return f"```json\n{json.dumps(req, indent=2, ensure_ascii=False)}\n```"


def _gather_text(
    refs: list[PageRef],
    ctx: HarnessContext,
    get_text_fn,
    tier_name: str,
) -> list[tuple[PageRef, str | None, str]]:
    """Pull text for each ref via `get_text_fn`. Returns a per-page list of
    (ref, printed_page, raw text). `printed_page` is the bulletin's printed
    footer text or None when unknown. Refs that yield no text are emitted
    as "no text" and skipped.
    """
    out: list[tuple[PageRef, str | None, str]] = []
    for ref in _refs_for_tier(refs, ctx):
        text = get_text_fn(ref, ctx)
        if text:
            printed = get_printed_page(ref, ctx)
            ctx.emit("extract", f"tier={tier_name} got text",
                     page=str(ref), chars=len(text), printed_page=printed)
            out.append((ref, printed, text))
        else:
            ctx.emit("extract", f"tier={tier_name} no text", page=str(ref))
    return out


def _group_consecutive_pages(
    pages: list[tuple[PageRef, str | None, str]],
) -> list[list[tuple[PageRef, str | None, str]]]:
    """Bucket pages into groups where consecutive entries share the same
    bulletin (month) and adjacent 1-based page numbers. Each group is a
    continuation table that should be sent to the LLM as one prompt; non-
    adjacent or different-bulletin pages start a new group.
    """
    groups: list[list[tuple[PageRef, str, str]]] = []
    for item in pages:
        ref, _, _ = item
        if groups:
            prev_ref = groups[-1][-1][0]
            same_bulletin = ref.month is not None and ref.month == prev_ref.month
            adjacent = (
                ref.page is not None
                and prev_ref.page is not None
                and ref.page == prev_ref.page + 1
            )
            if same_bulletin and adjacent:
                groups[-1].append(item)
                continue
        groups.append([item])
    return groups


# ---------------------------------------------------------------------------
# Call-site executors — one SkunkExecutor per LLM call.
# ---------------------------------------------------------------------------

class ExtractTextExecutor(SkunkExecutor):
    name: str = "extract.text"
    system_prompt: str = """\
You extract printed values from page text. The user message contains a
JSON request envelope (with `question`, optional `expected_shape`,
optional `focus` key/period, and a `pages` array of metadata) followed
by raw page text blocks. Each raw block is preceded by
`--- PDF page N ... ---` matching one `pdf_page` in the envelope. Emit
every printed value that could plausibly answer the question. Do not
compute or transform — extract only what is printed.

## Output format

A single JSON ARRAY of entries. One entry per distinct datum. Pick the
smallest shape that fits:

Scalar:
  {"description": "...", "tag": "...", "kind": "scalar",
   "value": <num|str>, "unit": "<unit>"}

Vector (1-D series, one varying dim):
  {"description": "...", "tag": "...", "kind": "vector",
   "index_name": "<dim>",
   "value": {"<index>": <value>, ...},
   "unit": "<unit>",
   "expected_index_range": "<first>..<last>"}

Table (2-D grid, two varying dims):
  {"description": "...", "tag": "...", "kind": "table",
   "row_name": "<dim>", "col_name": "<dim>",
   "value": {"<row>": {"<col>": <value>, ...}, ...},
   "unit": "<unit>"}

Cells MUST be primitive (number or string). For a third axis, emit
multiple separate entries — nested cells are rejected.

## Field semantics

description   natural-language label that uniquely identifies the
              datum (series + period + sub-category + any other
              distinguishing context). Prefer the page's exact printed
              row label / column header / caption phrase.
tag           short snake_case selection key shaped <series>:<period>;
              lowercase ASCII, no spaces. Two entries describing the
              same underlying series + period MUST share the same tag
              — downstream dedup keys off this.
index_name    (vector only) name of the varying dimension.
row_name /
col_name      (table only) names of the two varying dimensions.
expected_index_range
              (vector only, optional) "<first>..<last>" naming the
              FULL range the QUESTION asked for, in the same key
              format as `value`. Set when the question implies a
              range but the page carries only a partial series; omit
              when page range equals question range.
unit          single lowercase snake_case token describing the printed
              scale and base. Match what the page prints; invent a
              similar token when nothing typical fits. Use `text` for
              named-entity / string answers. Use `mixed` only when
              one cell genuinely combines incompatible units.

## Other rules

- Every cell in a vector/table shares one unit.
- Numbers in `value` are bare — no commas, no $, no %.
- If the page has nothing relevant, return [].
- Output ONLY the JSON array — no fences, no prose.
- Verbatim grounding: every numeric value emitted MUST appear on the
  page (with or without comma separators). Computed values are rejected.
"""

    def sample_groups(
        self,
        question: str,
        hints: _BranchHints,
        groups: list[list[tuple[PageRef, str | None, str]]],
        ctx: HarnessContext,
        tier_name: str,
    ) -> list[list[list[AnnotatedValue]]]:
        """Per-group × per-sample fan-out. Each `group` is a list of
        (ref, printed_page, text) tuples already known to share a bulletin
        and to be consecutive PDF pages — i.e. one continuation table. The
        group is sent as one user message: a JSON request envelope (question,
        shape, focus, page metadata) followed by raw page text bodies keyed
        by `pdf_page`. Returns `[group][sample] -> parsed entries list`.
        """
        n_samples = ctx.config.extract_n_samples
        temperature = ctx.config.extract_sample_temperature
        system = self.assemble_system_prompt(ctx)

        group_msgs: list[str] = []
        for group in groups:
            envelope = _request_envelope(question, hints, pages=group)
            page_blocks = "\n\n".join(
                f"{_page_header(ref, pp)}\n{text}" for ref, pp, text in group
            )
            group_msgs.append(
                f"Request:\n{envelope}\n\n"
                f"Page text (referenced by `pdf_page` in the request above):\n\n"
                f"{page_blocks}"
            )

        def _one(task: tuple[int, int]) -> tuple[int, int, str, list[AnnotatedValue]]:
            group_idx, sample_idx = task
            resp = ctx.llm_client.call(
                system, group_msgs[group_idx],
                temperature=temperature, thinking_budget=0, ctx=ctx,
            )
            raw = resp.text
            parsed = _parse_response_raw(raw, ctx)
            return group_idx, sample_idx, raw, (parsed if parsed is not None else [])

        tasks: list[tuple[int, int]] = [
            (g, s) for g in range(len(groups)) for s in range(n_samples)
        ]
        max_workers = max(1, min(len(tasks), ctx.config.max_parallel_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_one, tasks))

        group_runs: list[list[list[AnnotatedValue]]] = [
            [[] for _ in range(n_samples)] for _ in range(len(groups))
        ]
        for group_idx, sample_idx, raw, parsed in results:
            group_runs[group_idx][sample_idx] = parsed
            n_pages = len(groups[group_idx])
            ctx.emit(
                "extract",
                f"tier={tier_name} group {group_idx + 1}/{len(groups)} "
                f"(p={n_pages}) sample {sample_idx + 1}/{n_samples}",
                raw=raw,
                n_entries=len(parsed),
            )
        return group_runs

    def call_once(
        self,
        question: str,
        hints: _BranchHints,
        pages: list[tuple[PageRef, str | None, str]],
        ctx: HarnessContext,
        tier_name: str,
    ) -> list[AnnotatedValue]:
        """One deterministic (T=0) call over a concatenated set of pages.
        Used by the OCR tier where T=0.7 sampling amplifies noise on sparse
        scans. Returns parsed entries; `[]` if response was empty/malformed."""
        envelope = _request_envelope(question, hints, pages=pages)
        page_blocks = "\n\n".join(
            f"{_page_header(ref, pp)}\n{text}" for ref, pp, text in pages
        )
        user_msg = (
            f"Request:\n{envelope}\n\n"
            f"Page text (referenced by `pdf_page` in the request above):\n\n"
            f"{page_blocks}"
        )
        ctx.emit(
            "extract",
            f"tier={tier_name} single-call (T=0)",
            total_chars=sum(len(text) for _, _, text in pages),
            user_message=user_msg,
        )
        resp = ctx.llm_client.call(
            self.assemble_system_prompt(ctx), user_msg,
            temperature=0.0, thinking_budget=-1, ctx=ctx,
        )
        raw = resp.text
        parsed = _parse_response_raw(raw, ctx)
        ctx.emit(
            "extract",
            f"tier={tier_name} single-call",
            raw=raw,
            n_entries=0 if parsed is None else len(parsed),
        )
        return parsed if parsed is not None else []


class ExtractVisionExecutor(SkunkExecutor):
    name: str = "extract.vision"
    system_prompt: str = """\
You extract visible values from rendered page images. The user message
contains a JSON request envelope (with `question`, optional
`expected_shape`, optional `focus` key/period, and an `images` array of
metadata); the rendered page images themselves arrive as attachments in
the same order as `images`. Emit every visible value that could answer
the question. Do not compute or transform — extract only what is
visible.

## Output format

A single JSON ARRAY of entries:

  scalar: {"description":"...","tag":"...","kind":"scalar",
           "value":<num|str>,"unit":"..."}
  vector: {"description":"...","tag":"...","kind":"vector",
           "index_name":"<dim>",
           "value":{"<index>":<value>,...},"unit":"..."}
  table:  {"description":"...","tag":"...","kind":"table",
           "row_name":"<dim>","col_name":"<dim>",
           "value":{"<row>":{"<col>":<value>,...},...},"unit":"..."}

Cells MUST be primitive. For a third axis, emit multiple separate
entries.

## Field semantics

description   natural-language label uniquely identifying the datum
              (series + period + sub-category). Prefer the page's
              exact visible row label / column header / caption phrase.
tag           snake_case selection key shaped <series>:<period>. Two
              entries describing the same series + period MUST share
              the same tag.
unit          lowercase snake_case token describing the printed scale
              and base. Match what the page prints; invent a similar
              token when nothing typical fits. Use `text` for named-
              entity answers; `mixed` only when one cell genuinely
              combines incompatible units.

## Other rules

- Every cell in a vector/table shares one unit.
- Numbers in `value` are bare (no commas, $, %).
- If nothing relevant is visible, return [].
- Output ONLY the JSON array — no fences, no prose.
- Verbatim grounding: every printed numeric value emitted MUST be
  visibly printed on the page.

## When derivation is permitted

One narrow carve-out from verbatim grounding: if the question asks for
the count of features observable on a chart but NOT printed as a number
(local maxima, distinct lines, labeled regions, bars exceeding a
threshold), emit that count as kind="scalar", unit="count", with a
description naming what was counted and on which chart/page. This is
the only derived category permitted; everything else must be verbatim
from the page.
"""

    def call_once(
        self,
        question: str,
        hints: _BranchHints,
        images: list[tuple[str, str]],
        rendered_refs: list[PageRef],
        ctx: HarnessContext,
    ) -> list[AnnotatedValue]:
        """One deterministic (T=0) vision call over rendered page images.
        Sends a JSON request envelope describing the images (indexed in
        order); the images themselves come through as attachments via the
        LLM client's `images=` kwarg."""
        refs_with_printed = [(ref, get_printed_page(ref, ctx)) for ref in rendered_refs]
        envelope = _request_envelope(question, hints, rendered_refs=refs_with_printed)
        user_msg = (
            f"Request:\n{envelope}\n\n"
            f"Images are attached in the same order as the `images` array above. "
            f"Include the source bulletin and page in each entry's description "
            f"so a downstream consumer can tell which image the value came from."
        )

        ctx.emit("extract", "tier=vision single-call (T=0)", n_images=len(images))
        resp = ctx.llm_client.call(
            self.assemble_system_prompt(ctx), user_msg,
            images=images, temperature=0.0, thinking_budget=-1, ctx=ctx,
        )
        raw = resp.text
        parsed = _parse_response_raw(raw, ctx)
        ctx.emit(
            "extract",
            "tier=vision single-call",
            raw=raw,
            n_entries=0 if parsed is None else len(parsed),
        )
        return parsed if parsed is not None else []


class ExtractDedupExecutor(SkunkExecutor):
    name: str = "extract.dedup"
    system_prompt: str = """\
You consolidate redundant extraction entries. Multiple independent
passes over the same pages produced overlapping entries; collapse
wording duplicates into one representative per distinct datum.

You are a PICKER, not a calculator. Every value, key, and cell in your
output MUST appear verbatim in some input entry. Do not compute,
aggregate, average, derive, rescale, round, reformat, or invent values
or keys. A structural verifier rejects computed output post-hoc.

## Input

The user message contains a JSON array of AnnotatedValue entries inside
a ```json fenced block. Each entry has shape:

  scalar: {"description":"...","tag":"...","kind":"scalar",
           "value":<num|str>,"unit":"..."}
  vector: {"description":"...","tag":"...","kind":"vector",
           "index_name":"<dim>",
           "value":{"<index>":<value>,...},"unit":"..."}
  table:  {"description":"...","tag":"...","kind":"table",
           "row_name":"<dim>","col_name":"<dim>",
           "value":{"<row>":{"<col>":<value>,...},...},"unit":"..."}

## Output format

A JSON array in the same envelope as the inputs — same per-entry
fields (description, tag, kind, value, unit; index_name for vectors,
row_name + col_name for tables).

## Rules

- One output entry per distinct datum. Wording duplicates → one
  representative; prefer the clearest, most specific description.
- All cells of an output entry come from a SINGLE input entry — do not
  graft cells across inputs. If two inputs disagree on a value at the
  same key, keep both as separate output entries with disambiguating
  descriptions.
- N genuinely distinct datums → N entries.
- Output ONLY the JSON array — no fences, no commentary.
"""

    def dedup(
        self,
        merged_entries: list[AnnotatedValue],
        ctx: HarnessContext,
    ) -> list[AnnotatedValue]:
        """LLM-based semantic deduplication over a flat array of AnnotatedValue.

        Sends all merged entries to the LLM at T=0 with the contract: collapse
        wording duplicates into one representative each, but every output cell
        must be verbatim-derived from a single input entry. Output is then
        structurally verified against `merged_entries`; entries that fail
        verification are dropped.

        Failure mode: if the LLM response can't be parsed or every output entry
        fails verification, returns `[]` after emitting a `dedup failed` event.
        The caller's tier-fallback flow takes over from there — no silent
        fallback to the un-deduped inputs.
        """
        if len(merged_entries) <= 1:
            return merged_entries
        envelope = [
            e.model_dump(exclude_none=True, exclude={"expected_index_range"})
            for e in merged_entries
        ]
        user_msg = (
            f"These entries are the residue after quorum-based dedup already "
            f"collapsed obvious overlaps; focus on genuine wording duplicates "
            f"and value conflicts.\n\n"
            f"Input entries from multiple independent extraction passes:\n"
            f"```json\n{json.dumps(envelope, indent=2, default=str)}\n```\n\n"
            f"Output the consolidated set as a JSON array in the same envelope."
        )
        ctx.emit(
            "extract",
            "tier=parsed_json dedup call (T=0)",
            n_input_entries=len(envelope),
        )
        resp = ctx.llm_client.call(
            self.assemble_system_prompt(ctx), user_msg,
            temperature=0.0, thinking_budget=0, ctx=ctx,
        )
        raw = resp.text
        parsed = _parse_response_raw(raw, ctx)
        ctx.emit(
            "extract",
            "tier=parsed_json dedup response",
            raw=raw,
            n_entries=0 if parsed is None else len(parsed),
        )
        if not parsed:
            ctx.emit("extract", "tier=parsed_json dedup failed: empty or unparseable response",
                     n_input_entries=len(merged_entries))
            return []
        kept = [e for e in parsed if _output_entry_in_inputs(e, merged_entries)]
        n_dropped = len(parsed) - len(kept)
        if n_dropped:
            ctx.emit("extract", f"tier=parsed_json dedup verifier dropped {n_dropped}/{len(parsed)}")
        if not kept:
            ctx.emit("extract", "tier=parsed_json dedup failed: every output entry rejected by verifier",
                     n_input_entries=len(merged_entries), n_output_entries=len(parsed))
            return []
        return kept


# ---------------------------------------------------------------------------
# Operator-level orchestrator — tier dispatch.
# ---------------------------------------------------------------------------

class ExtractOperator:
    """Question-driven extraction. Owns one instance of each call-site
    executor and drives the parsed_json → ocr → vision tier fallback.
    Mirrors `PlannerExecutor` at the operator boundary: one class, one
    public `run()` method, no module-level state.
    """

    def __init__(self) -> None:
        self._text = ExtractTextExecutor()
        self._vision = ExtractVisionExecutor()
        self._dedup = ExtractDedupExecutor()

    def _parsed_json_tier(
        self, refs: list[PageRef], ctx: HarnessContext, hints: _BranchHints,
    ) -> list[AnnotatedValue] | None:
        """Tier 1 — group-aware page fan-out → per-cell text verifier → run-quorum
        split → LLM dedup on the leftover only.

        Pages are grouped into runs of consecutive same-bulletin refs; each group
        is one concatenated prompt (so continuation tables don't fragment). Per
        sample, the LLM sees one group; the (n_groups × n_samples) calls run in
        parallel through the shared ThreadPool, paced by the global token bucket.

        A structural bucket is "accepted" if ≥2 (group, sample) runs agree on its
        identity (description + kind + value + unit + axis-names); only the
        sole-run leftover hits the LLM dedup.
        """
        pages = _gather_text(refs, ctx, get_text_for_pdf_page, "parsed_json")
        if not pages:
            ctx.emit("extract", "tier=parsed_json skipped (no text from any ref)")
            return None
        # In golden mode, every page is required context. Skip the same-bulletin
        # adjacency grouping so the LLM sees all pages in one prompt and can
        # produce a coherent multi-month/multi-year extraction.
        if ctx.config.golden_pages is not None:
            groups = [pages]
        else:
            groups = _group_consecutive_pages(pages)
        n_samples = ctx.config.extract_n_samples
        ctx.emit(
            "extract",
            f"tier=parsed_json fan-out {len(groups)}g × {n_samples}s @ T={ctx.config.extract_sample_temperature}",
            n_groups=len(groups),
            n_pages=len(pages),
            group_sizes=[len(g) for g in groups],
            total_chars=sum(len(t) for _, _, t in pages),
            system_prompt=self._text.assemble_system_prompt(ctx),
        )
        group_runs = self._text.sample_groups(ctx.question, hints, groups, ctx, "parsed_json")

        # Per-group verification: each entry's cell values must appear in the
        # concatenated text of the group's pages.
        kept_runs: list[list[AnnotatedValue]] = []
        for group_idx, runs in enumerate(group_runs):
            verify_text = "\n\n".join(text for _, _, text in groups[group_idx])
            for sample_idx, run in enumerate(runs):
                kept = [e for e in run if _value_in_text(e, verify_text)]
                n_dropped = len(run) - len(kept)
                if n_dropped:
                    ctx.emit(
                        "extract",
                        f"tier=parsed_json verifier dropped {n_dropped}/{len(run)}",
                        group_idx=group_idx + 1,
                        sample_idx=sample_idx + 1,
                    )
                kept_runs.append(kept)

        total_entries = sum(len(r) for r in kept_runs)
        if total_entries == 0:
            ctx.emit("extract", "tier=parsed_json all samples empty after verifier")
            return None
        ctx.emit("extract", "tier=parsed_json merged",
                 n_runs=len(kept_runs), n_entries=total_entries)

        # Two-stage consolidation:
        # 1) Quorum: if EVERY entry reaches quorum (structurally agreed by ≥2 runs),
        #    voting fully consolidates the set and we skip the LLM dedup.
        # 2) Otherwise, voting can't cleanly partition — discard the quorum result
        #    and pass the entire merged set to the LLM dedup. The dedup's
        #    pick-only verifier (`_output_entry_in_inputs`) keeps it grounded.
        accepted, leftover = _run_quorum_split(kept_runs, quorum=2)
        ctx.emit("extract", "tier=parsed_json quorum split",
                 n_accepted=len(accepted), n_leftover=len(leftover))
        if not leftover:
            ctx.emit("extract", "tier=parsed_json quorum fully consolidated; skipping dedup")
            return _finalize_entries(accepted, ctx, "parsed_json")

        all_entries = [e for run in kept_runs for e in run]
        ctx.emit("extract", "tier=parsed_json quorum incomplete; dedup over full input",
                 n_full_input=len(all_entries))
        deduped = self._dedup.dedup(all_entries, ctx)
        return _finalize_entries(deduped, ctx, "parsed_json")

    def _ocr_tier(
        self, refs: list[PageRef], ctx: HarnessContext, hints: _BranchHints,
    ) -> list[AnnotatedValue] | None:
        """Tier 2 — single deterministic call (T=0) over PyMuPDF text + verbatim verifier.

        OCR text on old scans is sparse and noisy. Sampling at T=0.7 amplifies
        cross-run disagreement on noisy reads; a single T=0 call paired with the
        verbatim verifier is both cheaper and more reliable.
        """
        pages = _gather_text(refs, ctx, extract_pdf_text, "ocr")
        if not pages:
            ctx.emit("extract", "tier=ocr skipped (no text from any ref)")
            return None
        entries = self._text.call_once(ctx.question, hints, pages, ctx, "ocr")
        verify_text = "\n\n".join(text for _, _, text in pages)
        kept = [e for e in entries if _value_in_text(e, verify_text)]
        n_dropped = len(entries) - len(kept)
        if n_dropped:
            ctx.emit("extract", f"tier=ocr verifier dropped {n_dropped}/{len(entries)}")
        return _finalize_entries(kept, ctx, "ocr")

    def _vision_tier(
        self, refs: list[PageRef], ctx: HarnessContext, hints: _BranchHints,
    ) -> list[AnnotatedValue] | None:
        """Tier 3 — single deterministic call (T=0) over rendered page images, with
        per-image PageRef labels in the user message so the LLM can't conflate pages.
        """
        images: list[tuple[str, str]] = []
        rendered_refs: list[PageRef] = []
        for ref in _refs_for_tier(refs, ctx):
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

        entries = self._vision.call_once(ctx.question, hints, images, rendered_refs, ctx)
        return _finalize_entries(entries, ctx, "vision")

    def _check_shape_match(
        self,
        entries: list[AnnotatedValue],
        hints: _BranchHints,
        ctx: HarnessContext,
    ) -> None:
        """Advisory check: log a trace event when extract returned a different
        shape than the planner declared. Never raises — compute will adapt."""
        if hints.value_kind:
            mismatched = [e.kind for e in entries if e.kind != hints.value_kind]
            if mismatched:
                ctx.emit(
                    "extract", "value_kind mismatch (advisory)",
                    expected=hints.value_kind, got=mismatched[:5],
                    n_mismatched=len(mismatched), n_total=len(entries),
                )

    def run(
        self,
        prev: list[PageRef] | None,
        ctx: HarnessContext, *,
        visual_only: bool = False, key: str = "", period: str = "",
        value_kind: str | None = None,
    ) -> list[AnnotatedValue]:
        hints = _BranchHints(key=key, period=period, value_kind=value_kind or None)
        refs = prev or []
        if not refs:
            raise StepFailed("extract", "No page refs to extract from")

        ctx.emit("extract", "starting", visual_only=visual_only,
                 n_refs=len(refs), refs=[str(r) for r in refs],
                 key=hints.key or None, period=hints.period or None,
                 value_kind=hints.value_kind)

        if not visual_only:
            result = self._parsed_json_tier(refs, ctx, hints)
            if result is not None:
                ctx.emit("extract", "tier=parsed_json produced values",
                         descriptions=[e.description for e in result])
                self._check_shape_match(result, hints, ctx)
                return result

            result = self._ocr_tier(refs, ctx, hints)
            if result is not None:
                ctx.emit("extract", "tier=ocr produced values",
                         descriptions=[e.description for e in result])
                self._check_shape_match(result, hints, ctx)
                return result

        result = self._vision_tier(refs, ctx, hints)
        if result is not None:
            ctx.emit("extract", "tier=vision produced values",
                     descriptions=[e.description for e in result])
            self._check_shape_match(result, hints, ctx)
            return result

        raise StepFailed("extract", "no relevant values found across tiers")
