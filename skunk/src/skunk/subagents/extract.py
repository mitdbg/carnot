"""extract subagent — question-driven named extraction over page text/images.

The agent receives the user's full question plus the rendered page(s) and returns
a JSON object mapping snake_case names to entries. Each entry has one of three
KINDS — `scalar`, `vector` (1-D series, one varying dim), or `table` (2-D grid,
two varying dims) — picked to match the question's aggregation axis. Vector and
table cells are always primitive scalars; nesting beyond those shapes is rejected.

Each entry carries `unit`, a verbatim page `quote`, kind-specific axis-name fields
(`index_name` for vectors, `row_name`+`col_name` for tables), and an optional
`dims` dict of categorical labels (e.g. {"series": "Budget expenditures"}). For
vectors, `dims` is shared by all cells; the varying dim moves to `index_name`.

Downstream `compute` consumes the resulting TypedValue whose `.value[k]` is the
payload (scalar / dict / dict-of-dict per kind) and `.meta[k]` is a NamedEntry
carrying unit/quote/dims/kind/axis-names.

Per-tier strategy:
- parsed_json (Tier 1): rich JSON-table text → 5×T=0.7 sampling + consensus filter
  (an entry is kept only if (canonical_value, unit, canonical_dims) appears in at
  least ctx.config.extract_quorum samples).
- ocr (Tier 2): sparse PyMuPDF text → single deterministic call (T=0) + verbatim
  verifier against the OCR text.
- vision (Tier 3): rendered page images → single deterministic call (T=0). Sampling
  at high temperature amplifies cross-run disagreement on noisy reads and makes
  consensus structurally unreachable when multiple pages are batched together.
"""

from __future__ import annotations

import base64
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import fitz

from skunk.common.context import HarnessContext
from skunk.common.parsed_json import get_printed_page, get_text_for_pdf_page
from skunk.dsl import DocHandle, NamedEntry, OpNode, PageRef, TypedValue
from skunk.subagents.base import StepFailed

_MAX_QUOTES_PER_ENTRY = 3
_DPI_SCALE = 300 / 72  # PyMuPDF base is 72 DPI; render pages at 300 DPI for the vision tier


def _extract_pdf_text(ref: PageRef, ctx: HarnessContext) -> str | None:
    """PyMuPDF text for ref's PDF page, or None when unavailable. No disk cache."""
    if ref.month is None or ref.page is None or ref.page <= 0 or not ref.file_path:
        return None
    try:
        with fitz.open(ref.file_path) as doc:
            return doc[ref.page - 1].get_text()
    except Exception as e:
        ctx.emit("extract", "tier=ocr extraction failed", page=str(ref), error=str(e))
        return None


def _render_pdf_page_b64(ref: PageRef, ctx: HarnessContext) -> tuple[str, str] | None:
    """Render ref's PDF page to in-memory PNG bytes and return (mime, base64). No disk cache."""
    if ref.month is None or ref.page is None or ref.page <= 0 or not ref.file_path:
        return None
    try:
        with fitz.open(ref.file_path) as doc:
            pix = doc[ref.page - 1].get_pixmap(matrix=fitz.Matrix(_DPI_SCALE, _DPI_SCALE))
        return "image/png", base64.standard_b64encode(pix.tobytes("png")).decode()
    except Exception as e:
        ctx.emit("extract", "tier=vision render failed", page=str(ref), error=str(e))
        return None


_TEXT_SYSTEM = (
    "You are a precise data extraction assistant for U.S. Treasury Bulletins.\n"
    "You will be given the user's question and the text of one or more bulletin pages.\n\n"
    "Read the page(s) carefully and emit EVERY value (number or string) that could plausibly\n"
    "be needed to answer the question. Do NOT compute, sum, average, or otherwise transform —\n"
    "only extract what is printed.\n\n"
    "Each entry has one of three KINDS. Pick the smallest shape that fits the question.\n\n"
    "  kind=\"scalar\"  — a single number or string. Use for lookup-one-value questions.\n"
    "  kind=\"vector\"  — a 1-D series indexed by ONE varying dim (e.g. monthly series).\n"
    "                    Use when the question aggregates/filters across one axis.\n"
    "  kind=\"table\"   — a 2-D grid indexed by TWO varying dims (rows × cols).\n"
    "                    Use only when both axes actively vary in the question.\n\n"
    "STRONG bias toward `vector` for time series. Even if the page formats the data as a\n"
    "2-D year×month grid, if the question asks about a 1-D series (\"each month from X to Y\",\n"
    "\"every quarter\", \"all values in CY1940\"), emit ONE `vector` whose `index_name` is the\n"
    "combined dim. For year+month series, use combined ISO labels: index_name=\"month\",\n"
    "keys like \"1942-03\", \"1942-04\". Use kind=\"table\" only when the question literally asks\n"
    "to compare rows vs columns (e.g. \"compare Jan vs July across years\").\n\n"
    "Output a single JSON object mapping snake_case names to entries.\n\n"
    "Scalar entry:\n"
    '  "name": {"kind": "scalar",\n'
    '           "value": <number or string>,\n'
    '           "unit": "<unit>",\n'
    '           "quote": "<verbatim phrase from the page>",\n'
    '           "dims": {"<dim>": <label>, ...}}              # optional\n\n'
    "Vector entry:\n"
    '  "name": {"kind": "vector",\n'
    '           "index_name": "month",                          # the varying dim\n'
    '           "value": {"1942-03": 3515, "1942-04": 3939, ...},  # flat dict, scalar cells\n'
    '           "unit": "usd_millions",\n'
    '           "quote": "<verbatim phrase>",\n'
    '           "dims": {"series": "Budget expenditures"}}     # shared by all cells\n\n'
    "Table entry:\n"
    '  "name": {"kind": "table",\n'
    '           "row_name": "year", "col_name": "month",\n'
    '           "value": {"1942": {"03": 3515, "04": 3939, ...},\n'
    '                     "1943": {"03": 7746, "04": 7300, ...}},   # 2-level dict, scalar cells\n'
    '           "unit": "usd_millions",\n'
    '           "quote": "<verbatim phrase>"}\n\n'
    "DO NOT NEST. Vector cells and table cells MUST be a single number or string. Examples\n"
    "of what is FORBIDDEN and will be rejected:\n"
    '  "value": [[3515, 3939], [4100, 4810]]            ← WRONG (nested list)\n'
    '  "value": {"1942": [3515, 3939, 4100]}            ← WRONG (vector cell is a list;\n'
    "                                                     emit kind=\"table\" instead)\n"
    '  "value": {"1942": {"q1": {"jan": 1043}}}         ← WRONG (3-level nest; tables are\n'
    "                                                     exactly 2 levels)\n"
    '  "value": {"03": [3515, 3500]}                    ← WRONG (vector cell is a list)\n\n'
    "If you need a third axis, emit multiple separate vector/table entries — never nest.\n\n"
    "Field rules:\n"
    '- "name" is a snake_case identifier uniquely describing the datum.\n'
    '- "unit" is a single lowercase token from:\n'
    "    usd, usd_thousands, usd_millions, usd_billions, pct, count, year, rate, fx_rate, text, mixed\n"
    '- "quote" MUST be a verbatim substring of the page text that anchors this datum (row\n'
    "  label, column header, caption, or surrounding phrase). Copy character-for-character\n"
    "  (preserving punctuation, capitalization, spacing). Do NOT paraphrase. If you cannot\n"
    "  point to a concrete printed phrase, OMIT the entry — do not invent a quote.\n"
    '- "dims" (OPTIONAL, scalar/vector) is a small dict of categorical labels distinguishing\n'
    "  *this* entry among siblings. Canonical names: year (int), month (YYYY-MM),\n"
    "  denomination (number), series (string), country (string), sub_category (string).\n"
    "  For vectors, `dims` is shared by all cells (the varying dim moves to index_name).\n\n"
    "Rules:\n"
    '- If the column/section header says "in thousands of dollars", report unit=usd_thousands\n'
    "  (do NOT silently rescale the printed numbers). Every cell in a vector/table shares one unit.\n"
    "- For named-entity / string answers, use kind=\"scalar\", unit=\"text\", value as a JSON string.\n"
    "- Numbers in `value` are bare (no commas, no $, no %).\n"
    "- If the page contains nothing relevant to the question, return {} (empty object).\n"
    "- Output ONLY the JSON object — no markdown fences, no commentary, no leading prose.\n"
    "- CRITICAL — verbatim grounding: every numeric value you emit (every cell, in any kind)\n"
    "  MUST appear on the page (with or without comma separators). Do NOT compute, derive,\n"
    "  or aggregate values. A computed value will be rejected.\n"
)

_VISION_SYSTEM = (
    "You are a precise data extraction assistant. The images show scanned pages from\n"
    "U.S. Treasury Monthly Bulletins. You will be given the user's question and the\n"
    "rendered page image(s).\n\n"
    "Read the page(s) and emit every value that could plausibly answer the question.\n"
    "Do not compute or transform — extract only what is visible.\n\n"
    "Each entry has one of three KINDS — pick the smallest shape that fits:\n"
    "  kind=\"scalar\"  — single number or string.\n"
    "  kind=\"vector\"  — 1-D series indexed by ONE varying dim. Provide `index_name`.\n"
    "  kind=\"table\"   — 2-D grid indexed by TWO varying dims. Provide `row_name`, `col_name`.\n\n"
    "Shapes (same as the text-tier prompt):\n"
    '  scalar: {"kind":"scalar","value":<num|str>,"unit":...,"quote":...,"dims":{...}}\n'
    '  vector: {"kind":"vector","index_name":"month",\n'
    '           "value":{"1942-03":3515,"1942-04":3939,...},\n'
    '           "unit":...,"quote":...,"dims":{...}}\n'
    '  table:  {"kind":"table","row_name":"year","col_name":"month",\n'
    '           "value":{"1942":{"03":3515,...},...},\n'
    '           "unit":...,"quote":...}\n\n'
    "DO NOT NEST. Vector and table cells MUST be a primitive (number or string). FORBIDDEN:\n"
    '  "value": [[3515, 3939], [4100, 4810]]            ← nested list\n'
    '  "value": {"1942": [3515, 3939]}                  ← vector cell is a list\n'
    '  "value": {"1942": {"q1": {"jan": 1043}}}         ← 3-level nest\n'
    "If you need a third axis, emit multiple separate entries — never nest.\n\n"
    'The "quote" field MUST be verbatim text visible on the page image. Do NOT paraphrase.\n'
    "If you cannot point to a concrete printed phrase, OMIT the entry — do not invent a quote.\n\n"
    "Unit vocabulary: usd, usd_thousands, usd_millions, usd_billions, pct, count, year, rate, fx_rate, text, mixed.\n"
    'Dim vocabulary (scalar/vector only): "year", "month", "denomination", "series", "country", "sub_category".\n'
    "If the page header says values are in thousands/millions, use that as the unit; do not rescale.\n"
    "If nothing relevant is on the page, return {}.\n"
    "Output ONLY the JSON object — no fences, no prose.\n"
    "CRITICAL — verbatim grounding: every numeric value you emit (every cell, in any kind) must\n"
    "be visibly printed on the page. Do not compute, derive, or aggregate values.\n"
)

_RESOLVER_SYSTEM = (
    "You are a careful adjudicator for U.S. Treasury Bulletin data extraction.\n"
    "Three independent sampling runs over the SAME page text disagreed on shape, axis "
    "names, or cell values. Your job is to pick the single best answer.\n\n"
    "You receive:\n"
    "  - the user's question\n"
    "  - the full page text\n"
    "  - the candidate entries from each run (already filtered to values that appear "
    "verbatim on the page)\n\n"
    "Rules:\n"
    "- Output the SAME JSON envelope as the extractor: a single object mapping snake_case\n"
    "  names to entries. Each entry has kind, value, unit, quote, and optionally\n"
    "  index_name (vector) or row_name+col_name (table) and dims.\n"
    "- Prefer the candidate whose shape (scalar / vector / table), index/row/col axes,\n"
    "  unit, and dim labels are the best fit for the question — not whichever run had\n"
    "  the most entries.\n"
    "- For series questions (e.g. \"each month from X to Y\"), prefer kind=\"vector\"\n"
    "  with a combined ISO index over kind=\"table\".\n"
    "- Every numeric cell in your output MUST appear verbatim on the page (with or\n"
    "  without comma separators). You may COPY values across candidates and fill gaps\n"
    "  if a value is printed on the page but missing from a candidate. Do NOT compute\n"
    "  or aggregate.\n"
    "- The \"quote\" field MUST be a verbatim substring of the page text.\n"
    "- If none of the candidates is salvageable, return {}.\n"
    "- Output ONLY the JSON object — no fences, no commentary.\n"
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(raw: str) -> str:
    return _FENCE_RE.sub("", raw.strip()).strip()


@dataclass
class _Sample:
    name: str
    value: Any           # scalar / vector dict / table dict-of-dict — shape validated at parse time
    unit: str
    quote: str
    dims: dict[str, Any]
    kind: str = "scalar"             # "scalar" | "vector" | "table"
    index_name: str | None = None    # vector only
    row_name: str | None = None      # table only
    col_name: str | None = None      # table only


def _canon_dims(dims: dict[str, Any] | None) -> dict[str, Any]:
    """Canonicalize a `dims` dict so trivial cosmetic differences don't break quorum.

    Lowercases keys, strips string labels, drops empty values. Keeps numeric labels
    as-is. Returns a fresh dict (preserves insertion order after lowercasing keys).
    """
    if not dims:
        return {}
    out: dict[str, Any] = {}
    for k, v in dims.items():
        if v is None or v == "":
            continue
        key = str(k).strip().lower()
        if isinstance(v, str):
            out[key] = v.strip()
        else:
            out[key] = v
    return out


def _is_primitive_cell(v: Any) -> bool:
    """Vector/table cells (and scalar values) must be non-bool int/float/str.
    bool is an int subclass in Python — exclude it explicitly."""
    if isinstance(v, bool):
        return False
    return isinstance(v, (int, float, str))


def _validate_vector_payload(value: Any) -> bool:
    """True iff `value` is a flat dict[str, primitive scalar] — no nesting."""
    if not isinstance(value, dict):
        return False
    for k, cell in value.items():
        if not isinstance(k, str):
            return False
        if not _is_primitive_cell(cell):
            return False
    return True


def _validate_table_payload(value: Any) -> bool:
    """True iff `value` is a 2-level dict[str, dict[str, primitive scalar]].
    Ragged column sets are allowed (e.g. a fiscal-year table where the first year
    starts in March); the no-nesting invariant holds regardless."""
    if not isinstance(value, dict):
        return False
    for r, row in value.items():
        if not isinstance(r, str) or not isinstance(row, dict):
            return False
        for c, cell in row.items():
            if not isinstance(c, str):
                return False
            if not _is_primitive_cell(cell):
                return False
    return True


def _parse_response_raw(raw: str, ctx: HarnessContext | None = None) -> list[_Sample] | None:
    """Parse one Gemini response into a list of _Sample.

    Returns None when the agent emitted an empty object {} (signal: nothing
    relevant on the page). Drops entries whose payload doesn't match its declared
    `kind`'s flat shape — second of the two no-nesting enforcement layers. Raises
    StepFailed on malformed JSON or structurally-malformed entries.
    """
    cleaned = _strip_fences(raw)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise StepFailed("extract", f"Cannot parse JSON response: {e}\nRaw: {cleaned[:400]}") from e
    if not isinstance(obj, dict):
        raise StepFailed("extract", f"Expected JSON object, got {type(obj).__name__}")
    if not obj:
        return None

    samples: list[_Sample] = []
    for name, entry in obj.items():
        if not isinstance(entry, dict) or "value" not in entry:
            raise StepFailed("extract", f"Malformed entry {name!r}: {entry!r}")
        value = entry["value"]
        kind = str(entry.get("kind", "scalar")).strip().lower() or "scalar"

        # Shape validation — silently drop misshaped entries so a single misbehaving
        # entry doesn't poison the whole sample. Emit a single diagnostic per drop.
        if kind == "scalar":
            if not _is_primitive_cell(value):
                if ctx is not None:
                    ctx.emit("extract", "rejected non-scalar entry", name=name,
                             reason=f"kind=scalar requires int|float|str, got {type(value).__name__}")
                continue
            index_name = row_name = col_name = None
        elif kind == "vector":
            if not _validate_vector_payload(value):
                if ctx is not None:
                    ctx.emit("extract", "rejected nested vector entry", name=name,
                             reason="vector value must be flat dict[str, scalar]")
                continue
            index_name = entry.get("index_name")
            if not isinstance(index_name, str) or not index_name:
                if ctx is not None:
                    ctx.emit("extract", "rejected vector without index_name", name=name)
                continue
            row_name = col_name = None
        elif kind == "table":
            if not _validate_table_payload(value):
                if ctx is not None:
                    ctx.emit("extract", "rejected nested table entry", name=name,
                             reason="table value must be dict[str, dict[str, scalar]] (no nested containers)")
                continue
            row_name = entry.get("row_name")
            col_name = entry.get("col_name")
            if not (isinstance(row_name, str) and row_name and isinstance(col_name, str) and col_name):
                if ctx is not None:
                    ctx.emit("extract", "rejected table missing row_name/col_name", name=name)
                continue
            index_name = None
        else:
            if ctx is not None:
                ctx.emit("extract", "rejected entry with unknown kind", name=name, kind=kind)
            continue

        dims_raw = entry.get("dims") or {}
        if not isinstance(dims_raw, dict):
            dims_raw = {}
        samples.append(_Sample(
            name=str(name),
            value=value,
            unit=str(entry.get("unit", "")),
            quote=str(entry.get("quote", "")),
            dims=_canon_dims(dims_raw),
            kind=kind,
            index_name=index_name,
            row_name=row_name,
            col_name=col_name,
        ))
    return samples


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
    candidates = {str(value)}
    # float with no fractional part → also try int form (9.0 → "9")
    if isinstance(value, float) and value == int(value):
        candidates.add(str(int(value)))
    # Comma-formatted for integers ≥ 1000 (e.g. 2582 → "2,582")
    int_val = int(value)
    if abs(int_val) >= 1000:
        candidates.add(f"{int_val:,}")
    return any(c in text for c in candidates)


def _value_in_text(sample: "_Sample", text: str) -> bool:
    """Verbatim verifier: every primitive cell in `sample.value` must appear in
    `text`. For vector/table samples, all cells must match (all-or-nothing). One
    missing cell rejects the entry — matches the quorum philosophy."""
    if sample.kind == "scalar":
        return _cell_in_text(sample.value, text)
    if sample.kind == "vector":
        return all(_cell_in_text(c, text) for c in sample.value.values())
    if sample.kind == "table":
        for row in sample.value.values():
            for cell in row.values():
                if not _cell_in_text(cell, text):
                    return False
        return True
    return False


def _describe_kind(kind: str, index_name: str | None, row_name: str | None,
                   col_name: str | None, value: Any) -> str:
    """Short tag describing the entry's shape, used in extract's desc string."""
    if kind == "scalar":
        return "scalar"
    if kind == "vector":
        n = len(value) if isinstance(value, dict) else 0
        return f"vector index={index_name} cells={n}"
    if kind == "table":
        n_rows = len(value) if isinstance(value, dict) else 0
        n_cols = len(next(iter(value.values()))) if n_rows and isinstance(value, dict) else 0
        return f"table rows={row_name}({n_rows}) cols={col_name}({n_cols})"
    return kind


def _canon(v: Any) -> Any:
    """Canonical form for a value, used as part of the consensus bucket key."""
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, (int, float)):
        return ("num", float(v))
    if isinstance(v, str):
        return ("str", v.strip().lower())
    if isinstance(v, list):
        return ("list", tuple(_canon(x) for x in v))
    if isinstance(v, dict):
        return ("dict", tuple(sorted((str(k), _canon(val)) for k, val in v.items())))
    return ("raw", json.dumps(v, sort_keys=True, default=str))


def _consensus_to_typed_value(
    samples_per_run: list[list[_Sample]],
    ctx: HarnessContext,
    tier_name: str,
) -> TypedValue | None:
    """All-or-nothing consensus: bucket _Samples across runs by the canonical form
    of the whole entry — (kind, index_name, row_name, col_name, canonical_value,
    unit, canonical_dims) — and keep only buckets with count >= ctx.config.extract_quorum.
    A single-cell disagreement between samples puts them in different buckets; neither
    survives unless quorum is hit on the exact match. Emits one TypedValue with
    parallel `.value` (payloads) and `.meta` (NamedEntry records).

    Returns None when no bucket reaches quorum (so the caller can try the next tier).
    """
    buckets: dict[tuple, list[_Sample]] = {}
    for run in samples_per_run:
        for s in run:
            # All-or-nothing match: kind + axis names + every cell value + unit + dims
            # must match exactly across samples. _canon canonicalizes scalars,
            # vector-dicts, and table-dict-of-dicts deterministically.
            key = (
                s.kind, s.index_name, s.row_name, s.col_name,
                _canon(s.value), s.unit, _canon(s.dims),
            )
            buckets.setdefault(key, []).append(s)

    quorum = ctx.config.extract_quorum
    kept_keys = [k for k, bucket in buckets.items() if len(bucket) >= quorum]
    dropped_keys = [k for k, bucket in buckets.items() if len(bucket) < quorum]

    ctx.emit(
        "extract",
        f"tier={tier_name} consensus done",
        n_runs=len(samples_per_run),
        n_buckets=len(buckets),
        n_kept=len(kept_keys),
        n_dropped_singletons=len(dropped_keys),
        quorum=quorum,
    )

    if not kept_keys:
        return None

    values: dict[str, Any] = {}
    meta: dict[str, NamedEntry] = {}
    desc_parts: list[str] = []

    # Deterministic ordering: by descending bucket size, then by representative name.
    def _bucket_sort_key(k: tuple) -> tuple:
        bucket = buckets[k]
        rep_name = Counter(s.name for s in bucket).most_common(1)[0][0]
        return (-len(bucket), rep_name)

    for key in sorted(kept_keys, key=_bucket_sort_key):
        bucket = buckets[key]
        rep_name = Counter(s.name for s in bucket).most_common(1)[0][0]
        rep_unit = Counter(s.unit for s in bucket).most_common(1)[0][0]
        rep_value = next(s.value for s in bucket if s.name == rep_name)
        # All samples in a bucket share canonical_dims by construction; pull the
        # representative dims from any bucket member.
        rep_dims = dict(bucket[0].dims)

        seen_quotes: list[str] = []
        for s in bucket:
            q = s.quote.strip()
            if q and q not in seen_quotes:
                seen_quotes.append(q)
            if len(seen_quotes) >= _MAX_QUOTES_PER_ENTRY:
                break

        # If the LLM picked the same name twice across the bucket, disambiguate
        # later occurrences so we don't silently overwrite.
        out_name = rep_name
        suffix = 1
        while out_name in values:
            suffix += 1
            out_name = f"{rep_name}_{suffix}"

        # Kind + axis names are part of the bucket key, so they're identical across
        # samples in this bucket; safe to pull from any member.
        rep_kind = bucket[0].kind
        rep_index_name = bucket[0].index_name
        rep_row_name = bucket[0].row_name
        rep_col_name = bucket[0].col_name

        values[out_name] = rep_value
        rep_quote = seen_quotes[0] if seen_quotes else ""
        meta[out_name] = NamedEntry(
            unit=rep_unit, quote=rep_quote, dims=rep_dims,
            kind=rep_kind,
            index_name=rep_index_name,
            row_name=rep_row_name,
            col_name=rep_col_name,
        )

        dims_str = f", dims={rep_dims}" if rep_dims else ""
        quote_str = " | ".join(f'"{q}"' for q in seen_quotes) if seen_quotes else "<no quote>"
        kind_str = _describe_kind(rep_kind, rep_index_name, rep_row_name, rep_col_name, rep_value)
        desc_parts.append(
            f"{out_name} [{kind_str}] (unit={rep_unit}{dims_str}) — {quote_str}; {len(bucket)} runs"
        )

    return TypedValue(
        value=values,
        desc="; ".join(desc_parts),
        meta=meta,
    )


def _sample_to_entry_dict(s: _Sample) -> dict[str, Any]:
    """Reverse of `_parse_response_raw`: serialize a parsed _Sample back to the JSON
    envelope Gemini was originally asked to produce. Used to present candidates to
    the resolver."""
    entry: dict[str, Any] = {"kind": s.kind, "value": s.value, "unit": s.unit, "quote": s.quote}
    if s.kind == "vector" and s.index_name:
        entry["index_name"] = s.index_name
    elif s.kind == "table":
        if s.row_name:
            entry["row_name"] = s.row_name
        if s.col_name:
            entry["col_name"] = s.col_name
    if s.dims:
        entry["dims"] = s.dims
    return entry


def _resolve_disagreement(
    verified_runs: list[list[_Sample]],
    verify_text: str,
    question: str,
    ctx: HarnessContext,
) -> TypedValue | None:
    """Tier 1 fallback when consensus produces 0 buckets but at least one run
    survived the verbatim verifier. Asks Gemini once (T=0) to pick the best entry
    from the candidates, then runs the same verifier on its output.

    TODO: tighten the merge contract. Today the prompt allows cell-level grafting
    across candidates ("you may COPY values across candidates and fill gaps"),
    which the verbatim verifier cannot police — a Frankenstein vector with cells
    from the right column for some months and the wrong column for others looks
    self-consistent to it. Restrict to per-entry pick-wholesale (still allow
    multi-entry merge across concepts) and re-evaluate.
    """
    candidates_block_parts: list[str] = []
    for i, run in enumerate(verified_runs):
        if not run:
            candidates_block_parts.append(f"--- Run {i + 1} (empty) ---")
            continue
        envelope = {s.name: _sample_to_entry_dict(s) for s in run}
        candidates_block_parts.append(
            f"--- Run {i + 1} ---\n{json.dumps(envelope, indent=2, default=str)}"
        )
    candidates_block = "\n\n".join(candidates_block_parts)
    user_msg = (
        f"Question:\n{question}\n\n"
        f"Page text:\n{verify_text}\n\n"
        f"Candidate entries from {len(verified_runs)} independent sampling runs over the "
        f"same page text:\n\n{candidates_block}\n\n"
        f"Pick the best answer and output it in the same envelope format."
    )
    ctx.emit(
        "extract",
        "tier=parsed_json resolver call (T=0)",
        n_runs=len(verified_runs),
        n_candidates=sum(len(r) for r in verified_runs),
        user_message=user_msg[:3000],
    )
    raw = ctx.llm_client.call(_RESOLVER_SYSTEM, user_msg, temperature=0.0)
    try:
        parsed = _parse_response_raw(raw, ctx)
    except StepFailed as e:
        ctx.emit("extract", "tier=parsed_json resolver parse failed", raw=raw[:400], error=str(e))
        return None
    ctx.emit(
        "extract",
        "tier=parsed_json resolver response",
        raw=raw[:400],
        n_entries=0 if parsed is None else len(parsed),
    )
    if not parsed:
        return None
    kept = [s for s in parsed if _value_in_text(s, verify_text)]
    n_dropped = len(parsed) - len(kept)
    if n_dropped:
        ctx.emit("extract", f"tier=parsed_json resolver verifier dropped {n_dropped}/{len(parsed)}")
    return _samples_to_typed_value(kept, ctx, "parsed_json_resolver")


def _single_call(
    system: str,
    user: str,
    ctx: HarnessContext,
    tier_name: str,
    images: list[tuple[str, str]] | None = None,
) -> list[_Sample]:
    """One deterministic Gemini call (T=0). Returns parsed samples; [] if LLM emitted {}."""
    raw = ctx.llm_client.call(system, user, images=images, temperature=0.0)
    try:
        parsed = _parse_response_raw(raw, ctx)
    except StepFailed:
        ctx.emit("extract", f"tier={tier_name} single-call parse failed", raw=raw[:400])
        raise
    ctx.emit(
        "extract",
        f"tier={tier_name} single-call",
        raw=raw[:400],
        n_entries=0 if parsed is None else len(parsed),
    )
    return parsed if parsed is not None else []


def _samples_to_typed_value(
    samples: list[_Sample],
    ctx: HarnessContext,
    tier_name: str,
) -> TypedValue | None:
    """Build a TypedValue from a flat sample list — no bucketing.

    Disambiguates duplicate names with `_2`/`_3` suffixes (same scheme as
    `_consensus_to_typed_value`). Returns None when `samples` is empty.
    """
    if not samples:
        return None
    values: dict[str, Any] = {}
    meta: dict[str, NamedEntry] = {}
    desc_parts: list[str] = []
    for s in samples:
        out_name = s.name
        suffix = 1
        while out_name in values:
            suffix += 1
            out_name = f"{s.name}_{suffix}"
        values[out_name] = s.value
        meta[out_name] = NamedEntry(
            unit=s.unit, quote=s.quote, dims=dict(s.dims),
            kind=s.kind,
            index_name=s.index_name,
            row_name=s.row_name,
            col_name=s.col_name,
        )
        dims_str = f", dims={s.dims}" if s.dims else ""
        quote_str = f'"{s.quote}"' if s.quote else "<no quote>"
        kind_str = _describe_kind(s.kind, s.index_name, s.row_name, s.col_name, s.value)
        desc_parts.append(f"{out_name} [{kind_str}] (unit={s.unit}{dims_str}) — {quote_str}")
    ctx.emit("extract", f"tier={tier_name} built TypedValue", n_entries=len(values))
    return TypedValue(value=values, desc="; ".join(desc_parts), meta=meta)


def _sample_n(
    system: str,
    user: str,
    ctx: HarnessContext,
    tier_name: str,
    images: list[tuple[str, str]] | None = None,
) -> list[list[_Sample]]:
    """Call Gemini n_samples times in parallel and return the parsed _Sample lists per run.

    Per-sample event emission is deferred until after the fan-out completes so the
    log order remains `1/n, 2/n, ...` regardless of which call returned first.
    """
    n_samples = ctx.config.extract_n_samples
    temperature = ctx.config.extract_sample_temperature

    def _one(_i: int) -> tuple[str, list[_Sample] | None, StepFailed | None]:
        raw = ctx.llm_client.call(system, user, images=images, temperature=temperature)
        try:
            parsed = _parse_response_raw(raw, ctx)
        except StepFailed as e:
            return raw, None, e
        return raw, (parsed if parsed is not None else []), None

    max_workers = max(1, min(n_samples, ctx.config.max_parallel_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_one, range(n_samples)))

    runs: list[list[_Sample]] = []
    n_failed = 0
    for i, (raw, parsed, err) in enumerate(results):
        if err is not None:
            n_failed += 1
            ctx.emit(
                "extract",
                f"tier={tier_name} sample {i + 1}/{n_samples} failed",
                raw=raw[:400],
                error=str(err),
            )
            runs.append([])
            continue
        ctx.emit(
            "extract",
            f"tier={tier_name} sample {i + 1}/{n_samples}",
            raw=raw[:400],
            n_entries=len(parsed),
        )
        runs.append(parsed)
    if n_failed == n_samples:
        raise StepFailed("extract", f"all {n_samples} samples failed to parse")
    return runs


def _gather_text(
    refs: list[PageRef],
    ctx: HarnessContext,
    get_text_fn,
    tier_name: str,
) -> tuple[list[str], list[str]]:
    """Pull text for each ref via `get_text_fn`. Returns (raw_texts, texts).

    raw_texts: per-page plain text (used by the verifier).
    texts:     per-page text with a `--- PDF page N ---` header for the LLM.
    Refs that yield no text are emitted as "no text" and skipped.
    """
    raw_texts: list[str] = []
    texts: list[str] = []
    for ref in refs[:ctx.config.extract_max_pages]:
        text = get_text_fn(ref, ctx)
        if text:
            printed = get_printed_page(ref, ctx)
            ctx.emit("extract", f"tier={tier_name} got text",
                     page=str(ref), chars=len(text), printed_page=printed)
            header = f"--- PDF page {ref.page}"
            if printed:
                header += f' (bulletin printed page "{printed}")'
            header += " ---"
            raw_texts.append(text)
            texts.append(f"{header}\n{text}")
        else:
            ctx.emit("extract", f"tier={tier_name} no text",
                     page=str(ref), file_path=ref.file_path)
    return raw_texts, texts


def _parsed_json_tier(refs: list[PageRef], ctx: HarnessContext) -> TypedValue | None:
    """Tier 1 — 5×T=0.7 sampling + consensus over rich JSON-derived page text."""
    raw_texts, texts = _gather_text(refs, ctx, get_text_for_pdf_page, "parsed_json")
    if not texts:
        ctx.emit("extract", "tier=parsed_json skipped (no text from any ref)")
        return None
    user_msg = (
        f"Question:\n{ctx.question}\n\n"
        f"Page text:\n\n" + "\n\n".join(texts)
    )
    ctx.emit(
        "extract",
        f"tier=parsed_json sampling gemini {ctx.config.extract_n_samples}x @ T={ctx.config.extract_sample_temperature}",
        total_chars=sum(len(t) for t in texts),
        system_prompt=_TEXT_SYSTEM[:1500],
        user_message=user_msg[:3000],
    )
    verify_text = "\n\n".join(raw_texts)
    runs = _sample_n(_TEXT_SYSTEM, user_msg, ctx, "parsed_json")
    verified: list[list[_Sample]] = []
    for i, run in enumerate(runs):
        kept = [s for s in run if _value_in_text(s, verify_text)]
        n_dropped = len(run) - len(kept)
        if n_dropped:
            ctx.emit("extract", f"tier=parsed_json verifier dropped {n_dropped}/{len(run)}",
                     sample_idx=i + 1)
        verified.append(kept)
    if not any(verified):
        return None
    result = _consensus_to_typed_value(verified, ctx, "parsed_json")
    if result is not None:
        return result
    ctx.emit("extract", "tier=parsed_json consensus empty — invoking resolver")
    return _resolve_disagreement(verified, verify_text, ctx.question, ctx)


def _ocr_tier(refs: list[PageRef], ctx: HarnessContext) -> TypedValue | None:
    """Tier 2 — single deterministic call (T=0) over PyMuPDF text + verbatim verifier.

    OCR text on old scans is sparse and noisy. Sampling 5× at T=0.7 amplifies
    disagreement without earning consensus; a single T=0 call paired with the
    verbatim verifier is both cheaper and more reliable.
    """
    raw_texts, texts = _gather_text(refs, ctx, _extract_pdf_text, "ocr")
    if not texts:
        ctx.emit("extract", "tier=ocr skipped (no text from any ref)")
        return None
    user_msg = (
        f"Question:\n{ctx.question}\n\n"
        f"Page text:\n\n" + "\n\n".join(texts)
    )
    ctx.emit(
        "extract",
        "tier=ocr single-call (T=0)",
        total_chars=sum(len(t) for t in texts),
        user_message=user_msg[:3000],
    )
    samples = _single_call(_TEXT_SYSTEM, user_msg, ctx, "ocr")
    verify_text = "\n\n".join(raw_texts)
    kept = [s for s in samples if _value_in_text(s, verify_text)]
    n_dropped = len(samples) - len(kept)
    if n_dropped:
        ctx.emit("extract", f"tier=ocr verifier dropped {n_dropped}/{len(samples)}")
    return _samples_to_typed_value(kept, ctx, "ocr")


def _vision_tier(refs: list[PageRef], ctx: HarnessContext) -> TypedValue | None:
    """Tier 3 — single deterministic call (T=0) over rendered page images, with
    per-image PageRef labels in the user message so the LLM can't conflate pages.
    """
    images: list[tuple[str, str]] = []
    rendered_refs: list[PageRef] = []
    for ref in refs[:ctx.config.extract_max_pages]:
        img = _render_pdf_page_b64(ref, ctx)
        if img:
            ctx.emit("extract", "tier=vision rendered png", page=str(ref))
            images.append(img)
            rendered_refs.append(ref)
        else:
            ctx.emit("extract", "tier=vision no png",
                     page=str(ref), file_path=ref.file_path)

    if not images:
        ctx.emit("extract", "tier=vision skipped (no images)")
        return None

    labels: list[str] = []
    for i, ref in enumerate(rendered_refs):
        printed = get_printed_page(ref, ctx)
        line = f"Image {i + 1}: bulletin {ref.month}, PDF page {ref.page}"
        if printed:
            line += f' (printed page "{printed}")'
        labels.append(line)
    user_msg = (
        f"Question:\n{ctx.question}\n\n"
        f"Images provided in order:\n" + "\n".join(labels) + "\n\n"
        f"Set dims.year and dims.month on each entry based on which image the cell came from."
    )

    ctx.emit("extract", "tier=vision single-call (T=0)", n_images=len(images))
    samples = _single_call(_VISION_SYSTEM, user_msg, ctx, "vision", images=images)
    return _samples_to_typed_value(samples, ctx, "vision")


def run(op: OpNode, prev: DocHandle | None, ctx: HarnessContext) -> TypedValue:
    visual_only = bool(op.args.get("visual_only", False))
    refs = prev.refs if isinstance(prev, DocHandle) else []
    if not refs:
        raise StepFailed("extract", "No page refs to extract from")

    ctx.emit("extract", "starting", visual_only=visual_only,
             n_refs=len(refs), refs=[str(r) for r in refs])

    if not visual_only:
        result = _parsed_json_tier(refs, ctx)
        if result is not None:
            ctx.emit("extract", "tier=parsed_json produced values",
                     names=list(result.value.keys()) if isinstance(result.value, dict) else None)
            return result

        result = _ocr_tier(refs, ctx)
        if result is not None:
            ctx.emit("extract", "tier=ocr produced values",
                     names=list(result.value.keys()) if isinstance(result.value, dict) else None)
            return result

    result = _vision_tier(refs, ctx)
    if result is not None:
        ctx.emit("extract", "tier=vision produced values",
                 names=list(result.value.keys()) if isinstance(result.value, dict) else None)
        return result

    raise StepFailed("extract", "no relevant values found across tiers")
