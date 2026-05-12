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
- parsed_json (Tier 1): rich JSON-table text → N×T=0.7 sampling → per-cell
  verbatim verifier against the page text → merge all surviving entries from
  all runs into one TypedValue → semantic dedup via a T=0 LLM call whose
  output cells are structurally re-verified against the merged inputs (the
  dedup LLM cannot invent values; it can only pick representatives).
- ocr (Tier 2): sparse PyMuPDF text → single deterministic call (T=0) + verbatim
  verifier against the OCR text.
- vision (Tier 3): rendered page images → single deterministic call (T=0). No
  text corpus to verify against, so the only safeguard is the prompt's
  verbatim-grounding instruction.
"""

from __future__ import annotations

import base64
import json
import re
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

_DEDUP_SYSTEM = (
    "You consolidate redundant extraction entries from a financial QA pipeline.\n\n"
    "Three independent sampling runs over the SAME source page produced overlapping "
    "entries. Many describe the same datum with different naming, dim labels, or shapes. "
    "Your job: produce a single consolidated set with one representative entry per "
    "distinct datum.\n\n"
    "Output the SAME JSON envelope as the inputs: a single object mapping snake_case "
    "names to entries. Each entry has kind, value, unit, quote, and optionally "
    "index_name (vector) or row_name+col_name (table) and dims.\n\n"
    "Rules:\n"
    "- Keep ONE entry per distinct datum. Merge naming/decoration duplicates into a "
    "single representative.\n"
    "- DO NOT introduce new cell values. Every cell value in your output (scalar value, "
    "vector cell, table cell) MUST come VERBATIM from an input entry. Pick a "
    "representative; do not invent.\n"
    "- DO NOT compute, aggregate, transform, derive, or rescale.\n"
    "- DO NOT add cell keys not present in any input entry.\n"
    "- For each output entry, all of its cells must come from a SINGLE input entry — "
    "do not graft cells across inputs. If two inputs disagree on a cell value at the "
    "same key, keep both inputs as separate output entries with disambiguating names.\n"
    "- Prefer the entry with the most informative dims and a clear, specific name.\n"
    "- If everything in the inputs is redundant duplicates of a single datum, output one "
    "entry. If the inputs describe N genuinely distinct datums, output N entries.\n"
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
    """Canonicalize a `dims` dict to a consistent shape.

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

    Uniform failure unit: when the LLM doesn't produce a well-formed envelope,
    we drop the bad entries (or the whole sample) and emit a diagnostic — never
    raise. Tier fallback is the only mechanism that escalates "this sample
    yielded nothing" into a step failure.

    Returns None when the agent emitted an empty object {} (signal: nothing
    relevant on the page) or when the response failed to parse / wasn't a dict.
    Drops individual entries whose payload doesn't match its declared `kind`'s
    flat shape, or whose top-level shape (must be a dict with a 'value' key)
    is wrong.
    """
    cleaned = _strip_fences(raw)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        if ctx is not None:
            ctx.emit("extract", "rejected unparseable response",
                     error=str(e), raw=cleaned[:400])
        return None
    if not isinstance(obj, dict):
        if ctx is not None:
            ctx.emit("extract", "rejected non-object response",
                     got=type(obj).__name__, raw=cleaned[:400])
        return None
    if not obj:
        return None

    samples: list[_Sample] = []
    for name, entry in obj.items():
        if not isinstance(entry, dict) or "value" not in entry:
            if ctx is not None:
                ctx.emit("extract", "rejected malformed entry",
                         name=str(name),
                         reason="entry must be a dict containing a 'value' key")
            continue
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
    `text`. For vector/table samples, all cells must match (all-or-nothing).
    One missing cell rejects the entry."""
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


def _sample_to_entry_dict(s: _Sample) -> dict[str, Any]:
    """Reverse of `_parse_response_raw`: serialize a parsed _Sample back to the JSON
    envelope Gemini was originally asked to produce. Used to present the merged
    candidates to the dedup LLM."""
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


def _values_match(a: Any, b: Any) -> bool:
    """Cell-level equivalence used by the dedup input-grounded verifier.

    Numeric a/b are compared as floats so 4 / 4.0 / "4" / "4.0" all match.
    String/string is case-insensitive trimmed compare. Bool is not used as a
    cell type (filtered by `_is_primitive_cell` at parse time).
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


def _output_entry_in_inputs(output: _Sample, inputs: list[_Sample]) -> bool:
    """Input-grounded verifier: every cell in `output` must come verbatim from
    a SINGLE input entry.

    Scalar: output's value must equal some input scalar's value.
    Vector: there must exist some input vector whose value dict is a key-wise
            superset of output's value dict (same keys → same values).
    Table:  there must exist some input table whose value dict-of-dicts is a
            (row, col)-wise superset of output's value dict-of-dicts.

    No cross-entry grafting: the dedup LLM picks a representative wholesale; it
    cannot stitch cells from different inputs together. If it wants to combine
    inputs, it should emit them as separate output entries.
    """
    if output.kind == "scalar":
        for s in inputs:
            if s.kind == "scalar" and _values_match(output.value, s.value):
                return True
            # Also allow a scalar to be sourced from a single-cell vector/table —
            # the LLM may have flattened on dedup. Match by value only.
            if s.kind == "vector" and any(_values_match(output.value, v) for v in s.value.values()):
                return True
            if s.kind == "table":
                for row in s.value.values():
                    if any(_values_match(output.value, v) for v in row.values()):
                        return True
        return False
    if output.kind == "vector":
        for s in inputs:
            if s.kind != "vector":
                continue
            ok = all(
                k in s.value and _values_match(v, s.value[k])
                for k, v in output.value.items()
            )
            if ok:
                return True
        return False
    if output.kind == "table":
        for s in inputs:
            if s.kind != "table":
                continue
            ok = True
            for row_key, row in output.value.items():
                if row_key not in s.value:
                    ok = False; break
                for col_key, v in row.items():
                    if col_key not in s.value[row_key] or not _values_match(v, s.value[row_key][col_key]):
                        ok = False; break
                if not ok:
                    break
            if ok:
                return True
        return False
    return False


def _dedup_semantically(
    merged_samples: list[_Sample],
    ctx: HarnessContext,
) -> list[_Sample]:
    """LLM-based semantic deduplication over the merged sample list.

    Sends all merged entries to Gemini at T=0 with the contract: collapse
    naming/decoration duplicates into one representative each, but every output
    cell must be verbatim-derived from a single input entry. Output is then
    structurally verified against `merged_samples`; entries that fail
    verification are dropped.

    Returns the deduped sample list, or the original `merged_samples` unchanged
    if dedup produced nothing usable (best-effort fallback so a noisy LLM dedup
    response can't lose us cells the per-sample verifier already grounded).
    """
    if len(merged_samples) <= 1:
        return merged_samples
    # Build a name-suffixed envelope (raw sample names can collide across runs;
    # JSON object keys must be unique, so disambiguate before serializing).
    envelope: dict[str, dict[str, Any]] = {}
    for s in merged_samples:
        nm = s.name
        i = 1
        while nm in envelope:
            i += 1
            nm = f"{s.name}_{i}"
        envelope[nm] = _sample_to_entry_dict(s)
    user_msg = (
        f"Input entries from 3 independent extraction runs "
        f"(consolidate near-duplicates):\n"
        f"```json\n{json.dumps(envelope, indent=2, default=str)}\n```\n\n"
        f"Output the consolidated set in the same JSON envelope."
    )
    ctx.emit(
        "extract",
        "tier=parsed_json dedup call (T=0)",
        n_input_entries=len(envelope),
    )
    raw = ctx.llm_client.call(_DEDUP_SYSTEM, user_msg, temperature=0.0, thinking_budget=0)
    parsed = _parse_response_raw(raw, ctx)
    ctx.emit(
        "extract",
        "tier=parsed_json dedup response",
        raw=raw[:400],
        n_entries=0 if parsed is None else len(parsed),
    )
    if not parsed:
        return merged_samples
    kept = [s for s in parsed if _output_entry_in_inputs(s, merged_samples)]
    n_dropped = len(parsed) - len(kept)
    if n_dropped:
        ctx.emit("extract", f"tier=parsed_json dedup verifier dropped {n_dropped}/{len(parsed)}")
    if not kept:
        return merged_samples
    return kept


def _single_call(
    system: str,
    user: str,
    ctx: HarnessContext,
    tier_name: str,
    images: list[tuple[str, str]] | None = None,
) -> list[_Sample]:
    """One deterministic Gemini call (T=0). Returns parsed samples; [] if LLM
    emitted {} or response was malformed (drops are emitted as diagnostics by
    `_parse_response_raw`)."""
    raw = ctx.llm_client.call(system, user, images=images, temperature=0.0, thinking_budget=-1)
    parsed = _parse_response_raw(raw, ctx)
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

    Disambiguates duplicate names with `_2`/`_3` suffixes. Returns None when
    `samples` is empty.
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

    def _one(_i: int) -> tuple[str, list[_Sample]]:
        raw = ctx.llm_client.call(system, user, images=images, temperature=temperature, thinking_budget=0)
        parsed = _parse_response_raw(raw, ctx)
        return raw, (parsed if parsed is not None else [])

    max_workers = max(1, min(n_samples, ctx.config.max_parallel_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_one, range(n_samples)))

    runs: list[list[_Sample]] = []
    for i, (raw, parsed) in enumerate(results):
        ctx.emit(
            "extract",
            f"tier={tier_name} sample {i + 1}/{n_samples}",
            raw=raw[:400],
            n_entries=len(parsed),
        )
        runs.append(parsed)
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
    """Tier 1 — N×T=0.7 sampling → per-cell page-text verifier → merge across
    samples → LLM-based semantic dedup grounded against the merged inputs.

    Replaces the older all-or-nothing envelope consensus + resolver. The
    per-sample verifier already grounds every cell to the page; merge gives
    union recall; dedup collapses near-duplicate envelopes without inventing
    values (each dedup output entry must match some single merged input entry
    cell-for-cell).
    """
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
    merged: list[_Sample] = []
    for i, run in enumerate(runs):
        kept = [s for s in run if _value_in_text(s, verify_text)]
        n_dropped = len(run) - len(kept)
        if n_dropped:
            ctx.emit("extract", f"tier=parsed_json verifier dropped {n_dropped}/{len(run)}",
                     sample_idx=i + 1)
        merged.extend(kept)
    if not merged:
        ctx.emit("extract", "tier=parsed_json all samples empty after verifier")
        return None
    ctx.emit("extract", "tier=parsed_json merged",
             n_samples=len(runs), n_entries=len(merged))
    deduped = _dedup_semantically(merged, ctx)
    return _samples_to_typed_value(deduped, ctx, "parsed_json")


def _ocr_tier(refs: list[PageRef], ctx: HarnessContext) -> TypedValue | None:
    """Tier 2 — single deterministic call (T=0) over PyMuPDF text + verbatim verifier.

    OCR text on old scans is sparse and noisy. Sampling at T=0.7 amplifies
    cross-run disagreement on noisy reads; a single T=0 call paired with the
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
