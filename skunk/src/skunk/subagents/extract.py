"""extract subagent — question-driven extraction over page text/images.

The agent receives the user's full question plus the rendered page(s) and returns
a JSON array of entries. Each entry has one of three KINDS — `scalar`, `vector`
(1-D series, one varying dim), or `table` (2-D grid, two varying dims) — picked to
match the question's aggregation axis. Vector and table cells are always primitive
scalars; nesting beyond those shapes is rejected.

Each entry carries `description` (free-form natural-language label that uniquely
identifies the datum, including period/series/sub-category context), `unit`, and
kind-specific axis-name fields (`index_name` for vectors, `row_name`+`col_name`
for tables).

Downstream `compute` consumes the resulting list[AnnotatedValue].

Per-tier strategy:
- parsed_json (Tier 1): rich JSON-table text → N×T=0.7 sampling → per-cell
  verbatim verifier against the page text → merge all surviving entries from
  all runs into one list[AnnotatedValue] → semantic dedup via a T=0 LLM call whose
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
import os
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import fitz

from skunk.common import HarnessContext
from skunk.dsl import AnnotatedValue, DocHandle, OpNode, PageRef
from skunk.subagents.base import StepFailed

_PARSED_JSON_DEFAULT_DIR = Path.home() / "Desktop/officeqa/treasury_bulletins_parsed/jsons"


def _parsed_json_dir() -> Path:
    d = os.environ.get("OFFICEQA_PARSED_JSON_DIR")
    return Path(d) if d else _PARSED_JSON_DEFAULT_DIR


@lru_cache(maxsize=64)
def _load_parsed_doc(month_str: str) -> dict:
    year, mon = month_str.split("-")
    p = _parsed_json_dir() / f"treasury_bulletin_{year}_{mon}.json"
    if not p.exists():
        raise StepFailed("extract", f"parsed-JSON source not found: {p}")
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise StepFailed("extract", f"corrupt parsed-JSON for {month_str}: {e}") from e


@lru_cache(maxsize=64)
def _parsed_page_index(month_str: str) -> dict[int, list[dict]]:
    doc = _load_parsed_doc(month_str)
    by_page: dict[int, list[dict]] = {}
    for el in doc.get("document", {}).get("elements", []):
        bbox = el.get("bbox") or []
        if not bbox:
            continue
        pid = bbox[0].get("page_id")
        if pid is None:
            continue
        by_page.setdefault(int(pid), []).append(el)
    return by_page


def get_text_for_pdf_page(ref: PageRef, ctx: HarnessContext) -> str | None:
    """Concatenated content for ref's PDF page. HTML tables pass through verbatim.

    Raises StepFailed if the parsed-JSON source is missing or corrupt. Returns
    None when the source is healthy but this PDF page has no parsed elements.
    """
    if ref.month is None or ref.page is None:
        return None
    idx = _parsed_page_index(ref.month)
    elements = idx.get(int(ref.page))
    if not elements:
        return None
    parts = [el["content"] for el in elements if el.get("content") is not None]
    return "\n\n".join(parts) if parts else None


def get_printed_page(ref: PageRef, ctx: HarnessContext) -> str | None:
    """Reverse lookup: bulletin printed-page footer text on ref's PDF page, or None.

    Raises StepFailed if the parsed-JSON source is missing or corrupt.
    """
    if ref.month is None or ref.page is None:
        return None
    idx = _parsed_page_index(ref.month)
    for el in idx.get(int(ref.page), []):
        if el.get("type") == "page_number" and el.get("content"):
            return str(el["content"])
    return None


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
    "keys like \"1942-03\", \"1942-04\". Use kind=\"table\"  when the question asks\n"
    "to compare rows vs columns (e.g. \"compare Jan vs July across years\").\n\n"
    "Output a single JSON ARRAY of entries (one entry per distinct datum).\n\n"
    "Scalar entry:\n"
    "  {\n"
    '    "description": "<short natural-language label uniquely identifying this datum>",\n'
    '    "kind": "scalar",\n'
    '    "value": <number or string>,\n'
    '    "unit": "<unit>"\n'
    "  }\n\n"
    "Vector entry:\n"
    "  {\n"
    '    "description": "...",\n'
    '    "kind": "vector",\n'
    '    "index_name": "month",                              # the varying dim\n'
    '    "value": {"1942-03": 3515, "1942-04": 3939, ...},   # flat dict, scalar cells\n'
    '    "unit": "usd_millions"\n'
    "  }\n\n"
    "Table entry:\n"
    "  {\n"
    '    "description": "...",\n'
    '    "kind": "table",\n'
    '    "row_name": "year", "col_name": "month",\n'
    '    "value": {"1942": {"03": 3515, "04": 3939, ...},\n'
    '              "1943": {"03": 7746, "04": 7300, ...}},   # 2-level dict, scalar cells\n'
    '    "unit": "usd_millions"\n'
    "  }\n\n"
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
    '- "description" is a short natural-language label uniquely identifying the datum. It\n'
    "  must include ALL context that distinguishes this entry from siblings — the series\n"
    "  (e.g. \"New Aa corporate bonds\"), the period (e.g. \"CY1940\", \"January 1985\"), any\n"
    "  sub-category, and any other categorical labels. Examples:\n"
    "    \"Total US national defense expenditures, monthly, CY1940\"\n"
    "    \"New Aa corporate bonds, January yield percentages, CY1990–CY1999\"\n"
    "    \"Federal individual income tax receipts net of refunds, FY1929–FY1942\"\n"
    "  Be specific enough that a downstream consumer can identify the entry from\n"
    "  description alone. Where possible, prefer including the EXACT printed row label,\n"
    "  column header, or caption phrase from the page (e.g. \"Net budget outlays\",\n"
    "  \"Treasury 30-yr. bonds\") inside the description. This is a soft preference, not\n"
    "  a requirement — paraphrase only when no concise printed phrase fits.\n"
    '- "unit" is a single lowercase token describing the printed scale + base. Build it\n'
    '  from the page — common examples (not an exhaustive list):\n'
    "    usd, usd_thousands, usd_millions, usd_billions,\n"
    "    jpy_millions, jpy_billions, gbp_millions, eur_billions, cad_millions, (etc.),\n"
    "    pct, count, year, rate, fx_rate, text, mixed.\n"
    "  For foreign currency, follow the pattern <iso3>_<scale> (lowercase ISO code, e.g.\n"
    '  "jpy_billions" for "in billions of yen"). For dimensionless / qualitative fields\n'
    '  use pct, count, year, rate, fx_rate, or text. Use "mixed" only when a single cell\n'
    "  genuinely combines incompatible units.\n\n"
    "Rules:\n"
    "- The numeric scale AND currency MUST come from the printed page — a column header,\n"
    "  table caption, parenthetical legend, or footnote like \"(In millions of dollars)\"\n"
    "  or \"(In billions of yen)\". Use exactly the scale and currency the page prints;\n"
    "  do NOT guess them from the magnitude of the numbers. If the page shows \"74\"\n"
    '  under a header that says "billions of yen", emit value=74 and unit=jpy_billions —\n'
    "  even if 74 \"feels\" small or large for the quantity. Likewise, do not silently\n"
    "  convert foreign currency to USD; leave it in its native unit and let downstream\n"
    "  compute apply the FX rate.\n"
    '- If the column/section header says "in thousands of dollars", report unit=usd_thousands\n'
    "  (do NOT silently rescale the printed numbers). Every cell in a vector/table shares one unit.\n"
    "- For named-entity / string answers, use kind=\"scalar\", unit=\"text\", value as a JSON string.\n"
    "- Numbers in `value` are bare (no commas, no $, no %).\n"
    "- If the page contains nothing relevant to the question, return [] (empty array).\n"
    "- Output ONLY the JSON array — no markdown fences, no commentary, no leading prose.\n"
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
    "Output a single JSON ARRAY of entries. Each entry has this shape:\n"
    '  scalar: {"description":"...","kind":"scalar","value":<num|str>,"unit":...}\n'
    '  vector: {"description":"...","kind":"vector","index_name":"month",\n'
    '           "value":{"1942-03":3515,"1942-04":3939,...},"unit":...}\n'
    '  table:  {"description":"...","kind":"table","row_name":"year","col_name":"month",\n'
    '           "value":{"1942":{"03":3515,...},...},"unit":...}\n\n'
    "DO NOT NEST. Vector and table cells MUST be a primitive (number or string). FORBIDDEN:\n"
    '  "value": [[3515, 3939], [4100, 4810]]            ← nested list\n'
    '  "value": {"1942": [3515, 3939]}                  ← vector cell is a list\n'
    '  "value": {"1942": {"q1": {"jan": 1043}}}         ← 3-level nest\n'
    "If you need a third axis, emit multiple separate entries — never nest.\n\n"
    'The "description" field is a short natural-language label uniquely identifying the\n'
    "datum — include series, period, sub-category, and any other distinguishing context.\n"
    "Where possible, prefer including the EXACT visible row label, column header, or\n"
    "caption phrase from the page inside the description (soft preference; paraphrase\n"
    "only when no concise printed phrase fits).\n\n"
    "Unit token — single lowercase string describing the printed scale + base.\n"
    "  Common examples (not exhaustive): usd, usd_thousands, usd_millions, usd_billions,\n"
    "  jpy_millions, jpy_billions, gbp_millions, eur_billions, cad_millions,\n"
    "  pct, count, year, rate, fx_rate, text, mixed.\n"
    "  For foreign currency, follow the pattern <iso3>_<scale> (e.g. jpy_billions for\n"
    '  "in billions of yen"). Use "mixed" only when one cell genuinely combines incompatible units.\n'
    "The numeric scale AND currency MUST come from the printed page (column header, table\n"
    "caption, parenthetical legend, or footnote). Never guess scale from value magnitude.\n"
    'If "74" appears under a header that says "billions of yen", emit value=74 and unit=jpy_billions.\n'
    "Do not silently convert foreign currency to USD; leave it native and let compute apply FX.\n"
    "If nothing relevant is visible, return [].\n"
    "Output ONLY the JSON array — no fences, no prose.\n"
    "CRITICAL — verbatim grounding: every numeric value you emit (every cell, in any kind) must\n"
    "be visibly printed on the page. Do not compute, derive, or aggregate values.\n"
)

_DEDUP_SYSTEM = (
    "You consolidate redundant extraction entries from a financial QA pipeline.\n\n"
    "Multiple independent extraction passes over the same set of source pages produced "
    "overlapping entries. Many describe the same datum with different wording. Your job: "
    "produce a single consolidated array with one representative entry per distinct datum.\n\n"
    "YOU ARE A PICKER, NOT A CALCULATOR.\n"
    "You may ONLY select a representative entry (or a subset of cells from a single input "
    "entry) — you may NOT compute, average, sum, sort, scale, format, infer, or otherwise "
    "create any new value. Every value, key, and cell in your output MUST appear verbatim "
    "in some input entry. Any computed value will be rejected post-hoc.\n\n"
    "Output the SAME JSON envelope as the inputs: a JSON ARRAY of entries. Each entry has\n"
    "description, kind, value, unit, and optionally index_name (vector) or row_name+col_name\n"
    "(table).\n\n"
    "Rules:\n"
    "- Keep ONE entry per distinct datum. Merge wording duplicates into a single representative.\n"
    "- DO NOT introduce new cell values. Every cell value in your output (scalar value, "
    "vector cell, table cell) MUST come VERBATIM from an input entry. Pick a representative; "
    "do not invent. Do not pick the mean / median / sum / etc. as a representative.\n"
    "- DO NOT compute, aggregate, transform, derive, rescale, round, or reformat.\n"
    "- DO NOT add cell keys not present in any input entry.\n"
    "- For each output entry, all of its cells must come from a SINGLE input entry — "
    "do not graft cells across inputs. If two inputs disagree on a cell value at the "
    "same key, keep both inputs as separate output entries with disambiguating descriptions.\n"
    "- Prefer the entry with the clearest, most specific description.\n"
    "- If everything in the inputs is redundant duplicates of a single datum, output one "
    "entry. If the inputs describe N genuinely distinct datums, output N entries.\n"
    "- Output ONLY the JSON array — no fences, no commentary.\n"
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(raw: str) -> str:
    return _FENCE_RE.sub("", raw.strip()).strip()


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


def _parse_response_raw(raw: str, ctx: HarnessContext | None = None) -> list[AnnotatedValue] | None:
    """Parse one Gemini response into a list of AnnotatedValue.

    Expected envelope: a JSON array of entries, each shaped like:
        {"description": str, "kind": "scalar"|"vector"|"table",
         "value": <payload>, "unit": str, ...kind-specific axis-name fields}

    Returns None when the LLM emitted an empty array (signal: nothing relevant
    on the page) or when the response failed to parse / wasn't an array. Drops
    individual entries whose payload doesn't match the declared kind's flat
    shape, emitting a diagnostic per drop.
    """
    cleaned = _strip_fences(raw)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        if ctx is not None:
            ctx.emit("extract", "rejected unparseable response",
                     error=str(e), raw=cleaned[:400])
        return None
    if not isinstance(obj, list):
        if ctx is not None:
            ctx.emit("extract", "rejected non-array response",
                     got=type(obj).__name__, raw=cleaned[:400])
        return None
    if not obj:
        return None

    entries: list[AnnotatedValue] = []
    for i, entry in enumerate(obj):
        if not isinstance(entry, dict) or "value" not in entry:
            if ctx is not None:
                ctx.emit("extract", "rejected malformed entry",
                         entry_idx=i,
                         reason="entry must be a dict containing a 'value' key")
            continue
        description = str(entry.get("description", "")).strip()
        value = entry["value"]
        kind = str(entry.get("kind", "scalar")).strip().lower() or "scalar"

        # Shape validation — silently drop misshaped entries so a single misbehaving
        # entry doesn't poison the whole sample. Emit a single diagnostic per drop.
        if kind == "scalar":
            if not _is_primitive_cell(value):
                if ctx is not None:
                    ctx.emit("extract", "rejected non-scalar entry", entry_idx=i,
                             description=description,
                             reason=f"kind=scalar requires int|float|str, got {type(value).__name__}")
                continue
            index_name = row_name = col_name = None
        elif kind == "vector":
            if not _validate_vector_payload(value):
                if ctx is not None:
                    ctx.emit("extract", "rejected nested vector entry", entry_idx=i,
                             description=description,
                             reason="vector value must be flat dict[str, scalar]")
                continue
            index_name = entry.get("index_name")
            if not isinstance(index_name, str) or not index_name:
                if ctx is not None:
                    ctx.emit("extract", "rejected vector without index_name",
                             entry_idx=i, description=description)
                continue
            row_name = col_name = None
        elif kind == "table":
            if not _validate_table_payload(value):
                if ctx is not None:
                    ctx.emit("extract", "rejected nested table entry", entry_idx=i,
                             description=description,
                             reason="table value must be dict[str, dict[str, scalar]] (no nested containers)")
                continue
            row_name = entry.get("row_name")
            col_name = entry.get("col_name")
            if not (isinstance(row_name, str) and row_name and isinstance(col_name, str) and col_name):
                if ctx is not None:
                    ctx.emit("extract", "rejected table missing row_name/col_name",
                             entry_idx=i, description=description)
                continue
            index_name = None
        else:
            if ctx is not None:
                ctx.emit("extract", "rejected entry with unknown kind",
                         entry_idx=i, description=description, kind=kind)
            continue

        entries.append(AnnotatedValue(
            description=description,
            value=value,
            unit=str(entry.get("unit", "")),
            kind=kind,
            index_name=index_name,
            row_name=row_name,
            col_name=col_name,
        ))
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
    candidates = {str(value)}
    # float with no fractional part → also try int form (9.0 → "9")
    if isinstance(value, float) and value == int(value):
        candidates.add(str(int(value)))
    # Comma-formatted for integers ≥ 1000 (e.g. 2582 → "2,582")
    int_val = int(value)
    if abs(int_val) >= 1000:
        candidates.add(f"{int_val:,}")
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


def _entry_to_dict(e: AnnotatedValue) -> dict[str, Any]:
    """Serialize an AnnotatedValue back to the JSON envelope shape Gemini emits.
    Used to present merged candidates to the dedup LLM."""
    entry: dict[str, Any] = {
        "description": e.description,
        "kind": e.kind,
        "value": e.value,
        "unit": e.unit,
    }
    if e.kind == "vector" and e.index_name:
        entry["index_name"] = e.index_name
    elif e.kind == "table":
        if e.row_name:
            entry["row_name"] = e.row_name
        if e.col_name:
            entry["col_name"] = e.col_name
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


def _output_entry_in_inputs(output: AnnotatedValue, inputs: list[AnnotatedValue]) -> bool:
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


def _deep_values_match(a: Any, b: Any) -> bool:
    """Recursive cell-level match for scalar / vector dict / table dict-of-dicts."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_values_match(a[k], b[k]) for k in a)
    return _values_match(a, b)


_DESC_WS_RE = re.compile(r"\s+")


def _norm_desc(s: str) -> str:
    return _DESC_WS_RE.sub(" ", s.strip().lower()) if isinstance(s, str) else ""


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


def _dedup_semantically(
    merged_entries: list[AnnotatedValue],
    ctx: HarnessContext,
) -> list[AnnotatedValue]:
    """LLM-based semantic deduplication over a flat array of AnnotatedValue.

    Sends all merged entries to Gemini at T=0 with the contract: collapse
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
    envelope = [_entry_to_dict(e) for e in merged_entries]
    user_msg = (
        f"Input entries from multiple independent extraction passes "
        f"(consolidate near-duplicates):\n"
        f"```json\n{json.dumps(envelope, indent=2, default=str)}\n```\n\n"
        f"Output the consolidated set as a JSON array in the same envelope."
    )
    ctx.emit(
        "extract",
        "tier=parsed_json dedup call (T=0)",
        n_input_entries=len(envelope),
    )
    resp = ctx.llm_client.call(_DEDUP_SYSTEM, user_msg, temperature=0.0, thinking_budget=0)
    raw = resp.text
    parsed = _parse_response_raw(raw, ctx)
    ctx.emit(
        "extract",
        "tier=parsed_json dedup response",
        raw=raw[:400],
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


def _single_call(
    system: str,
    user: str,
    ctx: HarnessContext,
    tier_name: str,
    images: list[tuple[str, str]] | None = None,
) -> list[AnnotatedValue]:
    """One deterministic Gemini call (T=0). Returns parsed entries; [] if LLM
    emitted [] or response was malformed (drops are emitted as diagnostics by
    `_parse_response_raw`)."""
    resp = ctx.llm_client.call(system, user, images=images, temperature=0.0, thinking_budget=-1)
    raw = resp.text
    parsed = _parse_response_raw(raw, ctx)
    ctx.emit(
        "extract",
        f"tier={tier_name} single-call",
        raw=raw[:400],
        n_entries=0 if parsed is None else len(parsed),
    )
    return parsed if parsed is not None else []


def _finalize_entries(
    entries: list[AnnotatedValue],
    ctx: HarnessContext,
    tier_name: str,
) -> list[AnnotatedValue] | None:
    """Emit-and-log helper. Returns None when `entries` is empty so the caller
    can fall through to the next tier."""
    if not entries:
        return None
    ctx.emit("extract", f"tier={tier_name} built entries", n_entries=len(entries))
    return entries


def _sample_groups_n(
    system: str,
    question: str,
    groups: list[list[tuple[PageRef, str, str]]],
    ctx: HarnessContext,
    tier_name: str,
) -> list[list[list[AnnotatedValue]]]:
    """Per-group × per-sample fan-out. Each `group` is a list of
    (ref, header, text) tuples already known to share a bulletin and to be
    consecutive PDF pages — i.e. one continuation table. The group is sent
    to the LLM as a single concatenated user message. Returns
    `[group][sample] -> parsed entries list`.

    Groups whose pages come from independent bulletins run as independent
    prompts in parallel; consecutive same-bulletin pages stay together so
    continuation tables aren't fragmented. All (n_groups × n_samples) tasks
    fan out through the same ThreadPool, paced by the global token bucket.
    """
    n_samples = ctx.config.extract_n_samples
    temperature = ctx.config.extract_sample_temperature

    group_msgs: list[str] = []
    for group in groups:
        page_blocks = [f"{header}\n{text}" for _, header, text in group]
        group_msgs.append(
            f"Question:\n{question}\n\n"
            f"Page text:\n\n" + "\n\n".join(page_blocks)
        )

    def _one(task: tuple[int, int]) -> tuple[int, int, str, list[AnnotatedValue]]:
        group_idx, sample_idx = task
        resp = ctx.llm_client.call(
            system, group_msgs[group_idx], temperature=temperature, thinking_budget=0
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
            raw=raw[:400],
            n_entries=len(parsed),
        )
    return group_runs


def _sample_n(
    system: str,
    user: str,
    ctx: HarnessContext,
    tier_name: str,
    images: list[tuple[str, str]] | None = None,
) -> list[list[AnnotatedValue]]:
    """Single-prompt sampling (used by tiers that don't split per page, e.g.
    vision over rendered images). Call Gemini n_samples times in parallel and
    return the parsed _Sample lists per run.
    """
    n_samples = ctx.config.extract_n_samples
    temperature = ctx.config.extract_sample_temperature

    def _one(_i: int) -> tuple[str, list[_Sample]]:
        resp = ctx.llm_client.call(system, user, images=images, temperature=temperature, thinking_budget=0)
        raw = resp.text
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
) -> list[tuple[PageRef, str, str]]:
    """Pull text for each ref via `get_text_fn`. Returns a per-page list of
    (ref, header, raw text) — header is "--- PDF page N ---" (plus printed-page
    hint when known). Refs that yield no text are emitted as "no text" and
    skipped.
    """
    out: list[tuple[PageRef, str, str]] = []
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
            out.append((ref, header, text))
        else:
            ctx.emit("extract", f"tier={tier_name} no text",
                     page=str(ref), file_path=ref.file_path)
    return out


def _group_consecutive_pages(
    pages: list[tuple[PageRef, str, str]],
) -> list[list[tuple[PageRef, str, str]]]:
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


def _parsed_json_tier(refs: list[PageRef], ctx: HarnessContext) -> list[AnnotatedValue] | None:
    """Tier 1 — group-aware page fan-out → per-cell text verifier → run-quorum
    split → LLM dedup on the leftover only.

    Pages are grouped into runs of consecutive same-bulletin refs; each group
    is one concatenated prompt (so continuation tables don't fragment). Per
    sample, the LLM sees one group; the (n_groups × n_samples) calls run in
    parallel through the shared ThreadPool, paced by the Gemini token bucket.

    A structural bucket is "accepted" if ≥2 (group, sample) runs agree on its
    identity (description + kind + value + unit + axis-names); only the
    sole-run leftover hits the LLM dedup.
    """
    pages = _gather_text(refs, ctx, get_text_for_pdf_page, "parsed_json")
    if not pages:
        ctx.emit("extract", "tier=parsed_json skipped (no text from any ref)")
        return None
    groups = _group_consecutive_pages(pages)
    n_samples = ctx.config.extract_n_samples
    ctx.emit(
        "extract",
        f"tier=parsed_json fan-out {len(groups)}g × {n_samples}s @ T={ctx.config.extract_sample_temperature}",
        n_groups=len(groups),
        n_pages=len(pages),
        group_sizes=[len(g) for g in groups],
        total_chars=sum(len(t) for _, _, t in pages),
        system_prompt=_TEXT_SYSTEM[:1500],
    )
    group_runs = _sample_groups_n(_TEXT_SYSTEM, ctx.question, groups, ctx, "parsed_json")

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
    deduped = _dedup_semantically(all_entries, ctx)
    return _finalize_entries(deduped, ctx, "parsed_json")


def _ocr_tier(refs: list[PageRef], ctx: HarnessContext) -> list[AnnotatedValue] | None:
    """Tier 2 — single deterministic call (T=0) over PyMuPDF text + verbatim verifier.

    OCR text on old scans is sparse and noisy. Sampling at T=0.7 amplifies
    cross-run disagreement on noisy reads; a single T=0 call paired with the
    verbatim verifier is both cheaper and more reliable.
    """
    pages = _gather_text(refs, ctx, _extract_pdf_text, "ocr")
    if not pages:
        ctx.emit("extract", "tier=ocr skipped (no text from any ref)")
        return None
    page_blocks = [f"{header}\n{text}" for _, header, text in pages]
    user_msg = (
        f"Question:\n{ctx.question}\n\n"
        f"Page text:\n\n" + "\n\n".join(page_blocks)
    )
    ctx.emit(
        "extract",
        "tier=ocr single-call (T=0)",
        total_chars=sum(len(b) for b in page_blocks),
        user_message=user_msg[:3000],
    )
    entries = _single_call(_TEXT_SYSTEM, user_msg, ctx, "ocr")
    verify_text = "\n\n".join(text for _, _, text in pages)
    kept = [e for e in entries if _value_in_text(e, verify_text)]
    n_dropped = len(entries) - len(kept)
    if n_dropped:
        ctx.emit("extract", f"tier=ocr verifier dropped {n_dropped}/{len(entries)}")
    return _finalize_entries(kept, ctx, "ocr")


def _vision_tier(refs: list[PageRef], ctx: HarnessContext) -> list[AnnotatedValue] | None:
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
        f"Include the source bulletin and page in each entry's description so a downstream "
        f"consumer can tell which image the value came from."
    )

    ctx.emit("extract", "tier=vision single-call (T=0)", n_images=len(images))
    entries = _single_call(_VISION_SYSTEM, user_msg, ctx, "vision", images=images)
    return _finalize_entries(entries, ctx, "vision")


def run(op: OpNode, prev: DocHandle | None, ctx: HarnessContext) -> list[AnnotatedValue]:
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
                     descriptions=[e.description for e in result])
            return result

        result = _ocr_tier(refs, ctx)
        if result is not None:
            ctx.emit("extract", "tier=ocr produced values",
                     descriptions=[e.description for e in result])
            return result

    result = _vision_tier(refs, ctx)
    if result is not None:
        ctx.emit("extract", "tier=vision produced values",
                 descriptions=[e.description for e in result])
        return result

    raise StepFailed("extract", "no relevant values found across tiers")
