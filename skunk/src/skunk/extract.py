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

Three executors implement the three input modalities:
  `ExtractTextExecutor`   — parsed-table / OCR text.
  `ExtractVisionExecutor` — rendered page images.
  `ExtractDedupExecutor`  — consolidates redundant entries from multiple
                            sampling passes; cannot invent values, only
                            select representatives.
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
from dataclasses import dataclass

from skunk.common import HarnessContext
from skunk.plan import AnnotatedValue, PageRef
from skunk.executor import SkunkExecutor
from skunk.operator import OpNode, StepFailed

# ---------------------------------------------------------------------------
# Static system prompt blocks
# ---------------------------------------------------------------------------

_TEXT_SYSTEM = """\
You are a precise data extraction assistant. You receive a question and the
text of one or more pages. Emit every value that could plausibly answer the
question. Do not compute or transform — extract only what is printed.

Output a single JSON ARRAY of entries (one per distinct datum). Each entry
has one of three shapes — pick the smallest that fits:

Scalar:
  {"description": "...", "tag": "...", "kind": "scalar",
   "value": <num|str>, "unit": "<unit>"}

Vector (1-D series, one varying dim):
  {"description": "...", "tag": "...", "kind": "vector",
   "index_name": "month",
   "value": {"1942-03": 3515, "1942-04": 3939, ...},
   "unit": "usd_millions",
   "expected_index_range": "1942-03..1948-10"}

Table (2-D grid, two varying dims):
  {"description": "...", "tag": "...", "kind": "table",
   "row_name": "year", "col_name": "month",
   "value": {"1942": {"03": 3515, ...}, "1943": {"03": 7746, ...}},
   "unit": "usd_millions"}

Bias toward vector for time series even when the page prints a 2-D
year×month grid: if the question reads as a 1-D series ("every month from
X to Y"), emit ONE vector with combined ISO keys like "1942-03". Use table
only when the question genuinely compares rows vs columns.

Cells MUST be primitive (number or string). Nesting is rejected. Example
of what NOT to emit:
  "value": {"1942": [3515, 3939]}   ← cell is a list; emit kind="table" instead.
For a third axis, emit multiple separate entries.

Field rules:
- description: natural-language label that uniquely identifies this datum
  (series + period + sub-category + any other distinguishing context).
  Prefer the page's exact printed row label / column header / caption
  phrase. Example: "Total US national defense expenditures, monthly, CY1940".
- tag: short snake_case selection key shaped <series>:<period>, lowercase
  ASCII, no spaces. Example: "national_defense_expenditures:cy1940".
  Two entries describing the same underlying series + period MUST share
  the same tag (dedup/quorum keys off this).
- expected_index_range (vector only, optional): "<first>..<last>" naming
  the FULL range the QUESTION asked for, in the same key format as `value`
  (e.g. "1969-01..1980-01"). Set when the question implies a range but
  the page only has a partial series; omit when page range == question range.
- unit: single lowercase snake_case token describing the printed scale +
  base. Match what the page prints; invent a similar token when the
  page's unit doesn't fit a typical one. Examples seen in financial
  corpora: `usd`, `usd_thousands`, `usd_millions`, `usd_billions`, `pct`,
  `count`, `year`, `rate`, `fx_rate`, `text`. Foreign currency follows
  `<iso3>_<scale>` (e.g. `jpy_billions`, `gbp_millions`). Use `mixed`
  only when one cell genuinely combines incompatible units.

Unit and scale MUST come from the printed page — a column header, caption,
parenthetical legend, or footnote like "(In millions of dollars)" or
"(In billions of yen)". Never guess from the magnitude of numbers: if the
page shows "74" under a "billions of yen" header, emit value=74 and
unit=jpy_billions. Do not silently rescale or convert foreign currency to
USD — leave it native; let downstream compute apply the FX rate.

Other rules:
- Every cell in a vector/table shares one unit.
- Named-entity / string answers: kind="scalar", unit="text".
- Numbers in `value` are bare — no commas, no $, no %.
- If the page has nothing relevant, return [].
- Output ONLY the JSON array — no fences, no prose.
- Verbatim grounding: every numeric value emitted MUST appear on the page
  (with or without comma separators). Computed values are rejected.
"""

_VISION_SYSTEM = """\
You are a precise data extraction assistant. You receive a question and one
or more rendered page images. Emit every visible value that could answer the
question. Do not compute or transform — extract only what is visible.

Output a JSON ARRAY of entries. Each entry has one of three shapes:

  scalar: {"description":"...","tag":"...","kind":"scalar","value":<num|str>,"unit":"..."}
  vector: {"description":"...","tag":"...","kind":"vector","index_name":"month",
           "value":{"1942-03":3515,...},"unit":"..."}
  table:  {"description":"...","tag":"...","kind":"table","row_name":"year","col_name":"month",
           "value":{"1942":{"03":3515,...},...},"unit":"..."}

Cells MUST be primitive. Nesting beyond these shapes is rejected; for a
third axis, emit multiple separate entries.

description: natural-language label uniquely identifying the datum
  (series + period + sub-category). Prefer the page's exact visible row
  label / column header / caption phrase.
tag: snake_case selection key shaped <series>:<period>
  (e.g. national_defense_expenditures:cy1940). Two entries describing the
  same series + period MUST share the same tag.

unit: single lowercase snake_case token describing the printed scale +
  base. Match what the page prints; invent a similar token when nothing
  typical fits. Examples seen in financial corpora: `usd`,
  `usd_thousands`, `usd_millions`, `usd_billions`, `pct`, `count`,
  `year`, `rate`, `fx_rate`, `text`. Foreign currency follows
  `<iso3>_<scale>` (e.g. `jpy_billions`, `gbp_millions`). Use `mixed`
  only when one cell genuinely combines incompatible units.

Unit and scale come from the page (column header, caption, parenthetical
legend, footnote). Never guess from magnitude. If "74" appears under a
"billions of yen" header, emit value=74 and unit=jpy_billions. Do not
convert foreign currency to USD — leave it native.

Other rules:
- Every cell in a vector/table shares one unit.
- Named-entity / string answers: kind="scalar", unit="text".
- Numbers in `value` are bare (no commas, $, %).
- If nothing relevant is visible, return [].
- Output ONLY the JSON array — no fences, no prose.
- Verbatim grounding: every printed numeric value emitted MUST be visibly
  printed on the page. Do not compute, sum, average, or transform.
- EXCEPTION — visual chart-feature counts: if the question asks for the
  count of features observable but not printed as a number (local maxima,
  distinct lines, labeled regions, bars exceeding a threshold), emit that
  count as kind="scalar", unit="count", with a description naming what
  was counted and on which chart/page. This is the only derived value
  category permitted.
"""

_DEDUP_SYSTEM = """\
You consolidate redundant extraction entries. Multiple independent passes
over the same pages produced overlapping entries; collapse wording
duplicates into one representative entry per distinct datum.

You are a PICKER, not a calculator. Every value, key, and cell in your
output MUST appear verbatim in some input entry. Do not compute,
aggregate, average, derive, rescale, round, reformat, or invent values
or keys. Computed values are rejected post-hoc.

Output the same JSON envelope as the inputs: an array of entries
(description, tag, kind, value, unit; index_name for vectors,
row_name + col_name for tables).

Rules:
- One output entry per distinct datum. Wording duplicates → one
  representative; prefer the clearest, most specific description.
- All cells of an output entry come from a SINGLE input entry — do not
  graft cells across inputs. If two inputs disagree on a value at the
  same key, keep both as separate output entries with disambiguating
  descriptions.
- N genuinely distinct datums → N entries.
- Output ONLY the JSON array — no fences, no commentary.
"""

class ExtractTextExecutor(SkunkExecutor):
    name: str = "extract.text"
    system_prompt: str = _TEXT_SYSTEM

class ExtractVisionExecutor(SkunkExecutor):
    name: str = "extract.vision"
    system_prompt: str = _VISION_SYSTEM

class ExtractDedupExecutor(SkunkExecutor):
    name: str = "extract.dedup"
    system_prompt: str = _DEDUP_SYSTEM

@dataclass(frozen=True)
class _BranchHints:
    """Per-branch planner advisories threaded through the extract pipeline.
    Built once in `run()` from op.args; passed as a single object to each tier
    so adding a new advisory doesn't fan out as another kwarg on five functions."""
    key: str = ""
    period: str = ""
    value_kind: str | None = None

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

def _pdf_path_for_ref(ref: PageRef) -> Path | None:
    """Resolve ref → PDF path via $OFFICEQA_PDF_DIR (skunk.page_index.pdf)."""
    if ref.month is None or ref.page is None or ref.page <= 0:
        return None
    from skunk.page_index.pdf import pdf_path_for
    return pdf_path_for(ref.month)

def _extract_pdf_text(ref: PageRef, ctx: HarnessContext) -> str | None:
    """PyMuPDF text for ref's PDF page, or None when unavailable. No disk cache."""
    pdf_path = _pdf_path_for_ref(ref)
    if pdf_path is None:
        return None
    try:
        with fitz.open(pdf_path) as doc:
            return doc[ref.page - 1].get_text()
    except Exception as e:
        ctx.emit("extract", "tier=ocr extraction failed", page=str(ref), error=str(e))
        return None

def _render_pdf_page_b64(ref: PageRef, ctx: HarnessContext) -> tuple[str, str] | None:
    """Render ref's PDF page to in-memory PNG bytes and return (mime, base64). No disk cache."""
    pdf_path = _pdf_path_for_ref(ref)
    if pdf_path is None:
        return None
    try:
        with fitz.open(pdf_path) as doc:
            pix = doc[ref.page - 1].get_pixmap(matrix=fitz.Matrix(_DPI_SCALE, _DPI_SCALE))
        return "image/png", base64.standard_b64encode(pix.tobytes("png")).decode()
    except Exception as e:
        ctx.emit("extract", "tier=vision render failed", page=str(ref), error=str(e))
        return None

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
            tag=str(entry.get("tag", "")),
            expected_index_range=str(entry.get("expected_index_range", "")),
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
                    ok = False
                    break
                for col_key, v in row.items():
                    if col_key not in s.value[row_key] or not _values_match(v, s.value[row_key][col_key]):
                        ok = False
                        break
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
    resp = ctx.llm_client.call(ExtractDedupExecutor().assemble_system_prompt(ctx), user_msg, temperature=0.0, thinking_budget=0, ctx=ctx)
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
    resp = ctx.llm_client.call(system, user, images=images, temperature=0.0, thinking_budget=-1, ctx=ctx)
    raw = resp.text
    parsed = _parse_response_raw(raw, ctx)
    ctx.emit(
        "extract",
        f"tier={tier_name} single-call",
        raw=raw[:400],
        n_entries=0 if parsed is None else len(parsed),
    )
    return parsed if parsed is not None else []

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
    returning so downstream compute sees merged vectors when possible."""
    if not entries:
        return None
    entries = _reconcile_periods(entries, ctx)
    ctx.emit("extract", f"tier={tier_name} built entries", n_entries=len(entries))
    return entries

def _sample_groups_n(
    system: str,
    question: str,
    groups: list[list[tuple[PageRef, str, str]]],
    ctx: HarnessContext,
    tier_name: str,
    *,
    hints: _BranchHints = _BranchHints(),
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

    spec_block = _constraints_block(ctx)
    shape_block = _shape_block(hints.value_kind)
    focus_block = _focus_block(hints.key, hints.period)
    group_msgs: list[str] = []
    for group in groups:
        page_blocks = [f"{header}\n{text}" for _, header, text in group]
        group_msgs.append(
            f"Question:\n{question}\n\n"
            f"{spec_block}"
            f"{shape_block}"
            f"{focus_block}"
            f"Page text:\n\n" + "\n\n".join(page_blocks)
        )

    def _one(task: tuple[int, int]) -> tuple[int, int, str, list[AnnotatedValue]]:
        group_idx, sample_idx = task
        resp = ctx.llm_client.call(
            system, group_msgs[group_idx], temperature=temperature, thinking_budget=0, ctx=ctx
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
    return the parsed AnnotatedValue lists per run.
    """
    n_samples = ctx.config.extract_n_samples
    temperature = ctx.config.extract_sample_temperature

    def _one(_i: int) -> tuple[str, list[AnnotatedValue]]:
        resp = ctx.llm_client.call(system, user, images=images, temperature=temperature, thinking_budget=0, ctx=ctx)
        raw = resp.text
        parsed = _parse_response_raw(raw, ctx)
        return raw, (parsed if parsed is not None else [])

    max_workers = max(1, min(n_samples, ctx.config.max_parallel_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_one, range(n_samples)))

    runs: list[list[AnnotatedValue]] = []
    for i, (raw, parsed) in enumerate(results):
        ctx.emit(
            "extract",
            f"tier={tier_name} sample {i + 1}/{n_samples}",
            raw=raw[:400],
            n_entries=len(parsed),
        )
        runs.append(parsed)
    return runs

def _refs_for_tier(refs: list[PageRef], ctx: HarnessContext) -> list[PageRef]:
    # Golden mode: every requested page is gold; don't apply the page cap.
    if ctx.config.golden_pages is not None:
        return refs
    return refs[:ctx.config.extract_max_pages]

def _constraints_block(ctx: HarnessContext) -> str:
    """Render Plan-level answer-shape constraints (units_out, precision,
    answer_form) as a bullet block. Empty when ctx.plan is None (tests) or
    no constraints are set."""
    plan = ctx.plan
    if plan is None:
        return ""
    lines: list[str] = []
    if plan.units_out:
        lines.append(f"- units_out: {plan.units_out}")
    if plan.precision is not None:
        lines.append(f"- precision: {plan.precision} decimal places")
    if plan.answer_form != "scalar":
        lines.append(f"- answer_form: {plan.answer_form}")
    if not lines:
        return ""
    return "Parsed question constraints (extract values aligned with these):\n" + "\n".join(lines) + "\n\n"

def _shape_block(value_kind: str | None) -> str:
    """Render the planner's expected-shape declaration. Advisory: the
    extractor should aim for this shape but return what it actually finds
    if the page is structured differently (a mismatch is logged downstream)."""
    if not value_kind:
        return ""
    return (
        "Expected shape (advisory — planner's declaration):\n"
        f"- value_kind: {value_kind}\n"
        "If the page genuinely has a different shape, return what you "
        "actually find and flag it in the entry description; downstream "
        "compute will adapt.\n\n"
    )

def _focus_block(key: str, period: str) -> str:
    """Render the retrieve branch's key/period as a focus hint. These pages
    were selected because the planner asked for this key and period;
    surface that to the extractor so it can prioritize matching entries
    (without refusing to emit related ones)."""
    if not key and not period:
        return ""
    parts = []
    if key:
        parts.append(f"key={key!r}")
    if period:
        parts.append(f"period={period!r}")
    return (
        f"Retrieve context: these pages were selected because the planner asked for "
        f"{', '.join(parts)}. Prefer emitting values matching this key/period, "
        f"but still emit related values that share the same series — downstream may "
        f"need them.\n\n"
    )

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
    for ref in _refs_for_tier(refs, ctx):
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
            ctx.emit("extract", f"tier={tier_name} no text", page=str(ref))
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

def _parsed_json_tier(refs: list[PageRef], ctx: HarnessContext,
                       hints: _BranchHints = _BranchHints()) -> list[AnnotatedValue] | None:
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
    # In golden mode, every page is required context. Skip the same-bulletin
    # adjacency grouping so the LLM sees all pages in one prompt and can
    # produce a coherent multi-month/multi-year extraction.
    if ctx.config.golden_pages is not None:
        groups = [pages]
    else:
        groups = _group_consecutive_pages(pages)
    n_samples = ctx.config.extract_n_samples
    text_system = ExtractTextExecutor().assemble_system_prompt(ctx)
    ctx.emit(
        "extract",
        f"tier=parsed_json fan-out {len(groups)}g × {n_samples}s @ T={ctx.config.extract_sample_temperature}",
        n_groups=len(groups),
        n_pages=len(pages),
        group_sizes=[len(g) for g in groups],
        total_chars=sum(len(t) for _, _, t in pages),
        system_prompt=text_system[:1500],
    )
    group_runs = _sample_groups_n(text_system, ctx.question, groups, ctx, "parsed_json",
                                    hints=hints)

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

def _ocr_tier(refs: list[PageRef], ctx: HarnessContext,
              hints: _BranchHints = _BranchHints()) -> list[AnnotatedValue] | None:
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
        f"{_constraints_block(ctx)}"
        f"{_shape_block(hints.value_kind)}"
        f"{_focus_block(hints.key, hints.period)}"
        f"Page text:\n\n" + "\n\n".join(page_blocks)
    )
    ctx.emit(
        "extract",
        "tier=ocr single-call (T=0)",
        total_chars=sum(len(b) for b in page_blocks),
        user_message=user_msg[:3000],
    )
    entries = _single_call(ExtractTextExecutor().assemble_system_prompt(ctx), user_msg, ctx, "ocr")
    verify_text = "\n\n".join(text for _, _, text in pages)
    kept = [e for e in entries if _value_in_text(e, verify_text)]
    n_dropped = len(entries) - len(kept)
    if n_dropped:
        ctx.emit("extract", f"tier=ocr verifier dropped {n_dropped}/{len(entries)}")
    return _finalize_entries(kept, ctx, "ocr")

def _vision_tier(refs: list[PageRef], ctx: HarnessContext,
                 hints: _BranchHints = _BranchHints()) -> list[AnnotatedValue] | None:
    """Tier 3 — single deterministic call (T=0) over rendered page images, with
    per-image PageRef labels in the user message so the LLM can't conflate pages.
    """
    images: list[tuple[str, str]] = []
    rendered_refs: list[PageRef] = []
    for ref in _refs_for_tier(refs, ctx):
        img = _render_pdf_page_b64(ref, ctx)
        if img:
            ctx.emit("extract", "tier=vision rendered png", page=str(ref))
            images.append(img)
            rendered_refs.append(ref)
        else:
            ctx.emit("extract", "tier=vision no png", page=str(ref))

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
        f"{_constraints_block(ctx)}"
        f"{_shape_block(hints.value_kind)}"
        f"{_focus_block(hints.key, hints.period)}"
        "Images provided in order:\n" + "\n".join(labels) + "\n\n"
        "Include the source bulletin and page in each entry's description so a downstream "
        "consumer can tell which image the value came from."
    )

    ctx.emit("extract", "tier=vision single-call (T=0)", n_images=len(images))
    entries = _single_call(ExtractVisionExecutor().assemble_system_prompt(ctx), user_msg, ctx, "vision", images=images)
    return _finalize_entries(entries, ctx, "vision")

def run(op: OpNode, prev: list[PageRef] | None, ctx: HarnessContext) -> list[AnnotatedValue]:
    visual_only = bool(op.args.get("visual_only", False))
    hints = _BranchHints(
        key=str(op.args.get("key", "") or ""),
        period=str(op.args.get("period", "") or ""),
        value_kind=op.args.get("value_kind") or None,
    )
    refs = prev if isinstance(prev, list) else []
    if not refs:
        raise StepFailed("extract", "No page refs to extract from")

    ctx.emit("extract", "starting", visual_only=visual_only,
             n_refs=len(refs), refs=[str(r) for r in refs],
             key=hints.key or None, period=hints.period or None,
             value_kind=hints.value_kind)

    if not visual_only:
        result = _parsed_json_tier(refs, ctx, hints=hints)
        if result is not None:
            ctx.emit("extract", "tier=parsed_json produced values",
                     descriptions=[e.description for e in result])
            _check_shape_match(result, hints, ctx)
            return result

        result = _ocr_tier(refs, ctx, hints=hints)
        if result is not None:
            ctx.emit("extract", "tier=ocr produced values",
                     descriptions=[e.description for e in result])
            _check_shape_match(result, hints, ctx)
            return result

    result = _vision_tier(refs, ctx, hints=hints)
    if result is not None:
        ctx.emit("extract", "tier=vision produced values",
                 descriptions=[e.description for e in result])
        _check_shape_match(result, hints, ctx)
        return result

    raise StepFailed("extract", "no relevant values found across tiers")

def _check_shape_match(
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
