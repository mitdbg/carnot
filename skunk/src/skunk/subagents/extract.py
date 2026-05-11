"""extract subagent — question-driven named-scalar extraction over page text/images.

The agent receives the user's full question plus the rendered page(s) and returns
a JSON object mapping snake_case names to *flat scalars* (number or string), each
annotated with `unit`, a verbatim page `quote`, and an optional `dims` dict of
categorical labels (e.g. {"denomination": 1, "series": "Total"}). No nested lists
or tables — a row/column structure is emitted as one scalar per cell with `dims`
disambiguating siblings.

Downstream `compute` consumes the resulting TypedValue(dtype='named') whose
`.value` is dict[str, scalar] and whose `.meta` is dict[str, NamedEntry] carrying
unit/quote/dims.

To reduce single-call variance, each tier runs the Gemini call N times at non-zero
temperature and consensus-filters: an entry is kept only if (canonical_value,
unit, canonical_dims) appears in at least _QUORUM samples.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from skunk.common.context import HarnessContext
from skunk.common.parsed_json import get_printed_page, get_text_for_pdf_page
from skunk.common.pdf_text import get_ocr_text_for_pdf_page
from skunk.common.vision import get_png
from skunk.dsl import DocHandle, NamedEntry, OpNode, PageRef, TypedValue
from skunk.subagents.base import StepFailed, call_gemini, load_image_b64

_MAX_PAGES = 5
_N_SAMPLES = 5
_QUORUM = 2
_SAMPLE_TEMPERATURE = 0.7
_MAX_QUOTES_PER_ENTRY = 3

_TEXT_SYSTEM = (
    "You are a precise data extraction assistant for U.S. Treasury Bulletins.\n"
    "You will be given the user's question and the text of one or more bulletin pages.\n\n"
    "Read the page(s) carefully and emit EVERY scalar value (number or string) that could\n"
    "plausibly be needed to answer the question. Do NOT compute, sum, average, or otherwise\n"
    "transform — only extract what is printed.\n\n"
    "FLAT SCALARS ONLY. The value of each entry MUST be a single number or a single string.\n"
    "Never emit a JSON array or a nested object as the value. If the page contains a table\n"
    "or row/column structure relevant to the question, emit ONE entry per cell you need —\n"
    "with `dims` distinguishing the siblings (see below).\n\n"
    "Output a single JSON object mapping descriptive snake_case names to entries.\n"
    "Each entry has the shape:\n"
    '  "<name>": {"value": <number or string>,\n'
    '             "unit": "<unit>",\n'
    '             "quote": "<verbatim phrase from the page>",\n'
    '             "dims": {"<dim_name>": <label>, ...}}     # optional\n\n'
    "Where:\n"
    '- "name" is a snake_case identifier that uniquely identifies the datum, including\n'
    '  any disambiguating period or sub-category (e.g. "national_defense_cy1940",\n'
    '  "total_usd_in_circulation_denom_1", "jpy_holdings_mar_2025").\n'
    '- "value" is one of:\n'
    "    a bare number (no commas, no $, no %), or\n"
    '    a JSON string (for named entities / categorical answers, with unit="text").\n'
    "  NEVER an array, list, or object.\n"
    '- "unit" is a single lowercase token from:\n'
    "    usd, usd_thousands, usd_millions, usd_billions, pct, count, year, rate, fx_rate, text, mixed\n"
    '- "quote" MUST be a verbatim substring of the page text that anchors this datum — the\n'
    "  row label, column header, caption, or surrounding phrase. Copy it character-for-character\n"
    "  (preserving punctuation, capitalization, spacing). Do NOT paraphrase. If you cannot point\n"
    "  to a concrete printed phrase, OMIT the entry — do not invent a quote.\n"
    '- "dims" (OPTIONAL) is a small dict of categorical labels that identify *this* cell\n'
    "  among siblings emitted from the same table or series. Use a small canonical vocabulary:\n"
    '    "year" (int)            — calendar / fiscal year tag\n'
    '    "month" (string YYYY-MM)\n'
    '    "denomination" (number) — face value in USD for currency-denomination rows\n'
    '    "series" (string)       — short label for the column/series this cell belongs to\n'
    '                              (e.g. "Total", "Federal Reserve notes", "U.S. notes")\n'
    '    "country" (string)\n'
    '    "sub_category" (string) — anything else that disambiguates siblings\n'
    "  Use snake_case lower-case dim names. Only include dims you actually need to distinguish\n"
    "  this scalar from other scalars you emit. Omit `dims` entirely (or pass {}) when there\n"
    "  are no siblings.\n\n"
    "Rules:\n"
    '- If the column/section header says "in thousands of dollars", report unit=usd_thousands\n'
    "  (do NOT silently rescale the printed numbers).\n"
    "- For named-entity / string answers, use unit=text and value as a JSON string.\n"
    "- If the page contains nothing relevant to the question, return {} (empty object).\n"
    "- Output ONLY the JSON object — no markdown fences, no commentary, no leading prose.\n"
    "- CRITICAL — verbatim grounding: every numeric value you emit MUST be a number that is\n"
    "  printed on the page (with or without comma separators). Do NOT compute, derive, or\n"
    "  aggregate values. A value you computed rather than copied from the page will be rejected.\n"
)

_VISION_SYSTEM = (
    "You are a precise data extraction assistant. The images show scanned pages from\n"
    "U.S. Treasury Monthly Bulletins. You will be given the user's question and the\n"
    "rendered page image(s).\n\n"
    "Read the page(s) and emit every scalar value (number or string) that could plausibly\n"
    "be needed to answer the question. Do not compute or transform — extract only what is visible.\n\n"
    "FLAT SCALARS ONLY. The `value` field MUST be a single number or a single string.\n"
    "Never emit a JSON array or a nested object. For row/column structures, emit ONE entry per\n"
    "cell you need, with `dims` distinguishing siblings.\n\n"
    "Output a single JSON object with the same shape as the text-tier prompt:\n"
    '  {"<snake_case_name>": {"value": <number or string>,\n'
    '                          "unit": "<unit>",\n'
    '                          "quote": "<verbatim text from page image>",\n'
    '                          "dims": {"<dim>": <label>, ...}}}      # dims optional\n\n'
    'The "quote" field MUST be verbatim text visible on the page image. Do NOT paraphrase.\n'
    "If you cannot point to a concrete printed phrase, OMIT the entry — do not invent a quote.\n\n"
    "Unit vocabulary: usd, usd_thousands, usd_millions, usd_billions, pct, count, year, rate, fx_rate, text, mixed.\n"
    'Dim vocabulary (use sparingly, only to distinguish siblings): "year", "month", "denomination",\n'
    '"series", "country", "sub_category".\n'
    "If the page header says values are in thousands/millions, use that as the unit; do not rescale.\n"
    "If nothing relevant is on the page, return {}.\n"
    "Output ONLY the JSON object — no fences, no prose.\n"
    "CRITICAL — verbatim grounding: every numeric value you emit must be visibly printed on\n"
    "the page. Do not compute, derive, or aggregate values.\n"
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(raw: str) -> str:
    return _FENCE_RE.sub("", raw.strip()).strip()


@dataclass
class _Sample:
    name: str
    value: Any           # int | float | str — non-scalars are dropped at parse time
    unit: str
    quote: str
    dims: dict[str, Any]


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


def _parse_response_raw(raw: str) -> list[_Sample] | None:
    """Parse one Gemini response into a list of _Sample.

    Returns None when the agent emitted an empty object {} (signal: nothing
    relevant on the page). Drops entries whose `value` isn't a scalar — we no
    longer accept lists or nested objects, so any such entry would violate the
    extract contract. Raises StepFailed on malformed JSON or malformed entries.
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
        # Contract: scalars only. Silently drop list/dict values so a single
        # model misbehavior doesn't poison the whole sample.
        if not isinstance(value, (int, float, str, bool)) or isinstance(value, bool):
            # bool is an int subclass in Python — exclude it explicitly; treasury
            # data has no genuine boolean cells. Lists/dicts are likewise rejected.
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
        ))
    return samples


def _value_in_text(value: int | float | str, text: str) -> bool:
    """Return True if `value` appears verbatim in `text`.

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
    """Bucket _Samples across runs by (canonical_value, unit, canonical_dims),
    keep buckets with count >= _QUORUM, and emit one TypedValue with parallel
    `.value` (scalars) and `.meta` (NamedEntry records).

    Returns None when no bucket reaches quorum (so the caller can try the next tier).
    """
    buckets: dict[tuple, list[_Sample]] = {}
    for run in samples_per_run:
        for s in run:
            key = (_canon(s.value), s.unit, _canon(s.dims))
            buckets.setdefault(key, []).append(s)

    kept_keys = [k for k, bucket in buckets.items() if len(bucket) >= _QUORUM]
    dropped_keys = [k for k, bucket in buckets.items() if len(bucket) < _QUORUM]

    ctx.emit(
        "extract",
        f"tier={tier_name} consensus done",
        n_runs=len(samples_per_run),
        n_buckets=len(buckets),
        n_kept=len(kept_keys),
        n_dropped_singletons=len(dropped_keys),
        quorum=_QUORUM,
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

        values[out_name] = rep_value
        rep_quote = seen_quotes[0] if seen_quotes else ""
        meta[out_name] = NamedEntry(unit=rep_unit, quote=rep_quote, dims=rep_dims)

        dims_str = f", dims={rep_dims}" if rep_dims else ""
        quote_str = " | ".join(f'"{q}"' for q in seen_quotes) if seen_quotes else "<no quote>"
        desc_parts.append(
            f"{out_name} (unit={rep_unit}{dims_str}) — {quote_str}; {len(bucket)} runs"
        )

    return TypedValue(
        value=values,
        dtype="named",
        unit="",
        desc="; ".join(desc_parts),
        meta=meta,
    )


def _sample_n(
    system: str,
    user: str,
    ctx: HarnessContext,
    tier_name: str,
    images: list[tuple[str, str]] | None = None,
) -> list[list[_Sample]]:
    """Call Gemini _N_SAMPLES times and return the parsed _Sample lists per run."""
    runs: list[list[_Sample]] = []
    for i in range(_N_SAMPLES):
        raw = call_gemini(system, user, images=images, temperature=_SAMPLE_TEMPERATURE)
        parsed = _parse_response_raw(raw)
        ctx.emit(
            "extract",
            f"tier={tier_name} sample {i + 1}/{_N_SAMPLES}",
            raw=raw[:400],
            n_entries=0 if parsed is None else len(parsed),
        )
        runs.append(parsed if parsed is not None else [])
    return runs


def _text_tier(
    refs: list[PageRef],
    ctx: HarnessContext,
    get_text_fn,
    tier_name: str,
) -> TypedValue | None:
    raw_texts: list[str] = []   # plain page text, used for verifier lookups
    texts: list[str] = []       # header + page text, sent to LLM
    for ref in refs[:_MAX_PAGES]:
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
    if not texts:
        ctx.emit("extract", f"tier={tier_name} skipped (no text from any ref)")
        return None
    user_msg = (
        f"Question:\n{ctx.question}\n\n"
        f"Page text:\n\n" + "\n\n".join(texts)
    )
    ctx.emit(
        "extract",
        f"tier={tier_name} sampling gemini {_N_SAMPLES}x @ T={_SAMPLE_TEMPERATURE}",
        total_chars=sum(len(t) for t in texts),
        system_prompt=_TEXT_SYSTEM[:1500],
        user_message=user_msg[:3000],
    )
    verify_text = "\n\n".join(raw_texts)
    runs = _sample_n(_TEXT_SYSTEM, user_msg, ctx, tier_name)
    verified: list[list[_Sample]] = []
    for i, run in enumerate(runs):
        kept = [s for s in run if _value_in_text(s.value, verify_text)]
        n_dropped = len(run) - len(kept)
        if n_dropped:
            ctx.emit("extract", f"tier={tier_name} verifier dropped {n_dropped}/{len(run)}",
                     sample_idx=i + 1)
        verified.append(kept)
    if not any(verified):
        return None
    return _consensus_to_typed_value(verified, ctx, tier_name)


def _vision_tier(refs: list[PageRef], ctx: HarnessContext) -> TypedValue | None:
    images: list[tuple[str, str]] = []
    for ref in refs[:_MAX_PAGES]:
        png = get_png(ref, ctx)
        if png:
            ctx.emit("extract", "tier=vision rendered png", page=str(ref), png=png)
            images.append(load_image_b64(png))
        else:
            ctx.emit("extract", "tier=vision no png",
                     page=str(ref), file_path=ref.file_path)

    if not images:
        ctx.emit("extract", "tier=vision skipped (no images)")
        return None

    ctx.emit(
        "extract",
        f"tier=vision sampling gemini {_N_SAMPLES}x @ T={_SAMPLE_TEMPERATURE}",
        n_images=len(images),
    )
    user_msg = (
        f"Question:\n{ctx.question}\n\n"
        f"Pages provided: {len(images)} (of {len(refs)} total refs)."
    )
    runs = _sample_n(_VISION_SYSTEM, user_msg, ctx, "vision", images=images)
    if not any(runs):
        return None
    return _consensus_to_typed_value(runs, ctx, "vision")


def run(op: OpNode, prev: DocHandle | None, ctx: HarnessContext) -> TypedValue:
    visual_only = bool(op.args.get("visual_only", False))
    refs = prev.refs if isinstance(prev, DocHandle) else []
    if not refs:
        raise StepFailed("extract", "No page refs to extract from")

    ctx.emit("extract", "starting", visual_only=visual_only,
             n_refs=len(refs), refs=[str(r) for r in refs])

    if not visual_only:
        result = _text_tier(refs, ctx, get_text_for_pdf_page, "parsed_json")
        if result is not None:
            ctx.emit("extract", "tier=parsed_json produced values",
                     names=list(result.value.keys()) if isinstance(result.value, dict) else None)
            return result

        result = _text_tier(refs, ctx, get_ocr_text_for_pdf_page, "ocr")
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
