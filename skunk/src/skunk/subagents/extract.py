"""extract subagent — per-page tier dispatch: parsed text → OCR text → vision."""

from __future__ import annotations

import io

import pandas as pd

from skunk.common.context import HarnessContext
from skunk.common.parsed import get_parsed_text_for_page
from skunk.common.pdf_text import get_ocr_text_for_pdf_page
from skunk.common.vision import get_png
from skunk.dsl import DocHandle, OpNode, PageRef, TypedValue
from skunk.subagents.base import StepFailed, call_gemini, load_image_b64, parse_llm_value

_MAX_PAGES = 5

_BASE = (
    "You are a precise data extraction assistant for U.S. Treasury Bulletins.\n"
    "Extract the specific value requested from the provided page text.\n\n"
)

_SYSTEMS = {
    "value": _BASE
    + "Return exactly two lines — no labels, no JSON, no prose:\n"
    "  Line 1: the value as a Python literal (int, float, or \"quoted string\")\n"
    "  Line 2: the unit as a single lowercase word\n\n"
    "Unit vocabulary: usd_millions  usd_billions  pct  count  year  rate  text\n\n"
    "If the value is not present on these pages, return NOT_FOUND on line 1 (nothing else).\n",
    "list": _BASE
    + "Return exactly two lines — no labels, no JSON, no prose:\n"
    "  Line 1: a Python list literal e.g. [1.5, 2.3, 4.7]\n"
    "  Line 2: the unit as a single lowercase word\n\n"
    "If the values are not present on these pages, return NOT_FOUND on line 1 (nothing else).\n",
    "table": _BASE
    + "Return CSV format: one header row then data rows. No other prose or labels.\n"
    "If the table is not present on these pages, return NOT_FOUND on line 1 (nothing else).\n",
    "vision": (
        "You are a precise data extraction assistant. The images show scanned pages from "
        "U.S. Treasury Monthly Bulletins. Extract the specific value or information requested.\n\n"
        "Return exactly two lines — no labels, no JSON, no prose:\n"
        "  Line 1: the value as a Python literal (int, float, list of numbers, or \"quoted string\")\n"
        "  Line 2: the unit as a single lowercase word\n\n"
        "Unit vocabulary: usd_millions  usd_billions  pct  count  year  rate  text\n\n"
        "Rules:\n"
        "- Numeric values: bare numbers, no commas, no unit symbols in the value\n"
        "- Lists: Python list literal e.g. [1.5, 2.3, 4.7]\n"
        "- Text/string answers: wrap in double quotes e.g. \"increased\"\n"
        "- If the value is not found on these pages: return NOT_FOUND on line 1 (nothing else)\n"
        "- If uncertain between two rows, return the most specific match\n"
    ),
}


def _parse_response(raw: str, mode: str, concept: str) -> TypedValue | None:
    if raw.strip().split("\n", 1)[0].strip() == "NOT_FOUND":
        return None
    if mode == "table":
        try:
            df = pd.read_csv(io.StringIO(raw.strip()))
            return TypedValue(value=df, dtype="df", unit="", desc=concept)
        except Exception as e:
            raise StepFailed("extract", f"Cannot parse table response: {e}\nRaw: {raw[:300]}") from e
    try:
        value, dtype, unit = parse_llm_value(raw)
        return TypedValue(value=value, dtype=dtype, unit=unit, desc=concept)
    except ValueError as e:
        raise StepFailed("extract", f"Cannot parse LLM response: {e}\nRaw: {raw[:300]}") from e


def _text_tier(
    refs: list[PageRef],
    concept: str,
    mode: str,
    ctx: HarnessContext,
    get_text_fn,
) -> TypedValue | None:
    texts = []
    for ref in refs[:_MAX_PAGES]:
        text = get_text_fn(ref, ctx)
        if text:
            texts.append(f"--- Page {ref.page} ---\n{text}")
    if not texts:
        return None
    raw = call_gemini(_SYSTEMS[mode], f"Extract: {concept!r}\n\n" + "\n\n".join(texts))
    return _parse_response(raw, mode, concept)


def _vision_tier(refs: list[PageRef], concept: str, mode: str, ctx: HarnessContext) -> TypedValue | None:
    images: list[tuple[str, str]] = []
    for ref in refs[:_MAX_PAGES]:
        png = get_png(ref, ctx)
        if png:
            images.append(load_image_b64(png))

    if not images:
        return None

    raw = call_gemini(
        _SYSTEMS["vision"],
        f"Extract: {concept!r}\n\nPages provided: {len(images)} (of {len(refs)} total refs).",
        images=images,
    )
    return _parse_response(raw, mode, concept)


def run(op: OpNode, prev: DocHandle | None, ctx: HarnessContext) -> TypedValue:
    concept = op.args.get("concept") or op.args.get("source", "")
    if not concept:
        raise StepFailed("extract", "Missing 'concept' arg")
    mode = op.args.get("mode", "value")
    visual_only = bool(op.args.get("visual_only", False))

    refs = prev.refs if isinstance(prev, DocHandle) else []
    if not refs:
        raise StepFailed("extract", "No page refs to extract from")

    if not visual_only:
        result = _text_tier(refs, concept, mode, ctx, get_parsed_text_for_page)
        if result is not None:
            return result

        result = _text_tier(refs, concept, mode, ctx, get_ocr_text_for_pdf_page)
        if result is not None:
            return result

    result = _vision_tier(refs, concept, mode, ctx)
    if result is not None:
        return result

    raise StepFailed("extract", "value not found across tiers")
