"""extract subagent — question-driven named-value extraction over page text/images.

The agent receives the user's full question plus the rendered page(s) and returns
a JSON object mapping snake_case names to typed values relevant to answering the
question. Downstream `compute` consumes the resulting dict-valued TypedValue.

Tier dispatch (text → vision) is unchanged; only the prompt + parser changed.
"""

from __future__ import annotations

import json
import re

from skunk.common.context import HarnessContext
from skunk.common.parsed_json import get_printed_page, get_text_for_pdf_page
from skunk.common.pdf_text import get_ocr_text_for_pdf_page
from skunk.common.vision import get_png
from skunk.dsl import DocHandle, OpNode, PageRef, TypedValue
from skunk.subagents.base import StepFailed, call_gemini, load_image_b64

_MAX_PAGES = 5

_TEXT_SYSTEM = (
    "You are a precise data extraction assistant for U.S. Treasury Bulletins.\n"
    "You will be given the user's question and the text of one or more bulletin pages.\n\n"
    "Read the page(s) carefully and identify EVERY value, list, or table on the page(s)\n"
    "that could plausibly be needed to answer the question — including the row/column\n"
    "context that disambiguates each datum (year, sub-category, fiscal vs calendar, units).\n"
    "Do NOT compute, sum, average, or otherwise transform — only extract what is printed.\n\n"
    "Output a single JSON object mapping descriptive snake_case names to typed values.\n"
    "Each entry has the shape:\n"
    '  "<name>": {"value": <python literal>, "type": "<scalar|list|table>", "unit": "<unit>"}\n\n'
    "Where:\n"
    "- <name> is a snake_case identifier that uniquely describes the datum, including\n"
    '  any disambiguating period (e.g. "national_defense_cy1940", "total_budget_fy1940",\n'
    '  "jpy_holdings_mar_2025").\n'
    '- "value" is a Python literal:\n'
    "    type=scalar → bare number (no commas, no $) or a \"quoted string\"\n"
    "    type=list   → JSON array of numbers e.g. [1.5, 2.3, 4.7]\n"
    "    type=table  → JSON array of arrays, first row the header (strings), rest data rows\n"
    '- "unit" is a single lowercase token from:\n'
    "    usd, usd_thousands, usd_millions, usd_billions, pct, count, year, rate, fx_rate, text, mixed\n\n"
    "Rules:\n"
    "- If the table header says \"in thousands of dollars\", report unit=usd_thousands\n"
    "  (do NOT silently rescale the printed numbers).\n"
    "- For named-entity / string answers, type=scalar with unit=text and value as a quoted string.\n"
    "- If the page contains nothing relevant to the question, return {} (empty object).\n"
    "- Output ONLY the JSON object — no markdown fences, no commentary, no leading prose.\n"
)

_VISION_SYSTEM = (
    "You are a precise data extraction assistant. The images show scanned pages from\n"
    "U.S. Treasury Monthly Bulletins. You will be given the user's question and the\n"
    "rendered page image(s).\n\n"
    "Read the page(s) and identify every value/list/table that could plausibly be needed\n"
    "to answer the question, including disambiguating context (year, row, column, units).\n"
    "Do not compute or transform — extract only what is visible.\n\n"
    "Output a single JSON object with the same shape as the text-tier prompt:\n"
    '  {"<snake_case_name>": {"value": <literal>, "type": "<scalar|list|table>", "unit": "<unit>"}}\n\n'
    "Unit vocabulary: usd, usd_thousands, usd_millions, usd_billions, pct, count, year, rate, fx_rate, text, mixed.\n"
    "If the page header says values are in thousands/millions, use that as the unit; do not rescale.\n"
    "If nothing relevant is on the page, return {}.\n"
    "Output ONLY the JSON object — no fences, no prose.\n"
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(raw: str) -> str:
    return _FENCE_RE.sub("", raw.strip()).strip()


def _parse_response(raw: str, ctx: HarnessContext) -> TypedValue | None:
    """Parse the agent's JSON dict into a single TypedValue.

    Returns None when the agent emitted an empty object (signal: nothing relevant on page).
    Otherwise returns a TypedValue whose `value` is a dict {name: scalar/list/table}, and
    whose `desc` summarises each named entry's unit/type for downstream compute.
    """
    cleaned = _strip_fences(raw)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise StepFailed("extract", f"Cannot parse JSON response: {e}\nRaw: {cleaned[:400]}") from e
    if not isinstance(obj, dict):
        raise StepFailed("extract", f"Expected JSON object, got {type(obj).__name__}")
    if not obj:
        return None  # agent saw nothing relevant on this page

    values: dict[str, object] = {}
    desc_parts: list[str] = []
    for name, entry in obj.items():
        if not isinstance(entry, dict) or "value" not in entry:
            raise StepFailed("extract", f"Malformed entry {name!r}: {entry!r}")
        val = entry["value"]
        unit = entry.get("unit", "")
        typ = entry.get("type", "scalar")
        values[name] = val
        desc_parts.append(f"{name} ({typ}, unit={unit})")

    return TypedValue(
        value=values,
        dtype="named",
        unit="",
        desc="; ".join(desc_parts),
    )


def _text_tier(
    refs: list[PageRef],
    ctx: HarnessContext,
    get_text_fn,
    tier_name: str,
) -> TypedValue | None:
    texts = []
    for ref in refs[:_MAX_PAGES]:
        text = get_text_fn(ref, ctx)
        printed = get_printed_page(ref, ctx)
        if text:
            ctx.emit("extract", f"tier={tier_name} got text",
                     page=str(ref), chars=len(text), printed_page=printed)
            header = f"--- PDF page {ref.page}"
            if printed:
                header += f' (bulletin printed page "{printed}")'
            header += " ---"
            texts.append(f"{header}\n{text}")
        else:
            ctx.emit("extract", f"tier={tier_name} no text",
                     page=str(ref), file_path=ref.file_path, printed_page=printed)
    if not texts:
        ctx.emit("extract", f"tier={tier_name} skipped (no text from any ref)")
        return None
    user_msg = (
        f"Question:\n{ctx.question}\n\n"
        f"Page text:\n\n" + "\n\n".join(texts)
    )
    ctx.emit("extract", f"tier={tier_name} calling gemini",
             total_chars=sum(len(t) for t in texts),
             system_prompt=_TEXT_SYSTEM[:1500],
             user_message=user_msg[:3000])
    raw = call_gemini(_TEXT_SYSTEM, user_msg)
    ctx.emit("extract", f"tier={tier_name} gemini response", raw=raw[:800])
    return _parse_response(raw, ctx)


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

    ctx.emit("extract", "tier=vision calling gemini", n_images=len(images))
    user_msg = (
        f"Question:\n{ctx.question}\n\n"
        f"Pages provided: {len(images)} (of {len(refs)} total refs)."
    )
    raw = call_gemini(_VISION_SYSTEM, user_msg, images=images)
    ctx.emit("extract", "tier=vision gemini response", raw=raw[:800])
    return _parse_response(raw, ctx)


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
