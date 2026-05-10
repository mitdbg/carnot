"""read_visual subagent — vision LLM call over rendered page images."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from skunk.dsl import DocHandle, OpNode, TypedValue
from skunk.subagents.base import HarnessContext, StepFailed, call_gemini, load_image_b64, parse_llm_value

if TYPE_CHECKING:
    from skunk.dsl import PageRef

_SYSTEM = """\
You are a precise data extraction assistant. The images show scanned pages from \
U.S. Treasury Monthly Bulletins. Extract the specific value or information requested.

Return exactly two lines — no labels, no JSON, no prose:
  Line 1: the value as a Python literal (int, float, list of numbers, or "quoted string")
  Line 2: the unit as a single lowercase word

Unit vocabulary: usd_millions  usd_billions  pct  count  year  rate  text

Rules:
- Numeric values: bare numbers, no commas, no unit symbols in the value
- Lists: Python list literal e.g. [1.5, 2.3, 4.7]
- Text/string answers: wrap in double quotes e.g. "increased"
- If the value is not found: you MUST raise — do NOT return a null or placeholder
- If uncertain between two rows, return the most specific match
"""

_MAX_PAGES = 5


def _get_png(ref: "PageRef", ctx: HarnessContext) -> str | None:
    from skunk.common.page_map import pdf_page_for_ref

    pdf_idx = pdf_page_for_ref(ref, ctx.cache_dir)
    if pdf_idx is None:
        return None

    month = ref.month or "unknown"
    png_path = Path(ctx.cache_dir) / "pages" / month / f"p{pdf_idx:03d}.png"
    if png_path.exists():
        return str(png_path)

    if not ref.file_path:
        return None

    try:
        import fitz
        png_path.parent.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(ref.file_path)
        page = doc[pdf_idx - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        pix.save(str(png_path))
        doc.close()
        return str(png_path)
    except Exception as e:
        print(f"[read_visual] render failed for {ref}: {e}")
        return None


def run(op: OpNode, prev: DocHandle | None, ctx: HarnessContext) -> TypedValue:
    concept = op.args.get("concept") or op.args.get("source", "")
    if not concept:
        raise StepFailed("read_visual", "Missing 'concept' arg")

    refs = prev.refs if isinstance(prev, DocHandle) else []
    if not refs:
        raise StepFailed("read_visual", "No page refs to read")

    images: list[tuple[str, str]] = []
    for ref in refs[:_MAX_PAGES]:
        png = _get_png(ref, ctx)
        if png:
            images.append(load_image_b64(png))

    if not images:
        raise StepFailed("read_visual", f"Could not render any page images ({len(refs)} refs had no PNG)")

    raw = call_gemini(
        _SYSTEM,
        f"Extract: {concept!r}\n\nPages provided: {len(images)} (of {len(refs)} total refs).",
        images=images,
    )

    try:
        value, dtype, unit = parse_llm_value(raw)
    except ValueError as e:
        raise StepFailed("read_visual", f"Cannot parse LLM response: {e}\nRaw: {raw[:300]}") from e

    return TypedValue(value=value, dtype=dtype, unit=unit, desc=concept)
