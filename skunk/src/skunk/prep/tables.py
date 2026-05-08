"""Tier 2 — lazy table catalog via vision-LLM.

On first access for a given page, renders the page image, runs a vision LLM
to extract all tables as DataFrames, and caches them as CSV files.

Cache files are keyed by **PDF page index** (NNN = PDF page index, 1-based):
  cache/tables/{YYYY-MM}/p{NNN}-t{K}.csv
  cache/tables/{YYYY-MM}/p{NNN}-meta.json

Use prep/page_map.pdf_page_for_ref(ref, cache_dir) to convert a PageRef whose
.page field holds a bulletin printed page number into the PDF page index.

Usage (lazy — called by extract subagent):
    from skunk.prep.tables import get_tables_for_page
    tables = get_tables_for_page(ref, ctx)   # list of DataFrames

Usage (full pass):
    python -m skunk.prep.tables \
        --manifest ... --cache-dir ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skunk.dsl import PageRef
    from skunk.subagents.base import HarnessContext

from skunk.prep.page_map import pdf_page_for_ref

_EXTRACT_SYSTEM = """\
You are a table-extraction assistant. The image shows a scanned page from a \
U.S. Treasury Monthly Bulletin. Extract ALL tables you can see.

For each table, output a JSON object:
{
  "tables": [
    {
      "title": "brief table title",
      "description": "what this table contains, units, time axis",
      "csv": "header row\\ndata row 1\\ndata row 2\\n..."
    }
  ]
}

Use ',' as the CSV separator. Keep column headers as close to the printed text as possible.
Numeric values: preserve original formatting (commas in numbers are fine).
If you see no tables, return {"tables": []}.
"""


def get_tables_for_page(ref: "PageRef", ctx: "HarnessContext") -> list[Any]:
    """Return list of pandas DataFrames extracted from the given page (lazy)."""
    if ref.month is None:
        return []
    pdf_idx = pdf_page_for_ref(ref, ctx.cache_dir)
    if pdf_idx is None:
        return []

    cache_dir = Path(ctx.cache_dir)
    meta_path = cache_dir / "tables" / ref.month / f"p{pdf_idx:03d}-meta.json"

    if meta_path.exists():
        return _load_cached_tables(meta_path, cache_dir, ref)

    # Otherwise render page and extract
    png_path = _get_or_render_page(ref, ctx)
    if not png_path:
        return []

    return _extract_and_cache(png_path, ref, ctx, meta_path, cache_dir)


def _get_or_render_page(ref: "PageRef", ctx: "HarnessContext") -> str | None:
    cache_dir = Path(ctx.cache_dir)
    pdf_idx = pdf_page_for_ref(ref, ctx.cache_dir)
    if ref.month and pdf_idx is not None:
        png = cache_dir / "pages" / ref.month / f"p{pdf_idx:03d}.png"
        if png.exists():
            return str(png)
    # Try on-demand render
    if ref.file_path and pdf_idx is not None:
        from skunk.subagents.read_visual import _render_on_demand
        return _render_on_demand(ref, ctx)
    return None


def _extract_and_cache(
    png_path: str, ref: "PageRef", ctx: "HarnessContext",
    meta_path: Path, cache_dir: Path
) -> list[Any]:
    from skunk.subagents.base import call_llm, extract_json, load_image_b64

    pdf_idx = pdf_page_for_ref(ref, ctx.cache_dir)
    mime, b64 = load_image_b64(png_path)
    raw = call_llm(_EXTRACT_SYSTEM, "Extract all tables from this page.", ctx.llm_config, images=[(mime, b64)])

    try:
        parsed = extract_json(raw)
        tables_data = parsed.get("tables", [])
    except Exception:
        tables_data = []

    import io
    import pandas as pd

    dfs: list[Any] = []
    meta: list[dict] = []

    table_dir = cache_dir / "tables" / (ref.month or "unknown")
    table_dir.mkdir(parents=True, exist_ok=True)

    for k, tbl in enumerate(tables_data):
        csv_text = tbl.get("csv", "")
        if not csv_text.strip():
            continue
        try:
            df = pd.read_csv(io.StringIO(csv_text))
            csv_path = table_dir / f"p{pdf_idx:03d}-t{k}.csv"
            df.to_csv(csv_path, index=True)
            dfs.append(df)
            meta.append({
                "k": k,
                "title": tbl.get("title", ""),
                "description": tbl.get("description", ""),
                "csv": str(csv_path),
                "n_rows": len(df),
                "columns": list(df.columns),
            })
        except Exception as e:
            print(f"[tables] Failed to parse table {k} on {ref.month}/p{pdf_idx}: {e}")

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return dfs


def _load_cached_tables(meta_path: Path, cache_dir: Path, ref: "PageRef") -> list[Any]:
    import pandas as pd

    meta = json.loads(meta_path.read_text())
    dfs = []
    for entry in meta:
        csv = Path(entry.get("csv", ""))
        if csv.exists():
            try:
                dfs.append(pd.read_csv(csv, index_col=0))
            except Exception:
                pass
    return dfs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default="cache")
    args = parser.parse_args()

    import pandas as pd
    from skunk.subagents.base import HarnessContext, LLMConfig
    from skunk.dsl import PageRef

    df = pd.read_csv(args.manifest)
    ctx = HarnessContext(question="", cache_dir=args.cache_dir)

    total = 0
    for _, row in df.iterrows():
        n_pages = int(row.get("n_pages", 0))
        month = row.get("month", "")
        for pdf_page_num in range(1, n_pages + 1):
            ref = PageRef(year=int(row["year"]), month=month, pdf_page=pdf_page_num, file_path=row.get("pdf_path"))
            tables = get_tables_for_page(ref, ctx)
            total += len(tables)
            if tables:
                print(f"  {month}/p{pdf_page_num}: {len(tables)} tables")

    print(f"[tables] Done. Total tables extracted: {total}")
