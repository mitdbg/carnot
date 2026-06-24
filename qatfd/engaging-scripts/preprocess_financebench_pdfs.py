"""Preprocess FinanceBench PDFs into OfficeQA-like element JSONs (text / table / figure).

Each page is decomposed into elements via a hybrid pipeline:
  * TEXT  — taken straight from PyMuPDF's text layer (born-digital filings; no OCR), split on
            blank lines and merged into ~512-token elements (BrowseComp-Plus style).
  * TABLE / FIGURE — a two-stage vision cascade per page:
      1. GATE  (cheap multimodal model, default google/gemma-3-12b-it): the 200dpi page
         screenshot -> YES/NO "does this page contain a table or a figure?".
      2. EXTRACT (default google/gemini-3.1-flash-lite, only when gate=YES): the same screenshot ->
         markdown with one section per table/figure, in this exact shape:
             ## Table 1
             <markdown for table 1>
             ## Figure 1
             <summary of figure 1>
         which is parsed back into one element per section.

Output: one `{output_dir}/{doc_name}.json` per PDF:
    {"doc_name": "3M_2018_10K", "n_pages": 142,
     "elements": [{"id": 0, "page_id": 0, "type": "text", "content": "..."}, ...]}
consumed downstream by compute_financebench_element_embeddings.py.

LLM access uses the skunk `LLMClient` over OpenRouter, built from a minimal `SystemConfig`
constructed from CLI args (no SkunkConfig / env config). `OPENROUTER_API_KEY` is read from the
environment by the client. Models and request pacing are CLI args.

Usage:
    OPENROUTER_API_KEY=sk-or-... python preprocess_financebench_pdfs.py \\
        --input_dir  ../skunk/financebench/pdfs \\
        --output_dir ../skunk/financebench/financebench-elements \\
        --sample 5            # validate on the first few docs before the full run
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import glob
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from skunk.common import B64Image
from skunk.config import SystemConfig
from skunk.llm_client import LLMClient

# Text-element sizing (BrowseComp-Plus style): merge paragraphs up to a token target, hard-split
# anything beyond MAX. Estimated as chars/4 to avoid tokenizing on the CPU preprocessing box.
CHARS_PER_TOKEN_EST = 4
DEFAULT_TARGET_ELEMENT_TOKENS = 512

_MULTI_NL_RE = re.compile(r"\n{3,}")
_LONG_DOTS_RE = re.compile(r"\.{4,}")
# Section headers emitted by the extract model, e.g. "## Table 1" / "## Figure 2".
_SECTION_RE = re.compile(r"^[ \t]*#{1,6}[ \t]*(table|figure)[ \t]+\d+[ \t]*:?[ \t]*$", re.IGNORECASE | re.MULTILINE)

_GATE_SYSTEM = (
    "You are a fast document-layout classifier. You are shown a screenshot of a single page from "
    "a company financial filing (10-K / 10-Q / 8-K / earnings release). Decide whether the page "
    "contains at least one TABLE (a grid of rows and columns of data) or FIGURE (a chart, graph, "
    "plot, or diagram). Plain paragraphs of text, page headers/footers, and page numbers do NOT "
    "count. Answer with a single word: YES if the page contains a table or a figure, otherwise NO."
)
_GATE_USER = "Does this page contain a table or a figure? Answer with exactly one word: YES or NO."

_EXTRACT_SYSTEM = (
    "You extract the tables and figures from a single page of a company financial filing, shown "
    "as a screenshot. Produce GitHub-flavored markdown with ONE section per table or figure, using "
    "exactly this format and nothing else (no preamble, no commentary):\n\n"
    "## Table 1\n<the table as a markdown table>\n\n## Table 2\n<...>\n\n"
    "## Figure 1\n<one-paragraph summary of the figure>\n\n## Figure 2\n<...>\n\n"
    "Rules:\n"
    "- Number tables and figures separately, each starting at 1, in top-to-bottom reading order.\n"
    "- For a TABLE, faithfully reproduce its rows, columns, and values as a markdown table. Do not "
    "summarize or omit data. Keep units/currency markers that appear in headers or cells.\n"
    "- For a FIGURE (chart/graph/plot/diagram), write a concise summary covering: its title, what "
    "it plots, the axes and their units, the series/legend, and any date range(s) or key values.\n"
    "- If the page has no tables, omit all Table sections; if it has no figures, omit all Figure "
    "sections. If it has neither, output nothing.\n"
    "- Output ONLY the '## Table N' / '## Figure N' sections."
)
_EXTRACT_USER = "Extract every table (as markdown) and every figure (as a summary) from this page."


# ---------------------------------------------------------------------------
# text elements
# ---------------------------------------------------------------------------

def _clean_page_text(text: str) -> str:
    """Collapse 3+-newline runs to a blank line and drop dot-leader runs (4+ periods)."""
    text = _MULTI_NL_RE.sub("\n\n", text or "")
    text = _LONG_DOTS_RE.sub("", text)
    return text.strip()


def text_elements(page_text: str, target_tokens: int) -> list[str]:
    """Split cleaned page text on blank lines and merge consecutive paragraphs up to a token
    target; hard-split any merged element beyond 4x the target. Mirrors the BrowseComp-Plus job."""
    cleaned = _clean_page_text(page_text)
    if not cleaned:
        return []
    target_chars = CHARS_PER_TOKEN_EST * target_tokens
    max_chars = CHARS_PER_TOKEN_EST * target_tokens * 4

    merged: list[str] = []
    current = ""
    for para in cleaned.split("\n\n"):
        if current and len(current) + len(para) > target_chars:
            merged.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        merged.append(current.strip())

    out: list[str] = []
    for elt in merged:
        if len(elt) > max_chars:
            for start in range(0, len(elt), target_chars):
                piece = elt[start : start + target_chars].strip()
                if piece:
                    out.append(piece)
        elif elt:
            out.append(elt)
    return out


# ---------------------------------------------------------------------------
# vision-extraction parsing
# ---------------------------------------------------------------------------

def parse_extract_markdown(md: str) -> list[tuple[str, str]]:
    """Parse the model's '## Table N' / '## Figure N' markdown into [(type, content), ...] where
    type is 'table' or 'figure'. Tolerant of extra prose: any text before the first section header
    is ignored, and empty sections are dropped."""
    out: list[tuple[str, str]] = []
    matches = list(_SECTION_RE.finditer(md or ""))
    for i, m in enumerate(matches):
        kind = m.group(1).lower()  # "table" | "figure"
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        content = md[body_start:body_end].strip()
        if content:
            out.append((kind, content))
    return out


# ---------------------------------------------------------------------------
# rendering + LLM calls
# ---------------------------------------------------------------------------

def render_page_png_b64(page: fitz.Page, dpi: int) -> B64Image:
    """Render a PDF page to an in-memory PNG and wrap it as a base64 B64Image."""
    pix = page.get_pixmap(dpi=dpi)
    png = pix.tobytes("png")
    return B64Image(mime="image/png", data=base64.standard_b64encode(png).decode())


def _is_yes(text: str) -> bool:
    """True iff the gate model's reply contains YES before any NO."""
    m = re.search(r"\b(yes|no)\b", text or "", re.IGNORECASE)
    return bool(m) and m.group(1).lower() == "yes"


def build_config(args) -> SystemConfig:
    """Minimal SystemConfig for OpenRouter generation (no embedding/agent fields are exercised).
    The key comes from OPENROUTER_API_KEY in the env (read by LLMClient)."""
    return SystemConfig(
        name="financebench_preprocess",
        emb_provider="openrouter",  # unused — no embedding calls here
        emb_model_id="",
        llm_provider="openrouter",
        agent_model_id=args.extract_model,
        llm_model=args.extract_model,
        llm_max_retries=args.max_retries,
        llm_retry_initial_delay_s=1.0,
        llm_model_rpm={},
        llm_default_rpm=args.rpm,
        llm_model_tpm={},
        llm_default_tpm=None,
        llm_prices={
            "qwen3-embedding-8b": {"in": 0.01, "out": 0.00, "cached": 0.00},
            "google/gemma-3-12b-it": {"in": 0.05, "out": 0.15, "cached": 0.00},
        },
    )

@dataclass
class Stats:
    pages: int = 0
    gated_pages: int = 0
    gate_yes: int = 0
    tables: int = 0
    figures: int = 0
    text_elems: int = 0
    gate_in: int = 0
    gate_out: int = 0
    extract_in: int = 0
    extract_out: int = 0
    errors: int = 0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class _PageElements:
    page_id: int
    items: list[tuple[str, str]] = field(default_factory=list)  # (type, content) in doc order


async def process_page(
    page: fitz.Page,
    page_id: int,
    *,
    client: LLMClient,
    sem: asyncio.Semaphore,
    gate_model: str,
    extract_model: str,
    dpi: int,
    target_tokens: int,
    timeout_s: float,
    stats: Stats,
) -> _PageElements:
    """Build the element list for one page: text (always) + table/figure (gate then extract)."""
    result = _PageElements(page_id=page_id)
    raw_text = page.get_text("text")
    texts = text_elements(raw_text, target_tokens)

    # A page with no text AND no visual content is blank -> no elements, no LLM calls.
    has_visual = bool(page.get_images(full=True)) or len(page.get_drawings()) > 4
    if not texts and not has_visual:
        return result

    for t in texts:
        result.items.append(("text", t))
    stats.text_elems += len(texts)

    ext_text = ""
    async with sem:
        try:
            img = await asyncio.to_thread(render_page_png_b64, page, dpi)
            gate = await client.acall(
                system=_GATE_SYSTEM, user=_GATE_USER, images=[img], temperature=0.0,
                model=gate_model, max_output_tokens=4, timeout_s=timeout_s, ctx=None, call_site="fb_gate",
            )
            stats.gated_pages += 1
            stats.gate_in += gate.input_tokens or 0
            stats.gate_out += gate.output_tokens or 0
            if not _is_yes(gate.text):
                return result
            stats.gate_yes += 1

            ext = await client.acall(
                system=_EXTRACT_SYSTEM, user=_EXTRACT_USER, images=[img], temperature=0.0,
                model=extract_model, timeout_s=timeout_s, ctx=None, call_site="fb_extract",
            )
            ext_text = ext.text
            stats.extract_in += ext.input_tokens or 0
            stats.extract_out += ext.output_tokens or 0
        except Exception as e:  # noqa: BLE001 — one bad page shouldn't kill the doc
            stats.errors += 1
            print(f"  WARN page {page_id}: vision call failed: {e}", flush=True)
            return result

    for kind, content in parse_extract_markdown(ext_text):
        result.items.append((kind, content))
        if kind == "table":
            stats.tables += 1
        else:
            stats.figures += 1
    return result


async def process_pdf(
    pdf_path: str,
    out_path: str,
    *,
    client: LLMClient,
    sem: asyncio.Semaphore,
    args,
    stats: Stats,
) -> None:
    """Decompose one PDF into elements and write its JSON atomically."""
    doc_name = Path(pdf_path).stem
    doc = fitz.open(pdf_path)
    try:
        n_pages = doc.page_count
        stats.pages += n_pages
        tasks = [
            process_page(
                doc[i], i, client=client, sem=sem, gate_model=args.gate_model,
                extract_model=args.extract_model, dpi=args.dpi,
                target_tokens=args.target_element_tokens, timeout_s=args.timeout, stats=stats,
            )
            for i in range(n_pages)
        ]
        page_results = await asyncio.gather(*tasks)
    finally:
        doc.close()

    elements: list[dict] = []
    eid = 0
    for pr in page_results:
        for etype, content in pr.items:
            elements.append({"id": eid, "page_id": pr.page_id, "type": etype, "content": content})
            eid += 1

    payload = {"doc_name": doc_name, "n_pages": n_pages, "elements": elements}
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, out_path)
    n_tab = sum(1 for e in elements if e["type"] == "table")
    n_fig = sum(1 for e in elements if e["type"] == "figure")
    n_txt = sum(1 for e in elements if e["type"] == "text")
    print(f"[{doc_name}] {n_pages}p -> {len(elements)} elems (text={n_txt} table={n_tab} figure={n_fig})", flush=True)


async def run(pdfs: list[str], args, stats: Stats) -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set in the environment.")
    client = LLMClient(build_config(args))
    sem = asyncio.Semaphore(args.concurrency)

    # Docs run sequentially (write-as-you-go, resumable); pages within a doc fan out under `sem`,
    # which bounds in-flight renders + LLM calls across the whole job.
    for pdf in pdfs:
        out_path = os.path.join(args.output_dir, f"{Path(pdf).stem}.json")
        if os.path.exists(out_path):
            print(f"[{Path(pdf).stem}] exists, skipping", flush=True)
            continue
        await process_pdf(pdf, out_path, client=client, sem=sem, args=args, stats=stats)


def select_pdfs(args) -> list[str]:
    pdfs = sorted(glob.glob(os.path.join(args.input_dir, "*.pdf")))
    if args.docs:
        wanted = {d.strip() for d in args.docs.split(",") if d.strip()}
        pdfs = [p for p in pdfs if Path(p).stem in wanted]
    if args.world_size > 1:
        pdfs = pdfs[args.rank :: args.world_size]
    if args.sample is not None:
        pdfs = pdfs[: args.sample]
    return pdfs


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose FinanceBench PDFs into text/table/figure element JSONs.")
    parser.add_argument("--input_dir", required=True, help="Directory of FinanceBench *.pdf files")
    parser.add_argument("--output_dir", required=True, help="Directory to write {doc_name}.json element files")
    parser.add_argument("--gate-model", default="google/gemma-3-12b-it", help="Cheap multimodal model for the table/figure gate")
    parser.add_argument("--extract-model", default="google/gemini-3.1-flash-lite", help="Model for table-markdown / figure-summary extraction")
    parser.add_argument("--dpi", type=int, default=200, help="Page render DPI for the vision calls")
    parser.add_argument("--concurrency", type=int, default=32, help="Max concurrent pages (bounds in-flight LLM calls + renders)")
    parser.add_argument("--rpm", type=float, default=600.0, help="Per-model requests/min pacing (LLMClient token bucket)")
    parser.add_argument("--max-retries", type=int, default=4, help="Per-call transient-fault retries (429/5xx/transport)")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout (seconds)")
    parser.add_argument("--target-element-tokens", type=int, default=DEFAULT_TARGET_ELEMENT_TOKENS, help="Target token size for merged text elements")
    parser.add_argument("--sample", type=int, default=None, help="Process only the first N (post-filter) docs")
    parser.add_argument("--docs", type=str, default=None, help="Comma-separated doc_name stems to restrict to")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", 0)), help="Worker index for multi-process splitting")
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)), help="Number of parallel workers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pdfs = select_pdfs(args)
    print(f"Processing {len(pdfs)} PDFs (gate={args.gate_model}, extract={args.extract_model}, dpi={args.dpi}, concurrency={args.concurrency}).", flush=True)

    stats = Stats()
    asyncio.run(run(pdfs, args, stats))

    yes_rate = (stats.gate_yes / stats.gated_pages) if stats.gated_pages else 0.0
    print("\n=== summary ===", flush=True)
    print(f"pages={stats.pages} gated={stats.gated_pages} gate_yes={stats.gate_yes} ({yes_rate:.0%}) "
          f"tables={stats.tables} figures={stats.figures} text_elems={stats.text_elems} errors={stats.errors}", flush=True)
    print(f"tokens: gate in/out={stats.gate_in}/{stats.gate_out}  extract in/out={stats.extract_in}/{stats.extract_out}", flush=True)

    manifest = os.path.join(args.output_dir, f"_manifest_rank{args.rank}.json")
    with open(manifest, "w") as f:
        json.dump({"args": vars(args), "stats": stats.as_dict()}, f, indent=2)
    print(f"wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
