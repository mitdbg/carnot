"""Preprocess FinanceBench PDFs into OfficeQA-like element JSONs (text / table / figure).

Three resumable phases, each skipping work already persisted (so a rerun only does what's missing):

  1. RENDER (ProcessPool — PyMuPDF is process-safe, not thread-safe): one worker per PDF renders
     each page to a PNG and extracts its text-layer elements. Per page it writes
     `{renders_dir}/{doc}/{p}.png` (binary; omitted for blank pages) and a `{p}.json` sidecar
     ({n_pages, blank, text_items}). The sidecar is written LAST as the commit marker, so an
     interrupted page re-renders next run.
  2. LLM (ThreadPool — I/O-bound): one task per rendered (non-blank) page sends the PNG to the
     EXTRACT model, which returns `## Table N` / `## Figure N` markdown, parsed into elements and
     written (with the page's text elements) to `{pages_dir}/{doc}/{p}.json` ({n_pages, items}). A
     failed page persists nothing, so it is retried on the next run.
  3. ASSEMBLE: for each doc whose every page has an element artifact, concatenate them (in page +
     element order, assigning running ids) into the final `{output_dir}/{doc}.json`.

Output: one `{output_dir}/{doc_name}.json` per PDF:
    {"doc_name": "3M_2018_10K", "n_pages": 142,
     "elements": [{"id": 0, "page_id": 0, "type": "text", "content": "..."}, ...]}
consumed downstream by compute_financebench_element_embeddings.py.

LLM access uses the skunk `LLMClient` over OpenRouter, built from a minimal `SystemConfig` from CLI
args (no SkunkConfig). `OPENROUTER_API_KEY` is read from the env by the client. `--input_dir`,
`--output_dir`, `--renders_dir`, `--pages_dir` each accept a local path OR an `s3://bucket/prefix`
URI; with S3 everything streams in/out so the cluster keeps ~zero local disk (S3 needs `boto3` +
AWS creds in the env; boto3 is imported lazily so local runs don't require it).

Usage:
    OPENROUTER_API_KEY=sk-or-... python preprocess_financebench_pdfs.py \\
        --input_dir  ../skunk/financebench/pdfs \\
        --output_dir ../skunk/financebench/financebench-elements \\
        --sample 5            # validate on the first few docs before the full run
    # or stream to/from S3 (see run_financebench_preprocess.slurm):
    #   --input_dir s3://carnot-research/financebench/pdfs \\
    #   --output_dir s3://carnot-research/financebench/financebench-elements
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from urllib.parse import urlparse

import fitz  # PyMuPDF

from skunk.common import B64Image
from skunk.config import SystemConfig
from skunk.llm_client import EmptyCompletionError, LLMClient

# Text-element sizing (BrowseComp-Plus style): merge paragraphs up to a token target, hard-split
# anything beyond MAX. Estimated as chars/4 to avoid tokenizing on the CPU preprocessing box.
CHARS_PER_TOKEN_EST = 4
DEFAULT_TARGET_ELEMENT_TOKENS = 512

_MULTI_NL_RE = re.compile(r"\n{3,}")
_LONG_DOTS_RE = re.compile(r"\.{4,}")
# Section headers emitted by the extract model, e.g. "## Table 1" / "## Figure 2".
_SECTION_RE = re.compile(r"^[ \t]*#{1,6}[ \t]*(table|figure)[ \t]+\d+[ \t]*:?[ \t]*$", re.IGNORECASE | re.MULTILINE)

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
    "sections. If it has NEITHER a table nor a figure, respond with exactly: NONE\n"
    "- Output ONLY the '## Table N' / '## Figure N' sections (or NONE). Never reply empty."
)
_EXTRACT_USER = "Extract every table (as markdown) and every figure (as a summary) from this page."


# ---------------------------------------------------------------------------
# I/O layer: every dir arg may be a local path OR an `s3://bucket/prefix` URI, so the Engaging job
# streams everything in/out of S3 and keeps ~zero local disk. boto3 reads creds from the standard
# AWS_* env vars; it's imported lazily so local runs don't need it. (Mirrors compute_biogen_embeddings.py.)
# ---------------------------------------------------------------------------

_S3_CLIENT = None


def _is_s3(path: str) -> bool:
    return path.startswith("s3://")


def _s3_split(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


def _s3():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        import boto3

        _S3_CLIENT = boto3.client("s3")
    return _S3_CLIENT


def _join(base: str, name: str) -> str:
    """Join a path/URI with a sub-path (works for both local paths and s3:// URIs)."""
    return base.rstrip("/") + "/" + name


def _stem(path: str) -> str:
    """File stem of a local path or s3:// key (e.g. ".../3M_2018_10K.pdf" -> "3M_2018_10K")."""
    return os.path.splitext(os.path.basename(path))[0]


def _exists(path: str) -> bool:
    if _is_s3(path):
        bucket, key = _s3_split(path)
        try:
            _s3().head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False
    return os.path.exists(path)


def list_pdfs(input_dir: str) -> list[str]:
    """Sorted list of every *.pdf under a local dir or an s3:// prefix."""
    if _is_s3(input_dir):
        bucket, prefix = _s3_split(input_dir)
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        keys: list[str] = []
        for page in _s3().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            keys += [f"s3://{bucket}/{o['Key']}" for o in page.get("Contents", []) if o["Key"].endswith(".pdf")]
        return sorted(keys)
    return sorted(glob.glob(os.path.join(input_dir, "*.pdf")))


def read_bytes(path: str) -> bytes:
    """Read a file's bytes from a local path or an s3:// object (one GET, held in memory)."""
    if _is_s3(path):
        bucket, key = _s3_split(path)
        return _s3().get_object(Bucket=bucket, Key=key)["Body"].read()
    with open(path, "rb") as f:
        return f.read()


def write_output(dest: str, data: bytes) -> None:
    """Write bytes to a local path (atomically via tmp+rename, creating parents) or an s3:// object."""
    if _is_s3(dest):
        bucket, key = _s3_split(dest)
        _s3().upload_fileobj(io.BytesIO(data), bucket, key)
    else:
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        tmp = dest + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)


def page_ids_under(dir_prefix: str) -> set[int]:
    """Integer page ids of the `{n}.json` files directly under one `{prefix}/{doc}` directory."""
    ids: set[int] = set()
    if _is_s3(dir_prefix):
        bucket, prefix = _s3_split(dir_prefix)
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        for page in _s3().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for o in page.get("Contents", []):
                k = o["Key"][len(prefix):]
                if k.endswith(".json") and "/" not in k:
                    ids.add(int(k[:-5]))
        return ids
    for p in glob.glob(os.path.join(dir_prefix, "*.json")):
        ids.add(int(_stem(p)))
    return ids


def all_page_keys(prefix: str) -> set[str]:
    """Set of "{doc}/{page}" for every `{prefix}/{doc}/{page}.json` (one LIST over the whole prefix)."""
    keys: set[str] = set()
    if _is_s3(prefix):
        bucket, pfx = _s3_split(prefix)
        if pfx and not pfx.endswith("/"):
            pfx += "/"
        for page in _s3().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=pfx):
            for o in page.get("Contents", []):
                rel = o["Key"][len(pfx):]
                if rel.endswith(".json") and rel.count("/") == 1:
                    keys.add(rel[:-5])
        return keys
    for p in glob.glob(os.path.join(prefix, "*", "*.json")):
        keys.add(f"{os.path.basename(os.path.dirname(p))}/{_stem(p)}")
    return keys


def list_done_docs(output_dir: str) -> set[str]:
    """Doc names whose final {doc}.json already exists directly under output_dir."""
    docs: set[str] = set()
    if _is_s3(output_dir):
        bucket, prefix = _s3_split(output_dir)
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        for page in _s3().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for o in page.get("Contents", []):
                base = o["Key"][len(prefix):]
                if base.endswith(".json") and "/" not in base and not base.startswith("_manifest"):
                    docs.add(base[:-5])
        return docs
    for p in glob.glob(os.path.join(output_dir, "*.json")):
        b = os.path.basename(p)
        if not b.startswith("_manifest"):
            docs.add(b[:-5])
    return docs


# ---------------------------------------------------------------------------
# text elements + extraction parsing
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


def parse_extract_markdown(md: str) -> list[tuple[str, str]]:
    """Parse the model's '## Table N' / '## Figure N' markdown into [(type, content), ...] where
    type is 'table' or 'figure'. Tolerant of extra prose; empty sections are dropped."""
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


def build_config(args) -> SystemConfig:
    """Minimal SystemConfig for OpenRouter generation (no embedding/agent fields are exercised).
    The key comes from OPENROUTER_API_KEY in the env (read by LLMClient)."""
    return SystemConfig(
        name="financebench_preprocess",
        emb_provider="openrouter",  # unused — no embedding calls here
        emb_model_id="",
        llm_provider="openrouter",
        llm_model=args.extract_model,
        llm_max_retries=args.max_retries,
        llm_retry_initial_delay_s=1.0,
        llm_model_rpm={},
        llm_default_rpm=args.rpm,
        llm_model_tpm={},
        llm_default_tpm=None,
        llm_prices={},
    )


# ---------------------------------------------------------------------------
# phase 1 — render (ProcessPool worker; PyMuPDF kept to its own process)
# ---------------------------------------------------------------------------

def _render_png(page: fitz.Page, dpi: int) -> bytes:
    return page.get_pixmap(dpi=dpi).tobytes("png")


def render_doc(pdf_path: str, renders_dir: str, dpi: int, target_tokens: int) -> tuple[str, int, int, int]:
    """Render every not-yet-rendered page of one PDF: write `{doc}/{p}.png` (non-blank pages) then
    the `{doc}/{p}.json` sidecar ({n_pages, blank, text_items}) as the commit marker. Runs in a
    subprocess, so fitz never crosses threads. Returns (doc, rendered, blank, skipped)."""
    doc = _stem(pdf_path)
    doc_dir = _join(renders_dir, doc)
    already = page_ids_under(doc_dir)
    fdoc = fitz.open(stream=read_bytes(pdf_path), filetype="pdf")
    rendered = blank = skipped = 0
    try:
        n_pages = fdoc.page_count
        for p in range(n_pages):
            if p in already:
                skipped += 1
                continue
            page = fdoc[p]
            texts = text_elements(page.get_text("text"), target_tokens)
            has_visual = bool(page.get_images(full=True)) or len(page.get_drawings()) > 4
            is_blank = not texts and not has_visual
            if not is_blank:
                write_output(_join(doc_dir, f"{p}.png"), _render_png(page, dpi))
                rendered += 1
            else:
                blank += 1
            # sidecar last == commit marker (so a half-written page re-renders next run)
            write_output(_join(doc_dir, f"{p}.json"),
                         json.dumps({"n_pages": n_pages, "blank": is_blank, "text_items": texts}).encode())
    finally:
        fdoc.close()
    return doc, rendered, blank, skipped


# ---------------------------------------------------------------------------
# phase 2 — LLM (ThreadPool worker; sync `client.call`, which is thread-safe)
# ---------------------------------------------------------------------------

def llm_page(client: LLMClient, key: str, args) -> dict:
    """Extract tables/figures from one rendered page and persist its element artifact
    `{pages_dir}/{doc}/{p}.json`. On any error nothing is persisted -> the page retries next run."""
    doc, page_s = key.split("/")
    page_id = int(page_s)
    doc_renders = _join(args.renders_dir, doc)
    side = json.loads(read_bytes(_join(doc_renders, f"{page_id}.json")))
    n_pages, texts, is_blank = side["n_pages"], side["text_items"], side["blank"]
    items: list[list[str]] = [["text", t] for t in texts]
    summary = {"doc": doc, "errored": False, "n_table": 0, "n_figure": 0,
               "n_text": len(texts), "extract_in": 0, "extract_out": 0}

    if not is_blank:
        try:
            png = read_bytes(_join(doc_renders, f"{page_id}.png"))
            img = B64Image(mime="image/png", data=base64.standard_b64encode(png).decode())
            try:
                ext = client.call(system=_EXTRACT_SYSTEM, user=_EXTRACT_USER, images=[img], temperature=0.0,
                                  model=args.extract_model, ctx=None, call_site="fb_extract")
                summary["extract_in"], summary["extract_out"] = ext.input_tokens or 0, ext.output_tokens or 0
                items += [[k, c] for k, c in parse_extract_markdown(ext.text)]
            except EmptyCompletionError:
                # The prompt asks for a `NONE` sentinel on no-table/figure pages, but if the model
                # returns truly empty content anyway, that just means "no tables/figures" — keep the
                # text-only items and persist normally (NOT an error, so the page/doc isn't stuck).
                print(f"  note {key}: empty extract -> no tables/figures", flush=True)
        except Exception as e:  # noqa: BLE001 — one bad page shouldn't kill the run
            summary["errored"] = True
            print(f"  WARN {key}: extract call failed: {e}", flush=True)
            return summary

    try:
        write_output(_join(_join(args.pages_dir, doc), f"{page_id}.json"),
                     json.dumps({"n_pages": n_pages, "items": items}).encode())
    except Exception as e:  # noqa: BLE001 — persist failure -> not done, retry next run
        summary["errored"] = True
        print(f"  WARN {key}: persist failed: {e}", flush=True)
        return summary
    summary["n_table"] = sum(1 for k, _ in items if k == "table")
    summary["n_figure"] = sum(1 for k, _ in items if k == "figure")
    return summary


# ---------------------------------------------------------------------------
# phase 3 — assemble (ThreadPool: one task per doc reads its page artifacts)
# ---------------------------------------------------------------------------

def assemble_doc(doc: str, args) -> tuple[str, int, str]:
    """Assemble one doc's final JSON from its element artifacts. Returns (doc, n_elements, status)
    where status is 'written' | 'incomplete' | 'skip'."""
    out_path = _join(args.output_dir, f"{doc}.json")
    if _exists(out_path):
        return doc, 0, "skip"
    pages_doc = _join(args.pages_dir, doc)
    page_ids = page_ids_under(pages_doc)
    if not page_ids:
        return doc, 0, "skip"
    n_pages = json.loads(read_bytes(_join(pages_doc, f"{min(page_ids)}.json")))["n_pages"]
    if page_ids != set(range(n_pages)):
        return doc, len(set(range(n_pages)) - page_ids), "incomplete"

    elements: list[dict] = []
    eid = 0
    for p in range(n_pages):
        items = json.loads(read_bytes(_join(pages_doc, f"{p}.json")))["items"]
        for etype, content in items:
            elements.append({"id": eid, "page_id": p, "type": etype, "content": content})
            eid += 1
    write_output(out_path, json.dumps({"doc_name": doc, "n_pages": n_pages, "elements": elements}).encode())
    return doc, len(elements), "written"


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    render_pages: int = 0
    render_blank: int = 0
    render_skipped: int = 0
    llm_pages: int = 0
    tables: int = 0
    figures: int = 0
    text_elems: int = 0
    errors: int = 0
    docs_written: int = 0
    docs_incomplete: int = 0
    extract_in: int = 0
    extract_out: int = 0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def phase_render(pdfs: list[str], done_docs: set[str], args, stats: Stats) -> None:
    todo = [p for p in pdfs if _stem(p) not in done_docs]
    print(f"[render] {len(todo)} PDFs over {args.render_procs} processes...", flush=True)
    if not todo:
        return
    with ProcessPoolExecutor(max_workers=args.render_procs) as pex:
        futs = [pex.submit(render_doc, p, args.renders_dir, args.dpi, args.target_element_tokens) for p in todo]
        for fut in as_completed(futs):
            doc, rendered, blank, skipped = fut.result()
            stats.render_pages += rendered
            stats.render_blank += blank
            stats.render_skipped += skipped
            print(f"  [render] {doc}: +{rendered} rendered, {blank} blank, {skipped} already-done", flush=True)


def phase_llm(done_docs: set[str], args, stats: Stats) -> None:
    rendered = {k for k in all_page_keys(args.renders_dir) if k.split("/")[0] not in done_docs}
    done = all_page_keys(args.pages_dir)
    todo = sorted(rendered - done)
    print(f"[llm] {len(todo)} pages to process ({len(done)} already done) over {args.concurrency} threads...", flush=True)
    if not todo:
        return
    client = LLMClient(build_config(args))
    with ThreadPoolExecutor(max_workers=args.concurrency) as tex:
        futs = [tex.submit(llm_page, client, key, args) for key in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            stats.llm_pages += 1
            stats.extract_in += s["extract_in"]
            stats.extract_out += s["extract_out"]
            if s["errored"]:
                stats.errors += 1
                continue
            stats.tables += s["n_table"]
            stats.figures += s["n_figure"]
            stats.text_elems += s["n_text"]
            if i % 500 == 0:
                print(f"  [llm] {i}/{len(todo)} pages...", flush=True)


def phase_assemble(pdfs: list[str], args, stats: Stats) -> None:
    print(f"[assemble] {len(pdfs)} docs...", flush=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as tex:
        futs = [tex.submit(assemble_doc, _stem(p), args) for p in pdfs]
        for fut in as_completed(futs):
            doc, n_elem, status = fut.result()
            if status == "written":
                stats.docs_written += 1
                print(f"  [assemble] {doc}: {n_elem} elements", flush=True)
            elif status == "incomplete":
                stats.docs_incomplete += 1
                print(f"  [assemble] {doc}: INCOMPLETE ({n_elem} page(s) missing) -> rerun to finish", flush=True)


def select_pdfs(args) -> list[str]:
    pdfs = list_pdfs(args.input_dir)
    if args.docs:
        wanted = {d.strip() for d in args.docs.split(",") if d.strip()}
        pdfs = [p for p in pdfs if _stem(p) in wanted]
    if args.world_size > 1:
        pdfs = pdfs[args.rank :: args.world_size]
    if args.sample is not None:
        pdfs = pdfs[: args.sample]
    return pdfs


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose FinanceBench PDFs into text/table/figure element JSONs.")
    parser.add_argument("--input_dir", required=True, help="Local dir OR s3:// prefix of the FinanceBench *.pdf files")
    parser.add_argument("--output_dir", required=True, help="Local dir OR s3:// prefix for the final {doc}.json element files")
    parser.add_argument("--renders_dir", default=None, help="Phase-1 render cache (PNG + text sidecar). Default: {output_dir}-renders")
    parser.add_argument("--pages_dir", default=None, help="Phase-2 per-page element artifacts. Default: {output_dir}-pages")
    parser.add_argument("--phases", default="render,llm,assemble", help="Comma list of phases to run (render,llm,assemble)")
    parser.add_argument("--extract-model", default="google/gemini-3.1-flash-lite", help="Model for table-markdown / figure-summary extraction")
    parser.add_argument("--dpi", type=int, default=200, help="Page render DPI for the vision calls")
    parser.add_argument("--render-procs", type=int, default=os.cpu_count() or 4, help="Phase-1 render processes (CPU-bound)")
    parser.add_argument("--concurrency", type=int, default=32, help="Phase-2/3 threads (I/O-bound LLM + S3)")
    parser.add_argument("--rpm", type=float, default=600.0, help="Per-model requests/min pacing (LLMClient token bucket)")
    parser.add_argument("--max-retries", type=int, default=4, help="Per-call transient-fault retries (429/5xx/transport)")
    parser.add_argument("--target-element-tokens", type=int, default=DEFAULT_TARGET_ELEMENT_TOKENS, help="Target token size for merged text elements")
    parser.add_argument("--sample", type=int, default=None, help="Process only the first N (post-filter) docs")
    parser.add_argument("--docs", type=str, default=None, help="Comma-separated doc_name stems to restrict to")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", 0)), help="Worker index for multi-task splitting")
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)), help="Number of parallel tasks")
    args = parser.parse_args()

    args.renders_dir = args.renders_dir or (args.output_dir.rstrip("/") + "-renders")
    args.pages_dir = args.pages_dir or (args.output_dir.rstrip("/") + "-pages")
    phases = {p.strip() for p in args.phases.split(",") if p.strip()}

    pdfs = select_pdfs(args)
    done_docs = list_done_docs(args.output_dir)  # fully-assembled docs: skipped by every phase
    print(f"Selected {len(pdfs)} PDFs ({len(done_docs)} already assembled). phases={sorted(phases)} "
          f"extract={args.extract_model} dpi={args.dpi}.", flush=True)

    stats = Stats()
    if "render" in phases:
        phase_render(pdfs, done_docs, args, stats)
    if "llm" in phases:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise SystemExit("OPENROUTER_API_KEY is not set in the environment.")
        phase_llm(done_docs, args, stats)
    if "assemble" in phases:
        phase_assemble(pdfs, args, stats)

    print("\n=== summary ===", flush=True)
    print(f"render: +{stats.render_pages} pages, {stats.render_blank} blank, {stats.render_skipped} already-done", flush=True)
    print(f"llm: {stats.llm_pages} pages, tables={stats.tables} figures={stats.figures} "
          f"text_elems={stats.text_elems} errors={stats.errors}", flush=True)
    print(f"assemble: {stats.docs_written} written, {stats.docs_incomplete} incomplete", flush=True)
    print(f"tokens: extract in/out={stats.extract_in}/{stats.extract_out}", flush=True)

    manifest = _join(args.output_dir, f"_manifest_rank{args.rank}.json")
    write_output(manifest, json.dumps({"args": vars(args), "stats": stats.as_dict()}, indent=2).encode())
    print(f"wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
