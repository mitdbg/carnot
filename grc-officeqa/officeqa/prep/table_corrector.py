import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from skunk.common import B64Image, pdf_path_for
from officeqa.config import SkunkConfig
from skunk.llm_client import LLMClient

# Sibling of `page_cleaner.py`. Where the page cleaner re-orders a page's elements,
# this script repairs the extracted grid of each *table* element: it shows the LLM the
# parser's table (HTML) alongside both a full-page render (for context) and a tight crop
# of the table's bounding box (for legibility of small numbers), and asks for the table
# re-emitted as correct GitHub-flavored Markdown.

MAX_RETRIES = 3
MAX_WORKERS = 32
SERIALIZE_MAP_EVERY_N = 100

# Full page render: legible but compact (image-token budget). Crop: sharper, since the
# point of the crop is to read individual cells.
FULL_PAGE_DPI = 150
FULL_PAGE_JPG_QUALITY = 75
CROP_DPI = 220
CROP_JPG_QUALITY = 80
CROP_PAD_PTS = 10  # padding around the table bbox, in PDF points (~0.14 in)

# Parser-coordinate -> PDF-point scale. The parser emits bboxes in a pixel space that is a
# uniform scale of the PDF's point space (it rendered pages at ~190 DPI, so S ~= 2.6). We
# recover S per document from the data (see `estimate_coord_scale`) rather than hardcoding,
# but clamp to a sane band and fall back to this default if estimation fails.
DEFAULT_COORD_SCALE = 2.6
COORD_SCALE_MIN, COORD_SCALE_MAX = 2.0, 3.5

SYSTEM_PROMPT = (
    "You are a meticulous data-extraction proofreader working on tables extracted from "
    "scanned U.S. Treasury Bulletin PDFs. The automatic parser frequently makes structural "
    "mistakes: merged or split cells, columns shifted out of alignment, a number landing in "
    "the wrong column, dropped or duplicated rows, mis-read digits, and lost row/column "
    "headers. Your job is to reconstruct the table exactly as it appears on the page."
)

TABLE_CORRECTOR_PROMPT = """Below is a table that an automatic parser extracted from a PDF page (as HTML). It may contain extraction errors. You are given two images: the full PDF page for context, and a tight crop of the region the table was extracted from so you can read individual cells.

Re-emit the table as a correct GitHub-flavored Markdown table that faithfully matches what is printed on the page. Fix any structural errors (misaligned columns, merged/split cells, wrong or dropped values, missing headers). Preserve the original numbers, footnote markers, and units exactly as printed — do not round, reformat, or invent values. If a cell is genuinely blank on the page, leave it blank.

Output ONLY the corrected Markdown table — no commentary, no explanation, and no code fences.

Parser-extracted table (HTML):
{table_html}"""


def estimate_coord_scale(elements: list[dict], page_heights: dict[int, float]) -> float:
    """Recover the parser's (uniform) coord-space -> PDF-point scale S for one document.

    coord = point * S exactly (linear, no offset), so for any page whose content reaches the
    bottom margin (page numbers / footers almost always do), max(y1)/page_height == S. We take
    a high percentile across pages rather than the max: a slight UNDER-estimate is the safe
    direction (it widens the crop, never clips the table), whereas one anomalous element below
    the page edge would inflate the max and tighten the crop too far."""
    per_page = defaultdict(float)
    for elt in elements:
        for b in elt["bbox"]:
            pid = b["page_id"]
            if pid in page_heights:
                per_page[pid] = max(per_page[pid], b["coord"][3])

    ratios = sorted(
        per_page[pid] / page_heights[pid] for pid in per_page if page_heights[pid] > 0
    )
    if not ratios:
        return DEFAULT_COORD_SCALE
    s = ratios[int(0.90 * (len(ratios) - 1))]
    return min(max(s, COORD_SCALE_MIN), COORD_SCALE_MAX)


def _union_bbox(elt: dict) -> tuple[float, float, float, float]:
    """Union of an element's bbox rectangles (parser-coordinate space)."""
    xs0, ys0, xs1, ys1 = [], [], [], []
    for b in elt["bbox"]:
        x0, y0, x1, y1 = b["coord"]
        xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
    return min(xs0), min(ys0), max(xs1), max(ys1)


def render_page_and_crop(
    month: str, page: int, coord: tuple[float, float, float, float], scale: float, pdf_dir: str
) -> tuple[B64Image | None, B64Image | None]:
    """Open the PDF once and return (full-page image, table-crop image). The crop maps the
    table's parser-space bbox into PDF points (point = coord / scale), pads it, clamps to the
    page, and renders just that region. Returns (None, None) if the PDF/page is missing, and a
    None crop if the bbox is degenerate (worker falls back to the full page alone)."""
    pdf_path = pdf_path_for(month, pdf_dir)
    if not pdf_path.exists():
        return None, None
    import base64

    import fitz

    with fitz.open(pdf_path) as doc:
        if page < 1 or page > doc.page_count:
            return None, None
        pg = doc[page - 1]

        full_pix = pg.get_pixmap(matrix=fitz.Matrix(FULL_PAGE_DPI / 72, FULL_PAGE_DPI / 72))
        full = B64Image(
            mime="image/jpeg",
            data=base64.standard_b64encode(full_pix.tobytes("jpg", jpg_quality=FULL_PAGE_JPG_QUALITY)).decode(),
        )

        r = pg.rect
        x0, y0, x1, y1 = coord
        clip = fitz.Rect(
            max(0.0, x0 / scale - CROP_PAD_PTS),
            max(0.0, y0 / scale - CROP_PAD_PTS),
            min(r.width, x1 / scale + CROP_PAD_PTS),
            min(r.height, y1 / scale + CROP_PAD_PTS),
        )
        crop = None
        if clip.width > 2 and clip.height > 2:
            crop_pix = pg.get_pixmap(matrix=fitz.Matrix(CROP_DPI / 72, CROP_DPI / 72), clip=clip)
            if crop_pix.width > 0 and crop_pix.height > 0:
                crop = B64Image(
                    mime="image/jpeg",
                    data=base64.standard_b64encode(
                        crop_pix.tobytes("jpg", jpg_quality=CROP_JPG_QUALITY)
                    ).decode(),
                )
    return full, crop


def _strip_code_fence(text: str) -> str:
    """Drop a wrapping ```...``` fence if the model added one despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    return t.strip()


def correct_table(task: dict, pdfs_dir: str, llm: LLMClient) -> str | None:
    """Correct one table element. Returns the corrected Markdown, or None on failure."""
    try:
        full, crop = render_page_and_crop(
            task["month"], task["page_id"], task["coord"], task["scale"], pdfs_dir
        )
        if full is None:
            print(f"Error correcting table ({task['key']}): could not render page.")
            return None
        images = [full] + ([crop] if crop is not None else [])

        user_text = TABLE_CORRECTOR_PROMPT.format(table_html=task["content"])
        for _ in range(MAX_RETRIES):
            markdown = _strip_code_fence(llm.call(system=SYSTEM_PROMPT, user=user_text, images=images).text)
            if markdown:
                return markdown
        print(f"Error correcting table ({task['key']}): empty response after {MAX_RETRIES} retries.")
        return None
    except Exception as e:
        print(f"Error correcting table ({task['key']}): {e}")
        return None


def _page_heights(month: str, pdfs_dir: str) -> dict[int, float]:
    """1-indexed page_id -> page height in PDF points, used to calibrate the coord scale."""
    pdf_path = pdf_path_for(month, pdfs_dir)
    if not pdf_path.exists():
        return {}
    import fitz

    with fitz.open(pdf_path) as doc:
        return {i + 1: doc[i].rect.height for i in range(doc.page_count)}


def build_work_list(
    input_json_dir: str,
    pdfs_dir: str,
    done_keys: set[str],
    months: set[str] | None = None,
    pages: set[int] | None = None,
) -> list[dict]:
    """Walk the parsed-JSON directory and emit one task per (not-yet-done) table element.

    `months` (e.g. {"1970-06"}) and `pages` (1-indexed page_ids), when given, restrict the
    work list to that subset — handy for sampling a few pages before a full sweep."""
    tasks = []
    for _, _, files in sorted(os.walk(input_json_dir)):
        for file in sorted(files):
            match = re.match(r"treasury_bulletin_(\d{4})_(\d{2})\.json", file)
            if not match:
                continue
            year, month = match.groups()
            if months is not None and f"{year}-{month}" not in months:
                continue
            with open(os.path.join(input_json_dir, file)) as f:
                doc = json.load(f)

            elements = doc["document"]["elements"]
            heights = _page_heights(f"{year}-{month}", pdfs_dir)
            scale = estimate_coord_scale(elements, heights)

            for elt in elements:
                if elt.get("type") != "table" or not elt.get("content"):
                    continue
                page_id = elt["bbox"][0]["page_id"]
                if pages is not None and page_id not in pages:
                    continue
                # NOTE: element `id` is only unique WITHIN a page (it repeats across pages),
                # so the table's identity — and the map key / filename — must include page_id.
                key = f"{year}_{month}_{page_id}_{elt['id']}"
                if key in done_keys:
                    continue
                tasks.append({
                    "key": key,
                    "year": year,
                    "month": f"{year}-{month}",
                    "element_id": elt["id"],
                    "page_id": page_id,
                    "coord": _union_bbox(elt),
                    "content": elt["content"],
                    "scale": scale,
                })
    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correct extracted table elements into clean Markdown.")
    parser.add_argument("--input-json-dir", default="treasury_bulletins_parsed/jsons",
                        help="Directory with the parsed document JSONs.")
    parser.add_argument("--pdfs-dir", default="treasury_bulletin_pdfs",
                        help="Directory with the source PDFs (for page renders).")
    parser.add_argument("--output-dir", default="treasury_bulletins_tables_corrected",
                        help="Output directory for the corrected-table .md files and the map.")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--months", default=None,
                        help="Comma-separated YYYY-MM to restrict to (e.g. '1970-06'). Default: all.")
    parser.add_argument("--pages", default=None,
                        help="Comma-separated 1-indexed page_ids to restrict to (e.g. '89'). Default: all.")
    args = parser.parse_args()

    months = {m.strip() for m in args.months.split(",")} if args.months else None
    pages = {int(p.strip()) for p in args.pages.split(",")} if args.pages else None

    if not os.path.isdir(args.input_json_dir):
        print(f"Error: Input directory {args.input_json_dir} does not exist.")
        exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the existing map so a previous run can be resumed.
    table_map_path = os.path.join(args.output_dir, "table_corrections_map.json")
    table_map: dict[str, list] = {}
    if os.path.isfile(table_map_path):
        with open(table_map_path) as f:
            table_map = json.load(f)

    tasks = build_work_list(args.input_json_dir, args.pdfs_dir, set(table_map.keys()), months, pages)
    print(f"{len(tasks)} table(s) to correct ({len(table_map)} already done).")

    llm = LLMClient(SkunkConfig.from_env())

    completed, errored = 0, 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {executor.submit(correct_table, t, args.pdfs_dir, llm): t for t in tasks}

        with tqdm(total=len(tasks), desc="Correcting tables", unit="table") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                markdown = future.result()

                if markdown is not None:
                    completed += 1
                    md_path = os.path.join(
                        args.output_dir,
                        f"treasury_bulletin_{task['year']}_{task['month'][-2:]}"
                        f"_p{task['page_id']}_table_{task['element_id']}.md",
                    )
                    with open(md_path, "w") as f:
                        f.write(markdown)
                    table_map[task["key"]] = [md_path, task["page_id"]]

                    if len(table_map) % SERIALIZE_MAP_EVERY_N == 0:
                        with open(table_map_path, "w") as f:
                            json.dump(table_map, f)
                else:
                    errored += 1

                pbar.update(1)
                pbar.set_postfix(completed=completed, errored=errored)

    with open(table_map_path, "w") as f:
        json.dump(table_map, f)

    print(f"Done. Completed {completed}, errored {errored}.")
