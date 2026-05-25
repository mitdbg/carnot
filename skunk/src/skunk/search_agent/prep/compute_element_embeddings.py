"""
Compute embeddings for Treasury Bulletin elements using
``gemini-embedding-2`` on GCP Vertex AI via the ``google-genai`` SDK.

For each .json file in --input_dir:
  * Each non-skipped element's text is preprocessed (table HTML stripped, numeric
    cells dropped, dot-leaders removed, optional year stripping).
  * If the resulting text exceeds the model's 8k-token context limit it is
    split into non-overlapping chunks; the per-chunk embeddings are averaged
    and re-normalized to produce a single embedding for the element.

Requests are issued in parallel via a ThreadPoolExecutor (default 64 workers).
Embeddings are written to ``--output_dir`` split across ``--n_partitions``
.npz files so each file stays small enough to fit comfortably in memory.

Usage:
    python compute_element_embeddings.py \\
        --input_dir ./treasury_bulletins_parsed \\
        --output_dir ./treasury_bulletin_embeddings
"""

import argparse
import glob
import html as _html
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
from google import genai
from google.genai import types as genai_types  # noqa: F401
from skunk.common import _make_vertex_client

# Gemini Embedding 2 Preview context limit (tokens).
MAX_TOKENS = 8192
# Conservative chunk size; leaves headroom for the model's own tokenization.
CHUNK_TOKENS = 8000
# Rough chars-per-token used for chunking long inputs.
CHARS_PER_TOKEN = 3
CHUNK_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN
MAX_ATTEMPTS = 6
MODEL_NAME = "gemini-embedding-2"  # Vertex AI model ID

_MULTI_NL_RE = re.compile(r"\n{3,}")
_TD_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_LONG_DOTS_RE = re.compile(r"\.{4,}")
# Matches four-digit years in the range 1776–2026 as whole tokens.
_YEAR_RE = re.compile(r"\b(177[6-9]|17[89]\d|1[89]\d\d|200\d|201\d|202[0-6])\b")


def preprocess_text(text: str, elt_type: str, strip_years: bool = False) -> str:
    """Preprocess element text for embedding.

    For table elements: strips all HTML tags, drops purely-numeric cells
    (page numbers), and joins remaining cell text with spaces.
    For all elements: collapses 3+-newline runs and removes dot-leader
    sequences (4 or more consecutive periods).
    If *strip_years* is True, removes all four-digit years in the range
    1776–2026 from every element type.
    """
    if elt_type == "table":
        cells = [_HTML_TAG_RE.sub("", _html.unescape(m.group(1))).strip() for m in _TD_RE.finditer(text)]
        cells = [c for c in cells if not re.fullmatch(
            r"[+\-]?\$\s*[\d,]+(\.\d+)?%?"  # $1,234.56 / $ 194.3
            r"|[+\-]?[\d,]+(\.\d+)?%?"       # 1,234 / 3.5 / 50%
            r"|\d[\d,]*[\/\-]\d[\d,]*"        # 4-5 / 283/444
            r"|\$\s*-+"                        # $ -- / $ -
            r"|\$\s*\.\d+"                     # $ .6 / $.1
            r"|\.\d+"                          # .6 / .2 (bare decimals)
            r"|\*+"                            # * / ** (footnote markers)
            r"|-{2,}",                         # -- / --- (dash placeholders)
            c
        )]
        text = " ".join(cells)
    else:
        text = _MULTI_NL_RE.sub("\n\n", text)

    # Remove dot-leader runs (4+ consecutive periods) from all element types.
    text = _LONG_DOTS_RE.sub("", text)

    if strip_years:
        text = _YEAR_RE.sub("", text)

    return text.strip()


def _chunk_by_chars(text: str, chunk_chars: int) -> list[str]:
    """Split *text* into non-overlapping chunks of <= chunk_chars characters."""
    if len(text) <= chunk_chars:
        return [text]
    return [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]


def _embed_request(client: genai.Client, inputs: list[str]) -> list[np.ndarray]:
    """Embed a batch of strings with jittered exponential-backoff retry.

    Empty strings must never appear in *inputs* — the API rejects any batch
    that contains one. We detect rate-limits (429) by inspecting the error
    message and apply a longer base delay; all other errors use standard
    doubling back-off.
    """
    delay = 1.0
    for attempt in range(MAX_ATTEMPTS):
        try:
            result = client.models.embed_content(
                model=MODEL_NAME,
                contents=[
                    genai_types.Content(parts=[genai_types.Part.from_text(text=s)])
                    for s in inputs
                ],
            )
            return [np.asarray(emb.values, dtype=np.float32) for emb in result.embeddings]  # type: ignore

        except Exception as e:  # noqa: BLE001 - retry on any transient API error
            if attempt == MAX_ATTEMPTS - 1:
                raise
            is_rate_limit = "429" in str(e)
            base = 10.0 if is_rate_limit else delay
            jittered = base * (0.5 + random.random())  # [0.5×base, 1.5×base]
            print(f"  embedding request failed (attempt {attempt + 1}): {e}; retrying in {jittered:.1f}s")
            time.sleep(jittered)
            if not is_rate_limit:
                delay *= 2
    raise RuntimeError("unreachable")


def embed_text(text: str, client: genai.Client) -> np.ndarray:
    """Embed a single element, chunking and averaging if it exceeds context."""
    # The API rejects empty strings with a 400; replace with a single space so
    # elements that preprocess to nothing still produce a valid (near-zero) embedding.
    chunks = [c if c else " " for c in _chunk_by_chars(text, CHUNK_CHARS)]
    embs = _embed_request(client, chunks)
    v = embs[0] if len(embs) == 1 else np.stack(embs, axis=0).mean(axis=0)
    norm = float(np.linalg.norm(v))
    if norm > 0:
        v = v / norm
    return v.astype(np.float32)


def _partition_path(output_dir: str, p: int) -> str:
    return os.path.join(output_dir, f"embeddings_{p}.npz")


def _partition_bounds(p: int, partition_size: int, n_total: int) -> tuple[int, int]:
    start = p * partition_size
    end = min(start + partition_size, n_total)
    return start, end


def embed_all(
    texts: list[str],
    unique_element_ids: list[str],
    client: genai.Client,
    n_partitions: int,
    output_dir: str,
    max_workers: int,
) -> None:
    """Embed *texts* in parallel and save partitions incrementally to disk.

    Each partition contains up to ``ceil(n_total / n_partitions)`` consecutive
    elements and is saved as ``embeddings_{partition_id}.npz`` with arrays
    ``embeddings`` (float32, shape [n, d]) and ``unique_element_ids`` (str).
    Already-existing partition files are skipped so the job can resume after
    being killed.
    """
    n_total = len(texts)
    assert len(unique_element_ids) == n_total
    partition_size = max(1, (n_total + n_partitions - 1) // n_partitions)
    n_partitions = (n_total + partition_size - 1) // partition_size

    completed_partitions: set[int] = {
        p for p in range(n_partitions)
        if os.path.exists(_partition_path(output_dir, p))
    }
    if completed_partitions:
        print(
            f"Resume: skipping {len(completed_partitions)}/{n_partitions} "
            f"already-completed partitions."
        )

    todo_indices = [
        i for i in range(n_total)
        if (i // partition_size) not in completed_partitions
    ]
    n_to_do = len(todo_indices)
    if n_to_do == 0:
        print("Nothing to do; all partitions already exist.")
        return

    buffers: dict[int, dict[int, np.ndarray]] = {}
    buffers_lock = Lock()
    log_every = max(100, n_to_do // 1000)
    done_count = 0
    done_lock = Lock()

    def _save_partition(p: int) -> None:
        start, end = _partition_bounds(p, partition_size, n_total)
        buf = buffers.pop(p)
        embs = np.stack([buf[k] for k in range(end - start)], axis=0)
        ids = np.array(unique_element_ids[start:end])
        path = _partition_path(output_dir, p)
        np.savez_compressed(path, embeddings=embs, unique_element_ids=ids)
        print(f"  saved partition {p} -> {path} (shape: {embs.shape})")

    def _record(i: int, emb: np.ndarray) -> None:
        p = i // partition_size
        start, end = _partition_bounds(p, partition_size, n_total)
        with buffers_lock:
            buf = buffers.setdefault(p, {})
            buf[i - start] = emb
            ready = len(buf) == end - start
        if ready:
            _save_partition(p)

    def _task(i: int) -> int:
        emb = embed_text(texts[i], client)
        _record(i, emb)
        return i

    print(f"Embedding {n_to_do} elements with {max_workers} parallel workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_task, i) for i in todo_indices]
        for fut in as_completed(futures):
            fut.result()
            with done_lock:
                done_count += 1
                if done_count % log_every == 0 or done_count == n_to_do:
                    pct = 100 * done_count / n_to_do
                    print(f"  embedded {done_count}/{n_to_do} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute Gemini Embedding 2 embeddings for each Treasury "
            "Bulletin element via GCP Vertex AI."
        )
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing .json files (one document each)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save embedding outputs")
    parser.add_argument("--strip_years", action="store_true", default=False,
                        help="Remove four-digit years (1776–2026) from all elements before embedding.")
    parser.add_argument("--n_partitions", type=int, default=100,
                        help="Number of embedding partition files to produce.")
    parser.add_argument("--max_workers", type=int, default=16,
                        help="Maximum number of concurrent API requests.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Auth via ADC + GOOGLE_CLOUD_PROJECT (see skunk.common._make_vertex_client).
    client = _make_vertex_client()

    json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    print(f"Processing {len(json_files)} .json files.")

    metadata: dict[str, dict] = {}
    cleaned_elements: list[str] = []
    unique_element_ids: list[str] = []

    for json_path in json_files:
        # file_id format: "treasury_bulletin_yyyy_mm.json"
        file_id = Path(json_path).stem
        match = re.match(r"treasury_bulletin_(\d{4})_(\d{2})", file_id)
        year, month = match.groups()  # type: ignore
        with open(json_path) as f:
            data = json.load(f)

        elements_lst = data['document']['elements']
        for element in elements_lst:
            elt_type = element['type']
            if elt_type in {"page_number", "page_footer", "page_header", "figure"}:
                continue

            page_id = element['bbox'][0]['page_id']
            elt_id = element['id']
            unique_element_id = f"{file_id}_{page_id}_{elt_id}"
            raw = element['content']
            cleaned = preprocess_text(raw, elt_type, strip_years=args.strip_years)

            unique_element_ids.append(unique_element_id)
            cleaned_elements.append(cleaned)
            metadata[unique_element_id] = {
                "file_id": file_id,
                "raw": raw,
                "cleaned": cleaned,
                "year": year,
                "month": month,
                "page_id": page_id,
                "page_key": f"{year}_{month}_{page_id}",
                "element_id": elt_id,
                "type": elt_type,
            }

    assert len(cleaned_elements) == len(metadata), "Duplicate element IDs found; check input data for consistency."
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    print(f"Saved metadata for {len(metadata)} elements to {metadata_path}")
    del metadata

    embed_all(
        cleaned_elements,
        unique_element_ids,
        client,
        n_partitions=args.n_partitions,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    )
    print("Embedding complete.")


if __name__ == "__main__":
    main()
