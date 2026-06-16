"""Compute Qwen3-Embedding-8B element embeddings for the DAIS slim corpus via OpenRouter.

Follows the OfficeQA ``qwen-v2`` recipe, with two deliberate differences: the embedding
backend is OpenRouter (no GPU available) instead of a local ``SentenceTransformer``, and
table cells are stripped more aggressively (``drop_numeric_cells=True``):

  * each non-skipped parsed-JSON element's **raw** content is cleaned with
    ``skunk.corpus.preprocess_text(raw, elt_type, strip_years=False, drop_numeric_cells=True)``
    — same as the ``qwen-v2`` text except table cells that are entirely numeric (incl.
    multi-token values like dollars+cents "$216,370,286 77" or value+footnote "9,198 2/")
    are dropped, leaving mostly headers / row & column names / footnotes;
  * the cleaned text is embedded with ``qwen/qwen3-embedding-8b`` through OpenRouter (the
    same call the SearchAgent already uses at query time), chunked + averaged + L2-normalized
    if it exceeds the model context;
  * outputs match ``search_agent/prep/create_vector_db.py``'s expected layout: partitioned
    ``embeddings_{p}.npz`` (arrays ``embeddings`` float32 [n,4096], ``unique_element_ids``)
    plus a ``metadata.json`` mapping ``chunk_id -> per-element metadata``.

Identifiers are generic: ``page_key/doc_id = f"{file_id}_{page_id}"``,
``chunk_id = f"{file_id}_{page_id}_{element_id}"``.

Usage:
    OPENROUTER_API_KEY=... python3 -m skunk.dais.compute_qwen_embeddings \\
        --input_dir parsed/jsons --output_dir dais_embeddings
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np

from skunk.common import get_rate_limiter
from skunk.corpus import preprocess_text
from skunk.dais.dais_common import SKIP_TYPES, chunk_id_of, doc_id_of, file_id_of, source_of, year_of

MODEL_NAME = "qwen/qwen3-embedding-8b"
# Qwen3-Embedding-8B supports ~32k tokens; leave headroom and chunk in char-space
# (~4 chars/token, matching compute_browsecomp_plus_element_embeddings.py).
CHUNK_TOKENS = 30000
CHARS_PER_TOKEN = 4
CHUNK_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN
MAX_ATTEMPTS = 6
# Max chars across all inputs in one batched request (~15k tokens) — keeps each request
# well within the model context while letting many small elements share one HTTP round-trip.
BATCH_CHARS_BUDGET = 60000


def _make_openrouter_client():
    from openrouter import OpenRouter

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set (required for qwen embeddings via OpenRouter).")
    return OpenRouter(api_key=key)


def _chunk_by_chars(text: str, chunk_chars: int) -> list[str]:
    if len(text) <= chunk_chars:
        return [text]
    return [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]


def _embed_request(client, inputs: list[str]) -> list[np.ndarray]:
    """Embed a batch of strings with jittered exponential-backoff retry.

    Sends a single string for one input (the proven runtime shape) and a list otherwise;
    OpenAI-compatible endpoints return ``data`` aligned with the input order.
    """
    rate = get_rate_limiter("embed")
    delay = 1.0
    for attempt in range(MAX_ATTEMPTS):
        try:
            rate.acquire()
            arg = inputs[0] if len(inputs) == 1 else inputs
            resp = client.embeddings.generate(input=arg, model=MODEL_NAME)
            # OpenAI-compatible APIs may return data out of order; sort by `index` so the
            # returned vectors align 1:1 with `inputs` for batched requests.
            data = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
            return [np.asarray(d.embedding, dtype=np.float32) for d in data]
        except Exception as e:  # noqa: BLE001 - retry on any transient API error
            if attempt == MAX_ATTEMPTS - 1:
                raise
            is_rate_limit = "429" in str(e)
            base = 10.0 if is_rate_limit else delay
            jittered = base * (0.5 + random.random())
            print(f"  embedding request failed (attempt {attempt + 1}): {e}; retrying in {jittered:.1f}s")
            time.sleep(jittered)
            if not is_rate_limit:
                delay *= 2
    raise RuntimeError("unreachable")


def embed_text(text: str, client) -> np.ndarray:
    """Embed one element: chunk + average + L2-normalize (matches the qwen-v2 recipe)."""
    # Endpoints reject empty input; map empty cleaned text to a single space so the element
    # still yields a valid (near-degenerate) vector instead of erroring the whole batch.
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
    return start, min(start + partition_size, n_total)


def _make_batches(simple: list[int], texts: list[str], max_items: int) -> list[list[int]]:
    """Group single-chunk element indices into batches bounded by item count and char budget,
    so each request packs many small elements without exceeding the model context."""
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_chars = 0
    for i in simple:
        t = len(texts[i]) or 1
        if cur and (len(cur) >= max_items or cur_chars + t > BATCH_CHARS_BUDGET):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(i)
        cur_chars += t
    if cur:
        batches.append(cur)
    return batches


def embed_all(
    texts: list[str],
    unique_element_ids: list[str],
    client,
    n_partitions: int,
    output_dir: str,
    max_workers: int,
    batch_size: int,
) -> None:
    """Embed in parallel and save partitions incrementally (resume by skipping existing).

    Single-chunk elements are packed into one request each (up to ``batch_size`` items /
    ``BATCH_CHARS_BUDGET`` chars) — one rate-limiter token per *request*, so throughput is no
    longer one element per round-trip. Oversized elements (needing per-element chunk+average)
    are embedded individually.
    """
    n_total = len(texts)
    assert len(unique_element_ids) == n_total
    partition_size = max(1, (n_total + n_partitions - 1) // n_partitions)
    n_partitions = (n_total + partition_size - 1) // partition_size

    completed = {p for p in range(n_partitions) if os.path.exists(_partition_path(output_dir, p))}
    if completed:
        print(f"Resume: skipping {len(completed)}/{n_partitions} already-completed partitions.")

    todo = [i for i in range(n_total) if (i // partition_size) not in completed]
    if not todo:
        print("Nothing to do; all partitions already exist.")
        return

    buffers: dict[int, dict[int, np.ndarray]] = {}
    buffers_lock = Lock()
    done_lock = Lock()
    done_count = 0
    log_every = max(100, len(todo) // 1000)

    def _save_partition(p: int) -> None:
        start, end = _partition_bounds(p, partition_size, n_total)
        buf = buffers.pop(p)
        embs = np.stack([buf[k] for k in range(end - start)], axis=0)
        ids = np.array(unique_element_ids[start:end])
        np.savez_compressed(_partition_path(output_dir, p), embeddings=embs, unique_element_ids=ids)
        print(f"  saved partition {p} (shape {embs.shape})")

    def _record(i: int, emb: np.ndarray) -> None:
        p = i // partition_size
        start, end = _partition_bounds(p, partition_size, n_total)
        with buffers_lock:
            buf = buffers.setdefault(p, {})
            buf[i - start] = emb
            ready = len(buf) == end - start
        if ready:
            _save_partition(p)

    def _do_batch(idxs: list[int]) -> list[tuple[int, np.ndarray]]:
        inputs = [texts[i] if texts[i] else " " for i in idxs]
        vecs = _embed_request(client, inputs)
        out: list[tuple[int, np.ndarray]] = []
        for i, v in zip(idxs, vecs, strict=True):
            nrm = float(np.linalg.norm(v))
            if nrm > 0:
                v = v / nrm
            out.append((i, v.astype(np.float32)))
        return out

    def _do_oversized(i: int) -> list[tuple[int, np.ndarray]]:
        return [(i, embed_text(texts[i], client))]

    simple = [i for i in todo if len(texts[i]) <= CHUNK_CHARS]
    oversized = [i for i in todo if len(texts[i]) > CHUNK_CHARS]
    batches = _make_batches(simple, texts, max(1, batch_size))

    print(f"Embedding {len(todo)} elements: {len(batches)} batches "
          f"(<= {batch_size} items / {BATCH_CHARS_BUDGET} chars) + {len(oversized)} oversized, "
          f"{max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_do_batch, b) for b in batches]
        futures += [ex.submit(_do_oversized, i) for i in oversized]
        for fut in as_completed(futures):
            recs = fut.result()
            for i, emb in recs:
                _record(i, emb)
            with done_lock:
                prev = done_count
                done_count += len(recs)
                if done_count // log_every != prev // log_every or done_count == len(todo):
                    print(f"  embedded {done_count}/{len(todo)} ({100 * done_count / len(todo):.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Qwen3-Embedding-8B embeddings (OpenRouter).")
    parser.add_argument("--input_dir", required=True, help="Directory with parsed document JSONs.")
    parser.add_argument("--output_dir", required=True, help="Directory to save embedding outputs.")
    parser.add_argument("--n_partitions", type=int, default=100, help="Number of .npz partition files.")
    parser.add_argument("--max_workers", type=int, default=16, help="Concurrent embedding requests.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Max elements packed into one embedding request (1 = per-element).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    client = _make_openrouter_client()

    json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    print(f"Processing {len(json_files)} .json files.")

    metadata: dict[str, dict] = {}
    cleaned_elements: list[str] = []
    unique_element_ids: list[str] = []

    for json_path in json_files:
        file_id = file_id_of(json_path)
        year = year_of(file_id)        # annual corpus: int year (or None) parsed from file_id
        source = source_of(file_id)    # era/source family (historical/modern/transition/govinfo_receipts)
        with open(json_path) as f:
            data = json.load(f)
        for element in data["document"]["elements"]:
            elt_type = element["type"]
            if elt_type in SKIP_TYPES:
                continue
            page_id = element["bbox"][0]["page_id"]
            elt_id = element["id"]
            cid = chunk_id_of(file_id, page_id, elt_id)
            raw = element["content"] or ""
            # drop_numeric_cells=True: strip pure-value table cells (incl. dollars+cents and
            # value+footnote), keeping mostly headers/row-names/footnotes for the embedding.
            cleaned = preprocess_text(raw, elt_type, strip_years=False, drop_numeric_cells=True)

            unique_element_ids.append(cid)
            cleaned_elements.append(cleaned)
            metadata[cid] = {
                "file_id": file_id,
                "raw": raw,
                "cleaned": cleaned,
                "page_id": page_id,
                "page_key": doc_id_of(file_id, page_id),
                "element_id": elt_id,
                "type": elt_type,
                "year": year,
                "source": source,
            }

    assert len(cleaned_elements) == len(metadata), "Duplicate chunk ids found; check input consistency."
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
        batch_size=args.batch_size,
    )
    print("Embedding complete.")


if __name__ == "__main__":
    main()
