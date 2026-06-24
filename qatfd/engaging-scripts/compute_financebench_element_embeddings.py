"""Compute element-level embeddings for the FinanceBench corpus using Qwen3-Embedding-8B.

Input is the element JSONs produced by `preprocess_financebench_pdfs.py` — one `{doc_name}.json`
per filing under --input_dir, each:
    {"doc_name": ..., "n_pages": N, "elements": [{"id", "page_id", "type", "content"}, ...]}
where `type` is one of text / table / figure. Each element becomes one embedding:
  * its `content` is embedded directly (table markdown / figure summary / merged text paragraphs);
  * if an element exceeds the model's ~32k-token context it is split into non-overlapping
    ~32k-token chunks whose per-chunk embeddings are averaged and re-normalized.

Retrieval stays PAGE-level (matching KARL): the chroma `doc_id` is the page key
"{doc_name}::p{page_id}" (so it lines up with FinanceBench's page-level gold), while the chroma
row id / chunk_id is the per-element "{doc_name}::p{page_id}::e{element_id}". The emitted
metadata_rank{r}.json feeds `create_vector_db.py --benchmark finance_bench`, and the per-element
`cleaned` text is what the FinanceBench benchmark's document_map is rebuilt from (concatenated per
page in element order).

`--input_dir` and `--output_dir` each accept a local path OR an `s3://bucket/prefix` URI: with S3
the element JSONs are streamed in and the embeddings/metadata streamed out, so the Engaging GPU
job keeps ~zero local disk (only the model cache lives on disk). S3 needs `boto3` (pip install) +
AWS creds in the env; boto3 is imported lazily so local runs don't require it.

Usage:
    python compute_financebench_element_embeddings.py \\
        --input_dir /home/mdrusso/carnot/skunk/financebench/financebench-elements \\
        --output_dir /home/mdrusso/carnot/skunk/financebench/financebench-element-embeddings
    # or stream to/from S3 (see run_financebench_element_embeddings.slurm):
    #   --input_dir s3://carnot-research/financebench/financebench-elements \\
    #   --output_dir s3://carnot-research/financebench/financebench-element-embeddings
"""

import argparse
import dataclasses
import glob
import io
import json
import os
from urllib.parse import urlparse

import numpy as np
from sentence_transformers import SentenceTransformer

# Qwen3-Embedding-8B supports up to 32k tokens; leave a little headroom.
MAX_TOKENS = 32000
CHUNK_TOKENS = 31500
# Token budget per batch: batch_size × max_seq_len_in_batch must not exceed this.
BATCH_TOKEN_BUDGET = 16384

# Page key separator — MUST match qatfd.benchmarks.financebench._PAGE_SEP and the chroma adapter.
# It cannot occur in a PDF filename stem, so a key is unambiguously splittable into (doc, page).
_PAGE_SEP = "::p"


# ---------------------------------------------------------------------------
# I/O layer: --input_dir / --output_dir may be a local dir OR an `s3://bucket/prefix`
# URI, so the Engaging job can stream the element JSONs in and the embeddings/metadata
# out of S3 and keep ~zero local disk. boto3 reads credentials from the standard AWS_*
# env vars; it's imported lazily so local runs don't need it. (Mirrors compute_biogen_embeddings.py
# and preprocess_financebench_pdfs.py.)
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
    """Join a path/URI with a filename (works for both local paths and s3:// URIs)."""
    return base.rstrip("/") + "/" + name


def _exists(path: str) -> bool:
    if _is_s3(path):
        bucket, key = _s3_split(path)
        try:
            _s3().head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False
    return os.path.exists(path)


def list_json(input_dir: str) -> list[str]:
    """Sorted list of every *.json under a local dir or an s3:// prefix."""
    if _is_s3(input_dir):
        bucket, prefix = _s3_split(input_dir)
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        keys: list[str] = []
        for page in _s3().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            keys += [f"s3://{bucket}/{o['Key']}" for o in page.get("Contents", []) if o["Key"].endswith(".json")]
        return sorted(keys)
    return sorted(glob.glob(os.path.join(input_dir, "*.json")))


def read_bytes(path: str) -> bytes:
    """Read a file's bytes from a local path or an s3:// object (one GET, held in memory)."""
    if _is_s3(path):
        bucket, key = _s3_split(path)
        return _s3().get_object(Bucket=bucket, Key=key)["Body"].read()
    with open(path, "rb") as f:
        return f.read()


def write_bytes(dest: str, data: bytes) -> None:
    if _is_s3(dest):
        bucket, key = _s3_split(dest)
        # upload_fileobj does multipart automatically, so it handles the (potentially large)
        # per-rank metadata json without hitting S3's 5 GB single-PutObject limit.
        _s3().upload_fileobj(io.BytesIO(data), bucket, key)
    else:
        with open(dest, "wb") as f:
            f.write(data)


def write_npz(dest: str, embeddings: np.ndarray, ids: np.ndarray) -> None:
    buf = io.BytesIO()
    np.savez_compressed(buf, embeddings=embeddings, unique_element_ids=ids)
    write_bytes(dest, buf.getvalue())


def _chunk_by_tokens(text: str, tokenizer, chunk_tokens: int) -> list[str]:
    """Split `text` into non-overlapping chunks of <= chunk_tokens tokens."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= chunk_tokens:
        return [text]
    chunks = []
    for start in range(0, len(token_ids), chunk_tokens):
        piece_ids = token_ids[start : start + chunk_tokens]
        chunks.append(tokenizer.decode(piece_ids, skip_special_tokens=True))
    return chunks


def embed_oversized_text(text: str, model: SentenceTransformer) -> np.ndarray:
    """Embed a single element, chunking and averaging if it exceeds context."""
    tokenizer = model.tokenizer
    chunks = _chunk_by_tokens(text, tokenizer, CHUNK_TOKENS)
    if len(chunks) == 1:
        emb = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(emb, dtype=np.float32)

    chunk_embs = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False, batch_size=1)
    avg = np.asarray(chunk_embs, dtype=np.float32).mean(axis=0)
    norm = float(np.linalg.norm(avg))
    if norm > 0:
        avg = avg / norm
    return avg.astype(np.float32)


def _partition_path(output_dir: str, rank: int, p: int) -> str:
    return _join(output_dir, f"embeddings_{rank}_{p}.npz")


def _partition_bounds(p: int, partition_size: int, n_total: int) -> tuple[int, int]:
    start = p * partition_size
    end = min(start + partition_size, n_total)
    return start, end


def _save_partition(
    p: int,
    buffers: dict[int, dict[int, np.ndarray]],
    unique_element_ids: list[str],
    partition_size: int,
    n_total: int,
    output_dir: str,
    rank: int,
    log_prefix: str,
) -> None:
    start, end = _partition_bounds(p, partition_size, n_total)
    buf = buffers[p]
    embs = np.stack([buf[k] for k in range(end - start)], axis=0)
    ids = np.array(unique_element_ids[start:end])
    path = _partition_path(output_dir, rank, p)
    write_npz(path, embs, ids)
    print(f"{log_prefix}  saved partition {p} -> {path} (shape: {embs.shape})", flush=True)
    del buffers[p]


def _record(
    i: int,
    emb: np.ndarray,
    buffers: dict[int, dict[int, np.ndarray]],
    unique_element_ids: list[str],
    partition_size: int,
    n_total: int,
    output_dir: str,
    rank: int,
    log_prefix: str,
) -> None:
    p = i // partition_size
    buf = buffers.setdefault(p, {})
    buf[i - p * partition_size] = emb
    start, end = _partition_bounds(p, partition_size, n_total)
    if len(buf) == end - start:
        _save_partition(p, buffers, unique_element_ids, partition_size, n_total, output_dir, rank, log_prefix)


@dataclasses.dataclass
class _BatchState:
    indices: list[int] = dataclasses.field(default_factory=list)
    max_tokens: int = 0
    n_done: int = 0
    last_logged: int = 0


def _flush(
    state: _BatchState,
    texts: list[str],
    model: SentenceTransformer,
    buffers: dict[int, dict[int, np.ndarray]],
    unique_element_ids: list[str],
    partition_size: int,
    n_total: int,
    n_to_do: int,
    output_dir: str,
    rank: int,
    log_every: int,
    log_prefix: str,
) -> None:
    if not state.indices:
        return
    embs = model.encode([texts[i] for i in state.indices], normalize_embeddings=True, show_progress_bar=False)
    for j, i in enumerate(state.indices):
        _record(i, np.asarray(embs[j], dtype=np.float32), buffers, unique_element_ids, partition_size, n_total, output_dir, rank, log_prefix)
    state.n_done += len(state.indices)
    if state.n_done - state.last_logged >= log_every or state.n_done == n_to_do:
        pct = 100 * state.n_done / max(n_to_do, 1)
        print(f"{log_prefix}  embedded {state.n_done}/{n_to_do} ({pct:.1f}%)", flush=True)
        state.last_logged = state.n_done
    state.indices.clear()
    state.max_tokens = 0


def embed_all(
    texts: list[str],
    unique_element_ids: list[str],
    model: SentenceTransformer,
    partition_size: int,
    output_dir: str,
    rank: int,
    log_prefix: str = "",
) -> None:
    """Embed texts using token-budget batching and save partitions incrementally to disk. Already
    existing partition files are skipped so the job can resume after pre-emption."""
    n_total = len(texts)
    assert len(unique_element_ids) == n_total
    n_partitions = (n_total + partition_size - 1) // partition_size

    skip_partitions: set[int] = {
        p for p in range(n_partitions) if _exists(_partition_path(output_dir, rank, p))
    }
    if skip_partitions:
        print(f"{log_prefix}Resume: skipping {len(skip_partitions)}/{n_partitions} completed partitions.", flush=True)

    print(f"{log_prefix}Estimating token counts for {n_total} texts...", flush=True)
    token_counts = [len(t) // 4 for t in texts]

    n_skipped = sum(
        _partition_bounds(p, partition_size, n_total)[1] - _partition_bounds(p, partition_size, n_total)[0]
        for p in skip_partitions
    )
    n_to_do = n_total - n_skipped
    log_every = max(100, n_to_do // 1000) if n_to_do > 0 else 100

    buffers: dict[int, dict[int, np.ndarray]] = {}
    state = _BatchState()
    flush_args = (state, texts, model, buffers, unique_element_ids, partition_size, n_total, n_to_do, output_dir, rank, log_every, log_prefix)

    for i, n_tok in enumerate(token_counts):
        if (i // partition_size) in skip_partitions:
            continue
        if n_tok > CHUNK_TOKENS:
            _flush(*flush_args)
            _record(i, embed_oversized_text(texts[i], model), buffers, unique_element_ids, partition_size, n_total, output_dir, rank, log_prefix)
            state.n_done += 1
            print(f"{log_prefix}  embedded oversized text {i} ({n_tok} tokens); total {state.n_done}/{n_to_do}", flush=True)
            state.last_logged = state.n_done
            continue

        new_max = max(state.max_tokens, n_tok)
        if state.indices and new_max * (len(state.indices) + 1) > BATCH_TOKEN_BUDGET:
            _flush(*flush_args)
        state.indices.append(i)
        state.max_tokens = max(state.max_tokens, n_tok)

    _flush(*flush_args)


def main():
    parser = argparse.ArgumentParser(description="Compute element-level Qwen3 embeddings for FinanceBench.")
    parser.add_argument("--input_dir", type=str, required=True, help="Local dir OR s3:// prefix of {doc_name}.json element files from preprocess_financebench_pdfs.py")
    parser.add_argument("--output_dir", type=str, required=True, help="Local dir OR s3:// prefix for embeddings_{rank}_{p}.npz + metadata_rank{rank}.json")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", 0)),
                        help="Index of this worker (0-based). Defaults to $SLURM_PROCID.")
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)),
                        help="Total number of parallel workers. Defaults to $SLURM_NTASKS.")
    parser.add_argument("--n_partitions", type=int, default=20,
                        help="Total number of embedding partition files to produce across all ranks.")
    args = parser.parse_args()

    if not _is_s3(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    log_prefix = f"[rank {args.rank}/{args.world_size}] "

    if args.world_size > 1 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.rank)

    print(f"{log_prefix}Loading Qwen3-Embedding-8B model...", flush=True)
    model_kwargs = {"device_map": "auto"}
    try:
        import flash_attn  # type: ignore # noqa: F401

        model_kwargs["attn_implementation"] = "flash_attention_2"
        print(f"{log_prefix}Using flash_attention_2", flush=True)
    except ImportError:
        print(f"{log_prefix}flash_attention_2 unavailable; using default attention.", flush=True)

    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-8B",
        model_kwargs=model_kwargs,
        processor_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = MAX_TOKENS
    print(f"{log_prefix}Model loaded successfully.", flush=True)

    json_files = list_json(args.input_dir)
    # Skip the preprocessing manifest(s); only per-doc element files have a doc payload.
    json_files = [p for p in json_files if not os.path.basename(p).startswith("_manifest")]
    # Each worker handles its own slice of the file list.
    json_files = json_files[args.rank :: args.world_size]
    print(f"{log_prefix}Processing {len(json_files)} element JSONs.", flush=True)

    metadata: dict[str, dict] = {}
    cleaned_elements: list[str] = []
    unique_element_ids: list[str] = []
    n_empty = 0

    for json_path in json_files:
        try:
            doc = json.loads(read_bytes(json_path))
        except Exception as e:  # noqa: BLE001 — one bad file shouldn't kill the whole job
            print(f"{log_prefix}  WARN failed to read {json_path}: {e}", flush=True)
            continue
        doc_name = doc["doc_name"]
        for elt in doc["elements"]:
            cleaned = (elt.get("content") or "").strip()
            if not cleaned:
                n_empty += 1
                continue
            page_id = int(elt["page_id"])
            element_id = int(elt["id"])
            page_key = f"{doc_name}{_PAGE_SEP}{page_id}"
            # chunk_id is per-element; doc_id (page_key) stays page-level so recall is page-level.
            unique_element_id = f"{page_key}::e{element_id}"
            unique_element_ids.append(unique_element_id)
            cleaned_elements.append(cleaned)
            metadata[unique_element_id] = {
                "doc_name": doc_name,
                "page_num": page_id,
                "page_key": page_key,
                "element_id": element_id,
                "type": elt.get("type", "text"),
                "cleaned": cleaned,
            }

    assert len(cleaned_elements) == len(metadata), "Duplicate element ids found; check input JSONs for consistency."
    metadata_path = _join(args.output_dir, f"metadata_rank{args.rank}.json")
    write_bytes(metadata_path, json.dumps(metadata).encode())
    print(f"{log_prefix}Saved metadata for {len(metadata)} elements ({n_empty} empty skipped) to {metadata_path}", flush=True)
    del metadata

    n_partitions_rank = max(1, round(args.n_partitions / args.world_size))
    n_total = len(cleaned_elements)
    partition_size = (n_total + n_partitions_rank - 1) // n_partitions_rank
    print(f"{log_prefix}Embedding {n_total} elements in {n_partitions_rank} partitions of size ~{partition_size}...", flush=True)
    embed_all(
        cleaned_elements,
        unique_element_ids,
        model,
        partition_size=partition_size,
        output_dir=args.output_dir,
        rank=args.rank,
        log_prefix=log_prefix,
    )
    print(f"{log_prefix}Embedding complete.", flush=True)


if __name__ == "__main__":
    main()
