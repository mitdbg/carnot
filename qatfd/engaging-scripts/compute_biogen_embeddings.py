"""
Compute embeddings for the TREC-BioGen 2025 document collection using Qwen3-Embedding-0.6B.

The corpus is the 2025 BioGen Pyserini collection — 26,805,982 PubMed abstracts shipped as
`jsonl_collection/pubmed25n*.jsonl`, one JSON doc per line: `{"id": <pmid>, "contents": <text>}`
(the loader also tolerates `pmid`/`docid` and `text`/`title`+`abstract`). This is the exact
corpus KARL reports, and it matches KARL's retrieval setup: Qwen3-Embedding-0.6B, k=20.

Abstracts are short (~309 tokens avg), so — unlike the OfficeQA/BrowseComp-Plus element
splitter — each document is embedded as a SINGLE element (no segmentation); the rare
over-context doc is chunked and its chunk embeddings averaged + renormalized.

Sharding: whole `.jsonl` files are assigned round-robin to ranks (file-level, so a rank never
has to read the whole 10 GB corpus). Each rank writes `embeddings_{rank}_{p}.npz` +
`metadata_rank{rank}.json`, the same layout `create_vector_db.py` consumes.

Both `--collection_dir` and `--output_dir` accept a local path OR an `s3://bucket/prefix` URI.
With S3 the corpus is streamed in and the embeddings/metadata streamed out, so the Engaging job
needs ~zero local disk (we only borrow the GPUs); you then pull the output to a machine with disk
to build the Chroma index. S3 needs `boto3` (pip install) + AWS creds in the env (AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION).

Usage (one worker per GPU, launched by srun — see run_biogen_embeddings.slurm):
    python compute_biogen_embeddings.py \
        --collection_dir s3://carnot-research/biogen/jsonl_collection \
        --output_dir     s3://carnot-research/biogen/biogen-element-embeddings \
        --n_partitions   2000
"""

import argparse
import dataclasses
import glob
import io
import json
import os
from urllib.parse import urlparse

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
# Qwen3-Embedding supports up to 32k tokens; leave a little headroom.
MAX_TOKENS = 32000
# Batch sizing. Most abstracts are short (~309 tok), and SentenceTransformer.encode's default
# batch_size of 32 badly underutilizes a 0.6B model on a 46 GB GPU. We instead pack each encode
# call up to MAX_BATCH docs, bounded by a *compute* budget count × max_len² — eager attention is
# O(seq²) memory, so this keeps peak attention memory ~constant (≈ BUDGET × n_heads × 2 B)
# regardless of sequence length: short docs batch large (fast), the rare long doc batches small
# (can't OOM). With flash-attention-2 active you can safely raise both knobs further.
# Sized for ~22 GB peak on a 46 GB L40S. MAX_BATCH=512 ran at ~44 GB and OOM-killed any rank whose
# shard hit a batch of longer abstracts (genuine allocation, not fragmentation — expandable_segments
# is already on). Halving leaves real headroom; raise again only on a bigger card or after confirming
# peak memory has margin (watch `nvidia-smi` memory.used on the running job).
MAX_BATCH = 256
BATCH_COMPUTE_BUDGET = 256 * 512 * 512  # count × max_len² ceiling (~256 docs of 512 tokens)
TARGET_ELEMENT_TOKENS = 1024


# ---------------------------------------------------------------------------
# I/O layer: every path may be a local dir OR an `s3://bucket/prefix` URI, so the
# Engaging job can stream the corpus in and the embeddings/metadata out of S3 and
# keep ~zero local disk (the cluster only lends its GPUs). boto3 reads credentials
# from the standard AWS_* env vars; it's imported lazily so local runs don't need it.
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


def list_jsonl(collection: str) -> list[str]:
    """Sorted list of every *.jsonl under a local dir or an s3:// prefix."""
    if _is_s3(collection):
        bucket, prefix = _s3_split(collection)
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        keys: list[str] = []
        for page in _s3().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            keys += [f"s3://{bucket}/{o['Key']}" for o in page.get("Contents", []) if o["Key"].endswith(".jsonl")]
        return sorted(keys)
    return sorted(glob.glob(os.path.join(collection, "**", "*.jsonl"), recursive=True))


def iter_lines(path: str):
    """Yield text lines from a local file or an s3:// object. For S3 the (~30 MB) object is
    pulled in one GET and split in memory — nothing is buffered to local disk."""
    if _is_s3(path):
        bucket, key = _s3_split(path)
        body = _s3().get_object(Bucket=bucket, Key=key)["Body"].read()
        for raw in body.splitlines():
            yield raw.decode("utf-8")
    else:
        with open(path) as f:
            yield from f


def write_bytes(dest: str, data: bytes) -> None:
    if _is_s3(dest):
        bucket, key = _s3_split(dest)
        # upload_fileobj does multipart automatically, so it handles objects over S3's 5 GB
        # single-PutObject limit — the per-rank metadata json is ~10 GB. (put_object would
        # raise EntityTooLarge.)
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
    return [
        tokenizer.decode(token_ids[s : s + chunk_tokens], skip_special_tokens=True)
        for s in range(0, len(token_ids), chunk_tokens)
    ]


def embed_oversized_text(text: str, model: SentenceTransformer) -> np.ndarray:
    """Embed a single element, chunking and averaging if it exceeds context."""
    chunks = _chunk_by_tokens(text, model.tokenizer, TARGET_ELEMENT_TOKENS)
    if len(chunks) == 1:
        return np.asarray(model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)[0], dtype=np.float32)
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
    return start, min(start + partition_size, n_total)


def _save_partition(p, buffers, unique_element_ids, partition_size, n_total, output_dir, rank, log_prefix) -> None:
    start, end = _partition_bounds(p, partition_size, n_total)
    buf = buffers[p]
    embs = np.stack([buf[k] for k in range(end - start)], axis=0)
    ids = np.array(unique_element_ids[start:end])
    path = _partition_path(output_dir, rank, p)
    write_npz(path, embs, ids)
    print(f"{log_prefix}  saved partition {p} -> {path} (shape: {embs.shape})", flush=True)
    del buffers[p]


def _record(i, emb, buffers, unique_element_ids, partition_size, n_total, output_dir, rank, log_prefix) -> None:
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


def _flush(state, texts, model, buffers, unique_element_ids, partition_size, n_total, n_to_do, output_dir, rank, log_every, log_prefix) -> None:
    if not state.indices:
        return
    embs = model.encode(
        [texts[i] for i in state.indices],
        batch_size=MAX_BATCH,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    for j, i in enumerate(state.indices):
        _record(i, np.asarray(embs[j], dtype=np.float32), buffers, unique_element_ids, partition_size, n_total, output_dir, rank, log_prefix)
    state.n_done += len(state.indices)
    if state.n_done - state.last_logged >= log_every or state.n_done == n_to_do:
        print(f"{log_prefix}  embedded {state.n_done}/{n_to_do} ({100 * state.n_done / max(n_to_do, 1):.1f}%)", flush=True)
        state.last_logged = state.n_done
    state.indices.clear()
    state.max_tokens = 0


def embed_all(texts, unique_element_ids, model, partition_size, output_dir, rank, log_prefix="") -> None:
    """Embed texts with token-budget batching, saving partitions incrementally. Existing
    partition files are skipped so the job resumes after pre-emption."""
    n_total = len(texts)
    assert len(unique_element_ids) == n_total
    n_partitions = (n_total + partition_size - 1) // partition_size
    skip = {p for p in range(n_partitions) if _exists(_partition_path(output_dir, rank, p))}
    if skip:
        print(f"{log_prefix}Resume: skipping {len(skip)}/{n_partitions} completed partitions.", flush=True)

    token_counts = [len(t) // 4 for t in texts]
    n_skipped = sum(_partition_bounds(p, partition_size, n_total)[1] - _partition_bounds(p, partition_size, n_total)[0] for p in skip)
    n_to_do = n_total - n_skipped
    log_every = max(1000, n_to_do // 1000) if n_to_do > 0 else 1000

    buffers: dict[int, dict[int, np.ndarray]] = {}
    state = _BatchState()
    args = (state, texts, model, buffers, unique_element_ids, partition_size, n_total, n_to_do, output_dir, rank, log_every, log_prefix)

    for i, n_tok in enumerate(token_counts):
        if (i // partition_size) in skip:
            continue
        if n_tok > MAX_TOKENS:
            _flush(*args)
            _record(i, embed_oversized_text(texts[i], model), buffers, unique_element_ids, partition_size, n_total, output_dir, rank, log_prefix)
            state.n_done += 1
            state.last_logged = state.n_done
            continue
        new_max = max(state.max_tokens, n_tok)
        n_next = len(state.indices) + 1
        if state.indices and (n_next > MAX_BATCH or new_max * new_max * n_next > BATCH_COMPUTE_BUDGET):
            _flush(*args)
        state.indices.append(i)
        state.max_tokens = max(state.max_tokens, n_tok)
    _flush(*args)


def _doc_text(rec: dict) -> str:
    """Extract the abstract text from a Pyserini JSONL record, tolerant of field names."""
    if rec.get("contents"):
        return str(rec["contents"])
    if rec.get("text"):
        return str(rec["text"])
    title = str(rec.get("title", "") or "")
    abstract = str(rec.get("abstract", "") or "")
    return (title + "\n\n" + abstract).strip()


def _doc_id(rec: dict) -> str:
    for k in ("id", "pmid", "docid", "_id"):
        if rec.get(k) is not None:
            return str(rec[k])
    raise KeyError(f"no id field in record (keys={list(rec)})")


def load_shard(collection: str, rank: int, world_size: int, log_prefix: str) -> tuple[list[str], list[str]]:
    """Read this rank's round-robin slice of `*.jsonl` files (local dir or s3:// prefix).
    Streams each file line-by-line, so nothing lands on local disk. Returns (pmids, texts)."""
    files = list_jsonl(collection)
    if not files:
        raise FileNotFoundError(f"no *.jsonl files under {collection}")
    my_files = files[rank::world_size]
    print(f"{log_prefix}{len(my_files)}/{len(files)} jsonl files assigned.", flush=True)
    pmids: list[str] = []
    texts: list[str] = []
    for fi, path in enumerate(my_files):
        for line in iter_lines(path):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pmids.append(_doc_id(rec))
            texts.append(_doc_text(rec))
        if fi % 20 == 0:
            print(f"{log_prefix}  read {fi + 1}/{len(my_files)} files, {len(pmids)} docs...", flush=True)
    return pmids, texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Qwen3-0.6B embeddings for TREC-BioGen abstracts.")
    parser.add_argument("--collection_dir", required=True,
                        help="local dir OR s3:// prefix holding the Pyserini *.jsonl files")
    parser.add_argument("--output_dir", required=True,
                        help="local dir OR s3:// prefix for embeddings_{rank}_{p}.npz + metadata_rank{rank}.json")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)))
    parser.add_argument("--n_partitions", type=int, default=2000,
                        help="total partition files across all ranks (each rank writes ~n/world_size).")
    args = parser.parse_args()

    if not _is_s3(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    log_prefix = f"[rank {args.rank}/{args.world_size}] "

    if args.world_size > 1 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.rank)

    # Fail fast on a node where CUDA can't initialize (e.g. Fabric Manager down -> CUDA error 802):
    # otherwise SentenceTransformer silently falls back to CPU and 6.7M abstracts never finish.
    # Checked BEFORE the multi-minute corpus read so a bad node aborts in seconds (resume elsewhere).
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{log_prefix}CUDA is not available on this node — refusing to embed on CPU. This is "
            f"usually a broken GPU node (CUDA error 802 / Fabric Manager down). Resubmit, excluding "
            f"this node: `sbatch --exclude=$SLURMD_NODENAME run_biogen_embeddings.slurm`."
        )

    pmids, texts = load_shard(args.collection_dir, args.rank, args.world_size, log_prefix)

    # there are some duplicate pmids, so we default to keeping the latest as it most likely
    # means that the article was revised in some fashion
    by_pmid: dict[str, str] = {}
    for pmid, text in zip(pmids, texts, strict=True):
        by_pmid[pmid] = text
    n_dup = len(texts) - len(by_pmid)
    if n_dup:
        print(f"{log_prefix}collapsed {n_dup} duplicate PMIDs (kept latest revision).", flush=True)

    # metadata: one element per document (element_id 0). `cleaned` is the embedded text, so the
    # benchmark's document_map (and chroma `documents`) reconstruct exactly what was embedded.
    unique_element_ids = [f"{pmid}_0" for pmid in by_pmid]
    texts = list(by_pmid.values())
    metadata = {f"{pmid}_0": {"docid": pmid, "cleaned": text, "element_id": 0} for pmid, text in by_pmid.items()}
    metadata_path = _join(args.output_dir, f"metadata_rank{args.rank}.json")
    write_bytes(metadata_path, json.dumps(metadata).encode("utf-8"))
    print(f"{log_prefix}Saved metadata for {len(metadata)} docs to {metadata_path}", flush=True)
    del metadata

    print(f"{log_prefix}Loading {EMBED_MODEL}...", flush=True)
    model_kwargs = {"device_map": "auto"}
    try:
        import flash_attn  # type: ignore # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print(f"{log_prefix}Using flash_attention_2", flush=True)
    except ImportError:
        # No flash-attn -> use PyTorch SDPA (memory-efficient/flash backend, built into torch, no
        # install). This is REQUIRED for the batched encode: eager attention materializes the full
        # [batch, heads, seq, seq] score matrix in fp32 and OOMs a 44 GB L40S on large batches /
        # long abstracts; SDPA never materializes it (O(seq) memory), so big batches are safe + fast.
        model_kwargs["attn_implementation"] = "sdpa"
        print(f"{log_prefix}flash-attn not installed, using PyTorch SDPA attention", flush=True)
    model = SentenceTransformer(EMBED_MODEL, model_kwargs=model_kwargs, processor_kwargs={"padding_side": "left"})
    model.max_seq_length = MAX_TOKENS
    print(f"{log_prefix}Model loaded (dim={model.get_sentence_embedding_dimension()}).", flush=True)

    n_partitions_rank = max(1, round(args.n_partitions / args.world_size))
    n_total = len(texts)
    partition_size = (n_total + n_partitions_rank - 1) // n_partitions_rank
    print(f"{log_prefix}Embedding {n_total} abstracts in {n_partitions_rank} partitions of ~{partition_size}...", flush=True)
    embed_all(texts, unique_element_ids, model, partition_size, args.output_dir, args.rank, log_prefix)
    print(f"{log_prefix}Done.", flush=True)


if __name__ == "__main__":
    main()
