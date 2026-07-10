"""Compute embeddings for the FULL QAMPARI Wikipedia corpus with Qwen3-Embedding-0.6B.

Input is QAMPARI's released chunked Wikipedia DIRECTLY — `wikipedia_chunks/chunks_v5/
wikipedia_chunks_*.jsonl`, ~25.9M ~100-word passages, one JSON chunk per line:
    {"id": "<page_id>__<n>", "contents": "<title> <body>",
     "meta": {"revid","url","title","file_path","page_id","content","chunk_id"}}
We embed EVERY chunk (no filtering): KARL's "chunks containing a gold answer entity" subset can't be
reliably reproduced (the answer entities are common strings that match most of Wikipedia, and the
entity-LINK annotations that would scope it aren't in the chunk metadata), so — like TREC-BioGen's
26.8M abstracts — we index the whole corpus and let retrieval do the work. Each chunk is already
~100 words, so it is embedded as a SINGLE element (no segmentation); the rare over-context chunk is
split and its chunk embeddings averaged + renormalized. We embed `contents` (title + body), matching
QAMPARI's BM25/DPR index field.

The chunk's own id ("<page_id>__<n>") is the unique_element_id (= the ChromaDB row id / chunk_id).
Per-element metadata carries `cleaned` (the embedded text), `title`, `page_id`, and `url`.
create_vector_db's `qampari` adapter stores the Wikipedia ARTICLE (`page_id`) as the Chroma `doc_id`
(the retrieval unit) with the chunk's `__n` ordinal as `element_id`, and surfaces `title` — the
article identity the benchmark's gold uses for doc-recall (see CORPUS_MODEL.md).

Sharding (one worker per GPU, launched by srun): whole `.jsonl` files are assigned round-robin to
ranks; each rank writes `embeddings_{rank}_{p}.npz` + `metadata_rank{rank}.json`.

Both `--collection_dir` and `--output_dir` accept a local path OR an `s3://bucket/prefix` URI. With
S3 the corpus is streamed in and the embeddings/metadata streamed out (boto3 + AWS_* creds), so the
GPU box needs ~zero local disk; pull the output to a machine with disk to build the Chroma index.

Usage (see run_qampari_embeddings.slurm):
    python compute_qampari_embeddings.py \
        --collection_dir <skunk>/qampari/wikipedia_chunks/chunks_v5 \
        --output_dir     <skunk>/qampari/qampari-element-embeddings \
        --n_partitions   2000
"""

import argparse
import glob
import io
import json
import os
from urllib.parse import urlparse

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
# Hard cap on the sequence length fed to the model AND the threshold above which a chunk is split +
# averaged (embed_oversized_text). MUST stay modest: with PyTorch SDPA (no flash-attn) the attention
# scores are materialized as heads×seq², so a lone multi-thousand-token sequence can OOM an 80GB A100
# (job 16684620 died here at 75GB on a 32k-token chunk). QAMPARI chunks are ~100 words (~150 tokens);
# 2048 covers virtually all of them while a lone sequence at the cap needs only ~130MB of scores, and
# the rare longer chunk is split + averaged. Override with EMBED_MAX_TOKENS if needed.
MAX_TOKENS = int(os.environ.get("EMBED_MAX_TOKENS", "2048"))
MAX_BATCH = int(os.environ.get("EMBED_MAX_BATCH", "256"))
# count × max_len² ceiling for a batch (~256 chunks of 512 tokens); bounds per-batch attention memory.
BATCH_COMPUTE_BUDGET = int(os.environ.get("EMBED_BATCH_BUDGET", str(256 * 512 * 512)))
TARGET_ELEMENT_TOKENS = 1024


# ---------------------------------------------------------------------------
# I/O layer: local path OR s3:// URI (mirrors compute_biogen_embeddings.py).
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
        _s3().upload_fileobj(io.BytesIO(data), bucket, key)
    else:
        with open(dest, "wb") as f:
            f.write(data)


def write_npz(dest: str, embeddings: np.ndarray, ids: np.ndarray) -> None:
    buf = io.BytesIO()
    np.savez_compressed(buf, embeddings=embeddings, unique_element_ids=ids)
    write_bytes(dest, buf.getvalue())


# ---------------------------------------------------------------------------
# Embedding (token-budget batching; identical to compute_biogen_embeddings.py).
# ---------------------------------------------------------------------------


def _chunk_by_tokens(text: str, tokenizer, chunk_tokens: int) -> list[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= chunk_tokens:
        return [text]
    return [
        tokenizer.decode(token_ids[s : s + chunk_tokens], skip_special_tokens=True)
        for s in range(0, len(token_ids), chunk_tokens)
    ]


def embed_oversized_text(text: str, model: SentenceTransformer) -> np.ndarray:
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


class _BatchState:
    def __init__(self) -> None:
        self.indices: list[int] = []
        self.max_tokens = 0
        self.n_done = 0
        self.last_logged = 0


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


# ---------------------------------------------------------------------------
# QAMPARI corpus loading.
# ---------------------------------------------------------------------------


def _chunk_id(rec: dict) -> str:
    """The chunk's stable id ("<page_id>__<n>"): top-level `id`, else meta.chunk_id."""
    if rec.get("id"):
        return str(rec["id"])
    meta = rec.get("meta") or {}
    if meta.get("chunk_id"):
        return str(meta["chunk_id"])
    raise KeyError(f"no chunk id in record (keys={list(rec)})")


def _chunk_text(rec: dict) -> str:
    """The embedded text: `contents` (title + body), else title + meta.content."""
    if rec.get("contents"):
        return str(rec["contents"])
    meta = rec.get("meta") or {}
    return f"{meta.get('title', '')} {meta.get('content', '')}".strip()


def _element_id(chunk_id: str) -> int:
    """Element (chunk) index within its page — the trailing field of the chunk id
    "{page_id}__{elt_id}" (page_id is a numeric Wikipedia id, so it never contains "__")."""
    tail = chunk_id.rsplit("__", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def load_shard(collection: str, rank: int, world_size: int, log_prefix: str) -> tuple[list[str], list[str], dict[str, dict]]:
    """Read this rank's round-robin slice of `*.jsonl` files. Returns (chunk_ids, texts, meta_by_id),
    where meta_by_id maps chunk_id -> per-element metadata for create_vector_db's qampari adapter."""
    files = list_jsonl(collection)
    if not files:
        raise FileNotFoundError(f"no *.jsonl files under {collection}")
    my_files = files[rank::world_size]
    print(f"{log_prefix}{len(my_files)}/{len(files)} jsonl files assigned.", flush=True)
    chunk_ids: list[str] = []
    texts: list[str] = []
    meta_by_id: dict[str, dict] = {}
    for fi, path in enumerate(my_files):
        for line in iter_lines(path):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = _chunk_id(rec)
            text = _chunk_text(rec)
            meta = rec.get("meta") or {}
            chunk_ids.append(cid)
            texts.append(text)
            meta_by_id[cid] = {
                "chunk_id": cid,
                "cleaned": text,
                "title": str(meta.get("title", "")),
                "page_id": str(meta.get("page_id", "")),
                "url": str(meta.get("url", "")),
                "element_id": _element_id(cid),
            }
        if fi % 20 == 0:
            print(f"{log_prefix}  read {fi + 1}/{len(my_files)} files, {len(chunk_ids)} chunks...", flush=True)
    return chunk_ids, texts, meta_by_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Qwen3-0.6B embeddings for the QAMPARI corpus.")
    parser.add_argument("--collection_dir", required=True,
                        help="local dir OR s3:// prefix of QAMPARI's wikipedia_chunks_*.jsonl (chunks_v5)")
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

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{log_prefix}CUDA is not available on this node — refusing to embed on CPU. This is "
            f"usually a broken GPU node (CUDA error 802 / Fabric Manager down). Resubmit, excluding "
            f"this node: `sbatch --exclude=$SLURMD_NODENAME run_qampari_embeddings.slurm`."
        )

    chunk_ids, texts, meta_by_id = load_shard(args.collection_dir, args.rank, args.world_size, log_prefix)

    # chunk ids are globally unique already; collapse any accidental dup (keep last).
    if len(meta_by_id) != len(chunk_ids):
        seen: dict[str, str] = {}
        for cid, text in zip(chunk_ids, texts, strict=True):
            seen[cid] = text
        n_dup = len(chunk_ids) - len(seen)
        print(f"{log_prefix}collapsed {n_dup} duplicate chunk_ids.", flush=True)
        chunk_ids = list(seen.keys())
        texts = list(seen.values())

    unique_element_ids = list(chunk_ids)
    metadata_path = _join(args.output_dir, f"metadata_rank{args.rank}.json")
    write_bytes(metadata_path, json.dumps(meta_by_id).encode("utf-8"))
    print(f"{log_prefix}Saved metadata for {len(meta_by_id)} chunks to {metadata_path}", flush=True)
    del meta_by_id

    print(f"{log_prefix}Loading {EMBED_MODEL}...", flush=True)
    model_kwargs = {"device_map": "auto"}
    try:
        import flash_attn  # type: ignore # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print(f"{log_prefix}Using flash_attention_2", flush=True)
    except ImportError:
        model_kwargs["attn_implementation"] = "sdpa"
        print(f"{log_prefix}flash-attn not installed, using PyTorch SDPA attention", flush=True)
    model = SentenceTransformer(EMBED_MODEL, model_kwargs=model_kwargs, processor_kwargs={"padding_side": "left"})
    model.max_seq_length = MAX_TOKENS
    print(f"{log_prefix}Model loaded (dim={model.get_sentence_embedding_dimension()}).", flush=True)

    n_partitions_rank = max(1, round(args.n_partitions / args.world_size))
    n_total = len(texts)
    partition_size = (n_total + n_partitions_rank - 1) // n_partitions_rank
    print(f"{log_prefix}Embedding {n_total} chunks in {n_partitions_rank} partitions of ~{partition_size}...", flush=True)
    embed_all(texts, unique_element_ids, model, partition_size, args.output_dir, args.rank, log_prefix)
    print(f"{log_prefix}Done.", flush=True)


if __name__ == "__main__":
    main()
