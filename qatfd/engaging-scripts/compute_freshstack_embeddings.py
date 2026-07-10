"""Compute embeddings for one FreshStack topic's corpus with Qwen3-Embedding-0.6B.

Input is a FreshStack topic's `corpus.jsonl` (one JSON record per line, as written by
download_freshstack.py):
    {"_id": "azure-openai/LICENSE.md_0_1140", "text": "<chunk text>",
     "meta"/"metadata": {"url", "start_byte", "end_byte"}}
Each corpus record is ALREADY A CHUNK — the `_id` is "{repo}/{path}_{start_byte}_{end_byte}", a
byte-range slice of a source file (a file split into several chunks appears as several `_id`s). That
chunk is the unit of BOTH retrieval and relevance: FreshStack's gold (a nugget's relevant_corpus_ids)
references these same chunk `_id`s. So we embed ONE vector per corpus `_id` — i.e. at exactly the
corpus/gold granularity, no re-chunking. We embed the `text` field, and the `_id` is the
unique_element_id (= the ChromaDB row id). KARL retrieves FreshStack with Qwen3-0.6B, so we match that
model (1024-dim). The rare chunk longer than the model context is split and its piece-embeddings
averaged + renormalized (embed_oversized_text) — still ONE vector for that `_id` (a model-context
workaround, not a change in retrieval unit).

Per-element metadata carries `doc_id` (= `_id`, the chunk), `file_id` (the source file the chunk
belongs to, = the `_id` minus its "_{start}_{end}" byte-range suffix), `cleaned` (the embedded
text), and `url`. create_vector_db's `freshstack` adapter builds the collection with the source
FILE (`file_id`) as the Chroma `doc_id` (the retrieval unit) and the corpus `_id` as the row id /
chunk_id; the benchmark collapses its chunk-level gold to file_ids for recall (see CORPUS_MODEL.md).

Sharding (one worker per GPU, launched by srun): the single corpus.jsonl has no files to split across
ranks, so records are sharded round-robin BY LINE INDEX (`records[rank::world_size]`); each rank writes
`embeddings_{rank}_{p}.npz` + `metadata_rank{rank}.json`. Keep --ntasks fixed across resubmits (the
record-stripe + the resume key depend on it).

Both `--corpus_file` and `--output_dir` accept a local path OR an `s3://bucket/key` URI.

Usage (see run_freshstack_embeddings.slurm):
    srun python compute_freshstack_embeddings.py \
        --corpus_file <skunk>/freshstack/langchain/corpus.jsonl \
        --output_dir  <skunk>/freshstack/langchain/embeddings \
        --n_partitions 50
"""

import argparse
import io
import json
import os
from urllib.parse import urlparse

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
MAX_TOKENS = 32000
MAX_BATCH = 256
BATCH_COMPUTE_BUDGET = 256 * 512 * 512  # count × max_len² ceiling (~256 chunks of 512 tokens)
TARGET_ELEMENT_TOKENS = 1024


# ---------------------------------------------------------------------------
# I/O layer: local path OR s3:// URI (mirrors compute_qampari_embeddings.py).
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
# Embedding (token-budget batching; identical to compute_qampari_embeddings.py).
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
# FreshStack corpus loading.
# ---------------------------------------------------------------------------


def _file_id(doc_id: str) -> str:
    """The source FILE a chunk belongs to: the chunk `_id` minus its trailing "_{start}_{end}" byte
    range, e.g. "azure-openai/LICENSE.md_0_1140" -> "azure-openai/LICENSE.md". (File paths can contain
    underscores, so we only strip when the last two underscore-separated fields are both integers.)
    Local copy of qatfd.keys.freshstack_file_id (this script runs standalone on the cluster, so it
    cannot import qatfd) — keep the two identical."""
    parts = doc_id.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0]
    return doc_id


def load_shard(corpus_file: str, rank: int, world_size: int, log_prefix: str) -> tuple[list[str], list[str], dict[str, dict]]:
    """Read this rank's round-robin stripe of corpus records (`records[rank::world_size]`). The single
    corpus.jsonl has no files to split across ranks, so we shard BY LINE INDEX. Returns (doc_ids,
    texts, meta_by_id), where meta_by_id maps doc_id -> per-element metadata for the freshstack adapter."""
    doc_ids: list[str] = []
    texts: list[str] = []
    meta_by_id: dict[str, dict] = {}
    for i, line in enumerate(iter_lines(corpus_file)):
        if i % world_size != rank:
            continue
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        did = str(rec["_id"])
        text = str(rec.get("text", ""))
        meta = rec.get("metadata") or rec.get("meta") or {}
        doc_ids.append(did)
        texts.append(text)
        meta_by_id[did] = {
            "doc_id": did,
            "file_id": _file_id(did),
            "cleaned": text,
            "url": str(meta.get("url", "")),
            "element_id": 0,
        }
    print(f"{log_prefix}read {len(doc_ids)} docs (stripe {rank}/{world_size}) from {corpus_file}", flush=True)
    return doc_ids, texts, meta_by_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Qwen3-0.6B embeddings for a FreshStack topic corpus.")
    parser.add_argument("--corpus_file", required=True,
                        help="local path OR s3:// URI of the topic's corpus.jsonl ({_id, text, metadata})")
    parser.add_argument("--output_dir", required=True,
                        help="local dir OR s3:// prefix for embeddings_{rank}_{p}.npz + metadata_rank{rank}.json")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)))
    parser.add_argument("--n_partitions", type=int, default=50,
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
            f"this node: `sbatch --exclude=$SLURMD_NODENAME run_freshstack_embeddings.slurm <topic>`."
        )

    doc_ids, texts, meta_by_id = load_shard(args.corpus_file, args.rank, args.world_size, log_prefix)

    # corpus ids are unique already; collapse any accidental dup (keep last).
    if len(meta_by_id) != len(doc_ids):
        seen: dict[str, str] = {}
        for did, text in zip(doc_ids, texts, strict=True):
            seen[did] = text
        n_dup = len(doc_ids) - len(seen)
        print(f"{log_prefix}collapsed {n_dup} duplicate doc_ids.", flush=True)
        doc_ids = list(seen.keys())
        texts = list(seen.values())

    unique_element_ids = list(doc_ids)
    metadata_path = _join(args.output_dir, f"metadata_rank{args.rank}.json")
    write_bytes(metadata_path, json.dumps(meta_by_id).encode("utf-8"))
    print(f"{log_prefix}Saved metadata for {len(meta_by_id)} docs to {metadata_path}", flush=True)
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
    print(f"{log_prefix}Embedding {n_total} docs in {n_partitions_rank} partitions of ~{partition_size}...", flush=True)
    embed_all(texts, unique_element_ids, model, partition_size, args.output_dir, args.rank, log_prefix)
    print(f"{log_prefix}Done.", flush=True)


if __name__ == "__main__":
    main()
