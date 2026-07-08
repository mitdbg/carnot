"""Compute Qwen3-Embedding-8B embeddings for benchmark QA pairs.

For each QA pair in a benchmark, two embeddings are produced:

  {benchmark}_{idx}:q  — the question text
  {benchmark}_{idx}:a  — the answer text

For OfficeQA Pro the answer is a plain string from the CSV.
For BrowseComp-Plus the answer is the plain string from the JSONL.

The full benchmark dataset is embedded (not just the validation split), so
that the de-duplication pipeline can check synthetic questions against ALL
existing benchmark questions.

Output layout (identical to compute_browsecomp_plus_element_embeddings.py):

  <output_dir>/embeddings_{rank}_{partition}.npz
    embeddings:          float32  [n, d]
    unique_element_ids:  str      [n]   (e.g. "officeqa_0:q")

  <output_dir>/metadata_rank{rank}.json
    { unique_element_id: { kind, qa_id, is_synthetic, benchmark, text } }

The metadata schema matches the constants in skunk/datagen/dedup.py so the
output can be loaded directly by create_qa_vector_db.py.

Usage (single process):
    python compute_qa_embeddings.py \\
        --benchmark officeqa \\
        --output_dir ./officeqa-qa-embeddings

Usage (multi-GPU via SLURM, e.g. 2 tasks):
    srun python compute_qa_embeddings.py \\
        --benchmark browsecomp-plus \\
        --output_dir ./browsecomp-qa-embeddings
"""

import argparse
import dataclasses
import json
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Qwen3-Embedding-8B supports up to 32k tokens.
MAX_TOKENS = 32000

# Token budget per batch: batch_size × max_seq_len_in_batch must not exceed this.
BATCH_TOKEN_BUDGET = 16384
CHARS_PER_TOKEN_EST = 4

# Default source paths (relative to the skunk/ working directory).
OFFICEQA_QUESTIONS_PATH = "officeqa_pro.csv"
BROWSECOMP_PLUS_QUESTIONS_PATH = "browsecomp-plus/browsecomp_plus_decrypted.jsonl"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_qa_pairs(
    benchmark: str,
    officeqa_path: str,
    browsecomp_path: str,
) -> list[tuple[str, str, str]]:
    """Return (qa_id, question, answer) triples for every pair in the benchmark.

    The entire dataset is loaded (not just the validation split) so that the
    de-duplication collection covers all existing questions.
    """
    if benchmark == "officeqa":
        df = pd.read_csv(officeqa_path)
        return [
            (row["uid"], str(row["question"]), str(row["answer"]))
            for i, row in df.iterrows()
        ]
    elif benchmark == "browsecomp-plus":
        with open(browsecomp_path) as f:
            items = [json.loads(line) for line in f]
        return [
            (str(item["query_id"]), str(item["query"]), str(item["answer"]))
            for i, item in enumerate(items)
        ]
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark!r}")


# ---------------------------------------------------------------------------
# Partition helpers (mirrors compute_browsecomp_plus_element_embeddings.py)
# ---------------------------------------------------------------------------

def _partition_path(output_dir: str, rank: int, p: int) -> str:
    return os.path.join(output_dir, f"embeddings_{rank}_{p}.npz")


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
    np.savez_compressed(path, embeddings=embs, unique_element_ids=ids)
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
        _save_partition(
            p, buffers, unique_element_ids, partition_size, n_total, output_dir, rank, log_prefix
        )


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
    embs = model.encode(
        [texts[i] for i in state.indices],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    for j, i in enumerate(state.indices):
        _record(
            i, np.asarray(embs[j], dtype=np.float32),
            buffers, unique_element_ids, partition_size, n_total, output_dir, rank, log_prefix,
        )
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
    """Embed all texts using token-budget batching; save partitions incrementally.

    Already-existing partition files are skipped so the job can resume after
    preemption.
    """
    n_total = len(texts)
    assert len(unique_element_ids) == n_total
    n_partitions = (n_total + partition_size - 1) // partition_size

    skip_partitions: set[int] = {
        p for p in range(n_partitions)
        if os.path.exists(_partition_path(output_dir, rank, p))
    }
    if skip_partitions:
        print(
            f"{log_prefix}Resume: skipping {len(skip_partitions)}/{n_partitions} "
            f"already-completed partitions.",
            flush=True,
        )

    token_counts = [len(t) // CHARS_PER_TOKEN_EST for t in texts]
    n_skipped = sum(
        _partition_bounds(p, partition_size, n_total)[1] - _partition_bounds(p, partition_size, n_total)[0]
        for p in skip_partitions
    )
    n_to_do = n_total - n_skipped
    if n_to_do == 0:
        print(f"{log_prefix}Nothing to do; all partitions already exist.", flush=True)
        return
    log_every = max(10, n_to_do // 100)

    buffers: dict[int, dict[int, np.ndarray]] = {}
    state = _BatchState()
    flush_args = (
        state, texts, model, buffers, unique_element_ids,
        partition_size, n_total, n_to_do, output_dir, rank, log_every, log_prefix,
    )

    for i, n_tok in enumerate(token_counts):
        if (i // partition_size) in skip_partitions:
            continue
        new_max = max(state.max_tokens, n_tok)
        if state.indices and new_max * (len(state.indices) + 1) > BATCH_TOKEN_BUDGET:
            _flush(*flush_args)
        state.indices.append(i)
        state.max_tokens = max(state.max_tokens, n_tok)

    _flush(*flush_args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Qwen3-Embedding-8B embeddings for benchmark QA pairs."
    )
    parser.add_argument(
        "--benchmark", type=str, required=True,
        choices=["officeqa", "browsecomp-plus"],
        help="Which benchmark's QA pairs to embed.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save embedding outputs.",
    )
    parser.add_argument(
        "--officeqa_path", type=str, default=OFFICEQA_QUESTIONS_PATH,
        help=f"Path to the OfficeQA Pro CSV (default: {OFFICEQA_QUESTIONS_PATH})",
    )
    parser.add_argument(
        "--browsecomp_path", type=str, default=BROWSECOMP_PLUS_QUESTIONS_PATH,
        help=f"Path to the BrowseComp-Plus JSONL (default: {BROWSECOMP_PLUS_QUESTIONS_PATH})",
    )
    parser.add_argument(
        "--rank", type=int,
        default=int(os.environ.get("SLURM_PROCID", 0)),
        help="Index of this worker (0-based). Defaults to $SLURM_PROCID.",
    )
    parser.add_argument(
        "--world_size", type=int,
        default=int(os.environ.get("SLURM_NTASKS", 1)),
        help="Total number of parallel workers. Defaults to $SLURM_NTASKS.",
    )
    parser.add_argument(
        "--n_partitions", type=int, default=10,
        help="Total number of partition files to produce across all ranks (default: 10).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_prefix = f"[rank {args.rank}/{args.world_size}] "

    # Load all pairs, then shard by rank.
    all_pairs = load_qa_pairs(args.benchmark, args.officeqa_path, args.browsecomp_path)
    rank_pairs = all_pairs[args.rank::args.world_size]
    print(
        f"{log_prefix}Processing {len(rank_pairs)}/{len(all_pairs)} QA pairs "
        f"({args.benchmark}).",
        flush=True,
    )

    # Interleave Q and A so each pair's embeddings are computed together.
    # Layout: [q_0, a_0, q_1, a_1, ...]
    texts: list[str] = []
    unique_element_ids: list[str] = []
    metadata: dict[str, dict] = {}

    for qa_id, question, answer in rank_pairs:
        q_eid = f"{qa_id}:q"
        a_eid = f"{qa_id}:a"
        unique_element_ids.extend([q_eid, a_eid])
        texts.extend([question, answer])
        metadata[q_eid] = {
            "qa_id": qa_id,
            "kind": "question",
            "is_synthetic": False,
            "benchmark": args.benchmark,
            "text": question,
        }
        metadata[a_eid] = {
            "qa_id": qa_id,
            "kind": "answer",
            "is_synthetic": False,
            "benchmark": args.benchmark,
            "text": answer,
        }

    # Persist metadata for this rank before GPU work starts so it survives
    # preemption even if embedding is interrupted.
    metadata_path = os.path.join(args.output_dir, f"metadata_rank{args.rank}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    print(
        f"{log_prefix}Saved metadata for {len(metadata)} elements to {metadata_path}",
        flush=True,
    )
    del metadata

    # Pin each SLURM task to its own GPU. Honour a pre-set CUDA_VISIBLE_DEVICES
    # (e.g. for single-GPU recovery runs) and only assign when running with
    # multiple workers.
    if args.world_size > 1 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.rank)

    print(f"{log_prefix}Loading Qwen3-Embedding-8B model...", flush=True)
    model_kwargs: dict = {"device_map": "auto"}
    try:
        import flash_attn  # type: ignore  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print(f"{log_prefix}Using flash_attention_2", flush=True)
    except ImportError:
        print(f"{log_prefix}flash-attn not installed, using default attention", flush=True)

    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-8B",
        model_kwargs=model_kwargs,
        processor_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = MAX_TOKENS
    print(f"{log_prefix}Model loaded.", flush=True)

    n_total = len(texts)
    n_partitions_rank = max(1, round(args.n_partitions / args.world_size))
    partition_size = max(1, (n_total + n_partitions_rank - 1) // n_partitions_rank)
    print(
        f"{log_prefix}Embedding {n_total} elements in {n_partitions_rank} "
        f"partitions of size ~{partition_size}...",
        flush=True,
    )

    embed_all(
        texts,
        unique_element_ids,
        model,
        partition_size=partition_size,
        output_dir=args.output_dir,
        rank=args.rank,
        log_prefix=log_prefix,
    )
    print(f"{log_prefix}Done.", flush=True)


if __name__ == "__main__":
    main()
