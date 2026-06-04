"""
Compute embeddings for BrowseComp-Plus using Qwen3-Embedding-8B.

For each docid in the BrowseComp-Plus corpus
  * In-memory preprocessing splits each document on `\n\n` to produce initial element texts
  * The initial elements are then combined consecutively until their total token count exceeds 512 tokens
  * If any resulting element exceeds the model's ~32k-token context limit,
    it is split into non-overlapping ~512-token chunks; the per-chunk
    embeddings are averaged and re-normalized to produce a single embedding
    for the element.

Usage:
    python compute_element_embeddings.py \\
        --output_dir /home/mdrusso/carnot/skunk/browsecomp-plus-embedding-gen/browsecomp-plus-element-embeddings
"""

import argparse
import dataclasses
import json
import os

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Qwen3-Embedding-8B supports up to 32k tokens; leave a little headroom.
MAX_TOKENS = 32000
# Token budget per batch: batch_size × max_seq_len_in_batch must not exceed this.
# With flash_attention_2 memory scales linearly with this value; 16k is conservative.
BATCH_TOKEN_BUDGET = 16384
CHARS_PER_TOKEN_EST = 4
TARGET_ELEMENT_TOKENS = 1024
MAX_ELEMENT_TOKENS = 4096


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
    chunks = _chunk_by_tokens(text, tokenizer, TARGET_ELEMENT_TOKENS)
    if len(chunks) == 1:
        emb = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(emb, dtype=np.float32)

    chunk_embs = model.encode(
        chunks, normalize_embeddings=True, show_progress_bar=False, batch_size=1
    )
    avg = np.asarray(chunk_embs, dtype=np.float32).mean(axis=0)
    norm = float(np.linalg.norm(avg))
    if norm > 0:
        avg = avg / norm
    return avg.astype(np.float32)


def _partition_path(output_dir: str, rank: int, p: int) -> str:
    """Return the output file path for partition p of the given rank."""
    return os.path.join(output_dir, f"embeddings_{rank}_{p}.npz")


def _partition_bounds(p: int, partition_size: int, n_total: int) -> tuple[int, int]:
    """Return the (start, end) element indices for partition p."""
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
    """Stack a completed partition's embeddings and write them to a .npz file."""
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
    """Store an embedding in its partition buffer, saving the partition when complete."""
    p = i // partition_size
    buf = buffers.setdefault(p, {})
    buf[i - p * partition_size] = emb
    start, end = _partition_bounds(p, partition_size, n_total)
    if len(buf) == end - start:
        _save_partition(p, buffers, unique_element_ids, partition_size, n_total, output_dir, rank, log_prefix)


@dataclasses.dataclass
class _BatchState:
    """Mutable accumulator for the current in-flight batch."""
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
    """Encode the pending batch, record each embedding, and reset the batch state."""
    if not state.indices:
        return
    embs = model.encode(
        [texts[i] for i in state.indices],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
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
    """Embed texts using token-budget batching and save partitions incrementally to disk.

    Each partition contains up to ``partition_size`` consecutive elements and
    is saved as ``embeddings_{rank}_{partition_id}.npz`` with arrays
    ``embeddings`` (float32, shape [n, d]) and ``unique_element_ids`` (str). Already
    existing partition files are skipped so the job can resume after being
    pre-empted or killed.
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

    print(f"{log_prefix}Estimating token counts for {n_total} texts...", flush=True)
    token_counts = [len(t) // 4 for t in texts]
    print(f"{log_prefix}Token count estimation complete.", flush=True)

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
        # Oversized texts bypass batching entirely.
        if n_tok > MAX_TOKENS:
            _flush(*flush_args)
            _record(i, embed_oversized_text(texts[i], model), buffers, unique_element_ids, partition_size, n_total, output_dir, rank, log_prefix)
            state.n_done += 1
            print(
                f"{log_prefix}  embedded oversized text {i} "
                f"({n_tok} tokens); total {state.n_done}/{n_to_do}",
                flush=True,
            )
            state.last_logged = state.n_done
            continue

        new_max = max(state.max_tokens, n_tok)
        if state.indices and new_max * (len(state.indices) + 1) > BATCH_TOKEN_BUDGET:
            _flush(*flush_args)

        state.indices.append(i)
        state.max_tokens = max(state.max_tokens, n_tok)

    _flush(*flush_args)


def main():
    parser = argparse.ArgumentParser(
        description="Compute Qwen3 embeddings for each Treasury Bulletin page."
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save embedding outputs")
    parser.add_argument("--rank", type=int,
                        default=int(os.environ.get("SLURM_PROCID", 0)),
                        help="Index of this worker (0-based). Defaults to $SLURM_PROCID.")
    parser.add_argument("--world_size", type=int,
                        default=int(os.environ.get("SLURM_NTASKS", 1)),
                        help="Total number of parallel workers. Defaults to $SLURM_NTASKS.")
    parser.add_argument("--n_partitions", type=int, default=100,
                        help="Total number of embedding partition files to produce across all ranks. "
                             "Each rank writes roughly n_partitions / world_size files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_prefix = f"[rank {args.rank}/{args.world_size}] "

    # load the BrowseComp-Plus dataset
    ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")

    # Pin this process to its own GPU so each worker has exclusive use of one
    # device. CUDA_VISIBLE_DEVICES should already be set by SLURM when using
    # --gpus-per-task, but we set it explicitly here as a safety measure.
    # Honour a pre-set CUDA_VISIBLE_DEVICES so that single-GPU recovery runs
    # (e.g. --rank 1 --world_size 2 with CUDA_VISIBLE_DEVICES=0) work correctly.
    if args.world_size > 1 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.rank)

    print(f"{log_prefix}Loading Qwen3-Embedding-8B model...", flush=True)
    model_kwargs = {"device_map": "auto"}
    try:
        import flash_attn  # type: ignore # noqa: F401
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
    print(f"{log_prefix}Model loaded successfully.", flush=True)

    # each worker handles its own slice of the dataset
    ds = ds[args.rank::args.world_size]
    print(f"{log_prefix}Processing {len(ds['docid'])} documents.", flush=True)

    metadata: dict[str, dict] = {}
    final_elements: list[str] = []
    unique_element_ids: list[str] = []

    # create elements by splitting on double newlines and aggregating until the (est.) token count exceeds 512
    for doc_idx, (docid, text, url) in enumerate(zip(ds["docid"], ds["text"], ds["url"], strict=True)):
        if doc_idx % 1000 == 0:
            print(f"{log_prefix}Preprocessing document {doc_idx}/{len(ds['docid'])}...", flush=True)

        elements = text.split("\n\n")

        merged_elements = []
        current_elt = ""
        for element in elements:
            if len(current_elt) + len(element) > CHARS_PER_TOKEN_EST * TARGET_ELEMENT_TOKENS:
                merged_elements.append(current_elt)
                current_elt = ""
            current_elt += element + "\n\n"
        if current_elt:
            merged_elements.append(current_elt)

        # split any merged element that exceeds MAX_ELEMENT_TOKENS into ~TARGET_ELEMENT_TOKENS char chunks.
        chunk_chars = CHARS_PER_TOKEN_EST * TARGET_ELEMENT_TOKENS
        final_doc_elements = []
        for elt in merged_elements:
            if len(elt) > CHARS_PER_TOKEN_EST * MAX_ELEMENT_TOKENS:
                for start in range(0, len(elt), chunk_chars):
                    final_doc_elements.append(elt[start : start + chunk_chars])
            else:
                final_doc_elements.append(elt)

        for elt_id, element in enumerate(final_doc_elements):
            unique_element_id = f"{docid}_{elt_id}"
            unique_element_ids.append(unique_element_id)
            final_elements.append(element)
            metadata[unique_element_id] = {
                "docid": docid,
                "url": url,
                "cleaned": element,
                "element_id": elt_id,
            }

    assert len(final_elements) == len(metadata), "Duplicate element IDs found; check input data for consistency."
    metadata_path = os.path.join(args.output_dir, f"metadata_rank{args.rank}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    print(f"{log_prefix}Saved metadata for {len(metadata)} elements to {metadata_path}", flush=True)
    del metadata

    # Determine partition layout for this rank.
    n_partitions_rank = max(1, round(args.n_partitions / args.world_size))
    n_total = len(final_elements)
    partition_size = (n_total + n_partitions_rank - 1) // n_partitions_rank
    print(
        f"{log_prefix}Embedding {n_total} elements in {n_partitions_rank} "
        f"partitions of size ~{partition_size}...",
        flush=True,
    )
    embed_all(
        final_elements,
        unique_element_ids,
        model,
        partition_size=partition_size,
        output_dir=args.output_dir,
        rank=args.rank,
        log_prefix=log_prefix,
    )
    print(f"{log_prefix}Embedding complete.", flush=True)
    print(f"{log_prefix}Done.", flush=True)


if __name__ == "__main__":
    main()
