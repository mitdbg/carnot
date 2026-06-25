"""Filter QAMPARI's chunked Wikipedia down to KARL's QAMPARI corpus.

KARL indexes, for QAMPARI, only the Wikipedia chunks that contain at least one gold answer entity
(reported as 256,680 chunks). This script reproduces that: it reads QAMPARI's released chunked
Wikipedia (`chunked_wikipedia.tar.gz` -> `wikipedia_chunks/chunks_v5/wikipedia_chunks_*.jsonl`,
~21M ~100-token passages) and keeps every chunk whose text mentions a gold answer entity (the
`answer_text` + `aliases` of every answer across all 1000 test questions), as a whole-word,
case-insensitive match.

  NOTE: the exact count depends on the matching rule (whole-word, case-insensitive over the
  passage text). It will be ~256,680 but is not guaranteed to hit it exactly without KARL's own
  filter code; the printed kept-count tells you what you actually built.

Each released chunk is one JSON line:
    {"id": "<page_id>__<n>", "contents": "<title> <body>",
     "meta": {"revid","url","title","file_path","page_id","content","chunk_id"}}
Kept chunks are written through UNCHANGED (same schema), so compute_qampari_embeddings.py can embed
`contents` and carry `meta` straight into the index metadata.

Output layout (consumed by compute_qampari_embeddings.py): one `qampari_chunks_*.jsonl` per input
file, so a rerun resumes by skipping files whose output already exists.

Both `--chunks_dir` and `--output_dir` accept a local path OR an `s3://bucket/prefix` URI (boto3,
same I/O layer as compute_biogen_embeddings.py). A local run fans the input files across a process
pool (`--processes`); set `--rank`/`--world_size` to additionally shard across SLURM array tasks.

Matcher: prefers `flashtext` (KeywordProcessor, O(text) whole-word matching); falls back to batched
`re` alternations if flashtext is not installed (`pip install flashtext`).

Usage:
    python preprocess_qampari_corpus.py \
        --chunks_dir   /data/wikipedia_chunks/chunks_v5 \
        --questions    <skunk>/qampari/test_data.jsonl \
        --output_dir   <skunk>/qampari/qampari-corpus
"""

import argparse
import glob
import io
import json
import os
import re
from functools import partial
from multiprocessing import Pool
from urllib.parse import urlparse

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
    """Sorted list of every *.jsonl under a local dir (recursive) or an s3:// prefix."""
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
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)


# ---------------------------------------------------------------------------
# Gold-entity matcher.
# ---------------------------------------------------------------------------


def _collect_entities(questions_path: str) -> list[str]:
    """Every gold answer surface form (answer_text + aliases) across all questions, lowercased,
    whitespace-collapsed, de-duplicated. These are the phrases a kept chunk must mention."""
    forms: set[str] = set()
    for line in iter_lines(questions_path):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        for ans in rec.get("answer_list", []):
            surfaces = [ans.get("answer_text", "")] + list(ans.get("aliases", []))
            for s in surfaces:
                s = " ".join(str(s).split()).lower()
                if len(s) >= 2:  # drop empty / single-char forms (noise)
                    forms.add(s)
    return sorted(forms)


class _Matcher:
    """`contains(text) -> bool`: does `text` mention any gold entity as a whole word?

    Uses flashtext when available (O(len(text)) keyword scan), else batched case-insensitive `re`
    alternations with non-alphanumeric boundaries. Picklable, so it ships to pool workers."""

    def __init__(self, entities: list[str]) -> None:
        self.entities = entities
        self._kp = None
        self._regexes: list[re.Pattern] | None = None
        self._build()

    def _build(self) -> None:
        try:
            from flashtext import KeywordProcessor

            kp = KeywordProcessor(case_sensitive=False)
            for e in self.entities:
                kp.add_keyword(e)
            self._kp = kp
        except ImportError:
            # Fallback: batched alternations, each phrase bounded by non-alphanumeric edges.
            self._regexes = []
            batch = 4000
            for i in range(0, len(self.entities), batch):
                alts = "|".join(re.escape(e) for e in self.entities[i : i + batch])
                self._regexes.append(re.compile(rf"(?<![a-z0-9]){alts}(?![a-z0-9])", re.IGNORECASE))

    def __getstate__(self) -> dict:
        # KeywordProcessor / compiled regexes are rebuilt in the worker from `entities` (smaller,
        # faster to pickle than the built structures).
        return {"entities": self.entities}

    def __setstate__(self, state: dict) -> None:
        self.entities = state["entities"]
        self._kp = None
        self._regexes = None
        self._build()

    def contains(self, text: str) -> bool:
        if self._kp is not None:
            return len(self._kp.extract_keywords(text)) > 0
        low = text.lower()
        return any(r.search(low) for r in (self._regexes or []))


# ---------------------------------------------------------------------------
# Per-file filtering.
# ---------------------------------------------------------------------------


def _chunk_text(rec: dict) -> str:
    """Text a chunk is matched against: the indexed `contents` (title + body), falling back to the
    meta body, so a gold entity mention in either the title or the body keeps the chunk."""
    if rec.get("contents"):
        return str(rec["contents"])
    meta = rec.get("meta") or {}
    return f"{meta.get('title', '')} {meta.get('content', '')}".strip()


def _output_path(output_dir: str, input_path: str) -> str:
    base = os.path.basename(input_path).replace("wikipedia_chunks", "qampari_chunks")
    if not base.endswith(".jsonl"):
        base = base + ".jsonl"
    return _join(output_dir, base)


def _filter_file(input_path: str, matcher: _Matcher, output_dir: str) -> tuple[str, int, int]:
    """Filter one chunk file -> one output file. Returns (input_path, n_kept, n_total).
    Skips work (and reports n_kept=-1) if the output file already exists (resume)."""
    out_path = _output_path(output_dir, input_path)
    if _exists(out_path):
        return (input_path, -1, 0)
    kept: list[str] = []
    n_total = 0
    for line in iter_lines(input_path):
        line = line.strip()
        if not line:
            continue
        n_total += 1
        rec = json.loads(line)
        if matcher.contains(_chunk_text(rec)):
            kept.append(json.dumps(rec))
    write_bytes(out_path, ("\n".join(kept) + ("\n" if kept else "")).encode("utf-8"))
    return (input_path, len(kept), n_total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter QAMPARI chunked Wikipedia to chunks with a gold entity.")
    parser.add_argument("--chunks_dir", required=True, help="local dir OR s3:// prefix of wikipedia_chunks_*.jsonl")
    parser.add_argument("--questions", required=True, help="QAMPARI test_data.jsonl (gold entities source)")
    parser.add_argument("--output_dir", required=True, help="local dir OR s3:// prefix for qampari_chunks_*.jsonl")
    parser.add_argument("--processes", type=int, default=os.cpu_count(), help="local worker processes")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", 0)),
                        help="this task's index for SLURM-array file sharding")
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)),
                        help="number of SLURM-array tasks sharding the files")
    args = parser.parse_args()

    if not _is_s3(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"Collecting gold entities from {args.questions} ...", flush=True)
    entities = _collect_entities(args.questions)
    print(f"  {len(entities)} distinct gold answer surface forms.", flush=True)
    matcher = _Matcher(entities)

    files = list_jsonl(args.chunks_dir)
    if not files:
        raise FileNotFoundError(f"no *.jsonl files under {args.chunks_dir}")
    my_files = files[args.rank :: args.world_size]
    print(f"[rank {args.rank}/{args.world_size}] {len(my_files)}/{len(files)} chunk files to filter.", flush=True)

    n_kept_total = 0
    n_seen_total = 0
    n_files = 0
    worker = partial(_filter_file, matcher=matcher, output_dir=args.output_dir)
    procs = max(1, min(args.processes, len(my_files)))
    with Pool(procs) as pool:
        for input_path, n_kept, n_total in pool.imap_unordered(worker, my_files):
            n_files += 1
            if n_kept < 0:
                print(f"  [{n_files}/{len(my_files)}] skip (exists) {os.path.basename(input_path)}", flush=True)
                continue
            n_kept_total += n_kept
            n_seen_total += n_total
            if n_files % 50 == 0 or n_files == len(my_files):
                print(
                    f"  [{n_files}/{len(my_files)}] kept {n_kept_total} / {n_seen_total} chunks so far",
                    flush=True,
                )
    print(f"[rank {args.rank}] DONE. Kept {n_kept_total} chunks from {n_seen_total} scanned "
          f"(this rank's shard). Total across all ranks should be ~256,680.", flush=True)


if __name__ == "__main__":
    main()
