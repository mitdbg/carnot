"""Download FreshStack topics from HuggingFace to local JSONL the qatfd harness consumes.

FreshStack (https://fresh-stack.github.io/) ships one (corpus, queries) pair PER TOPIC; KARLBench
uses `langchain` (203 q / 49,514 docs) as its FreshStack, and `laravel` (184 q / 52,351 docs) is the
closest-sized companion we use as the dev set. This pulls each requested topic's parquet from
`freshstack/corpus-oct-2024` + `freshstack/queries-oct-2024` and writes, under {out_dir}/{topic}/:
    corpus.jsonl   one JSON doc per line  -> {"_id", "text", "metadata": {url, start_byte, end_byte}}
    queries.jsonl  one JSON query per line -> {"query_id", "query_title", "query_text", "nuggets": [...], ...}
The line schemas are passed through verbatim from HuggingFace, so the benchmark loader and the
embedding script (compute_freshstack_embeddings.py) read exactly the released fields.

Usage:
    python download_freshstack.py --topics langchain laravel --out_dir <skunk>/freshstack
Then (per topic) embed the corpus and build the Chroma collection — see run_freshstack_embeddings.slurm.
"""

import argparse
import json
import os

CORPUS_DATASET = "freshstack/corpus-oct-2024"
QUERIES_DATASET = "freshstack/queries-oct-2024"
ALL_TOPICS = ["langchain", "laravel", "angular", "godot", "yolo"]


def _write_jsonl(path: str, rows) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _dump_split(dataset: str, topic: str, split: str, out_path: str) -> int:
    from datasets import load_dataset

    ds = load_dataset(dataset, topic, split=split)
    n = _write_jsonl(out_path, (dict(row) for row in ds))
    print(f"  {dataset} [{topic}/{split}] -> {out_path} ({n} rows)", flush=True)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FreshStack topics to local JSONL.")
    parser.add_argument("--topics", nargs="+", default=["langchain", "laravel"],
                        help=f"FreshStack topics to download (default: langchain laravel). All: {ALL_TOPICS}")
    parser.add_argument("--out_dir", required=True, help="base dir; writes {out_dir}/{topic}/{corpus,queries}.jsonl")
    args = parser.parse_args()

    for topic in args.topics:
        if topic not in ALL_TOPICS:
            raise SystemExit(f"unknown topic {topic!r}; choose from {ALL_TOPICS}")
        print(f"Topic {topic}:", flush=True)
        topic_dir = os.path.join(args.out_dir, topic)
        # corpus is published under a single `train` split; queries under `test`.
        n_corpus = _dump_split(CORPUS_DATASET, topic, "train", os.path.join(topic_dir, "corpus.jsonl"))
        n_queries = _dump_split(QUERIES_DATASET, topic, "test", os.path.join(topic_dir, "queries.jsonl"))
        print(f"  done: {n_corpus} docs, {n_queries} queries.", flush=True)


if __name__ == "__main__":
    main()
