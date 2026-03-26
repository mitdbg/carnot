#!/usr/bin/env python3
"""Create a fixed corpus for BrowserComp+ evaluation.

This mirrors the QUEST corpus creation flow, but BrowserComp+ queries come from
`topics-qrels/queries.tsv` and evidence labels come from
`topics-qrels/qrel_evidence.txt`.

By default this script selects the first 25 queries, guarantees that all of
their evidence documents are included, and fills the remainder of the 1000-doc
budget with random corpus documents.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import tiktoken


def load_queries(queries_path: str, qrels_path: str) -> list[dict]:
    """Load BrowserComp+ queries and attach evidence docids."""
    qrels: dict[str, list[str]] = defaultdict(list)
    with open(qrels_path) as f:
        for line in f:
            qid, _, docid, rel = line.split()
            if rel != "0":
                qrels[qid].append(docid)

    queries = []
    with open(queries_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, query_text = line.split("\t", maxsplit=1)
            docs = qrels.get(qid, [])
            if not docs:
                continue
            queries.append(
                {
                    "query_id": qid,
                    "query": query_text,
                    "docs": docs,
                }
            )
    return queries


def iter_documents_jsonl(documents_path: str):
    """Yield documents from a local JSONL file."""
    with open(documents_path) as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def iter_documents_hf(dataset_name: str, split: str):
    """Yield documents from the BrowserComp+ Hugging Face corpus."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    for row in dataset:
        yield row


def normalize_document(doc: dict) -> dict:
    """Keep only the fields used by evaluation/indexing."""
    return {
        "docid": str(doc["docid"]),
        "text": doc.get("text", ""),
        "url": doc.get("url", ""),
    }


def count_text_tokens(text: str, encoding) -> int:
    """Return the token count for one document text."""
    return len(encoding.encode(text or "", disallowed_special=()))


def build_gt_doc_token_counts(documents_iter, gt_docids: set[str]) -> tuple[dict[str, int], list[str]]:
    """Collect token counts for all ground-truth docs referenced by the query set."""
    encoding = tiktoken.get_encoding("cl100k_base")
    token_counts: dict[str, int] = {}
    seen_docids: set[str] = set()

    for raw_doc in documents_iter:
        doc = normalize_document(raw_doc)
        docid = doc["docid"]
        if docid in seen_docids:
            continue
        seen_docids.add(docid)

        if docid in gt_docids:
            token_counts[docid] = count_text_tokens(doc["text"], encoding)

    missing_docids = sorted(gt_docids - set(token_counts))
    return token_counts, missing_docids


def filter_queries_by_gt_doc_tokens(
    queries: list[dict],
    gt_doc_token_counts: dict[str, int],
    max_gt_doc_tokens: int,
) -> tuple[list[dict], list[dict]]:
    """Split queries into eligible and excluded groups based on GT doc token lengths."""
    eligible_queries = []
    excluded_queries = []

    for query in queries:
        too_long_docs = []
        missing_docs = []
        for docid in query["docs"]:
            token_count = gt_doc_token_counts.get(docid)
            if token_count is None:
                missing_docs.append(docid)
            elif token_count > max_gt_doc_tokens:
                too_long_docs.append({"docid": docid, "token_count": token_count})

        if too_long_docs or missing_docs:
            excluded_queries.append(
                {
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "missing_docids": missing_docs,
                    "too_long_docs": too_long_docs,
                }
            )
        else:
            eligible_queries.append(query)

    return eligible_queries, excluded_queries


def select_queries_random(
    queries: list[dict],
    doc_budget: int,
    seed: int = 42,
) -> tuple[list[dict], set[str]]:
    """Randomly select queries until the evidence-doc budget is exhausted."""
    random.seed(seed)

    selected_queries = []
    selected_docs = set()

    shuffled = list(queries)
    random.shuffle(shuffled)

    for query in shuffled:
        new_docs = set(query["docs"]) - selected_docs
        if len(selected_docs) + len(new_docs) <= doc_budget:
            selected_queries.append(query)
            selected_docs.update(query["docs"])

    return selected_queries, selected_docs


def select_first_n_queries(
    queries: list[dict],
    n: int,
) -> tuple[list[dict], set[str]]:
    """Select the first N queries and collect all evidence docids."""
    selected_queries = queries[:n]
    selected_docs = set()

    for query in selected_queries:
        selected_docs.update(query["docs"])

    return selected_queries, selected_docs


def create_corpus(
    documents_iter,
    selected_docs: set[str],
    doc_budget: int,
    max_doc_tokens: int | None = None,
    seed: int = 42,
) -> tuple[list[dict], list[str], int]:
    """Create corpus from selected evidence docs plus random filler docs.

    This streams through the full BrowserComp+ corpus once, so it does not need
    to load the multi-GB source corpus into memory.
    """
    random.seed(seed)
    encoding = tiktoken.get_encoding("cl100k_base") if max_doc_tokens is not None else None

    required_docs: dict[str, dict] = {}
    filler_reservoir: list[dict] = []
    seen_docids: set[str] = set()
    filler_candidates_seen = 0
    skipped_too_long_docs = 0

    for raw_doc in documents_iter:
        doc = normalize_document(raw_doc)
        docid = doc["docid"]

        if docid in seen_docids:
            continue
        seen_docids.add(docid)

        if max_doc_tokens is not None:
            token_count = count_text_tokens(doc["text"], encoding)
            if token_count > max_doc_tokens:
                skipped_too_long_docs += 1
                continue

        if docid in selected_docs:
            required_docs[docid] = doc
            continue

        filler_candidates_seen += 1
        if len(filler_reservoir) < doc_budget:
            filler_reservoir.append(doc)
            continue

        replacement_idx = random.randint(0, filler_candidates_seen - 1)
        if replacement_idx < doc_budget:
            filler_reservoir[replacement_idx] = doc

    missing_docs = sorted(selected_docs - set(required_docs))

    corpus = list(required_docs.values())
    filler_needed = max(0, doc_budget - len(corpus))
    random.shuffle(filler_reservoir)
    corpus.extend(filler_reservoir[:filler_needed])
    random.shuffle(corpus)

    return corpus, missing_docs, skipped_too_long_docs


def main():
    parser = argparse.ArgumentParser(
        description="Create a fixed corpus for BrowserComp+ evaluation"
    )
    parser.add_argument(
        "--queries-path",
        type=str,
        default="topics-qrels/queries.tsv",
        help="Path to BrowserComp+ queries.tsv",
    )
    parser.add_argument(
        "--qrels-path",
        type=str,
        default="topics-qrels/qrel_evidence.txt",
        help="Path to BrowserComp+ evidence qrels",
    )
    parser.add_argument(
        "--documents-path",
        type=str,
        default=None,
        help="Optional local JSONL path for the full BrowserComp+ corpus",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="Tevatron/browsecomp-plus-corpus",
        help="Hugging Face dataset to use when --documents-path is omitted",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        help="Hugging Face split to use when --documents-path is omitted",
    )
    parser.add_argument(
        "--doc-budget",
        type=int,
        default=1000,
        help="Maximum number of documents in the corpus",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/corpus_first25",
        help="Output directory for corpus files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=["first_n", "random"],
        default="first_n",
        help="How to choose evaluation queries",
    )
    parser.add_argument(
        "--first-n-queries",
        type=int,
        default=25,
        help="Number of leading queries to keep in first_n mode",
    )
    parser.add_argument(
        "--max-gt-doc-tokens",
        type=int,
        default=None,
        help="Exclude queries whose GT docs exceed this limit, and only include corpus docs at or below this token count",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading queries from {args.queries_path}...")
    queries = load_queries(args.queries_path, args.qrels_path)
    print(f"Loaded {len(queries)} queries with evidence documents")

    excluded_queries = []
    missing_gt_docs_for_filter: list[str] = []

    if args.documents_path:
        print(f"Streaming documents from local JSONL: {args.documents_path}")
        source_documents = args.documents_path
    else:
        print(
            "Streaming documents from Hugging Face dataset: "
            f"{args.hf_dataset} [{args.hf_split}]"
        )
        source_documents = f"{args.hf_dataset}:{args.hf_split}"

    def make_documents_iter():
        if args.documents_path:
            return iter_documents_jsonl(args.documents_path)
        return iter_documents_hf(args.hf_dataset, args.hf_split)

    if args.max_gt_doc_tokens is not None:
        print(
            "\nFiltering queries by ground-truth doc length: "
            f"max {args.max_gt_doc_tokens} tokens per GT doc"
        )
        gt_docids = {docid for query in queries for docid in query["docs"]}
        gt_doc_token_counts, missing_gt_docs_for_filter = build_gt_doc_token_counts(
            make_documents_iter(),
            gt_docids,
        )
        queries, excluded_queries = filter_queries_by_gt_doc_tokens(
            queries,
            gt_doc_token_counts,
            args.max_gt_doc_tokens,
        )
        print(
            f"Eligible queries after GT token filtering: {len(queries)} "
            f"(excluded {len(excluded_queries)})"
        )
        if missing_gt_docs_for_filter:
            print(
                f"Warning: {len(missing_gt_docs_for_filter)} GT docs were not found "
                "while computing token lengths"
            )

    if args.selection_mode == "first_n":
        print(f"\nSelecting first {args.first_n_queries} queries...")
        selected_queries, selected_docs = select_first_n_queries(
            queries, args.first_n_queries
        )
    else:
        print(f"\nSelecting queries randomly with doc budget of {args.doc_budget}...")
        selected_queries, selected_docs = select_queries_random(
            queries, args.doc_budget, seed=args.seed
        )

    print(
        f"Selected {len(selected_queries)} queries with "
        f"{len(selected_docs)} evidence docs"
    )
    if len(selected_docs) > args.doc_budget:
        print(
            f"Warning: evidence docs ({len(selected_docs)}) exceed doc_budget "
            f"({args.doc_budget})"
        )
        print(f"         Corpus will contain at least {len(selected_docs)} docs")

    print("\nCreating corpus...")
    corpus, missing_docs, skipped_too_long_docs = create_corpus(
        make_documents_iter(),
        selected_docs,
        args.doc_budget,
        max_doc_tokens=args.max_gt_doc_tokens,
        seed=args.seed,
    )
    print(f"Corpus size: {len(corpus)} documents")
    if missing_docs:
        print(f"Warning: {len(missing_docs)} evidence docs not found in corpus source")
    if skipped_too_long_docs:
        print(
            f"Skipped {skipped_too_long_docs} source documents that exceeded "
            f"the {args.max_gt_doc_tokens}-token cap"
        )

    corpus_path = output_dir / "corpus.jsonl"
    queries_path = output_dir / "queries.jsonl"
    metadata_path = output_dir / "metadata.json"

    print(f"\nSaving corpus to {corpus_path}...")
    with open(corpus_path, "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc) + "\n")

    print(f"Saving queries to {queries_path}...")
    with open(queries_path, "w") as f:
        for query in selected_queries:
            f.write(json.dumps(query) + "\n")

    metadata = {
        "doc_budget": args.doc_budget,
        "seed": args.seed,
        "selection_mode": args.selection_mode,
        "first_n_queries": args.first_n_queries
        if args.selection_mode == "first_n"
        else None,
        "max_gt_doc_tokens": args.max_gt_doc_tokens,
        "num_queries": len(selected_queries),
        "num_queries_after_gt_token_filter": len(queries),
        "num_queries_excluded_by_gt_token_filter": len(excluded_queries),
        "num_gt_docs": len(selected_docs),
        "num_missing_gt_docs": len(missing_docs),
        "missing_gt_docids": missing_docs,
        "num_missing_gt_docs_for_filter": len(missing_gt_docs_for_filter),
        "missing_gt_docids_for_filter": missing_gt_docs_for_filter,
        "num_corpus_docs": len(corpus),
        "num_filler_docs": len(corpus) - (len(selected_docs) - len(missing_docs)),
        "num_source_docs_skipped_for_length": skipped_too_long_docs,
        "source_queries_path": args.queries_path,
        "source_qrels_path": args.qrels_path,
        "source_documents": source_documents,
        "selected_query_ids": [query["query_id"] for query in selected_queries],
        "excluded_queries_by_gt_token_filter": excluded_queries,
    }

    print(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone!")
    print("\nSummary:")
    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(selected_queries)}")
    print(f"  Evidence docs requested: {len(selected_docs)}")
    print(f"  Evidence docs missing: {len(missing_docs)}")
    print(f"  Filler docs: {metadata['num_filler_docs']}")
    print(f"  Source docs skipped for length: {skipped_too_long_docs}")


if __name__ == "__main__":
    main()
