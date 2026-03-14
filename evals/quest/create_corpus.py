#!/usr/bin/env python3
"""Create a fixed corpus for QUEST evaluation.

This script selects queries and builds a corpus containing all their gt_docs
plus random filler documents.

Two selection modes:
1. Random selection (default): Randomly select queries until doc budget is hit
2. First-N selection: Use exactly the first N queries (for PZ/LOTUS comparison)

Usage:
    # Random selection with 1000-doc budget
    python create_corpus.py --doc-budget 1000 --output-dir data/corpus_1000
    
    # First 25 queries (for matching PZ/LOTUS test set)
    python create_corpus.py --first-n-queries 25 --output-dir data/corpus_first25

Outputs:
    - corpus.jsonl: The fixed corpus of documents
    - queries.jsonl: The selected queries (all gt_docs guaranteed in corpus)
    - metadata.json: Stats about the corpus and query selection
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_queries(queries_path: str) -> list[dict]:
    """Load queries from a JSONL file."""
    queries = []
    with open(queries_path) as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def load_documents(documents_path: str) -> list[dict]:
    """Load documents from a JSONL file."""
    documents = []
    with open(documents_path) as f:
        for line in f:
            documents.append(json.loads(line))
    return documents


def select_queries_random(
    queries: list[dict],
    doc_budget: int,
    seed: int = 42,
) -> tuple[list[dict], set[str]]:
    """Randomly select queries until doc budget is exhausted.
    
    This preserves the natural distribution of query difficulty,
    avoiding bias toward queries with fewer ground truth documents.
    
    Args:
        queries: List of query dicts with 'docs' field
        doc_budget: Maximum number of unique gt_docs allowed
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (selected_queries, selected_doc_titles)
    """
    random.seed(seed)
    
    selected_queries = []
    selected_docs = set()
    
    # Shuffle queries randomly
    shuffled = list(queries)
    random.shuffle(shuffled)
    
    for query in shuffled:
        new_docs = set(query['docs']) - selected_docs
        if len(selected_docs) + len(new_docs) <= doc_budget:
            selected_queries.append(query)
            selected_docs.update(query['docs'])
    
    return selected_queries, selected_docs


def select_first_n_queries(
    queries: list[dict],
    n: int,
) -> tuple[list[dict], set[str]]:
    """Select the first N queries and collect all their gt_docs.
    
    Use this when you need to match a specific set of test queries
    (e.g., for comparison with other systems like PZ/LOTUS).
    
    Args:
        queries: List of query dicts with 'docs' field
        n: Number of queries to select from the beginning
    
    Returns:
        Tuple of (selected_queries, selected_doc_titles)
    """
    selected_queries = queries[:n]
    selected_docs = set()
    
    for query in selected_queries:
        selected_docs.update(query['docs'])
    
    return selected_queries, selected_docs


def create_corpus(
    documents: list[dict],
    selected_docs: set[str],
    doc_budget: int,
    seed: int = 42,
) -> list[dict]:
    """Create corpus from selected gt_docs plus random filler.
    
    Args:
        documents: All available documents
        selected_docs: Set of document titles that must be included
        doc_budget: Total corpus size
        seed: Random seed for reproducibility
    
    Returns:
        List of document dicts forming the corpus
    """
    random.seed(seed)
    
    # Build title -> document mapping
    title_to_doc = {doc['title']: doc for doc in documents}
    
    # Add all selected gt_docs
    corpus = []
    missing_docs = []
    for title in selected_docs:
        if title in title_to_doc:
            corpus.append(title_to_doc[title])
        else:
            missing_docs.append(title)
    
    if missing_docs:
        print(f"Warning: {len(missing_docs)} gt_docs not found in documents file")
    
    # Add random filler docs if needed
    filler_needed = doc_budget - len(corpus)
    if filler_needed > 0:
        available_fillers = [
            doc for doc in documents 
            if doc['title'] not in selected_docs
        ]
        random.shuffle(available_fillers)
        corpus.extend(available_fillers[:filler_needed])
    
    # Shuffle final corpus
    random.shuffle(corpus)
    
    return corpus


def main():
    parser = argparse.ArgumentParser(
        description="Create a fixed corpus for QUEST evaluation"
    )
    parser.add_argument(
        "--queries-path",
        type=str,
        default="data/test.jsonl",
        help="Path to the queries JSONL file",
    )
    parser.add_argument(
        "--documents-path",
        type=str,
        default="data/documents.jsonl",
        help="Path to the documents JSONL file",
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
        default="data/corpus_1000",
        help="Output directory for corpus files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter to a specific domain (films, books, plants, animals)",
    )
    parser.add_argument(
        "--first-n-queries",
        type=int,
        default=None,
        help="Use exactly the first N queries (for matching PZ/LOTUS test sets). "
             "Overrides random selection and doc-budget for query selection.",
    )
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading queries from {args.queries_path}...")
    queries = load_queries(args.queries_path)
    
    # Filter by domain if specified
    if args.domain:
        queries = [q for q in queries if q['metadata']['domain'] == args.domain]
        print(f"Filtered to {len(queries)} queries in domain '{args.domain}'")
    
    print(f"Loading documents from {args.documents_path}...")
    documents = load_documents(args.documents_path)
    print(f"Loaded {len(documents)} documents")
    
    # Select queries based on mode
    if args.first_n_queries is not None:
        print(f"\nSelecting first {args.first_n_queries} queries (fixed set for comparison)...")
        selected_queries, selected_docs = select_first_n_queries(
            queries, args.first_n_queries
        )
        print(f"Selected {len(selected_queries)} queries with {len(selected_docs)} gt_docs")
        if len(selected_docs) > args.doc_budget:
            print(f"Warning: gt_docs ({len(selected_docs)}) exceeds doc_budget ({args.doc_budget})")
            print(f"         Corpus will contain {len(selected_docs)} docs (no filler)")
    else:
        print(f"\nSelecting queries randomly with doc budget of {args.doc_budget}...")
        selected_queries, selected_docs = select_queries_random(
            queries, args.doc_budget, seed=args.seed
        )
        print(f"Selected {len(selected_queries)} queries with {len(selected_docs)} gt_docs")
    
    # Show domain distribution
    domains = Counter(q['metadata']['domain'] for q in selected_queries)
    print(f"Domain distribution: {dict(domains)}")
    
    print("\nCreating corpus...")
    corpus = create_corpus(
        documents, selected_docs, args.doc_budget, seed=args.seed
    )
    print(f"Corpus size: {len(corpus)} documents")
    
    # Save outputs
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
        "domain_filter": args.domain,
        "first_n_queries": args.first_n_queries,
        "selection_mode": "first_n" if args.first_n_queries else "random",
        "num_queries": len(selected_queries),
        "num_gt_docs": len(selected_docs),
        "num_corpus_docs": len(corpus),
        "num_filler_docs": len(corpus) - len(selected_docs),
        "domain_distribution": dict(domains),
        "source_queries_path": args.queries_path,
        "source_documents_path": args.documents_path,
    }
    
    print(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\nDone!")
    print("\nSummary:")
    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(selected_queries)} (all gt_docs in corpus)")
    print(f"  GT docs: {len(selected_docs)}")
    print(f"  Filler docs: {len(corpus) - len(selected_docs)}")


if __name__ == "__main__":
    main()
