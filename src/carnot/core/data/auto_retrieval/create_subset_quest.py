"""
Script to create a subset dataset from QUEST evaluation results.

Steps:
1. Read quest_eval_results_val_base.jsonl and randomly sample 100 entries
2. Collect queries, first 10 titles, and recall scores
3. Find corresponding QUEST queries using prepare_quest_queries
4. Combine titles from retrieved results with QUEST document titles
5. Extract corresponding documents from documents.jsonl
6. Save the subset to new JSONL files
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Set, Any

from quest_utils import prepare_quest_queries, QuestQuery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
random.seed(150)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON: {e}")
    return results


def write_jsonl(path: str, data: List[Dict[str, Any]]):
    """Write a list of dictionaries to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    here = Path(__file__).resolve().parent
    
    # Paths
    eval_results_path = here / "tmp/quest_eval_results_val_base.jsonl"
    documents_path = here / "tmp/documents.jsonl"
    output_queries_path = here / "tmp/subset_3_quest_queries.jsonl"
    output_documents_path = here / "tmp/subset_3_documents.jsonl"
    output_summary_path = here / "tmp/subset_3_summary.json"
    
    # ========================================
    # Step 1: Read and sample 100 eval results
    # ========================================
    logger.info("Step 1: Reading evaluation results...")
    eval_results = read_jsonl(str(eval_results_path))
    
    # Filter out any summary entries (those without "query" key or that have special keys)
    eval_entries = [entry for entry in eval_results if "query" in entry and "query_index" in entry]
    logger.info(f"Found {len(eval_entries)} query entries in eval results")
    
    # Randomly sample 100 entries
    if len(eval_entries) > 100:
        sampled_entries = random.sample(eval_entries, 100)
    else:
        sampled_entries = eval_entries
        logger.warning(f"Only {len(sampled_entries)} entries available, using all of them")
    
    logger.info(f"Sampled {len(sampled_entries)} entries")
    
    # ========================================
    # Step 2: Collect queries, titles, and recall scores
    # ========================================
    logger.info("Step 2: Collecting queries, titles, and recall scores...")
    
    sampled_queries = []
    all_retrieved_titles: Set[str] = set()
    recall_scores = []
    
    for entry in sampled_entries:
        query = entry.get("query", "")
        recall_score = entry.get("recall@100", 0.0)
        retrieved_top_100 = entry.get("retrieved_top_100", [])
        
        # Get first 10 titles from retrieved results
        top_10_titles = [item.get("title", "") for item in retrieved_top_100[:10]]
        
        sampled_queries.append(query)
        recall_scores.append(recall_score)
        all_retrieved_titles.update(top_10_titles)
    
    average_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    logger.info(f"Collected {len(sampled_queries)} queries")
    logger.info(f"Collected {len(all_retrieved_titles)} unique titles from top-10 retrieved results")
    logger.info(f"Average recall@100 for sampled queries: {average_recall:.4f}")
    
    # ========================================
    # Step 3: Find corresponding QUEST queries
    # ========================================
    logger.info("Step 3: Loading QUEST queries from val.jsonl...")
    
    quest_queries = prepare_quest_queries(
        source="https://storage.googleapis.com/gresearch/quest/val.jsonl"
    )
    
    # Build a map from query text to QuestQuery
    query_to_quest_query: Dict[str, QuestQuery] = {}
    for q in quest_queries:
        query_to_quest_query[q.query] = q
    
    # Find matching QuestQuery objects for sampled queries
    matched_quest_queries: List[QuestQuery] = []
    quest_doc_titles: Set[str] = set()
    
    for query in sampled_queries:
        if query in query_to_quest_query:
            quest_query = query_to_quest_query[query]
            matched_quest_queries.append(quest_query)
            # Collect document titles from the QUEST query
            quest_doc_titles.update(quest_query.docs)
        else:
            logger.warning(f"Query not found in QUEST val.jsonl: {query[:50]}...")
    
    logger.info(f"Matched {len(matched_quest_queries)} QUEST queries")
    logger.info(f"Collected {len(quest_doc_titles)} unique document titles from QUEST queries")
    
    # ========================================
    # Step 4: Combine titles
    # ========================================
    logger.info("Step 4: Combining titles from retrieved results and QUEST documents...")
    
    combined_titles = all_retrieved_titles.union(quest_doc_titles)
    logger.info(f"Combined set contains {len(combined_titles)} unique titles")
    
    # ========================================
    # Step 5: Extract corresponding documents
    # ========================================
    logger.info("Step 5: Extracting corresponding documents from documents.jsonl...")
    
    # Read all documents and filter by title
    all_documents = read_jsonl(str(documents_path))
    logger.info(f"Loaded {len(all_documents)} documents from {documents_path}")
    
    # Filter documents whose titles are in the combined set
    subset_documents = []
    found_titles: Set[str] = set()
    
    for doc in all_documents:
        title = doc.get("title", "").strip()
        if title in combined_titles:
            subset_documents.append(doc)
            found_titles.add(title)
    
    logger.info(f"Found {len(subset_documents)} documents matching combined titles")
    
    # Check for missing titles
    missing_titles = combined_titles - found_titles
    if missing_titles:
        logger.warning(f"{len(missing_titles)} titles not found in documents.jsonl")
        logger.debug(f"Missing titles sample: {list(missing_titles)[:5]}")
    
    # ========================================
    # Step 6: Save results
    # ========================================
    logger.info("Step 6: Saving results...")
    
    # Save subset documents (same format as original documents.jsonl)
    write_jsonl(str(output_documents_path), subset_documents)
    logger.info(f"Saved {len(subset_documents)} documents to {output_documents_path}")
    
    # Save matched QUEST queries as JSONL
    quest_queries_data = []
    for q in matched_quest_queries:
        quest_queries_data.append({
            "query": q.query,
            "docs": q.docs,
            "original_query": q.original_query,
            "scores": q.scores,
            "metadata": q.metadata
        })
    write_jsonl(str(output_queries_path), quest_queries_data)
    logger.info(f"Saved {len(quest_queries_data)} QUEST queries to {output_queries_path}")
    
    # Save summary
    summary = {
        "num_sampled_queries": len(sampled_queries),
        "average_recall_at_100": average_recall,
        "num_retrieved_titles_from_top_10": len(all_retrieved_titles),
        "num_quest_doc_titles": len(quest_doc_titles),
        "num_combined_titles": len(combined_titles),
        "num_documents_found": len(subset_documents),
        "num_missing_titles": len(missing_titles),
        "sampled_query_indices": [entry.get("query_index") for entry in sampled_entries]
    }
    
    with open(output_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {output_summary_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Number of sampled queries: {len(sampled_queries)}")
    print(f"Average recall@100: {average_recall:.4f}")
    print(f"Unique titles from top-10 retrieved: {len(all_retrieved_titles)}")
    print(f"Unique titles from QUEST documents: {len(quest_doc_titles)}")
    print(f"Combined unique titles: {len(combined_titles)}")
    print(f"Documents extracted: {len(subset_documents)}")
    print(f"Missing titles: {len(missing_titles)}")
    print("=" * 60)
    
    return {
        "sampled_entries": sampled_entries,
        "matched_quest_queries": matched_quest_queries,
        "subset_documents": subset_documents,
        "summary": summary
    }


if __name__ == "__main__":
    main()