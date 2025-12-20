from __future__ import annotations
import logging
import copy
from typing import List, Dict, Any, Optional

from _internal.chroma_store import ChromaStore
from quest_utils import (
    prepare_quest_documents,
    prepare_quest_queries,
    QuestQuery
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def expand_metas(metadata: Dict[str, Any]) -> Dict[str, Any]:
    # TODO: Implement expansion logic
    return metadata  # Return the original metadata as a fallback

def recall(predicted: List[str], gold: List[str]) -> float:
    predicted_set = set(predicted)
    gold_set = set(gold)
    
    if not gold_set:
        return 0.0
    if not predicted_set:
        return 0.0
    
    covered_docs = gold_set.intersection(predicted_set)
    return len(covered_docs) / len(gold_set)

def create_collections(documents_path: str, collection_name: str, persist_directory: str, if_expand_meta: bool = False) -> ChromaStore:
    store = ChromaStore(collection_name=collection_name, persist_directory=persist_directory)

    logger.info("Preparing documents for ingestion...")
    dataset = prepare_quest_documents(
        jsonl_path=documents_path,
        index_first_512=True
    )
        
    docs_all = []
    metas_all = []

    for doc_item in dataset:
        text = doc_item["text"]
        meta = doc_item["metadata"]
        
        if if_expand_meta:
            meta = expand_metas(meta)

        docs_all.append(text)
        metas_all.append(meta)

    # Upsert all documents at once
    if docs_all:
        store.upsert_documents(documents=docs_all, metadatas=metas_all)
        logger.info(f"Ingested {len(docs_all)} documents into '{collection_name}'")

    return store

import json

def evaluate_collection(store: ChromaStore, queries: List[QuestQuery], output_path: Optional[str] = None) -> float:
    """
    Evaluates a single ChromaStore collection against a list of queries.
    Returns the average recall.
    If output_path is provided, saves query-level results to a JSONL file.
    """
    # TODO: LLM-based query rewriter
    
    top_k = 100
    
    total_recall = 0.0
    
    # Open file if path provided
    f_out = open(output_path, "w", encoding="utf-8") if output_path else None
    
    if not queries:
        logger.warning("⚠️ No queries to evaluate.")
        if f_out: f_out.close()
        return 0.0
    
    for i, q in enumerate(queries):
        results = store.query(q.query, n_results=top_k)
        
        predicted = []
        retrieved_details = []
        
        for result in results:
            meta = result.get("metadata") or {}
            title = meta.get("title")
            predicted.append(title)
            
            if f_out:
                retrieved_details.append({
                    "title": title,
                    "source": meta.get("source"),
                    "score": result.get("distance")
                })
        
        score = recall(predicted, q.docs)
        total_recall += score

        # Save to file
        if f_out:
            record = {
                "query_index": i,
                "query": q.query,
                f"recall@{top_k}": score,
                f"retrieved_top_{top_k}": retrieved_details
            }
            f_out.write(json.dumps(record) + "\n")

        if i % 10 == 0:
            logger.info(f"Evaluated {i+1}/{len(queries)} queries.")
            
    logger.info(f"Total Recall@{top_k}: {total_recall / len(queries):.4f}")
    
    if f_out:
        f_out.write(json.dumps(
            {
                f"Total Recall@{top_k}": total_recall / len(queries)
            }
        ) + "\n")
        f_out.close()
        logger.info(f"Saved evaluation results to {output_path}")

    return total_recall / len(queries) if queries else 0.0


if __name__ == "__main__":
    # 1) Base Collection
    store_base = create_collections(
        documents_path="tmp/documents.jsonl",
        collection_name="quest_base",
        persist_directory="./chroma_collections",
        if_expand_meta=False
    )
    
    # # 2) Expanded Collection
    # store_expanded = create_collections(
    #     documents_path="dataset/quest/documents.jsonl",
    #     collection_name="quest_expanded",
    #     persist_directory="./chroma_quest_eval",
    #     if_expand_meta=True
    # )
    
    # 3) Evaluate
    queries = prepare_quest_queries(source="https://storage.googleapis.com/gresearch/quest/val.jsonl")
    avg_recall_base = evaluate_collection(store_base, queries, output_path="quest_eval_results_val_base.jsonl")
    # avg_recall_expanded = evaluate_collection(store_expanded, queries, output_path="quest_eval_results_val_expanded.jsonl")
    
    print(f"Average Recall (Base Collection): {avg_recall_base:.4f}")
    # print(f"Average Recall (Expanded Collection): {avg_recall_expanded:.4f}")
