from __future__ import annotations
import logging
import copy
from typing import List, Dict, Any, Optional

from pathlib import Path
import json
import os
import dspy
import palimpzest as pz
from typing import get_args, get_origin
from _internal.sem_map import SemMapStrategy
from _internal.sem_map import sem_map, expand_sem_map_results_to_tags
from _internal.chroma_store import ChromaStore
from quest_utils import (
    prepare_quest_documents,
    prepare_quest_queries,
    QuestQuery
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def expand_metas(metadata: List[Dict[str, Any]], data_rows: List[Dict[str, str]], strategy: SemMapStrategy) -> List[Dict[str, Any]]:
    def _type_to_str(tp: Any) -> str:
        if tp is str: return "str"
        if tp is int: return "int"
        if tp is float: return "float"
        if tp is bool: return "bool"
        origin = get_origin(tp)
        args = get_args(tp)
        if origin is list and len(args) == 1:
            return f"List[{_type_to_str(args[0])}]"
        return str(tp)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")

    lm = dspy.LM("openai/gpt-5.1", temperature=1.0, max_tokens=16000, api_key=api_key)
    dspy.configure(lm=lm)

    here = Path(__file__).resolve().parent

    # Load concepts (final_concepts)
    concepts_path = here / "tmp/concept_generation_artifacts.json"
    obj = json.loads(concepts_path.read_text(encoding="utf-8"))
    concepts = obj["final_concepts"] if isinstance(obj, dict) else obj
    concepts = [" ".join(c.strip().split()) for c in concepts if isinstance(c, str) and c.strip()]

    sem_results, concept_schema_cols = sem_map(concepts=concepts, data=data_rows, strategy=strategy)

    expanded_results, expanded_schema, expanded_stats = expand_sem_map_results_to_tags(
        sem_results, concept_schema_cols)

    for meta in metadata:
        entity_id = str(meta.get("entity_id", "")).strip()
        if not entity_id:
            continue

        expanded_meta = expanded_results.get(entity_id)
        if not expanded_meta:
            continue

        for k, v in expanded_meta.items():
            if isinstance(v, bool):
                if v:                
                    meta[k] = True
            else:
                if v is not None:
                    meta[k] = v

    sem_payload = {
        "strategy": strategy.value,
        "concepts": list(concepts),
        "concept_schema_cols": [
            {"name": c["name"], "type": _type_to_str(c["type"]), "desc": c.get("desc", "")}
            for c in concept_schema_cols
        ],
        "results": sem_results[:30],
    }
    expanded_payload = {
        "schema": [
            {"name": c["name"], "type": _type_to_str(c["type"])}
            for c in expanded_schema
        ],
        "results": expanded_results[:30],
        "stats": expanded_stats,
    }

    sem_out_path = here / "sem_map/quest_sem_map_output.json"
    expanded_out_path = here / "sem_map/quest_sem_map_tagified_output.json"
    sem_out_path.parent.mkdir(parents=True, exist_ok=True)
    expanded_out_path.parent.mkdir(parents=True, exist_ok=True)

    sem_out_path.write_text(json.dumps(sem_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    expanded_out_path.write_text(json.dumps(expanded_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return metadata

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
    
    data_rows = []

    for doc_item in dataset:
        text = doc_item["text"]
        meta = doc_item["metadata"]
        
        data_rows.append({"id": meta["entity_id"], "text": text})

        docs_all.append(text)
        metas_all.append(meta)
        
    if if_expand_meta:
        logger.info("Expanding metadata...")
        metas_all = expand_metas(metas_all, data_rows, SemMapStrategy.HIERARCHY_FIRST)
        logger.info("✅ Metadata expanded")
            
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
    # store_base = create_collections(
    #     documents_path="tmp/documents.jsonl",
    #     collection_name="quest_base",
    #     persist_directory="./chroma_collections",
    #     if_expand_meta=False
    # )
    
    # 2) Expanded Collection
    store_expanded = create_collections(
        documents_path="tmp/documents.jsonl",
        collection_name="quest_expanded_hierarchy_first",
        persist_directory="./chroma_collections",
        if_expand_meta=True
    )
    
    print(f"✅ Expanded collection created")
    exit(0)
    # 3) Evaluate
    queries = prepare_quest_queries(source="https://storage.googleapis.com/gresearch/quest/val.jsonl")
    # avg_recall_base = evaluate_collection(store_base, queries, output_path="quest_eval_results_val_base.jsonl")
    avg_recall_expanded = evaluate_collection(store_expanded, queries, output_path="quest_eval_results_val_expanded.jsonl")
    
    # print(f"Average Recall (Base Collection): {avg_recall_base:.4f}")
    print(f"Average Recall (Expanded Collection): {avg_recall_expanded:.4f}")
