from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from _internal.chroma_store import ChromaStore
from _internal.query_planner import LLMQueryPlanner
from _internal.sem_map import sem_map, expand_sem_map_results_to_tags
from _internal.hierarchy_augment import postprocess_sem_map
from quest_utils import prepare_quest_documents, QuestQuery, prepare_quest_queries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent


def build_filter_catalog(
    expanded_results: Dict[str, Dict[str, Any]],
    expanded_schema: List[Dict[str, Any]],
    expanded_stats: Dict[str, Dict[str, float]],
    min_frequency: int,
) -> tuple[Dict[str, Any], set[str]]:
    """Build a frequency-filtered catalog of queryable tags for the LLM planner."""
    type_by_name = {s["name"]: s["type"] for s in expanded_schema}
    frequent_tags = {k for k, s in expanded_stats.items() if s["present"] >= min_frequency}

    # Pre-collect allowed values for non-bool tags in one pass.
    non_bool_tags = {t for t in frequent_tags if type_by_name.get(t) is not bool}
    allowed_values: Dict[str, Set[Any]] = {t: set() for t in non_bool_tags}
    if allowed_values:
        for doc_meta in expanded_results.values():
            for k, v in doc_meta.items():
                if k in allowed_values and v is not None:
                    allowed_values[k].add(v)

    catalog: Dict[str, Any] = {}
    for tag in sorted(frequent_tags):
        tp = type_by_name[tag]
        freq = int(expanded_stats[tag]["present"])

        if tp is bool:
            base, value = tag.rsplit(":", 1)
            if base not in catalog:
                catalog[base] = {"type": "bool", "allowed_values": []}
            catalog[base]["allowed_values"].append({"value": value, "frequency": freq})
        else:
            catalog[tag] = {
                "type": tp.__name__ if hasattr(tp, "__name__") else str(tp),
                "frequency": freq,
                "allowed_values": sorted(allowed_values.get(tag, set())),
            }

    return catalog, frequent_tags


def expand_metas(
    metadata: List[Dict[str, Any]],
    data_rows: List[Dict[str, str]],
    min_frequency: int = 3,
    dump_intermediate: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Run the full post-processing pipeline on sem_map results.

    Transforms raw per-document concept extractions into stable, queryable
    metadata columns suitable for ChromaDB filtering.
    """
    out_dir = HERE / "sem_map_subset_3"
    out_dir.mkdir(parents=True, exist_ok=True)

    if dump_intermediate is None:
        dump_intermediate = os.environ.get("SEM_MAP_DUMP_INTERMEDIATE", "").strip() == "1"

    def _schema_cols_json(cols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def _type_to_json_name(tp: Any) -> str:
            if isinstance(tp, type):
                return tp.__name__
            return str(tp).replace("typing.", "")

        return [
            {
                "name": c["name"],
                "type": _type_to_json_name(c["type"]),
                "desc": c.get("desc", ""),
            }
            for c in cols
        ]

    def _dump_json(name: str, payload: Dict[str, Any]) -> None:
        if not dump_intermediate:
            return
        (out_dir / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _deserialize_type(type_str: str) -> Any:
        """Convert a JSON-serialized type name back to a Python type."""
        if type_str == "int":
            return int
        if type_str == "str":
            return str
        if type_str == "bool":
            return bool
        if type_str == "float":
            return float
        if type_str.startswith("list["):
            inner = type_str[5:-1]
            return list[_deserialize_type(inner)]
        return str  # safe fallback

    # Load concept schema, restoring Python types from their JSON-serialized names.
    concept_schema_cols = [
        {**col, "type": _deserialize_type(col["type"])}
        for col in json.loads(
            (HERE / "sem_map_subset_1/concept_schema_cols.json").read_text(encoding="utf-8")
        )
    ]
        
    # sem_results, concept_schema_cols = sem_map(data=data_rows, concept_schema_cols=concept_schema_cols)

    # logger.info("Postprocessing sem_map raw output...")
    # sem_results = postprocess_sem_map(
    #     sem_results,
    #     concept_schema_cols,
    #     out_dir=str(out_dir) if dump_intermediate else None
    #     )
    # logger.info("Postprocessing complete: step 1-3.")
    
    # Load augmented sem_map output (output of postprocess Steps 1–3).
    augmented_path = HERE / "sem_map_subset_3/postprocess_step3_augmented.json"
    if not augmented_path.exists():
        raise FileNotFoundError(
            f"Missing augmented sem_map file: {augmented_path}. "
            "Run postprocess_sem_map first to generate it."
        )
    sem_results = json.loads(augmented_path.read_text(encoding="utf-8"))

    # Step 4: Flatten string/list columns into boolean tag columns for ChromaDB.
    logger.info("Expanding sem_map results to tags...")
    expanded_results, expanded_schema, expanded_stats = expand_sem_map_results_to_tags(
        sem_results, concept_schema_cols
    )

    # Step 5: Build frequency-filtered catalog for the LLM query planner.
    filter_catalog, frequent_tags = build_filter_catalog(
        expanded_results, expanded_schema, expanded_stats, min_frequency
    )
    logger.info(
        "Filters: %d base filters, %d tags (from %d)",
        len(filter_catalog), len(frequent_tags), len(expanded_stats),
    )
    _dump_json("filter_catalog.json", filter_catalog)

    # Merge frequent expanded tags into the original metadata dicts.
    for meta in metadata:
        entity_id = str(meta.get("entity_id", "")).strip()
        expanded_meta = expanded_results.get(entity_id, {})
        for k, v in expanded_meta.items():
            if k in frequent_tags and v is not None:
                meta[k] = v

    return metadata


def recall_at_k(ranked_list: List[str], relevant_set: Set[str], k: int) -> float:
    """Fraction of all relevant items found in the top-K."""
    if not relevant_set:
        return 0.0
    top_k = ranked_list[:k]
    return sum(1 for doc in top_k if doc in relevant_set) / len(relevant_set)


def precision_at_k(ranked_list: List[str], relevant_set: Set[str], k: int) -> float:
    """Fraction of the top-K that are relevant."""
    top_k = ranked_list[:k]
    if not top_k:
        return 0.0
    return sum(1 for doc in top_k if doc in relevant_set) / len(top_k)


def reciprocal_rank_at_k(ranked_list: List[str], relevant_set: Set[str], k: int) -> float:
    """1/position of the first relevant hit within top K, or 0 if none."""
    for i, doc in enumerate(ranked_list[:k], start=1):
        if doc in relevant_set:
            return 1.0 / i
    return 0.0


def dcg_at_k(
    ranked_list: List[str],
    relevance_fn: Callable[[str], float],
    k: int,
) -> float:
    """Sum of relevance scores weighted by 1/log2(rank+1)."""
    dcg = 0.0
    for i, doc in enumerate(ranked_list[:k], start=1):
        rel = relevance_fn(doc)
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(
    ranked_list: List[str],
    relevance_fn: Callable[[str], float],
    all_relevant_docs: List[str],
    k: int,
) -> float:
    """DCG normalized by ideal DCG. Returns value in [0, 1]."""
    dcg_val = dcg_at_k(ranked_list, relevance_fn, k)
    # For binary relevance: ideal DCG = 1/log2(2) + 1/log2(3) + ... for min(|relevant|, k) terms
    n_ideal = min(len(all_relevant_docs), k)
    if n_ideal == 0:
        return 0.0
    idcg_val = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


def create_collection(
    documents_path: str,
    collection_name: str,
    persist_directory: str,
    embedding_model_name: str = "Qwen/Qwen3-Embedding-4B",
    expand_meta: bool = False,
    reset_collection: bool = False,
    max_docs: Optional[int] = None,
    dump_intermediate: bool = True,
) -> ChromaStore:
    if reset_collection:
        ChromaStore.reset_collection(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

    store = ChromaStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model_name=embedding_model_name,
    )

    existing_count = store.count()
    if (not expand_meta) and existing_count > 0:
        logger.info(f"Found Collection '{collection_name}', already has {existing_count} documents. Skipping ingestion.")
        return store
    
    logger.info("Preparing documents for ingestion...")
    dataset = prepare_quest_documents(
        jsonl_path=documents_path, index_first_512=True, max_docs=max_docs
    )

    docs, metas, data_rows = [], [], []
    for doc_item in dataset:
        text = doc_item["text"]
        meta = doc_item["metadata"]
        data_rows.append({"id": meta["entity_id"], "text": text})
        docs.append(text)
        metas.append(meta)

    if expand_meta:
        logger.info("Expanding metadata...")
        metas = expand_metas(
            metas,
            data_rows,
            min_frequency=3,
            dump_intermediate=dump_intermediate,
        )

    if docs:
        store.upsert_documents(documents=docs, metadatas=metas)
        logger.info(f"Ingested {len(docs)} documents into '{collection_name}'")

    return store


def evaluate_collection(
    store: ChromaStore,
    queries: List[QuestQuery],
    query_planner: Optional[LLMQueryPlanner] = None,
    output_path: Optional[str] = None,
    top_k: int = 20,
) -> float:
    if not queries:
        logger.warning("No queries to evaluate.")
        return 0.0

    total_recall = 0.0
    total_precision = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0
    f_out = (Path(output_path).parent.mkdir(parents=True, exist_ok=True) or open(output_path, "w", encoding="utf-8")) if output_path else None

    gold_set = set()
    relevance_fn = lambda d: 1.0 if d in gold_set else 0.0

    try:
        for i, q in enumerate(queries):
            logger.info(f"Evaluating query {i + 1}/{len(queries)}: {q.query}")
            where_clause = query_planner.plan(q.query) if query_planner else None
            results = store.query(q.query, n_results=top_k, where_filter=where_clause)
            # results = store.get(where=where_clause)
            if where_clause and len(results) == 0:
                logger.info(f"metadata filter did not return any results for query: {q.query}. Falling back to pure vector search.")
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
                        "score": result.get("distance"),
                    })

            gold_set.clear()
            gold_set.update(q.docs)
            rec = recall_at_k(predicted, gold_set, top_k)
            prec = precision_at_k(predicted, gold_set, top_k)
            rr = reciprocal_rank_at_k(predicted, gold_set, top_k)
            ndcg = ndcg_at_k(predicted, relevance_fn, q.docs, top_k)

            total_recall += rec
            total_precision += prec
            total_mrr += rr
            total_ndcg += ndcg

            if f_out:
                f_out.write(json.dumps({
                    "query_index": i,
                    "query": q.query,
                    f"recall@{top_k}": rec,
                    f"precision@{top_k}": prec,
                    f"mrr@{top_k}": rr,
                    f"ndcg@{top_k}": ndcg,
                    f"retrieved_top_{top_k}": retrieved_details,
                }) + "\n")
                f_out.flush()

            if i % 10 == 0:
                logger.info(f"Evaluated {i + 1}/{len(queries)} queries. Recall@{top_k}: {rec:.4f}")
    finally:
        if f_out:
            n = len(queries)
            avg_recall = total_recall / n
            avg_precision = total_precision / n
            avg_mrr = total_mrr / n
            avg_ndcg = total_ndcg / n
            f_out.write(json.dumps({
                f"Average Recall@{top_k}": avg_recall,
                f"Average Precision@{top_k}": avg_precision,
                f"Average MRR@{top_k}": avg_mrr,
                f"Average nDCG@{top_k}": avg_ndcg,
            }) + "\n")
            f_out.close()

    avg_recall = total_recall / len(queries)
    logger.info(f"Average Recall@{top_k}: {avg_recall:.4f}")
    if output_path:
        logger.info(f"Saved evaluation results to {output_path}")

    return avg_recall


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ontology-config",
        type=str,
        default=None,
        help="Optional override config YAML/JSON for low-confidence edge cases.",
    )
    parser.add_argument(
        "--profile-cache-mode",
        type=str,
        choices=["reuse", "refresh"],
        default="reuse",
        help="Profile cache strategy: reuse cached inferred profiles or refresh from Wikidata.",
    )
    parser.add_argument(
        "--kg-confidence-threshold",
        type=float,
        default=0.55,
        help="Confidence threshold (0-1) for applying KG-inferred profiles and hierarchy edges.",
    )
    parser.add_argument(
        "--dump-intermediate",
        action="store_true",
        default=True,
        help="Write step-wise sem_map intermediate JSON files and metrics.",
    )
    parser.add_argument("--full", action="store_true", help="Use full dataset (default: subset)")
    args = parser.parse_args()

    ontology_config_path = None
    if args.ontology_config:
        ontology_config_path = Path(args.ontology_config)
    elif os.environ.get("ONTOLOGY_CONFIG"):
        ontology_config_path = Path(os.environ["ONTOLOGY_CONFIG"])
    if ontology_config_path and not ontology_config_path.is_absolute():
        ontology_config_path = HERE / ontology_config_path

    # ── Normal mode: create collection + evaluate ────────────────────
    USE_SUBSET = not args.full

    if USE_SUBSET:
        documents_path = str(HERE / "tmp/subset_3_documents.jsonl")
        queries_source = str(HERE / "tmp/subset_3_quest_queries.jsonl")
        collection_suffix = "_subset_3"
        max_docs = None
    else:
        documents_path = str(HERE / "tmp/documents.jsonl")
        queries_source = "https://storage.googleapis.com/gresearch/quest/val.jsonl"
        collection_suffix = ""
        max_docs = 100

    queries = prepare_quest_queries(source=queries_source)
    embedding_model_name = "Qwen/Qwen3-Embedding-4B"
    
    # Base collection
    # store = create_collection(
    #     documents_path=documents_path,
    #     collection_name=f"quest_base{collection_suffix}",
    #     persist_directory=f"./chroma_collections_{embedding_model_name}",
    #     embedding_model_name=embedding_model_name,
    #     expand_meta=False,
    #     reset_collection=True,
    #     max_docs=max_docs,
    #     dump_intermediate=args.dump_intermediate,
    # )
    
    # avg_recall = evaluate_collection(
    #     store,
    #     queries,
    #     output_path=f"results_{embedding_model_name}/quest_eval_results_val_base{collection_suffix}.jsonl",
    # )
    # print(f"Average Recall (Base Collection): {avg_recall:.4f}")

    # # Expanded collection
    store_expanded = create_collection(
        documents_path=documents_path,
        collection_name=f"quest_expanded{collection_suffix}",
        persist_directory=f"./chroma_collections_{embedding_model_name}",
        embedding_model_name=embedding_model_name,
        expand_meta=True,
        reset_collection=True,
        max_docs=max_docs,
        dump_intermediate=args.dump_intermediate,
    )

    filter_catalog_path = HERE / "sem_map_subset_3/filter_catalog.json"
    query_planner = LLMQueryPlanner(filter_catalog_path) if filter_catalog_path.exists() else None

    avg_recall = evaluate_collection(
        store_expanded,
        queries,
        query_planner=query_planner,
        output_path=f"results_{embedding_model_name}/quest_eval_results_val_expanded{collection_suffix}.jsonl",
    )
    print(f"Average Recall (Expanded Collection): {avg_recall:.4f}")
