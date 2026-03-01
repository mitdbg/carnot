from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from _internal.chroma_store import ChromaStore
from _internal.query_planner import LLMQueryPlanner
from _internal.sem_map import expand_sem_map_results_to_tags
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
    ontology_config_path: Optional[Path] = None,
    dump_intermediate: Optional[bool] = None,
    profile_cache_mode: str = "reuse",
    kg_confidence_threshold: float = 0.55,
) -> List[Dict[str, Any]]:
    """Run the full post-processing pipeline on sem_map results.

    Transforms raw per-document concept extractions into stable, queryable
    metadata columns suitable for ChromaDB filtering.
    """
    out_dir = HERE / "sem_map"
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
            (HERE / "sem_map/concept_schema_cols.json").read_text(encoding="utf-8")
        )
    ]

    # Load augmented sem_map output (output of postprocess Steps 1–3).
    # logger.info("Postprocessing sem_map raw output...")
    # sem_results = postprocess_sem_map(
    #     sem_results,
    #     concept_schema_cols,
    #     out_dir=str(out_dir) if dump_intermediate else None
    #     )
    augmented_path = HERE / "sem_map/postprocess_step3_augmented.json"
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


def recall(predicted: List[str], gold: List[str]) -> float:
    if not gold or not predicted:
        return 0.0
    return len(set(predicted) & set(gold)) / len(gold)


def create_collection(
    documents_path: str,
    collection_name: str,
    persist_directory: str,
    expand_meta: bool = False,
    max_docs: Optional[int] = None,
    ontology_config_path: Optional[Path] = None,
    profile_cache_mode: str = "reuse",
    kg_confidence_threshold: float = 0.55,
    dump_intermediate: bool = True,
) -> ChromaStore:
    if expand_meta:
        ChromaStore.reset_collection(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

    store = ChromaStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
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
            ontology_config_path=ontology_config_path,
            dump_intermediate=dump_intermediate,
            profile_cache_mode=profile_cache_mode,
            kg_confidence_threshold=kg_confidence_threshold,
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
    f_out = open(output_path, "w", encoding="utf-8") if output_path else None

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

            score = recall(predicted, q.docs)
            total_recall += score

            if f_out:
                f_out.write(json.dumps({
                    "query_index": i,
                    "query": q.query,
                    f"recall@{top_k}": score,
                    f"retrieved_top_{top_k}": retrieved_details,
                }) + "\n")
                f_out.flush()

            if i % 10 == 0:
                logger.info(f"Evaluated {i + 1}/{len(queries)} queries. Recall@{top_k}: {score:.4f}")
    finally:
        if f_out:
            avg = total_recall / (i + 1)
            f_out.write(json.dumps({f"Average Recall@{top_k}": avg}) + "\n")
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
        documents_path = str(HERE / "tmp/subset_documents.jsonl")
        queries_source = str(HERE / "tmp/subset_quest_queries.jsonl")
        collection_suffix = "_subset"
        max_docs = None
    else:
        documents_path = str(HERE / "tmp/documents.jsonl")
        queries_source = "https://storage.googleapis.com/gresearch/quest/val.jsonl"
        collection_suffix = ""
        max_docs = 100

    store = create_collection(
        documents_path=documents_path,
        collection_name=f"quest_expanded{collection_suffix}",
        persist_directory="./chroma_collections",
        expand_meta=False,
        max_docs=max_docs,
        ontology_config_path=ontology_config_path,
        profile_cache_mode=args.profile_cache_mode,
        kg_confidence_threshold=args.kg_confidence_threshold,
        dump_intermediate=args.dump_intermediate,
    )

    queries = prepare_quest_queries(source=queries_source)
    filter_catalog_path = HERE / "sem_map/filter_catalog.json"
    query_planner = LLMQueryPlanner(filter_catalog_path) if filter_catalog_path.exists() else None

    avg_recall = evaluate_collection(
        store,
        queries,
        query_planner=query_planner,
        output_path=f"quest_eval_results_val_expanded{collection_suffix}.jsonl",
    )
    print(f"Average Recall (Expanded Collection): {avg_recall:.4f}")
