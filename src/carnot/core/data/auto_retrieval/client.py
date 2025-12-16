from __future__ import annotations
from typing import List, Dict, Any, Optional
import logging

import json
import os
import dspy

from .config import Config, load_config
from .types import Query, SearchResult
from .quest_data_prep import prepare_quest_documents
from ._internal.index_management import IndexManagementPipeline
from ._internal.query_planning import QueryPlanner, LogicalPlan
from ._internal.query_optimization import QueryOptimizer, PhysicalPlan
from ._internal.execution import QueryExecutor


def _load_concepts_from_json(path: str) -> List[str]:
    """Helper to load 'centroid' fields from the concept clustering JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    
    return sorted(list(set(data)))


class FilterGeneratorSignature(dspy.Signature):
    query: str = dspy.InputField(
        desc="User's search query in natural language."
    )

    concept_schema: str = dspy.InputField(
        desc=(
            "JSON string mapping concept fields to their schema, e.g. "
            '{"concept:film_genre":{"type":"string","allowed_values":["action","drama"]}, ...}. '
            "Use only these fields when building filters."
        )
    )

    where_json: str = dspy.OutputField(
        desc=(
            "Return ONLY a JSON object string (no prose, no backticks) for a ChromaDB `where` metadata filter.\n"
            "The top-level must be an object; metadata keys must come from concept_schema and must not start with '$'.\n"
            "Use only operators $eq,$ne,$gt,$gte,$lt,$lte,$in,$nin,$and,$or. Arrays are allowed only as values of $in/$nin.\n"
            "DO NOT use document filters or operators like $contains or $not_contains.\n"
            "Return {} if no filter is appropriate. \n"
        )
    )


class SearchClient:
    """Facade that exposes a clean search() API over the internal stack."""

    def __init__(
        self,
        config: Config,
        index_pipeline: IndexManagementPipeline,
        planner: QueryPlanner,
        optimizer: QueryOptimizer,
        executor: QueryExecutor,
    ) -> None:
        """Initialize a SearchClient with all internal components."""
        self._config = config
        self._index_pipeline = index_pipeline
        self._planner = planner
        self._optimizer = optimizer
        self._executor = executor
        
        lm = dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=16000, api_key="")
        dspy.configure(lm=lm)
        self._filter_gen = dspy.Predict(FilterGeneratorSignature)

    @classmethod
    def from_config(cls, config_path: str) -> "SearchClient":
        """Construct a SearchClient from a config file."""
        config = load_config(config_path)
        index_pipeline = IndexManagementPipeline.from_config(config)
        planner = QueryPlanner.from_config(config, index_pipeline)
        optimizer = QueryOptimizer.from_config(config, index_pipeline)
        executor = QueryExecutor(index_pipeline)
        return cls(
            config=config,
            index_pipeline=index_pipeline,
            planner=planner,
            optimizer=optimizer,
            executor=executor,
        )

    def search(
        self,
        text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        concept_filters: bool = False,
    ) -> List[SearchResult]:
        vector_index = self._index_pipeline.get_vector_index()

        if filters is None and concept_filters:
            concept_schema: Dict[str, Any] = {}
            if os.path.exists("metadata_concepts_schema.json"):
                try:
                    with open("metadata_concepts_schema.json", "r") as f:
                        concept_schema = json.load(f)
                except Exception as e:
                    logging.warning(f"Failed to load concept schema: {e}")
                    concept_schema = {}

            if concept_schema:
                try:
                    prediction = self._filter_gen(
                        query=text,
                        concept_schema=json.dumps(concept_schema),
                    )

                    raw_filter = getattr(prediction, "where_json", "{}")
                    generated_filters = json.loads(raw_filter)

                    if isinstance(generated_filters, dict) and generated_filters:
                        filters = generated_filters
                        logging.debug(f"Generated concept filters: {filters}")
                except Exception as e:
                    logging.warning(f"Filter generation failed, falling back to no filters: {e}")
                    filters = None

        if hasattr(vector_index, "_embed_fn"):
            query_embedding = vector_index._embed_fn([text])[0]
        else:
            raise NotImplementedError(
                "Vector index does not expose an embedding function."
            )

        result = vector_index.query(
            query_embedding,
            top_k=top_k,
            filters=filters,
            include=("documents", "metadatas", "distances"),
        )

        res_ids = result.ids or []
        res_docs = result.documents or []
        res_metas = result.metadatas or []
        res_dists = result.distances or []

        out: List[SearchResult] = []
        for i in range(len(res_ids)):
            meta = dict(res_metas[i]) if i < len(res_metas) and res_metas[i] else {}
            if "text" not in meta and i < len(res_docs):
                meta["text"] = res_docs[i]

            score = res_dists[i] if i < len(res_dists) else 0.0
            out.append(
                SearchResult(
                    doc_id=res_ids[i],
                    score=score,
                    metadata=meta,
                )
            )
        return out

    def ingest_dataset(self, path: str) -> None:
        """
        Load documents from a path and ingest it into the index pipeline. This prepares the documents and adds them to the Chroma collection.
        """

        docs = prepare_quest_documents(
            jsonl_path=path,
            tokenizer_model=self._config.tokenizer_model,
            index_first_512=self._config.index_first_512,
            chunk_size=self._config.chunk_size,
            overlap=self._config.overlap,
        )
        reset = self._config.clear_chroma_collection
        self._index_pipeline.add_documents(docs, reset=reset)
        
    
    def enrich_documents(self, queries: Optional[List[str]] = None, concepts_path: Optional[str] = None) -> None:
        """
        Enrich the documents with the concept fields.
        """
        concept_vocabulary = []
        if concepts_path:
            concept_vocabulary = _load_concepts_from_json(concepts_path)
        elif queries:
            concept_vocabulary = self._index_pipeline.concept_generator.generate_from_queries(queries=queries)
        self._index_pipeline.enrich_documents(concept_vocabulary=concept_vocabulary)
