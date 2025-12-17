from __future__ import annotations
import json
import re
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import dspy
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("concept_generation_outputs")

def _parse_concept_list(raw: str) -> List[str]:
    """Parse a JSON array of strings; salvage common non-JSON formats."""
    if not isinstance(raw, str):
        return []

    raw = raw.strip()
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except Exception:
        pass

    if "[" in raw and "]" in raw:
        candidate = raw[raw.find("[") : raw.rfind("]") + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return parsed
        except Exception:
            pass

    numbered_pattern = r"\[\d+\]\s*[«\"]([^»\"]+)[»\"]"
    matches = re.findall(numbered_pattern, raw)
    if matches:
        return [m.strip() for m in matches]

    return [raw]


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """Case-insensitive dedupe preserving original order."""
    seen = set()
    result: List[str] = []
    for item in items:
        norm = item.strip().lower()
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        result.append(item.strip())
    return result


class PerQueryConceptSignature(dspy.Signature):
    """
    Generate mid-granularity concepts for a single query.
    """
    query = dspy.InputField(
        desc="Natural language search query."
    )
    concepts = dspy.OutputField(
        desc=(
            "Return ONLY a JSON array of strings (no prose before/after). "
            "Each string is ONE self-contained, Boolean-friendly concept."
        )
    )


class PerQueryConceptModel(dspy.Module):
    """
    LLM wrapper that maps a single query → per-query concept list.
    """

    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(PerQueryConceptSignature)

        # Optional: built-in few-shot examples (generic, not dataset-specific).
        self._few_shot_examples = [
            dspy.Example(
                query="Birds of Kolombangara or of the Western Province (Solomon Islands)",
                concepts='['
                         '"Birds of Kolombangara",'
                         '"Birds on the New Georgia Islands group",'
                         '"Birds of the Western Province (Solomon Islands)",'
                         '"Birds of the Solomon Islands"'
                         ']'
            ).with_inputs("query"),
            dspy.Example(
                query="Trees of South Africa that are also in the south-central Pacific",
                concepts='['
                         '"Trees of Africa",'
                         '"Trees of South Africa",'
                         '"Trees of the South-Central Pacific",'
                         '"Trees in the Pacific",'
                         '"Coastal trees"'
                         ']'
            ).with_inputs("query"),
            dspy.Example(
                query="2010s adventure films set in the Southwestern United States but not in California",
                concepts='['
                         '"Adventure films",'
                         '"2010s films",'
                         '"Films set in the Southwestern United States",'
                         '"Films set in the United States",'
                         '"Films set in California"'
                         ']'
            ).with_inputs("query"),
        ]

    def forward(self, query: str) -> List[str]:
        """Run the LLM and return a parsed list of concepts."""
        result = self._predict(query=query, demos=self._few_shot_examples)
        raw = getattr(result, "concepts", "") or ""
        return _parse_concept_list(raw)


class BatchFinalConceptSignature(dspy.Signature):
    """
    Directly generate final abstract concepts from a list of queries.
    """
    queries = dspy.InputField(
        desc="A list of natural language queries."
    )
    final_concepts = dspy.OutputField(
        desc="Return ONLY a JSON list of UNIQUE short noun phrases."
    )


class BatchFinalConceptModel(dspy.Module):
    """
    ONE-SHOT: list of queries → deduped list of final concepts.
    """

    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(BatchFinalConceptSignature)
        self._few_shot_examples = [
            dspy.Example(
                queries=[
                    "Birds of Kolombangara or of the Western Province (Solomon Islands)",
                    "Trees of South Africa that are also in the south-central Pacific",
                    "2010s adventure films set in the Southwestern United States but not in California",
                ],
                final_concepts=[
                    "bird location",
                    "plant location",
                    "film genre",
                    "film decade",
                    "film location",
                ],
            ).with_inputs("queries"),
        ]

    def forward(self, queries: List[str]) -> List[str]:
        """Run the LLM and return a parsed, deduped list of final concepts."""
        result = self._predict(queries=queries, demos=self._few_shot_examples)
        raw = getattr(result, "final_concepts", "") or ""
        parsed = _parse_concept_list(raw)
        return _dedupe_preserve_order(parsed)


class ClusterCentroidSignature(dspy.Signature):
    concepts = dspy.InputField(
        desc="A list of mid-granularity concept strings that belong to ONE cluster."
    )
    centroid = dspy.OutputField(
        desc=(
            "Return EXACTLY ONE short noun phrase of the form '<subject> <facet>'. "
            "The subject is a generic singular noun. "
            "The facet is the dominant *attribute type* expressed by the cluster "
            "(e.g., location, nationality, habitat, genre, tear). "
            "Do NOT output specific entities/places/years. "
            "Do NOT output a bare topic like 'fish' or 'criminal films'. "
            "Do NOT combine facets (no 'and', no commas)."
        )
    )


class ClusterCentroidModel(dspy.Module):
    """
    LLM wrapper: cluster of concepts → centroid label.
    """

    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(ClusterCentroidSignature)
        self._few_shot_examples = [
            dspy.Example(
                concepts=[
                    "Birds of the Pacific Islands",
                    "Birds of North America",
                    "Birds found in Central Africa",
                ],
                centroid="bird location",
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=[
                    "Horror films",
                    "Historical films",
                    "Films set in the future",
                    "Black-and-white films",
                ],
                centroid="film genre",
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=[
                    "1990s films",
                    "Early 1960s films",
                    "Late 1990s films",
                ],
                centroid="film decade",
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=[
                    "2002 films",
                    "films released in 2024",
                    "films shot in 1999",
                ],
                centroid="film year",
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=[
                    "Flowers of the Crozet Islands",
                    "Trees of the Marshall Islands",
                    "Trees of the Line Islands",
                ],
                centroid="plant location",
            ).with_inputs("concepts"),
        ]

    def forward(self, concepts: List[str]) -> str:
        """Run the LLM and return a centroid label."""
        concepts_str = "\n".join(f"- {c}" for c in concepts)
        result = self._predict(concepts=concepts_str, demos=self._few_shot_examples)
        centroid = (getattr(result, "centroid", "") or "").strip()
        return centroid


class ConceptGenerationMode(Enum):
    TWO_STAGE = "two_stage"
    DIRECT = "direct"


class LLMConceptGenerator:
    """
    Minimal concept-generation API with exactly two paths:
    - DIRECT: queries -> final concepts
    - TWO_STAGE: per-query concepts -> clustering -> centroid labels
    """

    def __init__(self, config: Any) -> None:
        self.config = config

        mode_str = getattr(config, "concept_generation_mode", ConceptGenerationMode.TWO_STAGE.value)
        try:
            self.mode = ConceptGenerationMode(mode_str)
        except ValueError:
            self.mode = ConceptGenerationMode.TWO_STAGE

        self.n_clusters = getattr(config, "concept_cluster_count", 50)
        self.embedding_model_name = getattr(config, "concept_embedding_model", "all-MiniLM-L6-v2")

        self._per_query_model = PerQueryConceptModel()
        self._batch_final_model = BatchFinalConceptModel()
        self._centroid_model = ClusterCentroidModel()

        self._concept_vocabulary: List[str] = []

    def fit(
        self,
        queries: Iterable[str],
        *,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        queries_list = list(queries)
        logger.info(
            f"LLMConceptGenerator: fitting on {len(queries_list)} queries (mode={self.mode})."
        )

        if not queries_list:
            self._concept_vocabulary = []
            return

        if self.mode is ConceptGenerationMode.TWO_STAGE:
            concepts, artifacts = self._fit_two_stage(queries_list)
        else:
            concepts, artifacts = self._fit_direct(queries_list)

        self._concept_vocabulary = concepts
        logger.info(f"LLMConceptGenerator: learned {len(concepts)} concepts.")

        if output_dir is None:
            output_dir = (
                getattr(self.config, "concept_generation_output_dir", None)
                or getattr(self.config, "concept_output_dir", None)
                or DEFAULT_OUTPUT_DIR
            )
        self._persist(output_dir=output_dir, artifacts=artifacts)

    def _persist(self, *, output_dir: str | Path, artifacts: Mapping[str, Any]) -> None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "concept_generation_artifacts.json"
        try:
            # Keep insertion order for readability (e.g., clusters in numeric order).
            out_path.write_text(json.dumps(artifacts, indent=2, sort_keys=False))
        except Exception as e:
            logger.warning(f"Failed to write concept generation artifacts to {out_path}: {e}")

    def _fit_two_stage(self, queries: List[str]) -> tuple[List[str], Dict[str, Any]]:
        """
        TWO-STAGE STRATEGY:
        1. LLM generates per-query intermediate concepts.
        2. Concepts are embedded and clustered.
        3. LLM generates a centroid (final concept) for each cluster.
        """
        artifacts: Dict[str, Any] = {
            "mode": ConceptGenerationMode.TWO_STAGE.value,
            "n_clusters": int(self.n_clusters),
            "embedding_model_name": self.embedding_model_name,
        }

        all_concepts: List[str] = []
        per_query: Dict[str, List[str]] = {}
        for query in tqdm(queries, desc="two_stage per-query"):
            per_query_concepts = self._per_query_model(query)
            per_query[query] = list(per_query_concepts)
            all_concepts.extend(per_query_concepts)

        all_concepts = _dedupe_preserve_order(all_concepts)
        logger.info(f"LLMConceptGenerator: generated {len(all_concepts)} intermediate concepts.")
        if not all_concepts:
            artifacts["per_query_concepts"] = per_query
            artifacts["intermediate_concepts"] = []
            artifacts["clusters"] = {}
            artifacts["final_concepts"] = []
            return [], artifacts

        model = SentenceTransformer(self.embedding_model_name)
        embeddings = model.encode(all_concepts, show_progress_bar=False)

        n_clusters = min(self.n_clusters, len(all_concepts))
        logger.info(f"LLMConceptGenerator: clustering into {n_clusters} clusters.")
        if n_clusters <= 0:
            artifacts["per_query_concepts"] = per_query
            artifacts["intermediate_concepts"] = all_concepts
            artifacts["clusters"] = {}
            artifacts["final_concepts"] = []
            return [], artifacts

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        clusters: Mapping[int, List[str]] = {}
        for concept, label in zip(all_concepts, labels):
            clusters.setdefault(int(label), []).append(concept)  # type: ignore[attr-defined]

        cluster_ids = sorted(clusters.keys())
        cluster_centroids: Dict[str, str] = {}
        centroids_in_cluster_order: List[str] = []
        for cluster_id in tqdm(cluster_ids, desc="two_stage centroids"):
            members = clusters[cluster_id]
            if not members:
                continue
            centroid = self._centroid_model(members)
            if centroid:
                cluster_centroids[str(cluster_id)] = centroid
                centroids_in_cluster_order.append(centroid)

        final_concepts = _dedupe_preserve_order(centroids_in_cluster_order)

        artifacts["per_query_concepts"] = per_query
        artifacts["intermediate_concepts"] = all_concepts
        # Preserve numeric ordering for cluster ids in the JSON file.
        artifacts["clusters"] = {str(k): clusters[k] for k in cluster_ids}
        artifacts["cluster_centroids"] = cluster_centroids
        artifacts["final_concepts"] = final_concepts
        return final_concepts, artifacts

    def _fit_direct(self, queries: List[str]) -> tuple[List[str], Dict[str, Any]]:
        """
        DIRECT STRATEGY:
        Generate final concepts in batches and globally dedupe.
        """
        batch_size = int(getattr(self.config, "concept_direct_batch_size", 100) or 0)

        global_seen: set[str] = set()
        global_concepts: List[str] = []
        batch_results: List[Dict[str, Any]] = []

        n = len(queries)
        if batch_size <= 0:
            batch_size = n
        num_batches = (n + batch_size - 1) // batch_size if batch_size else 0

        for batch_idx in tqdm(range(num_batches), desc="direct batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n)
            batch_queries = queries[start_idx:end_idx]

            tqdm.write(f"direct batch {batch_idx + 1}/{num_batches}: queries[{start_idx}:{end_idx}]")
            batch_concepts = self._batch_final_model(batch_queries)

            new_concepts: List[str] = []
            for concept in batch_concepts:
                norm = concept.strip().lower()
                if not norm:
                    continue
                if norm in global_seen:
                    continue
                global_seen.add(norm)
                kept = concept.strip()
                global_concepts.append(kept)
                new_concepts.append(kept)

            batch_results.append(
                {
                    "batch_idx": batch_idx,
                    "raw_concepts": batch_concepts,
                    "new_concepts": new_concepts,
                    "num_new_concepts": len(new_concepts),
                }
            )

        artifacts: Dict[str, Any] = {
            "mode": ConceptGenerationMode.DIRECT.value,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "batches": batch_results,
            "final_concepts": global_concepts,
        }
        return global_concepts, artifacts


# Example usage:
if __name__ == "__main__":
    import os
    import requests

    print("Downloading queries...")
    url = "https://storage.googleapis.com/gresearch/quest/train.jsonl"
    lines = requests.get(url).content.decode("utf-8").strip().split("\n")
    queries = [json.loads(line).get("query", "") for line in lines if line.strip()]
    queries = [q for q in queries if q]
    print(f"✅ Queries downloaded, {len(queries)} queries found")
    
    api_key = os.environ.get("OPENAI_API_KEY", "")
    lm = dspy.LM("openai/gpt-5.1", temperature=1.0, max_tokens=16000, api_key=api_key)
    dspy.configure(lm=lm)

    class _Cfg:
        pass
    
    
    cfg_direct = _Cfg()
    cfg_direct.concept_generation_mode = ConceptGenerationMode.DIRECT.value
    cfg_direct.concept_direct_batch_size = 100

    cfg_two_stage = _Cfg()
    cfg_two_stage.concept_generation_mode = ConceptGenerationMode.TWO_STAGE.value
    cfg_two_stage.concept_cluster_count = 50
    cfg_two_stage.concept_embedding_model = "all-MiniLM-L6-v2"

    # print("Fitting DIRECT model...")
    # LLMConceptGenerator(cfg_direct).fit(queries, output_dir="results/quest_concepts/direct")
    # print("✅ DIRECT model fitted")
    
    print("[DEFAULT] Fitting TWO_STAGE model...")
    LLMConceptGenerator(cfg_two_stage).fit(queries, output_dir="results/quest_concepts/two_stage")
    print("[DEFAULT] ✅ TWO_STAGE model fitted")
    
    print("[DEFAULT] Wrote TWO_STAGE artifacts to results/quest_concepts/two_stage/concept_generation_artifacts.json")