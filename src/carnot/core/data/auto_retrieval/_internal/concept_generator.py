from __future__ import annotations
from dataclasses import dataclass
import json
import re
import logging
from enum import Enum
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union, Set

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import dspy
import palimpzest as pz

from ..config import Config

SCHEMA_PATH = "metadata_concepts_schema.json"

logger = logging.getLogger(__name__)


def _parse_concept_list(raw: str) -> List[str]:
    """Parse a JSON array of strings; attempt simple salvage if extra text appears."""
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

    # Heuristic salvage: extract first [...] and parse
    if "[" in raw and "]" in raw:
        candidate = raw[raw.find("["): raw.rfind("]") + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return parsed
        except Exception:
            pass

    # Numbered list format: [1] «concept» or [1] "concept"
    numbered_pattern = r"\[\d+\]\s*[«\"]([^»\"]+)[»\"]"
    matches = re.findall(numbered_pattern, raw)
    if matches:
        return [m.strip() for m in matches]

    # Fallback: treat whole string as one concept
    return [raw]


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """Case-insensitive deduplication while preserving original order."""
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
        desc="Natural language search query (may include set logic or filters)."
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
                         '"Vertebrates of Kolombangara",'
                         '"Birds on the New Georgia Islands group",'
                         '"Vertebrates of the Western Province (Solomon Islands)",'
                         '"Birds of the Solomon Islands"'
                         ']'
            ).with_inputs("query"),
            dspy.Example(
                query="Trees of South Africa that are also in the south-central Pacific",
                concepts='['
                         '"Trees of Africa",'
                         '"Flora of South Africa",'
                         '"Flora of the South-Central Pacific",'
                         '"Trees in the Pacific",'
                         '"Coastal trees"'
                         ']'
            ).with_inputs("query"),
            dspy.Example(
                query="2010s adventure films set in the Southwestern United States but not in California",
                concepts='['
                         '"Adventure films",'
                         '"2010s films",'
                         '"Films set in the U.S. Southwest",'
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
        desc="A list of natural language queries with implicit set operations."
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
                    "bird geographic distribution",
                    "plant geographic distribution",
                    "film genre",
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
    """
    Generate a compact centroid (short noun phrase) for a cluster of related concepts.
    """
    concepts = dspy.InputField(
        desc="A list of short concept strings that belong to ONE semantic cluster."
    )
    centroid = dspy.OutputField(
        desc=(
            "Return ONLY a SINGLE, short, singular noun phrase (2-5 words) "
            "describing the cluster."
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
                centroid="avian geographic region",
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=[
                    "Horror films",
                    "Historical films",
                    "Films set in the future",
                    "Black-and-white films",
                ],
                centroid="film genre or style",
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=[
                    "1990s films",
                    "1988 films",
                    "Films released in 1975",
                    "Early 1960s films",
                ],
                centroid="film release period",
            ).with_inputs("concepts"),
        ]

    def forward(self, concepts: List[str]) -> str:
        """Run the LLM and return a centroid label."""
        # Format as a bullet list for nicer prompting
        concepts_str = "\n".join(f"- {c}" for c in concepts)
        result = self._predict(concepts=concepts_str, demos=self._few_shot_examples)
        centroid = (getattr(result, "centroid", "") or "").strip()
        return centroid


class BatchConceptSchemaSignature(dspy.Signature):
    """
    Infer data schema for a list of concepts.
    """
    concepts = dspy.InputField(
        desc="A list of concept names."
    )
    schemas = dspy.OutputField(
        desc=(
            "Return ONLY a JSON object mapping each concept to its schema details.\n"
            "Each schema object must have:\n"
            "- 'type': One of ['categorical', 'int', 'float', 'hierarchy'].\n"
            "  * 'categorical': for tag-like fields (e.g. genre, nationality) that should be boolean flags.\n"
            "  * 'int'/'float': for numeric fields (e.g. year, rating) that should be scalar or list values.\n"
            "  * 'hierarchy': for hierarchical fields requiring level normalization.\n"
            "- 'desc': A concise description of the field.\n"
            "- 'levels': (Optional) If type is 'hierarchy', providing a list of level names.\n"
            "Determine the levels based on the concept (e.g. location vs organizational structure).\n"
        )
    )


class BatchConceptSchemaModel(dspy.Module):
    """
    LLM wrapper: list of concepts → dictionary of schemas.
    """

    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(BatchConceptSchemaSignature)
        self._few_shot_examples = [
            dspy.Example(
                concepts=[
                    "film genre",
                    "release year",
                    "average rating",
                    "is blockbuster",
                    "filming location"
                ],
                schemas='{\n'
                        '  "film genre": {"type": "categorical", "desc": "Genre of the film"},\n'
                        '  "release year": {"type": "int", "desc": "Year of release"},\n'
                        '  "average rating": {"type": "float", "desc": "Rating out of 10"},\n'
                        '  "is blockbuster": {"type": "categorical", "desc": "Whether it was a major commercial success"},\n'
                        '  "filming location": {"type": "hierarchy", "desc": "Where the film was shot", "levels": ["City", "State / Province", "Country", "Region / Subcontinent", "Continent"]}\n'
                        '}'
            ).with_inputs("concepts"),
        ]

    def forward(self, concepts: List[str]) -> Dict[str, Any]:
        """Run the LLM and return a mapping of concept -> schema."""
        concepts_str = "\n".join(f"- {c}" for c in concepts)
        result = self._predict(concepts=concepts_str, demos=self._few_shot_examples)
        raw = getattr(result, "schemas", "") or ""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {}


class CanonicalizeValuesSignature(dspy.Signature):
    """
    Canonicalize and deduplicate a list of raw entity values.
    """
    concept_name = dspy.InputField(desc="The name of the concept these values belong to.")
    raw_values = dspy.InputField(desc="A list of raw string values extracted from text.")
    canonical_map = dspy.OutputField(
        desc=(
            "Return ONLY a JSON object mapping each UNIQUE raw value to its canonical form. "
            "Merge synonyms, fix typos, and standardize formatting. "
            "Example: {'NYC': 'New York City', 'Manhattan, New York': 'New York City'}"
        )
    )


class CanonicalizeValuesModel(dspy.Module):
    """
    LLM wrapper: list of raw values → canonical map.
    """
    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(CanonicalizeValuesSignature)
    
    def forward(self, concept_name: str, raw_values: List[str]) -> Dict[str, str]:
        if not raw_values:
            return {}
        
        # Batching could be added here if list is huge, but assuming reasonable size per batch
        raw_values_str = json.dumps(list(set(raw_values))) # distinct input only
        result = self._predict(concept_name=concept_name, raw_values=raw_values_str)
        raw_output = getattr(result, "canonical_map", "{}")
        
        try:
            mapping = json.loads(raw_output)
            if isinstance(mapping, dict):
                 return {k: str(v) for k, v in mapping.items()}
        except Exception:
            pass
        
        # Fallback: identity mapping
        return {v: v for v in raw_values}


class ConceptGenerationMode(Enum):
    """Supported concept generation strategies."""

    TWO_STAGE = "two_stage"   # per-query → cluster → centroid (default)
    DIRECT = "direct"         # direct concepts from list of queries


@dataclass
class LLMConceptGenerator:
    """
    LLM-based component that learns and assigns workload-specific concepts.

    This wraps two strategies:
    - TWO_STAGE (default): per-query intermediate concepts → clustering →
      centroid labels (final concepts).
    - DIRECT: a single LLM pass over the full query list to get final concepts.
    """

    config: Config
    mode: ConceptGenerationMode = ConceptGenerationMode.DIRECT
    n_clusters: int = 50
    embedding_model_name: str = "all-MiniLM-L6-v2"

    def __init__(self, config: Config) -> None:
        """Initialize the concept generator from configuration."""
        # Extract mode and hyperparameters from config if present
        mode_str = getattr(config, "concept_generation_mode", ConceptGenerationMode.TWO_STAGE.value)
        try:
            mode = ConceptGenerationMode(mode_str)
        except ValueError:
            mode = ConceptGenerationMode.TWO_STAGE

        n_clusters = getattr(config, "concept_cluster_count", 50)
        embedding_model_name = getattr(config, "concept_embedding_model", "all-MiniLM-L6-v2")

        # dataclass-style manual initialization
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "n_clusters", n_clusters)
        object.__setattr__(self, "embedding_model_name", embedding_model_name)

        # LLM-based submodules
        object.__setattr__(self, "_per_query_model", PerQueryConceptModel())
        object.__setattr__(self, "_batch_final_model", BatchFinalConceptModel())
        object.__setattr__(self, "_centroid_model", ClusterCentroidModel())
        object.__setattr__(self, "_schema_inference_model", BatchConceptSchemaModel())
        object.__setattr__(self, "_canonicalize_model", CanonicalizeValuesModel())

        # Learned vocabulary of final concepts
        object.__setattr__(self, "_concept_vocabulary", [])  # type: ignore[var-annotated]

    def fit(
        self,
        queries: Iterable[str],
    ) -> None:
        """
        Infer a workload-specific vocabulary of semantic concepts.
        """
        logger.info(f"LLMConceptGenerator: fitting on {len(queries)} queries (mode={self.mode}).")

        if not queries:
            object.__setattr__(self, "_concept_vocabulary", [])
            return

        if self.mode is ConceptGenerationMode.TWO_STAGE:
            concepts = self._fit_two_stage(queries)
        else:
            concepts = self._fit_direct(queries)

        logger.info(f"LLMConceptGenerator: learned {len(concepts)} concepts.")
        object.__setattr__(self, "_concept_vocabulary", concepts)

    def concept_map(
        self,
        docs: Iterable[Mapping[str, Any]],
        concept_vocabulary: Optional[List[str]] = None,
        concept_schemas: Optional[Dict[str, Any]] = None,
        backfill_false: bool = True,
    ) -> Mapping[str, Mapping[str, Any]]:
        """
        Given docs and a (learned) concept vocabulary, extract per-doc values
        for each concept using Palimpzest and return a mapping:
            doc_id -> { "concept:<domain>:<attr>[:<val>]": <val|True|False> }

        Features:
        - Extracts LISTs of values per concept.
        - Canonicalizes string values (dedupe/normalize).
        - Maps to boolean keys for categorical/hierarchy.
        - Maps to scalar/list keys for numeric.
        - **Dense Boolean Mode (backfill_false=True)**:
          If True, any known boolean concept key (from schema or current batch)
          that is NOT present in a document will be explicitly set to False.
        """
        if concept_vocabulary is None:
            concept_vocabulary = self._concept_vocabulary
        if not concept_vocabulary:
            logger.info("LLMConceptGenerator.concept_map: empty concept vocabulary; nothing to do.")
            return {}

        docs_list: List[Mapping[str, Any]] = list(docs)
        if not docs_list:
            logger.info("LLMConceptGenerator.concept_map: no docs provided.")
            return {}

        pz_rows: List[Dict[str, Any]] = []
        for d in docs_list:
            doc_id = str(d.get("id"))
            if not doc_id:
                continue

            text = d.get("text")
            if not text:
                continue

            pz_rows.append({"id": doc_id, "text": text})

        if not pz_rows:
            logger.info("LLMConceptGenerator.concept_map: no docs with text.")
            return {}

        concept_schemas = self._infer_concept_schemas(concept_vocabulary) if concept_schemas is None else concept_schemas

        cols_spec: List[Dict[str, Any]] = []
        col_mapping: Dict[str, Tuple[str, Dict[str, Any]]] = {}

        # 1. Build PZ Schema
        for concept in concept_vocabulary:
            schema = concept_schemas.get(concept, {"type": "categorical"})
            stype = schema.get("type", "categorical")
            desc_text = schema.get("desc", f"The {concept} mentioned in the text.")
            base_col = f"c_{abs(hash(concept))}" 
            
            if stype == "hierarchy":
                levels = schema.get("levels", []) or ["Level1"]
                for lvl in levels:
                    safe_lvl = re.sub(r'\W+', '_', str(lvl)).lower()
                    pz_col = f"{base_col}_{safe_lvl}"
                    desc = f"List of {lvl}s for {concept} mentioned in the text. {desc_text}"
                    cols_spec.append({"name": pz_col, "type": List[str], "desc": desc})
                    col_mapping[pz_col] = (concept, {"type": "hierarchy", "level": str(lvl)})

            elif stype == "int":
                pz_col = f"{base_col}_int"
                # Always ask for a List to handle potential multi-values (e.g. Olympic years),
                # but we'll store as scalar if len==1 during post-processing.
                cols_spec.append({"name": pz_col, "type": List[int], "desc": f"List of {desc_text}"})
                col_mapping[pz_col] = (concept, {"type": "int"})

            elif stype == "float":
                pz_col = f"{base_col}_float"
                cols_spec.append({"name": pz_col, "type": List[float], "desc": f"List of {desc_text}"})
                col_mapping[pz_col] = (concept, {"type": "float"})

            else:
                # Default to categorical (string list) -> Boolean Tags
                pz_col = f"{base_col}_cat"
                desc = f"List of values for {concept}. {desc_text}"
                cols_spec.append({"name": pz_col, "type": List[str], "desc": desc})
                col_mapping[pz_col] = (concept, {"type": "categorical"})

        # 2. Run Palimpzest Extraction
        dataset = pz.MemoryDataset(id="concept-assignment", vals=pz_rows)
        dataset = dataset.sem_map(cols=cols_spec)
        output = dataset.run(max_quality=True)
        df = output.to_df()

        # 3. Post-processing & Canonicalization
        raw_values_by_concept: Dict[str, List[str]] = {}
        doc_raw_extractions: Dict[str, Dict[str, List[Any]]] = {}

        for _, row in df.iterrows():
            doc_id = str(row.get("id"))
            if not doc_id:
                continue
            doc_raw_extractions[doc_id] = {}
            for pz_col, raw_val in row.items():
                if pz_col not in col_mapping:
                    continue
                vals = raw_val if isinstance(raw_val, list) else [raw_val]
                vals = [v for v in vals if v is not None and v != ""]
                if not vals:
                    continue
                
                doc_raw_extractions[doc_id][pz_col] = vals
                concept, info = col_mapping[pz_col]
                stype = info["type"]
                
                if stype in ("categorical", "hierarchy"):
                    key = f"{concept}::{info.get('level', 'root')}"
                    raw_values_by_concept.setdefault(key, []).extend([str(v) for v in vals])

        # 4. Canonicalize
        canonical_maps: Dict[str, Dict[str, str]] = {}
        for key, all_raw in raw_values_by_concept.items():
            if not all_raw: continue
            concept_name, _ = key.split("::", 1)
            unique_raw = list(set(all_raw))
            if unique_raw:
                 mapping = self._canonicalize_model(concept_name=concept_name, raw_values=unique_raw)
                 canonical_maps[key] = mapping

        # 5. Build Initial True Assignments
        # doc_id -> key -> value
        assignments: Dict[str, Dict[str, Any]] = {}
        
        # Track all boolean keys seen/known for each concept prefix
        # Prefix -> Set of known keys (e.g. concept:film:genre -> { ...:Action, ...:Comedy })
        known_boolean_keys: Dict[str, Set[str]] = {}

        # 5a. Load existing keys from schema to ensure continuity
        current_schema = self._load_metadata_concepts_schema()
        for k, v in current_schema.items():
            if v.get("type") == "bool":
                # Find prefix by stripping last component
                if ":" in k:
                    prefix = k.rsplit(":", 1)[0]
                    known_boolean_keys.setdefault(prefix, set()).add(k)

        # 5b. Generate Assignments for current batch
        for doc_id, col_data in doc_raw_extractions.items():
            concept_attrs: Dict[str, Any] = {}
            for pz_col, vals in col_data.items():
                concept, info = col_mapping[pz_col]
                domain_attr = self._normalize_concept_key(concept)
                stype = info["type"]

                if stype in ("int", "float"):
                    key = f"concept:{domain_attr}"
                    clean_vals = []
                    for v in vals:
                        try:
                            clean_vals.append(int(v) if stype == "int" else float(v))
                        except (ValueError, TypeError):
                            pass
                    if clean_vals:
                        concept_attrs[key] = clean_vals[0] if len(clean_vals) == 1 else clean_vals

                elif stype == "categorical":
                    key_group = f"{concept}::root"
                    mapping = canonical_maps.get(key_group, {})
                    prefix = f"concept:{domain_attr}"
                    
                    for v in vals:
                        canon = mapping.get(str(v).strip(), str(v).strip())
                        if canon:
                             final_key = f"{prefix}:{canon}"
                             concept_attrs[final_key] = True
                             known_boolean_keys.setdefault(prefix, set()).add(final_key)

                elif stype == "hierarchy":
                    level = info["level"]
                    key_group = f"{concept}::{level}"
                    mapping = canonical_maps.get(key_group, {})
                    prefix = f"concept:{domain_attr}:{level.lower()}"

                    for v in vals:
                        canon = mapping.get(str(v).strip(), str(v).strip())
                        if canon:
                            final_key = f"{prefix}:{canon}"
                            concept_attrs[final_key] = True
                            known_boolean_keys.setdefault(prefix, set()).add(final_key)

            if concept_attrs:
                assignments[doc_id] = concept_attrs

        # 6. Backfill False values (Dense Mode)
        if backfill_false and assignments:
            for doc_id, concept_attrs in assignments.items():
                # For every prefix present in this doc's assignments, backfill missing siblings.
                # Identify which prefixes this doc has touched.
                touched_prefixes = set()
                for key in concept_attrs.keys():
                    # Only relevant for boolean keys
                    if concept_attrs[key] is True:
                         # Reconstruct prefix: splitting on last colon
                         if ":" in key:
                             prefix = key.rsplit(":", 1)[0]
                             touched_prefixes.add(prefix)
                
                # For each touched prefix, ensure all *known* keys in that prefix are present
                for prefix in touched_prefixes:
                    universe = known_boolean_keys.get(prefix, set())
                    for k in universe:
                        if k not in concept_attrs:
                            concept_attrs[k] = False
            
            # NOTE: Documents that did NOT touch a prefix at all will not get False keys for that prefix.
            # This is "Sparse at Concept Level, Dense at Value Level".
            # If the user wants FULL density (every doc gets every key even if concept is missing),
            # we would iterate `known_boolean_keys` instead of `touched_prefixes`.
            # Assuming "Sparse Concept, Dense Value" is safer to avoid exploding non-relevant metadata.

        if assignments:
            self._update_metadata_concepts_schema(assignments)

        return assignments

    def _normalize_concept_key(self, concept: str) -> str:
        """
        Normalize 'domain attribute' -> 'domain:attribute'.
        Heuristic: Split on first space. First part is domain. Rest is attribute (replace spaces with _).
        """
        parts = concept.strip().split(" ", 1)
        if len(parts) == 1:
            return parts[0]
        
        domain = parts[0]
        attr = parts[1].replace(" ", "_")
        return f"{domain}:{attr}"

    def _infer_concept_schemas(self, concepts: List[str]) -> Dict[str, Any]:
        """
        Infer schema for a list of concepts.
        """
        if not concepts:
            return {}

        return self._schema_inference_model(concepts)

    def _load_metadata_concepts_schema(self) -> Dict[str, Any]:
        if not os.path.exists(SCHEMA_PATH):
            return {}
        try:
            with open(SCHEMA_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
 
    def _update_metadata_concepts_schema(
        self,
        assignments: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """
        Maintain metadata_concepts_schema.json.
        """
        schema = self._load_metadata_concepts_schema()
        updated = False

        for _doc_id, concept_attrs in assignments.items():
            for key, value in concept_attrs.items():
                if key not in schema:
                    # Infer type from value
                    val_type = "string"
                    if isinstance(value, bool):
                        val_type = "bool"
                    elif isinstance(value, int):
                        val_type = "int"
                    elif isinstance(value, float):
                        val_type = "float"
                    elif isinstance(value, list):
                        # List of ints/floats
                        if value and isinstance(value[0], float):
                             val_type = "float_list"
                        else:
                             val_type = "int_list"

                    schema[key] = {
                        "type": val_type,
                        "allowed_values": [] 
                    }
                    updated = True

        if updated:
            try:
                with open(SCHEMA_PATH, "w") as f:
                    json.dump(schema, f, indent=2, sort_keys=True)
            except Exception as e:
                logger.warning(f"Failed to write metadata concepts schema: {e}")

    def generate_from_queries(self, queries: List[str]) -> List[str]:
        """
        Convenience method: given a list of raw query strings, learn and
        return the concept vocabulary.
        """
        self.fit(queries=queries)
        return list(self._concept_vocabulary)

    def get_concept_vocabulary(self) -> List[str]:
        """Return the learned concept vocabulary (final concepts)."""
        return list(self._concept_vocabulary)

    def _fit_two_stage(self, queries: List[str]) -> List[str]:
        """
        TWO-STAGE STRATEGY:
        1. LLM generates per-query intermediate concepts.
        2. Concepts are embedded and clustered.
        3. LLM generates a centroid (final concept) for each cluster.
        """
        # 1) Per-query concepts
        all_concepts: List[str] = []
        for query in queries:
            per_query_concepts = self._per_query_model(query)
            all_concepts.extend(per_query_concepts)

        all_concepts = _dedupe_preserve_order(all_concepts)
        logger.info(f"LLMConceptGenerator: generated {len(all_concepts)} intermediate concepts.")
        if not all_concepts:
            return []

        # 2) Embed + cluster concepts
        model = SentenceTransformer(self.embedding_model_name)
        embeddings = model.encode(all_concepts, show_progress_bar=False)

        # If fewer concepts than clusters, reduce cluster count
        n_clusters = min(self.n_clusters, len(all_concepts))
        logger.info(f"LLMConceptGenerator: clustering into {n_clusters} clusters.")
        if n_clusters <= 0:
            return []

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        clusters: Mapping[int, List[str]] = {}
        for concept, label in zip(all_concepts, labels):
            clusters.setdefault(int(label), []).append(concept)  # type: ignore[attr-defined]

        # 3) LLM centroid per cluster
        final_concepts: List[str] = []
        for cluster_id in sorted(clusters.keys()):
            members = clusters[cluster_id]
            if not members:
                continue
            centroid = self._centroid_model(members)
            if centroid:
                final_concepts.append(centroid)

        return _dedupe_preserve_order(final_concepts)

    def _fit_direct(self, queries: List[str]) -> List[str]:
        """
        DIRECT STRATEGY:
        Feed all queries to the LLM and ask for final concepts directly.
        """
        final_concepts = self._batch_final_model(queries)
        return _dedupe_preserve_order(final_concepts)
