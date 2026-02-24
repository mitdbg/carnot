from __future__ import annotations

import json
import re
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import dspy
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("concept_generation_outputs")

ConceptTriple = Tuple[str, str, str]


# ── Text helpers ──────────────────────────────────────────────────────────────


def _norm(text: str) -> str:
    return " ".join(str(text).strip().split())


def _canon(d: str, f: str, v: str) -> ConceptTriple:
    return (_norm(d), _norm(f), _norm(v))


def _key(t: ConceptTriple) -> ConceptTriple:
    return (_norm(t[0]).lower(), _norm(t[1]).lower(), _norm(t[2]).lower())


def _embed_text(t: ConceptTriple) -> str:
    return " | ".join(_canon(*t))


def _to_list(t: ConceptTriple) -> List[str]:
    return list(_canon(*t))


# ── Parsing helpers ───────────────────────────────────────────────────────────


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _json_to_triples(parsed: Any) -> List[ConceptTriple]:
    if not isinstance(parsed, list):
        return []
    triples: List[ConceptTriple] = []
    for item in parsed:
        if isinstance(item, list) and len(item) == 3:
            triples.append(_canon(str(item[0]), str(item[1]), str(item[2])))
        elif isinstance(item, dict):
            lm = {str(k).strip().lower(): v for k, v in item.items()}
            if {"domain", "facet", "value"} <= set(lm):
                triples.append(
                    _canon(str(lm["domain"]), str(lm["facet"]), str(lm["value"]))
                )
        elif isinstance(item, str):
            triples.append(_canon("unknown", "unknown", item))
    return triples


def _parse_triples(raw: str) -> List[ConceptTriple]:
    """Parse triples from strict JSON first, then salvage bullet/pipe formats."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    raw = raw.strip()

    candidates = [raw]
    if "[" in raw and "]" in raw:
        candidates.append(raw[raw.find("[") : raw.rfind("]") + 1])
    for c in candidates:
        parsed = _try_json(c)
        if parsed is not None:
            result = _json_to_triples(parsed)
            if result:
                return result

    triples: List[ConceptTriple] = []
    for line in raw.splitlines():
        line = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", line.strip()).strip()
        if "|" in line:
            parts = [_norm(p) for p in line.split("|")]
            if len(parts) == 3 and all(parts):
                triples.append(_canon(*parts))
    return triples or [_canon("unknown", "unknown", raw)]


def _parse_batch(raw: str, queries: List[str]) -> Dict[str, List[ConceptTriple]]:
    """Parse batched per-query output (JSON dict keyed by query, or aligned list)."""
    result: Dict[str, List[ConceptTriple]] = {q: [] for q in queries}
    if not isinstance(raw, str) or not raw.strip():
        return result

    parsed = _try_json(raw.strip())
    if parsed is None:
        for open_ch, close_ch in [("{", "}"), ("[", "]")]:
            if open_ch in raw and close_ch in raw:
                parsed = _try_json(raw[raw.find(open_ch) : raw.rfind(close_ch) + 1])
                if parsed is not None:
                    break
    if parsed is None:
        return result

    if isinstance(parsed, dict):
        for idx, q in enumerate(queries):
            triples = _json_to_triples(parsed.get(q, []))
            if not triples:
                triples = _json_to_triples(parsed.get(str(idx), []))
            result[q] = _dedupe(triples)
    elif isinstance(parsed, list):
        for idx, q in enumerate(queries):
            if idx < len(parsed):
                result[q] = _dedupe(_json_to_triples(parsed[idx]))
    return result


# ── Deduplication ─────────────────────────────────────────────────────────────


def _dedupe(items: Iterable[ConceptTriple]) -> List[ConceptTriple]:
    """Case-insensitive dedup preserving insertion order."""
    seen: set = set()
    out: List[ConceptTriple] = []
    for t in items:
        c = _canon(*t)
        if not all(c):
            continue
        k = _key(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _dedupe_strings(items: Iterable[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for s in items:
        s = _norm(s)
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


# ── DSPy signatures & modules ────────────────────────────────────────────────


class PerQueryConceptTripleSignature(dspy.Signature):
    """
    Generate mid-granularity concepts for a single query.
    """
    query = dspy.InputField(
        desc="Natural language search query."
    )
    concept_triples = dspy.OutputField(
        desc=(
            "Return ONLY a JSON array of triples: [domain, facet, value] (no prose before/after). "
            "domain: entity type being searched (e.g., film, book, plant_taxon, biological_taxon, work). "
            "facet: attribute dimension (e.g., genre, subject, location, language, time_period, taxonomy, audience, source). "
            "value: short canonical noun phrase/entity for that facet."
        )
    )


class PerQueryConceptTripleModel(dspy.Module):

    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(PerQueryConceptTripleSignature)

        self._few_shot_examples = [
            dspy.Example(
                query="Non-fiction books about elections excluding Books about North America",
                concept_triples='['
                                '["book","genre","non-fiction"],'
                                '["book","subject","elections"],'
                                '["book","location","North America"]'
                                ']'
            ).with_inputs("query"),
            dspy.Example(
                query="Orchids of Myanmar but not Flora of India",
                concept_triples='['
                                '["plant_taxon","taxonomy","Orchidaceae"],'
                                '["plant_taxon","location","Myanmar"],'
                                '["work","source","Flora of India"]'
                                ']'
            ).with_inputs("query"),
            dspy.Example(
                query="Canadian teen films that are not in the English language.",
                concept_triples='['
                                '["film","location","Canada"],'
                                '["film","audience","teenagers"],'
                                '["film","language","English"]'
                                ']'
            ).with_inputs("query"),
            dspy.Example(
                query="Prehistoric toothed whales not from the Miocene period",
                concept_triples='['
                                '["biological_taxon","taxonomy","Odontoceti"],'
                                '["biological_taxon","time_period","prehistory"],'
                                '["biological_taxon","time_period","Miocene"],'
                                '["biological_taxon","subject","toothed whales"]'
                                ']'
            ).with_inputs("query"),
        ]

    def forward(self, query: str) -> List[ConceptTriple]:
        result = self._predict(query=query, demos=self._few_shot_examples)
        raw = getattr(result, "concept_triples", "") or ""
        return _dedupe(_parse_triples(raw))


class BatchPerQueryConceptTripleSignature(dspy.Signature):
    queries = dspy.InputField(desc="List of natural language queries.")
    per_query_concepts = dspy.OutputField(
        desc=(
            "Return ONLY a JSON object keyed by the exact query string. "
            "Each value must be a JSON array of triples [domain, facet, value]. "
            "No prose before/after."
        )
    )


class BatchPerQueryConceptTripleModel(dspy.Module):

    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(BatchPerQueryConceptTripleSignature)
        self._few_shot_examples = [
            dspy.Example(
                queries=[
                    "Non-fiction books about elections excluding Books about North America",
                    "Orchids of Myanmar but not Flora of India",
                    "Canadian teen films that are not in the English language.",
                    "Prehistoric toothed whales not from the Miocene period",
                ],
                per_query_concepts='{"Non-fiction books about elections excluding Books about North America":[["book","genre","non-fiction"],["book","subject","elections"],["book","location","North America"]],"Orchids of Myanmar but not Flora of India":[["plant_taxon","taxonomy","Orchidaceae"],["plant_taxon","location","Myanmar"],["work","source","Flora of India"]],"Canadian teen films that are not in the English language.":[["film","location","Canada"],["film","audience","teenagers"],["film","language","English"]],"Prehistoric toothed whales not from the Miocene period":[["biological_taxon","taxonomy","Odontoceti"],["biological_taxon","time_period","prehistory"],["biological_taxon","time_period","Miocene"],["biological_taxon","subject","toothed whales"]]}',
            ).with_inputs("queries"),
        ]

    def forward(self, queries: List[str]) -> Dict[str, List[ConceptTriple]]:
        result = self._predict(queries=queries, demos=self._few_shot_examples)
        raw = getattr(result, "per_query_concepts", "") or ""
        return _parse_batch(raw, queries)


# ── Generator ─────────────────────────────────────────────────────────────────


class ConceptGenerationMode(Enum):
    TWO_STAGE = "two_stage"


class LLMConceptGenerator:
    """
    Concept-generation pipeline:
      1. LLM extracts per-query (domain, facet, value) triples.
      2. Global dedup with query provenance tracking.
      3. Group by *domain*, cluster within each group.
      4. Final concepts = unique domain strings (+ optional refined centroids).
    """

    def __init__(self, config: Any) -> None:
        self.config = config

        mode_str = getattr(config, "concept_generation_mode", ConceptGenerationMode.TWO_STAGE.value)
        try:
            self.mode = ConceptGenerationMode(mode_str)
        except ValueError:
            self.mode = ConceptGenerationMode.TWO_STAGE

        self.n_clusters = getattr(config, "concept_cluster_count", 3)
        self.min_cluster_size = int(getattr(config, "concept_min_cluster_size", 3) or 3)
        self.per_query_batch_size = int(getattr(config, "concept_per_query_batch_size", 16) or 16)
        self.enable_refined_centroids = bool(getattr(config, "concept_enable_refined_centroids", False))
        self.embedding_model_name = getattr(config, "concept_embedding_model", "all-MiniLM-L6-v2")

        self._per_query_model = PerQueryConceptTripleModel()
        self._batch_model = BatchPerQueryConceptTripleModel()
        self._concept_vocabulary: List[str] = []

    # ── public API ──

    def fit(self, queries: Iterable[str], *, output_dir: Optional[str | Path] = None) -> None:
        queries_list = list(queries)
        logger.info(f"Fitting on {len(queries_list)} queries (mode={self.mode}).")

        if not queries_list:
            self._concept_vocabulary = []
            return

        concepts, file_map = self._fit_two_stage(queries_list)
        self._concept_vocabulary = concepts
        logger.info(f"Learned {len(concepts)} concepts.")

        out = Path(
            output_dir
            or getattr(self.config, "concept_generation_output_dir", None)
            or getattr(self.config, "concept_output_dir", None)
            or DEFAULT_OUTPUT_DIR
        )
        self._save(out, file_map)

    # ── persistence ──

    def _save(self, out_dir: Path, file_map: Dict[str, Any]) -> None:
        """Write each key in *file_map* as a separate ``<key>.json`` file."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, data in file_map.items():
            path = out_dir / f"{name}.json"
            try:
                path.write_text(json.dumps(data, indent=2, sort_keys=False))
                logger.info(f"Wrote {path}")
            except Exception as e:
                logger.warning(f"Failed to write {path}: {e}")

    # ── clustering ──

    def _target_n_clusters(self, group_size: int) -> int:
        if group_size < self.min_cluster_size:
            return 1
        target = max(1, group_size // max(1, self.min_cluster_size))
        return max(1, min(int(self.n_clusters), target, group_size))

    def _cluster_triples(
        self,
        triples: List[ConceptTriple],
        embedding_model: Optional[SentenceTransformer],
    ) -> Dict[str, Any]:
        if not triples:
            return {"members": [], "n_clusters": 0, "clusters": {}}

        triples = sorted(triples, key=_key)
        n = len(triples)
        n_clusters = self._target_n_clusters(n)

        if n_clusters <= 1 or embedding_model is None:
            labels = [0] * n
            n_clusters = 1
            kmeans, embeddings = None, None
        else:
            texts = [_embed_text(t) for t in triples]
            embeddings = embedding_model.encode(texts, show_progress_bar=False)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = [int(x) for x in kmeans.fit_predict(embeddings)]

        buckets: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(labels):
            buckets.setdefault(lbl, []).append(idx)

        clusters: Dict[str, Any] = {}
        for cid in sorted(buckets):
            members = [triples[i] for i in buckets[cid]]
            entry: Dict[str, Any] = {"members": [_to_list(t) for t in members]}
            if self.enable_refined_centroids:
                if kmeans is not None and embeddings is not None:
                    center = kmeans.cluster_centers_[cid]
                    best = min(
                        buckets[cid],
                        key=lambda i: float(((embeddings[i] - center) ** 2).sum()),
                    )
                    entry["centroid_value"] = _norm(triples[best][2])
                else:
                    entry["centroid_value"] = _norm(members[0][2])
            clusters[str(cid)] = entry

        return {
            "members": [_to_list(t) for t in triples],
            "n_clusters": n_clusters,
            "clusters": clusters,
        }

    # ── two-stage pipeline ──

    def _fit_two_stage(self, queries: List[str]) -> tuple[List[str], Dict[str, Any]]:
        # ── Stage 1: per-query extraction via LLM ──
        per_query: Dict[str, List[ConceptTriple]] = {}
        batch_size = max(1, self.per_query_batch_size)

        for start in tqdm(range(0, len(queries), batch_size), desc="per-query triples"):
            batch = queries[start : start + batch_size]
            batch_result = self._batch_model(batch)
            batch_ok = any(batch_result.get(q) for q in batch)
            for q in batch:
                triples = _dedupe(batch_result.get(q, []))
                if not batch_ok and not triples:
                    triples = _dedupe(self._per_query_model(q))
                per_query[q] = triples

        # ── Stage 2: global dedup + query provenance ──
        key_to_triple: Dict[ConceptTriple, ConceptTriple] = {}
        key_to_queries: Dict[ConceptTriple, List[str]] = {}

        for q, triples in per_query.items():
            for t in triples:
                k = _key(t)
                if k not in key_to_triple:
                    key_to_triple[k] = _canon(*t)
                    key_to_queries[k] = []
                if q not in key_to_queries[k]:
                    key_to_queries[k].append(q)

        sorted_keys = sorted(key_to_triple)
        unique_triples = [key_to_triple[k] for k in sorted_keys]
        logger.info(f"Generated {len(unique_triples)} unique triples.")

        num_unique = len(unique_triples)

        facet_freq: Dict[str, int] = {}
        for t in unique_triples:
            pair = f"{_norm(t[0])}|{_norm(t[1])}"
            facet_freq[pair] = facet_freq.get(pair, 0) + 1
        facet_freq = dict(sorted(facet_freq.items(), key=lambda kv: (-kv[1], kv[0])))

        per_query_json: Dict[str, Any] = {
            "num_unique_triples": num_unique,
            "facet_frequencies": facet_freq,
            "queries": {q: [_to_list(t) for t in ts] for q, ts in per_query.items()},
        }

        if not unique_triples:
            return [], {
                "per_query_concepts": per_query_json,
                "domain_clusters": {"total_clusters": 0, "domains": {}},
            }

        # ── Stage 3: group by domain, cluster within each group ──
        domain_groups: Dict[str, List[ConceptTriple]] = {}
        for t in unique_triples:
            domain_groups.setdefault(_norm(t[0]), []).append(t)

        sorted_domains = sorted(domain_groups)
        embedding_model: Optional[SentenceTransformer] = None
        if any(len(ts) >= self.min_cluster_size for ts in domain_groups.values()):
            embedding_model = SentenceTransformer(self.embedding_model_name)

        domains_json: Dict[str, Any] = {}
        total_clusters = 0
        refined_concepts: List[str] = []
        for domain in tqdm(sorted_domains, desc="domain clustering"):
            cluster_result = self._cluster_triples(domain_groups[domain], embedding_model)
            total_clusters += cluster_result["n_clusters"]
            for cid, cdata in cluster_result["clusters"].items():
                cdata["num_members"] = len(cdata["members"])
                if self.enable_refined_centroids:
                    cv = cdata.get("centroid_value")
                    if cv:
                        refined_concepts.append(f"{domain} | {cv}")
            domains_json[domain] = cluster_result

        base_concepts = list(sorted_domains)
        final_concepts = _dedupe_strings(base_concepts + refined_concepts)

        domain_clusters_json: Dict[str, Any] = {
            "total_clusters": total_clusters,
            "domains": domains_json,
        }

        return final_concepts, {
            "per_query_concepts": per_query_json,
            "domain_clusters": domain_clusters_json,
        }


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import requests

    print("Downloading queries...")
    url = "https://storage.googleapis.com/gresearch/quest/train.jsonl"
    lines = requests.get(url).content.decode("utf-8").strip().split("\n")
    queries = [json.loads(line).get("query", "") for line in lines if line.strip()]
    queries = [q for q in queries if q]
    print(f"Queries downloaded, {len(queries)} queries found")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    lm = dspy.LM("openai/gpt-5.1", temperature=1.0, max_tokens=16000, api_key=api_key)
    dspy.configure(lm=lm)

    class _Cfg:
        concept_generation_mode = ConceptGenerationMode.TWO_STAGE.value
        concept_cluster_count = 3
        concept_per_query_batch_size = 20
        concept_embedding_model = "all-MiniLM-L6-v2"

    print("Fitting TWO_STAGE model...")
    LLMConceptGenerator(_Cfg()).fit(queries, output_dir="results/quest_concepts/two_stage")
    print("Done — wrote per_query_concepts.json, domain_clusters.json")
