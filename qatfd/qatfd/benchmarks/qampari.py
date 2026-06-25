"""QAMPARI benchmark (Tel Aviv University; the KARLBench sub-benchmark).

Questions: the QAMPARI test JSONL (`test_data.jsonl`) — 1000 open-domain questions over Wikipedia,
  each with MANY answers (>=5, ~14.7 on average). Per record: `qid`, `question_text`, and
  `answer_list[]`, where every answer carries `answer_text`, `aliases`, and `proof[]` (each proof
  names the supporting Wikipedia article via `found_in_url`). QAMPARI tests EXHAUSTIVE entity search
  — success depends on comprehensive recall, not on finding a single supporting mention.
Gold nuggets: the gold answer entities themselves — one nugget per `answer_text`. The score is
  nugget-completion recall (KARL "treats each entity as a separate nugget"), graded by the LLM judge
  in judge.py so an alias / paraphrase of the entity still counts.
Corpus + index: KARL's QAMPARI corpus is the subset of QAMPARI's chunked Wikipedia (~100-token
  passages) containing at least one gold answer entity — ~256,680 chunks — embedded with
  Qwen3-Embedding-0.6B into a Chroma collection. The chunk-text doc map is rebuilt from the
  embedding metadata (so it matches exactly what the vectors were built from); see
  engaging-scripts/{preprocess_qampari_corpus.py, compute_qampari_embeddings.py}.
Gold docs / recall: a proof identifies the supporting Wikipedia ARTICLE, not a corpus chunk (the
  proof `pid` is a per-question id, never a corpus chunk_id), and the released chunk `url` is a
  curid URL while proofs use title-slug URLs — so article identity is the normalized Wikipedia
  TITLE. gold_docs are normalized article titles; recall = fraction of gold articles whose chunks
  were retrieved (`doc_recall`), collapsing retrieved chunk_ids -> article title via the metadata.
Held-out test set: `test_ids_path` if given, else ALL 1000 questions (run with --split test).
"""

from __future__ import annotations

import chromadb
import glob
import json
import os
from urllib.parse import unquote, urlparse

from qatfd.benchmarks.base import Benchmark, BenchmarkResources, doc_recall
from qatfd.benchmarks.judge import judge_nugget_recall
from qatfd.config import QampariConfig
from qatfd.constants import QAMPARI
from qatfd.paths import resolve_under_skunk
from qatfd.types import Question

# Grader persona for the nugget judge: QAMPARI answers are encyclopedic entities (people, films,
# places), so a neutral evaluator — not TREC-BioGen's biomedical one — judges entity presence.
_JUDGE_SYSTEM = (
    "You are a careful answer-evaluation judge for open-domain, multi-answer questions over "
    "encyclopedic (Wikipedia) text. Each decompositional fact is one gold answer entity; judge "
    "whether the predicted answer names that entity (an alias or alternate surface form counts)."
)


def _norm_title(title: str) -> str:
    """Canonical form of a Wikipedia article title: underscores -> spaces, whitespace collapsed,
    lowercased. The single join key between a proof's article and a corpus chunk's article."""
    return " ".join(str(title).replace("_", " ").split()).lower()


def _article_from_url(url: str) -> str:
    """Normalized article identity for a proof's `found_in_url`. Proofs use title-slug URLs
    (".../wiki/The_Helen_Morgan_Story"), so take the slug after `/wiki/`, url-decode it, and
    normalize. Falls back to the raw (normalized) URL for the rare non-`/wiki/` form."""
    path = urlparse(str(url)).path
    marker = "/wiki/"
    if marker in path:
        return _norm_title(unquote(path.split(marker, 1)[1]))
    return _norm_title(unquote(url))


class QampariBenchmark(Benchmark):
    name = QAMPARI
    config: QampariConfig

    # QAMPARI questions have many correct answers; the metric is entity recall, so the answer must
    # be an exhaustive list. Push for breadth/recall over a single best answer.
    answer_format_hint = (
        "This question has MANY correct answers. List every entity that answers it that the "
        "documents support — as many as you can find — as a comma-separated list of entity names. "
        "Do not stop at one answer; prefer completeness (recall) over brevity, and include an "
        "entity whenever the documents support it."
    )

    def __init__(self, config: QampariConfig) -> None:
        config.chromadb_dir = str(resolve_under_skunk(config.chromadb_dir))
        config.questions_path = str(resolve_under_skunk(config.questions_path))
        config.qampari_metadata_glob = str(resolve_under_skunk(config.qampari_metadata_glob))
        if config.test_ids_path:
            config.test_ids_path = str(resolve_under_skunk(config.test_ids_path))
        if config.prompts_path:
            config.prompts_path = str(resolve_under_skunk(config.prompts_path))
        # chunk_id -> normalized article title, built from the embedding metadata in _build_resources;
        # used by recall_metrics to collapse retrieved chunk_ids to Wikipedia articles.
        self._chunk_to_article: dict[str, str] = {}
        super().__init__(config)

    # ---- questions ------------------------------------------------------------

    @staticmethod
    def _gold_articles(rec: dict) -> list[str]:
        """Normalized titles of every Wikipedia article cited by any answer's proof, de-duplicated
        in first-seen order. These are the gold supporting documents for retrieval recall."""
        articles: list[str] = []
        seen: set[str] = set()
        for ans in rec.get("answer_list", []):
            for pr in ans.get("proof", []):
                url = pr.get("found_in_url")
                if not url:
                    continue
                art = _article_from_url(url)
                if art and art not in seen:
                    seen.add(art)
                    articles.append(art)
        return articles

    def load_questions(self) -> list[Question]:
        questions: list[Question] = []
        with open(self.config.questions_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                answers = rec.get("answer_list", [])
                # one nugget per gold answer entity (KARL: "each entity is a separate nugget").
                nuggets = [str(a["answer_text"]).strip() for a in answers if str(a.get("answer_text", "")).strip()]
                # aliases per answer, for reference / debugging (the judge handles aliases semantically).
                aliases = {str(a.get("answer_text", "")): list(a.get("aliases", [])) for a in answers}
                questions.append(
                    Question(
                        qid=str(rec["qid"]),
                        text=str(rec["question_text"]),
                        gold=", ".join(nuggets),
                        gold_docs=self._gold_articles(rec),
                        meta={
                            "nuggets": nuggets,
                            "n_nuggets": len(nuggets),
                            "aliases": aliases,
                        },
                    )
                )
        return questions

    # ---- retrieval substrate --------------------------------------------------

    def _build_document_map(self) -> dict[str, str]:
        """chunk_id -> passage text, reconstructed from the embedding-job metadata
        (metadata_rank*.json: chunk_id -> {cleaned, title, page_id, url, element_id}). Side effect:
        populates `self._chunk_to_article` (chunk_id -> normalized article title) for doc-recall."""
        paths = sorted(glob.glob(self.config.qampari_metadata_glob))
        if not paths:
            raise FileNotFoundError(
                f"no QAMPARI element metadata found at {self.config.qampari_metadata_glob}; "
                f"expected metadata_rank*.json produced by compute_qampari_embeddings.py."
            )
        document_map: dict[str, str] = {}
        chunk_to_article: dict[str, str] = {}
        for p in paths:
            with open(p) as f:
                meta = json.load(f)
            for chunk_id, entry in meta.items():
                document_map[str(chunk_id)] = entry.get("cleaned", "")
                chunk_to_article[str(chunk_id)] = _norm_title(entry.get("title", ""))
        self._chunk_to_article = chunk_to_article
        return document_map

    def _build_resources(self) -> BenchmarkResources:
        if not os.path.exists(self.config.chromadb_dir):
            raise FileNotFoundError(f"chromadb_dir {self.config.chromadb_dir} does not exist")

        client = chromadb.PersistentClient(path=self.config.chromadb_dir)
        try:
            collection = client.get_collection(name=self.config.chromadb_collection)
        except Exception as e:
            raise RuntimeError(
                f"chroma collection {self.config.chromadb_collection!r} not found under {self.config.chromadb_dir}."
            ) from e

        return BenchmarkResources(
            chroma_collection=collection,
            document_map=self._build_document_map(),
            config=self.config,
        )

    # ---- scoring + metrics ----------------------------------------------------

    async def score(self, question: Question, predicted: str, ctx) -> dict:
        return await judge_nugget_recall(
            ctx,
            question=question.text,
            nuggets=question.meta.get("nuggets", []),
            predicted=predicted,
            model=self.config.judge_model,
            judge_system=_JUDGE_SYSTEM,
            partial_credit=self.config.partial_credit,
        )

    def recall_metrics(self, retrieved: list[str] | None, question: Question) -> dict[str, float]:
        # Relevance is labeled at the Wikipedia-article level (a proof cites a supporting article),
        # so collapse retrieved chunk_ids to their article title and report article-level doc_recall.
        ret_articles = None if retrieved is None else [self._chunk_to_article.get(c, c) for c in retrieved]
        return {"doc_recall": doc_recall(ret_articles, question.gold_docs)}

    def test_qids(self) -> set[str]:
        if self.config.test_ids_path:
            assert os.path.exists(self.config.test_ids_path), (
                f"test_ids_path {self.config.test_ids_path} does not exist; expected a JSON object with "
                f"a 'query_ids' field listing the held-out QAMPARI qids."
            )
            with open(self.config.test_ids_path) as f:
                data = json.load(f)
            assert "query_ids" in data, f"test_ids_path {self.config.test_ids_path} must have a 'query_ids' field."
            return {str(q) for q in data["query_ids"]}

        # no explicit split: the whole 1000-question test set is the held-out comparison set.
        return {q.qid for q in self.load_questions()}
