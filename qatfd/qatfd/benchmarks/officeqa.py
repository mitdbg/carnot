"""OfficeQA benchmark (Treasury Bulletin corpus).

Questions: skunk/officeqa_pro.csv (uid/question/answer/source_docs/source_files/difficulty).
Corpus + index: the prebuilt ChromaDB collection under skunk/.chromadb (element-level,
Qwen3-Embedding-8B) + the doc_id->text map from treasury_bulletins_cleaned/clean_page_map.json.
Scoring: the official cup-kit numeric scorer (deterministic, 0% relative-error tolerance).
Held-out test set: the 32 UIDs in skunk/eval/test_set_uids.json (mandatory exclusion).
"""

from __future__ import annotations

import chromadb
import json
import os
import re

import pandas as pd
from skunk.eval.scoring import SCORER_VERSION, score_correct

from qatfd.benchmarks.base import Benchmark, BenchmarkResources, doc_recall
from qatfd.config import OfficeQAConfig
from qatfd.constants import OFFICE_QA
from qatfd.paths import resolve_under_skunk
from qatfd.types import Question

# source_docs URL -> (month, year, page). Mirrors eval/eval_e2e.py's golden parsing.
_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}
_URL_RE = re.compile(
    r"/(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)"
    r"-(?P<year>\d{4})[^?]*\?page=(?P<page>\d+)",
    re.IGNORECASE,
)


def _parse_gold_page_keys(source_docs: str) -> list[str]:
    """Extract page keys ("YYYY_MM_page") from the source_docs URL(s) for doc-recall."""
    if not isinstance(source_docs, str):
        return []
    keys: list[str] = []
    for m in _URL_RE.finditer(source_docs):
        mm = _MONTH_MAP[m.group("month").lower()]
        keys.append(f"{m.group('year')}_{mm}_{int(m.group('page'))}")
    return keys


def _page_key_to_doc_key(page_key: str) -> str:
    """Collapse a page key "YYYY_MM_page" to its document (monthly bulletin) key
    "YYYY_MM" by dropping the trailing page component."""
    return "_".join(page_key.split("_")[:2])


def officeqa_recall_metrics(retrieved: list[str] | None, gold_page_keys: list[str]) -> dict[str, float]:
    """OfficeQA recall at two granularities, both over page keys ("YYYY_MM_page"):
    `page_recall` (exact page hit) and `doc_recall` (right monthly bulletin, page
    ignored). The gap between them isolates "found the right document but not the
    right page". Pure so the backfill script can reuse it on stored report rows."""
    page_recall = doc_recall(retrieved, gold_page_keys)
    ret_docs = None if retrieved is None else [_page_key_to_doc_key(k) for k in retrieved]
    gold_docs = [_page_key_to_doc_key(k) for k in gold_page_keys]
    return {"page_recall": page_recall, "doc_recall": doc_recall(ret_docs, gold_docs)}


class OfficeQABenchmark(Benchmark):
    name = OFFICE_QA
    config: OfficeQAConfig

    # The cup scorer extracts the final answer from a <FINAL_ANSWER> wrapper and
    # numerically compares it to a BARE gold value (digits, optional sign / decimal /
    # thousands-commas / a trailing %; lists like [1.2, 3.4]). It does NOT strip unit
    # or magnitude words, so "4962.46 million dollars" or "0.0 percentage points" score
    # wrong even when numerically right — the hint must forbid them.
    answer_format_hint = (
        "Wrap your final answer in <FINAL_ANSWER>...</FINAL_ANSWER> tags with no working "
        "or explanation. Inside the tags put ONLY the bare value, expressed in the units "
        "the question specifies: just the number, with no unit words, currency symbols, or "
        "magnitude words — write 4962.46, not '4962.46 million dollars', '$4,962,460,000', "
        "or '4962.46 USD' (a percentage may use a trailing '%' but not the word 'percent'). "
        "If several values are requested, list them in brackets like [1.2, 3.4]. For a "
        "non-numeric answer, give the date or short phrase alone."
    )

    def __init__(self, config: OfficeQAConfig) -> None:
        # resolve paths and store config
        config.chromadb_dir = str(resolve_under_skunk(config.chromadb_dir))
        config.csv_path = str(resolve_under_skunk(config.csv_path))
        config.clean_page_map_path = str(resolve_under_skunk(config.clean_page_map_path))
        config.test_set_uids_path = str(resolve_under_skunk(config.test_set_uids_path))
        if config.prompts_path:
            config.prompts_path = str(resolve_under_skunk(config.prompts_path))
        if config.pdf_dir:
            config.pdf_dir = str(resolve_under_skunk(config.pdf_dir))
        super().__init__(config)

    def load_questions(self) -> list[Question]:
        df = pd.read_csv(self.config.csv_path)
        questions: list[Question] = []
        for row in df.itertuples(index=False):
            questions.append(
                Question(
                    qid=str(row.uid),
                    text=str(row.question),
                    gold=str(row.answer),
                    gold_docs=_parse_gold_page_keys(getattr(row, "source_docs", "")),
                    meta={
                        "difficulty": getattr(row, "difficulty", ""),
                        "source_files": getattr(row, "source_files", ""),
                    },
                )
            )
        return questions

    def _build_document_map(self) -> dict[str, str]:
        """doc_id -> full document text, reconstructed by concatenating each document's
        element `cleaned` text (ordered by element_id) from the embedding metadata. This
        is the same text the vectors were built from, so it matches what search returns."""
        clean_page_map_path = self.config.clean_page_map_path
        if not os.path.exists(clean_page_map_path):
            raise FileNotFoundError(
                f"no OfficeQA clean page map found at {self.config.clean_page_map_path}."
            )

        with open(clean_page_map_path) as f:
            clean_page_map: dict = json.load(f)

        document_map: dict[str, str] = {}
        missing = 0
        example_missing = ""
        for doc_id, entry in clean_page_map.items():
            raw = entry[0] if isinstance(entry, (list, tuple)) else entry
            path = str(resolve_under_skunk(raw))
            try:
                with open(path) as pf:
                    document_map[doc_id] = pf.read()
            except OSError:
                missing += 1
                example_missing = example_missing or path
        # fail loudly if the document map is empty (likely a misconfiguration)
        if not document_map:
            raise FileNotFoundError(
                f"OfficeQA document_map is empty: all {missing} page files referenced by "
                f"{clean_page_map_path} were unreadable (e.g. {example_missing!r}). "
                f"Check the corpus is present under the skunk dir."
            )
        if missing:
            print(f"[officeqa] document_map: loaded {len(document_map)} pages, {missing} missing.")
        return document_map

    def _build_resources(self) -> BenchmarkResources:
        if not os.path.exists(self.config.chromadb_dir):
            raise FileNotFoundError(f"chromadb_dir {self.config.chromadb_dir} does not exist.")

        client = chromadb.PersistentClient(path=self.config.chromadb_dir)
        try:
            collection = client.get_collection(name=self.config.chromadb_collection)
        except Exception as e:
            raise RuntimeError(
                f"chroma collection {self.config.chromadb_collection!r} not found under {self.config.chromadb_dir}."
            ) from e

        document_map = self._build_document_map()

        return BenchmarkResources(
            chroma_collection=collection,
            document_map=document_map,
            config=self.config,
        )

    async def score(self, question: Question, predicted: str, ctx) -> dict:
        return {"score": float(score_correct(question.gold, predicted)), "scorer": SCORER_VERSION}

    def recall_metrics(self, retrieved: list[str] | None, question: Question) -> dict[str, float]:
        return officeqa_recall_metrics(retrieved, question.gold_docs)

    def test_qids(self) -> set[str]:
        if not os.path.exists(self.config.test_set_uids_path):
            return set()
        with open(self.config.test_set_uids_path) as f:
            data = json.load(f)
        assert "uids" in data, "test_set_uids_path must be a dict with 'uids' key."

        return set(data["uids"])
