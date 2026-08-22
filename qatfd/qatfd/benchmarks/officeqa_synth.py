"""Synthetic OfficeQA benchmark (LLM-generated QA pairs over the Treasury Bulletin corpus).

Questions: a `qa_pairs.json` produced by dataset_datagen (list of records with qid /
line_of_inquiry / question / answer / doc_ids), copied under benchmarks/officeqa/.
Each ~6-question block shares one "line of inquiry" over a common set of documents, so
consecutive questions are topically related — the intended testbed for working-set reuse.
Corpus, chroma collection, document map, scorer, and recall metrics are all inherited
from OfficeQABenchmark; only the question source differs.
"""

from __future__ import annotations

import json

from qatfd.benchmarks.officeqa import OfficeQABenchmark
from qatfd.config import OfficeQASynthConfig
from qatfd.constants import OFFICE_QA_SYNTH
from qatfd.paths import resolve_under_benchmarks
from qatfd.types import Question


class OfficeQASynthBenchmark(OfficeQABenchmark):
    name = OFFICE_QA_SYNTH
    config: OfficeQASynthConfig

    def __init__(self, config: OfficeQASynthConfig) -> None:
        config.qa_pairs_path = str(resolve_under_benchmarks(config.qa_pairs_path))
        super().__init__(config)

    def load_questions(self) -> list[Question]:
        with open(self.config.qa_pairs_path) as f:
            records: list[dict] = json.load(f)

        # The source `qid` is the OfficeQA uid the line of inquiry was seeded from, so all
        # ~6 questions in a block share it — but the runner keys results / resume / dedup by
        # qid, so we suffix a per-block index to make each question's qid unique. The block
        # ordering of the file is preserved (questions of one line of inquiry run
        # consecutively, which is what makes sequential working-set reuse possible).
        questions: list[Question] = []
        per_uid_counts: dict[str, int] = {}
        loi_indices: dict[str, int] = {}
        for rec in records:
            source_uid = str(rec["qid"])
            j = per_uid_counts.get(source_uid, 0)
            per_uid_counts[source_uid] = j + 1
            loi = rec["line_of_inquiry"]["line_of_inquiry"]
            loi_idx = loi_indices.setdefault(loi, len(loi_indices))
            # a rare answer is a list of acceptable phrasings; store it as JSON so the row
            # is still a string (the numeric cup scorer will score it 0 either way)
            gold = rec["answer"] if isinstance(rec["answer"], str) else json.dumps(rec["answer"])
            questions.append(
                Question(
                    qid=f"{source_uid}_q{j}",
                    text=str(rec["question"]),
                    gold=gold,
                    gold_docs=[str(d) for d in (rec.get("doc_ids") or [])],
                    meta={
                        "source_uid": source_uid,
                        "loi_idx": loi_idx,
                        "loi_doc_ids": list(rec["line_of_inquiry"].get("doc_ids", [])),
                    },
                )
            )
        return questions
