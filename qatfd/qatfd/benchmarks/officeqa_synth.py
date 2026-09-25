"""Synthetic OfficeQA benchmark (LLM-generated QA pairs over the Treasury Bulletin corpus).

Questions: a `qa_pairs.json` produced by dataset_datagen (list of records with qid /
idx / line_of_inquiry / question / answer / doc_ids), copied under benchmarks/officeqa/.
Each ~6-question block shares one "line of inquiry" over a common set of documents, so
consecutive questions are topically related — the intended testbed for working-set reuse.
Corpus, chroma collection, document map, and recall metrics are inherited from
OfficeQABenchmark. Scoring is NOT: a generated answer is a list of nugget strings (often
several, often short sentences), so the score is LLM-judged nugget recall (see judge.py),
and the prompt hints ask the system for a complete short answer rather than a bare number.
"""

from __future__ import annotations

import json

from skunk.common import ExecutionContext

from qatfd.benchmarks.judge import judge_nugget_recall
from qatfd.benchmarks.officeqa import OfficeQABenchmark
from qatfd.config import OfficeQASynthConfig
from qatfd.constants import OFFICE_QA_SYNTH
from qatfd.paths import resolve_under_benchmarks
from qatfd.types import Question

# The gold nuggets are atomic facts (a statistic, a date, an entity, a short phrase) that the
# generator verified against Treasury Bulletin tables; most are computed values (means, changes,
# ratios), so the judge must tolerate rounding and formatting differences but not wrong figures.
_JUDGE_SYSTEM = (
    "You are a careful answer-evaluation judge for quantitative questions over U.S. Treasury "
    "Bulletin data (federal receipts, expenditures, debt, trust funds, and similar statistics). "
    "Each nugget is one key fact of the correct answer — usually a computed figure, a date, or a "
    "named entity. Judge whether the predicted answer states that fact, allowing for equivalent "
    "units, rounding to a similar precision, and paraphrase, but not for a materially different "
    "value, period, or entity."
)


class OfficeQASynthBenchmark(OfficeQABenchmark):
    name = OFFICE_QA_SYNTH
    config: OfficeQASynthConfig  # type: ignore

    # Scored by nugget recall, so the answer must state every key fact — the officeqa "bare
    # number only" hint would strip exactly the facts the judge looks for.
    answer_format_hint = (
        "Give a short, direct answer that states every value the question asks for, with the "
        "units the question specifies (e.g. '4962.46 million dollars', '12.34%'), plus any entity, "
        "date, or period needed to make each value unambiguous. One or two sentences, or a short "
        "list when several values are requested; no working or explanation."
    )
    compute_objective = (
        "the final answer to this question will be graded on nugget recall — the fraction of the "
        "reference answer's key facts (values, entities, dates) that the final answer states."
    )

    def __init__(self, config: OfficeQASynthConfig) -> None:
        assert config.judge_model, (
            "officeqa_synth scores by LLM-judged nugget recall: pass benchmarks.judge_model=<model>"
        )
        config.qa_pairs_path = str(resolve_under_benchmarks(config.qa_pairs_path))
        super().__init__(config)

    def load_questions(self) -> list[Question]:
        with open(self.config.qa_pairs_path) as f:
            records: list[dict] = json.load(f)

        # The source `qid` is the OfficeQA uid the line of inquiry was seeded from, so all
        # ~6 questions in a block share it — but the runner keys results / resume / dedup by
        # qid, so we suffix a per-block index to make each question's qid unique. The counter
        # runs over EVERY record (seeds included, even when they are skipped) so a qid means
        # the same question whether or not `include_seeds` is set. The block ordering of the
        # file is preserved (questions of one line of inquiry run consecutively, which is what
        # makes sequential working-set reuse possible).
        questions: list[Question] = []
        per_uid_counts: dict[str, int] = {}
        loi_indices: dict[str, int] = {}
        for rec in records:
            source_uid = str(rec["qid"])
            j = per_uid_counts.get(source_uid, 0)
            per_uid_counts[source_uid] = j + 1
            is_seed = rec.get("idx") is None
            if is_seed and not self.config.include_seeds:
                continue
            loi = rec["line_of_inquiry"]["line_of_inquiry"]
            loi_idx = loi_indices.setdefault(loi, len(loi_indices))
            # generated answers are a list of nugget strings; a seed answer is one string
            nuggets = [str(n) for n in rec["answer"]] if isinstance(rec["answer"], list) else [str(rec["answer"])]
            questions.append(
                Question(
                    qid=f"{source_uid}_q{j}",
                    text=str(rec["question"]),
                    gold=" | ".join(nuggets),
                    gold_docs=[str(d) for d in (rec.get("doc_ids") or [])],
                    meta={
                        "source_uid": source_uid,
                        "idx": rec.get("idx"),
                        "is_seed": is_seed,
                        "loi_idx": loi_idx,
                        "loi_doc_ids": list(rec["line_of_inquiry"].get("doc_ids", [])),
                        "nuggets": nuggets,
                        "n_docs_target": rec.get("n_docs_target"),
                        "n_nuggets_target": rec.get("n_nuggets_target"),
                    },
                )
            )
        return questions

    # ---- scoring ------------------------------------------------------------------

    async def score(self, question: Question, predicted: str, ctx: ExecutionContext) -> dict:
        return await judge_nugget_recall(
            ctx,
            question=question.text,
            nuggets=question.meta.get("nuggets", []),
            predicted=predicted,
            model=self.config.judge_model,
            judge_system=_JUDGE_SYSTEM,
            partial_credit=self.config.partial_credit,
            timeout_s=self.config.judge_timeout_s,
        )
