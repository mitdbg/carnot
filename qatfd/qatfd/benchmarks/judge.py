"""LLM-as-judge for nugget-based benchmarks.

KARL (arXiv:2603.05218) unifies evaluation via nugget-based completion. Two flavors:
  - `judge_single_nugget` (BrowseComp-Plus): the answer is a single nugget that must be
    predicted correctly — a deterministic YES/NO judge.
  - `judge_nugget_recall` (TREC-BioGen): the answer is graded against many gold nuggets;
    the score is nugget recall (fraction supported), using KARL's D.1 completeness prompt.
"""

from __future__ import annotations

import ast
import re

_JUDGE_SYSTEM = (
    "You are grading a question-answering system. You are given a question, the "
    "gold (correct) answer, and a predicted answer. Decide whether the predicted "
    "answer contains the correct answer to the question — i.e. whether it is "
    "semantically equivalent to the gold answer (same entity / value / fact), "
    "allowing for paraphrase, extra context, and formatting differences. Ignore "
    "style, verbosity, and hedging; judge only correctness of the key nugget.\n\n"
    "Respond on the FIRST line with exactly 'YES' (correct) or 'NO' (incorrect), "
    "then optionally a one-line justification."
)

_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _parse_yes_no(text: str) -> bool:
    """True iff the judge said YES. Reads the first yes/no token; defaults to False."""
    m = _YESNO_RE.search(text or "")
    return bool(m) and m.group(1).lower() == "yes"


async def judge_single_nugget(ctx, *, question: str, gold: str, predicted: str, model: str) -> dict:
    user = f"Question:\n{question}\n\nGold answer:\n{gold}\n\nPredicted answer:\n{predicted or '(no answer)'}"
    resp = await ctx.llm_client.acall(
        system=_JUDGE_SYSTEM,
        user=user,
        temperature=0.0,
        model=model,
        ctx=ctx,
        call_site="bcp_judge",
    )
    rationale = (resp.text or "").strip()
    return {
        "score": float(_parse_yes_no(rationale)),
        "scorer": "karl.single_nugget.v1",
        "judge_rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Multi-nugget completion (TREC-BioGen)
# ---------------------------------------------------------------------------

# KARL's nugget-completeness prompt (paper appendix D.1, Figure 31), verbatim modulo the
# {placeholders}. The judge labels every gold nugget support / partial_support / not_support
# in one call, returning a Python list of label strings in nugget order.
_NUGGET_COMPLETENESS_PROMPT = """\
Your Role: You will evaluate whether an answer to a question (which can include a code snippet or documentation) sufficiently supports each decompositional fact.
Process:
1. Read the question and the answer.
2. Read each of the {length} decompositional facts carefully one by one.
3. Based on the question and answer, judge whether the answer supports, partially supports, or does not support each decompositional fact. Read every fact and document pair carefully as you would when proofreading.
It may be helpful to ask yourself: "Does the answer provide sufficient evidence required to support the decompositional fact?" Be sure to check all of the information in the answer.
Label Definitions:
- support: The answer fully captures and entails all necessary parts of the decompositional fact.
- partial_support: The answer partially captures the decompositional fact, but does not fully capture all necessary parts.
- not_support: The answer does not capture or does not provide information entailing the decompositional fact.
Output Format: Return the labels as a Python list of strings (List[str]), in the same order as the decompositional facts. Provide a label for each fact. Do not provide any explanation or reasoning.
["support", "not_support", "partial_support", ...]
Input:
Question: {question}
Answer: {answer}
Decompositional Facts: {nugget}
Labels:"""

_LABEL_WEIGHTS = {"support": 1.0, "partial_support": None, "not_support": 0.0}
_LIST_RE = re.compile(r"\[.*?\]", re.DOTALL)


def _parse_labels(text: str, n: int) -> list[str]:
    """Extract the judge's `["support", ...]` list. Returns a list of `n` labels, padding
    missing/unparseable entries with `not_support` (the conservative, zero-credit default)."""
    labels: list[str] = []
    m = _LIST_RE.search(text or "")
    if m:
        try:
            parsed = ast.literal_eval(m.group(0))
            if isinstance(parsed, (list, tuple)):
                labels = [str(x).strip().lower() for x in parsed]
        except (ValueError, SyntaxError):
            labels = []
    # normalize unknown labels to not_support; pad/truncate to exactly n.
    labels = [lab if lab in _LABEL_WEIGHTS else "not_support" for lab in labels]
    if len(labels) < n:
        labels += ["not_support"] * (n - len(labels))
    return labels[:n]


async def judge_nugget_recall(
    ctx,
    *,
    question: str,
    nuggets: list[str],
    predicted: str,
    model: str,
    judge_system: str,
    partial_credit: float = 0.0,
) -> dict:
    """Nugget recall (KARL TREC-BioGen): one judge call labels every gold nugget against the
    predicted answer; score = (n_support + partial_credit * n_partial) / n_nuggets.

    `judge_system` is the grader-persona system prompt, supplied per benchmark (TREC-BioGen uses a
    biomedical persona; QAMPARI a neutral encyclopedic one) so the same machinery grades each
    domain's nuggets appropriately."""
    if not nuggets:
        return {"score": 0.0, "scorer": "karl.nugget_completion.v1", "judge_rationale": "(no gold nuggets)"}

    resp = await ctx.llm_client.acall(
        system=judge_system,
        user=_NUGGET_COMPLETENESS_PROMPT.format(
            length=len(nuggets),
            question=question,
            answer=predicted or "(no answer)",
            nugget=nuggets,
        ),
        temperature=0.0,
        model=model,
        ctx=ctx,
        call_site="biogen_nugget_judge",
    )
    labels = _parse_labels(resp.text or "", len(nuggets))
    weights = [partial_credit if lab == "partial_support" else _LABEL_WEIGHTS[lab] for lab in labels]
    score = sum(weights) / len(nuggets)
    n_sup = labels.count("support")
    n_par = labels.count("partial_support")

    # Persist the per-nugget verdicts (discarded by the aggregate score) as a structured event, so
    # the trace viewer can show every gold nugget colored by whether the answer supported it.
    if ctx is not None:
        ctx.emit(
            f"nugget_judge n_support={n_sup} n_partial={n_par} n_nuggets={len(nuggets)} recall={score:.3f}",
            kind="observation",
            data={
                "n_support": n_sup,
                "n_partial": n_par,
                "n_nuggets": len(nuggets),
                "partial_credit": partial_credit,
                "recall": score,
                "nuggets": [
                    {"nugget": n, "label": lab} for n, lab in zip(nuggets, labels, strict=True)
                ],
            },
        )

    return {
        "score": score,
        "scorer": "karl.nugget_completion.v1",
        "judge_rationale": f"{n_sup} support + {n_par} partial / {len(nuggets)} nuggets -> recall {score:.3f}",
    }
