"""End-to-end eval. Runs the full pipeline (planner → orchestrator → 6 subagents)
per question and fuzzy-matches against the `answer` column.

The gap between this and eval_extraction.py quantifies retrieval cost.
The gap between eval_extraction.py and the answer floor quantifies extraction cost.

Usage:
  python -m eval.eval_e2e --csv data/officeqa_pro.csv --report eval/e2e_report.csv
"""

if __name__ == "__main__":
    # TODO: full pipeline; gap vs eval_extraction quantifies retrieval cost.
    # Write results to eval/e2e_report.csv.
    raise NotImplementedError("eval_e2e not yet implemented.")
