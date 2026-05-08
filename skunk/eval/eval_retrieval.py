"""Retrieve-only eval. Calls retrieve() per question and compares to golden pages.

Reports:
  - bulletin-precision        — retrieved bulletin matches golden bulletin
  - page-precision-strict     — retrieved page == golden page
  - page-precision-loose(±2)  — retrieved page within ±2 of golden
  - drift histogram           — retrieved month - golden month

Usage:
  python -m eval.eval_retrieval --csv data/officeqa_pro.csv --report eval/retrieval_report.csv
"""

if __name__ == "__main__":
    # TODO: per question, call retrieve(), compare to golden pages via eval/golden.py.
    # Report bulletin-precision, page-precision-strict, page-precision-loose(±2), offset histogram.
    # Write results to eval/retrieval_report.csv.
    raise NotImplementedError("eval_retrieval not yet implemented.")
