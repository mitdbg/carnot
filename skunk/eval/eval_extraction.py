"""Extract-only eval. Builds a synthetic DocHandle from golden pages
(parsed from source_docs URLs via eval/golden.py) and calls extract() / read_visual()
per question. Fuzzy-matches against the `answer` column.

This isolates extraction quality from retrieval quality.

Usage:
  python -m eval.eval_extraction --csv data/officeqa_pro.csv --report eval/extraction_report.csv
"""

if __name__ == "__main__":
    # TODO: per question, build DocHandle from golden pages, call extract()/read_visual().
    # Fuzzy-match against answer column. Write results to eval/extraction_report.csv.
    raise NotImplementedError("eval_extraction not yet implemented.")
