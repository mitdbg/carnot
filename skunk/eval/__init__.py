"""Independent eval harnesses for OfficeQA.

Three harnesses isolate the failure modes:
  eval.eval_retrieval   — retrieve only; compares to golden pages from source_docs URLs
  eval.eval_extraction  — extract only; assumes perfect retrieval (golden pages)
  eval.eval_e2e         — full pipeline; gap to extraction-only quantifies retrieval cost

"""
