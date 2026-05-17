"""OfficeQA — declarative QA pipeline over the U.S. Treasury Bulletin corpus.

Standalone Python package. No Palimpzest dependency.

Architecture:
  question -> PlannerExecutor (LLM, single shot; src/skunk/plan.py)
            -> Plan (one branches list + computation + presentation)
            -> orchestrator
            -> {retrieve, extract, lookup_external, compute}

`compute` is the chain terminator and subsumes formatting (it self-plans, codegens,
execs, then self-critiques the result against the question with full context).

See ARCHITECTURE.md for design intent + the canonical Plan shape.
"""
