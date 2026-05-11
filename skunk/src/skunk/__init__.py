"""OfficeQA — declarative QA pipeline over the U.S. Treasury Bulletin corpus.

Standalone Python package. No Palimpzest dependency.

Architecture:
  question -> planner (LLM, single shot)
            -> ChainNode AST (DSL, 4 ops)
            -> orchestrator
            -> {retrieve, extract, lookup_external, compute}

`compute` is the chain terminator and subsumes formatting (it self-plans, codegens,
execs, then a verifier LLM checks the output's format/unit fits the question).

See ARCHITECTURE.md for design intent and DSL.md for the formal grammar.
"""
