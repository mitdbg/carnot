"""OfficeQA — declarative QA pipeline over the U.S. Treasury Bulletin corpus.

Standalone Python package. No Palimpzest dependency.

Architecture:
  question -> planner (LLM, single shot)
            -> ChainNode AST (DSL, 6 ops)
            -> orchestrator
            -> {retrieve, extract, read_visual, lookup_external, compute, format}

See ARCHITECTURE.md for design intent and DSL.md for the formal grammar.
"""
