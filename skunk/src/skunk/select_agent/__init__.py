"""SelectAgent — agentic precision selection over page-index sem-filter survivors.

A `MultiTurnAgent` that mirrors the SearchAgent's architecture but runs over the catalog
+ PageStore (no ChromaDB / embeddings), seeded with the stage-1 flagged candidates.
"""

from skunk.select_agent.select_agent import SelectAgent

__all__ = ["SelectAgent"]
