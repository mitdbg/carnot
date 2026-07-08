"""Teammate's search agent — vendored under one subtree.

Wired into the framework by `skunk.retrieve.RetrieveOp` when
`config.retriever == "search_agent"`. The agent runs an iterative
tool-using LLM loop (vector_search / retrieve_page_info / run_grep, then
a final-answer ```json``` block) over a ChromaDB index + cleaned per-page
text, and returns a list of page keys.

The agent declares its prompt (`name` + `system_prompt` template) and lets
`MultiTurnAgent` assemble the `PromptedCall`; per-step events
(system / question / observation / error) flow through `ctx.emit(message)`
into the orchestrator's event stream, stamped with the enclosing `retrieve`
step's `op`.
The one remaining vendored piece — the `OpenRouter` shim — is a deferred
follow-up (see "TODO after merge" in ARCHITECTURE.md). The smolagents-derived
sandbox has been hoisted to the top-level `skunk.local_python_executor`, since
it is shared infrastructure (used by `pyexec` and `multi_turn_agent`), not
search-agent-specific.

`prep/` holds the offline corpus-prep scripts (page cleaner, vector
db builder, embedding computation, recall experiment). They are not
imported by the runtime path.
"""

from skunk.search_agent.search_agent import SearchAgent, doc_ids_from_payload

__all__ = ["SearchAgent", "doc_ids_from_payload"]
