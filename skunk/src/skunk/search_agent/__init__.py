"""Teammate's search agent — vendored under one subtree.

Wired into the framework by `skunk.retrieve.RetrieveExecutor` when
`config.retriever == "search_agent"`. The agent runs an iterative
tool-using LLM loop (vector_search / retrieve_page_info / run_grep /
final_answer) over a ChromaDB index + cleaned per-page text, and
returns a list of page keys.

The agent's system prompt is assembled by `SearchAgentPromptedCall`
(subclass of `skunk.prompted_call.PromptedCall`); per-step events
(system / question / assistant / observation / error) flow through
`ctx.emit("search_agent", …)` into the orchestrator's `QuestionTrace`.
The remaining vendored pieces — `OpenRouter` shim and `LocalPythonExecutor`
— are deferred follow-ups (see "TODO after merge" in ARCHITECTURE.md).

`prep/` holds the offline corpus-prep scripts (page cleaner, vector
db builder, embedding computation, recall experiment). They are not
imported by the runtime path.
"""

from skunk.search_agent.base import Retriever
from skunk.search_agent.search_agent import SearchAgent

__all__ = ["Retriever", "SearchAgent"]
