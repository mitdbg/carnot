"""Teammate's search agent — vendored under one subtree.

Wired into the framework by `skunk.retrieve.RetrieveExecutor` when
`config.retriever == "search_agent"`. The agent runs an iterative
tool-using LLM loop (vector_search / retrieve_page_info / run_grep /
final_answer) over a ChromaDB index + cleaned per-page text, and
returns a list of page keys.

This first-pass merge keeps the teammate's `OpenRouter` wrapper,
`LocalPythonExecutor`, and `Tracer` intact rather than porting to the
framework's `LLMClient` / `pyexec` / `trace`. See "TODO after merge"
in ARCHITECTURE.md for the follow-up cleanup list.

`prep/` holds the offline corpus-prep scripts (page cleaner, vector
db builder, embedding computation, recall experiment). They are not
imported by the runtime path.
"""

from skunk.search_agent.base import Retriever
from skunk.search_agent.search_agent import SearchAgent

__all__ = ["Retriever", "SearchAgent"]
