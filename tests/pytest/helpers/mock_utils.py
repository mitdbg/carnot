"""Utility functions for extracting content from LiteLLM-style messages.

These helpers are used by Tier 2 mocked operator tests to inspect the
messages passed to the mocked ``litellm.completion`` handler.
"""

from __future__ import annotations


def msg_text(messages: list[dict]) -> str:
    """Concatenate all text from a litellm-style *messages* list.

    Each message is a dict with a ``content`` key that is either a string
    or a list of ``{"type": "text", "text": "..."}`` dicts (the multimodal
    format used by ``LiteLLMModel._prepare_completion_kwargs``).

    Returns:
        A single string containing all text content, joined by newlines.
    """
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict):
                    parts.append(c.get("text", ""))
                else:
                    parts.append(str(c))
        else:
            parts.append(str(content))
    return "\n".join(parts)
