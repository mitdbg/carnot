"""Minimal parsing/utility helpers extracted from carnot.agents.utils.

Only the symbols actually needed by ``local_python_executor`` and the
``SearchAgent`` loop are kept.
"""

from __future__ import annotations

import ast
import re

BASE_BUILTIN_MODULES = [
    "collections",
    "datetime",
    "itertools",
    "math",
    "queue",
    "random",
    "re",
    "stat",
    "statistics",
    "time",
    "unicodedata",
]


MAX_LENGTH_TRUNCATE_CONTENT = 20000


def truncate_content(content: str, max_length: int = MAX_LENGTH_TRUNCATE_CONTENT) -> str:
    if len(content) <= max_length:
        return content
    return (
        content[: max_length // 2]
        + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
        + content[-max_length // 2 :]
    )


def extract_text_from_tags(text: str, tags: tuple[str, str]) -> str | None:
    """Extract text between tags from the LLM's output."""
    pattern = rf"{tags[0]}(.*?){tags[1]}"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(match.strip() for match in matches)
    return None


def parse_code_blobs(text: str, code_block_tags: tuple[str, str]) -> str:
    """Extract a python code block from the LLM's output.

    Tries the user-supplied tag pair first, then falls back to standard
    ```python / ``` markdown fences, then to parsing the entire text as
    Python.
    """
    output_code_blobs = extract_text_from_tags(text, code_block_tags)
    if not output_code_blobs:
        output_code_blobs = extract_text_from_tags(text, ("```(?:python|py)", "\n```"))
    if output_code_blobs:
        return output_code_blobs
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        pass

    if "final" in text and "answer" in text:
        raise ValueError(
            f"Your code snippet is invalid, because the regex pattern "
            f"{code_block_tags[0]}(.*?){code_block_tags[1]} was not found in it.\n"
            f"Here is your code snippet:\n{text}\n"
            f"It seems like you're trying to return the final answer, you can do it as follows:\n"
            f"{code_block_tags[0]}\nfinal_answer(\"YOUR FINAL ANSWER HERE\")\n{code_block_tags[1]}"
        )
    raise ValueError(
        f"Your code snippet is invalid, because the regex pattern "
        f"{code_block_tags[0]}(.*?){code_block_tags[1]} was not found in it.\n"
        f"Here is your code snippet:\n{text}\n"
        f"Make sure to include code with the correct pattern, for instance:\n"
        f"Thoughts: Your thoughts\n"
        f"{code_block_tags[0]}\n# Your python code here\n{code_block_tags[1]}"
    )
