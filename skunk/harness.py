from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
import pandas as pd
from chromadb.api.models.Collection import Collection
from google import genai
from google.genai import types as genai_types
from openrouter import OpenRouter
from utils import (
    CodeOutput,
    InterpreterError,
    LocalPythonExecutor,
    parse_code_blobs,
)

MODEL_ID = "google/gemini-3-flash-preview"

MODEL_CONTEXT_WINDOW = 1_000_000
MODEL_EFFECTIVE_CONTEXT_WINDOW = int(MODEL_CONTEXT_WINDOW * 0.5)
CHARS_PER_TOKEN_ESTIMATE = 4
MAX_STEPS = 20
MAX_STEPS_WARNING_STEPS_BEFORE = 3
MAX_PAGES_PER_TOOL_CALL = 20
TRACE_DIR = "search_agent_traces"
CODE_BLOCK_TAGS = ("```python", "```")
BULLETINS_DIR = "treasury_bulletins_cleaned"

# ANSI color codes for terminal output
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"


class Tracer:
    """Streams the agent trace to a file and, optionally, to the terminal.

    Use as a context manager so the file is always closed on exit::

        with Tracer("trace.txt", show_output=True) as tracer:
            agent(question, tracer=tracer)
    """

    def __init__(self, trace_filepath: str, show_output: bool = False) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(trace_filepath)), exist_ok=True)
        self._file = open(trace_filepath, "w")  # noqa: SIM115
        self.show_output = show_output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_file(self, text: str) -> None:
        self._file.write(text + "\n\n")
        self._file.flush()

    def _print(self, text: str, ansi: str = "") -> None:
        if self.show_output:
            if ansi:
                print(f"{ansi}{text}{_RESET}", flush=True)
            else:
                print(text, flush=True)

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_system(self, content: str) -> None:
        """Write the system prompt to the file only (never to the terminal)."""
        self._write_file(f"## system\n{content}")

    def log_question(self, content: str) -> None:
        """Initial user question — normal text."""
        self._write_file(f"## user\n{content}")
        self._print(content)

    def log_assistant(self, content: str) -> None:
        """LLM-generated output — green."""
        self._write_file(f"## assistant\n{content}")
        self._print(content, _GREEN)

    def log_observation(self, content: str) -> None:
        """Successful tool output — yellow."""
        self._write_file(f"## user\n{content}")
        self._print(content, _YELLOW)

    def log_error(self, content: str) -> None:
        """Parse / execution / generation error — red."""
        self._write_file(f"## error\n{content}")
        self._print(content, _RED)

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Tracer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# Tokens that are allowed to appear as the *command name* in each segment of a
# grep pipeline. Anything else causes the command to be rejected.
GREP_ALLOWED_COMMANDS = {"grep", "xargs", "head", "tail", "ls", "cat"}

# Any of these substrings, if present anywhere in a grep command, cause it to
# be rejected outright -- even if the leading command token is allowed.
GREP_FORBIDDEN_PATTERNS = [
    "rm", "mv", "cp", "dd", "mkfifo", "mkdir", "rmdir", "chmod", "chown",
    "ln ", "touch", "tee", "sudo", "su ", "kill", "pkill", "shutdown",
    "reboot", "curl", "wget", "ssh", "scp", "rsync", "ftp", "nc ", "ncat",
    "python", "node", "perl", "ruby", " sh ", "bash", "zsh", "exec",
    "eval", "source", "$(", "`", ">", "<", "&", ";",
]

OFFICEQA_SPECIAL_NOTES = """The files consist of U.S. Treasury bulletins from 1939 to 2025 Q3. From 1939-1982, every month had a single bulletin. Starting in 1983, bulletins were released quarterly (i.e. in months 3, 6, 9, and 12)."""

# ## Notes About These Files
# {special_notes}

# TODO: context compaction as a tool
SEARCH_AGENT_SYSTEM_PROMPT = """You are a helpful assistant for retrieving relevant information from a large collection of files. You will be given a question, and your task is to identify which files are relevant to answering the question. You will have {max_steps} steps to retrieve relevant information, so you do not need to retrieve everything in a single tool call. Instead, use early steps to explore files of potential relevance, and then refine your searches in later steps based on what you find. Some questions require looking up information that is not contained in the files; this will be handled by a separate agent, so you should only focus on retrieving relevant files for the remainder of the question.

## Response format
On every step, output exactly ONE fenced python code block of the form:

```python
# your code here
```

The code block must contain a single call to one of the tools listed below. Do not output any other code blocks, and do not output multiple tool calls in a single step. You may write a brief "Thoughts:" line before the code block.

## Tools
In each step, you must invoke one of the following tools:

### retrieve_page_info(year_month_page_tuples: list[tuple[int, int, int]])
This tool returns the cleaned text content for each page specified by the input list of (year, month, page_id) tuples. Don't read more than ~{max_pages} pages per tool call, as they may exceed your context window.

Example:
```python
# retrieve content for pages 19 and 26 of the December 2002 bulletin
retrieve_page_info([(2002, 12, 19), (2002, 12, 26)])

# retrieve pages 20 to 30 for every month of the 1970 bulletin
retrieve_page_info([(1970, m, p) for m in range(1, 13) for p in range(20, 31)])
```

### vector_search(query: str, top_k: int, filter_year_months: list[tuple[int, int]] | None = None, filter_page_ids: list[int] | None = None)
This tool performs a vector search over all elements (text chunks, page titles, tables, etc.) extracted from every file in the collection. The input query is embedded and the top_k most relevant elements and their year_month_page_ids are returned. You can optionally filter the search to only consider elements from certain year-months or from certain page_ids. This semantic search can help you find relevant information even when you don't know the exact keywords to grep for.

```python
# find the 100 elements most relevant to "interest rates" in the year 2008
vector_search("interest rates", top_k=100, filter_year_months=[(2008, m) for m in range(1, 13)])

# find the 50 elements most relevant to "inflation" on page 26 of every Q1 bulletin in the 2010s
vector_search("inflation", top_k=50, filter_year_months=[(y, 3) for y in range(2010, 2020)], filter_page_ids=[26])
```

### run_grep(grep_cmd: str)
This tool executes the given `grep_cmd` using `grep` (Mac OSX version) assuming the treasury bulletins are stored at `./treasury_bulletins_cleaned/treasury_bulletin_{{yyyy}}_{{mm}}_{{page_id}}.txt`. The grep utility searches any given input files, selecting lines that match one or more patterns. You may use pipes to chain commands together, but only the following commands are permitted as the leading token of each pipeline segment: grep, xargs, head, tail, ls, cat. Shell redirection, command substitution, and any other shell tokens are forbidden. Finally, use case insensitive searches when case sensitivity is not necessary to reduce the number of queries you need to make.

Example:
```python
# find all pages that mention "interest rates"
run_grep('grep -rli "interest rates" ./treasury_bulletins_cleaned/')

# find all pages that mention both "interest rates" and "inflation"
run_grep('grep -rli "interest rates" ./treasury_bulletins_cleaned/ | xargs grep -li "inflation"')
```

### final_answer(page_keys: list[str])
This tool should be called when you have retrieved all relevant information and are ready to provide a final answer to the question. The input is a list of page keys in the format "year_month_page_id" that correspond to the pages you have identified as relevant. Call this exactly once per question, and only when you are confident. Be sure to use the page_id that corresponds to the index of the page in the bulletin not the page number printed on the page itself, which may differ.

Example:
```python
final_answer(["2002_06_1", "2002_12_26"])
```

## Task
You will now be presented with the question and must execute your search before responding with a final list of relevant page keys in the format "year_month_page_id". Remember to use the tools iteratively to refine your search results, and only call final_answer when you are confident that you have identified all relevant information.
"""

# ### lookup_external(query: str)
# This tool executes a Google-backed search for the given query and returns an answer. Use this for statistics or facts that are not expected to live in the treasury bulletins themselves.

# Example:
# ```python
# # lookup the exchange rate between the USD and JPY on April 1st, 2000
# lookup_external("What was the exchange rate between the USD and JPY on April 1st, 2000?")
# ```


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _make_retrieve_page_info(clean_page_map: dict[str, list]):
    def retrieve_page_info(year_month_page_tuples):
        chunks = []
        for item in year_month_page_tuples:
            year, month, page_id = item
            key = f"{int(year)}-{int(month):02d}-{int(page_id)}"
            entry = clean_page_map.get(key)
            if entry is None:
                chunks.append(f"=== {key} ===\n[no such page (or no content on page)]")
                continue
            filepath = entry[0]
            try:
                with open(filepath) as f:
                    text = f.read()
            except OSError as e:
                chunks.append(f"=== {key} ===\n[error reading file: {e}]")
                continue
            chunks.append(f"=== {key} ({filepath}) ===\n{text}")
        return "\n\n".join(chunks)

    return retrieve_page_info


def _make_vector_search(chroma_collection: Collection, emb_model_id: str, openrouter_client: OpenRouter | genai.Client):
    def _build_where(
        filter_year_months: list[tuple[int, int]] | None,
        filter_page_ids: list[int] | None,
    ) -> dict | None:
        """Build a ChromaDB `where` filter from the optional year-month and page-id constraints.

        Metadata schema:
          year     – zero-padded string e.g. "1941"
          month    – zero-padded string e.g. "01"
          page_id  – int
        """
        clauses: list[dict] = []

        if filter_year_months:
            ym_conditions = [
                {"$and": [{"year": str(y)}, {"month": f"{m:02d}"}]}
                for y, m in filter_year_months
            ]
            # $or requires at least two items; use the condition directly for a single entry.
            if len(ym_conditions) == 1:
                clauses.append(ym_conditions[0])
            else:
                clauses.append({"$or": ym_conditions})

        if filter_page_ids:
            if len(filter_page_ids) == 1:
                clauses.append({"page_id": filter_page_ids[0]})
            else:
                clauses.append({"page_id": {"$in": filter_page_ids}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def vector_search(
        query: str,
        top_k: int,
        filter_year_months: list[tuple[int, int]] | None = None,
        filter_page_ids: list[int] | None = None,
    ) -> str:
        # embed the query using the same model that produced the stored embeddings.
        # resp = openrouter_client.embeddings.generate(input=query, model=emb_model_id)
        # query_embedding = resp.data[0].embedding  # type: ignore
        assert isinstance(openrouter_client, genai.Client)
        gemini_emb_model = emb_model_id.removeprefix("google/")
        emb_result = openrouter_client.models.embed_content(model=gemini_emb_model, contents=query)
        query_embedding = list(emb_result.embeddings[0].values)

        where = _build_where(filter_year_months, filter_page_ids)
        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where

        try:
            results = chroma_collection.query(**query_kwargs)
        except Exception as e:
            return f"[vector_search error: {e}]"

        metadatas = results["metadatas"][0]  # type: ignore
        distances = results["distances"][0]  # type: ignore

        chunks = []
        for rank, (meta, dist) in enumerate(zip(metadatas, distances, strict=True), 1):
            page_key = meta.get("page_key", "?")
            elt_type = meta.get("type", "?")
            cleaned = meta.get("cleaned", "")
            chunks.append(
                f"[{rank}] page_key={page_key} | type={elt_type} | distance={dist:.4f}\n{cleaned}"
            )
        return "\n\n".join(chunks)

    return vector_search


def _validate_grep_cmd(grep_cmd: str) -> None:
    if not isinstance(grep_cmd, str) or not grep_cmd.strip():
        raise ValueError("grep_cmd must be a non-empty string")

    for pat in GREP_FORBIDDEN_PATTERNS:
        if pat in grep_cmd:
            raise ValueError(
                f"grep_cmd rejected: contains forbidden token {pat!r}"
            )

    # Split on pipe characters that are NOT inside quotes by tokenizing first.
    try:
        all_tokens = shlex.split(grep_cmd)
    except ValueError as e:
        raise ValueError(f"grep_cmd is not valid shell syntax: {e}") from e

    # Reconstruct pipeline segments by splitting on bare "|" tokens.
    segments: list[list[str]] = []
    current: list[str] = []
    for tok in all_tokens:
        if tok == "|":
            segments.append(current)
            current = []
        else:
            current.append(tok)
    segments.append(current)

    for tokens in segments:
        if not tokens:
            raise ValueError("grep_cmd contains an empty pipeline segment")
        leading = os.path.basename(tokens[0])
        if leading not in GREP_ALLOWED_COMMANDS:
            raise ValueError(
                f"grep_cmd rejected: leading command {leading!r} is not in "
                f"the allowlist {sorted(GREP_ALLOWED_COMMANDS)}"
            )


def run_grep(grep_cmd: str) -> str:
    _validate_grep_cmd(grep_cmd)
    try:
        proc = subprocess.run(
            grep_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "[grep timed out after 30s]"
    out = proc.stdout
    if proc.returncode not in (0, 1):  # grep returns 1 when no matches found
        out += f"\n[stderr]\n{proc.stderr}"
    return out

# NOTE: tools=... may not be supported by OpenRouter
# def lookup_external(query: str) -> str:
#     """Query the model via OpenRouter."""
#     resp = _openrouter_client.chat.send(
#         model=MODEL_ID,
#         messages=[{"role": "user", "content": query}],
#         tools=[{"googleSearch": {}}],
#     )
#     return resp.choices[0].message.content or ""


def final_answer(page_keys):
    return page_keys


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
# Matches an opening code fence: ```python, ```py, or plain ```.
# Requires a newline immediately after the language tag so we don't accidentally
# match closing fences (which are followed by a newline too, but are never preceded
# by a language word — the pattern is unambiguous when used with finditer to find
# the *last* match in the accumulated buffer).
_OPEN_FENCE_RE = re.compile(r"```(?:python|py)?\n")


def _extract_all_code_blocks(text: str) -> list[str]:
    """Return every fenced python code block found in *text*, in order."""
    matches = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL)
    return [m.strip() for m in matches]


MULTIPLE_BLOCKS_REMINDER = (
    "Reminder: your previous response contained multiple code blocks. "
    "Only the first was executed. Please output exactly one "
    "```python ... ``` block per step."
)


# ---------------------------------------------------------------------------
# Analysis helpers (ground-truth comparison)
# ---------------------------------------------------------------------------

_MONTH_NAMES: dict[str, str] = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}


def _source_docs_to_page_keys(source_docs: str) -> list[str]:
    """Parse newline-separated source_docs URLs -> page keys like '1941_01_15'."""
    keys = []
    for url in source_docs.splitlines():
        url = url.strip()
        m = re.search(r"/([a-zA-Z]+)-(\d{4})-\d+\?page=(\d+)", url)
        if m:
            month_num = _MONTH_NAMES.get(m.group(1).lower())
            if month_num:
                keys.append(f"{m.group(2)}_{month_num}_{m.group(3)}")
    return keys


def _source_files_to_year_months(source_files: str) -> list[tuple[str, str]]:
    """Parse newline-separated source_files like 'treasury_bulletin_1941_01.txt' -> [(year, month)]."""
    year_months = []
    for fname in source_files.splitlines():
        fname = fname.strip()
        m = re.search(r"_(\d{4})_(\d{2})", fname)
        if m:
            year_months.append((m.group(1), m.group(2)))
    return year_months


def _coerce_page_keys(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(v) for v in value]
    except TypeError:
        return [str(value)]

class SearchAgent:
    def __init__(
        self,
        model_id: str,
        clean_page_map: dict[str, list],
        chroma_collection: Collection,
        emb_model_id: str,
        max_steps: int = MAX_STEPS,
        max_pages_per_tool_call: int = MAX_PAGES_PER_TOOL_CALL,
    ):
        self.model_id = model_id
        # self.client = OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])
        self.client: OpenRouter | genai.Client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.chroma_collection = chroma_collection
        self.clean_page_map = clean_page_map
        self.emb_model_id = emb_model_id
        self.max_steps = max_steps
        self.max_pages_per_tool_call = max_pages_per_tool_call
        self.system_prompt = SEARCH_AGENT_SYSTEM_PROMPT.format(max_steps=max_steps, max_pages=max_pages_per_tool_call)
        self.messages: list[dict] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def _build_executor(self) -> LocalPythonExecutor:
        executor = LocalPythonExecutor(additional_authorized_imports=[])
        executor.send_tools(
            {
                "retrieve_page_info": _make_retrieve_page_info(self.clean_page_map),
                "vector_search": _make_vector_search(self.chroma_collection, self.emb_model_id, self.client),
                "run_grep": run_grep,
                # "lookup_external": lookup_external,
                "final_answer": final_answer,
            }
        )
        return executor

    def _generate(self) -> str:
        # Cheap token estimate: avoids a count_tokens round-trip.
        content_str = "\n".join(m["content"] for m in self.messages)
        total_tokens = len(content_str) / CHARS_PER_TOKEN_ESTIMATE

        final_messages = self.messages
        if total_tokens > MODEL_EFFECTIVE_CONTEXT_WINDOW:
            # preserve the system prompt (index 0) and the question (index 1).
            # fill the remaining budget with as many of the most recent messages as
            # possible, then insert a placeholder between the question and those
            # recent messages to indicate that older steps were dropped.
            placeholder = {"role": "user", "content": "...(earlier steps truncated to fit context window)..."}
            fixed = [self.messages[0], self.messages[1], placeholder]
            fixed_tokens = sum(len(m["content"]) for m in fixed) / CHARS_PER_TOKEN_ESTIMATE
            remaining = MODEL_EFFECTIVE_CONTEXT_WINDOW - fixed_tokens

            recent: list[dict] = []
            for msg in reversed(self.messages[2:]):
                cost = len(msg["content"]) / CHARS_PER_TOKEN_ESTIMATE
                if remaining - cost < 0:
                    break
                recent.append(msg)
                remaining -= cost

            # recent is currently newest-first; reverse to restore chronological order.
            final_messages = fixed + recent[::-1]

        # Stream tokens and stop as soon as a complete ```python...``` block
        # has been received.  This avoids waiting for the model to finish its
        # full "thinking" output after the code block is already parseable.
        # stream = self.client.chat.send(
        #     model=self.model_id,
        #     messages=final_messages,  # type: ignore
        #     stream=True,
        # )  # type: ignore
        assert isinstance(self.client, genai.Client)
        gemini_chat_model = self.model_id.removeprefix("google/")
        _system_instruction = next(
            (m["content"] for m in final_messages if m["role"] == "system"), None
        )
        _genai_contents = [
            genai_types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[genai_types.Part.from_text(text=m["content"])],
            )
            for m in final_messages
            if m["role"] != "system"
        ]
        _genai_config = genai_types.GenerateContentConfig(
            system_instruction=_system_instruction,
        )
        stream = self.client.models.generate_content_stream(
            model=gemini_chat_model,
            contents=_genai_contents,
            config=_genai_config,
        )

        accumulated = ""
        code_block_closed = False
        in_code_block = False
        for chunk in stream:
            # delta = chunk.choices[0].delta.content or ""  # type: ignore (OpenRouter)
            delta = chunk.text or ""  # genai
            accumulated += delta

            # Track whether we're inside an opening code fence and detect
            # the closing ``` so we can stop early.  Handles ```python,
            # ```py, and plain ``` fences.
            if not in_code_block:
                if _OPEN_FENCE_RE.search(accumulated):
                    in_code_block = True
            else:
                first_fence = _OPEN_FENCE_RE.search(accumulated)
                if first_fence is not None:
                    # tail is everything after the full opening fence line
                    tail = accumulated[first_fence.end():]
                    close_idx = tail.find("```")
                    if close_idx != -1:
                        code_block_closed = True
                        break

        try:  # noqa: SIM105
            stream.close()  # type: ignore[union-attr]
        except Exception:
            pass

        if not accumulated:
            return ""

        # If we stopped early after detecting a closed code block, trim any
        # partial tokens that arrived after the closing fence.
        if code_block_closed:
            first_fence = _OPEN_FENCE_RE.search(accumulated)
            if first_fence is not None:
                tail = accumulated[first_fence.end():]
                close_idx_in_tail = tail.find("```")
                accumulated = accumulated[: first_fence.end() + close_idx_in_tail + 3]

        return accumulated

    def __call__(self, question: str, tracer: Tracer | None = None) -> list[str]:
        """
        Given a question, return a list of page keys (in the format
        "year_month_page_id") that are relevant to answering the question.

        If *tracer* is provided, every message is streamed to its file
        (and to the terminal when ``show_output=True``) as it is produced.
        """
        # Reset per-question state.
        self._completed = False
        self._num_steps = 0
        self._error: str | None = None

        # Reset per-question conversation but keep the system prompt.
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.messages.append({"role": "user", "content": f"Question: {question}"})

        if tracer is not None:
            tracer.log_system(self.system_prompt)
            tracer.log_question(f"Question: {question}")

        executor = self._build_executor()

        for step in range(self.max_steps):
            try:
                assistant_text = self._generate()
            except Exception as e:
                error_msg = f"[generation error: {e}]"
                self._error = error_msg
                self.messages.append({"role": "user", "content": error_msg})
                if tracer is not None:
                    tracer.log_error(error_msg)
                break

            self.messages.append({"role": "assistant", "content": assistant_text})
            if tracer is not None:
                tracer.log_assistant(assistant_text)

            all_blocks = _extract_all_code_blocks(assistant_text)
            multiple_blocks = len(all_blocks) > 1
            if all_blocks:
                code = all_blocks[0]
            else:
                # No fenced blocks — fall back to raw-python heuristic
                try:
                    code = parse_code_blobs(assistant_text, CODE_BLOCK_TAGS)
                except ValueError as e:
                    obs = (
                        f"Observation (step {step + 1}): could not parse a "
                        f"python code block from your response.\n{e}"
                    )
                    self.messages.append({"role": "user", "content": obs})
                    if tracer is not None:
                        tracer.log_error(obs)
                    continue

            if not code.strip():
                obs = (
                    f"Observation (step {step + 1}): your response contained an "
                    f"empty code block. Please output a non-empty "
                    f"```python ... ``` block with a single tool call."
                )
                self.messages.append({"role": "user", "content": obs})
                if tracer is not None:
                    tracer.log_error(obs)
                continue

            try:
                out: CodeOutput = executor(code)
            except InterpreterError as e:
                obs = f"Observation (step {step + 1}): execution failed.\n{e}"
                self.messages.append({"role": "user", "content": obs})
                if tracer is not None:
                    tracer.log_error(obs)
                continue
            except Exception as e:
                obs = (
                    f"Observation (step {step + 1}): tool raised "
                    f"{type(e).__name__}: {e}"
                )
                self.messages.append({"role": "user", "content": obs})
                if tracer is not None:
                    tracer.log_error(obs)
                continue

            observation_parts = []
            if out.logs:
                observation_parts.append(f"[stdout]\n{out.logs}")
            if out.output is not None:
                observation_parts.append(f"[result]\n{out.output}")
            observation = (
                "\n".join(observation_parts) if observation_parts else "[no output]"
            )
            observation = f"Observation (step {step + 1}):\n{observation}"

            self.messages.append({"role": "user", "content": observation})
            if tracer is not None:
                tracer.log_observation(observation)
            self._num_steps = step + 1

            if multiple_blocks:
                self.messages.append({"role": "user", "content": MULTIPLE_BLOCKS_REMINDER})
                if tracer is not None:
                    tracer.log_observation(MULTIPLE_BLOCKS_REMINDER)

            if out.is_final_answer:
                self._completed = True
                return _coerce_page_keys(out.output)

        if self._error is None:
            self._error = "max steps"
        return []


def _run_one(
    row: dict,
    model_id: str,
    clean_page_map: dict,
    chroma_collection: Collection,
    emb_model_id: str,
    show_output: bool,
) -> tuple[str, dict]:
    """Run the search agent for a single question and return an analysis dict."""
    start_time = time.perf_counter()
    uid = row["uid"]
    question = row["question"]
    trace_path = f"{TRACE_DIR}/{uid}_trace.txt"
    # Each thread gets its own SearchAgent so mutable state is never shared.
    agent = SearchAgent(
        model_id,
        clean_page_map=clean_page_map,
        chroma_collection=chroma_collection,
        emb_model_id=emb_model_id,
    )
    with Tracer(trace_path, show_output=show_output) as tracer:
        page_keys = agent(question, tracer=tracer)

    # Compute accuracy against ground truth.
    source_page_keys = _source_docs_to_page_keys(str(row.get("source_docs", "")))
    source_year_months = _source_files_to_year_months(str(row.get("source_files", "")))

    correct_page = 0.0
    correct_document = 0.0
    if agent._completed and page_keys:
        if source_page_keys:
            correct_page = sum(1 for k in source_page_keys if k in page_keys) / len(source_page_keys)
        if source_year_months:
            def _key_ym(k: str) -> tuple[str, str] | None:
                parts = k.split("_")
                return (parts[0], parts[1]) if len(parts) >= 2 else None
            final_yms = {_key_ym(k) for k in page_keys} - {None}
            correct_document = sum(1 for ym in source_year_months if ym in final_yms) / len(source_year_months)

    # persist the full message history for later debugging.
    messages_path = f"{TRACE_DIR}/{uid}_messages.json"
    with open(messages_path, "w") as f:
        json.dump(agent.messages, f, indent=2)

    analysis = {
        "uid": uid,
        "completed": agent._completed,
        "correct_page": correct_page,
        "correct_document": correct_document,
        "num_steps": agent._num_steps,
        "error": agent._error,
        "page_keys": page_keys,
        "total_time_sec": time.perf_counter() - start_time,
    }
    return uid, analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the treasury bulletin search agent.")
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=TRACE_DIR,
        help=f"Directory to save agent traces (default: {TRACE_DIR})",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Stream the agent trace to the terminal in addition to the trace file.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=4,
        help="Number of questions to process in parallel (default: 4).",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=".chromadb",
        help="Directory where ChromaDB stores its data (default: .chromadb)",
    )
    parser.add_argument(
        "--chroma-collection-name",
        type=str,
        default="gemini",
        help="Name of the ChromaDB collection to use for embeddings (default: gemini-embedding-2)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help=f"ID of the language model to use (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--emb-model-id",
        type=str,
        default="google/gemini-embedding-2-preview",
        help="ID of the embedding model to use (default: google/gemini-embedding-2-preview)",
    )
    args = parser.parse_args()

    # step 0: initialize directories for agent traces
    os.makedirs(args.trace_dir, exist_ok=True)

    # step 1: load questions
    officeqa_df = pd.read_csv("officeqa_pro.csv")

    # step 2: load mapping from "year-month-page_id" --> [clean page text file path, sorted elements order]
    with open(f"{BULLETINS_DIR}/clean_page_map.json") as f:
        clean_page_map = json.load(f)

    # step 2.5: load chromadb collection to ensure it's ready before we start processing questions
    client = chromadb.PersistentClient(path=args.chroma_dir)
    collection = client.get_collection(args.chroma_collection_name)

    # step 3: filter out questions whose traces already exist
    rows = [
        row
        for _, row in officeqa_df.iterrows()
        if not os.path.exists(f"{args.trace_dir}/{row['uid']}_trace.txt")
    ]
    skipped = len(officeqa_df) - len(rows)
    if skipped:
        print(f"Skipping {skipped} question(s) with existing traces.")

    # step 4: run questions (debug: sequential loop with pdb breakpoint)
    # with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
    #     futures = {
    #         pool.submit(_run_one, row, args.model_id, clean_page_map, collection, args.emb_model_id, args.show_output): row # type: ignore
    #         for row in rows
    #     }
    #     for future in as_completed(futures):
    #         row = futures[future]
    #         uid = row["uid"]
    #         try:
    #             uid, analysis = future.result()
    #         except Exception as e:
    #             print(f"ERROR for UID {uid}: {e}")
    #             analysis = {
    #                 "uid": uid,
    #                 "completed": False,
    #                 "correct_page": 0.0,
    #                 "correct_document": 0.0,
    #                 "num_steps": 0,
    #                 "error": str(e),
    #                 "page_keys": [],
    #             }
    #         with open(f"{args.trace_dir}/{uid}_analysis.json", "w") as f:
    #             json.dump(analysis, f, indent=2)
    #         print(f"============= END OF TRACE: {uid} ==================")
    #         print(f"UID: {uid}")
    #         print(f"Question: {row['question']}")
    #         print(f"Analysis: {json.dumps(analysis, indent=2)}")
    #         print(f"Ground Truth Source Docs: {row['source_docs']}")
    #         print(f"Ground Truth Source Files: {row['source_files']}")
    #         print("===========================================")
    # import pdb
    for row in rows:
        uid = row["uid"]
        try:
            uid, analysis = _run_one(row, args.model_id, clean_page_map, collection, args.emb_model_id, args.show_output)
        except Exception as e:
            print(f"ERROR for UID {uid}: {e}")
            analysis = {
                "uid": uid,
                "completed": False,
                "correct_page": 0.0,
                "correct_document": 0.0,
                "num_steps": 0,
                "error": str(e),
                "page_keys": [],
            }
        with open(f"{args.trace_dir}/{uid}_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        print(f"============= END OF TRACE: {uid} ==================")
        print(f"UID: {uid}")
        print(f"Question: {row['question']}")
        print(f"Analysis: {json.dumps(analysis, indent=2)}")
        print(f"Ground Truth Source Docs: {row['source_docs']}")
        print(f"Ground Truth Source Files: {row['source_files']}")
        print("===========================================")
        # pdb.set_trace()
