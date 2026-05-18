from __future__ import annotations

import os
import pathlib
import re

import yaml
from chromadb.api.models.Collection import Collection
from google import genai
from google.genai import types as genai_types  # noqa: F401
from jinja2 import Template
from openrouter import OpenRouter
from utils import (
    CodeOutput,
    InterpreterError,
    LocalPythonExecutor,
    parse_code_blobs,
)

from skunk.logging.tracer import Tracer
from skunk.retrieve.base import Retriever
from skunk.retrieve.search_tools import _make_retrieve_page_info, _make_vector_search, final_answer, run_grep

MODEL_CONTEXT_WINDOW = 1_000_000
MODEL_EFFECTIVE_CONTEXT_WINDOW = int(MODEL_CONTEXT_WINDOW * 0.5)
CHARS_PER_TOKEN_ESTIMATE = 4
MAX_STEPS = 20
MAX_STEPS_WARNING_STEPS_BEFORE = 3
MAX_PAGES_PER_TOOL_CALL = 20
CODE_BLOCK_TAGS = ("```python", "```")

_PROMPTS_FILE = pathlib.Path(__file__).parent / "prompts.yaml"
with _PROMPTS_FILE.open() as _f:
    _PROMPTS = yaml.safe_load(_f)

SEARCH_AGENT_SYSTEM_PROMPT: str = _PROMPTS["search_agent_system_prompt"]

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

def _coerce_page_keys(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(v) for v in value]
    except TypeError:
        return [str(value)]


class SearchAgent(Retriever):
    """
    A SearchAgent is a Retriever that uses an LLM with access to:

    - vector search
    - grep
    - page lookups

    To retrieve documents relevant to a question.
    """
    
    def __init__(
        self,
        model_id: str,
        clean_page_map: dict[str, list],
        chroma_collection: Collection,
        emb_model_id: str,
        tracer: Tracer | None = None,
        max_steps: int = MAX_STEPS,
        max_pages_per_tool_call: int = MAX_PAGES_PER_TOOL_CALL,
    ):
        self.model_id = model_id
        self.client: OpenRouter | genai.Client = OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])
        # self.client: OpenRouter | genai.Client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.chroma_collection = chroma_collection
        self.clean_page_map = clean_page_map
        self.emb_model_id = emb_model_id
        self.tracer = tracer
        self.max_steps = max_steps
        self.max_pages_per_tool_call = max_pages_per_tool_call
        self.system_prompt = Template(SEARCH_AGENT_SYSTEM_PROMPT).render(max_steps=max_steps, max_pages=max_pages_per_tool_call)
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
        stream = self.client.chat.send(  # type: ignore
            model=self.model_id,
            messages=final_messages,  # type: ignore
            stream=True,
        )  # type: ignore
        # assert isinstance(self.client, genai.Client)
        # gemini_chat_model = self.model_id.removeprefix("google/")
        # _system_instruction = next(
        #     (m["content"] for m in final_messages if m["role"] == "system"), None
        # )
        # _genai_contents = [
        #     genai_types.Content(
        #         role="model" if m["role"] == "assistant" else "user",
        #         parts=[genai_types.Part.from_text(text=m["content"])],
        #     )
        #     for m in final_messages
        #     if m["role"] != "system"
        # ]
        # _genai_config = genai_types.GenerateContentConfig(
        #     system_instruction=_system_instruction,
        # )
        # stream = self.client.models.generate_content_stream(
        #     model=gemini_chat_model,
        #     contents=_genai_contents,
        #     config=_genai_config,
        # )

        accumulated = ""
        code_block_closed = False
        in_code_block = False
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""  # type: ignore (OpenRouter)
            # delta = chunk.text or ""  # genai
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

    def retrieve(self, question: str) -> list[str]:
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

        if self.tracer is not None:
            self.tracer.log_system(self.system_prompt)
            self.tracer.log_question(f"Question: {question}")

        executor = self._build_executor()

        for step in range(self.max_steps):
            try:
                assistant_text = self._generate()
            except Exception as e:
                error_msg = f"[generation error: {e}]"
                self._error = error_msg
                self.messages.append({"role": "user", "content": error_msg})
                if self.tracer is not None:
                    self.tracer.log_error(error_msg)
                break

            self.messages.append({"role": "assistant", "content": assistant_text})
            if self.tracer is not None:
                self.tracer.log_assistant(assistant_text)

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
                    if self.tracer is not None:
                        self.tracer.log_error(obs)
                    continue

            if not code.strip():
                obs = (
                    f"Observation (step {step + 1}): your response contained an "
                    f"empty code block. Please output a non-empty "
                    f"```python ... ``` block with a single tool call."
                )
                self.messages.append({"role": "user", "content": obs})
                if self.tracer is not None:
                    self.tracer.log_error(obs)
                continue

            try:
                out: CodeOutput = executor(code)
            except InterpreterError as e:
                obs = f"Observation (step {step + 1}): execution failed.\n{e}"
                self.messages.append({"role": "user", "content": obs})
                if self.tracer is not None:
                    self.tracer.log_error(obs)
                continue
            except Exception as e:
                obs = (
                    f"Observation (step {step + 1}): tool raised "
                    f"{type(e).__name__}: {e}"
                )
                self.messages.append({"role": "user", "content": obs})
                if self.tracer is not None:
                    self.tracer.log_error(obs)
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
            if self.tracer is not None:
                self.tracer.log_observation(observation)
            self._num_steps = step + 1

            if multiple_blocks:
                self.messages.append({"role": "user", "content": MULTIPLE_BLOCKS_REMINDER})
                if self.tracer is not None:
                    self.tracer.log_observation(MULTIPLE_BLOCKS_REMINDER)

            if out.is_final_answer:
                self._completed = True
                return _coerce_page_keys(out.output)

        if self._error is None:
            self._error = "max steps"

        return []
