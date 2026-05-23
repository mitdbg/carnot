from __future__ import annotations

import os
import pathlib
import re
from dataclasses import dataclass

import yaml
from chromadb.api.models.Collection import Collection
from google import genai
from google.genai import types as genai_types  # noqa: F401
from jinja2 import Template
from openrouter import OpenRouter

from skunk.logging.tracer import Tracer
from skunk.retrieve.base import Retriever
from skunk.retrieve.search_tools import (
    GREP_RESULT_TAG,
    PRUNE_RESULT_TAG,
    READ_DOCUMENT_RESULT_TAG,
    SEARCH_RESULT_TAG,
    make_search_tools,
)
from skunk.utils import (
    CodeOutput,
    InterpreterError,
    LocalPythonExecutor,
    parse_code_blobs,
)

MODEL_CONTEXT_WINDOW = 1_000_000
EFFECTIVE_CONTEXT_FRACTION = 0.5
HARD_CONTEXT_FRACTION = 0.8
CHARS_PER_TOKEN_ESTIMATE = 4
MAX_STEPS = 20
MAX_STEPS_WARNING_STEPS_BEFORE = 3
MAX_PAGES_PER_TOOL_CALL = 20
CODE_BLOCK_TAGS = ("```python", "```")

# names of tools available to the agent. Used to detect which tool a code
# block invokes for the purposes of the hard-context-cutoff restriction.
_TOOL_NAMES = (
    "search_corpus",
    "grep_corpus",
    "read_document",
    "prune",
    "final_answer",
)
_TOOL_CALL_RE = re.compile(
    r"(?<![\w.])(" + "|".join(_TOOL_NAMES) + r")\s*\("
)
_RESTRICTED_HARD_TOOLS = {"prune", "final_answer"}

_PROMPTS_FILE = pathlib.Path(__file__).parent / "prompts.yaml"
with _PROMPTS_FILE.open() as _f:
    _PROMPTS = yaml.safe_load(_f)

SEARCH_AGENT_SYSTEM_PROMPT: str = _PROMPTS["search_agent_system_prompt"]
OFFICEQA_SPECIAL_NOTES: str = _PROMPTS["officeqa_special_notes"]

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


# ---------------------------------------------------------------------------
# Message-block representation
#
# We store `self.messages` as a list of {role, blocks} entries where each
# block is either a TextBlock (always rendered) or a ChunkBlock (filtered
# out at render time when its chunk_id or doc_id has been pruned).  This
# lets us preserve the complete search trajectory verbatim -- essential for
# downstream reward calculation -- while feeding the LLM a redacted view in
# `_render_for_llm`.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class ChunkBlock:
    chunk_id: str | None
    doc_id: str
    text: str


Block = TextBlock | ChunkBlock


@dataclass
class _StepOutcome:
    """Result of one `_run_step` iteration."""

    should_break: bool = False
    final_page_keys: list[str] | None = None


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
        bulletins_dir: str | None = None,
        model_context_window: int = MODEL_CONTEXT_WINDOW,
        effective_context_fraction: float = EFFECTIVE_CONTEXT_FRACTION,
        hard_context_fraction: float = HARD_CONTEXT_FRACTION,
        train: bool = False,
        special_notes: str = "",
    ):
        self.model_id = model_id
        self.client: OpenRouter | genai.Client = OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])
        # self.client: OpenRouter | genai.Client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.chroma_collection = chroma_collection
        self.clean_page_map = clean_page_map
        self.emb_model_id = emb_model_id
        self.bulletins_dir = bulletins_dir
        self.tracer = tracer
        self.max_steps = max_steps
        self.max_pages_per_tool_call = max_pages_per_tool_call
        self.model_context_window = model_context_window
        self.effective_context_fraction = effective_context_fraction
        self.hard_context_fraction = hard_context_fraction
        self.effective_context_tokens = int(model_context_window * effective_context_fraction)
        self.hard_context_tokens = int(model_context_window * hard_context_fraction)
        self.train = train
        self.system_prompt = Template(SEARCH_AGENT_SYSTEM_PROMPT).render(
            max_steps=max_steps,
            max_pages=max_pages_per_tool_call,
            special_notes=special_notes,
        )

        # NOTE: messages are stored as {role, blocks: list[Block]} so the
        # full trajectory is preserved. Use `_render_for_llm()` to get the
        # filtered, plain-text view sent to the chat API.
        self.messages: list[dict] = [
            {"role": "system", "blocks": [TextBlock(self.system_prompt)]}
        ]
        self._pruned_chunk_ids: set[str] = set()
        self._pruned_doc_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Message storage / rendering
    # ------------------------------------------------------------------

    def _append_message(self, role: str, blocks: list[Block]) -> None:
        self.messages.append({"role": role, "blocks": list(blocks)})

    def _append_text(self, role: str, text: str) -> None:
        self._append_message(role, [TextBlock(text)])

    def _block_is_visible(self, block: Block) -> bool:
        """Return True if *block* should appear in the LLM-facing render."""
        if isinstance(block, ChunkBlock):
            if block.chunk_id is not None and block.chunk_id in self._pruned_chunk_ids:
                return False
            if block.doc_id in self._pruned_doc_ids:
                return False
        return True

    def _render_for_llm(self) -> list[dict]:
        """Return the messages as a list of {role, content} for the chat API.

        Pruned `ChunkBlock`s are simply omitted (no '(redacted)' placeholder).
        Messages whose surviving blocks are all empty are dropped entirely.
        """
        rendered: list[dict] = []
        for msg in self.messages:
            parts: list[str] = []
            for block in msg["blocks"]:
                if not self._block_is_visible(block):
                    continue
                if block.text:
                    parts.append(block.text)
            if not parts:
                continue
            rendered.append({"role": msg["role"], "content": "\n\n".join(parts)})
        return rendered

    def _estimate_tokens(self, rendered: list[dict] | None = None) -> int:
        """Estimate token usage on the LLM-facing (post-prune) render."""
        if rendered is None:
            rendered = self._render_for_llm()
        return sum(len(m["content"]) for m in rendered) // CHARS_PER_TOKEN_ESTIMATE

    def messages_to_jsonable(self) -> list[dict]:
        """Return a JSON-serializable copy of the full message trajectory.

        Each block becomes a plain dict tagged with its type. Use this for
        persistence and downstream reward computation -- the raw
        `self.messages` contains dataclass instances that `json.dump`
        cannot serialize.
        """
        out: list[dict] = []
        for msg in self.messages:
            blocks_json: list[dict] = []
            for block in msg["blocks"]:
                if isinstance(block, ChunkBlock):
                    blocks_json.append(
                        {
                            "type": "chunk",
                            "chunk_id": block.chunk_id,
                            "doc_id": block.doc_id,
                            "text": block.text,
                        }
                    )
                else:
                    blocks_json.append({"type": "text", "text": block.text})
            out.append({"role": msg["role"], "blocks": blocks_json})
        return out

    @staticmethod
    def _detect_tool_call(code: str) -> str | None:
        """Return the name of the first tool invoked in *code*, if any."""
        m = _TOOL_CALL_RE.search(code)
        return m.group(1) if m else None

    def _build_executor(self) -> LocalPythonExecutor:
        executor = LocalPythonExecutor(additional_authorized_imports=[])
        tools = make_search_tools(
            chroma_collection=self.chroma_collection,
            emb_model_id=self.emb_model_id,
            openrouter_client=self.client,
            clean_page_map=self.clean_page_map,
            pruned_chunk_ids=self._pruned_chunk_ids,
            pruned_doc_ids=self._pruned_doc_ids,
            bulletins_dir=self.bulletins_dir,
        )
        executor.send_tools(tools)
        return executor

    # ------------------------------------------------------------------
    # Per-step helpers
    # ------------------------------------------------------------------

    def _generate(self) -> str:
        """Generate one assistant turn from the redacted message render.

        No message truncation is performed here: staying under the context
        window is the model's responsibility, exercised via `prune(...)`.
        """
        final_messages = self._render_for_llm()

        # Stream tokens and stop as soon as a complete ```python...``` block
        # has been received. Avoids waiting for the model to finish its
        # full "thinking" output after the code block is already parseable.
        stream = self.client.chat.send(  # type: ignore
            model=self.model_id,
            messages=final_messages,  # type: ignore
            stream=True,
        )  # type: ignore

        accumulated = ""
        code_block_closed = False
        in_code_block = False
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""  # type: ignore (OpenRouter)
            accumulated += delta

            if not in_code_block:
                if _OPEN_FENCE_RE.search(accumulated):
                    in_code_block = True
            else:
                first_fence = _OPEN_FENCE_RE.search(accumulated)
                if first_fence is not None:
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

        # trim partial tokens after the closing fence.
        if code_block_closed:
            first_fence = _OPEN_FENCE_RE.search(accumulated)
            if first_fence is not None:
                tail = accumulated[first_fence.end():]
                close_idx_in_tail = tail.find("```")
                accumulated = accumulated[: first_fence.end() + close_idx_in_tail + 3]

        return accumulated

    def _record_error(self, text: str) -> None:
        self._append_text("user", text)
        if self.tracer is not None:
            self.tracer.log_error(text)

    def _record_observation(self, blocks: list[Block]) -> None:
        self._append_message("user", blocks)
        if self.tracer is not None:
            # trace what the LLM will actually see for this message.
            visible = [b.text for b in blocks if self._block_is_visible(b) and b.text]
            self.tracer.log_observation("\n\n".join(visible))

    def _try_generate(self) -> str | None:
        try:
            assistant_text = self._generate()
        except Exception as e:
            error_msg = f"[generation error: {e}]"
            self._error = error_msg
            self._record_error(error_msg)
            return None

        self._append_text("assistant", assistant_text)
        if self.tracer is not None:
            self.tracer.log_assistant(assistant_text)
        return assistant_text

    def _extract_code(self, assistant_text: str, step: int) -> tuple[str, bool] | None:
        """Parse the first python code block from *assistant_text*.

        Returns (code, multiple_blocks_flag) on success, or None after
        recording an observation describing the failure.
        """
        all_blocks = _extract_all_code_blocks(assistant_text)
        multiple_blocks = len(all_blocks) > 1
        if all_blocks:
            code = all_blocks[0]
        else:
            try:
                code = parse_code_blobs(assistant_text, CODE_BLOCK_TAGS)
            except ValueError as e:
                self._record_error(
                    f"Observation (step {step + 1}): could not parse a "
                    f"python code block from your response.\n{e}"
                )
                return None

        if not code.strip():
            self._record_error(
                f"Observation (step {step + 1}): your response contained an "
                f"empty code block. Please output a non-empty "
                f"```python ... ``` block with a single tool call."
            )
            return None

        return code, multiple_blocks

    def _reject_if_hard_cutoff(
        self, code: str, step: int, pregen_tokens: int
    ) -> bool:
        """If we're over the hard context cutoff, reject non-prune/-final tool calls.

        Returns True if the step was rejected (caller should `continue`).
        """
        if pregen_tokens < self.hard_context_tokens:
            return False
        called = self._detect_tool_call(code)
        if called in _RESTRICTED_HARD_TOOLS:
            return False
        self._record_error(
            f"Observation (step {step + 1}): context window usage "
            f"(~{pregen_tokens:,} tokens) has exceeded the hard cutoff "
            f"({int(self.hard_context_fraction * 100)}% of "
            f"{self.model_context_window:,}). Until you reduce usage, "
            f"only `prune(...)` and `final_answer(...)` calls are "
            f"accepted. Your `{called or '<unknown>'}(...)` call was not "
            f"executed."
        )
        self._num_steps = step + 1
        return True

    def _execute_code(
        self, executor: LocalPythonExecutor, code: str, step: int
    ) -> CodeOutput | None:
        try:
            return executor(code)
        except InterpreterError as e:
            self._record_error(
                f"Observation (step {step + 1}): execution failed.\n{e}"
            )
            return None
        except Exception as e:
            self._record_error(
                f"Observation (step {step + 1}): tool raised "
                f"{type(e).__name__}: {e}"
            )
            return None

    def _apply_prune_result(self, payload: dict) -> tuple[int, int]:
        """Fold a `prune(...)` payload into the agent-owned prune sets.

        Returns (new_chunk_count, new_doc_count) actually added.
        """
        new_chunks = [
            c for c in payload.get("chunk_ids", [])
            if c not in self._pruned_chunk_ids
        ]
        new_docs = [
            d for d in payload.get("doc_ids", [])
            if d not in self._pruned_doc_ids
        ]
        self._pruned_chunk_ids.update(new_chunks)
        self._pruned_doc_ids.update(new_docs)
        return len(new_chunks), len(new_docs)

    def _blocks_from_output(self, out: CodeOutput, step: int) -> list[Block]:
        """Build the observation blocks for an executor result.

        Tool outputs that carry structured chunk lists (search_corpus /
        grep_corpus sentinels) are split into one TextBlock per header and
        one ChunkBlock per chunk so they can be redacted at render time.
        The `prune(...)` sentinel triggers a side-effect on the agent's
        prune sets and emits a single summary TextBlock.
        """
        blocks: list[Block] = [TextBlock(f"Observation (step {step + 1}):")]
        if out.logs:
            blocks.append(TextBlock(f"[stdout]\n{out.logs}"))

        output = out.output
        if output is None:
            if len(blocks) == 1:
                blocks.append(TextBlock("[no output]"))
            return blocks

        if isinstance(output, dict) and output.get(PRUNE_RESULT_TAG) is True:
            n_chunks, n_docs = self._apply_prune_result(output)
            blocks.append(
                TextBlock(
                    f"[result]\nPruned {n_chunks} chunk(s) and {n_docs} doc(s)."
                )
            )
            return blocks

        if isinstance(output, dict) and output.get(SEARCH_RESULT_TAG) is True:
            for chunk in output.get("chunks", []):
                blocks.append(
                    ChunkBlock(
                        chunk_id=chunk["chunk_id"],
                        doc_id=chunk["doc_id"],
                        text=chunk["text"],
                    )
                )
            return blocks

        if isinstance(output, dict) and output.get(GREP_RESULT_TAG) is True:
            for group in output.get("groups", []):
                blocks.append(TextBlock(group["header"]))
                for chunk in group["chunks"]:
                    blocks.append(
                        ChunkBlock(
                            chunk_id=chunk["chunk_id"],
                            doc_id=chunk["doc_id"],
                            text=chunk["text"],
                        )
                    )
            return blocks

        if isinstance(output, dict) and output.get(READ_DOCUMENT_RESULT_TAG) is True:
            for doc in output.get("docs", []):
                blocks.append(
                    ChunkBlock(
                        chunk_id=None, doc_id=doc["doc_id"], text=doc["text"]
                    )
                )
            return blocks

        blocks.append(TextBlock(f"[result]\n{output}"))
        return blocks

    def _post_step_advisories(self, step: int) -> str:
        """Build the end-of-turn advisory text (context usage, warnings)."""
        post_tokens = self._estimate_tokens()
        pct = 100.0 * post_tokens / self.model_context_window

        advisories: list[str] = [
            f"[context usage: ~{post_tokens:,} / {self.model_context_window:,} "
            f"tokens ({pct:.1f}%)]"
        ]
        if post_tokens >= self.hard_context_tokens:
            advisories.append(
                f"You have exceeded the hard context cutoff "
                f"({int(self.hard_context_fraction * 100)}% of "
                f"{self.model_context_window:,}). Until you reduce usage, only "
                f"`prune(...)` and `final_answer(...)` calls will be accepted."
            )
        elif post_tokens >= self.effective_context_tokens and self.train:
            advisories.append(
                f"You have exceeded the soft context cutoff "
                f"({int(self.effective_context_fraction * 100)}% of "
                f"{self.model_context_window:,}). Strongly consider calling "
                f"`prune(...)` on chunks or docs you no longer need so that "
                f"future searches stay focused."
            )

        steps_remaining = self.max_steps - (step + 1)
        if steps_remaining == 1:
            advisories.append(
                "Your next step is your FINAL step. You MUST call "
                "`final_answer(...)` on the next step."
            )
        elif 1 < steps_remaining <= MAX_STEPS_WARNING_STEPS_BEFORE:
            advisories.append(
                f"You have {steps_remaining} step(s) remaining before you must "
                f"return a `final_answer(...)`."
            )
        return "\n\n".join(advisories)

    # ------------------------------------------------------------------
    # Step orchestration
    # ------------------------------------------------------------------

    def _run_step(
        self, executor: LocalPythonExecutor, step: int
    ) -> _StepOutcome:
        # Hard cutoff check uses the pre-generation token count of the
        # message render that will actually be sent.
        pregen_tokens = self._estimate_tokens()

        assistant_text = self._try_generate()
        if assistant_text is None:
            return _StepOutcome(should_break=True)

        extracted = self._extract_code(assistant_text, step)
        if extracted is None:
            return _StepOutcome()
        code, multiple_blocks = extracted

        if self._reject_if_hard_cutoff(code, step, pregen_tokens):
            return _StepOutcome()

        out = self._execute_code(executor, code, step)
        if out is None:
            return _StepOutcome()

        blocks = self._blocks_from_output(out, step)
        self._record_observation(blocks)
        self._num_steps = step + 1

        if multiple_blocks:
            self._append_text("user", MULTIPLE_BLOCKS_REMINDER)
            if self.tracer is not None:
                self.tracer.log_observation(MULTIPLE_BLOCKS_REMINDER)

        if out.is_final_answer:
            return _StepOutcome(final_page_keys=_coerce_page_keys(out.output))

        advisory_msg = self._post_step_advisories(step)
        self._append_text("user", advisory_msg)
        if self.tracer is not None:
            self.tracer.log_observation(advisory_msg)

        return _StepOutcome()

    def _init_question_state(self, question: str) -> None:
        self._completed = False
        self._num_steps = 0
        self._error: str | None = None
        self.messages = [
            {"role": "system", "blocks": [TextBlock(self.system_prompt)]},
            {"role": "user", "blocks": [TextBlock(f"Question: {question}")]},
        ]
        self._pruned_chunk_ids.clear()
        self._pruned_doc_ids.clear()
        if self.tracer is not None:
            self.tracer.log_system(self.system_prompt)
            self.tracer.log_question(f"Question: {question}")

    def retrieve(self, question: str) -> list[str]:
        """Run the agent and return the page keys it identifies as relevant.

        Page keys have the format "year_month_page_id".  Per-step state is
        reset; the system prompt is preserved.
        """
        self._init_question_state(question)
        executor = self._build_executor()

        for step in range(self.max_steps):
            outcome = self._run_step(executor, step)
            if outcome.final_page_keys is not None:
                self._completed = True
                return outcome.final_page_keys
            if outcome.should_break:
                break

        if self._error is None:
            self._error = "max steps"
        return []
