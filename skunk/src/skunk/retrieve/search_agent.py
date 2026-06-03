from __future__ import annotations

import os
import pathlib
import random
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import yaml
from chromadb.api.models.Collection import Collection
from google import genai
from google.genai import types as genai_types  # noqa: F401
from jinja2 import Template
from openrouter import OpenRouter

from skunk.logging.tracer import Tracer
from skunk.retrieve.base import Retriever
from skunk.retrieve.search_tools import (
    EMPTY_RESULT_MESSAGE,
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
# The chars/token heuristic above is cheap but under-counts dense numeric
# corpora (e.g. treasury tables tokenize at ~2.3 chars/token, ~1.7x the
# heuristic's estimate). Once the cheap estimate crosses this fraction of the
# window, switch to an exact tokenizer count (when the backend provides one) so
# the prune / response-budget decisions are accurate near the limit. Set well
# below HARD_CONTEXT_FRACTION so the exact recount kicks in before the real
# token count reaches the hard cutoff even under heavy under-counting.
EXACT_RECOUNT_FRACTION = 0.4
MAX_STEPS = 20
MAX_STEPS_WARNING_STEPS_BEFORE = 3
MAX_PAGES_PER_TOOL_CALL = 20
CODE_BLOCK_TAGS = ("```python", "```")

# Maximum number of tool calls the agent may invoke in parallel within a
# single step. The agent is prompted to emit between 1 and this many fenced
# python code blocks per turn; we execute them concurrently and surface
# their observations in submission order.
MAX_PARALLEL_TOOL_CALLS = 3

# OpenRouter service-tier hints. ``flex`` is the cheap, best-effort tier that
# gets throttled / aborted under load (429 / 502 / 504 / finish_reason="error");
# ``STANDARD_SERVICE_TIER`` (no hint) routes through the provider's normal
# capacity. On the first flex failure within a run we permanently downgrade the
# agent to the standard tier so the remaining requests don't keep hitting the
# same flaky capacity pool.
FLEX_SERVICE_TIER = "flex"
STANDARD_SERVICE_TIER: str | None = None

# Generation retry/backoff. A single transient provider error used to abort the
# whole agent run (and, in datagen, throw away the entire seed); we now retry
# the streaming generation call with exponential backoff + jitter before giving
# up. Delay for attempt ``i`` (0-indexed) is
# ``min(BASE * 2**i, MAX) + uniform(0, JITTER)`` seconds.
DEFAULT_MAX_GENERATE_RETRIES = 4
GENERATE_BACKOFF_BASE_SEC = 1.0
GENERATE_BACKOFF_MAX_SEC = 30.0
GENERATE_BACKOFF_JITTER_SEC = 1.0

# names of tools available to the agent. Used to detect which tool a code
# block invokes for the purposes of the hard-context-cutoff restriction
# and the control-tool-isolation rule (see _CONTROL_ONLY_TOOLS below).
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
# Tools that may only ever appear by themselves within a single step. Mixing
# any of them with another tool call in the same parallel batch is rejected.
_CONTROL_ONLY_TOOLS = {"prune", "final_answer"}

# Tools that the agent is restricted to using when it reaches the hard context limit.
_RESTRICTED_HARD_TOOLS = {"prune", "final_answer"}

_PROMPTS_FILE = pathlib.Path(__file__).parent / "prompts.yaml"
with _PROMPTS_FILE.open() as _f:
    _PROMPTS = yaml.safe_load(_f)

SEARCH_AGENT_SYSTEM_PROMPT: str = _PROMPTS["search_agent_system_prompt"]
OFFICEQA_SPECIAL_NOTES: str = _PROMPTS["officeqa_special_notes"]
BROWSECOMP_PLUS_SPECIAL_NOTES: str = _PROMPTS["browsecomp_plus_special_notes"]

# Default sampling parameters for rollout generation, matching OpenRouter's
# documented defaults. Recording these explicitly in each RolloutRecord makes
# the importance-ratio denominator (pi_old) fully reproducible at train time.
DEFAULT_ROLLOUT_SAMPLING_PARAMS: dict[str, float | int] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}

# Number of top-token log-probability candidates to request from the provider
# alongside each generated token. Used only in rollout mode for entropy
# estimation and off-policy diagnostics. Does NOT affect the sampling
# distribution (i.e. does not change pi_old) so it is kept separate from
# DEFAULT_ROLLOUT_SAMPLING_PARAMS.
DEFAULT_ROLLOUT_TOP_LOGPROBS: int = 5

# Subset of DEFAULT_ROLLOUT_SAMPLING_PARAMS that the OpenRouter Chat.send()
# method accepts as direct keyword arguments. Provider-specific extensions
# (top_k, min_p, repetition_penalty) are recorded in RolloutRecord.sampling_params
# for documentation but cannot be forwarded to the SDK — they are all at their
# "off" defaults anyway so omitting them has no effect on pi_old.
_CHAT_SEND_SAMPLING_PARAMS: frozenset[str] = frozenset(
    {"temperature", "top_p", "frequency_penalty", "presence_penalty"}
)

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
    "Reminder: your previous response contained more than "
    f"{MAX_PARALLEL_TOOL_CALLS} code blocks. Only the first "
    f"{MAX_PARALLEL_TOOL_CALLS} were executed; the rest were dropped. "
    f"Please keep each step to at most {MAX_PARALLEL_TOOL_CALLS} "
    "```python ... ``` blocks."
)

CONTROL_TOOL_MIX_REMINDER = (
    "`prune(...)` and `final_answer(...)` must each be invoked in a step "
    "by themselves, not combined with other tool calls. Your step was not "
    "executed; please retry with one tool call per step (or up to "
    f"{MAX_PARALLEL_TOOL_CALLS} non-control tool calls in parallel)."
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
    raw_output: object = None


@dataclass
class _CallResult:
    """Result of executing one of the (1..MAX_PARALLEL_TOOL_CALLS) code
    blocks emitted by the model in a single step. Exactly one of ``output``
    / ``error`` is populated."""

    output: CodeOutput | None
    error: str | None


# ---------------------------------------------------------------------------
# Generation backends
#
# The agent loop, tool execution, pruning, and logprob persistence are all
# backend-agnostic; the only LLM seam is producing one assistant turn from the
# redacted message render. A `GenerationBackend` encapsulates that call so the
# agent can dispatch to OpenRouter (datagen synthesis, quality filter) or to
# Tinker (rollouts -- so the captured logprobs are exactly pi_old for the model
# we fine-tune) without the loop knowing which.
# ---------------------------------------------------------------------------


class GenerationBackend(Protocol):
    def generate(
        self,
        rendered_messages: list[dict],
        *,
        sampling_params: dict | None,
        capture_logprobs: bool,
    ) -> tuple[str, dict | None]:
        """Return ``(assistant_text, logprob_data)`` for one assistant turn."""
        ...

    def on_failure(self) -> bool:
        """Hook called after a failed generation before retry.

        Returns True if the backend mutated its routing/state in response (used
        only to annotate the retry log). Backends with nothing to adjust return
        False.
        """
        ...

    def count_tokens(self, text: str) -> int | None:
        """Exact token count for ``text``, or None if no tokenizer is available.

        The agent uses this to replace its chars/token heuristic with an exact
        count near the context limit. Backends without a local tokenizer (e.g.
        OpenRouter) return None and the agent keeps the heuristic.
        """
        ...


class OpenRouterBackend:
    """Streaming OpenRouter generation backend (the original `_generate`).

    Owns the OpenRouter `service_tier` / provider-pin state, including the
    first-failure flex->standard downgrade (`on_failure`). Streams tokens and
    early-stops as soon as one complete ```python ... ``` block is received.
    """

    def __init__(
        self,
        client: OpenRouter | genai.Client,
        model_id: str,
        service_tier: str | None,
        rollout_provider: str | None = None,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.service_tier = service_tier
        self.rollout_provider = rollout_provider

    def on_failure(self) -> bool:
        # On the first failure at the flex tier, permanently downgrade to the
        # standard tier (flex capacity is the usual culprit for transient
        # 429/502/504/finish_reason="error"); don't keep retrying that pool.
        if self.service_tier == FLEX_SERVICE_TIER:
            self.service_tier = STANDARD_SERVICE_TIER
            return True
        return False

    def count_tokens(self, text: str) -> int | None:
        # No local tokenizer over OpenRouter; the agent keeps its heuristic.
        return None

    def generate(
        self,
        rendered_messages: list[dict],
        *,
        sampling_params: dict | None,
        capture_logprobs: bool,
    ) -> tuple[str, dict | None]:
        final_messages = rendered_messages

        # Build extra kwargs for rollout mode: explicit sampling params,
        # logprob request, and optional provider pin for reproducibility.
        extra: dict = {}
        if capture_logprobs:
            extra.update(
                {
                    k: v
                    for k, v in (sampling_params or {}).items()
                    if k in _CHAT_SEND_SAMPLING_PARAMS
                }
            )
            extra["logprobs"] = True
            extra["top_logprobs"] = DEFAULT_ROLLOUT_TOP_LOGPROBS
        if self.rollout_provider is not None:
            extra["provider"] = {"only": [self.rollout_provider]}
        if self.service_tier is not None:
            extra["service_tier"] = self.service_tier

        # Stream tokens and stop as soon as a complete ```python...``` block
        # has been received. Avoids waiting for the model to finish its
        # full "thinking" output after the code block is already parseable.
        stream = self.client.chat.send(  # type: ignore
            model=self.model_id,
            messages=final_messages,  # type: ignore
            stream=True,
            **extra,
        )  # type: ignore

        accumulated = ""
        accumulated_lp: list = []  # raw TokenLogprob objects from the API
        finish_reason: str | None = None
        last_gen_id: str | None = None  # OpenRouter generation ID for API lookup
        system_fingerprint: str | None = None
        code_block_closed = False
        in_code_block = False
        for chunk in stream:
            last_gen_id = getattr(chunk, "id", None)
            # Surface any chunk-level error from the upstream provider before
            # touching choices. OpenRouter sets chunk.error when a non-2xx
            # response was forwarded (e.g. provider capacity errors, content
            # policy rejections).
            chunk_error = getattr(chunk, "error", None)
            if chunk_error is not None:
                err_code = getattr(chunk_error, "code", "unknown")
                err_msg = getattr(chunk_error, "message", str(chunk_error))
                raise RuntimeError(
                    f"upstream provider error (code={err_code}): {err_msg} "
                    f"[gen_id={last_gen_id}]"
                )

            # Some providers (e.g. parasail via OpenRouter) emit a terminal
            # chunk with choices=[] carrying only usage metadata. Skip it.
            if not chunk.choices:  # type: ignore[union-attr]
                if getattr(chunk, "system_fingerprint", None):
                    system_fingerprint = chunk.system_fingerprint  # type: ignore[union-attr]
                continue

            choice = chunk.choices[0]  # type: ignore[union-attr]

            # Track finish_reason on every chunk (not just in logprob mode)
            # so we can detect finish_reason="error" from the provider.
            if choice.finish_reason:
                finish_reason = choice.finish_reason
                if finish_reason == "error":
                    # Query https://openrouter.ai/api/v1/generation?id=<gen_id>
                    # for native_finish_reason and upstream error details.
                    raise RuntimeError(
                        f"upstream provider signalled finish_reason='error' "
                        f"for model {self.model_id!r} [gen_id={last_gen_id}]"
                    )

            # Some reasoning models (e.g. openai/gpt-oss-20b via OpenRouter) stream
            # their entire output through delta.reasoning while delta.content stays "".
            # Fall back to reasoning tokens when content is absent.
            _cd = choice.delta  # type: ignore[union-attr]
            delta = _cd.content or getattr(_cd, "reasoning", None) or ""  # type: ignore
            accumulated += delta

            if capture_logprobs:
                lp_content = getattr(getattr(choice, "logprobs", None), "content", None)
                if lp_content:
                    accumulated_lp.extend(lp_content)
                if getattr(chunk, "system_fingerprint", None):
                    system_fingerprint = chunk.system_fingerprint  # type: ignore[union-attr]

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
            raise RuntimeError(
                f"model {self.model_id!r} returned an empty response "
                f"(finish_reason={finish_reason!r}, gen_id={last_gen_id!r})"
            )

        # trim partial tokens after the closing fence.
        if code_block_closed:
            finish_reason = "code_block_closed"
            first_fence = _OPEN_FENCE_RE.search(accumulated)
            if first_fence is not None:
                tail = accumulated[first_fence.end():]
                close_idx_in_tail = tail.find("```")
                accumulated = accumulated[: first_fence.end() + close_idx_in_tail + 3]

        logprob_data: dict | None = None
        if capture_logprobs and accumulated_lp:
            # Align accumulated_lp to the (possibly truncated) text by
            # walking forward until the concatenated token strings consume
            # the full text length.  This is necessary because early-stop
            # may cut off mid-stream before the server sends a finish_reason.
            cumlen = 0
            cutoff = 0
            for i, tok in enumerate(accumulated_lp):
                cumlen += len(tok.token)  # type: ignore[union-attr]
                cutoff = i + 1
                if cumlen >= len(accumulated):
                    break
            accumulated_lp = accumulated_lp[:cutoff]

            logprob_data = {
                "tokens": [t.token for t in accumulated_lp],  # type: ignore[union-attr]
                "token_logprobs": [t.logprob for t in accumulated_lp],  # type: ignore[union-attr]
                "top_logprobs": [
                    [{"token": c.token, "logprob": c.logprob} for c in (t.top_logprobs or [])]  # type: ignore[union-attr]
                    for t in accumulated_lp
                ],
                "finish_reason": finish_reason,
                "system_fingerprint": system_fingerprint,
            }

        return accumulated, logprob_data


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
        document_map: dict[str, str],
        chroma_collection: Collection,
        emb_model_id: str,
        tracer: Tracer | None = None,
        max_steps: int = MAX_STEPS,
        max_pages_per_tool_call: int = MAX_PAGES_PER_TOOL_CALL,
        model_context_window: int = MODEL_CONTEXT_WINDOW,
        effective_context_fraction: float = EFFECTIVE_CONTEXT_FRACTION,
        hard_context_fraction: float = HARD_CONTEXT_FRACTION,
        train: bool = False,
        special_notes: str = "",
        system_prompt_override: str | None = None,
        final_answer_fn: Callable | None = None,
        additional_authorized_imports: list[str] | None = None,
        sampling_params: dict | None = None,
        service_tier: str | None = "flex",
        max_generate_retries: int = DEFAULT_MAX_GENERATE_RETRIES,
        tinker_backend: GenerationBackend | None = None,
    ):
        self.model_id = model_id
        # OpenRouter client is retained on every path -- query embeddings go
        # through it (see make_search_tools), independent of which backend
        # generates assistant turns.
        self.client: OpenRouter | genai.Client = OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])
        # self.client: OpenRouter | genai.Client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.chroma_collection = chroma_collection
        self.document_map = document_map
        self.emb_model_id = emb_model_id
        self.tracer = tracer
        self.max_steps = max_steps
        self.max_pages_per_tool_call = max_pages_per_tool_call
        self.model_context_window = model_context_window
        self.effective_context_fraction = effective_context_fraction
        self.hard_context_fraction = hard_context_fraction
        self.effective_context_tokens = int(model_context_window * effective_context_fraction)
        self.hard_context_tokens = int(model_context_window * hard_context_fraction)
        # Cheap-estimate threshold past which we pay for an exact tokenizer count
        # (only when the backend provides one). Keeps the common case cheap while
        # making the prune / budget decisions accurate in the upper region.
        self.exact_recount_tokens = int(model_context_window * EXACT_RECOUNT_FRACTION)
        self.train = train
        self.final_answer_fn = final_answer_fn
        self.additional_authorized_imports = list(additional_authorized_imports or [])
        self.sampling_params: dict | None = sampling_params
        self.max_generate_retries = max_generate_retries
        # Generation backend: Tinker for rollouts (so captured logprobs are
        # exactly pi_old for the fine-tuned model), OpenRouter otherwise.
        self._backend: GenerationBackend = tinker_backend or OpenRouterBackend(
            client=self.client,
            model_id=model_id,
            service_tier=service_tier,
        )
        if system_prompt_override is not None:
            self.system_prompt = system_prompt_override
        else:
            self.system_prompt = Template(SEARCH_AGENT_SYSTEM_PROMPT).render(
                max_steps=max_steps,
                max_pages=max_pages_per_tool_call,
                max_parallel_tool_calls=MAX_PARALLEL_TOOL_CALLS,
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
        """Token usage of the LLM-facing (post-prune) render.

        Cheap chars/token heuristic in the common case; once that estimate
        enters the upper region of the window (``exact_recount_tokens``), use the
        backend's exact tokenizer count if available. This catches the heuristic
        badly under-counting dense numeric corpora before the real token count
        slams into the hard cutoff.
        """
        if rendered is None:
            rendered = self._render_for_llm()
        cheap = sum(len(m["content"]) for m in rendered) // CHARS_PER_TOKEN_ESTIMATE
        if cheap >= self.exact_recount_tokens:
            exact = self._backend.count_tokens("\n\n".join(m["content"] for m in rendered))
            if exact is not None:
                return exact
        return cheap

    def messages_to_jsonable(self) -> list[dict]:
        """Return a JSON-serializable copy of the full message trajectory.

        Each block becomes a plain dict tagged with its type. Use this for
        persistence and downstream reward computation -- the raw
        `self.messages` contains dataclass instances that `json.dump`
        cannot serialize.

        For assistant messages produced in rollout mode (i.e. when
        ``sampling_params`` was set), an extra ``"logprobs"`` key is included:

        .. code-block:: json

            {
              "role": "assistant",
              "blocks": [...],
              "logprobs": {
                "tokens": ["...", ...],
                "token_logprobs": [-0.12, ...],
                "top_logprobs": [[{"token": "...", "logprob": -0.12}, ...], ...],
                "finish_reason": "code_block_closed",
                "system_fingerprint": "fp_..."
              }
            }

        ``token_logprobs[i]`` is ``log pi_old(token_i | context)`` -- the
        behaviour-policy log-probability needed for the importance-ratio
        denominator in GRPO / CISPO.  The loss mask (which tokens are
        model-generated vs. tool-observation) can be reconstructed from the
        ``role`` field at train time: supervise ``role == "assistant"`` spans
        only.
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
            entry: dict = {"role": msg["role"], "blocks": blocks_json}
            if "logprobs" in msg:
                entry["logprobs"] = msg["logprobs"]
            out.append(entry)
        return out

    @staticmethod
    def _detect_tool_call(code: str) -> str | None:
        """Return the name of the first tool invoked in *code*, if any."""
        m = _TOOL_CALL_RE.search(code)
        return m.group(1) if m else None

    def _build_executor(self) -> LocalPythonExecutor:
        executor = LocalPythonExecutor(
            additional_authorized_imports=self.additional_authorized_imports,
        )
        tools = make_search_tools(
            chroma_collection=self.chroma_collection,
            emb_model_id=self.emb_model_id,
            openrouter_client=self.client,
            document_map=self.document_map,
            pruned_chunk_ids=self._pruned_chunk_ids,
            pruned_doc_ids=self._pruned_doc_ids,
            final_answer_fn=self.final_answer_fn,
        )
        executor.send_tools(tools)
        return executor

    # ------------------------------------------------------------------
    # Per-step helpers
    # ------------------------------------------------------------------

    def _generate(self) -> tuple[str, dict | None]:
        """Generate one assistant turn from the redacted message render.

        Dispatches to the configured generation backend (OpenRouter for
        datagen / quality-filter roles, Tinker for rollouts). No message
        truncation is performed here: staying under the context window is the
        model's responsibility, exercised via `prune(...)`.

        Returns ``(text, logprob_data)``. ``logprob_data`` is non-None only
        when ``self.sampling_params`` is set (rollout mode); its
        ``token_logprobs`` are ``log pi_old(a_t)`` -- the importance-ratio
        denominator for GRPO / CISPO at train time.
        """
        return self._backend.generate(
            self._render_for_llm(),
            sampling_params=self.sampling_params,
            capture_logprobs=self.sampling_params is not None,
        )

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

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter for generation retry ``attempt``."""
        base = min(GENERATE_BACKOFF_BASE_SEC * (2 ** attempt), GENERATE_BACKOFF_MAX_SEC)
        return base + random.uniform(0.0, GENERATE_BACKOFF_JITTER_SEC)

    @staticmethod
    def _is_retryable_generation_error(e: Exception) -> bool:
        """Whether a failed generation is worth retrying.

        Client errors (HTTP 4xx: bad request, unprocessable, not found) are
        deterministic -- e.g. a prompt that exceeds the context window will fail
        identically on every retry -- so we don't waste backoff attempts on them.
        Everything else (transient 5xx, timeouts, stream errors) is retryable.
        """
        status = getattr(e, "status_code", None)
        if status is None:
            status = getattr(e, "code", None)
        # 4xx are deterministic client errors (except 429 rate-limit, retryable).
        return not (isinstance(status, int) and 400 <= status < 500 and status != 429)

    def _try_generate(self) -> str | None:
        # Retry the generation call with exponential backoff. The backend's
        # `on_failure()` hook handles any backend-specific recovery (the
        # OpenRouter backend downgrades flex->standard on the first failure,
        # since flex capacity is the usual culprit for transient
        # 429/502/504/finish_reason="error"; the Tinker backend is a no-op).
        assistant_text: str | None = None
        logprob_data: dict | None = None
        last_err: Exception | None = None
        for attempt in range(self.max_generate_retries + 1):
            try:
                assistant_text, logprob_data = self._generate()
                last_err = None
                break
            except Exception as e:
                last_err = e
                if not self._is_retryable_generation_error(e):
                    # Deterministic client error (e.g. prompt over the context
                    # window) -- retrying would fail identically. Stop now.
                    if self.tracer is not None:
                        self.tracer.log_error(f"[generation failed (non-retryable): {e}]")
                    break
                downgraded = self._backend.on_failure()
                if attempt >= self.max_generate_retries:
                    break
                delay = self._backoff_delay(attempt)
                if self.tracer is not None:
                    tier_note = " [downgraded flex->standard tier]" if downgraded else ""
                    self.tracer.log_error(
                        f"[generation retry {attempt + 1}/{self.max_generate_retries} "
                        f"in {delay:.1f}s{tier_note}: {e}]"
                    )
                time.sleep(delay)

        if last_err is not None:
            error_msg = f"[generation error: {last_err}]"
            self._error = error_msg
            self._record_error(error_msg)
            return None

        # Store the message with logprob data attached (if captured).  The
        # logprob dict is included in messages_to_jsonable so it gets persisted
        # alongside the full trajectory for RL training use.
        entry: dict = {"role": "assistant", "blocks": [TextBlock(assistant_text)]} # type: ignore
        if logprob_data is not None:
            entry["logprobs"] = logprob_data
        self.messages.append(entry)
        if self.tracer is not None:
            self.tracer.log_assistant(assistant_text) # type: ignore
        return assistant_text

    def _extract_code_blocks(
        self, assistant_text: str, step: int
    ) -> list[str] | None:
        """Parse all fenced python code blocks from *assistant_text*.

        Returns a list of non-empty code strings (capped at
        ``MAX_PARALLEL_TOOL_CALLS``), or ``None`` after recording an
        observation describing a parsing / emptiness failure. The caller
        is responsible for noting + warning when the original list was
        longer than the cap.
        """
        all_blocks = _extract_all_code_blocks(assistant_text)
        if not all_blocks:
            try:
                all_blocks = [parse_code_blobs(assistant_text, CODE_BLOCK_TAGS)]
            except ValueError as e:
                self._record_error(
                    f"Observation (step {step + 1}): could not parse a "
                    f"python code block from your response.\n{e}"
                )
                return None

        all_blocks = [b for b in all_blocks if b.strip()]
        if not all_blocks:
            self._record_error(
                f"Observation (step {step + 1}): your response contained an "
                f"empty code block. Please output a non-empty "
                f"```python ... ``` block."
            )
            return None
        return all_blocks

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

    def _blocks_from_output(self, out: CodeOutput) -> list[Block]:
        """Build the observation blocks for one executor result.

        Tool outputs that carry structured chunk lists (search_corpus /
        grep_corpus sentinels) are split into one TextBlock per header and
        one ChunkBlock per chunk so they can be redacted at render time.
        The `prune(...)` sentinel triggers a side-effect on the agent's
        prune sets and emits a single summary TextBlock. Empty search /
        grep results emit a generic ``EMPTY_RESULT_MESSAGE`` so the agent
        knows to widen its scope (or unprune chunks).
        """
        blocks: list[Block] = []
        if out.logs:
            blocks.append(TextBlock(f"[stdout]\n{out.logs}"))

        output = out.output
        if output is None:
            if not blocks:
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
            chunks = output.get("chunks", [])
            if not chunks:
                blocks.append(TextBlock(EMPTY_RESULT_MESSAGE))
                return blocks
            for chunk in chunks:
                blocks.append(
                    ChunkBlock(
                        chunk_id=chunk["chunk_id"],
                        doc_id=chunk["doc_id"],
                        text=chunk["text"],
                    )
                )
            return blocks

        if isinstance(output, dict) and output.get(GREP_RESULT_TAG) is True:
            groups = output.get("groups", [])
            if not groups:
                blocks.append(TextBlock(EMPTY_RESULT_MESSAGE))
                return blocks
            for group in groups:
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

    def _execute_blocks(self, code_blocks: list[str]) -> list[_CallResult]:
        """Execute one or more code blocks in parallel.

        Each block runs against its own fresh ``LocalPythonExecutor`` so
        parallel calls cannot race on interpreter state. The shared
        ``pruned_chunk_ids`` / ``pruned_doc_ids`` sets are read-only here
        (no parallel batch ever includes a ``prune(...)`` call -- the
        control-tool-isolation rule rejects such mixes before we get here).
        """
        def run_one(code: str) -> _CallResult:
            executor = self._build_executor()
            try:
                return _CallResult(output=executor(code), error=None)
            except InterpreterError as e:
                return _CallResult(output=None, error=f"execution failed.\n{e}")
            except Exception as e:
                return _CallResult(
                    output=None, error=f"tool raised {type(e).__name__}: {e}"
                )

        results: list[_CallResult | None] = [None] * len(code_blocks)
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_TOOL_CALLS) as pool:
            fut_to_idx = {
                pool.submit(run_one, code): i
                for i, code in enumerate(code_blocks)
            }
            for fut, i in fut_to_idx.items():
                results[i] = fut.result()
        return [r for r in results if r is not None]

    def _build_call_blocks(
        self,
        idx: int,
        n_calls: int,
        tool_name: str | None,
        result: _CallResult,
    ) -> list[Block]:
        """Render one parallel-call's observation blocks (with header)."""
        blocks: list[Block] = []
        if n_calls > 1:
            blocks.append(
                TextBlock(
                    f"--- Tool call {idx + 1}/{n_calls} "
                    f"({tool_name or '<unknown>'}(...)) ---"
                )
            )
        if result.error is not None:
            blocks.append(TextBlock(f"[error]\n{result.error}"))
            return blocks
        assert result.output is not None
        blocks.extend(self._blocks_from_output(result.output))
        return blocks

    def _call_visible_tokens(self, call_blocks: list[Block], exact: bool = False) -> int:
        """Token count for a single parallel-call's visible blocks.

        Cheap chars/token heuristic by default; when ``exact`` (set by the
        response-budget pass once the prompt is near the limit) and the backend
        has a tokenizer, count exactly so large observations are truncated
        accurately rather than slipping under an under-counted budget.
        """
        visible = "".join(b.text for b in call_blocks if self._block_is_visible(b))
        if exact:
            tokens = self._backend.count_tokens(visible)
            if tokens is not None:
                return tokens
        return len(visible) // CHARS_PER_TOKEN_ESTIMATE

    def _enforce_response_budget(
        self,
        per_call_blocks: list[list[Block]],
        tools: list[str | None],
        pregen_tokens: int,
        n_calls: int,
    ) -> list[list[Block]]:
        """Replace the largest tool responses with placeholders until the
        projected post-step context fits under the hard cutoff.

        Iterates: if the projected total exceeds ``hard_context_tokens``,
        pick the largest still-replaceable call and substitute its blocks
        for a single error TextBlock describing why. The header block (for
        parallel calls) is preserved so the agent still knows which call
        was truncated.
        """
        budget = self.hard_context_tokens
        replaced: set[int] = set()
        # Once the prompt is in the upper region, size observations exactly so a
        # single large tool result can't slip past an under-counted budget.
        exact = pregen_tokens >= self.exact_recount_tokens

        def projected_total() -> int:
            return pregen_tokens + sum(
                self._call_visible_tokens(cb, exact=exact) for cb in per_call_blocks
            )

        while projected_total() > budget:
            candidates = [i for i in range(n_calls) if i not in replaced]
            if not candidates:
                break
            i = max(
                candidates,
                key=lambda j: self._call_visible_tokens(per_call_blocks[j], exact=exact),
            )
            attempted_tokens = self._call_visible_tokens(per_call_blocks[i], exact=exact)
            tool_name = tools[i] or "<unknown>"
            placeholder = TextBlock(
                f"Tool call {i + 1}/{n_calls} ({tool_name}(...)) tried "
                f"returning ~{attempted_tokens:,} tokens which would exceed "
                f"the model's context limit; please scope the tool call more "
                f"narrowly or prune previous chunks/documents to make room "
                f"in the context."
            )
            # preserve the parallel-call header if there is one.
            kept: list[Block] = []
            if n_calls > 1 and per_call_blocks[i]:
                head = per_call_blocks[i][0]
                if isinstance(head, TextBlock) and head.text.startswith("--- Tool call "):
                    kept.append(head)
            kept.append(placeholder)
            per_call_blocks[i] = kept
            replaced.add(i)
        return per_call_blocks

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

    def _run_step(self, step: int) -> _StepOutcome:
        # Hard cutoff check uses the pre-generation token count of the
        # message render that will actually be sent.
        pregen_tokens = self._estimate_tokens()

        assistant_text = self._try_generate()
        if assistant_text is None:
            return _StepOutcome(should_break=True)

        code_blocks = self._extract_code_blocks(assistant_text, step)
        if code_blocks is None:
            return _StepOutcome()

        excess = max(0, len(code_blocks) - MAX_PARALLEL_TOOL_CALLS)
        code_blocks = code_blocks[:MAX_PARALLEL_TOOL_CALLS]
        tools = [self._detect_tool_call(c) for c in code_blocks]

        # Control-tool isolation: prune / final_answer must each be the only
        # tool call in a step. Mixing them with anything else is rejected.
        if len(code_blocks) > 1 and any(t in _CONTROL_ONLY_TOOLS for t in tools):
            self._record_error(
                f"Observation (step {step + 1}): {CONTROL_TOOL_MIX_REMINDER}"
            )
            return _StepOutcome()

        # Hard cutoff: every block in the batch must be a control tool when
        # context usage has exceeded the hard threshold.
        if pregen_tokens >= self.hard_context_tokens:
            offending = next(
                (t for t in tools if t not in _RESTRICTED_HARD_TOOLS), None
            )
            if offending is not None or any(t is None for t in tools):
                offending_name = offending or "<unknown>"
                self._record_error(
                    f"Observation (step {step + 1}): context window usage "
                    f"(~{pregen_tokens:,} tokens) has exceeded the hard cutoff "
                    f"({int(self.hard_context_fraction * 100)}% of "
                    f"{self.model_context_window:,}). Until you reduce usage, "
                    f"only `prune(...)` and `final_answer(...)` calls are "
                    f"accepted. Your `{offending_name}(...)` call was not "
                    f"executed."
                )
                self._num_steps = step + 1
                return _StepOutcome()

        # Execute (always through the thread pool, even when N==1).
        call_results = self._execute_blocks(code_blocks)

        # Build per-call observation blocks, enforce response-size budget.
        n_calls = len(code_blocks)
        per_call_blocks = [
            self._build_call_blocks(i, n_calls, tools[i], res)
            for i, res in enumerate(call_results)
        ]
        per_call_blocks = self._enforce_response_budget(
            per_call_blocks, tools, pregen_tokens, n_calls
        )

        # Flatten into a single observation message.
        observation_blocks: list[Block] = [
            TextBlock(f"Observation (step {step + 1}):")
        ]
        for call_blocks in per_call_blocks:
            observation_blocks.extend(call_blocks)
        self._record_observation(observation_blocks)
        self._num_steps = step + 1

        if excess > 0:
            self._append_text("user", MULTIPLE_BLOCKS_REMINDER)
            if self.tracer is not None:
                self.tracer.log_observation(MULTIPLE_BLOCKS_REMINDER)

        # final_answer can only appear when N==1 (control-tool isolation),
        # so this loop yields at most one terminal outcome.
        for res in call_results:
            if res.output is not None and res.output.is_final_answer:
                raw = res.output.output
                page_keys = (
                    [] if isinstance(raw, dict) else _coerce_page_keys(raw)
                )
                return _StepOutcome(final_page_keys=page_keys, raw_output=raw)

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

    def _run_loop(self, prompt: str) -> _StepOutcome:
        """Shared step loop used by `retrieve` and `qa_synthesis`.

        Resets per-question state, runs up to `max_steps`, and returns the
        final `_StepOutcome` (which carries `final_page_keys` and `raw_output`
        when the agent called `final_answer`, or default values on timeout).
        """
        self._init_question_state(prompt)

        for step in range(self.max_steps):
            outcome = self._run_step(step)
            if outcome.final_page_keys is not None:
                self._completed = True
                return outcome
            if outcome.should_break:
                break

        if self._error is None:
            self._error = "max steps"
        return _StepOutcome()

    def retrieve(self, question: str) -> list[str]:
        """Run the agent and return the page keys it identifies as relevant.

        Per-question state is reset; the system prompt is preserved.
        """
        return self._run_loop(question).final_page_keys or []

    def qa_synthesis(self, prompt: str) -> list[dict]:
        """Run the agent in QA-synthesis mode and return the generated pairs.

        Expects the agent to have been constructed with `final_answer_fn` set
        to `datagen_final_answer`, which returns a dict with key `qa_pairs`
        whose value is a list of dicts (each with `question`, `answer`, and
        `chunk_ids`).

        Returns the list of pair dicts; an empty list if the agent did not
        complete successfully or returned an unexpected payload.
        """
        outcome = self._run_loop(prompt)
        raw = outcome.raw_output
        if isinstance(raw, dict) and "qa_pairs" in raw:
            return list(raw["qa_pairs"])
        return []
