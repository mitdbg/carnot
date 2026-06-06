from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

from jinja2 import Environment, StrictUndefined

from skunk.common import Effort, ExecutionContext
from skunk.errors import ParseError, StepFailed
from skunk.prompted_call import PromptedCall
from skunk.local_python_executor import CodeOutput, LocalPythonExecutor

_ENV = Environment(autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined)


# ---------------------------------------------------------------------------
# Trajectory blocks
#
# The trajectory is stored as a list of {role, blocks} messages rather than
# flat strings, so the *full* record (every tool call + observation) is kept
# for downstream reward computation while the LLM-facing render can omit
# content the agent has pruned. A `ChunkBlock` carries the chunk_id / doc_id
# needed to redact it after a `prune(...)`; a `TextBlock` is always shown.
# `_render_for_llm()` flattens the visible blocks; `_block_is_visible()` is the
# per-subclass redaction hook (default: show everything).
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


_FENCE_RE = re.compile(r"```([a-zA-Z0-9_]*)\n(.*?)```", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"```[a-zA-Z0-9_]*\n")


def _has_complete_block(acc: str) -> bool:
    """True once `acc` holds a complete fenced block — the streaming `should_stop`."""
    open_m = _OPEN_FENCE_RE.search(acc)
    return bool(open_m and "```" in acc[open_m.end():])


@dataclass
class _StepOutput:
    """A successfully parsed model response, discriminated by `code`: a ```python``` tool-
    call body to execute (`code is not None`), or — when `code is None` — a ```json``` final
    answer in `result` (plain `Any`, so a literal `null` payload is represented as-is).
    `raw` carries the verbatim model text so the caller can append it to message history."""

    code: str | None = None
    result: Any = None
    raw: str = ""


def _parse_step(text: str, _: ExecutionContext) -> _StepOutput:
    """Parse the first fenced block and raise `ParseError` on bad format. Never executes."""
    m = _FENCE_RE.search(text)
    if m is None:
        raise ParseError(raw=text, detail="no fenced block — emit ONE ```python``` block (tool call) "
                         "or ```json``` block (final answer).")
    lang, body = m.group(1).lower(), m.group(2).strip()
    if lang != "json":
        return _StepOutput(code=body, raw=text)
    try:
        return _StepOutput(result=json.loads(body), raw=text)
    except json.JSONDecodeError as e:
        raise ParseError(raw=text, detail=f"final-answer JSON was malformed — {e}") from e


def _trim(messages: list[dict], budget: int) -> list[dict]:
    """Keep first user + the most recent messages fitting in `budget` chars;
    drop the middle behind a placeholder so observation history can't bloat unbounded."""
    if sum(len(m["content"]) for m in messages) <= budget:
        return messages
    head = [messages[0],  # first user question — always kept
            {"role": "user", "content": "...(earlier steps truncated)..."}]
    remaining = budget - sum(len(m["content"]) for m in head)
    tail: list[dict] = []
    for m in reversed(messages[1:]):
        if remaining - len(m["content"]) < 0:
            break
        tail.append(m)
        remaining -= len(m["content"])
    return head + tail[::-1]


class Tool(ABC):
    name: str
    doc: str

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """The tool's runtime behaviour."""


class GenerationBackend(Protocol):
    """Pluggable seam for producing one assistant turn from a redacted message
    render. The default (None) path uses `PromptedCall`; an RL rollout backend
    (e.g. Tinker) implements this to sample tokens and return `log pi_old`."""

    def generate(
        self,
        rendered_messages: list[dict],
        *,
        sampling_params: dict | None,
        capture_logprobs: bool,
    ) -> tuple[str, dict | None]:
        """Return `(assistant_text, logprob_data)`. `rendered_messages` is the
        `{role, content}` view (system message prepended). `logprob_data` is None
        unless `capture_logprobs`, else the schema `messages_to_jsonable` persists."""
        ...

class MultiTurnAgent(ABC):
    """A tool-loop agent that takes a series of tool call actions and produces a final answer."""
    name: str
    briefing: str
    final_answer_doc: str
    default_effort: Effort = "medium"

    _SYSTEM_TEMPLATE = """\
{{ briefing }}

## Tools (already imported)

{{ tools_doc }}

You have ≤{{ max_steps }} steps. On each step, output exactly ONE fenced block:
  - a ```python``` block containing a single tool call — the tool will be executed and its output appears
    as your next observation; or
  - a ```json``` block containing your final answer — emit this once, when you are
    ready to finish. It is parsed as data (not executed), so write plain JSON
    literals (no Python, no variables, no trailing commas).

Requirements for the final answer:
{{ final_answer_doc }}"""

    max_steps: int | None = 8
    # Format-error retries per step, delegated to PromptedCall.call(). A step whose
    # all attempts misfire still advances max_steps; execution errors are not retried.
    max_recover_retries: int = 1
    # TODO: Should probably eventually merge with visible_observations as they tackle the same challenge
    # Hard char cap on the message list, applied after `visible_observations` collapse
    # as the final safety net (see `_llm_step`). Complements, not duplicates, that knob.
    context_budget_chars: int = 200_000
    # If set, only the most recent N tool-result messages are shown; older ones collapse
    # to a placeholder (assistant turns kept). None = show all, rely on `_trim`. A small
    # window keeps loops with large observations lean; accumulator agents leave it None.
    visible_observations: int | None = None
    # Steps remaining at which to emit a low-budget warning. None disables the warning.
    warn_steps_remaining: int | None = 1
    # Extra imports authorized inside the per-step code sandbox. Default: none
    # (tool calls only). Compute-oriented agents (e.g. the task solver) widen
    # this to allow numpy / scipy / statistics / ... in their python steps.
    authorized_imports: list[str] = []

    def __init__(
        self,
        tools: list[Tool],
        *,
        max_steps: int | None = None,
        system_prompt_override: str | None = None,
        generation_backend: GenerationBackend | None = None,
        sampling_params: dict | None = None,
        capture_logprobs: bool = False,
    ) -> None:
        self._tools = tools
        self.max_steps = max_steps
        # Full block trajectory of the most recent `call()`; rebuilt per call.
        # Kept on the instance (one agent per question / branch) so callers can
        # read `messages_to_jsonable()` after the run for reward / persistence.
        self.messages: list[dict] = []
        # Optional pluggable generation backend (e.g. Tinker for RL rollouts).
        # When None, generation goes through `PromptedCall` (the genai/Vertex
        # path). When set, `_llm_step` samples from it and captures per-token
        # logprobs onto each assistant turn for `messages_to_jsonable()`.
        self._backend = generation_backend
        self._sampling_params = sampling_params
        self._capture_logprobs = capture_logprobs
        self._last_logprobs: dict | None = None
        tools_doc = "\n\n".join(t.doc for t in tools)
        if system_prompt_override is not None:
            # The agent supplies its complete, already-rendered system prompt
            # (e.g. the datagen judge / solver prompts). We only splice tool docs
            # where it places `{{ tools_doc }}`, and deliberately do NOT jinja-
            # render it — those prompts contain literal `{...}` (JSON / filter
            # examples) that StrictUndefined would choke on.
            system_prompt = system_prompt_override.replace("{{ tools_doc }}", tools_doc)
        else:
            template = self._SYSTEM_TEMPLATE.replace("{{ tools_doc }}", tools_doc)
            system_prompt = _ENV.from_string(template).render(
                briefing=self.briefing,
                max_steps=self.max_steps,
                final_answer_doc=self.final_answer_doc,
            )
        self._prompt: PromptedCall[_StepOutput] = PromptedCall(
            name=self.name,
            system_prompt=system_prompt,
            default_effort=self.default_effort,
            parse=_parse_step,
            max_parse_retries=self.max_recover_retries,
        )

    def validate_final_answer(self, payload: object, observations: list[str]) -> str | None:
        """Return None to accept, or feedback string to reject (loop continues with it as an observation)."""
        return None

    # ------------------------------------------------------------------
    # Trajectory: blocks, redaction, rendering
    # ------------------------------------------------------------------

    def _block_is_visible(self, block: Block) -> bool:
        """Whether `block` appears in the LLM-facing render. Default: always.
        Subclasses (e.g. `SearchAgent`) override to redact pruned `ChunkBlock`s."""
        return True

    def _blocks_from_output(self, out: CodeOutput) -> list[Block]:
        """Turn one tool-execution result into observation blocks.

        Default: stdout / result rendered as `TextBlock`s — identical to the
        pre-block flat-string observation. Subclasses override to emit
        `ChunkBlock`s for structured (chunk-bearing) tool payloads so the
        chunks can be redacted after a prune."""
        blocks: list[Block] = []
        stdout_s = (out.logs or "").strip()
        if stdout_s:
            blocks.append(TextBlock(f"[stdout]\n{stdout_s}"))
        result_s = "" if out.output is None else str(out.output).strip()
        if result_s and result_s != stdout_s and result_s not in stdout_s:
            blocks.append(TextBlock(f"[result]\n{result_s}"))
        if not blocks:
            blocks.append(TextBlock("[no output]"))
        return blocks

    def _render_for_llm(self) -> list[dict]:
        """Flatten the trajectory to `{role, content}`, dropping invisible blocks
        and any message left empty after redaction."""
        rendered: list[dict] = []
        for msg in self.messages:
            parts = [b.text for b in msg["blocks"] if b.text and self._block_is_visible(b)]
            if not parts:
                continue
            rendered.append({"role": msg["role"], "content": "\n\n".join(parts)})
        return rendered

    def messages_to_jsonable(self) -> list[dict]:
        """JSON-serializable copy of the full (unredacted) trajectory, for
        persistence + downstream reward computation. `self.messages` holds
        dataclass blocks that `json.dump` cannot serialize directly."""
        out: list[dict] = []
        for msg in self.messages:
            blocks_json: list[dict] = []
            for b in msg["blocks"]:
                if isinstance(b, ChunkBlock):
                    blocks_json.append(
                        {"type": "chunk", "chunk_id": b.chunk_id, "doc_id": b.doc_id, "text": b.text}
                    )
                else:
                    blocks_json.append({"type": "text", "text": b.text})
            entry: dict = {"role": msg["role"], "blocks": blocks_json}
            # Assistant turns sampled via a rollout backend carry per-token
            # `token_logprobs` (= log pi_old); the loss mask is reconstructable
            # from `role` at train time (supervise assistant spans only).
            if "logprobs" in msg:
                entry["logprobs"] = msg["logprobs"]
            out.append(entry)
        return out

    async def call(self, ctx: ExecutionContext, user: str, **_) -> Any:
        """Run the multi-turn loop, returning the parsed json final-answer payload. Extra
        kwargs are ignored (signature compat with single-shot calls)."""
        executor = LocalPythonExecutor(additional_authorized_imports=self.authorized_imports)
        # The final answer is parsed outside the sandbox, so it is NOT bound here.
        executor.send_tools({t.name: t for t in self._tools})

        # Full block trajectory (no system message; call() assembles it each turn).
        # `_render_for_llm()` produces the redacted, flattened view sent to the model.
        self.messages = [{"role": "user", "blocks": [TextBlock(user)]}]
        observations: list[str] = []

        ctx.emit(f"question {user!r}")

        # `step` counts only turns that progressed (observation or final answer); `turn`
        # numbers every attempt, including misfired re-prompts.
        step = turn = 0
        warned = False
        while self.max_steps is None or step < self.max_steps:
            step += 1
            if self.warn_steps_remaining is not None and not warned and self.max_steps is not None and self.max_steps - step + 1 <= self.warn_steps_remaining:
                warned = True
                left = self.max_steps - step + 1
                warn = (
                    f"Only {left} of {self.max_steps} steps remain. You should focus your remaining on"
                    f"your most promising lead and avoid wasting time on exploration."
                )
                self.messages.append({"role": "user", "blocks": [TextBlock(warn)]})
                ctx.emit(f"steps_low_warning left={left}")
            # Generate → parse (retried by PromptedCall on format errors) → execute.
            # `done` is set on success; errors append an observation and advance the step.
            done: _StepOutput | None = None
            out: CodeOutput | None = None
            turn += 1
            try:
                step_out = await self._llm_step(ctx)
                assistant_msg: dict = {"role": "assistant", "blocks": [TextBlock(step_out.raw)]}
                # `_last_logprobs` is set by the backend path of `_llm_step`
                # (None on the PromptedCall path); attach it so the trajectory
                # carries `log pi_old` for downstream RL reward computation.
                if self._last_logprobs is not None:
                    assistant_msg["logprobs"] = self._last_logprobs
                self.messages.append(assistant_msg)
                if step_out.code is not None:
                    ctx.emit(f"tool_code {step_out.code!r}")
                    # Inline (not to_thread): tool code runs on this question's worker
                    # thread; blocking here only affects sibling branches of the question.
                    out = executor(step_out.code)
                done = step_out
            except ParseError as e:
                obs = f"Observation (step {turn}): {e.detail}"
                self.messages.append({"role": "user", "blocks": [TextBlock(obs)]})
                ctx.emit(f"error {obs!r}")
            except Exception as e:  # tool code raised
                obs = f"Observation (step {turn}): exec failed — {type(e).__name__}: {e}"
                self.messages.append({"role": "user", "blocks": [TextBlock(obs)]})
                ctx.emit(f"error {obs!r}")

            if done is None:
                continue  # step spent on misfires; advance

            if done.code is None:  # final-answer block
                payload = done.result
                feedback = self.validate_final_answer(payload, observations)
                if feedback is None:
                    return payload
                fb = f"Observation (step {turn}, validation): {feedback}"
                self.messages.append({"role": "user", "blocks": [TextBlock(fb)]})
                observations.append(fb)
                ctx.emit(f"validation_failed {feedback!r}")
                continue

            assert out is not None  # a non-final step that ended ⇒ tool exec succeeded
            # Structured observation: subclasses may emit redactable ChunkBlocks; the
            # default renders stdout / result as TextBlocks (see `_blocks_from_output`).
            obs_blocks: list[Block] = [TextBlock(f"Observation (step {turn}):"), *self._blocks_from_output(out)]
            self.messages.append({"role": "user", "blocks": obs_blocks})
            obs_text = "\n\n".join(b.text for b in obs_blocks if b.text and self._block_is_visible(b))
            observations.append(obs_text)
            ctx.emit(f"observation {obs_text!r}")

        # Out of steps: one forced terminal turn that either commits an answer from the
        # existing observations or hands off to the planner with a diagnostic.
        return await self._terminal_turn(ctx, observations)

    _TERMINAL_PROMPT = (
        "You are out of steps. Do NOT call any tool now — emit exactly ONE ```json``` "
        "block, either:\n"
        "  • COMMIT: if a value already in your observations answers the request, your "
        "best final answer in the required format; or\n"
        '  • NO RESULT: an envelope {"error": "<note>"} with a 2-4 sentence note describing which tools/series you tried, any '
        "candidate values you found, and what blocked you."
    )

    async def _terminal_turn(
        self, ctx: ExecutionContext, observations: list[str]
    ) -> Any:
        diagnostic = ""
        try:
            step_out = await self._llm_step(
                ctx, extra=[{"role": "user", "content": self._TERMINAL_PROMPT}]
            )
            diagnostic = step_out.raw.strip()  # default: the whole reply is the hand-off note
            if step_out.code is None:  # json block
                result = step_out.result
                # `{"error": <note>}` is the explicit give-up envelope (a reserved key the
                # real answer schemas don't use); anything else is a commit candidate.
                if isinstance(result, dict) and list(result) == ["error"]:
                    diagnostic = str(result["error"]).strip()
                    ctx.emit(f"terminal_giveup {diagnostic!r}")
                elif self.validate_final_answer(result, observations) is None:
                    ctx.emit(f"terminal_commit {str(result)!r}")
                    return result
            # else: python block → fall through to StepFailed
        except Exception as e:  # never let the terminal turn mask the real failure
            ctx.emit(f"terminal_turn_failed error={str(e)!r}")
        if not diagnostic:
            tail = observations[-2:]
            diagnostic = "(summary unavailable) recent observations:\n" + "\n".join(tail) if tail else ""
        raise StepFailed(
            self.name, "max steps without accepted final answer", diagnostic=diagnostic
        )

    async def _llm_step(self, ctx: ExecutionContext, extra: list[dict] | None = None) -> _StepOutput:
        """Render the visible trajectory (`_render_for_llm`), collapse stale observations,
        trim to `context_budget_chars`, then route through `PromptedCall.call()` — stopping
        at the first complete fenced block. `extra` appends transient messages (e.g. the
        terminal-turn prompt) that are deliberately NOT stored in `self.messages`."""
        messages = self._render_for_llm()
        # Collapse all but the most recent `visible_observations` tool results to a
        # placeholder (None = keep all), keeping the question + every assistant turn.
        if self.visible_observations is not None:
            user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"]
            # user_idxs[0] is the initial question — always kept; the rest are results.
            keep = set(user_idxs[1:][-self.visible_observations:]) | {user_idxs[0]}
            messages = [
                m if (m["role"] != "user" or i in keep)
                else {"role": "user", "content": "[earlier tool result hidden]"}
                for i, m in enumerate(messages)
            ]
        trimmed = _trim(messages, self.context_budget_chars)
        if extra:
            trimmed = trimmed + extra
        if self._backend is None:
            self._last_logprobs = None
            return await self._prompt.call(ctx, messages=trimmed, should_stop=_has_complete_block)
        # Backend path (e.g. Tinker rollouts): prepend the assembled system
        # prompt, sample one turn synchronously (the rollout owns its thread +
        # loop), stash logprobs for `call()`, then parse. A bad parse raises
        # `ParseError`, which `call()` turns into a recoverable observation.
        system = self._prompt._assemble_system_prompt(ctx)
        rendered = [{"role": "system", "content": system}, *trimmed]
        text, self._last_logprobs = self._backend.generate(
            rendered,
            sampling_params=self._sampling_params,
            capture_logprobs=self._capture_logprobs,
        )
        return _parse_step(text, ctx)
