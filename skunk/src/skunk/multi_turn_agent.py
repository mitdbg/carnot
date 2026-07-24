from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from jinja2 import Environment, StrictUndefined

from skunk.common import B64Image, Effort, ExecutionContext
from skunk.errors import ParseError, StepFailed
from skunk.prompted_call import PromptedCall
from skunk.prompts import load_prompts
from skunk.sandbox.local_python_executor import CodeOutput, LocalPythonExecutor

_ENV = Environment(
    autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined
)

_PROMPTS = load_prompts("multi_turn_agent")
OUT_OF_STEPS_TERMINATE_REASON = "You are out of steps."
OVER_COST_BUDGET_TERMINATE_REASON = "You are over your cost budget."
OVER_LATENCY_BUDGET_TERMINATE_REASON = "You are over your latency budget."


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


@dataclass(frozen=True)
class ImageBlock:
    """An image observation (e.g. a rendered page) the agent can "view". `text` is the
    caption shown in the flattened text stream; `image` carries the base64 payload that
    `_render_for_llm` lifts into the message's `images` list for the multimodal LLM call.
    The raw base64 is deliberately kept out of the JSON-able trajectory (see
    `_block_to_jsonable`)."""

    doc_id: str
    figure_id: str | int | None
    image: B64Image
    text: str


Block = TextBlock | ChunkBlock | ImageBlock


def _block_to_jsonable(b: Block) -> dict:
    """One trajectory block as a JSON-serializable dict (`{type, text, ...}`). Shared
    by `messages_to_jsonable` (full-trajectory persistence) and the per-step
    observation event the trace viewer renders."""
    if isinstance(b, ChunkBlock):
        return {
            "type": "chunk",
            "chunk_id": b.chunk_id,
            "doc_id": b.doc_id,
            "text": b.text,
        }
    if isinstance(b, ImageBlock):
        # Reference only — the base64 payload would bloat the trajectory JSON and the
        # trace-viewer event, so we record just enough to identify the image.
        return {
            "type": "image",
            "doc_id": b.doc_id,
            "figure_id": b.figure_id,
            "mime": b.image.mime,
            "caption": b.text,
            "bytes": len(b.image.data),
        }
    return {"type": "text", "text": b.text}


_FENCE_RE = re.compile(r"```([a-zA-Z0-9_]*)\n(.*?)```", re.DOTALL)


def _stop_at_first_block(acc: str) -> bool:
    """Streaming stop: end generation as soon as one complete fenced block has streamed in.
    The agent acts one tool call (or one final-answer block) per step, so there is never a
    reason to keep generating past the first block."""
    return bool(_FENCE_RE.findall(acc))


@dataclass
class _StepOutput:
    """A successfully parsed model response. `is_final` ⇒ a ```json``` final answer in
    `result` (plain `Any`, so a literal `null` payload is represented as-is). Otherwise
    `code` is the single ```python``` tool-call body to execute. `notice` is an optional
    one-line nudge appended to the observation (set when the reply carried more than one
    fenced block — only the first ran). `raw` carries the verbatim model text so the caller
    can append it to message history."""

    code: str | None = None
    result: Any = None
    is_final: bool = False
    notice: str | None = None
    raw: str = ""


@dataclass
class _CallResult:
    """Outcome of executing a step's single tool-call block. Exactly one of `output` /
    `error` is populated."""

    output: CodeOutput | None = None
    error: str | None = None


def _parse_step(text: str, _: ExecutionContext) -> _StepOutput:
    """Parse the model's fenced block(s) and raise `ParseError` on bad format. Never executes.
    A lone ```json``` block is the final answer; otherwise the FIRST ```python``` block is the
    step's single tool call. Extra blocks beyond the first are dropped with a `notice`."""
    blocks = _FENCE_RE.findall(text)
    if not blocks:
        raise ParseError(
            raw=text,
            detail="Your reply had no fenced block, so nothing ran. Reasoning/prose alone "
            "is not an action. Re-issue your intended action now as a fenced block: a "
            "```python``` block (a single tool call) to act, or ONE ```json``` block to "
            "give your final answer.",
        )
    # A single json block is the final answer; parse it as data.
    if len(blocks) == 1 and blocks[0][0].lower() == "json":
        body = blocks[0][1].strip()
        try:
            return _StepOutput(result=json.loads(body), is_final=True, raw=text)
        except json.JSONDecodeError as e:
            raise ParseError(
                raw=text, detail=f"final-answer JSON was malformed — {e}"
            ) from e
    codes = [body.strip() for lang, body in blocks if lang.lower() != "json"]
    if not codes:
        # Only json block(s), but not a single one ⇒ ambiguous final answer.
        raise ParseError(
            raw=text,
            detail="emit your final answer as a single ```json``` block by itself.",
        )
    notice = None
    if len(blocks) > 1:
        notice = (
            f"You emitted {len(blocks)} fenced blocks; only the first tool call ran. Emit "
            "exactly one ```python``` block per step (or a lone ```json``` final answer)."
        )
    return _StepOutput(code=codes[0], notice=notice, raw=text)


def _trim(messages: list[dict], budget: int) -> list[dict]:
    """Keep first user + the most recent messages fitting in `budget` chars;
    drop the middle behind a placeholder so observation history can't bloat unbounded."""
    if sum(len(m["content"]) for m in messages) <= budget:
        return messages
    head = [
        messages[0],  # first user question — always kept
        {"role": "user", "content": "...(earlier steps truncated)..."},
    ]
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

    # Per-step generation caps, threaded to `PromptedCall.call` → the LLM stream.
    # Both None → provider defaults (uncapped output, no wall-clock timeout),
    # preserving behaviour for every agent that doesn't opt in. `SearchAgent` sets
    # these to bound runaway generations (output ran to the 65535-token ceiling,
    # 200–800s per call) and to cap genuinely hung requests.
    max_output_tokens: int | None = None
    request_timeout_s: float | None = None

    # System-prompt template from prompts/multi_turn_agent.yaml; rendered in __init__();
    # (Jinja `{{ briefing }}` / `{{ tools_doc }}` / `{{ max_steps }}` / `{{ final_answer_doc }}`)
    _SYSTEM_PROMPT_TEMPLATE = _PROMPTS["system_prompt"]

    # Forced-terminal-turn template from prompts/multi_turn_agent.yaml; rendered per call
    # so the reason (out of steps / over budget) can vary.
    # (Jinja `{{ terminate_reason }}`)
    _TERMINAL_PROMPT_TEMPLATE = _PROMPTS["terminal_prompt"]

    max_steps: int = 8
    # Per-step sampling temperature, threaded to `PromptedCall.call` → `astream`.
    # 1.0 per Gemini 3.x guidance: thinking-enabled calls below 1.0 can trap the
    # model in a degenerate reasoning loop that burns the whole output budget
    # (https://ai.google.dev/gemini-api/docs/gemini-3). Subclasses may still override.
    temperature: float = 1.0
    # Hard cap on NON-progressing attempts (parse-misfires / exec-machinery failures) over
    # the whole run. These do NOT consume `max_steps` (only turns that ran a tool call or
    # parsed a final-answer block do); this is the backstop so a model that never emits a
    # valid action can't loop forever. Total attempts ≤ max_steps + max_misfires.
    max_misfires: int = 6
    # Format-error retries per step, delegated to PromptedCall.call(). A step whose
    # all attempts misfire still advances max_steps; execution errors are not retried.
    max_recover_retries: int = 1
    # Hard character cap on the message list as the final safety net (see `_llm_step`).
    context_budget_chars: int = 200_000
    # steps remaining at which to emit a low-budget warning
    warn_steps_remaining: int = 3
    # Extra imports authorized inside the per-step code sandbox. Default: none
    # (tool calls only). Compute-oriented agents (e.g. the task solver) widen
    # this to allow numpy / scipy / statistics / ... in their python steps.
    # A tuple (not a list): class-level mutable defaults are shared across every
    # instance, so an in-place append would leak between agents.
    authorized_imports: Sequence[str] = ()

    def __init__(
        self,
        tools: list[Tool],
        *,
        max_steps: int | None = None,
        max_misfires: int | None = None,
        cost_budget: float | None = None,
        latency_budget: float | None = None,
        agent_id: str | None = None,
        system_prompt_override: str | None = None,
        generation_backend: GenerationBackend | None = None,
        sampling_params: dict | None = None,
        capture_logprobs: bool = False,
    ) -> None:
        # Stable usage-attribution key for this instance. Caller-supplied (e.g. from config) to
        # pin a deterministic key, else a fresh uuid4 so parallel agents sharing one LLMClient
        # keep their per-model token/cost tracking separate.
        self.agent_id = agent_id if agent_id is not None else str(uuid.uuid4())

        self._tools = tools
        # None = keep the class default (mirrors max_misfires) — assigning the bare
        # param used to clobber the declared default to None (an UNBOUNDED loop).
        if max_steps is not None:
            self.max_steps = max_steps
        if max_misfires is not None:
            self.max_misfires = max_misfires

        # state for cost and latency budget awareness
        assert cost_budget is None or cost_budget > 0.0
        assert latency_budget is None or latency_budget > 0.0
        self.cost_budget = cost_budget
        self.latency_budget = latency_budget

        # set start time to None; updated by first invocation of call()
        self.agent_start_time = None

        # state which records the reason for termination; defaults to "finished",
        # updated in _terminal_turn() if need be.
        self.terminate_state = "finished"

        # Full block trajectory of the most recent `call()`; rebuilt per call.
        # Kept on the instance (one agent per question / branch) so callers can
        # read `messages_to_jsonable()` after the run for reward / persistence.
        self.messages: list[dict] = []

        # Optional pluggable generation backend (e.g. Tinker for RL rollouts).
        # When None, generation goes through `PromptedCall` (the LLMClient
        # path). When set, `_llm_step` samples from it and captures per-token
        # logprobs onto each assistant turn for `messages_to_jsonable()`.
        self._backend = generation_backend
        self._sampling_params = sampling_params
        self._capture_logprobs = capture_logprobs
        self._last_logprobs: dict | None = None

        # Overrides are complete prompts, already fully rendered by their caller (datagen /
        # qatfd), so use them verbatim: the base must NOT jinja-render them — they carry
        # literal `{...}` JSON (StrictUndefined chokes) and their own `{{ ... }}` were already
        # filled by the caller — nor read briefing/final_answer_doc (override agents don't set
        # them). Only the built-in template is rendered, with tool docs spliced in as a var.
        if system_prompt_override is not None:
            system_prompt = system_prompt_override
        else:
            tools_doc = "\n\n".join(t.doc for t in tools)
            system_prompt = _ENV.from_string(self._SYSTEM_PROMPT_TEMPLATE).render(
                briefing=self.briefing,
                tools_doc=tools_doc,
                max_steps=self.max_steps,
                final_answer_doc=self.final_answer_doc,
                has_cost_budget=self.cost_budget is not None,
                has_latency_budget=self.latency_budget is not None,
            )
        self._prompt: PromptedCall[_StepOutput] = PromptedCall(
            name=self.name,
            system_prompt=system_prompt,
            default_effort=self.default_effort,
            parse=_parse_step,
            max_parse_retries=self.max_recover_retries,
        )
        # Persistent sandbox — cross-step interpreter state survives across the loop.
        # `call()` swaps in a fresh one per NEW run (a `resume` keeps it, so resumed
        # turns still see the variables earlier steps defined).
        self._executor = self._build_executor()

    def _build_executor(self) -> LocalPythonExecutor:
        """A fresh sandbox with the agent's tools bound. The loop keeps one persistent
        executor across steps so cross-step interpreter state survives."""
        executor = LocalPythonExecutor(
            additional_authorized_imports=list(self.authorized_imports)
        )
        executor.send_tools({t.name: t for t in self._tools})
        return executor

    @staticmethod
    def _run_block(executor: LocalPythonExecutor, code: str) -> _CallResult:
        """Run one tool-call block, capturing the result or the error (never raising)."""
        try:
            return _CallResult(output=executor(code))
        except Exception as e:
            return _CallResult(error=f"{type(e).__name__}: {e}")

    async def _execute_code(self, code: str) -> _CallResult:
        """Run the step's single tool-call block on the persistent executor, offloaded from
        the event loop so this question's sibling branches keep progressing while the
        blocking tool I/O runs."""
        return await asyncio.to_thread(self._run_block, self._executor, code)

    def validate_final_answer(
        self, payload: object, observations: list[str]
    ) -> str | None:
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
        """Flatten the trajectory to `{role, content}` (+ optional `images`), dropping
        invisible blocks and any message left empty after redaction. `ImageBlock`s
        contribute their caption to `content` and their base64 payload to an `images`
        list the multimodal LLM path consumes (text-only callers ignore it)."""
        rendered: list[dict] = []
        for msg in self.messages:
            visible = [b for b in msg["blocks"] if self._block_is_visible(b)]
            parts = [b.text for b in visible if b.text]
            if not parts:
                continue
            entry: dict = {"role": msg["role"], "content": "\n\n".join(parts)}
            images = [b.image for b in visible if isinstance(b, ImageBlock)]
            if images:
                entry["images"] = images
            rendered.append(entry)
        return rendered

    def messages_to_jsonable(self) -> list[dict]:
        """JSON-serializable copy of the full (unredacted) trajectory, for
        persistence + downstream reward computation. `self.messages` holds
        dataclass blocks that `json.dump` cannot serialize directly."""
        out: list[dict] = []
        for msg in self.messages:
            blocks_json = [_block_to_jsonable(b) for b in msg["blocks"]]
            entry: dict = {"role": msg["role"], "blocks": blocks_json}
            # Assistant turns sampled via a rollout backend carry per-token
            # `token_logprobs` (= log pi_old); the loss mask is reconstructable
            # from `role` at train time (supervise assistant spans only).
            if "logprobs" in msg:
                entry["logprobs"] = msg["logprobs"]
            out.append(entry)
        return out

    async def call(
        self, ctx: ExecutionContext, user: str, *, resume: bool = False,
        max_steps: int | None = None, **_
    ) -> Any:
        """Run the multi-turn loop, returning the parsed json final-answer payload.
        `resume=True` appends `user` to the existing trajectory (with a fresh step budget)
        instead of starting over — the seam for a reviewer sending the agent back with
        feedback without it re-deriving everything it already saw. `max_steps` overrides the
        agent's default budget for this call only (e.g. a short, resumed correction turn that
        should not get a full search budget); it defaults to `self.max_steps`. Extra kwargs
        are ignored (signature compat with single-shot calls)."""
        # mark the time at which the agent started executing; do not overwrite the start time
        # if it has already been set (this may happen if an agent uses call() for retries)
        if self.agent_start_time is None:
            self.agent_start_time = time.monotonic()

        # Full block trajectory (no system message; call() assembles it each turn).
        # `_render_for_llm()` produces the redacted, flattened view sent to the model.
        if resume and self.messages:
            # Keep the executor too — a resumed turn can still read the variables
            # earlier steps defined (rebuilding here used to silently drop them).
            self.messages.append({"role": "user", "blocks": [TextBlock(user)]})
        else:
            self.messages = [{"role": "user", "blocks": [TextBlock(user)]}]
            # Fresh sandbox per new run; the final answer is parsed outside the
            # sandbox, so it is NOT bound here.
            self._executor = self._build_executor()

        # Capture the system prompt + opening question into the event stream so the
        # trace viewer can show them (the console keeps only the one-liners —
        # the full text rides in `data`). The system prompt is static for the call.
        system_prompt = self._prompt.assemble_system_prompt(ctx)
        ctx.emit(
            f"system_prompt chars={len(system_prompt)}",
            kind="system",
            data={"text": system_prompt},
        )
        ctx.emit(f"question {user!r}", kind="user", data={"text": user})
        return await self._run_loop(ctx, self.max_steps if max_steps is None else max_steps)

    @staticmethod
    def steps_and_turns_left(step: int, max_steps: int, turn: int, max_turns: int) -> bool:
        """True iff the agent still has steps (number of actions without error) or turns
        (number of actions including errors) left."""
        return step < max_steps and turn < max_turns

    def cost_budget_left(self, ctx: ExecutionContext) -> bool:
        """True iff the agent has not exceeded its cost budget."""
        return self.cost_budget is None or ctx.llm_client.usage.cost(key=str(self.agent_id)) < self.cost_budget

    def latency_budget_left(self) -> bool:
        """True iff the agent has not exceeded its latency budget."""
        assert self.agent_start_time is not None
        return self.latency_budget is None or time.monotonic() - self.agent_start_time < self.latency_budget

    def warn_low_steps(self, step: int, max_steps: int) -> bool:
        """Warn the model if it needs to produce a final answer if it has <= self.warn_steps_remaining left."""
        return max_steps - step <= self.warn_steps_remaining

    def _get_terminate_reason(self, out_of_steps: bool, over_cost_budget: bool, over_latency_budget: bool) -> tuple[str, str]:
        """Returns a string and explaining the reason(s) for termination and an identifier for the emitted log."""
        terminate_reason, terminate_state = "", ""
        if out_of_steps:
            terminate_reason = OUT_OF_STEPS_TERMINATE_REASON
            terminate_state = "out_of_steps"
        if over_cost_budget:
            terminate_reason += f" {OVER_COST_BUDGET_TERMINATE_REASON}"
            terminate_reason = terminate_reason.lstrip()
            terminate_state += "|over_cost_budget"
            terminate_state = terminate_state.lstrip("|")
        if over_latency_budget:
            terminate_reason += f" {OVER_LATENCY_BUDGET_TERMINATE_REASON}"
            terminate_reason = terminate_reason.lstrip()
            terminate_state += "|over_latency_budget"
            terminate_state = terminate_state.lstrip("|")

        assert len(terminate_reason) > 0 and len(terminate_state) > 0
        return terminate_reason, terminate_state

    def add_budget_observations(self, ctx: ExecutionContext, obs_blocks: list[Block]) -> None:
        """Add TextBlocks tracking the context, cost, and latency usage of the agent.
        Cost and latency are tracked iff self.cost_budget and self.latency_budget are not None, respectively."""
        ctx_chars = sum(len(m["content"]) for m in self._render_for_llm())
        obs_blocks.append(TextBlock(
            f"[Context usage: {ctx_chars:,}/{self.context_budget_chars:,} chars "
            f"({ctx_chars / self.context_budget_chars * 100:.1f}%)]"
        ))
        if self.cost_budget is not None:
            cost_usage = ctx.llm_client.usage.cost(key=str(self.agent_id))
            obs_blocks.append(TextBlock(f"[Cost usage: ${cost_usage:.2f}/${self.cost_budget} ({(cost_usage / self.cost_budget * 100):.1f}%)]"))
        if self.latency_budget is not None:
            assert self.agent_start_time is not None
            latency_usage = time.monotonic() - self.agent_start_time
            obs_blocks.append(TextBlock(f"[Latency usage: {latency_usage:.1f}/{self.latency_budget}s ({(latency_usage / self.latency_budget * 100):.1f}%)]"))

    async def _run_loop(self, ctx: ExecutionContext, max_steps: int) -> Any:
        """The step loop, bounded by `max_steps`. Assumes `self.messages` and
        `self._executor` are already established. `step` counts only turns that PROGRESSED
        (ran a tool call or parsed a final-answer block); parse-misfires / exec-machinery
        failures advance `turn` but not `step`, so they don't burn the step budget. `turn`
        is hard-capped at `max_steps + max_misfires` so a never-progressing model still
        terminates. Returns the validated payload, or hands off to `_terminal_turn()`."""
        observations: list[str] = []

        # `step` counts only turns that progressed (observation or final answer); `turn`
        # numbers every attempt, including misfired re-prompts (which don't cost a step).
        step = turn = 0
        max_turns = max_steps + max(0, self.max_misfires)
        while self.steps_and_turns_left(step, max_steps, turn, max_turns) and self.cost_budget_left(ctx) and self.latency_budget_left():
            if self.warn_low_steps(step, max_steps):
                # add message to agent and emit so it can also be shown in trace viewer
                left = max_steps - step
                warn = (
                    f"Only {left} of {max_steps} non-error steps remain. You should focus your "
                    f"remaining steps on your most promising lead and avoid wasting time on exploration."
                )
                self.messages.append({"role": "user", "blocks": [TextBlock(warn)]})
                ctx.emit(f"steps_low_warning left={left}", data={"text": warn})

            # Generate → parse (retried by PromptedCall on format errors) → execute.
            # `done` is set on success; errors append an observation and advance `turn` only.
            done: _StepOutput | None = None
            result: _CallResult | None = None
            turn += 1
            try:
                step_out = await self._llm_step(ctx)
                assistant_msg: dict = {
                    "role": "assistant",
                    "blocks": [TextBlock(step_out.raw)],
                }
                # `_last_logprobs` is set by the backend path of `_llm_step`
                # (None on the PromptedCall path); attach it so the trajectory
                # carries `log pi_old` for downstream RL reward computation.
                if self._last_logprobs is not None:
                    assistant_msg["logprobs"] = self._last_logprobs
                self.messages.append(assistant_msg)
                ctx.emit(
                    f"assistant step={turn} chars={len(step_out.raw)}",
                    kind="assistant",
                    data={"text": step_out.raw},
                )
                if not step_out.is_final:
                    # Run the step's single tool-call block, offloaded so the event loop stays
                    # free for this question's sibling branches (see `_execute_code`).
                    assert step_out.code is not None
                    ctx.emit(f"tool_code {step_out.code!r}")
                    result = await self._execute_code(step_out.code)
                done = step_out
            except ParseError as e:
                obs = f"Observation (step {turn}): {e.detail}"
                obs_blocks = [TextBlock(obs)]
                self.add_budget_observations(ctx, obs_blocks)
                self.messages.append({"role": "user", "blocks": obs_blocks})
                ctx.emit(f"error {obs!r}")
            except Exception as e:  # LLM step / batch machinery failed
                obs = f"Observation (step {turn}): exec failed — {type(e).__name__}: {e}"
                obs_blocks = [TextBlock(obs)]
                self.add_budget_observations(ctx, obs_blocks)
                self.messages.append({"role": "user", "blocks": obs_blocks})
                ctx.emit(f"error {obs!r}")

            if done is None:
                continue  # misfire: `turn` advanced, `step` budget untouched

            # The turn progressed (a tool call ran, or a final-answer block parsed) — only
            # now does it count against the step budget. A validation-failed final answer
            # still progressed, so it too costs a step.
            step += 1

            if done.is_final:  # final-answer block
                payload = done.result
                feedback = self.validate_final_answer(payload, observations)
                if feedback is None:
                    return payload
                obs = f"Observation (step {turn}, validation): {feedback}"
                obs_blocks = [TextBlock(obs)]
                self.add_budget_observations(ctx, obs_blocks)
                self.messages.append({"role": "user", "blocks": obs_blocks})
                observations.append(obs)
                ctx.emit(f"validation_failed {feedback!r}")
                continue

            assert result is not None  # non-final step ⇒ the tool call ran
            # The step's observation: the tool result rendered via `_blocks_from_output`
            # (subclasses may emit redactable ChunkBlocks), or an `[error]` block on failure,
            # plus an over-emission `[notice]` if the model sent more than one block.
            obs_blocks: list[Block] = [TextBlock(f"Observation (step {turn}):")]
            if result.error is not None:
                obs_blocks.append(TextBlock(f"[error]\n{result.error}"))
            else:
                assert result.output is not None
                obs_blocks.extend(self._blocks_from_output(result.output))
            if done.notice:
                obs_blocks.append(TextBlock(f"[notice] {done.notice}"))

            # add messages to inform the agent of its budget usage
            self.add_budget_observations(ctx, obs_blocks)
            
            self.messages.append({"role": "user", "blocks": obs_blocks})
            visible_blocks = [
                b for b in obs_blocks if b.text and self._block_is_visible(b)
            ]
            obs_text = "\n\n".join(b.text for b in visible_blocks)
            observations.append(obs_text)
            # Structured observation for the viewer: keep each block typed (chunk blocks
            # carry chunk_id/doc_id) so it can render them like the agent saw them.
            ctx.emit(
                f"observation {obs_text!r}",
                kind="observation",
                data={"blocks": [_block_to_jsonable(b) for b in visible_blocks]},
            )

        # Out of steps or budget: one forced terminal turn that either commits an answer from the
        # existing observations or hands off to the planner with a diagnostic.
        out_of_steps = not self.steps_and_turns_left(step, max_steps, turn, max_turns)
        over_cost_budget = not self.cost_budget_left(ctx)
        over_latency_budget = not self.latency_budget_left()
        return await self._terminal_turn(ctx, observations, out_of_steps, over_cost_budget, over_latency_budget)

    async def _terminal_turn(
        self,
        ctx: ExecutionContext,
        observations: list[str],
        out_of_steps: bool,
        over_cost_budget: bool,
        over_latency_budget: bool,
    ) -> Any:
        # hand-off note; set from the reply below, but the try may raise first
        diagnostic = ""

        # get the reason for termination
        terminate_reason, self.terminate_state = self._get_terminate_reason(out_of_steps, over_cost_budget, over_latency_budget)
        terminal_prompt = _ENV.from_string(self._TERMINAL_PROMPT_TEMPLATE).render(
            terminate_reason=terminate_reason
        )

        # Record the terminal prompt the agent was shown, so the viewer can tell the
        # full story of the forced terminal turn (prompt → reply → outcome).
        ctx.emit(f"terminal_prompt {self.terminate_state}", data={"text": terminal_prompt})
        try:
            step_out = await self._llm_step(
                ctx, extra=[{"role": "user", "content": terminal_prompt}]
            )
            ctx.emit(
                f"terminal_reply chars={len(step_out.raw)}", data={"text": step_out.raw}
            )
            diagnostic = (
                step_out.raw.strip()
            )  # default: the whole reply is the hand-off note
            if step_out.is_final:  # json block
                result = step_out.result
                # `{"error": <note>}` is the explicit give-up envelope (a reserved key the
                # real answer schemas don't use); anything else is a commit candidate.
                if isinstance(result, dict) and list(result) == ["error"]:
                    diagnostic = str(result["error"]).strip()
                    ctx.emit(f"terminal_giveup {diagnostic!r}", data={"text": diagnostic})
                elif self.validate_final_answer(result, observations) is None:
                    ctx.emit(
                        f"terminal_commit {str(result)!r}", data={"text": str(result)}
                    )
                    return result
            # else: python block → fall through to StepFailed
        except Exception as e:  # never let the terminal turn mask the real failure
            ctx.emit(
                f"terminal_turn_failed error={str(e)!r}",
                data={"text": f"{type(e).__name__}: {e}"},
            )
        if not diagnostic:
            tail = observations[-2:]
            diagnostic = (
                "(summary unavailable) recent observations:\n" + "\n".join(tail)
                if tail
                else ""
            )
        raise StepFailed(
            self.name, "max steps without accepted final answer", diagnostic=diagnostic
        )

    async def _llm_step(
        self, ctx: ExecutionContext, extra: list[dict] | None = None
    ) -> _StepOutput:
        """Render the visible trajectory (`_render_for_llm`), trim to `context_budget_chars`,
        and then route through `PromptedCall.call()` — stopping once the step's block completes
        (`_stop_at_first_block`). `extra` appends transient messages (e.g. the terminal-turn prompt)
        that are deliberately NOT stored in `self.messages`."""
        messages = self._render_for_llm()
        trimmed = _trim(messages, self.context_budget_chars)
        if extra:
            trimmed = trimmed + extra
        if self._backend is None:
            self._last_logprobs = None
            # Model resolution is delegated to the prompt layer (`_resolve_model`):
            # `model_overrides.get(call_site_name, llm_model)`, same as every other call.
            return await self._prompt.call_multi_turn(
                ctx, trimmed, usage_key=str(self.agent_id), should_stop=_stop_at_first_block,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                timeout_s=self.request_timeout_s,
            )
        # Backend path (e.g. Tinker rollouts): prepend the assembled system
        # prompt, sample one turn synchronously (the rollout owns its thread +
        # loop), stash logprobs for `call()`, then parse. A bad parse raises
        # `ParseError`, which `call()` turns into a recoverable observation.
        system = self._prompt.assemble_system_prompt(ctx)
        rendered = [{"role": "system", "content": system}, *trimmed]
        text, self._last_logprobs = self._backend.generate(
            rendered,
            sampling_params=self._sampling_params,
            capture_logprobs=self._capture_logprobs,
        )
        return _parse_step(text, ctx)
