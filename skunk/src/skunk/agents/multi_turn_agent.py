from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from skunk.common import B64Image, Effort, ExecutionContext, estimate_tokens
from skunk.config import AgentConfig
from skunk.constants import IMAGE_TOKENS_EST
from skunk.errors import ParseError, StepFailed
from skunk.llm_client import LLMResponse
from skunk.usage import match_model_entry
from skunk.sandbox.local_python_executor import CodeOutput, LocalPythonExecutor

OUT_OF_STEPS_TERMINATE_REASON = "You are out of steps."
OVER_COST_BUDGET_TERMINATE_REASON = "You are over your cost budget."
OVER_LATENCY_BUDGET_TERMINATE_REASON = "You are over your latency budget."


# ---------------------------------------------------------------------------
# Trajectory blocks and messages
#
# The trajectory is stored as a list of {role, blocks} messages. Blocks contain
# the content of the message (text / image) as well as metadata (doc_id, chunk_id)
# which help with tracking statistics like document and chunk recall within an
# agent's trace. Additionally, each block maintains its visibility status via a
# _visible flag. Blocks may be marked invisible to make room when the agent's context
# window overflows. Some agents may also use this feature to remove information that
# is irrelevant to their work.
# ---------------------------------------------------------------------------

@dataclass
class Block:
    visible: bool = field(default=True, kw_only=True)

@dataclass
class TextBlock(Block):
    text: str 

@dataclass
class ChunkBlock(Block):
    chunk_id: str | None
    doc_id: str
    text: str

@dataclass
class ImageBlock(Block):
    """An image observation (e.g. a rendered page) the agent can "view". `text` is the
    caption shown in the flattened text stream; `image` carries the base64 payload that
    `_render_for_llm` lifts into the message's `images` list for the multimodal LLM call.
    The raw base64 is deliberately kept out of the JSON-able trajectory (see
    `_block_to_jsonable`)."""

    doc_id: str
    image: B64Image
    text: str

@dataclass
class Message:
    role: str
    blocks: list[Block]

def _block_to_jsonable(b: Block) -> dict:
    """One trajectory block as a JSON-serializable dict (`{type, text, ...}`). Shared
    by `_messages_to_jsonable` (full-trajectory persistence) and the per-step
    observation event the trace viewer renders."""
    if isinstance(b, ChunkBlock):
        return {
            "type": "chunk",
            "chunk_id": b.chunk_id,
            "doc_id": b.doc_id,
            "text": b.text,
        }
    elif isinstance(b, ImageBlock):
        # Reference only — the base64 payload would bloat the trajectory JSON and the
        # trace-viewer event, so we record just enough to identify the image.
        return {
            "type": "image",
            "doc_id": b.doc_id,
            "mime": b.image.mime,
            "caption": b.text,
            "bytes": len(b.image.data),
        }
    elif isinstance(b, TextBlock):
        return {"type": "text", "text": b.text}

    else:
        raise Exception(f"Unexpected block type: {type(b)}")

@dataclass
class StepOutput:
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
class _Observation:
    """The outcome of a single step. If the LLM generated a single tool-call block and it
    executed without an exception, the output is stored in `output`. Otherwise, if the LLM
    failed to generate a proper tool call, or executing the tool call produced an error,
    then an error message is stored in `error`. Exactly one of `output` / `error` is populated."""

    output: CodeOutput | None = None
    error: str | None = None


def _count_tokens(blocks: list[Block]) -> int:
    """Count the tokens in each block via `common.estimate_tokens` (exact BPE count,
    chars/4 above its size cutoff). An `ImageBlock` counts its caption text plus a flat
    per-image estimate for the base64 payload (`IMAGE_TOKENS_EST` — the payload's real
    token cost is provider-specific)."""
    total_tokens = 0
    for block in blocks:
        assert isinstance(block, (TextBlock, ChunkBlock, ImageBlock)), "Can only estimate TextBlock | ChunkBlock | ImageBlock"
        total_tokens += estimate_tokens(block.text)
        if isinstance(block, ImageBlock):
            total_tokens += IMAGE_TOKENS_EST

    return total_tokens


class Tool(ABC):
    # the name of the tool; should match the function name
    name: str
    # the tool docstring
    doc: str
    # True if the tool reclaims space in the context window and False otherwise
    reclaimer: bool = False

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """The tool's runtime behaviour."""

_FENCE_RE = re.compile(r"```([a-zA-Z0-9_]*)\n(.*?)```", re.DOTALL)

def parse_step(text: str) -> StepOutput:
    """Parse the model's fenced block(s) and raise `ParseError` on bad format. Expects a
    lone ```json``` block to be the final answer; otherwise the FIRST ```python``` block is the
    step's single tool call. Extra blocks beyond the first are dropped with a `notice`."""
    blocks = _FENCE_RE.findall(text)
    if not blocks:
        raise ParseError(
            detail="Your reply had no fenced block, so nothing ran. Reasoning/prose alone "
            "is not an action. Re-issue your intended action now as a fenced block: a "
            "```python``` block (a single tool call) to act, or ONE ```json``` block to "
            "give your final answer.",
        )
    # A single json block is the final answer; parse it as data.
    if len(blocks) == 1 and blocks[0][0].lower() == "json":
        body = blocks[0][1].strip()
        try:
            return StepOutput(result=json.loads(body), is_final=True)
        except json.JSONDecodeError as e:
            raise ParseError(
                detail=f"final-answer JSON was malformed — {e}"
            ) from e
    codes = [body.strip() for lang, body in blocks if lang.lower() != "json"]
    if not codes:
        # Only json block(s), but not a single one ⇒ ambiguous final answer.
        raise ParseError(
            detail="emit your final answer as a single ```json``` block by itself.",
        )
    notice = None
    if len(blocks) > 1:
        notice = (
            f"You emitted {len(blocks)} fenced blocks; only the first tool call ran. Emit "
            "exactly one ```python``` block per step (or a lone ```json``` final answer)."
        )
    return StepOutput(code=codes[0], notice=notice)


class MultiTurnAgent(ABC):
    """A tool-loop agent that takes a series of tool call actions and produces a final answer.
    
    Accepts an AgentConfig, set of tools, system prompt, terminal prompt, and parser. The system
    prompt instructs the agent what the expected format is for the output at each step. The parser
    is responsible for parsing the output and putting it into a StepOutput or raising a ParseError
    which stores any feedback for the agent in `ParseError.detail`. The terminal prompt informs
    the agent how to act in its final turn after it has exhausted its step budget. The output from
    this terminal step will also be parsed by the parser.

    TODO: tie system_prompt, terminal_prompt, and parse into a `StepProtocol` to make their
    coupling more obvious.
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: list[Tool],
        system_prompt: str,
        terminal_prompt: str,
        parse: Callable[[str], StepOutput],
    ) -> None:
        self.config = config
        self.agent_id = config.agent_id
        self._tools = tools
        self._system_prompt = system_prompt
        self._terminal_prompt = terminal_prompt
        self._parse = parse

        # current step, turn, and number of misfires (turn tracks progress within a step)
        self._step = self._turn = self._misfires = 0

        # set start time to None; updated by first invocation of call()
        self._agent_start_time = None

        # state which records the reason for termination; defaults to "finished",
        # updated in _terminal_turn() if need be
        self._terminate_state = "finished"

        # full block trajectory of the most recent `call()`, rebuilt per call;
        # callers can read `_messages_to_jsonable()` after the run for reward / persistence
        self._messages: list[Message] = []

        # Number of tokens in the agent's context window. RESET to the provider-reported
        # total (input + output) after every LLM call in `_llm_step`; incremented with
        # tiktoken/`IMAGE_TOKENS_EST` estimates as observations are appended between calls.
        # Transient drift is expected and bounded by one step, since every reset re-anchors
        # to ground truth: parse retries fold their retry exchange into the reported total
        # though it is never persisted to `self._messages`, and the estimator's tokenizer
        # (o200k_base) differs from the actual model's. The Tinker backend path never
        # updates it (RL rollouts don't read it).
        self._tokens_in_context = 0

        # Persistent sandbox — cross-step interpreter state survives across the loop.
        # `call()` swaps in a fresh one per NEW run (a `resume` keeps it, so resumed
        # turns still see the variables earlier steps defined).
        self._executor = self._build_executor()

    def _resolve_effort(self, ctx: ExecutionContext) -> Effort:
        """Effort setting for LLM calls; falls back to inference config default if not specified."""
        return ctx.config.inference.effort if self.config.effort is None else self.config.effort

    def _resolve_model(self, ctx: ExecutionContext) -> str:
        """Model to use for generation; falls back to inference config default if not specified."""
        return ctx.config.inference.llm_model if self.config.llm_model is None else self.config.llm_model

    def _get_context_limit(self, ctx: ExecutionContext) -> int:
        """Returns the context limit for the agent based on the context limit of its model.
        `llm_context_limits` is an id-or-SUBSTRING map (e.g. a bare "gemini-3.5-flash" key must
        match the full "google/gemini-3.5-flash" id), so resolve via `match_model_entry` like
        every other per-model map — an exact-key lookup here breaks on substring keys."""
        model = self._resolve_model(ctx)
        limit = match_model_entry(model, ctx.config.inference.llm_context_limits)
        assert limit is not None, f"Model {model} not found in context limits"
        return int(limit)

    def _get_soft_token_limit(self, ctx: ExecutionContext) -> int:
        """Returns the soft token limit for the agent based on the context limit of its model."""
        context_limit = self._get_context_limit(ctx)
        return int(context_limit * self.config.context_soft_safety_frac)

    def _get_hard_token_limit(self, ctx: ExecutionContext) -> int:
        """Returns the hard token limit for the agent based on the context limit of its model."""
        context_limit = self._get_context_limit(ctx)
        return int(context_limit * self.config.context_hard_safety_frac)

    def _build_executor(self) -> LocalPythonExecutor:
        """A fresh sandbox with the agent's tools bound. The loop keeps one persistent
        executor across steps so cross-step interpreter state survives."""
        executor = LocalPythonExecutor(
            additional_authorized_imports=list(self.config.authorized_imports)
        )
        executor.send_tools({t.name: t for t in self._tools})
        return executor

    @staticmethod
    def _run_block(executor: LocalPythonExecutor, code: str) -> _Observation:
        """Run one tool-call block, capturing the result or the error (never raising)."""
        try:
            return _Observation(output=executor(code))
        except Exception as e:
            return _Observation(error=f"{type(e).__name__}: {e}")

    async def _execute_code(self, code: str) -> tuple[_Observation, float]:
        """Run the step's single tool-call block on the persistent executor, offloaded from
        the event loop so this question's sibling branches keep progressing while the
        blocking tool I/O runs."""
        tool_call_start_time = time.monotonic()
        observation = await asyncio.to_thread(self._run_block, self._executor, code)
        return observation, round(time.monotonic() - tool_call_start_time, 3)

    def _make_blocks_invisible(self, block: Block | None = None, doc_ids: Sequence[str] | None = None, chunk_ids: Sequence[str] | None = None) -> None:
        """Sets the block's visibility to False. If `doc_id` or `chunk_id` are provided, all blocks in
        self._messages with the doc/chunk_id have their visibility set to False."""
        if block is not None:
            block.visible = False

        if doc_ids is not None or chunk_ids is not None:
            for msg in self._messages:
                for block in msg.blocks:
                    if isinstance(block, ChunkBlock) and doc_ids is not None and block.doc_id in doc_ids:
                        block.visible = False
                    if isinstance(block, ChunkBlock) and chunk_ids is not None and block.chunk_id in chunk_ids:
                        block.visible = False
                    if isinstance(block, ImageBlock) and doc_ids is not None and block.doc_id in doc_ids:
                        block.visible = False

    def _blocks_from_output(self, ctx: ExecutionContext, out: CodeOutput) -> list[Block]:
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
        for msg in self._messages:
            visible = [b for b in msg.blocks if b.visible]
            text_parts = [b.text for b in visible if isinstance(b, (ChunkBlock, TextBlock, ImageBlock)) and b.text]
            if not text_parts:
                continue
            entry: dict = {"role": msg.role, "content": "\n\n".join(text_parts)}
            images = [b.image for b in visible if isinstance(b, ImageBlock)]
            if images:
                entry["images"] = images
            rendered.append(entry)
        return rendered

    def _messages_to_jsonable(self) -> list[dict]:
        """JSON-serializable copy of the full (unredacted) trajectory, for
        persistence + downstream reward computation. `self._messages` holds
        dataclass blocks that `json.dump` cannot serialize directly."""
        out: list[dict] = []
        for msg in self._messages:
            blocks_json = [_block_to_jsonable(b) for b in msg.blocks]
            entry: dict = {"role": msg.role, "blocks": blocks_json}
            out.append(entry)
        return out

    def _steps_and_misfires_left(self) -> bool:
        """True iff the agent still has steps (number of actions without error) or misfires
        (number of actions producing errors) left."""
        return self._step < self.config.max_steps and self._misfires < self.config.max_misfires

    def _cost_budget_left(self, ctx: ExecutionContext) -> bool:
        """True iff the agent has not exceeded its cost budget."""
        return self.config.cost_budget is None or ctx.llm_client.usage.cost(key=self.agent_id) < self.config.cost_budget

    def _latency_budget_left(self) -> bool:
        """True iff the agent has not exceeded its latency budget."""
        assert self._agent_start_time is not None
        return self.config.latency_budget is None or time.monotonic() - self._agent_start_time < self.config.latency_budget

    def _warn_low_steps(self) -> bool:
        """Warn the model if it needs to produce a final answer if it has <= self.warn_steps_remaining left."""
        return self.config.max_steps - self._step <= self.config.warn_steps_remaining

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

    def _warn_context_limits(self, ctx: ExecutionContext) -> None:
        """Warn the model if the context usage exceeds the soft or hard token limits."""
        soft_token_max_after_step = self._tokens_in_context + (self.config.max_output_tokens or 0)
        hard_token_max_after_step = self._tokens_in_context + (self.config.max_output_tokens or 0)
        soft_token_limit = self._get_soft_token_limit(ctx)
        hard_token_limit = self._get_hard_token_limit(ctx)
        warn_msg = ""
        if hard_token_max_after_step >= hard_token_limit:
            usage_msg = "context usage" if self.config.max_output_tokens is None else "context usage + max output tokens"
            reclaim_tools = [t.name for t in self._tools if t.reclaimer]
            if reclaim_tools:
                reclaim_msg = f"You may only invoke a tool which reclaims context space (one of: {reclaim_tools}) or return your final answer."
            else:
                reclaim_msg = "You must return your final answer."
            warn_msg = f"[warning] {usage_msg} {hard_token_max_after_step:,} tokens exceeds hard limit {hard_token_limit:,} tokens. {reclaim_msg} All other tool calls will be rejected on your next step."
            ctx.tracer.emit(
                id="context_hard_limit_exceeded",
                level="warning",
                kind="lifecycle",
                step=self._step,
                turn=self._turn,
                message=warn_msg,
                data={"text": warn_msg, "tokens": hard_token_max_after_step, "limit": hard_token_limit}
            )
        elif soft_token_max_after_step >= soft_token_limit:
            usage_msg = "context usage" if self.config.max_output_tokens is None else "context usage + max output tokens"
            reclaim_tools = [t.name for t in self._tools if t.reclaimer]
            reclaim_msg = ""
            if reclaim_tools:
                reclaim_msg = f" Consider using a tool which reclaims context space (one of: {reclaim_tools}) to reduce the tokens in your context window on the next step."
            warn_msg = f"[warning] {usage_msg} {soft_token_max_after_step:,} tokens exceeds soft limit {soft_token_limit:,} tokens.{reclaim_msg}"
            ctx.tracer.emit(
                id="context_soft_limit_exceeded",
                level="warning",
                kind="lifecycle",
                step=self._step,
                turn=self._turn,
                message=warn_msg,
                data={"text": warn_msg, "tokens": soft_token_max_after_step, "limit": soft_token_limit}
            )
        if warn_msg:
            self._messages.append(Message(role="user", blocks=[TextBlock(warn_msg)]))

    def _trim(self, ctx: ExecutionContext) -> None:
        """Trim the messages to fit within the model's context limit. Go through the list of messages
        in reverse and redact until our estimate is under the model's context limit."""
        context_limit = self._get_context_limit(ctx)
        redacted_msg = f"[notice]\nIn order to trim your context to fit within the {context_limit:,} token limit, we've redacted messages which contained the following docs/chunks:\n"
        trimmed = False
        for msg in reversed(self._messages):
            # break once we're estimated to be below the token limit
            redacted_msg_tokens = _count_tokens([TextBlock(redacted_msg)]) if trimmed else 0
            if self._tokens_in_context + redacted_msg_tokens + (self.config.max_output_tokens or 0) < context_limit:
                break

            trimmed = True
            for block in msg.blocks:
                if isinstance(block, TextBlock) or not block.visible:
                    continue

                # redact block
                est_tokens = _count_tokens([block])
                self._make_blocks_invisible(block)
                if isinstance(block, ChunkBlock) and block.chunk_id is not None:
                    redacted_msg += f" - chunk_id={block.chunk_id} | est. tokens={est_tokens}\n"
                elif isinstance(block, (ImageBlock, ChunkBlock)):
                    redacted_msg += f" - doc_id={block.doc_id} | est. tokens={est_tokens}\n"

                # adjust block tokens
                self._tokens_in_context -= est_tokens

                # update redacted message, recompute tokens, and break early if under limit
                redacted_msg_tokens = _count_tokens([TextBlock(redacted_msg)])
                if self._tokens_in_context + redacted_msg_tokens + (self.config.max_output_tokens or 0) < context_limit:
                    break

        if trimmed:
            self._messages.append(Message(role="user", blocks=[TextBlock(redacted_msg)]))

    def _add_budget_observations(self, ctx: ExecutionContext, obs_blocks: list[Block]) -> tuple[float | None, float | None]:
        """Add TextBlocks tracking the context, cost, and latency usage of the agent.
        Cost and latency are tracked iff self.config.cost_budget and self.config.latency_budget are not None, respectively."""
        context_limit = self._get_context_limit(ctx)
        obs_blocks.append(TextBlock(
            f"[Context usage: ~{self._tokens_in_context:,}/{context_limit:,} tokens "
            f"({self._tokens_in_context / context_limit * 100:.1f}%)]"
        ))
        cost_usage, latency_usage = None, None
        if self.config.cost_budget is not None:
            cost_usage = ctx.llm_client.usage.cost(key=self.agent_id)
            obs_blocks.append(TextBlock(f"[Cost usage: ${cost_usage:.2f}/${self.config.cost_budget} ({(cost_usage / self.config.cost_budget * 100):.1f}%)]"))
        if self.config.latency_budget is not None:
            assert self._agent_start_time is not None
            latency_usage = time.monotonic() - self._agent_start_time
            obs_blocks.append(TextBlock(f"[Latency usage: {latency_usage:.1f}/{self.config.latency_budget}s ({(latency_usage / self.config.latency_budget * 100):.1f}%)]"))

        return cost_usage, latency_usage

    async def _terminal_turn(
        self,
        ctx: ExecutionContext,
        cost_usage: float | None,
        latency_usage: float | None,
        out_of_steps: bool,
        over_cost_budget: bool,
        over_latency_budget: bool,
    ) -> StepOutput:
        step_start_time = time.monotonic()

        # get the reason for termination
        terminate_reason, self._terminate_state = self._get_terminate_reason(out_of_steps, over_cost_budget, over_latency_budget)

        # add termination notice and terminal prompt to the agent's messages
        obs_blocks: list[Block] = [
            TextBlock(f"[notice]\n You are being forced to terminate because: {terminate_reason}"),
            TextBlock(self._terminal_prompt),
        ]
        self._messages.append(Message(role="user", blocks=obs_blocks))
        self._tokens_in_context += _count_tokens(obs_blocks)

        # TODO: if we bring back `resume=`, then we should maintain the state fully (i.e. self._messages,
        #       self._tokens_in_context, etc.) to help the agent after resumption
        resp = step_output = observation = None
        try:
            resp = await self._llm_step(ctx)
            step_output = self._parse(resp.text)

            if not step_output.is_final:
                raise Exception("Failed to issue a final answer in the terminal step.")

            return step_output.result

        except ParseError as e:
            observation = _Observation(error=e.detail)
        except Exception as e:
            observation = _Observation(error=f"{type(e).__name__}: {e}")

        # emit an event for the termination step
        ctx.tracer.emit(
            id="agent_step",
            kind="assistant",
            step=self._step,
            turn=self._turn,
            data={
                # raw output generated by llm
                "text": resp.text if resp else None,
                # StepOutput fields (is_final, result, parsed tool code, notice)
                # NOTE: should enforce that `step_output.result` is JSON serializable before logging
                "is_final": step_output.is_final if step_output else None,
                "result": str(step_output.result) if step_output else None,
                "code": step_output.code if step_output else None,
                "notice": step_output.notice if step_output else None,
                # _Observation 
                "observation_blocks": [_block_to_jsonable(b) for b in obs_blocks] if observation.output else None,
                "error": observation.error,
                # latencies
                "step_latency_s": round(time.monotonic() - step_start_time, 3),
                "tool_latency_s": None,
                # budget constraints
                "tokens_in_context": self._tokens_in_context,
                "context_limit": self._get_context_limit(ctx),
                "cost_usage": cost_usage,
                "cost_budget": self.config.cost_budget,
                "latency_usage": latency_usage,
                "latency_budget": self.config.latency_budget,
                # state
                "step": self._step,
                "max_steps": self.config.max_steps,
                "turn": self._turn,
                "misfires": self._misfires,
                "max_misfires": self.config.max_misfires,
                # termination state
                "terminate_state": self._terminate_state,
            }
        )

        raise StepFailed(
            self.config.name, "max steps without accepted final answer", diagnostic=observation.error
        )

    # TODO: we still may 400 if our estimate(s) are off and trim does not actually get under the model's context limit
    #       leave as-is for now, but eventually we should have a backstop that will auto compact if we hit a context limit 400 error
    async def _llm_step(self, ctx: ExecutionContext) -> LLMResponse:
        """Render the visible trajectory (`_render_for_llm`) and invoke the agent's model."""
        # cut the trajectory to guarantee it fits within the model's context limit
        self._trim(ctx)

        # add warning messages to the agent if the context usage exceeds the soft or hard token limits
        self._warn_context_limits(ctx)

        # render the remaining messages to send to the model
        messages = self._render_for_llm()
        resp = await ctx.llm_client.acall(
            messages=messages,
            model=self._resolve_model(ctx),
            temperature=self.config.temperature,
            effort=self._resolve_effort(ctx),
            call_site=self.config.name,
            max_output_tokens=self.config.max_output_tokens,
            timeout_s=self.config.request_timeout_s,
            usage_key=self.agent_id,
        )

        return resp

    async def _run_loop(self, ctx: ExecutionContext) -> Any:
        """Executes the agent's multi-turn loop. Assumes `self._messages` and `self._executor`
        have been initialized. `self._step` counts only turns that successfully executed a tool
        or parsed a final answer block. Failures in execution / parsing advance `self._turn` and
        increment `self._misfires` but not `self._step`, so they don't burn the step budget.

        Loop exits when:
            1. final answer payload is returned
            2. max steps is reached
            3. max misfires is reached
            4. a cost or latency budget is tripped

        Returns the validated payload (case 1.) or hands off to `_terminal_turn()` (cases 2-4.)

        raises StepFailed (via `_terminal_turn()`) if it fails to generate a result within its budget.
        """
        # reset self._step, self._turn, and self._misfires
        cost_usage = latency_usage = None
        self._step = self._turn = self._misfires = 0
        while self._steps_and_misfires_left() and self._cost_budget_left(ctx) and self._latency_budget_left():
            step_start_time = time.monotonic()
            if self._warn_low_steps():
                # add message to agent and emit so it can also be shown in trace viewer
                left = self.config.max_steps - self._step
                warn = (
                    f"Only {left} of {self.config.max_steps} non-error steps remain. You should focus your "
                    f"remaining steps on your most promising lead and avoid wasting time on exploration."
                )
                self._messages.append(Message(role="user", blocks=[TextBlock(warn)]))
                ctx.tracer.emit(id="steps_low_warning", level="warning", kind="lifecycle", step=self._step, turn=self._turn, data={"text": warn, "steps_left": left})

            # generate -> parse -> execute;
            step_output: StepOutput | None = None
            observation: _Observation | None = None
            resp = None
            try:
                # generate a response from the llm
                resp = await self._llm_step(ctx)
            except Exception as e:
                observation = _Observation(error=f"{type(e).__name__}: {e}")

            if resp:
                # add message to context and update self._tokens_in_context
                self._messages.append(Message(role="assistant", blocks=[TextBlock(resp.text)]))

                # TODO: check if this is still true (re: usage provider returning no usage on 200)?
                # usage fields are `int | None` (a provider CAN omit usage even on a 200);
                total_tokens = (resp.input_tokens or 0) + (resp.output_tokens or 0)
                if total_tokens > 0:
                    self._tokens_in_context = total_tokens
                else:
                    # if we don't get usage from the provider, estimate tokens based on raw output
                    self._tokens_in_context += estimate_tokens(resp.text)

                # parse the response
                try:
                    step_output = self._parse(resp.text)

                except ParseError as e:
                    observation = _Observation(error=e.detail)

            tool_latency_s = None
            try:
                # execute tool call if this is not a final answer
                if step_output and not step_output.is_final:
                    # run the step's single tool-call block, offloaded so the event loop stays
                    # free for this agent's sibling branches (see `_execute_code`).
                    assert step_output.code is not None

                    # if we are above the hard context limit, block any non-prune tool call
                    hard_token_max_after_step = self._tokens_in_context + (self.config.max_output_tokens or 0)
                    above_limit = hard_token_max_after_step >= self._get_hard_token_limit(ctx)
                    if above_limit:
                        # raise an exception if there are no tools for reclaiming context (this isn't a final answer)
                        reclaim_names = [t.name for t in self._tools if t.reclaimer]
                        if not reclaim_names:
                            raise Exception("You are above the hard token limit for the model but did not produce a final answer.")
        
                        # regex is an OR across any tool name that reclaims context
                        reclaim_tools_re = re.compile(
                            rf"\b(?:{'|'.join(re.escape(n) for n in reclaim_names)})\(.*\)", re.DOTALL
                        )
                        if not reclaim_tools_re.findall(step_output.code):
                            raise Exception(
                                f"You are above the hard token limit for the model but did not invoke a "
                                f"tool to reduce the context (one of: {reclaim_names}) nor produce a final answer."
                            )

                    # execute the tool call and decrement any newly reclaimed blocks from self._tokens_in_context
                    vis_before = [
                        (b, b.visible) for m in self._messages for b in m.blocks
                    ]
                    observation, tool_latency_s = await self._execute_code(step_output.code)
                    newly_hidden = [
                        b for b, was_visible in vis_before
                        if was_visible and not b.visible
                    ]
                    if newly_hidden:
                        self._tokens_in_context -= _count_tokens(newly_hidden)

            # handle tool selection or code execution errors
            except Exception as e:
                observation = _Observation(error=f"{type(e).__name__}: {e}")

            # NOTE: observation is None iff step produced a final answer
            obs_blocks: list[Block] = []
            if observation is not None:
                # construct the step's observation block; the step either executed a tool call successfully,
                # or it produced an error somewhere between generating a response and executing the tool call;
                # some generations may produce a StepOutput.notice if they produced more than a single code block
                obs_blocks.append(TextBlock(f"Observation (step {self._step}):"))
                if observation.error:
                    obs_blocks.append(TextBlock(f"[error]\n{observation.error}"))
                else:
                    assert observation.output is not None
                    obs_blocks.extend(self._blocks_from_output(ctx, observation.output))
                if step_output and step_output.notice:
                    obs_blocks.append(TextBlock(f"[notice]\n{step_output.notice}"))

                # add observation blocks to inform the agent of its budget usage
                # NOTE: we compute _tokens_in_context before calling _add_budget_observations because the
                #       latter needs to report the number of tokens in the context window up to that point
                self._tokens_in_context += _count_tokens(obs_blocks)
                cost_usage, latency_usage = self._add_budget_observations(ctx, obs_blocks)

                # add the observation to the agent's messages
                self._messages.append(Message(role="user", blocks=obs_blocks))

            # emit an event for the entire step --> (generation + tool call + output / error) or (final result);
            ctx.tracer.emit(
                id="agent_step",
                kind="assistant",
                step=self._step,
                turn=self._turn,
                data={
                    # raw output generated by llm
                    "text": resp.text if resp else None,
                    # StepOutput fields (is_final, result, parsed tool code, notice)
                    # NOTE: should enforce that `step_output.result` is JSON serializable before logging
                    "is_final": step_output.is_final if step_output else None,
                    "result": str(step_output.result) if step_output else None,
                    "code": step_output.code if step_output else None,
                    "notice": step_output.notice if step_output else None,
                    # _Observation 
                    "observation_blocks": [_block_to_jsonable(b) for b in obs_blocks] if observation and observation.output else None,
                    "error": observation.error if observation else None,
                    # latencies
                    "step_latency_s": round(time.monotonic() - step_start_time, 3),
                    "tool_latency_s": tool_latency_s,
                    # budget constraints
                    "tokens_in_context": self._tokens_in_context,
                    "context_limit": self._get_context_limit(ctx),
                    "cost_usage": cost_usage,
                    "cost_budget": self.config.cost_budget,
                    "latency_usage": latency_usage,
                    "latency_budget": self.config.latency_budget,
                    # state
                    "step": self._step,
                    "max_steps": self.config.max_steps,
                    "turn": self._turn,
                    "misfires": self._misfires,
                    "max_misfires": self.config.max_misfires,
                    # termination state
                    "terminate_state": self._terminate_state,
                }
            )

            # if the step produced a final output, validate it and return or add an error observation
            if step_output and step_output.is_final:
                return step_output.result

            # update loop state
            assert observation is not None
            if observation.output:
                self._step += 1
                self._turn = 0
            else:
                self._turn += 1
                self._misfires += 1

        # out of steps or budget: one forced terminal turn that either commits an answer from the
        # existing messages or hands off to the planner with a diagnostic.
        out_of_steps = not self._steps_and_misfires_left()
        over_cost_budget = not self._cost_budget_left(ctx)
        over_latency_budget = not self._latency_budget_left()
        return await self._terminal_turn(ctx, cost_usage, latency_usage, out_of_steps, over_cost_budget, over_latency_budget)

    async def call(self, ctx: ExecutionContext, input: str, *, resume: bool = False, **_) -> Any:
        """Run the multi-turn loop, returning the parsed json final-answer payload.
        `resume=True` appends `input` to the existing trajectory (with a fresh step budget).
        """
        # mark the time at which the agent started executing; do not overwrite the start time
        # if it has already been set (this may happen if an agent uses call() for retries)
        if self._agent_start_time is None:
            self._agent_start_time = time.monotonic()

        # if we're resuming execution, reuse messages, executor state, and _tokens_in_context
        if resume and self._messages:
            self._messages.append(Message(role="user", blocks=[TextBlock(input)]))
            self._tokens_in_context += _count_tokens(self._messages[-1].blocks)

        # otherwise, start a new message history and create a fresh executor
        else:
            self._messages = [
                Message(role="system", blocks=[TextBlock(self._system_prompt)]),
                Message(role="user", blocks=[TextBlock(input)]),
            ]
            self._tokens_in_context = _count_tokens(self._messages[0].blocks + self._messages[1].blocks)
            self._executor = self._build_executor()

        # capture the system prompt + opening question into the event stream
        ctx.tracer.emit(id="system_prompt", kind="system", data={"text": self._system_prompt})
        ctx.tracer.emit(id="question", kind="user", data={"text": input})

        # run the agent loop
        return await self._run_loop(ctx)
