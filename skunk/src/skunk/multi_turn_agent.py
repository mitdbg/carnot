from __future__ import annotations

import ast
import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from jinja2 import Environment, StrictUndefined

from skunk.common import B64Image, Effort, ExecutionContext, PendingHumanIntervention
from skunk.errors import ParseError, StepFailed
from skunk.prompted_call import PromptedCall
from skunk.local_python_executor import CodeOutput, LocalPythonExecutor

_ENV = Environment(
    autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined
)


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


def _make_should_stop(max_parallel: int) -> "Callable[[str], bool]":
    """Build the streaming `should_stop` for an agent that may emit up to `max_parallel`
    parallel tool blocks per step. Stop once a ```json``` final-answer block completes
    (it is always emitted on its own) or once `max_parallel` fenced blocks are complete.
    For `max_parallel == 1` this is the original behaviour — stop at the first complete
    block."""

    def should_stop(acc: str) -> bool:
        blocks = _FENCE_RE.findall(acc)
        if any(lang.lower() == "json" for lang, _ in blocks):
            return True
        return len(blocks) >= max_parallel

    return should_stop


def _tool_name(code: str) -> str | None:
    """Best-effort name of the first call in a tool-call block (e.g. `prune` /
    `search_corpus`), for prune-isolation checks and per-call observation headers.
    Returns None when the body does not parse or contains no call."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                return fn.id
            if isinstance(fn, ast.Attribute):
                return fn.attr
    return None


@dataclass
class _StepOutput:
    """A successfully parsed model response. `is_final` ⇒ a ```json``` final answer in
    `result` (plain `Any`, so a literal `null` payload is represented as-is). Otherwise
    `codes` holds the (1..N) ```python``` tool-call bodies to execute. `mixed_final` flags
    a ```json``` block that appeared alongside tool calls (ignored, with a warning, since
    the final answer must be emitted on its own). `raw` carries the verbatim model text so
    the caller can append it to message history."""

    codes: list[str] = field(default_factory=list)
    result: Any = None
    is_final: bool = False
    mixed_final: bool = False
    raw: str = ""


@dataclass
class _CallResult:
    """Outcome of executing one tool-call block in a (possibly parallel) batch. Exactly
    one of `output` / `error` is populated."""

    output: CodeOutput | None = None
    error: str | None = None


def _parse_step(text: str, _: ExecutionContext) -> _StepOutput:
    """Parse every fenced block and raise `ParseError` on bad format. Never executes.
    A lone ```json``` block is the final answer; otherwise all ```python``` blocks are
    returned as `codes` (capping / prune-isolation are applied by the loop, which knows
    the agent's `max_parallel_tool_calls`)."""
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
    mixed_final = any(lang.lower() == "json" for lang, _ in blocks)
    return _StepOutput(codes=codes, mixed_final=mixed_final, raw=text)


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

    _SYSTEM_TEMPLATE = """\
{{ briefing }}

## Tools (already imported)

{{ tools_doc }}

## How to act

You act ONLY by emitting fenced code blocks. **Every reply MUST contain at least one
fenced block.** A reply that is only reasoning or prose, with no fenced block, is not a
valid action: it changes nothing and wastes one of your limited steps. Do any brief
thinking *before* the block, then always finish your reply with the block(s).

A tool call is a ```python``` block whose body is a single call, opened by a line that is
exactly three backticks then `python`, and closed by a line of three backticks:

```python
some_tool(arg="value", count=10)
```

{% if max_parallel_tool_calls > 1 -%}
You have ≤{{ max_steps }} steps. On each step, emit ONE of these:
  - 1 to {{ max_parallel_tool_calls }} ```python``` blocks, each a single tool call. Emit
    more than one to run them **in parallel** — their outputs come back together as one
    batched observation; use this for independent calls (e.g. different sub-topics or time
    periods). Do not exceed {{ max_parallel_tool_calls }} blocks (extras are dropped), and
    since parallel blocks share no state, no block may depend on another's result.
  - exactly one ```json``` block — your final answer. Emit it once, when you are ready to
    finish, in a step **by itself** (never alongside ```python``` blocks). It is parsed as
    data, not executed, so write plain JSON literals (no Python, no variables, no trailing
    commas).
{%- else -%}
You have ≤{{ max_steps }} steps. On each step, emit exactly ONE fenced block:
  - a ```python``` block containing a single tool call — it is executed and its output
    becomes your next observation; or
  - a ```json``` block containing your final answer — emit it once, when you are ready to
    finish. It is parsed as data, not executed, so write plain JSON literals (no Python, no
    variables, no trailing commas).
{%- endif %}

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

    # Max tool-call blocks the agent may emit (and run in parallel) per step. 1 keeps the
    # classic one-call-per-step behaviour. Subclasses pass a config-derived value; stateful
    # compute agents (e.g. the task solver) keep this at 1 so cross-step interpreter state
    # survives in the single persistent sandbox.
    max_parallel_tool_calls: int = 1

    def __init__(
        self,
        tools: list[Tool],
        *,
        max_steps: int | None = None,
        max_parallel_tool_calls: int = 1,
        system_prompt_override: str | None = None,
        generation_backend: GenerationBackend | None = None,
        sampling_params: dict | None = None,
        capture_logprobs: bool = False,
    ) -> None:
        self._tools = tools
        self.max_steps = max_steps
        self.max_parallel_tool_calls = max(1, max_parallel_tool_calls)
        # Streaming stop: allow up to N complete blocks (or one json final answer).
        self._should_stop = _make_should_stop(self.max_parallel_tool_calls)
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
                max_parallel_tool_calls=self.max_parallel_tool_calls,
                final_answer_doc=self.final_answer_doc,
            )
        self._prompt: PromptedCall[_StepOutput] = PromptedCall(
            name=self.name,
            system_prompt=system_prompt,
            default_effort=self.default_effort,
            parse=_parse_step,
            max_parse_retries=self.max_recover_retries,
        )

    def _build_executor(self) -> LocalPythonExecutor:
        """A fresh sandbox with the agent's tools bound. The loop keeps one persistent
        executor for single-call steps (cross-step state survives); parallel batches use
        one fresh executor per block so concurrent calls cannot race on interpreter state."""
        executor = LocalPythonExecutor(
            additional_authorized_imports=self.authorized_imports
        )
        executor.send_tools({t.name: t for t in self._tools})
        return executor

    def _prepare_codes(self, step_out: _StepOutput) -> tuple[list[str], list[str]]:
        """Apply the over-cap and prune-isolation rules to a step's tool-call blocks.
        Returns `(codes_to_run, notices)` — notices are appended to the observation so the
        model can correct course (they never abort the step)."""
        codes = list(step_out.codes)
        notices: list[str] = []
        if step_out.mixed_final:
            notices.append(
                "Your ```json``` final-answer block was ignored — emit the final answer "
                "in a step by itself, with no tool calls."
            )
        if len(codes) > self.max_parallel_tool_calls:
            notices.append(
                f"You emitted {len(codes)} tool-call blocks; only the first "
                f"{self.max_parallel_tool_calls} were executed. Keep each step to at most "
                f"{self.max_parallel_tool_calls} ```python``` block(s)."
            )
            codes = codes[: self.max_parallel_tool_calls]
        # `prune(...)` mutates state the other tools read; only safe to run on its own.
        if len(codes) > 1:
            kept = [c for c in codes if _tool_name(c) != "prune"]
            if len(kept) != len(codes):
                notices.append(
                    "`prune(...)` was ignored — it mutates shared search state and must be "
                    "issued in a step by itself, not combined with other tool calls."
                )
                codes = kept
        # `request_human(...)` pauses the step for a human; pairing it with speculative tool
        # calls is confusing, so isolate it the same way as `prune`.
        if len(codes) > 1:
            kept = [c for c in codes if _tool_name(c) != "request_human"]
            if len(kept) != len(codes):
                notices.append(
                    "`request_human(...)` was ignored — it pauses the step for a human and "
                    "must be issued in a step by itself, not combined with other tool calls."
                )
                codes = kept
        return codes, notices

    @staticmethod
    def _run_block(executor: LocalPythonExecutor, code: str) -> _CallResult:
        """Run one tool-call block, capturing the result or the error (never raising)."""
        try:
            return _CallResult(output=executor(code))
        except Exception as e:
            return _CallResult(error=f"{type(e).__name__}: {e}")

    async def _execute_codes(self, codes: list[str]) -> list[_CallResult]:
        """Execute a step's tool-call blocks, offloaded from the event loop so this
        question's sibling branches keep progressing while the blocking tool I/O runs.

        A single block reuses the persistent executor (cross-step state survives). A
        parallel batch runs each block in its own fresh executor — concurrent calls can't
        then race on interpreter state — and `gather` + `to_thread` keeps the loop free."""
        if not codes:
            return []
        if len(codes) == 1:
            return [await asyncio.to_thread(self._run_block, self._executor, codes[0])]
        return list(
            await asyncio.gather(
                *(
                    asyncio.to_thread(self._run_block, self._build_executor(), c)
                    for c in codes
                )
            )
        )

    async def _resolve_human_interventions(
        self, ctx: ExecutionContext, results: list[_CallResult]
    ) -> list[_CallResult]:
        """Await any `PendingHumanIntervention` a tool returned (the `request_human` tool),
        replacing it in-place with the human's resolved response before the observation is
        rendered. Pending awaits run concurrently so a parallel batch is not serialized on
        each human round-trip."""

        async def _resolve(res: _CallResult) -> _CallResult:
            if res.output is not None and isinstance(
                res.output.output, PendingHumanIntervention
            ):
                ctx.emit("human_intervention_waiting", kind="note")
                res.output.output = await res.output.output.response
                ctx.emit("human_intervention_resolved", kind="note")
            return res

        return list(await asyncio.gather(*(_resolve(r) for r in results)))

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

    async def call(self, ctx: ExecutionContext, user: str, **_) -> Any:
        """Run the multi-turn loop, returning the parsed json final-answer payload. Extra
        kwargs are ignored (signature compat with single-shot calls)."""
        # Persistent sandbox for single-call steps — cross-step interpreter state survives
        # here. Parallel batches use one fresh executor per block (see `_execute_codes`).
        # The final answer is parsed outside the sandbox, so it is NOT bound here.
        self._executor = self._build_executor()

        # Full block trajectory (no system message; call() assembles it each turn).
        # `_render_for_llm()` produces the redacted, flattened view sent to the model.
        self.messages = [{"role": "user", "blocks": [TextBlock(user)]}]
        observations: list[str] = []

        # Capture the system prompt + opening question into the event stream so the
        # trace viewer can show them (the console / `.log` keep only the one-liners —
        # the full text rides in `data`). The system prompt is static for the call.
        system_prompt = self._prompt._assemble_system_prompt(ctx)
        ctx.emit(
            f"system_prompt chars={len(system_prompt)}",
            kind="system",
            data={"text": system_prompt},
        )
        ctx.emit(f"question {user!r}", kind="user", data={"text": user})

        # `step` counts only turns that progressed (observation or final answer); `turn`
        # numbers every attempt, including misfired re-prompts.
        step = turn = 0
        warned = False
        while self.max_steps is None or step < self.max_steps:
            step += 1
            if (
                self.warn_steps_remaining is not None
                and not warned
                and self.max_steps is not None
                and self.max_steps - step + 1 <= self.warn_steps_remaining
            ):
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
            results: list[_CallResult] | None = None
            executed_codes: list[str] = []
            notices: list[str] = []
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
                    # Apply the cap + prune-isolation rules, then run the (1..N) blocks.
                    # Parallel batches are offloaded so the event loop stays free for
                    # this question's sibling branches (see `_execute_codes`).
                    executed_codes, notices = self._prepare_codes(step_out)
                    for c in executed_codes:
                        ctx.emit(f"tool_code {c!r}")
                    results = await self._execute_codes(executed_codes)
                    # A `request_human` call returns a `PendingHumanIntervention`; await the
                    # human's response and splice it back in before the observation renders.
                    # `request_human` is isolation-only (see `_prepare_codes`), so at most one
                    # runs per step, but resolve concurrently to stay general.
                    results = await self._resolve_human_interventions(ctx, results)
                done = step_out
            except ParseError as e:
                obs = f"Observation (step {turn}): {e.detail}"
                self.messages.append({"role": "user", "blocks": [TextBlock(obs)]})
                ctx.emit(f"error {obs!r}")
            except Exception as e:  # LLM step / batch machinery failed
                obs = (
                    f"Observation (step {turn}): exec failed — {type(e).__name__}: {e}"
                )
                self.messages.append({"role": "user", "blocks": [TextBlock(obs)]})
                ctx.emit(f"error {obs!r}")

            if done is None:
                continue  # step spent on misfires; advance

            if done.is_final:  # final-answer block
                payload = done.result
                feedback = self.validate_final_answer(payload, observations)
                if feedback is None:
                    return payload
                fb = f"Observation (step {turn}, validation): {feedback}"
                self.messages.append({"role": "user", "blocks": [TextBlock(fb)]})
                observations.append(fb)
                ctx.emit(f"validation_failed {feedback!r}")
                continue

            assert results is not None  # non-final step ⇒ a (possibly empty) batch ran
            # One batched observation for the whole step. Each call gets a header when
            # >1 ran; successes render via `_blocks_from_output` (subclasses may emit
            # redactable ChunkBlocks), failures as `[error]` blocks. Cap / prune-isolation
            # notices are appended so the model can correct course next step.
            obs_blocks: list[Block] = [TextBlock(f"Observation (step {turn}):")]
            n = len(results)
            for i, (code, res) in enumerate(zip(executed_codes, results)):
                if n > 1:
                    obs_blocks.append(
                        TextBlock(
                            f"--- Tool call {i + 1}/{n} ({_tool_name(code) or '<unknown>'}(...)) ---"
                        )
                    )
                if res.error is not None:
                    obs_blocks.append(TextBlock(f"[error]\n{res.error}"))
                else:
                    assert res.output is not None
                    obs_blocks.extend(self._blocks_from_output(res.output))
            for notice in notices:
                obs_blocks.append(TextBlock(f"[notice] {notice}"))
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
            diagnostic = (
                step_out.raw.strip()
            )  # default: the whole reply is the hand-off note
            if step_out.is_final:  # json block
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
        """Render the visible trajectory (`_render_for_llm`), collapse stale observations,
        trim to `context_budget_chars`, then route through `PromptedCall.call()` — stopping
        once the step's block(s) complete (`self._should_stop`). `extra` appends transient
        messages (e.g. the terminal-turn prompt) that are deliberately NOT stored in
        `self.messages`."""
        messages = self._render_for_llm()
        # Collapse all but the most recent `visible_observations` tool results to a
        # placeholder (None = keep all), keeping the question + every assistant turn.
        if self.visible_observations is not None:
            user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"]
            # user_idxs[0] is the initial question — always kept; the rest are results.
            keep = set(user_idxs[1:][-self.visible_observations :]) | {user_idxs[0]}
            messages = [
                m
                if (m["role"] != "user" or i in keep)
                else {"role": "user", "content": "[earlier tool result hidden]"}
                for i, m in enumerate(messages)
            ]
        trimmed = _trim(messages, self.context_budget_chars)
        if extra:
            trimmed = trimmed + extra
        if self._backend is None:
            self._last_logprobs = None
            return await self._prompt.call(
                ctx, messages=trimmed, should_stop=self._should_stop,
                max_output_tokens=self.max_output_tokens,
                timeout_s=self.request_timeout_s,
            )
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
