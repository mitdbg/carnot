from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from jinja2 import Environment, StrictUndefined

from skunk.common import Effort, ExecutionContext
from skunk.errors import ParseError, StepFailed
from skunk.prompted_call import PromptedCall
from skunk.local_python_executor import CodeOutput, LocalPythonExecutor

_ENV = Environment(autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined)


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

    def __init__(self, tools: list[Tool], *, max_steps: int | None = None) -> None:
        self._tools = tools
        self.max_steps = max_steps
        template = self._SYSTEM_TEMPLATE.replace(
            "{{ tools_doc }}", "\n\n".join(t.doc for t in tools))
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

    async def call(self, ctx: ExecutionContext, user: str, **_) -> Any:
        """Run the multi-turn loop, returning the parsed json final-answer payload. Extra
        kwargs are ignored (signature compat with single-shot calls)."""
        executor = LocalPythonExecutor(additional_authorized_imports=[])
        # The final answer is parsed outside the sandbox, so it is NOT bound here.
        executor.send_tools({t.name: t for t in self._tools})

        # No system message in the list; call() assembles it internally each turn.
        messages: list[dict] = [{"role": "user", "content": user}]
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
                messages.append({"role": "user", "content": warn})
                ctx.emit(f"steps_low_warning left={left}")
            # Generate → parse (retried by PromptedCall on format errors) → execute.
            # `done` is set on success; errors append an observation and advance the step.
            done: _StepOutput | None = None
            out: CodeOutput | None = None
            turn += 1
            try:
                step_out = await self._llm_step(ctx, messages)
                messages.append({"role": "assistant", "content": step_out.raw})
                if step_out.code is not None:
                    # Inline (not to_thread): tool code runs on this question's worker
                    # thread; blocking here only affects sibling branches of the question.
                    out = executor(step_out.code)
                done = step_out
            except ParseError as e:
                obs = f"Observation (step {turn}): {e.detail}"
                messages.append({"role": "user", "content": obs})
                ctx.emit(f"error {obs!r}")
            except Exception as e:  # tool code raised
                obs = f"Observation (step {turn}): exec failed — {type(e).__name__}: {e}"
                messages.append({"role": "user", "content": obs})
                ctx.emit(f"error {obs!r}")

            if done is None:
                continue  # step spent on misfires; advance

            if done.code is None:  # final-answer block
                payload = done.result
                feedback = self.validate_final_answer(payload, observations)
                if feedback is None:
                    return payload
                fb = f"Observation (step {turn}, validation): {feedback}"
                messages.append({"role": "user", "content": fb})
                observations.append(fb)
                ctx.emit(f"validation_failed {feedback!r}")
                continue

            assert out is not None  # a non-final step that ended ⇒ tool exec succeeded
            # Render the tool output; skip the [result] echo when stdout already serializes it
            # (model wrote `print(tool_call(...))`).
            stdout_s = (out.logs or "").strip()
            result_s = "" if out.output is None else str(out.output).strip()
            parts = []
            if stdout_s:
                parts.append(f"[stdout]\n{stdout_s}")
            if result_s and result_s != stdout_s and result_s not in stdout_s:
                parts.append(f"[result]\n{result_s}")
            obs = f"Observation (step {turn}):\n" + ("\n".join(parts) or "[no output]")
            messages.append({"role": "user", "content": obs})
            observations.append(obs)
            ctx.emit(f"observation {obs!r}")

        # Out of steps: one forced terminal turn that either commits an answer from the
        # existing observations or hands off to the planner with a diagnostic.
        return await self._terminal_turn(ctx, messages, observations)

    _TERMINAL_PROMPT = (
        "You are out of steps. Do NOT call any tool now — emit exactly ONE ```json``` "
        "block, either:\n"
        "  • COMMIT: if a value already in your observations answers the request, your "
        "best final answer in the required format; or\n"
        '  • NO RESULT: an envelope {"error": "<note>"} with a 2-4 sentence note describing which tools/series you tried, any '
        "candidate values you found, and what blocked you."
    )

    async def _terminal_turn(
        self, ctx: ExecutionContext, messages: list[dict], observations: list[str]
    ) -> Any:
        diagnostic = ""
        try:
            msgs = messages + [{"role": "user", "content": self._TERMINAL_PROMPT}]
            step_out = await self._llm_step(ctx, msgs)
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

    async def _llm_step(self, ctx: ExecutionContext, messages: list[dict]) -> _StepOutput:
        """Route through `PromptedCall.call()`, stopping at the first complete fenced
        block. Collapses stale observations, then trims to `context_budget_chars`."""
        # Collapse all but the most recent `visible_observations` tool results to a
        # placeholder (None = keep all), keeping the question + every assistant turn; the
        # raw `messages` the caller holds stay intact for the terminal-turn diagnostic.
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
        return await self._prompt.call(ctx, messages=trimmed, should_stop=_has_complete_block)
