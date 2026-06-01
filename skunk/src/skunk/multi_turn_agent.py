"""MultiTurnAgent — a tool-loop agent that HAS-A `PromptedCall`. Each step: the model
emits one ```python``` block → exec in a LocalPythonExecutor with `self.tools()` bound
→ tool output appended as an observation → repeat until `final_answer(...)` passes
`validate_final_answer`. Raises `StepFailed` after `max_steps` without an accepted answer.

`self._observations` exposes the running observation list to `tools()` closures."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from skunk.errors import StepFailed
from skunk.prompted_call import PromptedCall
from skunk.search_agent.utils import LocalPythonExecutor

if TYPE_CHECKING:
    from skunk.common import HarnessContext


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"```(?:python|py)?\n")


def _has_complete_code_block(acc: str) -> bool:
    """True once `acc` holds a complete ```python block — the `should_stop` signal
    for the agent's one-block-per-step protocol."""
    open_m = _OPEN_FENCE_RE.search(acc)
    return bool(open_m and "```" in acc[open_m.end():])


def final_answer(payload: dict) -> dict:
    """Multi-turn loop terminator. The executor detects it by identity
    (`is_final_answer=True`); the dict flows to `validate_final_answer` and `call()`."""
    return payload


class Tool(ABC):
    """One agent tool. Subclasses set two class attributes — `name` (the binding
    the model calls in code) and `doc` (the markdown `### name(args)` block shown
    in the system prompt; it MAY contain Jinja placeholders like `{{ max_pages }}`
    that the host agent's `template_vars` render) — and implement `__call__` (the
    API). Dependencies are captured in the subclass `__init__`; a `Tool` instance
    is callable, so it binds into the executor exactly like a plain function.
    `final_answer` is NOT a `Tool` — it is the always-injected loop terminator."""

    name: str
    doc: str

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """The tool's runtime behaviour."""


def _trim(messages: list[dict], budget: int) -> list[dict]:
    """Keep system + first user + the most recent messages fitting in `budget` chars;
    drop the middle behind a placeholder so observation history can't bloat unbounded."""
    if sum(len(m["content"]) for m in messages) <= budget:
        return messages
    head = [messages[0], messages[1],
            {"role": "user", "content": "...(earlier steps truncated)..."}]
    remaining = budget - sum(len(m["content"]) for m in head)
    tail: list[dict] = []
    for m in reversed(messages[2:]):
        if remaining - len(m["content"]) < 0:
            break
        tail.append(m)
        remaining -= len(m["content"])
    return head + tail[::-1]


class MultiTurnAgent(ABC):
    """A tool-loop agent that HAS-A `PromptedCall` and a list of `Tool`s. Subclasses
    pass both to `super().__init__` (the base derives `tools()`, injecting
    `final_answer`) and may override `validate_final_answer()`; `call()` is the shared
    loop. The agent path ignores the `PromptedCall`'s effort resolution."""

    max_steps: int = 8
    # Char budget on the message list; oldest middle messages dropped beyond it.
    context_budget_chars: int = 200_000

    def __init__(self, prompt: PromptedCall, tools: list[Tool]) -> None:
        self._prompt = prompt
        self._tools = tools
        self._observations: list[str] = []

    def tools(self) -> dict[str, Callable]:
        """Executor bindings: each `self._tools` entry by `name`, plus the
        always-injected `final_answer` terminator. Override only for non-standard
        wiring."""
        return {t.name: t for t in self._tools} | {"final_answer": final_answer}

    def validate_final_answer(
        self, payload: dict, observations: list[str]
    ) -> str | None:
        """Return None to accept, or a feedback string to reject (loop continues with
        it as an observation). Default: always accept."""
        return None

    def call(self, ctx: "HarnessContext", user: str, **_) -> dict:
        """Run the multi-turn loop, returning the dict from `final_answer`. Extra
        kwargs are ignored (signature compat with single-shot calls)."""
        system_prompt = self._prompt.assemble_system_prompt(ctx)
        executor = LocalPythonExecutor(additional_authorized_imports=[])
        executor.send_tools(self.tools())

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]
        observations: list[str] = []
        self._observations = observations

        ctx.emit(self._prompt.name, "system", content=system_prompt)
        ctx.emit(self._prompt.name, "question", content=user)

        for step in range(self.max_steps):
            text = self._generate(ctx, messages)
            messages.append({"role": "assistant", "content": text})

            blocks = _CODE_BLOCK_RE.findall(text)
            if not blocks:
                obs = f"Observation (step {step + 1}): no ```python``` block in your response."
                messages.append({"role": "user", "content": obs})
                ctx.emit(self._prompt.name, "error", content=obs)
                continue
            code = blocks[0].strip()

            try:
                out = executor(code)
            except Exception as e:
                obs = f"Observation (step {step + 1}): exec failed — {type(e).__name__}: {e}"
                messages.append({"role": "user", "content": obs})
                ctx.emit(self._prompt.name, "error", content=obs)
                continue

            # Skip the [result] echo when stdout already serializes `out.output`
            # (typical when the model wrote `print(tool_call(...))`).
            stdout_s = (out.logs or "").strip()
            result_s = "" if out.output is None else str(out.output).strip()
            obs_parts = []
            if stdout_s:
                obs_parts.append(f"[stdout]\n{stdout_s}")
            if result_s and result_s != stdout_s and result_s not in stdout_s:
                obs_parts.append(f"[result]\n{result_s}")
            obs = f"Observation (step {step + 1}):\n" + ("\n".join(obs_parts) or "[no output]")
            messages.append({"role": "user", "content": obs})
            observations.append(obs)
            ctx.emit(self._prompt.name, "observation", content=obs)

            if out.is_final_answer:
                feedback = self.validate_final_answer(out.output, observations)
                if feedback is None:
                    return out.output  # type: ignore[return-value]
                fb = f"Observation (step {step + 1}, validation): {feedback}"
                messages.append({"role": "user", "content": fb})
                observations.append(fb)
                ctx.emit(self._prompt.name, "validation_failed", content=feedback)

        raise StepFailed(self._prompt.name, "max steps without accepted final_answer")

    def _generate(self, ctx: "HarnessContext", messages: list[dict]) -> str:
        """Stream tokens via the shared `LLMClient`, stopping once a complete
        ```python``` block arrives. Trims to `context_budget_chars` first."""
        messages = _trim(messages, self.context_budget_chars)
        resp = ctx.llm_client.stream(
            system=messages[0]["content"],  # system is always [0]
            messages=messages[1:],
            model=ctx.config.agent_model_id or ctx.config.llm_model,
            should_stop=_has_complete_code_block,
            ctx=ctx,
            call_site=self._prompt.name,
        )
        return resp.text
