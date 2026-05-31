"""MultiTurnAgent — a PromptedCall whose `call()` runs a tool loop.

Each step: model emits one ```python``` block → exec in a
LocalPythonExecutor with `self.tools()` bound → tool output appended
as a user observation → repeat until the model calls `final_answer(...)`
and `validate_final_answer` (default: accept) passes. Returns the
dict that `final_answer` produced. Raises `StepFailed` after
`max_steps` without an accepted answer.

`self._observations` exposes the running observation list to `tools()`
closures (used by `LookupAgent`'s grounder).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from google.genai import types as genai_types

from skunk.common import _make_genai_client
from skunk.errors import StepFailed
from skunk.prompted_call import PromptedCall
from skunk.search_agent.utils import LocalPythonExecutor

if TYPE_CHECKING:
    from skunk.models import HarnessContext


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"```(?:python|py)?\n")


def final_answer(payload: dict) -> dict:
    """Multi-turn loop terminator. The executor detects this tool by
    identity (`is_final_answer=True`); the dict is passed through to
    the agent's `validate_final_answer` and returned from `call()`.
    Each subclass declares the expected dict shape in its system prompt."""
    return payload


class MultiTurnAgent(PromptedCall, ABC):
    max_steps: int = 8

    @abstractmethod
    def tools(self) -> dict[str, Callable]:
        """Tools bound into the executor. MUST include `final_answer`."""

    def validate_final_answer(
        self, payload: dict, observations: list[str]
    ) -> str | None:
        """Return None to accept; return a feedback string to reject
        (loop continues with that feedback as an observation). Default:
        always accept."""
        return None

    def call(self, ctx: "HarnessContext", user: str, **_) -> dict:
        """Run the multi-turn loop. Returns the dict from `final_answer`.
        Extra kwargs are ignored (signature compat with PromptedCall)."""
        system_prompt = self.assemble_system_prompt(ctx)
        executor = LocalPythonExecutor(additional_authorized_imports=[])
        executor.send_tools(self.tools())

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]
        observations: list[str] = []
        self._observations = observations

        ctx.emit(self.name, "system", content=system_prompt)
        ctx.emit(self.name, "question", content=user)

        for step in range(self.max_steps):
            text = self._generate(ctx, messages)
            messages.append({"role": "assistant", "content": text})
            ctx.emit(self.name, "assistant", content=text)

            blocks = _CODE_BLOCK_RE.findall(text)
            if not blocks:
                obs = f"Observation (step {step + 1}): no ```python``` block in your response."
                messages.append({"role": "user", "content": obs})
                ctx.emit(self.name, "error", content=obs)
                continue
            code = blocks[0].strip()

            try:
                out = executor(code)
            except Exception as e:
                obs = f"Observation (step {step + 1}): exec failed — {type(e).__name__}: {e}"
                messages.append({"role": "user", "content": obs})
                ctx.emit(self.name, "error", content=obs)
                continue

            obs_parts = []
            if out.logs:
                obs_parts.append(f"[stdout]\n{out.logs}")
            if out.output is not None:
                obs_parts.append(f"[result]\n{out.output}")
            obs = f"Observation (step {step + 1}):\n" + ("\n".join(obs_parts) or "[no output]")
            messages.append({"role": "user", "content": obs})
            observations.append(obs)
            ctx.emit(self.name, "observation", content=obs)

            if out.is_final_answer:
                feedback = self.validate_final_answer(out.output, observations)
                if feedback is None:
                    return out.output  # type: ignore[return-value]
                fb = f"Observation (step {step + 1}, validation): {feedback}"
                messages.append({"role": "user", "content": fb})
                observations.append(fb)
                ctx.emit(self.name, "validation_failed", content=feedback)

        raise StepFailed(self.name, "max steps without accepted final_answer")

    def _generate(self, ctx: "HarnessContext", messages: list[dict]) -> str:
        """Stream tokens; stop once a complete ```python``` block has
        been received. Subclasses can override to add behaviour like
        token-budget message trimming."""
        client = _make_genai_client()
        model_id = (ctx.config.agent_model_id or ctx.config.llm_model).removeprefix("google/")
        system_instruction = messages[0]["content"]  # system is always [0]
        contents = [
            genai_types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[genai_types.Part.from_text(text=m["content"])],
            )
            for m in messages[1:]
        ]
        stream = client.models.generate_content_stream(
            model=model_id, contents=contents,
            config=genai_types.GenerateContentConfig(system_instruction=system_instruction),
        )
        accumulated = ""
        for chunk in stream:
            accumulated += chunk.text or ""
            open_m = _OPEN_FENCE_RE.search(accumulated)
            if open_m and "```" in accumulated[open_m.end():]:
                tail_close = accumulated[open_m.end():].find("```")
                accumulated = accumulated[: open_m.end() + tail_close + 3]
                break
        try:  # noqa: SIM105
            stream.close()  # type: ignore[union-attr]
        except Exception:
            pass
        return accumulated
