"""Tinker generation backend for SearchAgent rollouts.

Rollouts must sample from the *same* engine we fine-tune with Tinker so the
per-token logprobs we capture are exactly ``pi_old`` -- the behaviour-policy
denominator in the GRPO / CISPO importance ratio. Sampling from OpenRouter
instead (an fp8 build, behind a single pinned provider) gave us both a
single-point-of-failure and a train/inference precision mismatch; going direct
to Tinker removes both.

This module exposes a synchronous ``GenerationBackend`` (see
``search_agent.GenerationBackend``) backed by a Tinker ``SamplingClient``. The
SearchAgent loop, tool execution, pruning, and logprob persistence are all
backend-agnostic; only ``_generate`` dispatches here.

The ``tinker`` and ``tinker_cookbook`` imports are deferred to
``build_tinker_backend`` / ``TinkerBackend.generate`` so importing SearchAgent
for the OpenRouter-backed roles (datagen synthesis, quality filter) never
requires Tinker to be installed.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import tinker

# Defaults mirror the model we fine-tune with Tinker. The OpenRouter id
# ("qwen/qwen3.6-35b-a3b"), the Tinker base_model ("Qwen/Qwen3.6-35B-A3B"), and
# the renderer family ("qwen3") are three distinct identifiers -- a wrong
# mapping silently yields the wrong tokenizer and garbage logprobs.
DEFAULT_TINKER_BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_TINKER_RENDERER = "qwen3"
# Per-turn generation cap. Tinker does not stream, so a turn ends at the
# renderer's turn-terminator token (e.g. <|im_end|>) or this cap, whichever
# comes first. This is a runaway backstop, not the primary stop condition.
DEFAULT_ROLLOUT_MAX_TOKENS = 4096

_INSTALL_HINT = (
    "The Tinker rollout backend requires the `tinker` and `tinker_cookbook` "
    "packages. Install them (e.g. `pip install tinker tinker_cookbook`) and set "
    "the TINKER_API_KEY environment variable."
)


@dataclass
class TinkerUsage:
    """Cumulative Tinker sampling usage across a run (for cost tracking).

    ``prefill_tokens`` is the total prompt tokens fed across all ``sample``
    calls (Tinker bills these at the prefill rate); ``sample_tokens`` is the
    total generated tokens (billed at the sample rate). In a multi-turn agent
    each turn re-sends the growing prompt, so prefill accumulates per call --
    matching Tinker's per-call billing.
    """

    prefill_tokens: int = 0
    sample_tokens: int = 0
    n_calls: int = 0


class TinkerBackend:
    """Synchronous generation backend that samples from a Tinker model.

    One instance holds a single ``SamplingClient`` (plus its tokenizer and
    renderer) and is shared across every rollout ``SearchAgent`` in a run, so
    the underlying client is created once and concurrent ``sample(...)`` calls
    (issued from the rollout thread pool) can be batched server-side.
    """

    def __init__(
        self,
        sampling_client: Any,
        tokenizer: Any,
        renderer: Any,
        base_model: str,
        max_tokens: int = DEFAULT_ROLLOUT_MAX_TOKENS,
        context_window: int = 65_536,
    ) -> None:
        self.sampling_client = sampling_client
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.base_model = base_model
        self.max_tokens = max_tokens
        # Tinker rejects any request where prompt_tokens + max_tokens exceeds the
        # model's context window. We clamp max_tokens to the room left by the
        # prompt (and raise if the prompt alone won't fit) so a slightly-too-long
        # prompt degrades to a shorter generation instead of a hard 400.
        self.context_window = context_window
        # Cumulative sampling usage for cost tracking. Rollouts share one backend
        # across threads, so guard the counters with a lock.
        self.usage = TinkerUsage()
        self._usage_lock = threading.Lock()

    def on_failure(self) -> bool:
        """No tier/provider state to downgrade on Tinker; never downgrades."""
        return False

    def count_tokens(self, text: str) -> int:
        """Exact token count via the model's tokenizer.

        Lets the agent replace its cheap chars/token heuristic with an exact
        count near the context limit (dense numeric corpora tokenize at far
        fewer chars/token than the heuristic assumes).
        """
        return len(self.tokenizer.encode(text))

    def generate(
        self,
        rendered_messages: list[dict],
        *,
        sampling_params: dict | None,
        capture_logprobs: bool,
    ) -> tuple[str, dict | None]:
        """Sample one assistant turn from Tinker.

        ``rendered_messages`` is the SearchAgent's redacted ``{role, content}``
        view (``Message`` is a ``TypedDict`` so these dicts are accepted as-is).
        Returns ``(assistant_text, logprob_data)`` where ``logprob_data`` matches
        the schema ``SearchAgent.messages_to_jsonable`` persists, with exact
        ``token_ids`` + aligned ``token_logprobs`` (= ``log pi_old``).
        """
        import tinker  # deferred; see module docstring

        prompt = self.renderer.build_generation_prompt(rendered_messages)
        stop = self.renderer.get_stop_sequences()

        # Reserve room for generation: prompt_tokens + max_tokens must stay within
        # the context window or Tinker returns a 400. Clamp max_tokens to the
        # remaining budget; if the prompt alone overflows, fail clearly (the
        # agent should have pruned -- this is the backend-level safety net).
        prompt_len = prompt.length  # ModelInput.length is a property (int), not a method
        room = self.context_window - prompt_len
        if room <= 0:
            raise RuntimeError(
                f"prompt ({prompt_len} tokens) does not fit the {self.context_window}-token "
                f"context window for {self.base_model!r}; the agent must prune before generating"
            )
        effective_max_tokens = min(self.max_tokens, room)

        sp_kwargs: dict[str, Any] = {"max_tokens": effective_max_tokens, "stop": stop}
        if sampling_params:
            if "temperature" in sampling_params:
                sp_kwargs["temperature"] = sampling_params["temperature"]
            if "top_p" in sampling_params:
                sp_kwargs["top_p"] = sampling_params["top_p"]
            # OpenRouter uses top_k=0 to mean "disabled"; Tinker uses -1. Only
            # forward a positive top_k; otherwise leave Tinker's default.
            top_k = sampling_params.get("top_k")
            if top_k and top_k > 0:
                sp_kwargs["top_k"] = top_k
        sp = tinker.SamplingParams(**sp_kwargs)

        # `.sample(...)` returns a concurrent Future; `.result()` blocks this
        # thread (the rollout thread pool provides the concurrency).
        resp = self.sampling_client.sample(
            prompt=prompt, num_samples=1, sampling_params=sp
        ).result()

        sequences = getattr(resp, "sequences", None) or []
        if not sequences:
            raise RuntimeError(
                f"Tinker returned no sequences for base_model {self.base_model!r}"
            )
        seq = sequences[0]
        token_ids = list(seq.tokens)
        if not token_ids:
            raise RuntimeError(
                f"Tinker returned an empty sequence for base_model {self.base_model!r}"
            )
        token_logprobs = list(seq.logprobs)

        # Record usage for cost tracking (prefill = prompt tokens billed this
        # call, sample = generated tokens).
        with self._usage_lock:
            self.usage.prefill_tokens += prompt_len
            self.usage.sample_tokens += len(token_ids)
            self.usage.n_calls += 1

        # Raw assistant text (control tokens like <|im_end|> dropped); code-block
        # extraction downstream only cares about the ```python ... ``` fences.
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        logprob_data: dict | None = None
        if capture_logprobs:
            stop_reason = getattr(seq, "stop_reason", "stop")
            logprob_data = {
                "token_ids": token_ids,
                "tokens": [self.tokenizer.decode([t]) for t in token_ids],
                "token_logprobs": token_logprobs,
                # Tinker does not expose top-k alternatives; downstream only uses
                # top_logprobs for entropy diagnostics, never for pi_old.
                "top_logprobs": [],
                "finish_reason": stop_reason if isinstance(stop_reason, str) else str(stop_reason),
                "system_fingerprint": self.base_model,
            }
        return text, logprob_data


def build_tinker_backend(
    base_model: str = DEFAULT_TINKER_BASE_MODEL,
    renderer_name: str = DEFAULT_TINKER_RENDERER,
    max_tokens: int = DEFAULT_ROLLOUT_MAX_TOKENS,
    context_window: int = 65_536,
) -> TinkerBackend:
    """Construct a shared ``TinkerBackend`` once per harness run.

    Builds one ``ServiceClient`` -> ``SamplingClient`` for ``base_model``, pulls
    the matching tokenizer off the sampling client, and pairs it with the named
    renderer. Raises a clear error if Tinker isn't installed or ``TINKER_API_KEY``
    is unset.
    """
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError(f"TINKER_API_KEY is not set. {_INSTALL_HINT}")
    try:
        import tinker
        from tinker_cookbook import renderers
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError(_INSTALL_HINT) from e

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=base_model)
    tokenizer = sampling_client.get_tokenizer()
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    return TinkerBackend(
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        renderer=renderer,
        base_model=base_model,
        max_tokens=max_tokens,
        context_window=context_window,
    )
