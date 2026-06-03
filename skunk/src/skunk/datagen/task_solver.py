"""TaskSolver + nugget-completion evaluation for rollout post-processing.

For every SearchAgent rollout we run two extra LLM passes:

1. **TaskSolver** -- a plain LLM (no tools) which produces a nuggetized
   final answer given the question and the full text of the documents the
   rollout returned in its final answer. This isolates "did the agent
   retrieve the right context?" from "can a competent reader extract the
   answer from that context?".

2. **NuggetCompletion** -- a judge that labels each gold nugget as
   ``support`` / ``partial_support`` / ``not_support`` against the
   TaskSolver's answer (KARL paper, Appendix D1, Figure 31). The numeric
   score is then either:

       strict      score = #support / #nuggets                (default)
       non-strict  score = (#support + 0.5 * #partial) / #nuggets

The output of these two passes feeds both
``RolloutRecord.eval_score`` (which can drive global binarization when
``binarization_mode="eval-score"``) and the per-attempt evidence shown to
the QualityFilter agent.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml
from jinja2 import Template

from skunk.logging.tracer import Tracer
from skunk.retrieve.search_agent import SearchAgent
from skunk.retrieve.search_tools import (
    TASK_SOLVER_FINAL_ANSWER_TAG,
    task_solver_final_answer,
)
from skunk.utils.local_python_executor import LocalPythonExecutor

if TYPE_CHECKING:
    from openrouter import OpenRouter

# We reuse the same character/token estimate as the SearchAgent so the
# "does this fit?" check here is consistent with the agent's own pruning
# behaviour.
_CHARS_PER_TOKEN_ESTIMATE = 4

# Headroom (in characters) we leave inside the model context for the
# question + system prompt + the model's own generated answer. Anything
# larger than (context_chars - this) triggers round-robin halving of the
# longest document. Generous on purpose: TaskSolver output is short.
_TASK_SOLVER_HEADROOM_CHARS = 8_000
_QUESTION_HEADROOM_CHARS = 1_000

# JSON object regex; matches the smallest balanced-looking ``{...}`` block
# anywhere in the model output. The judge is asked to emit pure JSON but we
# tolerate a code fence or short preamble.
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


_PROMPTS_FILE = pathlib.Path(__file__).parent / "prompts.yaml"
with _PROMPTS_FILE.open() as _f:
    _PROMPTS = yaml.safe_load(_f)

TASK_SOLVER_SYSTEM_PROMPT: str = _PROMPTS["task_solver_system_prompt"]
TASK_SOLVER_USER_PROMPT: str = _PROMPTS["task_solver_user_prompt"]
NUGGET_COMPLETION_PROMPT: str = _PROMPTS["nugget_completion_prompt"]


# ---------------------------------------------------------------------------
# Config + result types
# ---------------------------------------------------------------------------


@dataclass
class TaskSolverConfig:
    """Config for the TaskSolver + NuggetCompletion passes.

    A single ``OpenRouter`` client is shared across both passes; pass two
    distinct model ids if you want a smaller solver and a larger judge.
    """

    or_client: OpenRouter
    task_solver_model_id: str = "google/gemini-3.5-flash"
    nugget_judge_model_id: str = "google/gemini-3.5-flash"
    # Approx input context window for the task-solver model (tokens). When
    # the rendered prompt exceeds (model_context_window - headroom) tokens
    # we round-robin halve the longest document until it fits.
    model_context_window: int = 1_000_000
    # Max number of agent steps (code-execution turns) for the TaskSolver.
    task_solver_max_steps: int = 30
    # Strict scoring (no partial credit) is the default.
    strict_nugget_scoring: bool = True
    # OpenRouter service tier hint applied to every TaskSolver agent turn
    # and every NuggetCompletion judge call. ``None`` disables the hint.
    service_tier: str | None = "flex"


# ---------------------------------------------------------------------------
# TaskSolverAgent
# ---------------------------------------------------------------------------


class TaskSolverAgent(SearchAgent):
    """SearchAgent subclass with only Python code execution.

    The corpus tools (``search_corpus``, ``grep_corpus``, ``read_document``,
    ``prune``) are stripped from the executor.  The only callable exposed to
    the model is ``final_answer(answer)``.  Document context is provided
    directly in the initial user message rather than fetched at runtime.

    The Python interpreter supports common scientific libraries so the agent
    can compute statistics, regressions, and other numerical results from
    values found in those documents.
    """

    _AUTHORIZED_IMPORTS: list[str] = [
        "math",
        "statistics",
        "numpy",
        "scipy",
        "statsmodels",
        "decimal",
        "fractions",
        "itertools",
        "collections",
        "json",
        "re",
    ]

    def __init__(
        self,
        model_id: str,
        system_prompt: str,
        max_steps: int = 30,
        model_context_window: int = 1_000_000,
        tracer: Tracer | None = None,
        service_tier: str | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            document_map={},
            chroma_collection=None,  # type: ignore[arg-type]
            emb_model_id="",
            tracer=tracer,
            max_steps=max_steps,
            model_context_window=model_context_window,
            system_prompt_override=system_prompt,
            final_answer_fn=task_solver_final_answer,
            additional_authorized_imports=self._AUTHORIZED_IMPORTS,
            service_tier=service_tier,
        )

    def _build_executor(self) -> LocalPythonExecutor:  # type: ignore[override]
        executor = LocalPythonExecutor(
            additional_authorized_imports=self.additional_authorized_imports,
        )
        executor.send_tools({"final_answer": task_solver_final_answer})
        return executor


# ---------------------------------------------------------------------------
# TaskSolver
# ---------------------------------------------------------------------------


def run_task_solver(
    question: str,
    output_doc_ids: list[str],
    document_map: dict[str, str],
    cfg: TaskSolverConfig,
) -> tuple[str, str | None]:
    """Run the TaskSolver agent on one rollout's final-answer document set.

    Returns ``(answer, error)``.  ``answer`` is the string produced by the
    agent's ``final_answer(...)`` call; if ``output_doc_ids`` is empty (the
    rollout returned no documents) we short-circuit without spending an API
    call.  ``error`` is non-None if the agent failed or hit max steps.
    """
    if not output_doc_ids:
        return "I cannot answer this question with the provided documents.", None

    doc_texts: list[tuple[str, str]] = [
        (did, document_map[did]) for did in output_doc_ids if did in document_map
    ]
    if not doc_texts:
        return "I cannot answer this question with the provided documents.", None

    packed = _pack_documents(
        question=question,
        doc_texts=doc_texts,
        model_context_window=cfg.model_context_window,
    )
    user_message = Template(TASK_SOLVER_USER_PROMPT).render(
        question=question,
        documents=packed,
    )

    try:
        agent = TaskSolverAgent(
            model_id=cfg.task_solver_model_id,
            system_prompt=TASK_SOLVER_SYSTEM_PROMPT,
            max_steps=cfg.task_solver_max_steps,
            model_context_window=cfg.model_context_window,
            service_tier=cfg.service_tier,
        )
        outcome = agent._run_loop(user_message)
    except Exception as e:
        return "", f"task_solver agent crashed: {e}"

    raw = outcome.raw_output
    if isinstance(raw, dict) and raw.get(TASK_SOLVER_FINAL_ANSWER_TAG):
        return str(raw.get("answer") or ""), agent._error
    error = agent._error or "task_solver: agent did not call final_answer"
    return "", error


def _pack_documents(
    question: str,
    doc_texts: list[tuple[str, str]],
    model_context_window: int,
) -> str:
    """Render the documents block, halving the longest doc until it fits.

    Each doc is rendered as ``[doc_id: X]\\n<text>\\n``. We do NOT cap any
    single document unless the full set would exceed the model context
    window; in that case we repeatedly halve the currently-longest document
    (round-robin) until the rendered prompt fits within
    ``(context - headroom)`` characters.
    """
    max_chars = max(
        1, model_context_window * _CHARS_PER_TOKEN_ESTIMATE - _TASK_SOLVER_HEADROOM_CHARS
    )

    # Account for the question + prompt scaffolding length so the headroom
    # absorbs only template + completion overhead, not the question itself.
    overhead = len(question) + _QUESTION_HEADROOM_CHARS
    budget = max(1, max_chars - overhead)

    texts: list[list[str]] = [[t] for _, t in doc_texts]
    ids = [did for did, _ in doc_texts]

    def total_chars() -> int:
        total = 0
        for did, parts in zip(ids, texts, strict=True):
            total += len(did) + len("".join(parts)) + 16  # header + newlines
        return total

    # Round-robin halve the currently-longest document until we fit.
    # Each halving truncates the last surviving fragment to its first
    # half (rounding up so we always keep at least 1 char).
    while total_chars() > budget:
        # find longest doc by current rendered length
        longest_idx = max(
            range(len(texts)),
            key=lambda i: sum(len(p) for p in texts[i]),
        )
        current = "".join(texts[longest_idx])
        if len(current) <= 1:
            # nothing more we can halve; bail out and accept overflow.
            break
        half = max(1, (len(current) + 1) // 2)
        texts[longest_idx] = [current[:half]]

    rendered: list[str] = []
    for did, parts in zip(ids, texts, strict=True):
        body = "".join(parts)
        rendered.append(f"[doc_id: {did}]\n{body}\n")
    return "\n".join(rendered)


# ---------------------------------------------------------------------------
# NuggetCompletion
# ---------------------------------------------------------------------------


def run_nugget_completion(
    question: str,
    candidate_answer: str,
    nuggets: list[str],
    cfg: TaskSolverConfig,
    strict: bool | None = None,
) -> tuple[float | None, list[str] | None, str | None]:
    """Judge ``candidate_answer`` against the gold ``nuggets``.

    Returns ``(score, labels, error)``.

    * ``score`` is in ``[0, 1]``.
      - strict (default): ``#support / #nuggets``.
      - non-strict:       ``(#support + 0.5 * #partial_support) / #nuggets``.
    * ``labels`` is the per-nugget judgement list, same length / order as
      ``nuggets``.
    * Empty ``nuggets`` -> ``(None, [], None)``.
    * Malformed judge output -> ``(None, None, "...")``.
    """
    if strict is None:
        strict = cfg.strict_nugget_scoring

    cleaned = [n.strip() for n in nuggets if n and n.strip()]
    if not cleaned:
        return None, [], None

    prompt = Template(NUGGET_COMPLETION_PROMPT).render(
        question=question,
        candidate_answer=candidate_answer,
        nuggets=cleaned,
    )

    try:
        send_kwargs: dict = {
            "model": cfg.nugget_judge_model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if cfg.service_tier is not None:
            send_kwargs["service_tier"] = cfg.service_tier
        resp = cfg.or_client.chat.send(**send_kwargs)  # type: ignore[attr-defined]
        text = (resp.choices[0].message.content or "").strip()  # type: ignore[union-attr]
    except Exception as e:
        return None, None, f"nugget_completion call failed: {e}"

    labels = _parse_labels(text, expected_len=len(cleaned))
    if labels is None:
        return None, None, "nugget_completion: malformed judge output"

    n_support = sum(1 for label in labels if label == "support")
    n_partial = sum(1 for label in labels if label == "partial_support")
    score = n_support / len(cleaned) if strict else (n_support + 0.5 * n_partial) / len(cleaned)
    return score, labels, None


def _parse_labels(text: str, expected_len: int) -> list[str] | None:
    """Extract the ``labels`` array from the judge's JSON output."""
    m = _JSON_BLOCK_RE.search(text)
    if m is None:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    labels = obj.get("labels") if isinstance(obj, dict) else None
    if not isinstance(labels, list) or len(labels) != expected_len:
        return None
    out: list[str] = []
    for lbl in labels:
        s = str(lbl).strip().lower()
        if s not in {"support", "partial_support", "not_support"}:
            return None
        out.append(s)
    return out
