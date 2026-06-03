"""QualityFilter agent + persistence.

Final stage of the QA-synthesis funnel. For every QA pair that survives
the rollout pass-rate filter we instantiate a fresh ``SearchAgent``
configured as a *Quality Filter*: it has the same corpus tools as the
rollout agent but its system prompt instructs it to decide whether the
candidate pair is unambiguous AND correct (KARL paper, Appendix D2,
Figures 35 / 36). The verdict is committed via a single call to
``quality_filter_final_answer(valid, reasoning)``.

The QF agent has access to the corpus so it can resolve genuine
ambiguities (e.g. an OfficeQA question that targets a data slice where
different bulletin pages disagree because one of them is pre-revision).

Results are persisted incrementally to
``{seed}_qa_pairs_quality.json`` since the pass-rate threshold can lower
mid-run, which schedules additional QF tasks for previously-discarded
pairs without re-judging the ones already on disk.
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import yaml
from jinja2 import Template

from skunk.logging.tracer import Tracer
from skunk.retrieve.search_agent import MAX_PARALLEL_TOOL_CALLS, SearchAgent
from skunk.retrieve.search_tools import (
    QUALITY_FILTER_FINAL_ANSWER_TAG,
    quality_filter_final_answer,
)

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

    from skunk.datagen.harness import QAPair
    from skunk.datagen.rollout import RolloutRecord


_PROMPTS_FILE = pathlib.Path(__file__).parent / "prompts.yaml"
with _PROMPTS_FILE.open() as _f:
    _PROMPTS = yaml.safe_load(_f)

QUALITY_FILTER_USER_PROMPT: str = _PROMPTS["quality_filter_user_prompt"]

# Map benchmark name to the QF *system* prompt template. The template is
# rendered with the benchmark's ``special_notes`` so the judge knows the
# corpus-specific rules (data revisions, identifier formats, ...).
BENCHMARK_QUALITY_FILTER_SYSTEM_PROMPT: dict[str, str] = {
    "officeqa":        _PROMPTS["quality_filter_system_prompt_officeqa"],
    "browsecomp-plus": _PROMPTS["quality_filter_system_prompt_browsecomp_plus"],
}


# ---------------------------------------------------------------------------
# Config + result types
# ---------------------------------------------------------------------------


@dataclass
class QualityFilterConfig:
    """Static config for QF runs within one harness invocation."""

    # SearchAgent construction args (mirror RolloutConfig).
    model_id: str
    emb_model_id: str
    document_map: dict[str, str]
    chroma_collection: Collection

    # QF system prompt template (Jinja2). Rendered with ``special_notes``.
    system_prompt_template: str
    # Benchmark-specific corpus notes templated into the system prompt.
    special_notes: str

    # SearchAgent runtime knobs.
    max_steps: int = 30
    max_pages_per_tool_call: int = 20
    model_context_window: int = 1_000_000

    # Which per-attempt score to surface to the judge in the user prompt:
    #   "doc-recall" -> RolloutRecord.doc_output_recall
    #   "eval-score" -> RolloutRecord.eval_score
    binarization_mode: str = "doc-recall"

    # OpenRouter service tier hint applied to every QF agent turn.
    service_tier: str | None = "flex"


@dataclass
class QualityFilterResult:
    """Verdict for one QA pair."""

    qa_id: str
    valid: bool | None
    reasoning: str
    error: str | None
    completed: bool
    num_steps: int
    trace_path: str
    messages_path: str
    # The score field used to populate the per-attempt evidence shown to
    # the judge (so the on-disk record matches what the judge actually saw).
    binarization_mode: str = "doc-recall"


@dataclass
class QualityFilterFunnel:
    """Per-seed funnel record persisted to ``{seed}_qa_pairs_quality.json``."""

    seed: int
    results: list[QualityFilterResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-pair entry point
# ---------------------------------------------------------------------------


def run_quality_filter_for_pair(
    pair: QAPair,
    rollout_records: list[RolloutRecord],
    cfg: QualityFilterConfig,
    trace_dir: str,
    show_output: bool = False,
) -> QualityFilterResult:
    """Run the QualityFilter agent on one QA pair + its rollout attempts."""
    trace_path = os.path.join(trace_dir, f"{pair.qa_id}_qf_trace.txt")
    messages_path = os.path.join(trace_dir, f"{pair.qa_id}_qf_messages.json")

    system_prompt = Template(cfg.system_prompt_template).render(
        special_notes=cfg.special_notes,
        max_parallel_tool_calls=MAX_PARALLEL_TOOL_CALLS,
    )
    user_prompt = Template(QUALITY_FILTER_USER_PROMPT).render(
        question=pair.question,
        ground_truth=_render_nuggets(pair.answer),
        attempts=_render_attempts(rollout_records, cfg.binarization_mode),
    )

    try:
        with Tracer(trace_path, show_output=show_output) as tracer:
            agent = SearchAgent(
                cfg.model_id,
                document_map=cfg.document_map,
                chroma_collection=cfg.chroma_collection,
                emb_model_id=cfg.emb_model_id,
                tracer=tracer,
                max_steps=cfg.max_steps,
                max_pages_per_tool_call=cfg.max_pages_per_tool_call,
                model_context_window=cfg.model_context_window,
                system_prompt_override=system_prompt,
                final_answer_fn=quality_filter_final_answer,
                service_tier=cfg.service_tier,
            )
            outcome = agent._run_loop(user_prompt)
        raw = outcome.raw_output
        valid: bool | None = None
        reasoning: str = ""
        if isinstance(raw, dict) and raw.get(QUALITY_FILTER_FINAL_ANSWER_TAG):
            valid = bool(raw.get("valid"))
            reasoning = str(raw.get("reasoning") or "")
        completed = bool(agent._completed)
        num_steps = int(agent._num_steps)
        error = agent._error
    except Exception as e:
        valid = None
        reasoning = ""
        completed = False
        num_steps = 0
        error = f"quality_filter agent crashed: {e}"

    # persist the message trajectory alongside the trace, when available.
    try:
        if "agent" in locals():
            with open(messages_path, "w") as f:
                json.dump(agent.messages_to_jsonable(), f, indent=2)  # type: ignore[name-defined]
    except Exception:
        pass

    return QualityFilterResult(
        qa_id=pair.qa_id,
        valid=valid,
        reasoning=reasoning,
        error=error,
        completed=completed,
        num_steps=num_steps,
        trace_path=trace_path,
        messages_path=messages_path,
        binarization_mode=cfg.binarization_mode,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def load_quality_funnel(path: str) -> dict[str, QualityFilterResult]:
    """Load ``{seed}_qa_pairs_quality.json`` into ``{qa_id: result}``.

    Returns an empty dict if the file does not exist or is corrupt.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return {}
    out: dict[str, QualityFilterResult] = {}
    for r in payload.get("results", []):
        try:
            res = QualityFilterResult(**r)
        except TypeError:
            continue
        out[res.qa_id] = res
    return out


def persist_quality_funnel(
    seed: int,
    results: dict[str, QualityFilterResult],
    path: str,
) -> None:
    """(Re-)write ``{seed}_qa_pairs_quality.json`` with all known QF results.

    The file is written atomically (via a temp file + rename) since the
    harness may invoke this concurrently with future QF tasks that
    re-trigger when the pass-rate threshold lowers.
    """
    payload = {
        "seed": seed,
        "n_results": len(results),
        "n_valid": sum(1 for r in results.values() if r.valid is True),
        "n_invalid": sum(1 for r in results.values() if r.valid is False),
        "n_error": sum(1 for r in results.values() if r.valid is None),
        "results": [asdict(r) for r in results.values()],
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_nuggets(answer: list[str]) -> str:
    items = [n.strip() for n in answer if n and n.strip()]
    if not items:
        return "  - (no ground truth recorded)"
    return "\n".join(f"  - {n}" for n in items)


def _render_attempts(
    rollouts: list[RolloutRecord],
    binarization_mode: str,
) -> str:
    """Pre-render the per-attempt evidence block shown to the QF judge.

    Each attempt includes:
      - the TaskSolver's nuggetized answer,
      - the rollout's retrieved doc_ids,
      - either ``doc_output_recall`` or ``eval_score`` (per binarization mode),
      - the nugget judgement labels (if available).

    The full document text is NOT inlined here -- the judge has the corpus
    tools and the doc_ids, and can call ``read_document(...)`` if it wants
    a closer look.
    """
    if not rollouts:
        return "(no rollouts available for this pair)"

    score_attr = "eval_score" if binarization_mode == "eval-score" else "doc_output_recall"
    score_label = "eval_score" if binarization_mode == "eval-score" else "doc_output_recall"

    blocks: list[str] = []
    for r in sorted(rollouts, key=lambda x: x.rollout_idx):
        score = getattr(r, score_attr, None)
        score_str = "n/a" if score is None else f"{score:.3f}"
        answer = (getattr(r, "task_solver_answer", None) or "").strip() or "(no task_solver answer)"
        labels = getattr(r, "eval_labels", None)
        labels_str = ", ".join(labels) if labels else "n/a"
        docs = ", ".join(r.output_doc_ids) if r.output_doc_ids else "(none)"
        blocks.append(
            f"--- Attempt {r.rollout_idx} ---\n"
            f"retrieved doc_ids: {docs}\n"
            f"{score_label}: {score_str}\n"
            f"nugget labels: {labels_str}\n"
            f"task_solver answer:\n{answer}\n"
        )
    return "\n".join(blocks)
