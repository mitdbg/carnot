"""Run identity shared by the table scripts in this package.

Each SearchAgent variant used to be its own qatfd *system* (``search_agent``,
``qatfd_search_agent``, ``ablation_search_agent``), so a run's directory —
``results/<benchmark>/<system>/<run>/`` — doubled as its identity and the tables keyed
their rows on that directory name. The variants are now a single ``search_agent`` system
configured by its retrieval-tool flags, so every SearchAgent run lands under
``results/<benchmark>/search_agent/`` and the identity has to be recovered from the run's
persisted ``config.yaml`` instead. This module is the one place that does that mapping.

``rag_llm`` is still a distinct system and keeps its own row.
"""

from __future__ import annotations

from dataclasses import dataclass

# Retrieval tool set ``(vector, grep, sem)`` -> (sort order, short label, descriptive label).
# ``read_document`` and ``prune`` are always present, so they are not part of the identity.
# Two points in the lattice are the systems the paper names, hence their short labels.
TOOLSETS: dict[tuple[bool, bool, bool], tuple[int, str, str]] = {
    (False, True, False): (1, "Grep-only", "grep+read"),
    (True, False, False): (2, "Vector-only", "vector+read"),
    (False, False, True): (3, "Sem-only", "sem+read"),
    (True, True, False): (4, "SearchAgent", "grep+vector+read (SearchAgent)"),
    (False, True, True): (5, "QATFD", "grep+sem+read (QATFD)"),
    (True, True, True): (6, "All tools", "all tools"),
}

# Systems that are not the configurable SearchAgent get a fixed label and sort ahead of it.
SYSTEM_LABELS: dict[str, str] = {"rag_llm": "RAG-LLM"}
_SYSTEM_ORDER: dict[str, int] = {"rag_llm": 0}
_UNKNOWN_ORDER = 99

# Working-set mode -> label suffix. Part of the identity so a working-set ablation never
# averages into the row of an otherwise-identical run that had the abstraction on. Labels
# stay LaTeX-safe (no ``_``/``%``/``&``) because they are emitted into tables unescaped.
_WS_SUFFIX: dict[str, str] = {"full": "", "ids": " (WS: ids only)", "off": " (WS: off)"}

AGENT_SYSTEM = "search_agent"


@dataclass(frozen=True)
class Variant:
    """What a run *is*, independent of where it was written."""

    system: str
    # (vector, grep, sem) for the SearchAgent; None for systems with a fixed tool set.
    tools: tuple[bool, bool, bool] | None
    ws_mode: str  # "full" | "ids" | "off"
    order: int
    label: str  # short label, for the main tables
    tools_label: str  # descriptive tool-set label, for the ablation tables


def _adhoc_label(tools: tuple[bool, bool, bool]) -> str:
    """Fallback label for a tool set outside :data:`TOOLSETS` (e.g. read-only)."""
    vector, grep, sem = tools
    parts = [name for name, on in (("grep", grep), ("vector", vector), ("sem", sem)) if on]
    return "+".join([*parts, "read"]) if parts else "read-only"


def _ws_mode(systems: dict) -> str:
    if systems.get("working_set_off"):
        return "off"
    if systems.get("id_tracking_only"):
        return "ids"
    return "full"


def variant_of(cfg: dict, *, fallback_system: str = "") -> Variant:
    """The variant a run represents, read from its persisted ``config.yaml``.

    ``fallback_system`` is used when the snapshot has no ``systems.name`` (pass the run
    dir's parent name, which is what the runner names after the system).
    """
    systems = cfg.get("systems") or {}
    system = systems.get("name") or fallback_system
    if system != AGENT_SYSTEM:
        label = SYSTEM_LABELS.get(system, system or "unknown")
        return Variant(
            system=system,
            tools=None,
            ws_mode="full",
            order=_SYSTEM_ORDER.get(system, _UNKNOWN_ORDER),
            label=label,
            tools_label=label,
        )

    tools = (
        bool(systems.get("include_search_corpus")),
        bool(systems.get("include_grep_corpus")),
        bool(systems.get("include_semantic_filter")),
    )
    adhoc = _adhoc_label(tools)
    order, short, descriptive = TOOLSETS.get(tools, (_UNKNOWN_ORDER, adhoc, adhoc))
    ws_mode = _ws_mode(systems)
    suffix = _WS_SUFFIX[ws_mode]
    return Variant(
        system=system,
        tools=tools,
        ws_mode=ws_mode,
        order=order,
        label=short + suffix,
        tools_label=descriptive + suffix,
    )


def llm_of(cfg: dict) -> str:
    """The run's agent model. It lives under ``inference`` since that config group was split
    out of ``systems``; the ``systems`` lookups are the pre-split fallback."""
    inference = cfg.get("inference") or {}
    systems = cfg.get("systems") or {}
    return inference.get("llm_model") or systems.get("llm_model") or systems.get("agent_model_id") or "unknown"


def judge_of(cfg: dict, agent_llm: str) -> str | None:
    """The semantic-filter judge model, or None when the run has no semantic filter. An unset
    ``semantic_filter_llm_model`` means the judge shares the agent model."""
    systems = cfg.get("systems") or {}
    if not systems.get("include_semantic_filter"):
        return None
    return systems.get("semantic_filter_llm_model") or agent_llm


def model_label(llm: str, judge: str | None, *, short: bool = False) -> str:
    """``agent`` normally, ``agent / judge`` when the judge ran on a different model."""
    fmt = (lambda m: m.split("/")[-1]) if short else (lambda m: m)
    if judge and judge != llm:
        return f"{fmt(llm)} / {fmt(judge)}"
    return fmt(llm)


def model_sort_key(model: str, preferred: list[str]) -> tuple[int, str]:
    """Sort models by a preferred order first, then alphabetically."""
    return (preferred.index(model) if model in preferred else len(preferred), model)
