"""Structured progress events emitted during planning and execution.

This module defines lightweight dataclasses that represent progress
updates.  They are yielded by generator methods such as
:meth:`Execution.plan_stream` and :meth:`Execution.run_stream` so that
callers (e.g. the web backend) can forward them to the frontend over
SSE without coupling to agent internals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from carnot.core.models import OperatorStats


@dataclass
class PlanningProgress:
    """A single progress event produced during the planning phase.

    Representation invariant:
        - ``phase`` is one of ``"data_discovery"``, ``"logical_plan"``,
          or ``"paraphrase"``.
        - ``message`` is a non-empty string describing the current activity.
        - ``step`` >= 1 when present; ``None`` for phase-level events.

    Abstraction function:
        Represents a human-readable status update about what the planner
        is currently doing, tagged with the phase and (optionally) step
        number within that phase.
    """

    phase: str
    """Which planning sub-phase produced this event."""

    message: str
    """A short, user-facing description of the current activity."""

    step: int | None = None
    """The 1-based step number within the phase, if applicable."""

    total_steps: int | None = None
    """The maximum number of steps allowed for this phase."""

    step_cost_usd: float | None = None
    """Cost of this step in USD."""

    code_action: str | None = None
    """The code the agent generated and executed in this step,
    if applicable."""

    observations: str | None = None
    """The text output / observations from executing the code,
    if applicable.  This is the same string stored on the
    ``ActionStep`` and fed back to the LLM."""

    error: str | None = None
    """Error message if the step failed."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON encoding.

        Returns:
            A dict with all fields; ``None`` values are omitted for
            cleaner JSON output.

        Raises:
            None.
        """
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class ExecutionProgress:
    """A single progress event produced during the execution phase.

    Representation invariant:
        - ``operator_index`` >= 0 when present.
        - ``total_operators`` >= 1 when present.
        - ``operator_index <= total_operators`` when both are present.

    Abstraction function:
        Represents a human-readable status update about what the
        execution engine is currently doing — which operator is running,
        how far through the plan it is, and any per-operator metadata
        (e.g. item count, operator type).
    """

    message: str
    """A short, user-facing description of the current activity."""

    operator_index: int | None = None
    """0-based index of the operator currently being executed."""

    total_operators: int | None = None
    """Total number of operators (including datasets) in the plan."""

    operator_name: str | None = None
    """The human-readable name/type of the current operator."""

    step_cost_usd: float | None = None
    """Cost of this operator step only, in USD.  Derived from
    ``operator_stats.total_cost_usd`` when available."""

    detail: dict[str, Any] = field(default_factory=dict)
    """Optional machine-readable metadata (e.g. item counts)."""

    operator_stats: OperatorStats | None = None
    """Stats for the operator that just completed, if available."""

    item_count: int | None = None
    """Number of items in the output dataset after this operator, if known."""

    preview_items: list[dict] | None = None
    """A small sample (≤ 5) of output items for preview."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON encoding.

        Returns:
            A dict with all fields; ``None`` values are omitted for
            cleaner JSON output.  ``operator_stats`` is serialized via
            its Pydantic ``.model_dump()`` method.

        Raises:
            None.
        """
        d = asdict(self)
        # Replace the dataclass-serialized operator_stats with Pydantic dict
        if self.operator_stats is not None:
            d["operator_stats"] = self.operator_stats.model_dump()
        return {k: v for k, v in d.items() if v is not None}
