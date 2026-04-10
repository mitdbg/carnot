"""Plan feedback types for re-optimisation.

This module defines data classes that represent user-provided feedback
for plan refinement.  They are consumed by
:meth:`Execution.reoptimize` and (in future) by
:meth:`Execution.plan` when a ``PlanFeedback`` is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from carnot.plan.physical_plan import PhysicalPlan


@dataclass
class NodeEdit:
    """A parameter change to a specific plan node.

    Representation invariant:
        - ``node_id`` is a valid node ID in the target plan.
        - ``new_params`` is non-empty.

    Abstraction function:
        Represents the user's intent to change one or more parameters
        of a plan node (e.g., changing a filter condition or a map
        field description).  Only the keys present in ``new_params``
        are updated; the rest of the node's params are preserved.
    """

    node_id: str
    new_params: dict


@dataclass
class StructuralChange:
    """An insertion or deletion of a plan node.

    Representation invariant:
        - ``change_type`` is ``"insert"`` or ``"delete"``.
        - If ``"insert"``, ``after_node_id`` and ``new_node_params``
          are not ``None``.
        - If ``"delete"``, ``node_id`` is not ``None``.

    Abstraction function:
        Represents the user's intent to add or remove a step in the
        plan.  For insertions, the new node is spliced in after the
        given node and before its children.  For deletions, the node
        is removed and its parents are rewired to its children.
    """

    change_type: str  # "insert" | "delete"
    node_id: str | None = None  # For delete
    after_node_id: str | None = None  # For insert
    new_node_params: dict | None = None  # For insert


@dataclass
class PlanFeedback:
    """Bundle of user signals for plan refinement.

    Representation invariant:
        - At least one field is non-``None`` / non-empty.

    Abstraction function:
        Represents a bundle of user signals (chat messages, parameter
        edits, structural changes, constraint updates) that the planner
        should incorporate when generating or refining a plan.
    """

    chat_message: str | None = None
    node_edits: list[NodeEdit] = field(default_factory=list)
    structural_changes: list[StructuralChange] = field(default_factory=list)
    cost_budget: float | None = None
    previous_plan: PhysicalPlan | None = None
