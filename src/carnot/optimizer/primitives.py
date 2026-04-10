from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from carnot.core.models import PlanCost
from carnot.operators.logical import LogicalOperator
from carnot.operators.physical import PhysicalOperator
from carnot.utils.hash_helpers import hash_for_id

if TYPE_CHECKING:
    from carnot.optimizer import rules


@dataclass(frozen=True)
class CostEntry:
    """One costed configuration of a physical expression.

    The optimizer allocates one ``CostEntry`` for every (physical-expression,
    input-cost-combination) pair it evaluates.  Each entry stores the
    cumulative ``PlanCost`` and back-pointers (as integer IDs) to the
    ``CostEntry`` objects chosen for the input groups.  This decouples
    pareto-frontier bookkeeping from plan reconstruction.

    Representation invariant:
        - ``entry_id`` is globally unique and non-negative.
        - ``plan_cost`` is the cumulative cost of the sub-plan rooted
          at the owning expression with the specific input choices
          encoded in ``input_entry_ids``.
        - ``len(input_entry_ids)`` equals the number of input groups
          of the owning expression (0 for scans, 1 for unary, 2 for
          joins).

    Abstraction function:
        Represents a single point in the (cost, time, quality) space
        for a physical expression, together with the exact input
        choices that produce that point.
    """

    entry_id: int
    plan_cost: PlanCost
    input_entry_ids: tuple[int, ...]


class Expression:
    """A multi-expression in the optimizer's group tree.

    An ``Expression`` wraps either a logical operator (if it's a
    ``LogicalExpression``) or a physical operator (if it's a
    ``PhysicalExpression``), together with the group IDs of its inputs.

    After costing, ``pareto_entry_ids`` contains the IDs of this
    expression's pareto-optimal ``CostEntry`` objects (stored in the
    optimizer's global ``cost_entries`` dict).

    Representation invariant:
        - ``expr_id`` is deterministic given ``(operator, input_group_ids, class)``.
        - ``pareto_entry_ids`` is ``None`` before costing and a non-empty
          ``list[int]`` after costing.

    Abstraction function:
        Represents one way to compute the logical sub-query associated
        with its owning ``Group``, using ``operator`` applied to the
        sub-queries represented by ``input_group_ids``.
    """

    def __init__(
        self,
        operator: LogicalOperator | PhysicalOperator,
        input_group_ids: list[int],
        group_id: int | None = None,
    ):
        self.operator = operator
        self.input_group_ids = input_group_ids
        self.group_id = group_id
        self.rules_applied = set()

        # IDs into the optimizer's global cost_entries dict; populated by
        # OptimizePhysicalExpression after costing.
        self.pareto_entry_ids: list[int] | None = None

        # compute the expression id
        self.expr_id = self._compute_expr_id()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Expression) and self.expr_id == other.expr_id

    def __str__(self):
        expr_str = f"{self.__class__.__name__}(group_id={self.group_id}, expr_id={self.expr_id})"
        expr_str += f"\n  - input_group_ids: {self.input_group_ids}"
        expr_str += f"\n  - operator:\n{str(self.operator)}"
        return expr_str

    def __hash__(self):
        op_id = self.operator.get_logical_op_id() if isinstance(self.operator, LogicalOperator) else self.operator.get_full_op_id()
        hash_str = str(tuple(sorted(self.input_group_ids)) + (op_id, str(self.__class__.__name__)))
        hash_id = int(hash_for_id(hash_str), 16)
        return hash_id

    def _compute_expr_id(self) -> int:
        return self.__hash__()

    def add_applied_rule(self, rule: type[rules.Rule]):
        self.rules_applied.add(rule.get_rule_id())

    def set_group_id(self, group_id: int) -> None:
        self.group_id = group_id


class LogicalExpression(Expression):
    pass


class PhysicalExpression(Expression):
    
    @classmethod
    def from_op_and_logical_expr(cls, op: PhysicalOperator, logical_expression: LogicalExpression) -> PhysicalExpression:
        """Construct a PhysicalExpression given a physical operator and a logical expression."""
        return cls(
            operator=op,
            input_group_ids=logical_expression.input_group_ids,
            group_id=logical_expression.group_id,
        )


class Group:
    """A group of logically equivalent expressions.

    A group maintains sets of logical and physical multi-expressions that
    are all semantically equivalent (they produce the same result set).
    During optimization, the group accumulates a *pareto frontier* —
    the set of (expression, cost-entry) pairs that are not dominated in
    (cost, time, quality) space.

    Representation invariant:
        - ``group_id`` is unique within the optimizer.
        - ``pareto_frontier`` is ``None`` before costing and a
          ``list[tuple[int, int]]`` after at least one physical
          expression has been costed, where each element is
          ``(expr_id, cost_entry_id)``.

    Abstraction function:
        Represents all known ways (logical and physical) to compute a
        particular sub-query, together with the pareto-optimal cost
        points discovered so far.
    """

    def __init__(self, logical_expressions: list[LogicalExpression], group_id: int):
        self.logical_expressions: set[LogicalExpression] = set(logical_expressions)
        self.physical_expressions: set[PhysicalExpression] = set()
        self.explored = False
        self.optimized = False
        self.group_id = group_id

        # Pareto frontier: list of (expr_id, cost_entry_id) pairs.
        # None means "not yet costed"; an empty list is valid (no feasible plans).
        self.pareto_frontier: list[tuple[int, int]] | None = None

    def set_explored(self):
        self.explored = True
