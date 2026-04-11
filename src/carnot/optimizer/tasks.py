from __future__ import annotations

from typing import TYPE_CHECKING, Any

from carnot.core.models import PlanCost
from carnot.optimizer.cost_model import CostModel

if TYPE_CHECKING:
    from carnot.optimizer.optimizer import Optimizer
from carnot.optimizer.primitives import CostEntry, Expression, Group
from carnot.optimizer.rules import ImplementationRule, Rule, TransformationRule


class Task:
    """
    Base class for a task. Each task has a method called perform() which executes the task.
    Examples of tasks include optimizing and exploring groups, optimizing expressions, applying
    rules, and optimizing inputs / costing the full group tree.
    """

    def perform(self, groups: dict[int, Group], context: dict[str, Any] | None = None) -> list[Task]:
        """
        NOTE: At the moment we do not make use of the context, but in the future
        this can be used to store required physical properties (e.g. sort conditions
        for the query) and bounds (e.g. the operator should not cost more than X).
        """
        raise NotImplementedError("Calling this method from an abstract base class.")


class OptimizeGroup(Task):
    """
    The task to optimize a group.

    This task pushes optimization tasks for the group's current logical and physical
    expressions onto the tasks stack. This will fully expand the space of possible
    logical and physical expressions for the group, because OptimizeLogicalExpression
    and OptimizePhysicalExpression tasks will indirectly schedule new tasks to apply
    rules and to optimize input groups and expressions.
    """

    def __init__(self, group_id: int):
        self.group_id = group_id

    def perform(self, groups: dict[int, Group], context: dict[str, Any] | None = None) -> list[Task]:
        # get updated instance of the group to be optimized
        context = context or {}
        group = groups[self.group_id]

        # if this group has already been optimized, there's nothing more to do
        if group.optimized:
            return []

        # otherwise, optimize all the logical expressions for the group
        new_tasks = []
        for logical_expr in group.logical_expressions:
            task = OptimizeLogicalExpression(logical_expr)
            new_tasks.append(task)

        # and optimize all of the physical expressions in the group
        for physical_expr in group.physical_expressions:
            task = OptimizePhysicalExpression(physical_expr)
            new_tasks.append(task)

        # and first explore the group if it hasn't been explored yet
        if not group.explored:
            task = ExploreGroup(self.group_id)
            new_tasks.append(task)

        return new_tasks


class ExploreGroup(Task):
    """
    The task to explore a group and add additional logical expressions.
    """

    def __init__(self, group_id: int):
        self.group_id = group_id

    def perform(self, groups: dict[int, Group], context: dict[str, Any] | None = None) -> list[Task]:
        # fetch group
        context = context or {}
        group = groups[self.group_id]

        # if the group has been explored before, return []
        if group.explored:
            return []

        # for each logical_expr in the group, add a new OptimizeLogicalExpression() task to the queue
        new_tasks = []
        for logical_expr in group.logical_expressions:
            task = OptimizeLogicalExpression(logical_expr, exploring=True)
            new_tasks.append(task)

        # but first (tasks are LIFO), we recursively explore input groups of logical expressions in this group
        for logical_expr in group.logical_expressions:
            for input_group_id in logical_expr.input_group_ids:
                task = ExploreGroup(input_group_id)
                new_tasks.append(task)

        # mark the group as explored and return tasks
        group.set_explored()

        return new_tasks


class OptimizeLogicalExpression(Task):
    """
    The task to optimize a (multi-)expression.

    This task filters for the subset of rules which may be applied to the given logical expression
    and schedules ApplyRule tasks for each rule.
    """

    def __init__(self, logical_expression: Expression, exploring: bool = False):
        self.logical_expression = logical_expression
        self.exploring = exploring

    def perform(
        self,
        transformation_rules: list[type[TransformationRule]],
        implementation_rules: list[type[ImplementationRule]],
        context: dict[str, Any] | None = None,
    ) -> list[Task]:
        # if we're exploring, only apply transformation rules which are tagged for exploration
        context = context or {}
        rules = (
            [rule for rule in transformation_rules if rule.is_exploration_rule()]
            if self.exploring
            else transformation_rules + implementation_rules
        )

        # filter out rules that have already been applied to logical expression
        rules = list(filter(lambda rule: rule.get_rule_id() not in self.logical_expression.rules_applied, rules))

        # filter for rules that match on this logical expression
        rules = list(filter(lambda rule: rule.matches_pattern(self.logical_expression, **context), rules))

        # TODO compute priority (i.e. "promise") of the rules and sort in order of priority

        # apply rules, exploring the input group(s) of each pattern if necessary
        new_tasks = []
        for rule in rules:
            # TODO: if necessary, expand the input groups of the logical expression to see if they need to be expanded
            apply_rule_task = ApplyRule(rule, self.logical_expression, self.exploring)
            new_tasks.append(apply_rule_task)

        return new_tasks


class ApplyRule(Task):
    """
    The task to apply a transformation or implementation rule to a (multi-)expression.

    For TransformationRules, this task will:
    - apply the substitution, receiving new expressions and groups
    - filter the new expressions for ones which may already exist
      - NOTE: we don't filter new groups because this is already done by the
              transformation rule in order to assign the correct group_id to any
              new expressions.
    - add new expressions to their group's set of logical expressions
    - schedule OptimizeGroup and OptimizeLogicalExpression tasks

    For ImplementationRules, this task will:
    - apply the substitution, receiving new expressions and groups
    - filter the new expressions for ones which may already exist
    - add new expressions to their group's set of physical expressions
    - schedule OptimizePhysicalExpression tasks
    """

    def __init__(self, rule: type[Rule], logical_expression: Expression, exploring: bool = False):
        self.rule = rule
        self.logical_expression = logical_expression
        self.exploring = exploring

    def _apply_transformation_rule(
        self,
        groups: dict[int, Group],
        expressions: dict[int, Expression],
        next_group_id: int,
        physical_op_params: dict[str, Any],
    ) -> tuple[list[Task], int]:
        new_tasks = []

        # apply transformation rule
        new_expressions, new_groups, next_group_id = self.rule.substitute(
            self.logical_expression, groups, expressions, next_group_id, **physical_op_params
        )

        # filter out any expressions which are duplicates (i.e. they've been previously computed)
        new_expressions = [expr for expr in new_expressions if expr.expr_id not in expressions]
        expressions.update({expr.expr_id: expr for expr in new_expressions})

        # add all new groups to the groups mapping
        for group in new_groups:
            groups[group.group_id] = group
            task = OptimizeGroup(group.group_id)

        # add new expressions to their respective groups
        for expr in new_expressions:
            group = groups[expr.group_id]
            group.logical_expressions.add(expr)

        # create new tasks for optimizing new logical expressions
        for expr in new_expressions:
            task = OptimizeLogicalExpression(expr, self.exploring)
            new_tasks.append(task)

        # NOTE: we place new tasks for groups on the top of the stack so that they may be
        #       optimized before we optimize expressions which take new groups as inputs
        # create new tasks for optimizing new groups
        for group in new_groups:
            task = OptimizeGroup(group.group_id)
            new_tasks.append(task)

        return new_tasks, next_group_id

    def _apply_implementation_rule(self, group: Group, expressions: dict[int, Expression], physical_op_params: dict[str, Any]) -> list[Task]:
        new_tasks = []

        # apply implementation rule
        new_expressions = self.rule.substitute(self.logical_expression, **physical_op_params)
        new_expressions = [expr for expr in new_expressions if expr.expr_id not in expressions]

        # update expressions mapping and add new physical expressions to the group
        expressions.update({expr.expr_id: expr for expr in new_expressions})
        group.physical_expressions.update(new_expressions)

        # create new task
        for expr in new_expressions:
            task = OptimizePhysicalExpression(expr)
            new_tasks.append(task)

        return new_tasks

    def perform(
        self,
        groups: dict[int, Group],
        expressions: dict[int, Expression],
        next_group_id: int = 0,
        **physical_op_params,
    ) -> tuple[list[Task], int]:
        # check if rule has already been applied to this logical expression; return [] if so
        if self.rule.get_rule_id() in self.logical_expression.rules_applied:
            return [], next_group_id

        # get the group of the logical expression
        group_id = self.logical_expression.group_id
        group = groups[group_id]

        # process new expressions; update groups and create new tasks as needed
        if issubclass(self.rule, TransformationRule):
            new_tasks, next_group_id = self._apply_transformation_rule(
                groups, expressions, next_group_id, physical_op_params
            )
        else:
            new_tasks = self._apply_implementation_rule(group, expressions, physical_op_params)

        # mark that the rule has been applied to the logical expression
        self.logical_expression.add_applied_rule(self.rule)

        return new_tasks, next_group_id


class OptimizePhysicalExpression(Task):
    """
    The task to optimize a physical expression and derive its cost.

    This task computes the cost of input groups for the given physical expression (scheduling
    OptimizeGroup tasks if needed), computes the cost of the given expression, and then updates
    the expression's group depending on whether this expression is in its `pareto_optimal_physical_expressions`.
    """

    def __init__(self, physical_expression: Expression):
        self.physical_expression = physical_expression

    def perform(
        self,
        cost_model: CostModel,
        groups: dict[int, Group],
        optimizer: Optimizer,
        context: dict[str, Any] | None = None,
    ) -> list[Task]:
        """Cost this physical expression using CostEntry-based bookkeeping.

        Computes costs for all pareto-optimal combinations of input
        group entries, writes ``CostEntry`` objects into the optimizer's
        global store, and updates the expression's and group's pareto
        frontiers.

        Requires:
            - ``cost_model`` is a callable ``CostModel``.
            - ``groups`` maps group IDs to ``Group`` instances.
            - ``optimizer`` exposes ``cost_entries: dict[int, CostEntry]``,
              ``next_entry_id: int``.

        Returns:
            A list of ``Task`` objects to push onto the task stack.
            Empty when costing is complete; contains ``[self] + [OptimizeGroup(...)]``
            when input groups still need optimization.

        Raises:
            None.
        """
        context = context or {}
        expr = self.physical_expression

        # return if we've already computed costs for this expression
        if expr.pareto_entry_ids is not None:
            return []

        # Collect pareto entries from input groups.
        input_group_entries: list[list[CostEntry]] = []
        new_tasks: list[Task] = []
        for gid in expr.input_group_ids:
            g = groups[gid]
            if g.pareto_frontier is None:
                new_tasks.append(OptimizeGroup(gid))
            else:
                entries = [optimizer.cost_entries[eid] for _, eid in g.pareto_frontier]
                input_group_entries.append(entries)

        # If not all input groups have been costed, schedule them first then retry.
        if new_tasks:
            return [self] + new_tasks

        # Compute cross-product of input pareto entries.
        if len(input_group_entries) == 0:
            combos: list[tuple[CostEntry, ...]] = [()]  # leaf (scan) — no inputs
        elif len(input_group_entries) == 1:
            combos = [(e,) for e in input_group_entries[0]]
        else:
            # join: 2 inputs
            combos = [
                (le, ri) for le in input_group_entries[0] for ri in input_group_entries[1]
            ]

        # Cost each combination.
        candidate_entries: list[CostEntry] = []
        for combo in combos:
            if len(combo) == 0:
                plan_cost = cost_model(expr.operator)
            elif len(combo) == 1:
                plan_cost = cost_model(expr.operator, combo[0].plan_cost)
            else:
                plan_cost = cost_model(expr.operator, combo[0].plan_cost, combo[1].plan_cost)

            entry = CostEntry(
                entry_id=optimizer.next_entry_id,
                plan_cost=plan_cost,
                input_entry_ids=tuple(e.entry_id for e in combo),
            )
            optimizer.next_entry_id += 1
            optimizer.cost_entries[entry.entry_id] = entry
            candidate_entries.append(entry)

        # Keep only pareto-optimal entries for this expression.
        pareto = _pareto_filter(candidate_entries)
        expr.pareto_entry_ids = [e.entry_id for e in pareto]

        # Update group frontier.
        group = groups[expr.group_id]
        _update_group_frontier(group, expr, pareto, optimizer)

        group.optimized = True
        return []


# ── Module-level helpers for pareto bookkeeping ───────────────────────


def _dominates(a: PlanCost, b: PlanCost) -> bool:
    """Return ``True`` if *a* dominates *b* in (cost, time, quality) space.

    *a* dominates *b* iff *a* is at least as good on every objective and
    strictly better on at least one.  Lower cost and time are better;
    higher quality is better.

    Requires:
        - *a* and *b* are valid ``PlanCost`` objects.

    Returns:
        ``True`` if *a* dominates *b*, ``False`` otherwise.

    Raises:
        None.
    """
    cost_ok = a.cost <= b.cost
    time_ok = a.time <= b.time
    quality_ok = a.quality >= b.quality
    cardinality_ok = a.output_cardinality <= b.output_cardinality
    if not (cost_ok and time_ok and quality_ok and cardinality_ok):
        return False
    # Must be strictly better on at least one dimension.
    return (
        a.cost < b.cost or a.time < b.time or a.quality > b.quality or a.output_cardinality < b.output_cardinality
    )


def _pareto_filter(entries: list[CostEntry]) -> list[CostEntry]:
    """Return the subset of *entries* that are not dominated by any other entry.

    Uses an $O(n^2)$ pairwise dominance check.  This is acceptable
    because the number of entries per expression is small (bounded by
    the cross-product of input pareto frontiers).

    Requires:
        - *entries* is a non-empty list of ``CostEntry`` objects.

    Returns:
        A list of ``CostEntry`` objects forming the pareto frontier.

    Raises:
        None.
    """
    result: list[CostEntry] = []
    for e in entries:
        if not any(_dominates(o.plan_cost, e.plan_cost) for o in entries if o is not e):
            result.append(e)
    return result


def _update_group_frontier(
    group: Group,
    expr: Expression,
    new_pareto_entries: list[CostEntry],
    optimizer: Optimizer,
) -> None:
    """Merge *new_pareto_entries* into the group's pareto frontier.

    Each entry becomes a ``(expr_id, entry_id)`` pair on the group.  If
    the group already has a frontier, the old and new pairs are merged
    and re-filtered for dominance.

    Requires:
        - *group* is the ``Group`` that owns *expr*.
        - *new_pareto_entries* are the pareto-optimal ``CostEntry``
          objects for *expr*.
        - ``optimizer.cost_entries`` contains all referenced entries.

    Returns:
        None (mutates *group* in place).

    Raises:
        None.
    """
    new_pairs = [(expr.expr_id, e.entry_id) for e in new_pareto_entries]

    if group.pareto_frontier is None:
        group.pareto_frontier = new_pairs
        return

    # Merge existing frontier with new entries, then re-filter.
    all_pairs = group.pareto_frontier + new_pairs
    all_entries = [optimizer.cost_entries[eid] for _, eid in all_pairs]
    pareto_set = {e.entry_id for e in _pareto_filter(all_entries)}
    group.pareto_frontier = [(xid, eid) for xid, eid in all_pairs if eid in pareto_set]
