import logging
from collections.abc import Callable

import tiktoken

from carnot.agents.models import Model
from carnot.core.models import PlanCost
from carnot.data.dataset import Dataset
from carnot.operators.logical import Scan
from carnot.operators.scan import ScanOp
from carnot.optimizer import (
    IMPLEMENTATION_RULES,
    TRANSFORMATION_RULES,
)
from carnot.optimizer.cost_model import CostModel
from carnot.optimizer.primitives import CostEntry, Expression, Group, LogicalExpression
from carnot.optimizer.tasks import (
    ApplyRule,
    ExploreGroup,
    OptimizeGroup,
    OptimizeLogicalExpression,
    OptimizePhysicalExpression,
)
from carnot.plan import PhysicalPlan
from carnot.plan.node import PlanNode

logger = logging.getLogger(__name__)

# Tokenizer used for estimating per-item token counts during scan creation.
# cl100k_base is a good general-purpose encoding; absolute counts don't need
# to match the downstream LLM's tokenizer — we only care about ratios.
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Number of items to sample when estimating tokens-per-item for a dataset.
_TOKEN_SAMPLE_SIZE = 20


class Optimizer:
    def __init__(
        self,
        model: Model,
        available_model_ids: list[str],
        llm_config: dict | None = None,
        max_workers: int = 64,
        cost_model: CostModel | None = None,
        max_tasks: int = 10_000,
    ):
        self.model = model
        self.available_model_ids = available_model_ids
        self.llm_config = llm_config or {}
        self.max_workers = max_workers
        self.cost_model = cost_model or CostModel()
        self.max_tasks = max_tasks

        # track the expressions and groups created during optimization
        self.expressions: dict[int, Expression] = {}
        self.groups: dict[int, Group] = {}
        self.next_group_id: int = 0

        # CostEntry storage for plan reconstruction (populated by OptimizePhysicalExpression._perform)
        self.cost_entries: dict[int, CostEntry] = {}
        self.next_entry_id: int = 0

        # the stack of tasks to perform during optimization
        self.tasks_stack = []

        # the lists of implementation and transformation rules that the optimizer can apply
        self.implementation_rules = IMPLEMENTATION_RULES
        self.transformation_rules = TRANSFORMATION_RULES

    def update_cost_model(self, cost_model: CostModel):
        self.cost_model = cost_model

    def get_physical_op_params(self):
        return {
            "available_model_ids": self.available_model_ids,
            "llm_config": self.llm_config,
            "max_workers": self.max_workers,
        }

    def _select_best_plan(self, final_group_id: int, policy: str = "min_cost") -> PhysicalPlan:
        """Reconstruct the best physical plan from the optimized group tree.

        Walks the pareto frontier of the final group, picks the best
        ``(expr_id, cost_entry_id)`` pair according to *policy*, and
        recursively builds a ``PhysicalPlan`` by following
        ``CostEntry.input_entry_ids`` back-pointers.

        Requires:
            - Optimization has completed (all reachable groups have
              ``pareto_frontier is not None``).
            - *policy* is one of ``"min_cost"``, ``"min_time"``, or
              ``"max_quality"``.

        Returns:
            A ``PhysicalPlan`` assembled from the pareto-optimal
            expressions chosen by the given policy.

        Raises:
            ValueError: If the final group has no pareto frontier or
            *policy* is unrecognised.
        """
        group = self.groups[final_group_id]
        if not group.pareto_frontier:
            raise ValueError(
                f"Final group {final_group_id} has no pareto frontier — "
                "was the optimisation search run?"
            )

        scorer = _get_scorer(policy)

        # Pick the best (expr_id, entry_id) from the final group.
        best_expr_id, best_entry_id = min(
            group.pareto_frontier,
            key=lambda pair: scorer(self.cost_entries[pair[1]].plan_cost),
        )

        # Recursively build the PlanNode DAG.
        nodes: dict[str, PlanNode] = {}
        counter = {"n": 0}
        self._build_plan_recursive(best_expr_id, best_entry_id, nodes, counter)
        return PhysicalPlan(nodes=nodes)

    def _build_plan_recursive(
        self,
        expr_id: int,
        entry_id: int,
        nodes: dict[str, PlanNode],
        counter: dict[str, int],
    ) -> str:
        """Recursively convert an expression + cost-entry pair into ``PlanNode`` objects.

        Requires:
            - ``expr_id`` is a valid key in ``self.expressions``.
            - ``entry_id`` is a valid key in ``self.cost_entries``.
            - ``counter`` is a mutable dict with key ``"n"`` for unique node ID generation.

        Returns:
            The ``node_id`` of the newly created node.

        Raises:
            None.
        """
        expr = self.expressions[expr_id]
        entry = self.cost_entries[entry_id]

        # Recurse into input groups.
        parent_node_ids: list[str] = []
        for i, input_gid in enumerate(expr.input_group_ids):
            input_entry_id = entry.input_entry_ids[i]

            # Find which expression in the input group owns this entry.
            input_group = self.groups[input_gid]
            input_expr_id = next(
                xid for xid, eid in input_group.pareto_frontier
                if eid == input_entry_id
            )
            child_node_id = self._build_plan_recursive(
                input_expr_id, input_entry_id, nodes, counter,
            )
            parent_node_ids.append(child_node_id)

        # Create PlanNode for this expression.
        node_id = f"node-{counter['n']}"
        counter["n"] += 1

        node = _expr_to_plan_node(expr, node_id, parent_node_ids)
        nodes[node_id] = node
        return node_id

    def _construct_group_tree(self, logical_plan: Dataset) -> int:
        """Recursively convert a Dataset DAG into a group tree for cascades optimization.

        Leaf datasets (``operator is None``) are wrapped in a ``Scan``
        logical operator so the optimizer has a concrete expression to
        attach implementation rules and cost estimates to.

        Requires:
            - ``logical_plan`` is a valid ``Dataset`` node.
            - Leaf datasets have ``operator is None`` and ``parents == []``.

        Returns:
            The group id of the newly created root group.

        Raises:
            None.
        """
        if logical_plan.operator is None:
            # Leaf dataset — create a Scan logical operator.
            num_items = len(logical_plan.items)
            est_tokens_per_item = self._estimate_tokens_per_item(logical_plan)
            op = Scan(
                name=logical_plan.name,
                dataset_id=logical_plan.dataset_id,
                num_items=num_items,
                est_tokens_per_item=est_tokens_per_item,
            )
            input_group_ids = []
        else:
            op = logical_plan.operator
            input_group_ids = [
                self._construct_group_tree(parent) for parent in logical_plan.parents
            ]

        # assign a fresh monotonic group id
        group_id = self.next_group_id
        self.next_group_id += 1

        # construct the logical expression and group
        logical_expression = LogicalExpression(
            operator=op,
            input_group_ids=input_group_ids,
            group_id=group_id,
        )
        group = Group(logical_expressions=[logical_expression], group_id=group_id)

        # add the expression and group to the optimizer's expressions and groups and return
        self.expressions[logical_expression.expr_id] = logical_expression
        self.groups[group.group_id] = group

        return group.group_id

    @staticmethod
    def _estimate_tokens_per_item(dataset: Dataset) -> float:
        """Estimate the average token count per item by sampling.

        Materializes up to ``_TOKEN_SAMPLE_SIZE`` items, concatenates all
        string values in each item dict, and counts tokens using the
        module-level ``_TOKENIZER`` (``cl100k_base``).

        Requires:
            - ``dataset`` is a ``Dataset`` with at least one item, **or**
              an empty dataset (in which case the estimate is ``0.0``).

        Returns:
            The mean token count across sampled items.  Returns ``0.0``
            when the dataset is empty.

        Raises:
            None.  Errors during sampling are logged and cause a
            fallback return of ``0.0``.
        """
        if not dataset.items:
            return 0.0

        try:
            sampled = dataset.sample(n=_TOKEN_SAMPLE_SIZE, random=True)
        except Exception:
            logger.warning(
                "Failed to sample dataset '%s' for token estimation; defaulting to 0.0",
                dataset.dataset_id,
                exc_info=True,
            )
            return 0.0

        if not sampled:
            return 0.0

        total_tokens = 0
        for item in sampled:
            # Concatenate all string-valued fields in the item dict.
            text = " ".join(str(v) for v in item.values() if isinstance(v, str))
            total_tokens += len(_TOKENIZER.encode(text))

        return total_tokens / len(sampled)

    def _convert_query_plan_to_group_tree(self, logical_plan: Dataset) -> int:
        # NOTE: previously we computed each operator's entire set of upstream
        # dependencies here; we may need to revisit this if we want to automatically
        # be able to push filters down through e.g. map operations
        # construct tree of groups
        final_group_id = self._construct_group_tree(logical_plan)

        return final_group_id

    def _search_optimization_space(self, group_id: int, cost_budget: float) -> None:
        # begin the search for an optimal plan with a task to optimize the final group
        initial_task = OptimizeGroup(group_id)
        self.tasks_stack.append(initial_task)

        # stop after exhaustive search or after reaching the max number of tasks, whichever comes first
        task_count = 0
        while len(self.tasks_stack) > 0 and task_count < self.max_tasks:
            task = self.tasks_stack.pop(-1)
            task_count += 1

            new_tasks = []
            if isinstance(task, (OptimizeGroup, ExploreGroup)):
                new_tasks = task.perform(self.groups)
            elif isinstance(task, OptimizeLogicalExpression):
                new_tasks = task.perform(self.transformation_rules, self.implementation_rules)
            elif isinstance(task, ApplyRule):
                new_tasks, self.next_group_id = task.perform(
                    self.groups, self.expressions, self.next_group_id, **self.get_physical_op_params()
                )
            elif isinstance(task, OptimizePhysicalExpression):
                new_tasks = task.perform(self.cost_model, self.groups, self)

            self.tasks_stack.extend(new_tasks)

    def optimize(self, logical_plan: Dataset, cost_budget: float | None = None, policy: str = "min_cost") -> dict:
        """Run cascades optimization and return the best physical plan.

        Requires:
            - *logical_plan* is a valid ``Dataset`` DAG.
            - *policy* is one of ``"min_cost"``, ``"min_time"``, ``"max_quality"``.

        Returns:
            A dict representation of the best ``PhysicalPlan`` (via
            ``PhysicalPlan.to_dict()``).

        Raises:
            ValueError: If the final group has no pareto frontier after
            the search completes.
        """
        # use unlimited budget if none is provided
        cost_budget = cost_budget or float("inf")

        # create the initial group tree from the logical plan and get the final group id
        final_group_id = self._convert_query_plan_to_group_tree(logical_plan)

        # use a cascades-inspired approach to quickly enumerate a space of physical plans
        self._search_optimization_space(final_group_id, cost_budget)

        # select the best physical plan that satisfies the cost budget
        best_plan = self._select_best_plan(final_group_id, policy)

        return best_plan.to_dict()


# ── Module-level helpers ─────────────────────────────────────────────


# Scorer functions: return a float where *lower is better*.
_SCORERS: dict[str, Callable[[PlanCost], float]] = {
    "min_cost": lambda pc: pc.cost,
    "min_time": lambda pc: pc.time,
    "max_quality": lambda pc: -pc.quality,  # negate so min() picks highest quality
}


def _get_scorer(policy: str) -> Callable[[PlanCost], float]:
    """Return a scorer function for *policy*.

    Requires:
        - *policy* is one of ``"min_cost"``, ``"min_time"``, or
          ``"max_quality"``.

    Returns:
        A callable ``(PlanCost) -> float`` where lower is better.

    Raises:
        ValueError: If *policy* is not recognised.
    """
    if policy not in _SCORERS:
        raise ValueError(
            f"Unknown policy '{policy}'. Choose from: {sorted(_SCORERS)}"
        )
    return _SCORERS[policy]


def _expr_to_plan_node(
    expr: Expression,
    node_id: str,
    parent_node_ids: list[str],
) -> PlanNode:
    """Convert an optimizer ``Expression`` into a ``PlanNode``.

    Scan expressions become ``node_type="dataset"``; all other physical
    expressions become ``node_type="operator"``.

    Requires:
        - *expr* is a ``PhysicalExpression`` (has a ``PhysicalOperator``).

    Returns:
        A ``PlanNode`` with the given *node_id* and *parent_node_ids*.

    Raises:
        None.
    """
    op = expr.operator

    if isinstance(op, ScanOp):
        return PlanNode(
            node_id=node_id,
            node_type="dataset",
            operator_type=None,
            name=op.dataset_id,
            description=f"Load dataset: {op.dataset_id}",
            params={"est_total_tokens": op.est_total_tokens},
            parent_ids=list(parent_node_ids),
            dataset_id=op.dataset_id,
        )

    # Determine operator_type from the logical_op_class_name stored on the physical operator.
    operator_type = getattr(op, "logical_op_class_name", None) or op.op_name()
    description = getattr(op, "task", "") or op.op_name()

    return PlanNode(
        node_id=node_id,
        node_type="operator",
        operator_type=operator_type,
        name=getattr(op, "dataset_id", ""),
        description=description,
        params=op.get_op_params(),
        parent_ids=list(parent_node_ids),
        dataset_id=getattr(op, "dataset_id", ""),
    )
