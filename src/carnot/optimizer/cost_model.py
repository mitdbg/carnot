from carnot.core.models import PlanCost
from carnot.operators.physical import PhysicalOperator
from carnot.operators.scan import ScanOp


class CostModel:
    """Base class for cost models used by the optimizer.

    Every ``CostModel`` must implement ``__call__`` which receives a
    ``PhysicalOperator`` and the ``PlanCost``(s) of its input(s), then
    returns the cumulative ``PlanCost`` for the sub-plan rooted at that
    operator.

    The base class provides a default implementation for ``ScanOp``
    (the leaf / base case) so subclasses only need to handle non-scan
    operators.

    Representation invariant:
        None (stateless base class).

    Abstraction function:
        Represents a strategy for predicting the runtime, dollar cost,
        and quality of a physical operator given its input plan costs.
    """

    def __init__(self):
        """Construct a ``CostModel``.

        Subclass constructors may accept additional arguments depending
        on whether they rely on historical execution data, learned
        models, or analytical formulas.

        Requires:
            None.

        Returns:
            A new ``CostModel`` instance.

        Raises:
            None.
        """
        pass

    def __call__(
        self,
        operator: PhysicalOperator,
        input_plan_cost: PlanCost | None = None,
        right_input_plan_cost: PlanCost | None = None,
    ) -> PlanCost:
        """Estimate the cost of *operator* given its input plan cost(s).

        For ``ScanOp`` operators the base class returns a zero-cost,
        zero-time ``PlanCost`` whose token fields are seeded from the
        scan's estimated total tokens.  All other operators delegate to
        ``_estimate``, which subclasses must override.

        Requires:
            - *operator* is a concrete ``PhysicalOperator``.
            - For unary operators, *input_plan_cost* is a ``PlanCost``.
            - For binary (join) operators, both *input_plan_cost* and
              *right_input_plan_cost* are ``PlanCost`` objects.
            - For leaf (scan) operators, both may be ``None``.

        Returns:
            A ``PlanCost`` representing the cumulative cost of the
            sub-plan rooted at *operator*.

        Raises:
            NotImplementedError: If *operator* is not a ``ScanOp`` and
            the subclass has not overridden ``_estimate``.
        """
        if isinstance(operator, ScanOp):
            return self._estimate_scan(operator)

        return self._estimate(operator, input_plan_cost, right_input_plan_cost)

    @staticmethod
    def _estimate_scan(operator: ScanOp) -> PlanCost:
        """Return the base-case ``PlanCost`` for a scan operator.

        A scan is lazy — it has zero dollar cost and zero execution
        time.  Its token fields seed the quality metric for downstream
        operators: ``total_input_tokens`` equals the estimated total
        tokens in the dataset, and ``total_scanned_input_tokens`` is
        the same value (a scan conceptually "reads everything").

        Requires:
            - *operator* is a ``ScanOp``.

        Returns:
            A ``PlanCost`` with ``cost=0``, ``time=0``,
            ``total_input_tokens = operator.est_total_tokens``,
            ``total_scanned_input_tokens = operator.est_total_tokens``,
            and per-operator fields all zero.

        Raises:
            None.
        """
        return PlanCost(
            cost=0.0,
            time=0.0,
            total_input_tokens=operator.est_total_tokens,
            total_scanned_input_tokens=operator.est_total_tokens,
            cardinality=float(operator.num_items),
            cost_per_record=0.0,
            time_per_record=0.0,
        )

    def _estimate(
        self,
        operator: PhysicalOperator,
        input_plan_cost: PlanCost | None = None,
        right_input_plan_cost: PlanCost | None = None,
    ) -> PlanCost:
        """Estimate the cost for a non-scan operator.

        Subclasses should override this method with model-specific
        logic (analytical formulas, learned models, etc.).

        Requires:
            - *operator* is a concrete ``PhysicalOperator`` that is
              **not** a ``ScanOp``.

        Returns:
            A ``PlanCost`` for the sub-plan rooted at *operator*.

        Raises:
            NotImplementedError: Always in the base class.
        """
        raise NotImplementedError("Subclasses must override _estimate for non-scan operators.")
