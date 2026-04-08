from carnot.core.models import PlanCost
from carnot.operators.physical import PhysicalOperator
from carnot.operators.scan import ScanOp
from carnot.operators.sem_agg import SemAggOperator
from carnot.operators.sem_filter import SemFilterOperator
from carnot.operators.sem_flat_map import SemFlatMapOperator
from carnot.operators.sem_groupby import SemGroupByOperator
from carnot.operators.sem_join import SemJoinOperator
from carnot.operators.sem_map import SemMapOperator
from carnot.operators.sem_topk import SemTopKOperator
from carnot.optimizer.pricing import ModelPricing, ModelPricingLookup

# ── Selectivity defaults ──────────────────────────────────────────────
DEFAULT_FILTER_SELECTIVITY = 0.5
DEFAULT_JOIN_SELECTIVITY = 0.1
DEFAULT_FLATMAP_FAN_OUT = 3.0

# ── Output token estimates (per LLM call) ─────────────────────────────
DEFAULT_FILTER_OUTPUT_TOKENS = 10
DEFAULT_MAP_OUTPUT_TOKENS = 100
DEFAULT_FLATMAP_OUTPUT_TOKENS = 200
DEFAULT_AGG_OUTPUT_TOKENS = 200
DEFAULT_GROUPBY_OUTPUT_TOKENS = 100
DEFAULT_JOIN_OUTPUT_TOKENS = 10

# ── GroupBy-specific ──────────────────────────────────────────────────
DEFAULT_GROUPBY_NUM_GROUPS = 5

# ── Time-per-record (seconds) ────────────────────────────────────────
DEFAULT_TIME_PER_RECORD = 0.5
DEFAULT_EMBEDDING_TIME_PER_RECORD = 0.05

# ── Fallback pricing (gpt-4o-mini rates) ─────────────────────────────
DEFAULT_PRICING = ModelPricing(
    input_cost_per_token=1.5e-07,
    output_cost_per_token=6e-07,
    max_input_tokens=128_000,
    max_output_tokens=16_384,
)


class CostModel:
    """Analytical cost model for Carnot's cascades-style optimizer.

    Estimates dollar cost, wall-clock time, and token-quality metrics
    for every physical operator.  Uses ``ModelPricingLookup`` to
    resolve per-token prices and applies operator-specific formulas
    to produce a cumulative ``PlanCost`` for each sub-plan.

    Representation invariant:
        - ``pricing`` is a ``ModelPricingLookup`` instance.

    Abstraction function:
        Represents an analytical strategy for predicting the runtime,
        dollar cost, and quality of a physical operator given its
        input plan costs and model pricing data.
    """

    def __init__(self, pricing: ModelPricingLookup | None = None):
        """Construct a ``CostModel``.

        Requires:
            - *pricing*, if provided, is a ``ModelPricingLookup``.

        Returns:
            A new ``CostModel`` with the given (or default) pricing
            lookup.

        Raises:
            None.
        """
        self.pricing = pricing or ModelPricingLookup(default_pricing=DEFAULT_PRICING)

    def __call__(
        self,
        operator: PhysicalOperator,
        input_plan_cost: PlanCost | None = None,
        right_input_plan_cost: PlanCost | None = None,
    ) -> PlanCost:
        """Estimate the cost of *operator* given its input plan cost(s).

        Dispatches to the appropriate ``_estimate_*`` method based on
        operator type.

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
            NotImplementedError: If the operator type is not recognized.
        """
        if isinstance(operator, ScanOp):
            return self._estimate_scan(operator)
        elif isinstance(operator, SemFilterOperator):
            return self._estimate_filter(operator, input_plan_cost)
        elif isinstance(operator, SemMapOperator):
            return self._estimate_map(operator, input_plan_cost)
        elif isinstance(operator, SemFlatMapOperator):
            return self._estimate_flatmap(operator, input_plan_cost)
        elif isinstance(operator, SemJoinOperator):
            return self._estimate_join(operator, input_plan_cost, right_input_plan_cost)
        elif isinstance(operator, SemAggOperator):
            return self._estimate_agg(operator, input_plan_cost)
        elif isinstance(operator, SemGroupByOperator):
            return self._estimate_groupby(operator, input_plan_cost)
        elif isinstance(operator, SemTopKOperator):
            return self._estimate_topk(operator, input_plan_cost)

        raise NotImplementedError(f"CostModel does not handle {type(operator).__name__}")

    # ── Leaf operator ─────────────────────────────────────────────────

    @staticmethod
    def _estimate_scan(operator: ScanOp) -> PlanCost:
        """Return the base-case ``PlanCost`` for a scan operator.

        A scan is lazy — zero dollar cost and zero execution time.
        Its token fields seed the quality metric for downstream
        operators.

        Requires:
            - *operator* is a ``ScanOp``.

        Returns:
            A ``PlanCost`` with ``cost=0``, ``time=0``, cardinality
            seeded from the scan's item count, and token fields from
            ``est_tokens_per_item``.

        Raises:
            None.
        """
        n = float(operator.num_items)
        est_tpr = float(operator.est_tokens_per_item)
        total_tokens = n * est_tpr
        return PlanCost(
            cost=0.0,
            time=0.0,
            total_input_tokens=total_tokens,
            total_scanned_input_tokens=total_tokens,
            input_cardinality=n,
            output_cardinality=n,
            selectivity=1.0,
            avg_tokens_per_record=est_tpr,
            cost_per_record=0.0,
            time_per_record=0.0,
        )

    # ── Unary semantic operators ──────────────────────────────────────

    def _estimate_filter(self, operator: SemFilterOperator, input_plan_cost: PlanCost) -> PlanCost:
        """Estimate the cost of a semantic filter (one LLM call per record).

        Requires:
            - *operator* is a ``SemFilterOperator`` with a ``model_id``.
            - *input_plan_cost* is a valid ``PlanCost``.

        Returns:
            A ``PlanCost`` with selectivity ``DEFAULT_FILTER_SELECTIVITY``,
            unchanged ``avg_tokens_per_record``, and cumulative cost/time/tokens.

        Raises:
            None.
        """
        pricing = self.pricing.get(operator.model_id)
        n = input_plan_cost.output_cardinality
        t_in = input_plan_cost.avg_tokens_per_record

        cost_per_record = t_in * pricing.input_cost_per_token + DEFAULT_FILTER_OUTPUT_TOKENS * pricing.output_cost_per_token
        time_per_record = DEFAULT_TIME_PER_RECORD
        selectivity = DEFAULT_FILTER_SELECTIVITY

        return PlanCost(
            cost=cost_per_record * n + input_plan_cost.cost,
            time=time_per_record * n + input_plan_cost.time,
            total_input_tokens=n * t_in + input_plan_cost.total_input_tokens,
            total_scanned_input_tokens=n * t_in + input_plan_cost.total_scanned_input_tokens,
            input_cardinality=n,
            output_cardinality=n * selectivity,
            selectivity=selectivity,
            avg_tokens_per_record=t_in,
            cost_per_record=cost_per_record,
            time_per_record=time_per_record,
        )

    def _estimate_map(self, operator: SemMapOperator, input_plan_cost: PlanCost) -> PlanCost:
        """Estimate the cost of a semantic map (one LLM call per record).

        Requires:
            - *operator* is a ``SemMapOperator`` with a ``model_id``.
            - *input_plan_cost* is a valid ``PlanCost``.

        Returns:
            A ``PlanCost`` with selectivity 1.0, enriched
            ``avg_tokens_per_record``, and cumulative cost/time/tokens.

        Raises:
            None.
        """
        pricing = self.pricing.get(operator.model_id)
        n = input_plan_cost.output_cardinality
        t_in = input_plan_cost.avg_tokens_per_record

        cost_per_record = t_in * pricing.input_cost_per_token + DEFAULT_MAP_OUTPUT_TOKENS * pricing.output_cost_per_token
        time_per_record = DEFAULT_TIME_PER_RECORD

        return PlanCost(
            cost=cost_per_record * n + input_plan_cost.cost,
            time=time_per_record * n + input_plan_cost.time,
            total_input_tokens=n * t_in + input_plan_cost.total_input_tokens,
            total_scanned_input_tokens=n * t_in + input_plan_cost.total_scanned_input_tokens,
            input_cardinality=n,
            output_cardinality=n,
            selectivity=1.0,
            avg_tokens_per_record=t_in + DEFAULT_MAP_OUTPUT_TOKENS,
            cost_per_record=cost_per_record,
            time_per_record=time_per_record,
        )

    def _estimate_flatmap(self, operator: SemFlatMapOperator, input_plan_cost: PlanCost) -> PlanCost:
        """Estimate the cost of a semantic flat-map (one LLM call per record, fan-out output).

        Requires:
            - *operator* is a ``SemFlatMapOperator`` with a ``model_id``.
            - *input_plan_cost* is a valid ``PlanCost``.

        Returns:
            A ``PlanCost`` with selectivity ``DEFAULT_FLATMAP_FAN_OUT``,
            output-only ``avg_tokens_per_record``, and cumulative cost/time/tokens.

        Raises:
            None.
        """
        pricing = self.pricing.get(operator.model_id)
        n = input_plan_cost.output_cardinality
        t_in = input_plan_cost.avg_tokens_per_record

        cost_per_record = (
            t_in * pricing.input_cost_per_token + DEFAULT_FLATMAP_OUTPUT_TOKENS * pricing.output_cost_per_token
        )
        time_per_record = DEFAULT_TIME_PER_RECORD

        return PlanCost(
            cost=cost_per_record * n + input_plan_cost.cost,
            time=time_per_record * n + input_plan_cost.time,
            total_input_tokens=n * t_in + input_plan_cost.total_input_tokens,
            total_scanned_input_tokens=n * t_in + input_plan_cost.total_scanned_input_tokens,
            input_cardinality=n,
            output_cardinality=n * DEFAULT_FLATMAP_FAN_OUT,
            selectivity=DEFAULT_FLATMAP_FAN_OUT,
            avg_tokens_per_record=DEFAULT_FLATMAP_OUTPUT_TOKENS,
            cost_per_record=cost_per_record,
            time_per_record=time_per_record,
        )

    # ── Binary operator ───────────────────────────────────────────────

    def _estimate_join(
        self,
        operator: SemJoinOperator,
        left_plan_cost: PlanCost,
        right_plan_cost: PlanCost,
    ) -> PlanCost:
        """Estimate the cost of a semantic join (one LLM call per cross-product pair).

        Requires:
            - *operator* is a ``SemJoinOperator`` with a ``model_id``.
            - *left_plan_cost* and *right_plan_cost* are valid ``PlanCost`` objects.

        Returns:
            A ``PlanCost`` with input cardinality ``N_L × N_R``, selectivity
            ``DEFAULT_JOIN_SELECTIVITY``, and cumulative cost/time/tokens from both inputs.

        Raises:
            None.
        """
        pricing = self.pricing.get(operator.model_id)
        n_l = left_plan_cost.output_cardinality
        n_r = right_plan_cost.output_cardinality
        t_l = left_plan_cost.avg_tokens_per_record
        t_r = right_plan_cost.avg_tokens_per_record

        cross = n_l * n_r
        t_pair = t_l + t_r
        cost_per_record = t_pair * pricing.input_cost_per_token + DEFAULT_JOIN_OUTPUT_TOKENS * pricing.output_cost_per_token
        time_per_record = DEFAULT_TIME_PER_RECORD
        selectivity = DEFAULT_JOIN_SELECTIVITY

        return PlanCost(
            cost=cost_per_record * cross + left_plan_cost.cost + right_plan_cost.cost,
            time=time_per_record * cross + left_plan_cost.time + right_plan_cost.time,
            total_input_tokens=(
                cross * t_pair + left_plan_cost.total_input_tokens + right_plan_cost.total_input_tokens
            ),
            total_scanned_input_tokens=(
                cross * t_pair
                + left_plan_cost.total_scanned_input_tokens
                + right_plan_cost.total_scanned_input_tokens
            ),
            input_cardinality=cross,
            output_cardinality=cross * selectivity,
            selectivity=selectivity,
            avg_tokens_per_record=t_pair,
            cost_per_record=cost_per_record,
            time_per_record=time_per_record,
        )

    # ── Aggregation operators ─────────────────────────────────────────

    def _estimate_agg(self, operator: SemAggOperator, input_plan_cost: PlanCost) -> PlanCost:
        """Estimate the cost of a semantic aggregation (single LLM call over all items).

        Requires:
            - *operator* is a ``SemAggOperator`` with a ``model_id``.
            - *input_plan_cost* is a valid ``PlanCost``.

        Returns:
            A ``PlanCost`` with ``output_cardinality=1``, and cost reflecting
            a single LLM call that reads all input tokens.

        Raises:
            None.
        """
        pricing = self.pricing.get(operator.model_id)
        n = input_plan_cost.output_cardinality
        t_in = input_plan_cost.avg_tokens_per_record

        # Single LLM call that reads all N records
        cost_per_record = n * t_in * pricing.input_cost_per_token + DEFAULT_AGG_OUTPUT_TOKENS * pricing.output_cost_per_token
        time_per_record = DEFAULT_TIME_PER_RECORD

        return PlanCost(
            cost=cost_per_record + input_plan_cost.cost,
            time=time_per_record + input_plan_cost.time,
            total_input_tokens=n * t_in + input_plan_cost.total_input_tokens,
            total_scanned_input_tokens=n * t_in + input_plan_cost.total_scanned_input_tokens,
            input_cardinality=n,
            output_cardinality=1.0,
            selectivity=1.0 / n if n > 0 else 1.0,
            avg_tokens_per_record=DEFAULT_AGG_OUTPUT_TOKENS,
            cost_per_record=cost_per_record,
            time_per_record=time_per_record,
        )

    def _estimate_groupby(self, operator: SemGroupByOperator, input_plan_cost: PlanCost) -> PlanCost:
        """Estimate the cost of a semantic group-by (N grouping calls + G aggregation calls).

        Requires:
            - *operator* is a ``SemGroupByOperator`` with a ``model_id``.
            - *input_plan_cost* is a valid ``PlanCost``.

        Returns:
            A ``PlanCost`` with ``output_cardinality=G`` (estimated groups),
            and cost reflecting two phases: grouping (N calls) and
            per-group aggregation (G calls).

        Raises:
            None.
        """
        pricing = self.pricing.get(operator.model_id)
        n = input_plan_cost.output_cardinality
        t_in = input_plan_cost.avg_tokens_per_record
        g = float(DEFAULT_GROUPBY_NUM_GROUPS)

        # Phase 1: N grouping calls (one per record)
        grouping_cost = n * (t_in * pricing.input_cost_per_token + DEFAULT_GROUPBY_OUTPUT_TOKENS * pricing.output_cost_per_token)

        # Phase 2: G aggregation calls (each reads ~N/G records)
        records_per_group = n / g if g > 0 else n
        agg_cost = g * (
            records_per_group * t_in * pricing.input_cost_per_token
            + DEFAULT_GROUPBY_OUTPUT_TOKENS * pricing.output_cost_per_token
        )
        total_cost = grouping_cost + agg_cost + input_plan_cost.cost
        total_time = (n + g) * DEFAULT_TIME_PER_RECORD + input_plan_cost.time

        # cost_per_record is the average cost per LLM call across both phases
        total_calls = n + g
        cost_per_record = (grouping_cost + agg_cost) / total_calls if total_calls > 0 else 0.0

        return PlanCost(
            cost=total_cost,
            time=total_time,
            total_input_tokens=n * t_in + input_plan_cost.total_input_tokens,
            total_scanned_input_tokens=n * t_in + input_plan_cost.total_scanned_input_tokens,
            input_cardinality=n,
            output_cardinality=g,
            selectivity=g / n if n > 0 else 1.0,
            avg_tokens_per_record=DEFAULT_GROUPBY_OUTPUT_TOKENS,
            cost_per_record=cost_per_record,
            time_per_record=DEFAULT_TIME_PER_RECORD,
        )

    # ── Embedding-based operator ──────────────────────────────────────

    def _estimate_topk(self, operator: SemTopKOperator, input_plan_cost: PlanCost) -> PlanCost:
        """Estimate the cost of a semantic top-k (embedding-based ranking).

        Uses the embedding model price (not completion price).  Only
        ``k`` out of ``N`` records are "attended to", so
        ``total_scanned_input_tokens`` grows by ``k × T_in`` while
        ``total_input_tokens`` grows by ``N × T_in``.

        Requires:
            - *operator* is a ``SemTopKOperator`` with ``model_id`` and ``k``.
            - *input_plan_cost* is a valid ``PlanCost``.

        Returns:
            A ``PlanCost`` with ``output_cardinality=k``, embedding-based
            cost, and reduced ``total_scanned_input_tokens``.

        Raises:
            None.
        """
        pricing = self.pricing.get(operator.model_id)
        n = input_plan_cost.output_cardinality
        t_in = input_plan_cost.avg_tokens_per_record
        k = float(operator.k)

        # Embedding cost: all N records are embedded
        cost_per_record = t_in * pricing.input_cost_per_token
        time_per_record = DEFAULT_EMBEDDING_TIME_PER_RECORD

        return PlanCost(
            cost=cost_per_record * n + input_plan_cost.cost,
            time=time_per_record * n + input_plan_cost.time,
            total_input_tokens=n * t_in + input_plan_cost.total_input_tokens,
            total_scanned_input_tokens=k * t_in + input_plan_cost.total_scanned_input_tokens,
            input_cardinality=n,
            output_cardinality=k,
            selectivity=k / n if n > 0 else 1.0,
            avg_tokens_per_record=t_in,
            cost_per_record=cost_per_record,
            time_per_record=time_per_record,
        )
