from copy import deepcopy

from carnot.operators.code import CodeOperator
from carnot.operators.limit import LimitOperator
from carnot.operators.logical import (
    Aggregate,
    Code,
    Filter,
    FlatMap,
    GroupBy,
    Join,
    Limit,
    Map,
    Reason,
    Scan,
    TopK,
)
from carnot.operators.reasoning import ReasoningOperator
from carnot.operators.scan import ScanOp
from carnot.operators.sem_agg import SemAggOperator
from carnot.operators.sem_filter import SemFilterOperator
from carnot.operators.sem_flat_map import SemFlatMapOperator
from carnot.operators.sem_groupby import SemGroupByOperator
from carnot.operators.sem_join import SemJoinOperator
from carnot.operators.sem_map import SemMapOperator
from carnot.operators.sem_topk import SemTopKOperator
from carnot.optimizer.primitives import Expression, Group, LogicalExpression, PhysicalExpression


class Rule:
    """
    The abstract base class for transformation and implementation rules.
    """

    @classmethod
    def get_rule_id(cls):
        return cls.__name__

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression, **kwargs) -> bool:
        raise NotImplementedError("Calling this method from an abstract base class.")

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **kwargs: dict) -> set[Expression]:
        raise NotImplementedError("Calling this method from an abstract base class.")


class TransformationRule(Rule):
    """
    Base class for transformation rules which convert a logical expression to another logical expression.
    The substitute method for a TransformationRule should return all new expressions, all new groups,
    and the updated next_group_id counter.
    """

    @classmethod
    def is_exploration_rule(cls) -> bool:
        """Returns True if this rule is an exploration rule and False otherwise. Default is False."""
        return False

    @classmethod
    def substitute(
        cls,
        logical_expression: LogicalExpression,
        groups: dict[int, Group],
        expressions: dict[int, Expression],
        next_group_id: int = 0,
        **kwargs,
    ) -> tuple[set[LogicalExpression], set[Group], int]:
        """
        This function applies the transformation rule to the logical expression, which
        potentially creates new intermediate expressions and groups.

        The function returns a tuple containing:
        - the set of all new logical expressions created when applying the transformation
        - the set of all new groups created when applying the transformation
        - the next group id (after creating any new groups)
        """
        raise NotImplementedError("Calling this method from an abstract base class.")


class PushDownFilter(TransformationRule):
    """Push a Filter below another Filter in the input group.

    Filters are independent of one another (no field dependencies to track),
    so any filter can safely be pushed past any other filter.  This is the
    only swap allowed — we do *not* push filters past converts or joins
    because that would require tracking field dependencies which Carnot
    does not yet support.

    Representation invariant:
        - ``next_group_id`` is always ≥ the id of every group in *groups*
          at the time of the call.

    Abstraction function:
        Produces an equivalent plan where *logical_expression* (a Filter)
        is evaluated below a sibling Filter in one of its input groups,
        potentially reducing the number of rows processed by the more
        expensive operator above it.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: Expression, **kwargs) -> bool:
        return isinstance(logical_expression.operator, Filter)

    @classmethod
    def substitute(
        cls,
        logical_expression: LogicalExpression,
        groups: dict[int, Group],
        expressions: dict[int, Expression],
        next_group_id: int = 0,
        **kwargs,
    ) -> tuple[set[LogicalExpression], set[Group], int]:
        """Push *logical_expression* (a Filter) below sibling Filters in its input groups.

        Requires:
            - ``logical_expression.operator`` is a ``Filter``.
            - ``next_group_id`` is greater than the id of every group currently
              in *groups*.

        Returns:
            A 3-tuple ``(new_logical_expressions, new_groups, next_group_id)``
            where *new_logical_expressions* contains every newly created
            ``LogicalExpression``, *new_groups* contains every newly created
            ``Group``, and *next_group_id* is the updated monotonic counter
            (incremented once per new group).

        Raises:
            None.
        """
        new_logical_expressions: set[LogicalExpression] = set()
        new_groups: set[Group] = set()

        filter_operator: Filter = logical_expression.operator

        for input_group_id in logical_expression.input_group_ids:
            input_group = groups[input_group_id]

            for expr in input_group.logical_expressions:
                # Only push filters past other filters (no dependency tracking needed)
                if not isinstance(expr.operator, Filter):
                    continue

                # Create a new logical expression with the filter pushed below the sibling filter
                new_input_group_ids = deepcopy(expr.input_group_ids)
                new_filter_expr = LogicalExpression(
                    filter_operator,
                    input_group_ids=new_input_group_ids,
                    group_id=None,
                )
                # Propagate applied rules from the original expression so that
                # rules like FilterToTopKFilter are not re-applied to reorderings.
                for rule_id in logical_expression.rules_applied:
                    new_filter_expr.rules_applied.add(rule_id)

                new_logical_expressions.add(new_filter_expr)

                # If this expression already exists, reuse its group
                if new_filter_expr.expr_id in expressions:
                    group_id = expressions[new_filter_expr.expr_id].group_id
                    new_filter_expr.set_group_id(group_id)
                else:
                    # Assign a fresh monotonic group id for the new intermediate group
                    group_id = next_group_id
                    next_group_id += 1

                    new_filter_expr.set_group_id(group_id)
                    group = Group(logical_expressions=[new_filter_expr], group_id=group_id)

                    # Register the new group
                    groups[group_id] = group
                    new_groups.add(group)

                # Create the pulled-up expression: the sibling filter now sits on top,
                # reading from the new intermediate group that contains our pushed-down filter
                remaining_input_ids = [
                    g_id for g_id in logical_expression.input_group_ids if g_id != input_group_id
                ]
                new_expr = LogicalExpression(
                    expr.operator.copy(),
                    input_group_ids=[group_id] + remaining_input_ids,
                    group_id=logical_expression.group_id,
                )
                # Propagate applied rules from the sibling expression.
                for rule_id in expr.rules_applied:
                    new_expr.rules_applied.add(rule_id)
                new_logical_expressions.add(new_expr)

        return new_logical_expressions, new_groups, next_group_id


class FilterToTopKFilter(TransformationRule):
    """Insert a TopK pre-filter below a Filter to trade quality for speed.

    For each input group of a Filter expression, this rule creates
    alternative plans of the form
    ``Filter(TopK(k, task=filter_text))`` for each fixed *k* in
    ``k_values``.  The TopK uses embedding similarity to cheaply
    narrow the candidate set before the expensive LLM-based filter
    runs.

    Multiple *k* values are explored so the optimizer's Pareto search
    can evaluate different quality-cost trade-offs.

    Representation invariant:
        - ``k_values`` is a non-empty list of positive integers,
          sorted in ascending order.
        - ``next_group_id`` is always ≥ the id of every group in
          *groups* at the time of the call.

    Abstraction function:
        For a given ``Filter(input_group)`` expression, produces a set
        of equivalent ``Filter(TopK_group)`` alternatives where
        ``TopK_group`` contains a ``TopK`` operator reading from the
        original ``input_group``.
    """

    k_values: list[int] = [10, 50, 100, 500, 1000]

    @classmethod
    def _has_openai_key(cls, llm_config: dict | None) -> bool:
        return bool(llm_config and llm_config.get("OPENAI_API_KEY"))

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression, **kwargs) -> bool:
        if not isinstance(logical_expression.operator, Filter):
            return False
        llm_config = kwargs.get("llm_config")
        return cls._has_openai_key(llm_config)

    @classmethod
    def substitute(
        cls,
        logical_expression: LogicalExpression,
        groups: dict[int, Group],
        expressions: dict[int, Expression],
        next_group_id: int = 0,
        **kwargs,
    ) -> tuple[set[LogicalExpression], set[Group], int]:
        """Create ``TopK → Filter`` alternatives at fixed *k* values.

        For each input group of the Filter and for each *k* in
        ``k_values``, creates:

        1. A new ``TopK`` logical expression reading from the original
           input group.
        2. A new intermediate group containing that ``TopK``.
        3. A copy of the ``Filter`` expression reading from the new
           intermediate group.

        Requires:
            - ``logical_expression.operator`` is a ``Filter``.
            - ``next_group_id`` > every group id currently in *groups*.

        Returns:
            A 3-tuple ``(new_logical_expressions, new_groups,
            next_group_id)`` following the ``TransformationRule``
            contract.

        Raises:
            None.
        """
        new_logical_expressions: set[LogicalExpression] = set()
        new_groups: set[Group] = set()

        filter_operator: Filter = logical_expression.operator

        for input_group_id in logical_expression.input_group_ids:
            for k in cls.k_values:
                # Create a TopK logical operator that uses the filter
                # text as the semantic search query
                topk_op = TopK(
                    name=f"{filter_operator.name}_topk_{k}",
                    task=filter_operator.filter,
                    k=k,
                )

                topk_expr = LogicalExpression(
                    topk_op,
                    input_group_ids=[input_group_id],
                    group_id=None,
                )

                # If this TopK expression already exists, reuse its group
                if topk_expr.expr_id in expressions:
                    topk_group_id = expressions[topk_expr.expr_id].group_id
                    topk_expr.set_group_id(topk_group_id)
                else:
                    topk_group_id = next_group_id
                    next_group_id += 1

                    topk_expr.set_group_id(topk_group_id)
                    topk_group = Group(logical_expressions=[topk_expr], group_id=topk_group_id)

                    groups[topk_group_id] = topk_group
                    new_groups.add(topk_group)

                new_logical_expressions.add(topk_expr)

                # Create a Filter reading from the TopK group instead of
                # the original input group; add this rule to the applied rules of the new Filter expression
                # so that the optimizer does not apply this same transformation again on the new Filter expression
                remaining_input_ids = [
                    g_id for g_id in logical_expression.input_group_ids if g_id != input_group_id
                ]
                new_filter_expr = LogicalExpression(
                    filter_operator.copy(),
                    input_group_ids=[topk_group_id] + remaining_input_ids,
                    group_id=logical_expression.group_id,
                )
                new_filter_expr.add_applied_rule(cls)
                new_logical_expressions.add(new_filter_expr)

        return new_logical_expressions, new_groups, next_group_id


class ImplementationRule(Rule):
    """Base class for implementation rules which convert a logical expression to a physical expression.

    Subclasses must implement ``matches_pattern`` and ``substitute``.  Each
    ``substitute`` method is responsible for translating logical-operator
    parameters to the physical-operator constructor kwargs and passing them
    to ``_perform_substitution``.
    """

    @classmethod
    def _perform_substitution(
        cls,
        logical_expression: LogicalExpression,
        physical_op_class: type,
        fixed_op_kwargs: dict,
        variable_op_kwargs: list[dict] | dict | None = None,
    ) -> set[PhysicalExpression]:
        """Create physical expressions by instantiating *physical_op_class* with the given kwargs.

        For each dict in *variable_op_kwargs* the dict is merged with
        *fixed_op_kwargs* and the combined kwargs are passed to the physical
        operator constructor.

        Requires:
            - *fixed_op_kwargs* contains every keyword argument that is
              constant across all physical alternatives for this logical
              expression.
            - Each entry in *variable_op_kwargs* provides the keyword
              arguments that differ between physical alternatives
              (e.g. ``model_id``).

        Returns:
            The unique set of ``PhysicalExpression`` instances produced.

        Raises:
            TypeError: If *physical_op_class* cannot be constructed with
            the merged kwargs.
        """
        if variable_op_kwargs is None:
            variable_op_kwargs = [{}]
        elif isinstance(variable_op_kwargs, dict):
            variable_op_kwargs = [variable_op_kwargs]

        physical_expressions = []
        for var_op_kwargs in variable_op_kwargs:
            op_kwargs = {**fixed_op_kwargs, **var_op_kwargs}
            op = physical_op_class(**op_kwargs)
            expression = PhysicalExpression.from_op_and_logical_expr(op, logical_expression)
            physical_expressions.append(expression)

        return set(physical_expressions)


class CodeRule(ImplementationRule):
    """Substitute a logical ``Code`` expression with a ``CodeOperator`` physical implementation."""

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression, **kwargs) -> bool:
        return isinstance(logical_expression.operator, Code)

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logical_op: Code = logical_expression.operator
        fixed_op_kwargs = {
            "task": logical_op.task,
            "dataset_id": logical_op.name,
            "llm_config": runtime_kwargs["llm_config"],
            "logical_op_id": logical_op.get_logical_op_id(),
            "logical_op_class_name": logical_op.logical_op_class_name(),
        }
        variable_op_kwargs = [{"model_id": model_id} for model_id in runtime_kwargs["available_model_ids"]]
        return cls._perform_substitution(logical_expression, CodeOperator, fixed_op_kwargs, variable_op_kwargs)


class ReasoningRule(ImplementationRule):
    """Substitute a logical ``Reason`` expression with a ``ReasoningOperator`` physical implementation."""

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression, **kwargs) -> bool:
        return isinstance(logical_expression.operator, Reason)

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logical_op: Reason = logical_expression.operator
        fixed_op_kwargs = {
            "task": logical_op.task,
            "dataset_id": logical_op.name,
            "llm_config": runtime_kwargs["llm_config"],
            "logical_op_id": logical_op.get_logical_op_id(),
            "logical_op_class_name": logical_op.logical_op_class_name(),
        }
        variable_op_kwargs = [{"model_id": model_id} for model_id in runtime_kwargs["available_model_ids"]]
        return cls._perform_substitution(logical_expression, ReasoningOperator, fixed_op_kwargs, variable_op_kwargs)


# class MapRule(ImplementationRule):
#     """
#     Substitute a logical expression for a non-semantic map with its physical implementation.
#     """

#     @classmethod
#     def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
#         is_match = isinstance(logical_expression.operator, Map) and logical_expression.operator.udf is not None
#         return is_match

#     @classmethod
#     def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
#         return cls._perform_substitution(logical_expression, Map, runtime_kwargs)


class SemMapRule(ImplementationRule):
    """
    Substitute a logical expression for a semantic map with its physical implementation.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression, **kwargs) -> bool:
        return isinstance(logical_expression.operator, Map) # and logical_expression.operator.udf is None

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logical_op: Map = logical_expression.operator
        fixed_op_kwargs = {
            "task": "Execute the map operation to compute the following output field(s).",
            "output_fields": logical_op.fields,
            "dataset_id": logical_op.name,
            "llm_config": runtime_kwargs["llm_config"],
            "max_workers": runtime_kwargs["max_workers"],
            "logical_op_id": logical_op.get_logical_op_id(),
            "logical_op_class_name": logical_op.logical_op_class_name(),
        }
        variable_op_kwargs = [{"model_id": model_id} for model_id in runtime_kwargs["available_model_ids"]]
        return cls._perform_substitution(logical_expression, SemMapOperator, fixed_op_kwargs, variable_op_kwargs)


class SemFlatMapRule(ImplementationRule):
    """
    Substitute a logical expression for a semantic flat map with its physical implementation.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression, **kwargs) -> bool:
        return isinstance(logical_expression.operator, FlatMap) # and logical_expression.operator.udf is None

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logical_op: FlatMap = logical_expression.operator
        fixed_op_kwargs = {
            "task": "Execute the flat map operation to compute the following output field(s).",
            "output_fields": logical_op.fields,
            "dataset_id": logical_op.name,
            "llm_config": runtime_kwargs["llm_config"],
            "max_workers": runtime_kwargs["max_workers"],
            "logical_op_id": logical_op.get_logical_op_id(),
            "logical_op_class_name": logical_op.logical_op_class_name(),
        }
        variable_op_kwargs = [{"model_id": model_id} for model_id in runtime_kwargs["available_model_ids"]]
        return cls._perform_substitution(logical_expression, SemFlatMapOperator, fixed_op_kwargs, variable_op_kwargs)


class SemTopKRule(ImplementationRule):
    """Substitute a logical ``TopK`` expression with a ``SemTopKOperator`` physical implementation."""

    k_budgets = [1, 5, 10, 25, 100]

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression, **kwargs) -> bool:
        if not isinstance(logical_expression.operator, TopK):
            return False
        llm_config = kwargs.get("llm_config")
        return bool(llm_config and llm_config.get("OPENAI_API_KEY"))

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logical_op: TopK = logical_expression.operator
        fixed_op_kwargs = {
            "task": logical_op.task,
            "dataset_id": logical_op.name,
            "llm_config": runtime_kwargs["llm_config"],
            "max_workers": runtime_kwargs["max_workers"],
            "index_name": logical_op.index_name,
            "logical_op_id": logical_op.get_logical_op_id(),
            "logical_op_class_name": logical_op.logical_op_class_name(),
        }
        ks = cls.k_budgets if logical_op.k == -1 else [logical_op.k]
        variable_op_kwargs = [{"k": k} for k in ks]
        return cls._perform_substitution(logical_expression, SemTopKOperator, fixed_op_kwargs, variable_op_kwargs)


# class FilterRule(ImplementationRule):
#     """
#     Substitute a logical expression for a non-semantic filter with its physical implementation.
#     """

#     @classmethod
#     def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
#         logical_op = logical_expression.operator
#         return isinstance(logical_op, Filter) and logical_op.filter.filter_fn is not None

#     @classmethod
#     def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
#         return cls._perform_substitution(logical_expression, NonLLMFilter, runtime_kwargs)


class SemFilterRule(ImplementationRule):
    """Substitute a logical ``Filter`` expression with a ``SemFilterOperator`` physical implementation."""

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression, **kwargs) -> bool:
        return isinstance(logical_expression.operator, Filter)

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logical_op: Filter = logical_expression.operator
        fixed_op_kwargs = {
            "task": logical_op.filter,
            "dataset_id": logical_op.name,
            "llm_config": runtime_kwargs["llm_config"],
            "max_workers": runtime_kwargs["max_workers"],
            "logical_op_id": logical_op.get_logical_op_id(),
            "logical_op_class_name": logical_op.logical_op_class_name(),
        }
        variable_op_kwargs = [{"model_id": model_id} for model_id in runtime_kwargs["available_model_ids"]]
        return cls._perform_substitution(logical_expression, SemFilterOperator, fixed_op_kwargs, variable_op_kwargs)


# class RelationalJoinRule(ImplementationRule):
#     """
#     Substitute a logical expression for a non-semantic join with its physical implementation.
#     """

#     @classmethod
#     def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
#         return isinstance(logical_expression.operator, Join) and logical_expression.operator.condition == ""

#     @classmethod
#     def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
#         return cls._perform_substitution(logical_expression, RelationalJoin, runtime_kwargs)


class SemJoinRule(ImplementationRule):
    """
    Substitute a logical expression for a semantic join with its physical implementation.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression, **kwargs) -> bool:
        return isinstance(logical_expression.operator, Join)

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logical_op: Join = logical_expression.operator
        fixed_op_kwargs = {
            "task": logical_op.condition,
            "dataset_id": logical_op.name,
            "llm_config": runtime_kwargs["llm_config"],
            "max_workers": runtime_kwargs["max_workers"],
            "logical_op_id": logical_op.get_logical_op_id(),
            "logical_op_class_name": logical_op.logical_op_class_name(),
        }
        variable_op_kwargs = [{"model_id": model_id} for model_id in runtime_kwargs["available_model_ids"]]
        return cls._perform_substitution(logical_expression, SemJoinOperator, fixed_op_kwargs, variable_op_kwargs)


# class EmbeddingJoinRule(ImplementationRule):
#     """
#     Substitute a logical expression for a semantic join with its EmbeddingJoin physical implementation.
#     """

#     @classmethod
#     def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
#         return isinstance(logical_expression.operator, JoinOp) and logical_expression.operator.condition != "" and not cls._is_audio_operation(logical_expression)

#     @classmethod
#     def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:

#         # create variable physical operator kwargs for each  (model, embedding_model) which can implement this logical_expression
#         provided_models: list[Model] = runtime_kwargs["available_model_ids"]
#         models = [model for model in provided_models if cls._model_matches_input(model, logical_expression)]
#         embedding_models = [model for model in provided_models if cls._embedding_model_matches_input(model, logical_expression)]
#         variable_op_kwargs = []

#         for (model, embedding_model) in product(models, embedding_models):
#             reasoning_prompt_strategy = use_reasoning_prompt(runtime_kwargs["reasoning_effort"])
#             prompt_strategy = PromptStrategy.JOIN if reasoning_prompt_strategy else PromptStrategy.JOIN_NO_REASONING
#             variable_op_kwargs.append(
#                 {
#                     "model": model,
#                     "embedding_model": embedding_model,
#                     "prompt_strategy": prompt_strategy,
#                     "join_parallelism": runtime_kwargs["join_parallelism"],
#                     "reasoning_effort": runtime_kwargs["reasoning_effort"],
#                     "retain_inputs": not runtime_kwargs["is_validation"],
#                     "num_samples": 10, # TODO: iterate over different choices of num_samples
#                 }
#             )

#         return cls._perform_substitution(logical_expression, EmbeddingJoin, runtime_kwargs, variable_op_kwargs)


class SemAggRule(ImplementationRule):
    """
    Substitute a logical expression for a semantic aggregation with its physical implementation.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression, **kwargs) -> bool:
        return isinstance(logical_expression.operator, Aggregate)

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logical_op: Aggregate = logical_expression.operator
        field_names = [f["name"] for f in logical_op.agg_fields]
        fixed_op_kwargs = {
            "task": f"Compute the following aggregation fields: {field_names}",
            "agg_fields": logical_op.agg_fields,
            "dataset_id": logical_op.name,
            "llm_config": runtime_kwargs["llm_config"],
            "max_workers": runtime_kwargs["max_workers"],
            "logical_op_id": logical_op.get_logical_op_id(),
            "logical_op_class_name": logical_op.logical_op_class_name(),
        }
        variable_op_kwargs = [{"model_id": model_id} for model_id in runtime_kwargs["available_model_ids"]]
        return cls._perform_substitution(logical_expression, SemAggOperator, fixed_op_kwargs, variable_op_kwargs)


class SemGroupByRule(ImplementationRule):
    """
    Substitute a logical expression for a group-by aggregate with its physical implementation.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression, **kwargs) -> bool:
        return isinstance(logical_expression.operator, GroupBy)

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logical_op: GroupBy = logical_expression.operator
        gby_field_names = [f["name"] for f in logical_op.gby_fields]
        agg_field_names = [f["name"] for f in logical_op.agg_fields]
        agg_funcs = [f["func"] for f in logical_op.agg_fields]
        task = (
            f"Group by fields {gby_field_names} with aggregations on "
            f"{agg_field_names} using {agg_funcs} for each aggregation "
            f"field, respectively."
        )
        fixed_op_kwargs = {
            "task": task,
            "group_by_fields": logical_op.gby_fields,
            "agg_fields": logical_op.agg_fields,
            "dataset_id": logical_op.name,
            "llm_config": runtime_kwargs["llm_config"],
            "max_workers": runtime_kwargs["max_workers"],
            "logical_op_id": logical_op.get_logical_op_id(),
            "logical_op_class_name": logical_op.logical_op_class_name(),
        }
        variable_op_kwargs = [{"model_id": model_id} for model_id in runtime_kwargs["available_model_ids"]]
        return cls._perform_substitution(logical_expression, SemGroupByOperator, fixed_op_kwargs, variable_op_kwargs)

# class AggregateRule(ImplementationRule):
#     """
#     Substitute the logical expression for an aggregate with its physical counterpart.
#     """

#     @classmethod
#     def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
#         return isinstance(logical_expression.operator, Aggregate) and logical_expression.operator.agg_func is not None

#     @classmethod
#     def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
#         # get the physical op class based on the aggregation function
#         physical_op_class = None
#         if logical_expression.operator.agg_func == AggFunc.COUNT:
#             physical_op_class = CountAggregateOp
#         elif logical_expression.operator.agg_func == AggFunc.AVERAGE:
#             physical_op_class = AverageAggregateOp
#         elif logical_expression.operator.agg_func == AggFunc.SUM:
#             physical_op_class = SumAggregateOp
#         elif logical_expression.operator.agg_func == AggFunc.MIN:
#             physical_op_class = MinAggregateOp
#         elif logical_expression.operator.agg_func == AggFunc.MAX:
#             physical_op_class = MaxAggregateOp
#         else:
#             raise Exception(f"Cannot support aggregate function: {logical_expression.operator.agg_func}")

#         # perform the substitution
#         return cls._perform_substitution(logical_expression, physical_op_class, runtime_kwargs)


# TODO: move into BasicSubstitutionRule?
class ScanRule(ImplementationRule):
    """Substitute a logical ``Scan`` expression with a ``ScanOp`` physical implementation.

    Unlike other implementation rules, ``ScanRule`` does not consume any
    ``runtime_kwargs`` — a scan has exactly one physical alternative with
    no model variation.

    Representation invariant:
        - ``matches_pattern`` returns ``True`` iff the logical expression's
          operator is an instance of ``Scan``.

    Abstraction function:
        Converts the planner's logical intent to read a dataset into a
        concrete ``ScanOp`` that carries token-count metadata for the
        cost model.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression, **kwargs) -> bool:
        return isinstance(logical_expression.operator, Scan)

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        """Create a single ``ScanOp`` physical expression from the logical ``Scan``.

        Requires:
            - ``logical_expression.operator`` is a ``Scan``.

        Returns:
            A singleton set containing the ``PhysicalExpression`` wrapping
            a ``ScanOp`` with the same ``dataset_id``, ``num_items``, and
            ``est_tokens_per_item`` as the logical operator.

        Raises:
            None.
        """
        logical_op: Scan = logical_expression.operator
        fixed_op_kwargs = {
            "dataset_id": logical_op.dataset_id,
            "num_items": logical_op.num_items,
            "est_tokens_per_item": logical_op.est_tokens_per_item,
            "logical_op_id": logical_op.get_logical_op_id(),
            "logical_op_class_name": logical_op.logical_op_class_name(),
        }
        # Only one physical alternative for a scan — no model variation.
        return cls._perform_substitution(logical_expression, ScanOp, fixed_op_kwargs)


class BasicSubstitutionRule(ImplementationRule):
    """For logical operators with a single physical implementation, substitute the
    logical expression with its physical counterpart.

    Uses a class-level mapping from logical operator classes to physical
    operator classes, plus an optional parameter-name translation table.
    """

    LOGICAL_OP_CLASS_TO_PHYSICAL_OP_CLASS_MAP = {
        Limit: LimitOperator,
    }

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression, **kwargs) -> bool:
        logical_op_class = logical_expression.operator.__class__
        return logical_op_class in cls.LOGICAL_OP_CLASS_TO_PHYSICAL_OP_CLASS_MAP

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logical_op = logical_expression.operator
        physical_op_class = cls.LOGICAL_OP_CLASS_TO_PHYSICAL_OP_CLASS_MAP[logical_op.__class__]

        # translate name --> dataset_id for all logical operators since all physical operators use dataset_id
        fixed_op_kwargs = logical_op.get_logical_op_params()
        fixed_op_kwargs["logical_op_id"] = logical_op.get_logical_op_id()
        fixed_op_kwargs["logical_op_class_name"] = logical_op.logical_op_class_name()
        if "name" in fixed_op_kwargs:
            fixed_op_kwargs["dataset_id"] = fixed_op_kwargs.pop("name")

        return cls._perform_substitution(logical_expression, physical_op_class, fixed_op_kwargs)
