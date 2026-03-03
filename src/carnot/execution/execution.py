from __future__ import annotations

import logging
from collections.abc import Generator

from smolagents.tools import Tool

from carnot.agents.models import LiteLLMModel
from carnot.agents.planner import Planner
from carnot.conversation.conversation import Conversation
from carnot.data.dataset import Dataset
from carnot.execution.progress import ExecutionProgress, PlanningProgress
from carnot.index.index import CarnotIndex
from carnot.memory.memory import Memory
from carnot.operators.code import CodeOperator
from carnot.operators.limit import LimitOperator
from carnot.operators.reasoning import ReasoningOperator
from carnot.operators.sem_agg import SemAggOperator
from carnot.operators.sem_filter import SemFilterOperator
from carnot.operators.sem_flat_map import SemFlatMapOperator
from carnot.operators.sem_groupby import SemGroupByOperator
from carnot.operators.sem_join import SemJoinOperator
from carnot.operators.sem_map import SemMapOperator
from carnot.operators.sem_topk import SemTopKOperator
from carnot.storage.catalog import IndexCatalog
from carnot.storage.config import StorageConfig
from carnot.storage.tiered import TieredStorageManager

Operator = CodeOperator | ReasoningOperator | SemAggOperator | SemFilterOperator | SemFlatMapOperator | SemGroupByOperator | SemJoinOperator | SemMapOperator | SemTopKOperator
logger = logging.getLogger('uvicorn.error')

class Execution:
    """Class for managing the execution of a query in Carnot.

    The optional *storage_config* parameter (:class:`StorageConfig`)
    controls how catalogs and storage backends are wired.  When omitted
    a default in-memory configuration is used (no external database
    required).  Callers can also pass pre-built *storage* and
    *index_catalog* objects to override the config-driven defaults.
    """
    def __init__(
            self,
            query: str,
            datasets: list[Dataset],
            plan: dict | None = None,
            tools: list[Tool] | None = None,
            conversation: Conversation | None = None,
            memory: Memory | None = None,
            indices: list[CarnotIndex] | None = None,
            llm_config: dict | None = None,
            progress_log_file: str | None = None,
            cost_budget: float | None = None,
            storage: TieredStorageManager | None = None,
            index_catalog: IndexCatalog | None = None,
            storage_config: StorageConfig | None = None,
        ):
        self.query = query
        self.datasets = datasets
        self._plan = plan or {}
        self.tools = tools or []
        self.conversation = conversation
        self.memory = memory or Memory()
        self.indices = indices or []
        self.llm_config = llm_config or {}
        self.progress_log_file = progress_log_file
        self.cost_budget = cost_budget
        self.storage_config = storage_config or StorageConfig()

        # Use explicitly passed objects, or derive from storage_config
        self.storage = storage
        if self.storage is None and self.storage_config is not None:
            # Build a default storage manager if not provided
            pass  # Caller can set up storage externally; None is valid

        db_factory = None
        if self.storage_config.has_postgres:
            db_factory = self.storage_config.get_db_session_factory()

        self.index_catalog = index_catalog or IndexCatalog(
            storage=self.storage,
            db_session_factory=db_factory,
        )

        self.planner_model_id = "openai/gpt-5-2025-08-07"
        self.api_key_name = "OPENAI_API_KEY"
        if "OPENAI_API_KEY" not in self.llm_config and "ANTHROPIC_API_KEY" in self.llm_config:
            self.planner_model_id = "anthropic/claude-sonnet-4-5-20250929"
            self.api_key_name = "ANTHROPIC_API_KEY"
        elif "OPENAI_API_KEY" not in self.llm_config and "GEMINI_API_KEY" in self.llm_config:
            self.planner_model_id = "google/gemini-2.5-flash"
            self.api_key_name = "GEMINI_API_KEY"
        elif "OPENAI_API_KEY" not in self.llm_config and "GOOGLE_API_KEY" in self.llm_config:
            self.planner_model_id = "google/gemini-2.5-flash"
            self.api_key_name = "GOOGLE_API_KEY"
        self.planner = Planner(
            datasets=self.datasets,
            tools=self.tools, 
            model=LiteLLMModel(model_id=self.planner_model_id, api_key=llm_config.get(self.api_key_name))
        )

    def plan(self) -> tuple[str, dict]:
        """
        Generate a logical execution plan for the query.
        
        This method uses a two-phase approach:
        1. Generate a code-based logical plan (the Planner can call its managed 
           DataDiscoveryAgent to explore datasets during planning)
        2. Translate the logical plan into a natural language description for the user
        
        Returns:
            A tuple of (natural_language_plan, logical_plan_dict)
        """
        # Phase 1: Generate the code-based logical plan
        # The Planner can call its DataDiscoveryAgent as needed during planning
        logical_plan = self.planner.generate_logical_plan(
            self.query, 
            self.datasets, 
            conversation=self.conversation,
            cost_budget=self.cost_budget,
        )

        # Phase 2: Translate the logical plan to natural language for the user
        nl_plan = self.planner.paraphrase_logical_plan(
            self.query, 
            logical_plan, 
            self.datasets,
            conversation=self.conversation,
            cost_budget=self.cost_budget,
        )

        return nl_plan, logical_plan

    def plan_stream(self) -> Generator[PlanningProgress, None, tuple[str, dict]]:
        """Generate a logical execution plan, yielding progress events.

        This is the streaming counterpart of :meth:`plan`.  It performs
        the same two-phase approach (logical plan generation → paraphrase)
        but yields :class:`PlanningProgress` events between steps so
        that callers can keep the user informed of progress.

        The **return value** (accessed via ``StopIteration.value`` or by
        collecting the generator with a helper) is the same
        ``(natural_language_plan, logical_plan_dict)`` tuple that
        :meth:`plan` returns.

        Typical usage from an async context::

            gen = execution.plan_stream()
            result = None
            try:
                while True:
                    progress = next(gen)
                    # forward ``progress`` to SSE / websocket
            except StopIteration as exc:
                result = exc.value  # (nl_plan, logical_plan)

        Requires:
            - ``self.query`` is a non-empty string.
            - ``self.datasets`` is a non-empty list.

        Returns:
            A generator that yields :class:`PlanningProgress` objects.
            The generator's return value is
            ``(natural_language_plan, logical_plan_dict)``.

        Raises:
            AgentGenerationError: If the LLM fails to produce valid
            output during either phase.
        """
        # ----------------------------------------------------------
        # Phase 1: Generate the code-based logical plan
        # ----------------------------------------------------------
        logical_plan = None
        for event in self.planner.generate_logical_plan_stream(
            self.query,
            self.datasets,
            conversation=self.conversation,
            cost_budget=self.cost_budget,
        ):
            if isinstance(event, PlanningProgress):
                yield event
            else:
                # Terminal value — the logical plan dict
                logical_plan = event

        yield PlanningProgress(
            phase="logical_plan",
            message="Logical plan generated. Preparing summary…",
        )

        # ----------------------------------------------------------
        # Phase 2: Translate the logical plan to natural language
        # ----------------------------------------------------------
        nl_plan = None
        for event in self.planner.paraphrase_logical_plan_stream(
            self.query,
            logical_plan,
            self.datasets,
            conversation=self.conversation,
            cost_budget=self.cost_budget,
        ):
            if isinstance(event, PlanningProgress):
                yield event
            else:
                # Terminal value — the NL plan string
                nl_plan = event

        yield PlanningProgress(
            phase="paraphrase",
            message="Plan summary complete.",
        )

        return nl_plan, logical_plan

    def _get_op_from_plan_dict(self, plan: dict) -> tuple[Operator | Dataset, list[str]]:
        """Return the physical operator (or Dataset) for a single plan node.

        The *plan* dict must have the following schema::

            {
                "name": str,                  # node name
                "output_dataset_id": str,     # output identifier
                "params": {                   # operator-specific params
                    "operator": str,          # one of the recognized names below
                    ...                       # operator-specific keys
                },
                "parents": [<plan dict>, ...] # parent plan nodes
            }

        Recognized ``operator`` values and corresponding physical classes:

        - ``"Code"`` → :class:`CodeOperator` (requires ``"task"``).
        - ``"Limit"`` → :class:`LimitOperator` (requires ``"n"``).
        - ``"SemanticAgg"`` → :class:`SemAggOperator`.
        - ``"SemanticFilter"`` → :class:`SemFilterOperator`.
        - ``"SemanticMap"`` → :class:`SemMapOperator`.
        - ``"SemanticFlatMap"`` → :class:`SemFlatMapOperator`.
        - ``"SemanticGroupBy"`` → :class:`SemGroupByOperator`.
        - ``"SemanticJoin"`` → :class:`SemJoinOperator`.
        - ``"SemanticTopK"`` → :class:`SemTopKOperator`.

        If no ``"operator"`` key is present in ``params``, the node is
        treated as a dataset reference — the ``name`` is looked up in
        ``self.datasets``.

        Returns:
            A tuple ``(operator_or_dataset, parent_dataset_ids)`` where
            *parent_dataset_ids* is a list of ``output_dataset_id``
            strings from the node's parents.

        Raises:
            ValueError: if the operator name is unrecognized and does
            not match any dataset name.
        """
        # TODO: filter for model_id and max_workers from llm_config
        operator = None
        op_params = plan['params']
        op_name = op_params.get('operator', plan['name'])
        if op_name == "Code":
            operator = CodeOperator(task=op_params['task'], output_dataset_id=plan['output_dataset_id'], model_id="openai/gpt-5-mini", llm_config=self.llm_config)

        elif op_name == "Limit":
            operator = LimitOperator(n=op_params['n'], output_dataset_id=plan['output_dataset_id'])

        elif op_name == "SemanticAgg":
            operator = SemAggOperator(task=op_params['task'], agg_fields=op_params['agg_fields'], output_dataset_id=plan['output_dataset_id'], model_id="openai/gpt-5-mini", llm_config=self.llm_config, max_workers=4)

        elif op_name == "SemanticFilter":
            operator = SemFilterOperator(task=op_params['condition'], output_dataset_id=plan['output_dataset_id'], model_id="openai/gpt-5-mini", llm_config=self.llm_config, max_workers=4)

        elif op_name == "SemanticMap":
            output_fields = [{"name": op_params['field'], "type": op_params['type'], "description": op_params['field_desc']}]
            operator = SemMapOperator(
                task="Execute the map operation to compute the following output field.",
                output_fields=output_fields,
                output_dataset_id=plan['output_dataset_id'],
                model_id="openai/gpt-5-mini",
                llm_config=self.llm_config,
                max_workers=4,
            )

        elif op_name == "SemanticFlatMap":
            output_fields = [{"name": op_params['field'], "type": op_params['type'], "description": op_params['field_desc']}]
            operator = SemFlatMapOperator(
                task="Execute the flat map operation to compute the following output field.",
                output_fields=output_fields,
                output_dataset_id=plan['output_dataset_id'],
                model_id="openai/gpt-5-mini",
                llm_config=self.llm_config,
                max_workers=4,
            )

        elif op_name == "SemanticGroupBy":
            gby_field_names = [field['name'] for field in op_params['gby_fields']]
            agg_field_names = [field['name'] for field in op_params['agg_fields']]
            agg_funcs = [field['func'] for field in op_params['agg_fields']]
            task = f"Group by fields {gby_field_names} with aggregations on {agg_field_names} using {agg_funcs} for each aggregation field, respectively."
            operator = SemGroupByOperator(
                task=task,
                group_by_fields=op_params['gby_fields'],
                agg_fields=op_params['agg_fields'],
                output_dataset_id=plan['output_dataset_id'],
                model_id="openai/gpt-5-mini",
                llm_config=self.llm_config,
                max_workers=4,
            )

        elif op_name == "SemanticJoin":
            operator = SemJoinOperator(task=op_params['condition'], output_dataset_id=plan['output_dataset_id'], model_id="openai/gpt-5-mini", llm_config=self.llm_config, max_workers=4)

        elif op_name == "SemanticTopK":
            operator = SemTopKOperator(task=op_params['search_str'], k=op_params['k'], output_dataset_id=plan['output_dataset_id'], model_id="openai/text-embedding-3-small", llm_config=self.llm_config, max_workers=4, index_name=op_params["index_name"], catalog=self.index_catalog)

        else:
            for dataset in self.datasets:
                if dataset.name == op_name:
                    operator = dataset

        if operator is None:
            raise ValueError(f"Unknown operator or dataset name: {op_name}")

        return operator, [subplan['output_dataset_id'] for subplan in plan['parents']]

    def _get_ops_in_topological_order(self, plan: dict) -> list[tuple[Operator | Dataset, list[str]]]:
        """Linearise a plan DAG into topological (dependency-first) order.

        Uses a recursive DFS: for each node, all parents are visited
        before the node itself.

        Requires:
            - *plan* is a valid plan dict (see :meth:`_get_op_from_plan_dict`).

        Returns:
            A list of ``(operator_or_dataset, parent_dataset_ids)``
            tuples in topological order (leaves first, root last).
        """
        # base case: this operator has no parents
        parents = plan.get('parents', [])
        if not parents:
            return [self._get_op_from_plan_dict(plan)]
        
        # recursive case: use DFS to get topological order, get parents first and then append this operator
        ops = []
        for parent in parents:
            ops.extend(self._get_ops_in_topological_order(parent))
        ops.append(self._get_op_from_plan_dict(plan))
        return ops

    def run(self) -> tuple[list[dict], str]: # physical_plan: PhysicalPlan -> str
        """
        Execute the physical plan and return the result.
        """
        # TODO: scope input_datasets based on children / parents in plan
        input_datasets = {}
        operators = self._get_ops_in_topological_order(self._plan)
        for operator, parent_ids in operators:
            if isinstance(operator, Dataset):
                # materialize items through the storage layer (or fallback)
                input_datasets[operator.name] = operator.materialize(self.storage)
            elif isinstance(operator, CodeOperator):
                input_datasets = operator(input_datasets)
            elif isinstance(operator, SemJoinOperator):
                left_dataset_id = parent_ids[0]
                right_dataset_id = parent_ids[1]
                input_datasets = operator(left_dataset_id, right_dataset_id, input_datasets)
            else:
                dataset_id = parent_ids[0]
                input_datasets = operator(dataset_id, input_datasets)

        # Use a reasoning operator to return a final output dataset which has all of the items, code_state, and/or answer text
        # from the input datasets which is relevant to the user for interpreting the final answer
        # - list of items (dicts) can be written to a pd.DataFrame --> csv
        # - text can be displayed to the user
        # - for now, assume code state is debug only (exposed in the future)
        final_answer_operator = ReasoningOperator(task=self.query, output_dataset_id="final_dataset", model_id="openai/gpt-5-mini", llm_config=self.llm_config)
        output_datasets = final_answer_operator(input_datasets)
        final_dataset = output_datasets["final_dataset"]

        return final_dataset.items, final_dataset.code_state.get("final_answer_str", "")

    # -- operator display helpers ------------------------------------------------

    _OPERATOR_DISPLAY_NAMES: dict[type, str] = {
        SemFilterOperator: "Semantic Filter",
        SemMapOperator: "Semantic Map",
        SemFlatMapOperator: "Semantic Flat Map",
        SemGroupByOperator: "Semantic Group By",
        SemJoinOperator: "Semantic Join",
        SemTopKOperator: "Semantic Top-K",
        SemAggOperator: "Semantic Aggregation",
        CodeOperator: "Code",
        ReasoningOperator: "Reasoning",
        LimitOperator: "Limit",
    }

    @staticmethod
    def _operator_display_name(operator: Operator | Dataset) -> str:
        """Return a human-readable label for an operator or dataset.

        Requires:
            - *operator* is an instance of a known operator type or
              :class:`Dataset`.

        Returns:
            A short display string such as ``"Semantic Filter"`` or
            ``"Dataset: Movies"``.

        Raises:
            None.
        """
        if isinstance(operator, Dataset):
            return f"Dataset: {operator.name}"
        return Execution._OPERATOR_DISPLAY_NAMES.get(
            type(operator), type(operator).__name__
        )

    # -- streaming run -----------------------------------------------------------

    def run_stream(self) -> Generator[ExecutionProgress, None, tuple[list[dict], str]]:
        """Execute the physical plan, yielding progress events.

        This is the streaming counterpart of :meth:`run`.  It performs
        the same operator-by-operator execution but yields
        :class:`ExecutionProgress` events between operators so that
        callers can keep the user informed of progress.

        The **return value** (accessed via ``StopIteration.value``) is
        the same ``(items, answer_str)`` tuple that :meth:`run` returns.

        Typical usage from an async context::

            gen = execution.run_stream()
            result = None
            try:
                while True:
                    progress = next(gen)
                    # forward ``progress`` to SSE / websocket
            except StopIteration as exc:
                result = exc.value  # (items, answer_str)

        Requires:
            - ``self._plan`` is a valid plan dict.
            - ``self.datasets`` is a non-empty list.

        Returns:
            A generator that yields :class:`ExecutionProgress` objects.
            The generator's return value is ``(items, answer_str)``.

        Raises:
            ValueError: If the plan contains unrecognized operators.
        """
        input_datasets: dict[str, Dataset] = {}
        operators = self._get_ops_in_topological_order(self._plan)
        total = len(operators)

        yield ExecutionProgress(
            message=f"Starting execution — {total} step(s) in plan.",
            total_operators=total,
        )

        for idx, (operator, parent_ids) in enumerate(operators):
            display = self._operator_display_name(operator)

            yield ExecutionProgress(
                message=f"Running step {idx + 1}/{total}: {display}…",
                operator_index=idx,
                total_operators=total,
                operator_name=display,
            )

            if isinstance(operator, Dataset):
                input_datasets[operator.name] = operator.materialize(self.storage)
            elif isinstance(operator, CodeOperator):
                input_datasets = operator(input_datasets)
            elif isinstance(operator, SemJoinOperator):
                left_dataset_id = parent_ids[0]
                right_dataset_id = parent_ids[1]
                input_datasets = operator(left_dataset_id, right_dataset_id, input_datasets)
            else:
                dataset_id = parent_ids[0]
                input_datasets = operator(dataset_id, input_datasets)

            yield ExecutionProgress(
                message=f"Completed step {idx + 1}/{total}: {display}.",
                operator_index=idx,
                total_operators=total,
                operator_name=display,
            )

        # Final reasoning step
        yield ExecutionProgress(
            message="Generating final answer…",
            operator_index=total,
            total_operators=total + 1,
            operator_name="Reasoning",
        )

        final_answer_operator = ReasoningOperator(
            task=self.query,
            output_dataset_id="final_dataset",
            model_id="openai/gpt-5-mini",
            llm_config=self.llm_config,
        )
        output_datasets = final_answer_operator(input_datasets)
        final_dataset = output_datasets["final_dataset"]

        yield ExecutionProgress(
            message="Execution complete.",
            operator_index=total + 1,
            total_operators=total + 1,
            operator_name="Reasoning",
        )

        return final_dataset.items, final_dataset.code_state.get("final_answer_str", "")
