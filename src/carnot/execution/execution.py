from __future__ import annotations

import logging
import time
from collections.abc import Generator

from smolagents.tools import Tool

from carnot.agents.models import LiteLLMModel
from carnot.agents.planner import Planner
from carnot.conversation.conversation import Conversation
from carnot.core.models import ExecutionStats, LLMCallStats, OperatorStats, PhaseStats
from carnot.data.dataset import Dataset
from carnot.execution.progress import ExecutionProgress, PlanningProgress
from carnot.index.index import CarnotIndex
from carnot.memory.memory import Memory
from carnot.plan import PhysicalPlan
from carnot.plan.feedback import PlanFeedback
from carnot.plan.node import PlanNode
from carnot.storage.catalog import IndexCatalog
from carnot.storage.config import StorageConfig
from carnot.storage.tiered import TieredStorageManager

logger = logging.getLogger('uvicorn.error')


class Execution:
    """Orchestrates planning and execution of Carnot queries.

    The optional *storage_config* parameter (:class:`StorageConfig`)
    controls how catalogs and storage backends are wired.  When omitted
    a default in-memory configuration is used (no external database
    required).  Callers can also pass pre-built *storage* and
    *index_catalog* objects to override the config-driven defaults.

    Representation invariant:
        - ``datasets`` is a list of ``Dataset`` objects.
        - ``llm_config`` contains the necessary API keys.
        - ``_physical_plan`` is either ``None`` (no plan yet) or a
          ``PhysicalPlan`` instance.

    Abstraction function:
        Represents the ability to plan and execute a Carnot query over a
        set of datasets using a configured LLM provider.  The plan may be
        supplied as a raw dict (backward compatibility) or as a
        ``PhysicalPlan``.
    """
    def __init__(
            self,
            query: str,
            datasets: list[Dataset],
            plan: dict | PhysicalPlan | None = None,
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
        self.tools = tools or []
        self.conversation = conversation
        self.memory = memory or Memory()
        self.indices = indices or []
        self.llm_config = llm_config or {}
        self.progress_log_file = progress_log_file
        self.cost_budget = cost_budget
        self.storage_config = storage_config or StorageConfig()

        # Build the PhysicalPlan from whatever was passed in.
        # Uses the _plan property setter which handles dict->PhysicalPlan
        # conversion.
        self._physical_plan: PhysicalPlan | None = None
        self._plan = plan

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

    # -- _plan property (backward-compatible) --------------------------------

    @property
    def _plan(self) -> dict | PhysicalPlan | None:
        """Return the current plan.

        External callers that assign ``execution._plan = some_dict``
        (common in evals and tests) trigger the setter which converts
        the dict to a ``PhysicalPlan`` automatically.

        Requires:
            None.

        Returns:
            The ``PhysicalPlan`` if one exists, else ``None``.

        Raises:
            None.
        """
        return self._physical_plan

    @_plan.setter
    def _plan(self, value: dict | PhysicalPlan | None) -> None:
        """Set the plan, converting dicts to ``PhysicalPlan`` automatically.

        Requires:
            - *value* is ``None``, a valid plan dict, or a ``PhysicalPlan``.

        Returns:
            None.

        Raises:
            None.
        """
        if value is None or (isinstance(value, dict) and not value):
            self._physical_plan = None
        elif isinstance(value, PhysicalPlan):
            self._physical_plan = value
        elif isinstance(value, dict):
            self._physical_plan = PhysicalPlan.from_plan_dict(
                value, self.datasets, query=self.query,
            )
        else:
            self._physical_plan = None

    # -- planning ------------------------------------------------------------

    def plan(self) -> tuple[str, dict]:
        """Generate a logical execution plan for the query.

        This method uses a two-phase approach:
        1. Generate a code-based logical plan (the Planner can call its managed 
           DataDiscoveryAgent to explore datasets during planning)
        2. Translate the logical plan into a natural language description for the user

        After both phases complete, LLM call statistics are collected
        from the Planner's and DataDiscoveryAgent's memories and stored
        in ``self._planning_stats`` for later assembly into
        :class:`ExecutionStats`.

        Returns:
            A tuple of (natural_language_plan, logical_plan_dict)
        """
        plan_start = time.perf_counter()

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

        plan_wall_clock = time.perf_counter() - plan_start

        # Collect stats from planner memories
        self._planning_stats = self._build_planning_stats(plan_wall_clock)

        return nl_plan, logical_plan

    def plan_stream(self) -> Generator[PlanningProgress, None, tuple[str, dict]]:
        """Generate a logical execution plan, yielding progress events.

        This is the streaming counterpart of :meth:`plan`.  It performs
        the same two-phase approach (logical plan generation -> paraphrase)
        but yields :class:`PlanningProgress` events between steps so
        that callers can keep the user informed of progress.

        After both phases complete, LLM call statistics are collected
        and stored in ``self._planning_stats``.

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
        plan_start = time.perf_counter()

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
                # Terminal value -- the logical plan dict
                logical_plan = event

        yield PlanningProgress(
            phase="logical_plan",
            message="Logical plan generated. Preparing summary...",
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
        ):
            if isinstance(event, PlanningProgress):
                yield event
            else:
                # Terminal value -- the NL plan string
                nl_plan = event

        plan_wall_clock = time.perf_counter() - plan_start
        self._planning_stats = self._build_planning_stats(plan_wall_clock)

        yield PlanningProgress(
            phase="paraphrase",
            message="Plan summary complete.",
            cumulative_cost_usd=self._planning_stats.total_cost_usd,
        )

        return nl_plan, logical_plan

    # -- stats helpers -------------------------------------------------------

    @staticmethod
    def _collect_llm_calls_from_memory(memory) -> list[LLMCallStats]:
        """Extract ``LLMCallStats`` from every step in an ``AgentMemory``.

        Iterates over all steps in *memory* and collects
        ``llm_call_stats`` from each ``ActionStep`` whose
        ``model_output_message`` carries an ``LLMCallStats`` object.

        Requires:
            - *memory* is an ``AgentMemory`` instance (or ``None``).

        Returns:
            A (possibly empty) list of ``LLMCallStats`` objects.

        Raises:
            None.
        """
        if memory is None:
            return []
        calls: list[LLMCallStats] = []
        for step in memory.steps:
            msg = getattr(step, "model_output_message", None)
            if msg is not None and getattr(msg, "llm_call_stats", None) is not None:
                calls.append(msg.llm_call_stats)
        return calls

    def _build_planning_stats(self, wall_clock_secs: float) -> PhaseStats:
        """Assemble ``PhaseStats`` for the planning phase.

        Collects ``LLMCallStats`` from the Planner's planning memory,
        paraphrase memory, and the DataDiscoveryAgent's memory.

        Requires:
            - ``self.planner`` has been used for planning (memories
              may still be ``None`` if planning was skipped).
            - *wall_clock_secs* >= 0.

        Returns:
            A ``PhaseStats`` with ``phase="planning"`` and per-agent
            ``OperatorStats``.

        Raises:
            None.
        """
        # Data discovery agent calls
        dd_memory = getattr(self.planner._data_discovery_agent, "memory", None)
        dd_calls = self._collect_llm_calls_from_memory(dd_memory)

        # Planner's own calls (planning + paraphrase)
        planning_calls = self._collect_llm_calls_from_memory(
            getattr(self.planner, "planning_memory", None)
        )
        paraphrase_calls = self._collect_llm_calls_from_memory(
            getattr(self.planner, "paraphrase_memory", None)
        )

        operator_stats: list[OperatorStats] = []
        if dd_calls:
            operator_stats.append(
                OperatorStats(
                    operator_name="DataDiscovery",
                    operator_id="data_discovery",
                    llm_calls=dd_calls,
                )
            )
        operator_stats.append(
            OperatorStats(
                operator_name="Planner",
                operator_id="planner",
                llm_calls=planning_calls + paraphrase_calls,
            )
        )

        return PhaseStats(
            phase="planning",
            wall_clock_secs=wall_clock_secs,
            operator_stats=operator_stats,
        )

    # -- execution -----------------------------------------------------------

    def run_node(
        self,
        node_id: str,
        datasets_store: dict[str, Dataset],
    ) -> tuple[dict[str, Dataset], OperatorStats | None]:
        """Execute a single plan node.

        This is the primitive that :meth:`run` composes and which external
        callers may use directly for node-by-node execution.

        Requires:
            - ``self._physical_plan`` is not None.
            - *node_id* exists in the plan.
            - All parent datasets for that node exist in
              *datasets_store*.

        Returns:
            ``(updated_store, op_stats)`` -- the store with the new
            output dataset added, and the operator's stats (``None``
            for dataset loads).

        Raises:
            KeyError: If a required parent dataset is missing.
            ValueError: If the operator type is unrecognized.
        """
        plan = self._physical_plan
        node = plan.get_node(node_id)

        # Resolve parent output_dataset_ids for operator calling convention
        parent_output_ids = [
            plan.get_node(pid).output_dataset_id
            for pid in node.parent_ids
        ]

        return node.execute(
            datasets_store,
            self.llm_config,
            leaf_datasets={ds.name: ds for ds in self.datasets},
            storage=self.storage,
            index_catalog=self.index_catalog,
            parent_output_ids=parent_output_ids,
        )

    def run(self) -> tuple[list[dict], str, ExecutionStats]:
        """Execute the physical plan and return the result with stats.

        Runs every node in topological order via :meth:`run_node`,
        collects ``OperatorStats`` from each, and assembles an
        ``ExecutionStats`` that combines planning and execution
        phase statistics.

        Requires:
            - ``self._physical_plan`` is not None (either set via
              constructor, ``_plan`` setter, or :meth:`plan`).

        Returns:
            A 3-tuple ``(items, answer_str, stats)`` where *items* is
            the list of result dicts, *answer_str* is the final
            answer text, and *stats* is the full ``ExecutionStats``.

        Raises:
            ValueError: If an operator fails.
        """
        exec_start = time.perf_counter()
        datasets_store: dict[str, Dataset] = {}
        all_operator_stats: list[OperatorStats] = []

        for node in self._physical_plan.topo_order():
            datasets_store, op_stats = self.run_node(
                node.node_id, datasets_store,
            )
            if op_stats is not None:
                all_operator_stats.append(op_stats)

        final_dataset = datasets_store.get("final_dataset")

        exec_wall_clock = time.perf_counter() - exec_start

        execution_phase = PhaseStats(
            phase="execution",
            wall_clock_secs=exec_wall_clock,
            operator_stats=all_operator_stats,
        )

        stats = ExecutionStats(
            query=self.query,
            planning=getattr(self, "_planning_stats", PhaseStats(phase="planning")),
            execution=execution_phase,
        )

        items = final_dataset.items if final_dataset else []
        answer_str = final_dataset.code_state.get("final_answer_str", "") if final_dataset else ""

        return items, answer_str, stats

    # -- notebook / interactive plan helpers ---------------------------------

    def get_physical_plan(self) -> list[dict]:
        """Serialise the physical plan into a list of node descriptors.

        Delegates to ``PhysicalPlan.to_node_dicts()`` which produces a
        node descriptor for each node in topological order.

        Each descriptor is a dict with keys: ``node_id``,
        ``node_type`` (``"dataset"`` | ``"operator"`` | ``"reasoning"``),
        ``operator_name``, ``operator_type``, ``description``, ``code``,
        ``original_code``, ``params``, ``parent_dataset_ids``, and
        ``output_dataset_id``.

        Requires:
            - ``self._physical_plan`` is not None.

        Returns:
            A list of node-descriptor dicts suitable for JSON
            serialization.

        Raises:
            None.
        """
        return self._physical_plan.to_node_dicts()

    def execute_cell(
        self,
        cell: dict,
        input_datasets: dict[str, Dataset],
    ) -> tuple[dict[str, Dataset], OperatorStats | None, dict]:
        """Execute a single cell and return the updated datasets and output preview.

        Delegates to :meth:`run_node` for the actual execution, using
        the cell's ``node_id`` to look up the corresponding plan node.

        Requires:
            - ``cell`` is a valid node descriptor dict (must contain
              ``"node_id"``).
            - All datasets referenced by the node's parents exist in
              ``input_datasets``.

        Returns:
            A 3-tuple ``(updated_datasets, op_stats, output_preview)``
            where ``output_preview`` contains ``items_count``,
            ``preview`` (first 10 items), and ``schema``.

        Raises:
            KeyError: If a required parent dataset is missing.
            ValueError: If the operator type is unrecognized.
        """
        node_id = cell["node_id"]
        node = self._physical_plan.get_node(node_id)

        updated, op_stats = self.run_node(node_id, input_datasets)

        output_dataset = updated.get(node.output_dataset_id)
        preview = self._build_output_preview(output_dataset) if output_dataset else {}

        # Add answer text for reasoning nodes
        if node.node_type == "reasoning" and output_dataset:
            preview["answer"] = output_dataset.code_state.get("final_answer_str", "")

        return updated, op_stats, preview

    @staticmethod
    def _build_output_preview(dataset: Dataset) -> dict:
        """Build a preview dict for a dataset's items.

        Returns:
            A dict with ``items_count``, ``preview`` (first 10 items),
            and ``schema`` (list of field names from the first item).

        Raises:
            None.
        """
        items = dataset.items if dataset else []
        schema = list(items[0].keys()) if items else []
        return {
            "items_count": len(items),
            "preview": items[:10],
            "schema": schema,
        }

    # -- re-optimisation -------------------------------------------------------

    def reoptimize(
        self,
        feedback: PlanFeedback,
    ) -> dict[str, list[str]]:
        """Apply user feedback to the current physical plan.

        Processes the feedback in order:

        1. **Node edits** (param changes) are applied in-place via
           ``PhysicalPlan.edit_node()``.  No LLM call is made.
        2. **Structural deletions** are applied via
           ``PhysicalPlan.delete_node()``.
        3. **Structural insertions** are applied via
           ``PhysicalPlan.insert_node()``.

        All mutations return a set of invalidated downstream node IDs.
        The caller (typically the web backend) uses these to reset
        application-level state (e.g., cell status in the frontend).

        Requires:
            - ``self._physical_plan`` is not ``None``.
            - ``feedback`` contains at least one edit or structural
              change.

        Returns:
            A dict mapping edit labels to their invalidated-node-ID
            lists::

                {
                    "edit:<node_id>": ["n3", "n4"],
                    "delete:<node_id>": ["n3"],
                    "insert:<new_node_id>": ["n5"],
                }

        Raises:
            KeyError: If any referenced node ID does not exist.
            ValueError: If an edit has empty params, or a dataset node
                is deleted, or an inserted node ID already exists.
        """
        plan = self._physical_plan
        invalidation_map: dict[str, list[str]] = {}

        # 1. Apply param edits.
        for edit in feedback.node_edits:
            inv = plan.edit_node(edit.node_id, edit.new_params)
            invalidation_map[f"edit:{edit.node_id}"] = inv

        # 2. Apply deletions (before insertions so that IDs remain stable).
        for sc in feedback.structural_changes:
            if sc.change_type == "delete" and sc.node_id is not None:
                inv = plan.delete_node(sc.node_id)
                invalidation_map[f"delete:{sc.node_id}"] = inv

        # 3. Apply insertions.
        for sc in feedback.structural_changes:
            if (
                sc.change_type == "insert"
                and sc.after_node_id is not None
                and sc.new_node_params is not None
            ):
                new_node = self._node_from_params(sc.new_node_params, plan)
                inv = plan.insert_node(sc.after_node_id, new_node)
                invalidation_map[f"insert:{new_node.node_id}"] = inv

        return invalidation_map

    @staticmethod
    def _node_from_params(params: dict, plan: PhysicalPlan) -> PlanNode:
        """Create a ``PlanNode`` from a user-supplied params dict.

        Requires:
            - *params* has at least ``"operator"`` and ``"name"`` keys.

        Returns:
            A new ``PlanNode`` with a unique auto-generated ``node_id``.

        Raises:
            KeyError: If required keys are missing from *params*.
        """
        # Generate a unique node ID.
        existing_ids = {n.node_id for n in plan.nodes}
        counter = 0
        while f"node-{counter}" in existing_ids:
            counter += 1
        node_id = f"node-{counter}"

        operator_type = params.get("operator", "")
        name = params.get("name", operator_type)
        description = params.get("description", name)
        output_dataset_id = params.get("output_dataset_id", name)
        node_type = "reasoning" if operator_type == "Reasoning" else "operator"

        return PlanNode(
            node_id=node_id,
            node_type=node_type,
            operator_type=operator_type or None,
            name=name,
            description=description,
            params=dict(params),
            parent_ids=[],  # Will be set by insert_node()
            output_dataset_id=output_dataset_id,
        )

    # -- streaming run -------------------------------------------------------

    def run_stream(self) -> Generator[ExecutionProgress, None, tuple[list[dict], str, ExecutionStats]]:
        """Execute the physical plan, yielding progress events.

        This is the streaming counterpart of :meth:`run`.  It performs
        the same node-by-node execution but yields
        :class:`ExecutionProgress` events between nodes so that
        callers can keep the user informed of progress.

        After each node completes, the ``"Completed step"`` progress
        event includes the ``OperatorStats`` for that operator.

        The **return value** (accessed via ``StopIteration.value``) is
        the same ``(items, answer_str, stats)`` 3-tuple that :meth:`run`
        returns.

        Typical usage from an async context::

            gen = execution.run_stream()
            result = None
            try:
                while True:
                    progress = next(gen)
                    # forward ``progress`` to SSE / websocket
            except StopIteration as exc:
                result = exc.value  # (items, answer_str, stats)

        Requires:
            - ``self._physical_plan`` is not None.
            - ``self.datasets`` is a non-empty list.

        Returns:
            A generator that yields :class:`ExecutionProgress` objects.
            The generator's return value is
            ``(items, answer_str, stats)``.

        Raises:
            ValueError: If the plan contains unrecognized operators.
        """
        exec_start = time.perf_counter()
        datasets_store: dict[str, Dataset] = {}
        all_operator_stats: list[OperatorStats] = []
        nodes = self._physical_plan.topo_order()
        total = len(nodes)

        yield ExecutionProgress(
            message=f"Starting execution -- {total} step(s) in plan.",
            total_operators=total,
        )

        for idx, node in enumerate(nodes):
            display = node.display_name()

            yield ExecutionProgress(
                message=f"Running step {idx + 1}/{total}: {display}...",
                operator_index=idx,
                total_operators=total,
                operator_name=display,
            )

            datasets_store, op_stats = self.run_node(
                node.node_id, datasets_store,
            )
            if op_stats is not None:
                all_operator_stats.append(op_stats)

            yield ExecutionProgress(
                message=f"Completed step {idx + 1}/{total}: {display}.",
                operator_index=idx,
                total_operators=total,
                operator_name=display,
                operator_stats=op_stats,
            )

        final_dataset = datasets_store.get("final_dataset")

        exec_wall_clock = time.perf_counter() - exec_start

        execution_phase = PhaseStats(
            phase="execution",
            wall_clock_secs=exec_wall_clock,
            operator_stats=all_operator_stats,
        )

        stats = ExecutionStats(
            query=self.query,
            planning=getattr(self, "_planning_stats", PhaseStats(phase="planning")),
            execution=execution_phase,
        )

        items = final_dataset.items if final_dataset else []
        answer_str = final_dataset.code_state.get("final_answer_str", "") if final_dataset else ""

        return items, answer_str, stats
