from __future__ import annotations

import logging
import queue
import time
from collections.abc import Callable

from smolagents.tools import Tool

from carnot.agents.models import LiteLLMModel
from carnot.agents.planner import Planner
from carnot.conversation.conversation import Conversation
from carnot.core.models import ExecutionStats, LLMCallStats, OperatorStats, PhaseStats
from carnot.data.dataset import Dataset
from carnot.execution.progress import ExecutionProgress, PlanningProgress
from carnot.index.index import CarnotIndex
from carnot.memory.memory import Memory
from carnot.optimizer.model_ids import get_available_model_ids
from carnot.optimizer.optimizer import Optimizer
from carnot.plan import PhysicalPlan
from carnot.plan.feedback import PlanFeedback
from carnot.plan.node import PlanNode
from carnot.storage.catalog import IndexCatalog
from carnot.storage.config import StorageConfig
from carnot.storage.tiered import TieredStorageManager

logger = logging.getLogger('uvicorn.error')

# Maximum number of items to include in ExecutionProgress preview
_PREVIEW_ITEM_LIMIT = 5


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
            planning_stats: PhaseStats | None = None,
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
            available_model_ids: list[str] | None = None,
            max_workers: int = 64
        ):
        self.query = query
        self.datasets = datasets
        self.tools = tools or []
        self.conversation = conversation

        # If the caller supplies planning stats (e.g. from a prior plan()
        # call on a separate Execution instance), store them so that
        # run() / run_stream() can include them in the final
        # ExecutionStats.  When plan_stream() is called on *this*
        # instance it will overwrite _planning_stats as before.
        if planning_stats is not None:
            self._planning_stats = planning_stats
        self.memory = memory or Memory()
        self.indices = indices or []
        self.llm_config = llm_config or {}
        self.progress_log_file = progress_log_file
        self.cost_budget = cost_budget
        self.storage_config = storage_config or StorageConfig()
        self.max_workers = max_workers

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

        # get model for the planner and optimizer based on llm_config
        self.model_id = "openai/gpt-5-2025-08-07"
        self.api_key_name = "OPENAI_API_KEY"
        if "OPENAI_API_KEY" not in self.llm_config and "ANTHROPIC_API_KEY" in self.llm_config:
            self.model_id = "anthropic/claude-sonnet-4-5-20250929"
            self.api_key_name = "ANTHROPIC_API_KEY"
        elif "OPENAI_API_KEY" not in self.llm_config and "GEMINI_API_KEY" in self.llm_config:
            self.model_id = "gemini/gemini-2.5-flash"
            self.api_key_name = "GEMINI_API_KEY"
        elif "OPENAI_API_KEY" not in self.llm_config and "GOOGLE_API_KEY" in self.llm_config:
            self.model_id = "gemini/gemini-2.5-flash"
            self.api_key_name = "GOOGLE_API_KEY"

        # create instance of the planner with the appropriate model and API key based on llm_config
        self.planner = Planner(
            datasets=self.datasets,
            tools=self.tools,
            model=LiteLLMModel(model_id=self.model_id, api_key=llm_config.get(self.api_key_name))
        )

        # Resolve available model IDs for the optimizer.  If the caller
        # supplied an explicit list, use it.  Otherwise auto-detect from
        # the API keys present in llm_config (spanning cost/quality tiers
        # per provider).  Fall back to just the planner model when no
        # recognised keys are found.
        if available_model_ids is not None:
            self.available_model_ids = available_model_ids
        else:
            detected = get_available_model_ids(self.llm_config)
            self.available_model_ids = detected if detected else [self.model_id]

        self.optimizer = Optimizer(
            model=LiteLLMModel(model_id=self.model_id, api_key=llm_config.get(self.api_key_name)),
            available_model_ids=self.available_model_ids,
            llm_config=self.llm_config,
            max_workers=self.max_workers,
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
            if "nodes" in value:
                # Flat format produced by PhysicalPlan.to_dict()
                self._physical_plan = PhysicalPlan.from_dict(value)
            else:
                # Recursive format produced by Dataset.serialize()
                self._physical_plan = PhysicalPlan.from_plan_dict(
                    value, self.datasets, query=self.query,
                )
        else:
            self._physical_plan = None

    def _update_progress(self, progress_queue: queue.Queue | None, is_planning: bool = False, **kwargs) -> None:
        """Helper to put a PlanningProgress or ExecutionProgress update on the queue if it exists."""
        if progress_queue is None:
            return

        if is_planning:
            progress_event = PlanningProgress(**kwargs)
            progress_queue.put(progress_event.to_dict())

        else:
            progress_event = ExecutionProgress(**kwargs)
            progress_queue.put(progress_event.to_dict())

    # ------------------------------------------------------------------
    # Generate logical plan (and its natural language paraphrasing)
    # ------------------------------------------------------------------

    def plan(
        self,
        progress_queue: queue.Queue | None = None,
    ) -> tuple[str, dict]:
        """Generate a logical execution plan for *self.query*.

        The function performs two phases: first it generates a code-based logical plan,
        then it paraphrases that plan into natural language.  If the caller provides a
        *progress_queue*, intermediate progress updates (from both the Planner and its
        managed agents) are pushed to the queue as serialized ``PlanningProgress`` dicts.

        After both phases complete, LLM call statistics are collected
        and stored in ``self._planning_stats``.

        Requires:
            - ``self.query`` is a non-empty string.
            - ``self.datasets`` is a non-empty list.
            - *progress_queue*, if provided, is a thread-safe ``queue.Queue``.

        Returns:
            The ``(natural_language_plan, logical_plan_dict)`` tuple.

        Raises:
            AgentGenerationError: If the LLM fails to produce valid
            output during either phase.
        """
        plan_start = time.perf_counter()

        # generate the code-based logical plan
        logical_plan = self.planner.generate_logical_plan(
            self.query,
            conversation=self.conversation,
            progress_queue=progress_queue,
        )
        self._update_progress(
            progress_queue,
            is_planning=True,
            phase="logical_plan",
            message="Logical plan generated. Optimizing implementation...",
        )

        # generate a physical plan which satisfies the cost budget
        physical_plan = self.optimizer.optimize(logical_plan, self.cost_budget)
        self._update_progress(
            progress_queue,
            is_planning=True,
            phase="optimizing",
            message="Physical plan generated. Preparing plan summary...",
        )

        # translate the logical plan to natural language
        nl_plan = self.planner.paraphrase_plan(
            self.query,
            physical_plan,
            conversation=self.conversation,
            progress_queue=progress_queue,
        )
        self._update_progress(
            progress_queue,
            is_planning=True,
            phase="paraphrase",
            message="Plan summary complete.",
        )

        plan_wall_clock = time.perf_counter() - plan_start
        self._planning_stats = self._build_planning_stats(plan_wall_clock)

        # Store the physical plan so that callers who use the same
        # Execution instance for both plan() and run() don't need to
        # set _plan manually.
        self._plan = physical_plan

        return nl_plan, physical_plan

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
        on_item_complete: Callable[[], None] | None = None,
    ) -> tuple[dict[str, Dataset], OperatorStats | None]:
        """Execute a single plan node.

        This is the primitive that :meth:`run` composes and which external
        callers may use directly for node-by-node execution.

        After execution, the output ``Dataset`` is enriched with
        ``parents`` (resolved from the plan's parent node IDs) so
        that callers can inspect the lineage of any dataset in the
        store.

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

        # Resolve parent dataset_ids for operator calling convention
        parent_output_ids = [
            plan.get_node(pid).dataset_id
            for pid in node.parent_ids
        ]

        updated_store, op_stats = node.execute(
            datasets_store,
            self.llm_config,
            leaf_datasets={ds.name: ds for ds in self.datasets},
            storage=self.storage,
            index_catalog=self.index_catalog,
            parent_output_ids=parent_output_ids,
            on_item_complete=on_item_complete,
        )

        # Enrich the output dataset with parent references so callers
        # can inspect lineage (e.g. during debugging).
        output_ds = updated_store.get(node.dataset_id)
        if output_ds is not None and not output_ds.parents:
            output_ds.parents = [
                updated_store[pid]
                for pid in parent_output_ids
                if pid in updated_store
            ]

        return updated_store, op_stats

    # -- notebook / interactive plan helpers ---------------------------------

    def get_physical_plan(self) -> list[dict]:
        """Serialise the physical plan into a list of node descriptors.

        Delegates to ``PhysicalPlan.to_node_dicts()`` which produces a
        node descriptor for each node in topological order.

        Each descriptor is a dict with keys: ``node_id``,
        ``node_type`` (``"dataset"`` | ``"operator"`` | ``"reasoning"``),
        ``operator_name``, ``operator_type``, ``description``, ``code``,
        ``original_code``, ``params``, ``parent_dataset_ids``, and
        ``dataset_id``.

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

        output_dataset = updated.get(node.dataset_id)
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

    def _resolve_final_dataset(self, datasets_store: dict[str, Dataset]) -> Dataset | None:
        """Identify the final output dataset from the executed plan.

        Locates the terminal nodes of the physical plan (nodes with no
        children) and returns their merged output.  When a single
        terminal node exists its output dataset is returned directly.
        When multiple terminal nodes exist their items are concatenated
        into a new ``Dataset``.

        Requires:
            - ``self._physical_plan`` is not None.

        Returns:
            The final ``Dataset``, or ``None`` if no terminal node
            produced output in *datasets_store*.

        Raises:
            None.
        """
        terminal = self._physical_plan.terminal_nodes
        final_datasets = [
            datasets_store[n.dataset_id]
            for n in terminal
            if n.dataset_id in datasets_store
        ]

        if not final_datasets:
            return None

        if len(final_datasets) == 1:
            return final_datasets[0]

        # Merge items from multiple terminal datasets.
        merged_items: list[dict] = []
        merged_code_state: dict = {}
        for ds in final_datasets:
            merged_items.extend(ds.items)
            # Preserve code_state from reasoning nodes if present.
            if ds.code_state:
                merged_code_state.update(ds.code_state)

        return Dataset(
            name="final_result",
            annotation="Merged output from terminal plan nodes",
            items=merged_items,
            code_state=merged_code_state,
        )

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
        dataset_id = params.get("dataset_id", name)
        node_type = "reasoning" if operator_type == "Reasoning" else "operator"

        return PlanNode(
            node_id=node_id,
            node_type=node_type,
            operator_type=operator_type or None,
            name=name,
            description=description,
            params=dict(params),
            parent_ids=[],  # Will be set by insert_node()
            dataset_id=dataset_id,
        )

    # ------------------------------------------------------------------
    # Execute the physical plan and produce final results with stats
    # ------------------------------------------------------------------

    def _estimate_items_total(
        self, node: PlanNode, datasets_store: dict[str, Dataset],
    ) -> int:
        """Estimate the number of work items for progress tracking.

        For joins this is left * right; for other operators it is the
        input item count. Returns 0 when the count cannot be determined
        (e.g. for code / reasoning operator).
        """
        plan = self._physical_plan
        parent_ids = [plan.get_node(pid).dataset_id for pid in node.parent_ids]

        if node.operator_type == "Join" and len(parent_ids) >= 2:
            left_ds = datasets_store.get(parent_ids[0])
            right_ds = datasets_store.get(parent_ids[1])
            left_n = len(left_ds.items) if left_ds else 0
            right_n = len(right_ds.items) if right_ds else 0
            return left_n * right_n

        if node.operator_type == "GroupBy" and parent_ids:
            # Phase 1 has one future per item; phase 2 has one per group
            # (unknown ahead of time).  Use input size as a lower bound.
            ds = datasets_store.get(parent_ids[0])
            return len(ds.items) if ds else 0

        if parent_ids:
            ds = datasets_store.get(parent_ids[0])
            return len(ds.items) if ds else 0

        # for leaf nodes with no parents return the number of items in the dataset
        if node.node_type == "dataset":
            ds = datasets_store.get(node.dataset_id)
            return len(ds.items) if ds else 0

        return 0

    def run(
        self,
        progress_queue: queue.Queue | None = None,
        show_progress: bool = False,
    ) -> tuple[list[dict], str, ExecutionStats]:
        """Execute the physical plan and return the result with stats.

        Runs every node in topological order via :meth:`run_node`,
        collects ``OperatorStats`` from each, and assembles an
        ``ExecutionStats`` that combines planning and execution
        phase statistics.

        If a *progress_queue* is provided, yields intermediate progress updates
        as serialized ``ExecutionProgress`` dicts after each node completes,
        including operator stats, item counts, and output previews when available.

        When *show_progress* is ``True``, a Rich live table is rendered
        in the terminal showing operator status, item progress, latency,
        and running cost.

        Requires:
            - ``self._physical_plan`` is not None.
            - ``self.datasets`` is a non-empty list.

        Returns:
            A 3-tuple ``(items, answer_str, stats)`` where *items* is
            the list of result dicts, *answer_str* is the final
            answer text, and *stats* is the full ``ExecutionStats``.

        Raises:
            ValueError: If the plan contains unrecognized operators.
        """
        exec_start = time.perf_counter()
        datasets_store: dict[str, Dataset] = {}
        all_operator_stats: list[OperatorStats] = []
        nodes = self._physical_plan.topo_order()
        total = len(nodes)

        # Set up Rich progress display if requested.
        progress_display = None
        if show_progress:
            from carnot.execution.rich_progress import RichProgressDisplay
            progress_display = RichProgressDisplay()
            for node in nodes:
                progress_display.register_node(node.node_id, node.display_name())
            progress_display.start()

        self._update_progress(
            progress_queue,
            message=f"Starting execution -- {total} step(s) in plan.",
            total_operators=total,
        )

        try:
            for idx, node in enumerate(nodes):
                display = node.display_name()

                self._update_progress(
                    progress_queue,
                    message=f"Running step {idx + 1}/{total}: {display}...",
                    operator_index=idx,
                    total_operators=total,
                    operator_name=display,
                )

                # Determine input item count for progress tracking.
                item_callback = None
                items_total = 0
                if progress_display is not None:
                    items_total = self._estimate_items_total(node, datasets_store)
                    progress_display.mark_running(node.node_id, items_total)
                    item_callback = progress_display.make_item_callback(node.node_id)

                datasets_store, op_stats = self.run_node(
                    node.node_id, datasets_store, on_item_complete=item_callback,
                )
                if op_stats is not None:
                    all_operator_stats.append(op_stats)

                if progress_display is not None:
                    if op_stats is not None:
                        progress_display.mark_done(
                            node.node_id,
                            cost_usd=op_stats.total_cost_usd,
                            items_out=op_stats.items_out,
                        )
                    else:
                        progress_display.mark_skipped(node.node_id)

                # peek at the output dataset for item count / preview
                output_ds = datasets_store.get(node.dataset_id)
                item_count, preview = None, None
                if output_ds is not None and hasattr(output_ds, "items"):
                    try:
                        items_list = output_ds.items
                        item_count = len(items_list)
                        raw_preview = items_list[:_PREVIEW_ITEM_LIMIT]
                        preview = [
                            item if isinstance(item, dict) else {"value": str(item)}
                            for item in raw_preview
                        ]
                    except Exception:
                        pass  # Non-critical — skip preview on error

                self._update_progress(
                    progress_queue,
                    message=f"Completed step {idx + 1}/{total}: {display}.",
                    operator_index=idx,
                    total_operators=total,
                    operator_name=display,
                    step_cost_usd=op_stats.total_cost_usd if op_stats else None,
                    operator_stats=op_stats,
                    item_count=item_count,
                    preview_items=preview,
                )
        finally:
            if progress_display is not None:
                progress_display.stop()

        final_dataset = self._resolve_final_dataset(datasets_store)

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
