"""Plan node: a single step in a physical execution plan.

This module defines :class:`PlanNode`, the fundamental building block of a
:class:`PhysicalPlan`.  Each node represents one concrete step — loading a
dataset, running a semantic/code operator, or performing final reasoning —
and knows how to instantiate its operator, generate display code, and
serialise itself to a cell descriptor for the notebook frontend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from carnot.core.models import OperatorStats
    from carnot.data.dataset import Dataset
    from carnot.storage.catalog import IndexCatalog
    from carnot.storage.tiered import TieredStorageManager

# ---------------------------------------------------------------------------
# Display name mapping (keyed by operator type *string*, not class)
# ---------------------------------------------------------------------------

OPERATOR_DISPLAY_NAMES: dict[str, str] = {
    "SemanticFilter": "Semantic Filter",
    "SemanticMap": "Semantic Map",
    "SemanticFlatMap": "Semantic Flat Map",
    "SemanticGroupBy": "Semantic Group By",
    "SemanticJoin": "Semantic Join",
    "SemanticTopK": "Semantic Top-K",
    "SemanticAgg": "Semantic Aggregation",
    "Code": "Code",
    "Reasoning": "Reasoning",
    "Limit": "Limit",
}


class PlanNode:
    """A single node in a physical execution plan.

    Representation invariant:
        - ``node_id`` is unique within its parent ``PhysicalPlan``.
        - ``node_type`` is one of ``"dataset"``, ``"operator"``, or
          ``"reasoning"``.
        - If ``node_type == "dataset"``, then ``operator_type is None``
          and ``parent_ids`` is empty.
        - If ``node_type == "operator"``, then ``operator_type`` is a
          valid operator type string and ``parent_ids`` has at least one
          element.
        - If ``node_type == "reasoning"``, then ``parent_ids`` has
          exactly one element.

    Abstraction function:
        Represents one step of a physical query plan: either loading a
        dataset, running a semantic/code operator, or performing final
        reasoning.  The node knows its own parameters and can instantiate
        its corresponding operator, generate its display code, and
        serialise itself to a cell descriptor.
    """
    def __init__(self, node_id: str, node_type: str, operator_type: str | None, name: str, description: str, params: dict | None = None, parent_ids: list[str] | None = None, output_dataset_id: str = ""):
         self.node_id = node_id
         self.node_type = node_type          # "dataset" | "operator" | "reasoning"
         self.operator_type = operator_type  # "SemanticFilter", "Code", etc.; None for datasets
         self.name = name                    # Human-readable display name
         self.description = description
         self.params = params or {}
         self.parent_ids = parent_ids or []
         self.output_dataset_id = output_dataset_id

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def display_name(self) -> str:
        """Return a human-readable label for this node.

        Requires:
            None.

        Returns:
            A short display string such as ``"Semantic Filter"`` or
            ``"Dataset: Movies"``.

        Raises:
            None.
        """
        if self.node_type == "dataset":
            return f"Dataset: {self.name}"
        return OPERATOR_DISPLAY_NAMES.get(
            self.operator_type or "", self.operator_type or self.name,
        )

    # ------------------------------------------------------------------
    # Operator instantiation
    # ------------------------------------------------------------------

    def to_operator(
        self,
        llm_config: dict,
        *,
        index_catalog: IndexCatalog | None = None,
    ) -> Any:
        """Instantiate the physical operator for this node.

        Requires:
            - ``self.node_type`` is ``"operator"`` or ``"reasoning"``.
            - ``llm_config`` contains the necessary API keys.

        Returns:
            An operator instance ready to be called.

        Raises:
            ValueError: If ``self.node_type`` is ``"dataset"`` (use
            ``execute()`` instead) or ``operator_type`` is unrecognised.
        """
        # Deferred imports to avoid circular dependencies at module level.
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

        if self.node_type == "dataset":
            raise ValueError(
                "Cannot create an operator for a dataset node.  "
                "Use execute() or handle dataset nodes directly."
            )

        op = self.operator_type
        params = self.params
        out = self.output_dataset_id

        if op == "Code":
            return CodeOperator(
                task=params["task"], output_dataset_id=out,
                model_id="openai/gpt-5-mini", llm_config=llm_config,
            )

        if op == "Limit":
            return LimitOperator(n=params["n"], output_dataset_id=out)

        if op == "SemanticAgg":
            return SemAggOperator(
                task=params["task"], agg_fields=params["agg_fields"],
                output_dataset_id=out, model_id="openai/gpt-5-mini",
                llm_config=llm_config, max_workers=64,
            )

        if op == "SemanticFilter":
            return SemFilterOperator(
                task=params["condition"], output_dataset_id=out,
                model_id="openai/gpt-5-mini", llm_config=llm_config,
                max_workers=64,
            )

        if op == "SemanticMap":
            output_fields = [{
                "name": params["field"], "type": params["type"],
                "description": params["field_desc"],
            }]
            return SemMapOperator(
                task="Execute the map operation to compute the following output field.",
                output_fields=output_fields, output_dataset_id=out,
                model_id="openai/gpt-5-mini", llm_config=llm_config,
                max_workers=64,
            )

        if op == "SemanticFlatMap":
            output_fields = [{
                "name": params["field"], "type": params["type"],
                "description": params["field_desc"],
            }]
            return SemFlatMapOperator(
                task="Execute the flat map operation to compute the following output field.",
                output_fields=output_fields, output_dataset_id=out,
                model_id="openai/gpt-5-mini", llm_config=llm_config,
                max_workers=64,
            )

        if op == "SemanticGroupBy":
            gby_field_names = [f["name"] for f in params["gby_fields"]]
            agg_field_names = [f["name"] for f in params["agg_fields"]]
            agg_funcs = [f["func"] for f in params["agg_fields"]]
            task = (
                f"Group by fields {gby_field_names} with aggregations on "
                f"{agg_field_names} using {agg_funcs} for each aggregation "
                f"field, respectively."
            )
            return SemGroupByOperator(
                task=task, group_by_fields=params["gby_fields"],
                agg_fields=params["agg_fields"], output_dataset_id=out,
                model_id="openai/gpt-5-mini", llm_config=llm_config,
                max_workers=64,
            )

        if op == "SemanticJoin":
            return SemJoinOperator(
                task=params["condition"], output_dataset_id=out,
                model_id="openai/gpt-5-mini", llm_config=llm_config,
                max_workers=64,
            )

        if op == "SemanticTopK":
            return SemTopKOperator(
                task=params["search_str"], k=params["k"], output_dataset_id=out,
                model_id="openai/text-embedding-3-small",
                llm_config=llm_config, max_workers=64,
                index_name=params["index_name"], catalog=index_catalog,
            )

        if op == "Reasoning" or self.node_type == "reasoning":
            from carnot.operators.reasoning import ReasoningOperator
            return ReasoningOperator(
                task=params.get("task", self.description),
                output_dataset_id=out,
                model_id="openai/gpt-5-mini", llm_config=llm_config,
            )

        raise ValueError(f"Unknown operator type: {op}")

    # ------------------------------------------------------------------
    # Serialization to dict
    # ------------------------------------------------------------------

    def to_dict(self, *, parent_output_map: dict[str, str] | None = None) -> dict:
        """Serialise to a dict

        Requires:
            None.

        Returns:
            A dict with keys: ``node_id``, ``node_type``,
            ``operator_name``, ``operator_type``, ``description``,
            ``code``, ``original_code``, ``params``,
            ``parent_dataset_ids``, ``output_dataset_id``.

        Raises:
            None.
        """
        code = self.to_code(parent_output_map=parent_output_map)
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "operator_name": self.display_name(),
            "operator_type": self.operator_type,
            "description": self.description,
            "code": code,
            "original_code": code,
            "params": self.params,
            "parent_dataset_ids": self.parent_ids,
            "output_dataset_id": self.output_dataset_id,
        }

    # ------------------------------------------------------------------
    # Code generation (pseudocode for display)
    # ------------------------------------------------------------------

    @staticmethod
    def _smart_quote(value: str) -> str:
        """Wrap *value* in quotes, choosing a delimiter that avoids conflicts.

        Requires:
            None.

        Returns:
            A quoted string using single quotes if *value* contains no
            single quotes, double quotes if it contains single but not
            double, or escaped-single-quote form otherwise.

        Raises:
            None.
        """
        if "'" not in value:
            return f"'{value}'"
        if '"' not in value:
            return f'"{value}"'
        escaped = value.replace("'", "\\'")
        return f"'{escaped}'"

    def to_code(self, *, parent_output_map: dict[str, str] | None = None) -> str:
        """Generate readable pseudocode for this node.

        Requires:
            None.

        Returns:
            A multi-line Python-like string describing the operation.

        Raises:
            None.
        """
        def _resolve_parent(pid: str) -> str:
            """Map a parent node ID to the parent's output_dataset_id."""
            if parent_output_map:
                return parent_output_map.get(pid, pid)
            return pid
        if self.node_type == "dataset":
            safe_name = self.name.replace("'", "\\'")
            return (
                f"import carnot\n"
                f"\n"
                f"# Load dataset: {self.name}\n"
                f"datasets['{safe_name}'] = carnot.load_dataset(\"{safe_name}\")"
            )

        if self.node_type == "reasoning":
            query = self.params.get("task", self.description)
            safe_query = query.replace('"', '\\"')
            return (
                f"# Final Reasoning\n"
                f"# Synthesize results to answer: {query}\n"
                f"datasets['final_dataset'] = reasoning_operator(\n"
                f"    query=\"{safe_query}\",\n"
                f"    datasets=datasets\n"
                f")"
            )

        op = self.operator_type or ""
        out = self.output_dataset_id
        p = self.params
        parent_ids = self.parent_ids
        display = self.display_name()
        lines: list[str] = [f"# {display}"]

        if op == "Code":
            task = p.get('task', '')
            # Format comment: prefix each line with #
            for task_line in task.splitlines():
                lines.append(f"# Task: {task_line}" if task_line == task.splitlines()[0] else f"#   {task_line}")
            # Use triple-quotes for the task parameter to handle newlines
            if '\n' in task:
                lines.append(f"datasets['{out}'] = code_operator(")
                lines.append(f'    task="""\n{task}\n""",')  
                lines.append("    datasets=datasets")
                lines.append(")")
            else:
                safe_task = task.replace('"', '\\"')
                lines.append(f"datasets['{out}'] = code_operator(")
                lines.append(f"    task=\"{safe_task}\",")
                lines.append("    datasets=datasets")
                lines.append(")")

        elif op == "Limit":
            parent = _resolve_parent(parent_ids[0]) if parent_ids else "?"
            lines.append(
                f"datasets['{out}'] = limit(datasets['{parent}'], "
                f"n={p.get('n', '?')})"
            )

        elif op == "SemanticFilter":
            parent = _resolve_parent(parent_ids[0]) if parent_ids else "?"
            lines.append(f"# Condition: {p.get('condition', '')}")
            lines.append(f"datasets['{out}'] = sem_filter(")
            lines.append(f"    dataset=datasets['{parent}'],")
            lines.append(f"    condition={self._smart_quote(p.get('condition', ''))}")
            lines.append(")")

        elif op == "SemanticMap":
            parent = _resolve_parent(parent_ids[0]) if parent_ids else "?"
            lines.append(
                f"# Map field: {p.get('field', '')} ({p.get('type', '')})"
            )
            lines.append(f"# Description: {p.get('field_desc', '')}")
            lines.append(f"datasets['{out}'] = sem_map(")
            lines.append(f"    dataset=datasets['{parent}'],")
            lines.append(f"    field='{p.get('field', '')}',")
            lines.append(f"    type='{p.get('type', '')}',")
            lines.append(f"    description={self._smart_quote(p.get('field_desc', ''))}")
            lines.append(")")

        elif op == "SemanticFlatMap":
            parent = _resolve_parent(parent_ids[0]) if parent_ids else "?"
            lines.append(
                f"# Flat map field: {p.get('field', '')} ({p.get('type', '')})"
            )
            lines.append(f"# Description: {p.get('field_desc', '')}")
            lines.append(f"datasets['{out}'] = sem_flat_map(")
            lines.append(f"    dataset=datasets['{parent}'],")
            lines.append(f"    field='{p.get('field', '')}',")
            lines.append(f"    type='{p.get('type', '')}',")
            lines.append(f"    description={self._smart_quote(p.get('field_desc', ''))}")
            lines.append(")")

        elif op == "SemanticGroupBy":
            parent = _resolve_parent(parent_ids[0]) if parent_ids else "?"
            gby = [f["name"] for f in p.get("gby_fields", [])]
            agg = [
                f"{f['name']}({f.get('func', '?')})"
                for f in p.get("agg_fields", [])
            ]
            lines.append(f"datasets['{out}'] = sem_group_by(")
            lines.append(f"    dataset=datasets['{parent}'],")
            lines.append(f"    group_by={gby},")
            lines.append(f"    aggregations={agg}")
            lines.append(")")

        elif op == "SemanticJoin":
            left = _resolve_parent(parent_ids[0]) if len(parent_ids) > 0 else "?"
            right = _resolve_parent(parent_ids[1]) if len(parent_ids) > 1 else "?"
            lines.append(f"# Condition: {p.get('condition', '')}")
            lines.append(f"datasets['{out}'] = sem_join(")
            lines.append(f"    left=datasets['{left}'],")
            lines.append(f"    right=datasets['{right}'],")
            lines.append(f"    condition={self._smart_quote(p.get('condition', ''))}")
            lines.append(")")

        elif op == "SemanticTopK":
            parent = _resolve_parent(parent_ids[0]) if parent_ids else "?"
            lines.append(f"# Search: {p.get('search_str', '')}")
            lines.append(f"datasets['{out}'] = sem_topk(")
            lines.append(f"    dataset=datasets['{parent}'],")
            lines.append(f"    search={self._smart_quote(p.get('search_str', ''))},")
            lines.append(f"    k={p.get('k', '?')}")
            lines.append(")")

        elif op == "SemanticAgg":
            parent = _resolve_parent(parent_ids[0]) if parent_ids else "?"
            lines.append(f"# Task: {p.get('task', '')}")
            lines.append(f"datasets['{out}'] = sem_agg(")
            lines.append(f"    dataset=datasets['{parent}'],")
            lines.append(f"    agg_fields={p.get('agg_fields', [])}")
            lines.append(")")

        else:
            lines.append(f"datasets['{out}'] = {op}(datasets)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Unified execution
    # ------------------------------------------------------------------

    def execute(
        self,
        datasets_store: dict[str, Dataset],
        llm_config: dict,
        *,
        leaf_datasets: dict[str, Dataset] | None = None,
        storage: TieredStorageManager | None = None,
        index_catalog: IndexCatalog | None = None,
        parent_output_ids: list[str] | None = None,
    ) -> tuple[dict[str, Dataset], OperatorStats | None]:
        """Execute this node against the datasets store.

        This is the unified entry point that absorbs the three-way
        dispatch (CodeOperator / SemJoinOperator / default) that was
        previously replicated across ``run()``, ``run_stream()``, and
        ``execute_cell()``.

        Requires:
            - For dataset nodes: the corresponding ``Dataset`` object
              exists in *leaf_datasets*.
            - For operator/reasoning nodes: all datasets referenced by
              *parent_output_ids* exist in *datasets_store*.

        Returns:
            ``(updated_store, op_stats)`` where ``updated_store`` is
            ``datasets_store`` merged with the new output dataset, and
            ``op_stats`` is ``None`` for dataset-load nodes.

        Raises:
            ValueError: If operator type is unrecognised.
            KeyError: If a required parent dataset is missing.
        """
        if self.node_type == "dataset":
            leaf_datasets = leaf_datasets or {}
            ds = leaf_datasets.get(self.name)
            if ds is None:
                raise KeyError(
                    f"Dataset '{self.name}' not found in leaf_datasets"
                )
            materialized = ds.materialize(storage)
            return {
                **datasets_store,
                self.output_dataset_id: materialized,
            }, None

        op = self.to_operator(llm_config, index_catalog=index_catalog)
        parent_output_ids = parent_output_ids or []

        # CodeOperator and ReasoningOperator take the full store.
        if self.operator_type == "Code" or self.node_type == "reasoning":
            updated, stats = op(datasets_store)
            return {**datasets_store, **updated}, stats

        # SemJoinOperator takes two explicit parent IDs.
        if self.operator_type == "SemanticJoin":
            left_id = parent_output_ids[0] if len(parent_output_ids) > 0 else ""
            right_id = parent_output_ids[1] if len(parent_output_ids) > 1 else ""
            updated, stats = op(left_id, right_id, datasets_store)
            return {**datasets_store, **updated}, stats

        # All other operators take a single parent ID.
        dataset_id = parent_output_ids[0] if parent_output_ids else ""
        updated, stats = op(dataset_id, datasets_store)
        return {**datasets_store, **updated}, stats
