from __future__ import annotations

import textwrap
from collections.abc import Iterator
from random import sample as random_sample

from carnot.data.item import DataItem
from carnot.index.index import CarnotIndex
from carnot.storage.catalog import IndexCatalog
from carnot.storage.tiered import TieredStorageManager


class Dataset:
    """Unified dataset class inspired by Ray Data and PySpark.

    A Dataset is **lazy by default**:

    - Plan-building methods (``.sem_filter``, ``.sem_map``, etc.) return new
      ``Dataset`` nodes without executing anything.
    - ``.materialize()`` triggers execution of a leaf dataset and pins results.
    - ``.items`` is a lazy property: accessing it on an unmaterialized leaf
      returns ``DataItem`` references; on a materialized dataset it returns
      the already-parsed ``list[dict]``.

    Three usage patterns, one class:

    1. **Plan building** (Planner agent): ``ds.sem_filter(...).serialize()``
    2. **Data exploration** (DataDiscoveryAgent): ``ds.items``, ``ds.has_indices()``
    3. **Imperative scripting**: ``ds.sem_filter(...).materialize().items``

    Public attributes (always available after construction):

    - ``name`` (``str``): display name for this dataset node.
    - ``annotation`` (``str``): human-readable description.
    - ``dataset_id`` (``int | None``): optional numeric identifier.
    - ``parents`` (``list[Dataset]``): parent nodes in the plan DAG
      (empty for leaf datasets).
    - ``params`` (``dict``): operator parameters.  For derived datasets
      this always contains ``"operator"`` (the operator type string)
      plus operator-specific keys (see each operator method's docstring).
      For leaf datasets this is empty.
    - ``output_dataset_id`` (``str``): identifier for this node's output.

    Representation invariant:
        - ``is_materialized`` is ``True`` iff ``items`` returns
          ``list[dict]`` (not ``list[DataItem]``).
        - A Dataset constructed with ``list[dict]`` items is immediately
          materialized.
        - A Dataset constructed with ``list[DataItem]`` items (or no
          items) is **not** materialized.
        - Assigning ``list[dict]`` to ``.items`` marks the dataset as
          materialized; assigning ``list[DataItem]`` or ``None``
          marks it as not materialized.

    Abstraction function:
        Represents a node in a logical query plan.  Leaf nodes wrap raw
        data; derived nodes record the operator and its parameters
        together with references to parent nodes.
    """

    def __init__(
        self,
        name: str = "",
        annotation: str = "",
        items: list[DataItem] | list[dict] | None = None,
        indices: dict[str, CarnotIndex] | None = None,
        code: str | None = None,
        code_state: dict | None = None,
        parents: list[Dataset] | None = None,
        id_params: dict | None = None,
        storage: TieredStorageManager | None = None,
        index_catalog: IndexCatalog | None = None,
        dataset_id: int | None = None,
        **kwargs,
    ):
        # dataset metadata
        self.name = name
        self.annotation = annotation
        self.dataset_id = dataset_id

        # determine whether we were given pre-materialized dicts or DataItem refs
        if items and isinstance(items[0], dict):
            self._item_refs: list[DataItem] = []
            self._materialized_items: list[dict] | None = list(items)
            self._is_materialized = True
        else:
            self._item_refs = list(items) if items else []
            self._materialized_items = None
            self._is_materialized = False

        # code execution state (for code operators which write and execute code)
        self.code = code
        self.code_state = code_state or {}

        # index layer (lazy-loaded from catalog if not provided)
        self._indices: dict[str, CarnotIndex] | None = dict(indices) if indices else None

        # storage layer
        self._storage = storage
        self._index_catalog = index_catalog

        # plan graph
        self.parents = parents or []
        self.id_params = id_params or {
            "limit_id": 0,
            "merge_id": 0,
            "code_id": 0,
            "reason_id": 0,
            "sem_agg_id": 0,
            "sem_filter_id": 0,
            "sem_map_id": 0,
            "sem_flat_map_id": 0,
            "sem_groupby_id": 0,
            "sem_join_id": 0,
            "sem_topk_id": 0,
        }
        self.params = kwargs
        self.output_dataset_id = kwargs.get("output_dataset_id") or name

    # ═══════════════════════════════════════════════════════════════════
    # DATA ACCESS — Lazy materialization
    # ═══════════════════════════════════════════════════════════════════

    @property
    def items(self) -> list:
        """Lazy-loading property.

        - If materialized: returns ``list[dict]`` (already-parsed data).
        - If not materialized: returns ``list[DataItem]`` (lightweight refs).
        - If empty / uninitialized: returns ``[]``.

        The Planner never accesses this (zero cost during planning).
        The DataDiscoveryAgent accesses it for sampling (loads on first access).
        """
        if self._is_materialized:
            return self._materialized_items
        return self._item_refs

    @items.setter
    def items(self, value: list) -> None:
        """Setter — used by execution to inject materialized items."""
        if value and isinstance(value[0], dict):
            self._materialized_items = value
            self._is_materialized = True
        else:
            self._item_refs = value if value else []
            self._materialized_items = None
            self._is_materialized = False

    @property
    def is_materialized(self) -> bool:
        """Whether this dataset's items have been materialized (read + parsed)."""
        return self._is_materialized

    def sample(self, n: int, random: bool = True) -> list[dict]:
        """Return a sample of *n* items as dicts, materializing if needed.

        If random is true, returns a random sample; otherwise returns the first *n* items.
        For each item, the dataset will materialize it if it is not already.

        Requires:
            - *n* is a positive integer.

        Returns:
            A list of *n* items as dicts.  If the dataset has fewer than *n* items,
            returns all items.
        """
        if n <= 0:
            raise ValueError("Sample size n must be a positive integer.")

        sample_indices = (random_sample(range(len(self.items)), min(n, len(self.items))) if random else range(min(n, len(self.items))))
        sampled_items = []
        for idx in sample_indices:
            item = self.items[idx]
            if isinstance(item, DataItem):
                sampled_items.append(item.materialize(self._storage))
            elif isinstance(item, dict):
                sampled_items.append(item)
            else:
                sampled_items.append(item)

        return sampled_items

    def materialize(self, storage: TieredStorageManager | None = None) -> Dataset:
        """Trigger materialization and pin results.  Returns *self* for chaining.

        Inspired by Spark's ``df.persist()`` — materializes in-place and returns
        the same Dataset, now backed by computed results.

        For leaf datasets (no parents): reads items through the storage layer.
        For derived datasets: should be called by ``Execution.run()`` which
        handles the full DAG.

        If this dataset is already materialized, returns *self* immediately
        (no-op).

        Requires:
            - For leaf datasets: *storage* (or ``self._storage``) is a
              valid :class:`TieredStorageManager`.

        Returns:
            *self*, with ``is_materialized == True`` and ``items``
            returning ``list[dict]``.

        Raises:
            ValueError: if called on a derived dataset (one with
            parents).
        """
        if self._is_materialized:
            return self

        storage = storage or self._storage

        # leaf dataset: materialize items directly
        if not self.parents:
            materialized: list[dict] = []
            for item_ref in self._item_refs:
                if isinstance(item_ref, DataItem):
                    materialized.append(item_ref.materialize(storage))
                elif isinstance(item_ref, dict):
                    materialized.append(item_ref)
                else:
                    materialized.append(item_ref)
            self._materialized_items = materialized
            self._is_materialized = True
        else:
            raise ValueError(
                "Cannot materialize a derived Dataset directly. "
                "Use Execution.run() to execute the full plan."
            )

        return self

    # ═══════════════════════════════════════════════════════════════════
    # INDEX ACCESS — Lazy loading from catalog
    # ═══════════════════════════════════════════════════════════════════

    @property
    def indices(self) -> dict[str, CarnotIndex]:
        """Lazy-loading indices.  Loaded from IndexCatalog on first access."""
        if self._indices is None:
            self._indices = {}
            if self._index_catalog and self.dataset_id:
                index_metas = self._index_catalog.list_indices(self.dataset_id)
                for meta in index_metas:
                    if not meta.is_stale:
                        loaded = self._index_catalog.load_index(meta.id)
                        if loaded is not None:
                            self._indices[meta.name] = loaded
        return self._indices

    @indices.setter
    def indices(self, value: dict[str, CarnotIndex]) -> None:
        self._indices = value

    def has_indices(self) -> bool:
        """Return True if any index exists."""
        return bool(self.indices)

    def list_indices(self) -> list[str]:
        """Return names of available indices."""
        return list(self.indices.keys())

    def get_indices_info(self) -> list[dict]:
        """Return info about available indices.

        Returns:
            A list of dicts, each with ``"index_name"``,
            ``"index_type"`` (the class name), and ``"description"``.

        Raises:
            None.
        """
        return [
            {
                "index_name": name,
                "index_type": index.__class__.__name__,
                "description": index.description,
            }
            for name, index in self.indices.items()
        ]

    def get_index_type_string(self, index_name: str) -> str:
        """Return the class name for a given index name.

        Requires:
            - *index_name* exists in ``self.indices``.

        Returns:
            The class name string (e.g. ``"ChromaIndex"``).

        Raises:
            ValueError: if *index_name* is not found.
        """
        index = self.indices.get(index_name)
        if index is None:
            raise ValueError(f"Index '{index_name}' not found in dataset '{self.name}'")
        return index.__class__.__name__

    # ═══════════════════════════════════════════════════════════════════
    # PLAN BUILDING — Creation of logical plan through lazy operators
    # ═══════════════════════════════════════════════════════════════════

    def serialize(self) -> dict:
        """Serialize the logical plan structure of this dataset.

        Returns:
            A ``dict`` with keys:
            - ``"name"``: the dataset's name (``str``).
            - ``"output_dataset_id"``: the output dataset id (``str``).
            - ``"params"``: the operator params ``dict`` (empty for leaf nodes).
            - ``"parents"``: a ``list`` of serialized parent dicts
              (empty for leaf nodes, one entry per parent otherwise).
        """
        return {
            "name": self.name,
            "output_dataset_id": self.output_dataset_id,
            "params": self.params,
            "parents": [p.serialize() for p in self.parents],
        }

    def format_description(self, code_block_tags: list[str]) -> str:
        code_str = " None" if self.code is None else f"\n{code_block_tags[0]}\n{self.code}\n{code_block_tags[1]}"
        index_info = "yes" if self.has_indices() else "no"
        if self.indices:
            index_info += f" ({', '.join(self.list_indices())})"
        return textwrap.dedent(
            f"Dataset Name: {self.name}\n"
            f"Annotation: {self.annotation}\n"
            f"Number of Items: {len(self.items)}\n"
            f"Indices: {index_info}\n"
            f"Code that Generated Code State: {code_str}\n"
            f"Available Code State Vars: {list(self.code_state.keys())}\n"
        )

    def __iter__(self) -> Iterator[DataItem]:
        return iter(self.items)

    # ── Semantic and Non-Semantic Operators ──────────────────────────────────────────────

    def limit(self, n: int) -> Dataset:
        """Return a new derived Dataset representing a limit operation.

        The child's ``params`` dict contains:
        - ``"operator": "Limit"``
        - ``"n"``: the *n* integer.

        The child's ``name`` is ``"LimitOperation{n}"``.

        Returns:
            A new :class:`Dataset` whose sole parent is *self*.
        """
        limited_name = f"LimitOperation{self.id_params['limit_id'] + 1}"
        self.id_params["limit_id"] += 1
        params = {"operator": "Limit", "description": f"Limited {self.name} to first {n} records", "n": n}
        return Dataset(
            name=limited_name,
            annotation=f"Limited version of ({self.annotation})",
            parents=[self],
            id_params=self.id_params,
            output_dataset_id=limited_name,
            **params,
        )

    # TODO: implement logical and physical operators for merge (and rename to `union`)
    # def merge(self, other: Dataset) -> Dataset:
    #     """Merge this dataset with another dataset."""
    #     merged_name = f"MergeOperation{self.id_params['merge_id'] + 1}"
    #     self.id_params["merge_id"] += 1
    #     params = {"operator": "Merge", "description": f"Merged {self.name} with {other.name}"}
    #     return Dataset(
    #         name=merged_name,
    #         annotation=f"Merge of ({self.annotation}) and ({other.annotation})",
    #         parents=[self, other],
    #         id_params=self.id_params,
    #         output_dataset_id=merged_name,
    #         **params,
    #     )

    def write_code(self, task: str) -> Dataset:
        """Return a new derived Dataset representing a code operation.

        The child's ``params`` dict contains:
        - ``"operator": "Code"``
        - ``"task"``: the *task* string.

        The child's ``name`` is ``"CodeOperation{n}"``.

        Returns:
            A new :class:`Dataset` whose sole parent is *self*.
        """
        coded_name = f"CodeOperation{self.id_params['code_id'] + 1}"
        self.id_params["code_id"] += 1
        params = {"operator": "Code", "description": f"Coded {self.name} for task: {task}", "task": task}
        return Dataset(
            name=coded_name,
            annotation=f"Code operation on ({self.annotation})",
            parents=[self],
            id_params=self.id_params,
            output_dataset_id=coded_name,
            **params,
        )

    def reason(self, task: str) -> Dataset:
        """Apply a reasoning operation to the dataset."""
        reasoned_name = f"ReasonOperation{self.id_params['reason_id'] + 1}"
        self.id_params["reason_id"] += 1
        params = {"operator": "Reason", "description": f"Reasoned {self.name} for task: {task}", "task": task}
        return Dataset(
            name=reasoned_name,
            annotation=f"Reasoning operation on ({self.annotation})",
            parents=[self],
            id_params=self.id_params,
            output_dataset_id=reasoned_name,
            **params,
        )

    def sem_aggregate(self, task: str, agg_fields: list[dict]) -> Dataset:
        """Apply a semantic aggregation."""
        agg_name = f"AggregateOperation{self.id_params['sem_agg_id'] + 1}"
        self.id_params["sem_agg_id"] += 1
        for field_dict in agg_fields:
            field_dict["type"] = field_dict["type"].__name__
        params = {"operator": "SemanticAgg", "description": f"Aggregated {self.name} on fields: {agg_fields}", "task": task, "agg_fields": agg_fields}
        return Dataset(
            name=agg_name,
            annotation=f"Aggregation on ({self.annotation})",
            parents=[self],
            id_params=self.id_params,
            output_dataset_id=agg_name,
            **params,
        )

    def sem_filter(self, condition: str) -> Dataset:
        """Return a new derived Dataset representing a semantic filter.

        The child's ``params`` dict contains:
        - ``"operator": "SemanticFilter"``
        - ``"condition"``: the *condition* string.

        The child's ``name`` is ``"FilterOperation{n}"``.

        Returns:
            A new :class:`Dataset` whose sole parent is *self*.
        """
        filtered_name = f"FilterOperation{self.id_params['sem_filter_id'] + 1}"
        self.id_params["sem_filter_id"] += 1
        params = {"operator": "SemanticFilter", "description": f"Filtered {self.name} by condition: {condition}", "condition": condition}
        return Dataset(
            name=filtered_name,
            annotation=f"Filtered ({self.annotation})",
            parents=[self],
            id_params=self.id_params,
            output_dataset_id=filtered_name,
            **params,
        )

    def sem_map(self, field: str, type: type, description: str) -> Dataset:
        """Return a new derived Dataset representing a semantic map.

        The child's ``params`` dict contains:
        - ``"operator": "SemanticMap"``
        - ``"field"``: the *field* name (``str``).
        - ``"type"``: ``type.__name__`` (``str``).
        - ``"field_desc"``: the *description* string.

        The child's ``name`` is ``"MapOperation{n}"``.

        Returns:
            A new :class:`Dataset` whose sole parent is *self*.
        """
        mapped_name = f"MapOperation{self.id_params['sem_map_id'] + 1}"
        self.id_params["sem_map_id"] += 1
        params = {
            "operator": "SemanticMap",
            "description": f"Created field {field} with type {type.__name__} and description {description}",
            "field": field,
            "type": type.__name__,
            "field_desc": description,
        }
        return Dataset(
            name=mapped_name,
            annotation=f"Mapped ({self.annotation})",
            parents=[self],
            id_params=self.id_params,
            output_dataset_id=mapped_name,
            **params,
        )

    def sem_flat_map(self, field: str, type: type, description: str) -> Dataset:
        """Return a new derived Dataset representing a semantic flat map.

        The child's ``params`` dict contains:
        - ``"operator": "SemanticFlatMap"``
        - ``"field"``: the *field* name (``str``).
        - ``"type"``: ``type.__name__`` (``str``).
        - ``"field_desc"``: the *description* string.

        The child's ``name`` is ``"FlatMapOperation{n}"``.

        Returns:
            A new :class:`Dataset` whose sole parent is *self*.
        """
        flat_mapped_name = f"FlatMapOperation{self.id_params['sem_flat_map_id'] + 1}"
        self.id_params["sem_flat_map_id"] += 1
        params = {
            "operator": "SemanticFlatMap",
            "description": f"Flat mapped field {field} with type {type.__name__} and description {description}",
            "field": field,
            "type": type.__name__,
            "field_desc": description,
        }
        return Dataset(
            name=flat_mapped_name,
            annotation=f"Flat mapped ({self.annotation})",
            parents=[self],
            id_params=self.id_params,
            output_dataset_id=flat_mapped_name,
            **params,
        )

    def sem_groupby(self, gby_fields: list[dict], agg_fields: list[dict]) -> Dataset:
        """Apply a semantic group by operation."""
        gby_name = f"GroupByOperation{self.id_params['sem_groupby_id'] + 1}"
        self.id_params["sem_groupby_id"] += 1
        gby_field_names = [field['name'] for field in gby_fields]
        agg_field_names = [field['name'] for field in agg_fields]
        for field_dict in gby_fields:
            field_dict["type"] = field_dict["type"].__name__
        for field_dict in agg_fields:
            field_dict["type"] = field_dict["type"].__name__
        params = {
            "operator": "SemanticGroupBy",
            "description": f"Grouped {self.name} by fields {gby_field_names} with aggregations on {agg_field_names}",
            "gby_fields": gby_fields,
            "agg_fields": agg_fields,
        }
        return Dataset(
            name=gby_name,
            annotation=f"Grouped ({self.annotation})",
            parents=[self],
            id_params=self.id_params,
            output_dataset_id=gby_name,
            **params,
        )

    def sem_join(self, other: Dataset, condition: str) -> Dataset:
        """Return a new derived Dataset representing a semantic join.

        The child's ``params`` dict contains:
        - ``"operator": "SemanticJoin"``
        - ``"condition"``: the *condition* string.

        The child's ``name`` is ``"JoinOperation{n}"``.

        Returns:
            A new :class:`Dataset` whose parents are ``[self, other]``.
        """
        joined_name = f"JoinOperation{self.id_params['sem_join_id'] + 1}"
        self.id_params["sem_join_id"] += 1
        params = {
            "operator": "SemanticJoin",
            "description": f"Joined {self.name} with {other.name} on condition: {condition}",
            "condition": condition,
        }
        return Dataset(
            name=joined_name,
            annotation=f"Join of ({self.annotation}) and ({other.annotation})",
            parents=[self, other],
            id_params=self.id_params,
            output_dataset_id=joined_name,
            **params,
        )

    def sem_topk(self, index_name: str, search_str: str, k: int = 5) -> Dataset:
        """Return a new derived Dataset representing a semantic top-k operation.

        The child's ``params`` dict contains:
        - ``"operator": "SemanticTopK"``
        - ``"index_name"``: the *index_name* string (passed through as-is;
          validation against available index types happens at the physical
          operator layer, not here).
        - ``"search_str"``: the *search_str* string.
        - ``"k"``: the *k* integer.

        The child's ``name`` is ``"TopKOperation{n}"``.

        Returns:
            A new :class:`Dataset` whose sole parent is *self*.
        """
        top_k_name = f"TopKOperation{self.id_params['sem_topk_id'] + 1}"
        self.id_params["sem_topk_id"] += 1
        params = {
            "operator": "SemanticTopK",
            "index_name": index_name,
            "description": f"Top-{k} items from {self.name} for search string: {search_str}",
            "search_str": search_str,
            "k": k,
        }

        return Dataset(
            name=top_k_name,
            annotation=f"Top-{k} from ({self.annotation})",
            parents=[self],
            id_params=self.id_params,
            output_dataset_id=top_k_name,
            **params,
        )
