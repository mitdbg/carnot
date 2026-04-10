from __future__ import annotations

import textwrap
from collections.abc import Iterator
from random import sample as random_sample

from carnot.data.item import DataItem
from carnot.index.index import CarnotIndex
from carnot.operators.logical import (
    Aggregate,
    Code,
    Filter,
    FlatMap,
    GroupBy,
    Join,
    Limit,
    LogicalOperator,
    Map,
    Reason,
    TopK,
)
from carnot.storage.catalog import IndexCatalog
from carnot.storage.tiered import TieredStorageManager


def _type_to_str(t: type | str) -> str:
    """Convert a type annotation to a JSON-safe string.

    Handles plain types (``str``, ``int``), parameterised generics
    (``list[str]``), and values that are already strings.

    Requires:
        - *t* is a type, a generic alias, or a string.

    Returns:
        A string representation (e.g. ``"str"``, ``"list[str]"``).

    Raises:
        None.
    """
    if isinstance(t, str):
        return t
    return getattr(t, "__name__", None) or str(t)


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

    - ``name`` (``str``): display name for this dataset node. The name is useful for
      producing human readable code in code operators (datasets are accessed by name).
    - ``annotation`` (``str``): human-readable description.
    - ``dataset_id`` (``str``): unique identifier for this dataset node in the
      plan DAG.  Defaults to ``name`` when not explicitly provided.
    - ``parents`` (``list[Dataset]``): parent nodes in the plan DAG
      (empty for leaf datasets).
    - ``operator`` (``LogicalOperator | None``): the logical operator that produces
      this dataset from its parents (``None`` for leaf datasets).

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
        data; derived nodes record the operator and parent datasets that produce them.
    """

    def __init__(
        self,
        name: str,
        annotation: str = "",
        items: list[DataItem] | list[dict] | None = None,
        indices: dict[str, CarnotIndex] | None = None,
        code: str | None = None,
        code_state: dict | None = None,
        parents: list[Dataset] | None = None,
        operator: LogicalOperator | None = None,
        storage: TieredStorageManager | None = None,
        index_catalog: IndexCatalog | None = None,
        dataset_id: str | None = None,
        op_counts: dict[str, int] | None = None,
    ):
        # dataset metadata
        self.name = name
        self.annotation = annotation
        self.dataset_id = dataset_id or name

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
        self.operator = operator
        self.op_counts = op_counts or {}

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
            - ``"dataset_id"``: the dataset id (``str``).
            - ``"operator"``: the logical operator that produces this dataset, serialized as a dict.
            - ``"parents"``: a ``list`` of serialized parent dicts
              (empty for leaf nodes, one entry per parent otherwise).
        """
        return {
            "name": self.name,
            "dataset_id": self.dataset_id,
            "operator": self.operator.serialize() if self.operator else {},
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
        op_count = self.op_counts.get("Limit", 1)
        operator = Limit(name=f"Limit{op_count}", n=n)
        return Dataset(
            name=operator.name,
            annotation=f"Limit {n} applied to ({self.name})",
            parents=[self],
            operator=operator,
            dataset_id=operator.name,
            op_counts={**self.op_counts, "Limit": op_count + 1},
        )

    def write_code(self, task: str) -> Dataset:
        """Return a new derived Dataset representing a code operation.

        The child's ``name`` is ``"Code{n}"``.

        Returns:
            A new :class:`Dataset` whose sole parent is *self*.
        """
        op_count = self.op_counts.get("Code", 1)
        operator = Code(name=f"Code{op_count}", task=task)
        return Dataset(
            name=operator.name,
            annotation=f"Code operation on ({self.name})",
            parents=[self],
            operator=operator,
            dataset_id=operator.name,
            op_counts={**self.op_counts, "Code": op_count + 1},
        )

    def reason(self, task: str) -> Dataset:
        """Return a new derived Dataset representing a reasoning operation.

        The child's ``name`` is ``"Reason{n}"``.

        Returns:
            A new :class:`Dataset` whose sole parent is *self*.
        """
        op_count = self.op_counts.get("Reason", 1)
        operator = Reason(name=f"Reason{op_count}", task=task)
        return Dataset(
            name=operator.name,
            annotation=f"Reasoning operation on ({self.name})",
            parents=[self],
            operator=operator,
            dataset_id=operator.name,
            op_counts={**self.op_counts, "Reason": op_count + 1},
        )

    def sem_aggregate(self, task: str, agg_fields: list[dict]) -> Dataset:
        """Return a new derived Dataset representing a semantic aggregation.

        The child's ``name`` is ``"Aggregate{n}"``.

        Returns:
            A new :class:`Dataset` whose sole parent is *self*.
        """
        for field_dict in agg_fields:
            field_dict["type"] = _type_to_str(field_dict["type"])

        op_count = self.op_counts.get("Aggregate", 1)
        operator = Aggregate(name=f"Aggregate{op_count}", agg_fields=agg_fields)
        return Dataset(
            name=operator.name,
            annotation=f"Aggregation on ({self.name})",
            parents=[self],
            operator=operator,
            dataset_id=operator.name,
            op_counts={**self.op_counts, "Aggregate": op_count + 1},
        )

    def sem_filter(self, condition: str) -> Dataset:
        """Return a new derived Dataset representing a semantic filter.

        The child's ``name`` is ``"Filter{n}"``.

        Returns:
            A new :class:`Dataset` whose sole parent is *self*.
        """
        op_count = self.op_counts.get("Filter", 1)
        operator = Filter(name=f"Filter{op_count}", filter=condition)
        return Dataset(
            name=operator.name,
            annotation=f"Filtered ({self.name})",
            parents=[self],
            operator=operator,
            dataset_id=operator.name,
            op_counts={**self.op_counts, "Filter": op_count + 1},
        )

    def sem_map(self, fields: list[dict]) -> Dataset:
        """Return a new derived Dataset representing a semantic map.

        The child's ``name`` is ``"Map{n}"``.

        Returns:
            A new :class:`Dataset` whose sole parent is *self*.
        """
        for field_dict in fields:
            field_dict["type"] = _type_to_str(field_dict["type"])

        op_count = self.op_counts.get("Map", 1)
        operator = Map(name=f"Map{op_count}", fields=fields, desc=f"Created fields: {fields}")
        return Dataset(
            name=operator.name,
            annotation=f"Mapped ({self.name})",
            parents=[self],
            operator=operator,
            dataset_id=operator.name,
            op_counts={**self.op_counts, "Map": op_count + 1},
        )

    def sem_flat_map(self, fields: list[dict]) -> Dataset:
        """Return a new derived Dataset representing a semantic flat map.

        The child's ``name`` is ``"FlatMap{n}"``.

        Returns:
            A new :class:`Dataset` whose sole parent is *self*.
        """
        for field_dict in fields:
            field_dict["type"] = _type_to_str(field_dict["type"])

        op_count = self.op_counts.get("FlatMap", 1)
        operator = FlatMap(name=f"FlatMap{op_count}", fields=fields, desc=f"Created fields: {fields}")
        return Dataset(
            name=operator.name,
            annotation=f"Flat mapped ({self.name})",
            parents=[self],
            operator=operator,
            dataset_id=operator.name,
            op_counts={**self.op_counts, "FlatMap": op_count + 1},
        )

    def sem_groupby(self, gby_fields: list[dict], agg_fields: list[dict]) -> Dataset:
        """Return a new derived Dataset representing a semantic group-by operation.

        The child's ``name`` is ``"GroupBy{n}"``.

        Returns:
            A new :class:`Dataset` whose sole parent is *self*.
        """
        for field_dict in gby_fields:
            field_dict["type"] = _type_to_str(field_dict["type"])
        for field_dict in agg_fields:
            field_dict["type"] = _type_to_str(field_dict["type"])

        op_count = self.op_counts.get("GroupBy", 1)
        operator = GroupBy(name=f"GroupBy{op_count}", gby_fields=gby_fields, agg_fields=agg_fields)
        return Dataset(
            name=operator.name,
            annotation=f"Grouped ({self.name})",
            parents=[self],
            operator=operator,
            dataset_id=operator.name,
            op_counts={**self.op_counts, "GroupBy": op_count + 1},
        )

    def sem_join(self, other: Dataset, condition: str) -> Dataset:
        """Return a new derived Dataset representing a semantic join.

        The child's ``name`` is ``"Join{n}"``.

        Returns:
            A new :class:`Dataset` whose parents are ``[self, other]``.
        """
        op_count = self.op_counts.get("Join", 1)
        operator = Join(name=f"Join{op_count}", condition=condition)
        return Dataset(
            name=operator.name,
            annotation=f"Join of ({self.name}) and ({other.name})",
            parents=[self, other],
            operator=operator,
            dataset_id=operator.name,
            op_counts={**self.op_counts, "Join": op_count + 1},
        )

    def sem_topk(self, index_name: str, search_str: str, k: int = 5) -> Dataset:
        """Return a new derived Dataset representing a semantic top-k operation.

        The child's ``name`` is ``"TopK{n}"``.

        Returns:
            A new :class:`Dataset` whose sole parent is *self*.
        """
        op_count = self.op_counts.get("TopK", 1)
        operator = TopK(name=f"TopK{op_count}", task=search_str, k=k, index_name=index_name)
        return Dataset(
            name=operator.name,
            annotation=f"Top-{k} from ({self.name})",
            parents=[self],
            operator=operator,
            dataset_id=operator.name,
            op_counts={**self.op_counts, "TopK": op_count + 1},
        )
