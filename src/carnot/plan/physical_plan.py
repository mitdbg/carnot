"""Physical plan: a DAG of :class:`PlanNode` objects.

This module defines :class:`PhysicalPlan`, a typed, serialisable,
inspectable plan DAG that owns its nodes and provides traversal,
serialisation, and (in later phases) mutation operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from carnot.plan.node import PlanNode

if TYPE_CHECKING:
    from carnot.data.dataset import Dataset


class PhysicalPlan:
    """A physical execution plan: a DAG of ``PlanNode`` objects.

    Representation invariant:
        - ``_nodes`` is a dict mapping ``node_id → PlanNode``.
        - The graph defined by ``PlanNode.parent_ids`` is a DAG (no
          cycles).
        - Every ``parent_id`` referenced by any node exists in
          ``_nodes``.
        - There is at most one reasoning node.

    Abstraction function:
        Represents a complete physical execution plan where each node
        is a concrete step (load data, run operator, reason) with
        explicit data-flow dependencies.
    """

    def __init__(
        self,
        nodes: dict[str, PlanNode] | None = None,
        query: str = "",
    ):
        self._nodes: dict[str, PlanNode] = nodes or {}
        self.query = query

    # ------------------------------------------------------------------
    # Construction from plan dict
    # ------------------------------------------------------------------

    @classmethod
    def from_plan_dict(
        cls,
        plan_dict: dict,
        leaf_datasets: list[Dataset],
        *,
        include_reasoning: bool = True,
        query: str = "",
    ) -> PhysicalPlan:
        """Construct a ``PhysicalPlan`` from a serialised plan dict.

        The *plan_dict* is the recursive structure produced by
        ``Dataset.serialize()``::

            {
                "name": str,
                "dataset_id": str,
                "operator": {
                    "logical_op_class_name": str,
                    ...operator params...
                },
                "parents": [ <plan_dict>, ... ],
            }

        The method performs a post-order traversal, creating one
        ``PlanNode`` per dict node.  Leaf nodes (empty ``operator``
        dict and matching a dataset name) become ``"dataset"`` nodes;
        ``Reason`` nodes become ``"reasoning"`` nodes; all others
        become ``"operator"`` nodes.  An optional trailing
        ``"reasoning"`` node is appended when the root is not already
        a reasoning node.

        Requires:
            - *plan_dict* is a valid recursive plan dict.
            - *leaf_datasets* contains every ``Dataset`` referenced as
              a leaf in the plan.

        Returns:
            A ``PhysicalPlan`` with one ``PlanNode`` per plan-dict
            node, plus an appended reasoning node if
            *include_reasoning* is ``True``.

        Raises:
            None.
        """
        dataset_names = {ds.name for ds in leaf_datasets}
        nodes: dict[str, PlanNode] = {}
        # Counter for generating unique node IDs.
        counter = {"n": 0}

        def _walk(d: dict) -> str:
            """Recursively create PlanNodes; return the node_id."""
            # Process parents first (post-order).
            parent_node_ids: list[str] = []
            for parent_dict in d.get("parents", []):
                parent_node_ids.append(_walk(parent_dict))

            node_id = f"node-{counter['n']}"
            counter["n"] += 1

            op_dict = d.get("operator", {})
            op_class_name = op_dict.get("logical_op_class_name", "")
            name = d.get("name", "")
            dataset_id = d.get("dataset_id", name)

            # Determine node type.
            if not op_class_name and name in dataset_names:
                node_type = "dataset"
                operator_type = None
                description = f"Load dataset: {name}"
            elif op_class_name == "Reason":
                node_type = "reasoning"
                operator_type = "Reason"
                description = op_dict.get("task", name)
            else:
                node_type = "operator"
                operator_type = op_class_name
                description = op_dict.get("desc") or op_dict.get("task") or name

            node = PlanNode(
                node_id=node_id,
                node_type=node_type,
                operator_type=operator_type,
                name=name,
                description=description,
                params=dict(op_dict),
                parent_ids=list(parent_node_ids),
                dataset_id=dataset_id,
            )
            nodes[node_id] = node
            return node_id

        root_id = _walk(plan_dict)

        # Optionally append a reasoning node.
        if include_reasoning and nodes[root_id].node_type != "reasoning":
            reasoning_id = f"node-{counter['n']}"
            counter["n"] += 1
            reasoning_node = PlanNode(
                node_id=reasoning_id,
                node_type="reasoning",
                operator_type="Reason",
                name="Reasoning",
                description="Generate final answer",
                params={"task": query} if query else {},
                parent_ids=[root_id],
                dataset_id="final_dataset",
            )
            nodes[reasoning_id] = reasoning_node

        return cls(nodes=nodes, query=query)

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def topo_order(self) -> list[PlanNode]:
        """Return nodes in topological (dependency-safe) order.

        Uses Kahn's algorithm.  Nodes with no parents come first;
        nodes that depend on others come after their dependencies.

        Requires:
            The graph is a DAG (representation invariant).

        Returns:
            A list of ``PlanNode`` objects in dependency-safe execution
            order.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        # Build in-degree map and adjacency list.
        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        children: dict[str, list[str]] = {nid: [] for nid in self._nodes}
        for nid, node in self._nodes.items():
            for pid in node.parent_ids:
                children[pid].append(nid)
                in_degree[nid] += 1

        # Seed with zero-in-degree nodes.
        queue: list[str] = [
            nid for nid, deg in in_degree.items() if deg == 0
        ]
        # Sort for deterministic output.
        queue.sort()
        result: list[PlanNode] = []

        while queue:
            nid = queue.pop(0)
            result.append(self._nodes[nid])
            for child_id in sorted(children[nid]):
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        if len(result) != len(self._nodes):
            raise ValueError(
                "Plan graph contains a cycle — cannot produce "
                "topological order."
            )

        return result

    def get_node(self, node_id: str) -> PlanNode:
        """Look up a single node by ID.

        Requires:
            - *node_id* exists in this plan.

        Returns:
            The ``PlanNode`` with the given ID.

        Raises:
            KeyError: If *node_id* is not found.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found in plan.")
        return self._nodes[node_id]

    @property
    def nodes(self) -> list[PlanNode]:
        """All nodes (unordered).

        Requires:
            None.

        Returns:
            A list of all ``PlanNode`` objects in no particular order.

        Raises:
            None.
        """
        return list(self._nodes.values())

    @property
    def leaf_nodes(self) -> list[PlanNode]:
        """Dataset-load nodes (no parents).

        Requires:
            None.

        Returns:
            A list of ``PlanNode`` objects whose ``node_type`` is
            ``"dataset"``.

        Raises:
            None.
        """
        return [
            n for n in self._nodes.values() if n.node_type == "dataset"
        ]

    @property
    def reasoning_node(self) -> PlanNode | None:
        """The final reasoning node, if present.

        Requires:
            None.

        Returns:
            The reasoning ``PlanNode``, or ``None`` if the plan has no
            reasoning step.

        Raises:
            None.
        """
        return next(
            (n for n in self._nodes.values() if n.node_type == "reasoning"),
            None,
        )

    def children_of(self, node_id: str) -> list[PlanNode]:
        """Return direct children of a node.

        Requires:
            - *node_id* exists in this plan.

        Returns:
            A list of ``PlanNode`` objects that have *node_id* in
            their ``parent_ids``.

        Raises:
            None.
        """
        return [
            n for n in self._nodes.values() if node_id in n.parent_ids
        ]

    def invalidated_downstream(self, node_id: str) -> list[str]:
        """Return the transitive downstream closure of *node_id*.

        Performs a BFS from the children of *node_id* and returns all
        reachable node IDs (not including *node_id* itself).

        Requires:
            - *node_id* exists in this plan.

        Returns:
            A list of node IDs that transitively depend on *node_id*.

        Raises:
            None.
        """
        visited: set[str] = set()
        queue = [
            n.node_id for n in self.children_of(node_id)
        ]
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            queue.extend(n.node_id for n in self.children_of(nid))
        return sorted(visited)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def edit_node(self, node_id: str, new_params: dict) -> list[str]:
        """Update a node's params in-place and return invalidated IDs.

        Merges *new_params* into the existing ``params`` dict of the
        node (existing keys are overwritten, new keys are added).

        Requires:
            - *node_id* exists in this plan.
            - *new_params* is non-empty.

        Returns:
            A sorted list of node IDs that are transitively downstream
            of the edited node (not including *node_id* itself).

        Raises:
            KeyError: If *node_id* is not found.
            ValueError: If *new_params* is empty.
        """
        if not new_params:
            raise ValueError("new_params must be non-empty.")
        node = self.get_node(node_id)  # raises KeyError
        node.params.update(new_params)
        return self.invalidated_downstream(node_id)

    def insert_node(self, after_node_id: str, new_node: PlanNode) -> list[str]:
        """Insert a node after *after_node_id*, rewiring edges.

        The new node is spliced between *after_node_id* and all of
        its current children.  After the call:

        - ``new_node.parent_ids == [after_node_id]``
        - Every former child of *after_node_id* now lists
          ``new_node.node_id`` in place of *after_node_id* in its
          ``parent_ids``.

        Requires:
            - *after_node_id* exists in this plan.
            - ``new_node.node_id`` does not already exist in this plan.

        Returns:
            A sorted list of node IDs that are transitively downstream
            of the newly inserted node (not including the new node
            itself).

        Raises:
            KeyError: If *after_node_id* is not found.
            ValueError: If ``new_node.node_id`` already exists.
        """
        if after_node_id not in self._nodes:
            raise KeyError(f"Node '{after_node_id}' not found in plan.")
        if new_node.node_id in self._nodes:
            raise ValueError(
                f"Node '{new_node.node_id}' already exists in plan."
            )

        # Wire new_node as child of after_node.
        new_node.parent_ids = [after_node_id]

        # Rewire former children of after_node → new_node.
        for child in self.children_of(after_node_id):
            child.parent_ids = [
                new_node.node_id if pid == after_node_id else pid
                for pid in child.parent_ids
            ]

        self._nodes[new_node.node_id] = new_node
        return self.invalidated_downstream(new_node.node_id)

    def delete_node(self, node_id: str) -> list[str]:
        """Remove a node, rewiring its children to its parents.

        After the call, every former child of *node_id* that listed
        *node_id* in its ``parent_ids`` now lists the deleted node's
        parents instead.

        Requires:
            - *node_id* exists in this plan.
            - *node_id* is not a leaf (dataset) node (deleting a data
              source would leave the plan disconnected).

        Returns:
            A sorted list of node IDs that are transitively downstream
            of the deleted node (computed **before** deletion).

        Raises:
            KeyError: If *node_id* is not found.
            ValueError: If the node is a dataset node.
        """
        node = self.get_node(node_id)  # raises KeyError
        if node.node_type == "dataset":
            raise ValueError(
                f"Cannot delete dataset node '{node_id}'."
            )

        # Compute invalidation set before removing the node.
        invalidated = self.invalidated_downstream(node_id)

        # Rewire children: replace node_id with the deleted node's parents.
        for child in self.children_of(node_id):
            new_parents: list[str] = []
            for pid in child.parent_ids:
                if pid == node_id:
                    new_parents.extend(node.parent_ids)
                else:
                    new_parents.append(pid)
            child.parent_ids = new_parents

        del self._nodes[node_id]
        return invalidated

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_node_dicts(self) -> list[dict]:
        """Serialise all nodes to a list of dict descriptors.

        Requires:
            None.

        Returns:
            A list of node-descriptor dicts in topological order.

        Raises:
            None.
        """
        return [node.to_dict() for node in self.topo_order()]

    def to_dict(self) -> dict:
        """Flat JSON-serialisable representation (for API transport).

        Requires:
            None.

        Returns:
            A dict with ``query`` and ``nodes`` keys, where ``nodes``
            is a list of node dicts in topological order.

        Raises:
            None.
        """
        return {
            "query": self.query,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "operator_type": n.operator_type,
                    "name": n.name,
                    "description": n.description,
                    "params": n.params,
                    "parent_ids": n.parent_ids,
                    "dataset_id": n.dataset_id,
                }
                for n in self.topo_order()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> PhysicalPlan:
        """Reconstruct a ``PhysicalPlan`` from a flat JSON representation.

        This is the inverse of :meth:`to_dict`.

        Requires:
            - *data* has the schema produced by ``to_dict()``.

        Returns:
            A ``PhysicalPlan`` with the same nodes and structure.

        Raises:
            KeyError: If required keys are missing.
        """
        nodes: dict[str, PlanNode] = {}
        for nd in data["nodes"]:
            node = PlanNode(
                node_id=nd["node_id"],
                node_type=nd["node_type"],
                operator_type=nd.get("operator_type"),
                name=nd["name"],
                description=nd["description"],
                params=nd.get("params", {}),
                parent_ids=nd.get("parent_ids", []),
                dataset_id=nd.get("dataset_id", ""),
            )
            nodes[node.node_id] = node
        return cls(nodes=nodes, query=data.get("query", ""))

    def __len__(self) -> int:
        """Return the number of nodes in this plan.

        Requires:
            None.

        Returns:
            Non-negative integer.

        Raises:
            None.
        """
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        """Return whether *node_id* exists in this plan.

        Requires:
            None.

        Returns:
            ``True`` if *node_id* is a key in ``_nodes``.

        Raises:
            None.
        """
        return node_id in self._nodes
