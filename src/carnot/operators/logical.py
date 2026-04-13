from __future__ import annotations

import textwrap
from typing import Any


class LogicalOperator:
    """
    A logical operator is an operator that operates on Dataset(s).
    """

    def __init__(self, name: str):
        # unique identifier for the operator; the caller must guarantee that this is unique
        # across all operators within a given query plan
        self.name = name

    def desc(self) -> str:
        raise NotImplementedError("Abstract method")

    def __str__(self) -> str:
        raise NotImplementedError("Abstract method")

    def __eq__(self, other: Any) -> bool:
        """Two logical operators are equal if they have the same class and the same parameters (not including name)."""
        return isinstance(other, self.__class__) and self.get_logical_op_params() == other.get_logical_op_params()

    def copy(self) -> LogicalOperator:
        return self.__class__(**self.get_logical_op_params())

    def logical_op_class_name(self) -> str:
        """Name of the logical operator."""
        return str(self.__class__.__name__)

    def serialize(self) -> dict:
        return {
            "logical_op_class_name": self.logical_op_class_name(),
            **self.get_logical_op_params(),
        }

    def get_logical_op_params(self) -> dict:
        """
        Returns a dictionary mapping of logical operator parameters which may be used to
        implement a physical operator associated with this logical operation.

        NOTE: Should be overriden by subclasses to include class-specific parameters.
        """
        return {"name": self.name}

    def get_logical_op_id(self):
        """
        TODO: remove and use name instead throughout the code base
        """
        return self.name


class Aggregate(LogicalOperator):
    """
    Logical operator that applies an aggregation to the input Dataset and yields a single result.
    """

    def __init__(self, name: str, agg_fields: list[dict]):
        super().__init__(name)
        self.agg_fields = agg_fields

    @staticmethod
    def desc() -> str:
        return textwrap.dedent(
            """
            Aggregate Operator:
                Description: Applies one or more semantic aggregations to the entire input dataset.
                    Useful for summarization or finding extremal input(s). For example, finding the most negative reviews
                    in a set of product reviews, or summarizing a set of legal contracts. Takes in a `task` string that
                    describes the aggregation(s) to be performed in natural language, as well as a list of the aggregation
                    fields to be computed (provide the "name", "type", and "description", and always set `"func": "sem_agg"`).
                Syntax: ds.sem_aggregate(task: str, agg_fields: list[dict])
                Example: ds.sem_aggregate(
                  task="Summarize the legal contracts and identify the most problematic clauses.",
                  agg_fields=[
                    {"name": "summary", "type": str, "description": "A concise summary of the legal contracts", "func": "sem_agg"},
                    {"name": "worst_clauses", "type": list[str], "description": "A list of the most problematic clauses in the legal contracts", "func": "sem_agg"},
                  ],
                )
                Example: ds.sem_aggregate(
                  task="Identify the most negative product review.",
                  agg_fields=[
                    {"name": "most_negative_review", "type": str, "description": "The text of the most negative product review", "func": "sem_agg"},
                  ],
                )
            """
        )

    def __str__(self):
        return f"Aggregate(agg fields: {self.agg_fields})"

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {"agg_fields": self.agg_fields, **logical_op_params}
        return logical_op_params


class Code(LogicalOperator):
    """Logical operator that represents a code operation on the input Dataset."""

    def __init__(self, name: str, task: str):
        super().__init__(name)
        self.task = task

    @staticmethod
    def desc() -> str:
        return textwrap.dedent(
            """
            Code Operator:
                Description: Use a coding agent to perform computation over structured data based on a natural language task description.
                Syntax: ds.write_code(task: str)
                Example: ds.write_code(task="Compute the total revenue for each product category in the sales dataset.")
            """
        )

    def __str__(self):
        return f"Code(task: {self.task})"

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {"task": self.task, **logical_op_params}

        return logical_op_params


class Filter(LogicalOperator):
    """Logical operator that represents applying a semantic filter to the input Dataset based on a natural language condition."""

    def __init__(self, name: str, filter: str, desc: str | None = None):
        super().__init__(name)
        self.filter = filter
        self._desc = desc

    @staticmethod
    def desc() -> str:
        return textwrap.dedent(
            """
            Filter Operator:
                Description: Applies a semantic filter to the input set based on a provided natural language condition.
                Syntax: ds.sem_filter(condition: str)
                Example: ds.sem_filter("The image contains a sunset over the mountains")
            """
        )

    def __str__(self):
        return f"Filter({str(self.filter)})"

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {"filter": self.filter, "desc": self._desc, **logical_op_params}

        return logical_op_params


class FlatMap(LogicalOperator):
    """Logical operator that represents applying a flat map operation to an input Dataset."""

    def __init__(self, name: str, fields: list[dict], desc: str | None = None):
        super().__init__(name)
        self.fields = fields
        self._desc = desc

    @staticmethod
    def desc() -> str:
        return textwrap.dedent(
            """
            Flat Map Operator:
                Description: Executes a semantic flat map operation by applying an LLM to each record in the input set to compute multiple output records per input record.
                  Each field consists of three keys: "name" (the name of the field to be created), "type" (the type of the field, e.g. str, int, list[str], etc.), and "description" (a natural language description of the field).
                Syntax: ds.sem_flat_map(field: str, type: type, description: str)
                Example: ds.sem_flat_map(fields=[{"name": "key_points", "type": str, "description": "Key points extracted from the legal contract"}])
            """
        )

    def __str__(self):
        return f"FlatMap({str(self.fields)})"

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {"fields": self.fields, "desc": self._desc, **logical_op_params}

        return logical_op_params


class GroupBy(LogicalOperator):
    """
    Logical operator that applies a group by operation to the input Dataset and yields a result.
    """

    def __init__(self, name: str, gby_fields: list[dict], agg_fields: list[dict]):
        super().__init__(name)
        self.gby_fields = gby_fields
        self.agg_fields = agg_fields

    @staticmethod
    def desc() -> str:
        return textwrap.dedent(
            """
            GroupBy Operator:
                Description: Groups the input set by specified columns and applies an aggregation function to each group.
                    The supported aggregation `func`s are: "min", "max", "mean", "count", and "sum".
                    You may compute a semantic aggregation by specifying `func` as "sem_agg".
                    The `agg_fields` must exist in the dataset already, but the `gby_fields` can be new fields derived from existing fields.
                Syntax: ds.sem_groupby(gby_fields: list[dict], agg_fields: list[dict])
                Example (semantic group, relational aggregate):
                  # assume input has a "review_text" column which can be used to compute "sentiment"
                  ds.sem_groupby(
                    gby_fields=[{"name": "sentiment", "type": str, "description": "the sentiment (POSITIVE or NEGATIVE) of the movie review"}],
                    agg_fields=[{"name": "count", "type": int, "description": "the count of each review sentiment", "func": "count"}],
                  )
                Example (semantic group, semantic aggregate):
                  # assume the same example as before
                  ds.sem_groupby(
                    gby_fields=[{"name": "sentiment", "type": str, "description": "the sentiment (POSITIVE or NEGATIVE) of the movie review"}],
                    agg_fields=[
                      {"name": "count", "type": int, "description": "the count of each review sentiment", "func": "count"}
                      {"name": "summary", "type": str, "description": "a concise summary of the reviews for each sentiment", "func": "sem_agg"},
                    ],
                  )
                Example (relational group, relational aggregate):
                  # assume we have a "sales" dataset with "region", "product_category", and "amount" columns
                  ds.sem_groupby(
                    gby_fields=[
                      {"name": "region", "type": str, "description": "the sales region"},
                      {"name": "product_category", "type": str, "description": "the product category"},
                    ],
                    agg_fields=[
                      {"name": "total_sales", "type": float, "description": "the total sales amount for the region and product category", "func": "sum"},
                    ],
                  )
            """
        )

    def __str__(self):
        return f"GroupBy(gby_fields: {self.gby_fields}, agg fields: {self.agg_fields})"

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {"gby_fields": self.gby_fields, "agg_fields": self.agg_fields, **logical_op_params}
        return logical_op_params


class Join(LogicalOperator):
    """Logical operator that represents a semantic join between two Datasets based on a natural language condition."""
    def __init__(self, name: str, condition: str, desc: str | None = None):
        super().__init__(name)
        self.condition = condition
        self._desc = desc

    @staticmethod
    def desc():
        return textwrap.dedent(
            """
            Join Operator:
                Description: Semantically joins two input sets based on a provided natural language condition.
                Syntax: ds.sem_join(other: Dataset, condition: str)
                Example: ds.sem_join(other=orders_ds, condition="customer_id matches id in the order PDF")
            """
        )

    def __str__(self):
        return f"Join({self.condition})"

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {"condition": self.condition, "desc": self._desc, **logical_op_params}

        return logical_op_params


class Limit(LogicalOperator):
    def __init__(self, name: str, n: int):
        super().__init__(name)
        self.n = n

    @staticmethod
    def desc() -> str:
        return textwrap.dedent(
            """
            Limit Operator:
                Description: return the first N records from the input set. Useful for limiting the number of records returned or for debugging.
                Syntax: ds.limit(n: int)
                Example: ds.limit(n=10)
            """
        )

    def __str__(self):
        return f"Limit({self.n})"

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {"n": self.n, **logical_op_params}

        return logical_op_params


class Map(LogicalOperator):
    """Logical operator that represents applying a map operation to an input Dataset."""

    def __init__(self, name: str, fields: list[dict], desc: str | None = None):
        super().__init__(name)
        self.fields = fields
        self._desc = desc

    @staticmethod
    def desc() -> str:
        return textwrap.dedent(
            """
            Map Operator:
                Description: Executes a semantic map operation by applying an LLM to each record in the input set to compute one or more output field(s).
                  Each field consists of three keys: "name" (the name of the field to be created), "type" (the type of the field, e.g. str, int, list[str], etc.), and "description" (a natural language description of the field).
                  The map operator can be used to create new fields from existing fields, or to overwrite existing fields with new values.
                Syntax: ds.sem_map(fields: list[dict])
                Example: ds.sem_map(fields=[{"name": "summary", "type": str, "description": "A concise summary of the legal contract"}])
            """
        )

    def __str__(self):
        return f"Map({str(self.fields)})"

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {"fields": self.fields, "desc": self._desc, **logical_op_params}

        return logical_op_params


class Reason(LogicalOperator):
    """Logical operator that represents a reasoning operation on the input Dataset."""

    def __init__(self, name: str, task: str):
        super().__init__(name)
        self.task = task

    @staticmethod
    def desc() -> str:
        return textwrap.dedent(
            """
            Reason Operator:
                Description: Use a reasoning agent (which has the ability to write code) to perform a reasoning task.
                Syntax: ds.reason(task: str)
                Example: ds.reason(task="Return the final answer to the query given the intermediate results.")
            """
        )

    def __str__(self):
        return f"Reason(task: {self.task})"

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {"task": self.task, **logical_op_params}

        return logical_op_params


class Scan(LogicalOperator):
    """Logical operator that represents reading an entire dataset.

    Every leaf Dataset node in the plan DAG is implicitly a scan.  Making
    this explicit gives the optimizer a concrete logical expression to
    attach implementation rules and cost estimates to.

    The operator carries lightweight metadata — record count and an
    estimated per-item token count — so the cost model can compute a
    base-case ``PlanCost`` without materializing any data.

    Representation invariant:
        - ``num_items >= 0``.
        - ``est_tokens_per_item >= 0``.

    Abstraction function:
        Represents the act of reading all items from dataset
        ``dataset_id``, exposing an estimated total token count of
        ``num_items * est_tokens_per_item``.
    """

    def __init__(self, name: str, dataset_id: str, num_items: int, est_tokens_per_item: float):
        super().__init__(name)
        self.dataset_id = dataset_id
        self.num_items = num_items
        self.est_tokens_per_item = est_tokens_per_item

    def desc(self) -> str:
        return f"Scan dataset '{self.dataset_id}' ({self.num_items} items)"

    def __str__(self):
        return f"Scan({self.dataset_id}, items={self.num_items})"

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {
            "dataset_id": self.dataset_id,
            "num_items": self.num_items,
            "est_tokens_per_item": self.est_tokens_per_item,
            **logical_op_params,
        }

        return logical_op_params


class TopK(LogicalOperator):
    """A TopK is a logical operator that represents a semantic search over the input dataset for the top-k most relevant items."""

    def __init__(self, name: str, task: str, k: int, index_name: str = "chroma"):
        super().__init__(name)
        self.task = task
        self.k = k
        self.index_name = index_name

    @staticmethod
    def desc():
        return textwrap.dedent(
            """
            Top-K Operator:
                Description: Searches for the top-k most semantically relevant items to a given search string. Uses the dataset's index (flat, hierarchical, chroma, or faiss). When a dataset has multiple indices, specify which to use with index_name.
                Syntax: ds.sem_topk(index_name: str, search_str: str, k: int)
                Example: ds.sem_topk(index_name="chroma", search_str="order contains self-care products", k=5)
            """
        )

    def __str__(self):
        return f"TopK(task: {self.task}, k: {self.k}, index: {self.index_name})"

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {
            "task": self.task,
            "k": self.k,
            "index_name": self.index_name,
            **logical_op_params,
        }

        return logical_op_params
