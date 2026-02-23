from __future__ import annotations

import textwrap
from collections.abc import Iterator

from carnot.data.item import DataItem
from carnot.index.index import CarnotIndex


class Dataset:
    """
    A collection of files (DataItems) and associated metadata.
    
    This class serves dual purposes:
    1. Physical dataset: Contains actual data items and metadata
    2. Logical dataset: Supports building logical query plans through chained operations
    """
    def __init__(
        self,
        name: str,
        annotation: str,
        items: list[DataItem] | None = None,
        index: CarnotIndex | None = None,
        code: str | None = None,
        code_state: dict | None = None,
        parents: list[Dataset] | None = None,
        id_params: dict | None = None,
        **kwargs
    ):
        # Physical dataset attributes
        self.name = name
        self.annotation = annotation
        self.items = items or []
        self._index = index
        self.code = code
        self.code_state = code_state or {}
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

    def serialize(self) -> dict:
        """Serialize the logical plan structure of this dataset."""
        return {
            "name": self.name,
            "output_dataset_id": self.output_dataset_id,
            "params": self.params,
            "parents": [p.serialize() for p in self.parents],
        }

    def format_description(self, code_block_tags: list[str]) -> str:
        code_str = " None" if self.code is None else f"\n{code_block_tags[0]}\n{self.code}\n{code_block_tags[1]}"
        return textwrap.dedent(
            f"Dataset Name: {self.name}\n"
            f"Annotation: {self.annotation}\n"
            f"Number of Items: {len(self.items)}\n"
            f"Index: {'yes' if self.has_index() else 'no'}\n"
            f"Code that Generated Code State: {code_str}\n"
            f"Available Code State Vars: {list(self.code_state.keys())}\n"
        )

    def has_index(self) -> bool:
        return self._index is not None

    def index(self, query: str, k: int = 5) -> list[DataItem]:
        if self._index is None:
            raise NotImplementedError("Dataset does not have an index constructed.")
        return self._index.search(query, k=k)

    def __iter__(self) -> Iterator[DataItem]:
        return iter(self.items)
    
    def limit(self, n: int) -> Dataset:
        """
        Apply a limit operation to the dataset, returning only the first n records.
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
            **params
        )

    def merge(self, other: Dataset) -> Dataset:
        """
        Merge this dataset with another dataset.
        """
        merged_name = f"MergeOperation{self.id_params['merge_id'] + 1}"
        self.id_params["merge_id"] += 1
        params = {"operator": "Merge", "description": f"Merged {self.name} with {other.name}"}
        return Dataset(
            name=merged_name,
            annotation=f"Merge of ({self.annotation}) and ({other.annotation})",
            parents=[self, other],
            id_params=self.id_params,
            output_dataset_id=merged_name,
            **params
        )

    def write_code(self, task: str) -> Dataset:
        """
        Apply a write code operation to the dataset based on the given task.
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
            **params
        )

    def reason(self, task: str) -> Dataset:
        """
        Apply a reasoning operation to the dataset based on the given task.
        """
        reasoned_name = f"ReasonOperation{self.id_params['reason_id'] + 1}"
        self.id_params["reason_id"] += 1
        params = {"operator": "Reason", "description": f"Reasoned {self.name} for task: {task}", "task": task}
        return Dataset(
            name=reasoned_name,
            annotation=f"Reasoning operation on ({self.annotation})",
            parents=[self],
            id_params=self.id_params,
            output_dataset_id=reasoned_name,
            **params
        )

    def sem_aggregate(self, task: str, agg_fields: list[dict]) -> Dataset:
        """
        Apply a semantic aggregation to the dataset based on the given aggregation fields.
        """
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
            **params
        )

    def sem_filter(self, condition: str) -> Dataset:
        """
        Apply a semantic filter to the dataset based on the given condition.
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
            **params
        )

    def sem_map(self, field: str, type: type, description: str) -> Dataset:
        """
        Apply a semantic map to the dataset, adding or transforming a field.
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
            **params
        )

    def sem_flat_map(self, field: str, type: type, description: str) -> Dataset:
        """
        Apply a semantic flat map to the dataset, expanding a field into multiple entries.
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
            **params
        )

    def sem_groupby(self, gby_fields: list[dict], agg_fields: list[dict]) -> Dataset:
        """
        Apply a semantic group by operation to the dataset.
        """
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
            **params
        )

    def sem_join(self, other: Dataset, condition: str) -> Dataset:
        """
        Apply a semantic join with another dataset based on the given condition.
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
            **params
        )

    def sem_topk(self, search_str: str, k: int = 5) -> Dataset:
        """
        Apply a semantic top-k operation with the given search string and k value.
        """
        top_k_name = f"TopKOperation{self.id_params['sem_topk_id'] + 1}"
        self.id_params["sem_topk_id"] += 1
        params = {
            "operator": "SemanticTopK",
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
            **params
        )

