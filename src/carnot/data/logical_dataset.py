from __future__ import annotations


class LogicalDataset:
    def __init__(self, name: str, parents: list[LogicalDataset] | None = None, **kwargs):
        self.name = name
        self.parents = parents or []
        self.params = kwargs

    def serialize(self) -> dict:
        return {"name": self.name, "params": self.params, "parents": [p.serialize() for p in self.parents]}

    def merge(self, other: LogicalDataset) -> LogicalDataset:
        """
        Merge this dataset with another dataset.
        """
        merged_name = f"{self.name}_merged_with_{other.name}"
        params = {"operator": "Merge", "description": f"Merged {self.name} with {other.name}"}
        return LogicalDataset(merged_name, parents=[self, other], **params)

    def code(self, task: str) -> LogicalDataset:
        """
        Apply a code operation to the dataset based on the given task.
        """
        coded_name = f"{self.name}_coded_for_{task}"
        params = {"operator": "Code", "description": f"Coded {self.name} for task: {task}", "task": task}
        return LogicalDataset(coded_name, parents=[self], **params)

    def reason(self, task: str) -> LogicalDataset:
        """
        Apply a reasoning operation to the dataset based on the given task.
        """
        reasoned_name = f"{self.name}_reasoned_for_{task}"
        params = {"operator": "Reason", "description": f"Reasoned {self.name} for task: {task}", "task": task}
        return LogicalDataset(reasoned_name, parents=[self], **params)

    def sem_filter(self, condition: str) -> LogicalDataset:
        """
        Apply a semantic filter to the dataset based on the given condition.
        """
        filtered_name = f"{self.name}_filtered_by_{condition}"
        params = {"operator": "SemanticFilter", "description": f"Filtered {self.name} by condition: {condition}", "condition": condition}
        return LogicalDataset(filtered_name, parents=[self], **params)

    def sem_map(self, field: str, type: type, description: str) -> LogicalDataset:
        """
        Apply a semantic map to the dataset, adding or transforming a field.
        """
        mapped_name = f"{self.name}_mapped_{field}"
        params = {
            "operator": "SemanticMap",
            "description": f"Created field {field} with type {type.__name__} and description {description}",
            "field": field,
            "type": type.__name__,
            "field_desc": description,
        }
        return LogicalDataset(mapped_name, parents=[self], **params)

    def sem_flat_map(self, field: str, type: type, description: str) -> LogicalDataset:
        """
        Apply a semantic flat map to the dataset, expanding a field into multiple entries.
        """
        flat_mapped_name = f"{self.name}_flat_mapped_{field}"
        params = {
            "operator": "SemanticFlatMap",
            "description": f"Flat mapped field {field} with type {type.__name__} and description {description}",
            "field": field,
            "type": type.__name__,
            "field_desc": description,
        }
        return LogicalDataset(flat_mapped_name, parents=[self], **params)

    def sem_groupby(self, gby_fields: list[str], agg_fields: list[str], agg_funcs: list[str]) -> LogicalDataset:
        """
        Apply a semantic group by operation to the dataset.
        """
        gby_name = f"{self.name}_grouped_by_{'_'.join(gby_fields)}"
        params = {
            "operator": "SemanticGroupBy",
            "description": f"Grouped {self.name} by fields {gby_fields} with aggregations on {agg_fields} using functions {agg_funcs}",
            "gby_fields": gby_fields,
            "agg_fields": agg_fields,
            "agg_funcs": agg_funcs,
        }
        return LogicalDataset(gby_name, parents=[self], **params)

    def sem_join(self, other: LogicalDataset, condition: str) -> LogicalDataset:
        """
        Apply a semantic join with another dataset based on the given condition.
        """
        joined_name = f"{self.name}_joined_with_{other.name}"
        params = {
            "operator": "SemanticJoin",
            "description": f"Joined {self.name} with {other.name} on condition: {condition}",
            "condition": condition,
        }
        return LogicalDataset(joined_name, parents=[self, other], **params)
