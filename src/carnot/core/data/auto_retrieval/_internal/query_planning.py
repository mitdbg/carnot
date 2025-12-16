# Internal query planning (logical plans only)
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Mapping, Any, Optional

from ..types import Query
from .index_management import IndexManagementPipeline


@dataclass
class LogicalOperator:
    """Single logical operator, e.g., selection, intersection, or vector probe."""
    op_type: str
    params: Mapping[str, Any]


@dataclass
class LogicalPlan:
    """Logical plan composed of selections, intersections, unions, and vector probes."""
    operators: List[LogicalOperator]


class QueryPlanner:
    """Maps natural-language queries to hybrid logical plans over available operators."""

    def __init__(self, index_pipeline: IndexManagementPipeline) -> None:
        """Initialize the planner with access to index metadata and concepts."""
        pass

    @classmethod
    def from_config(
        cls,
        config: Any,
        index_pipeline: IndexManagementPipeline,
    ) -> "QueryPlanner":
        """Construct a QueryPlanner from configuration and index statistics."""
        pass

    def plan(self, query: Query) -> LogicalPlan:
        """Produce a logical plan encoding filters, concept selections, and vector probes."""
        pass
