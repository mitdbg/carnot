# Executes a PhysicalPlan against the index portfolio. No need to expose this publicly.
from __future__ import annotations
from typing import List

from ..types import SearchResult
from .index_management import IndexManagementPipeline
from .query_optimization import PhysicalPlan


class QueryExecutor:
    """Executes physical plans against the index portfolio and returns results."""

    def __init__(self, index_pipeline: IndexManagementPipeline) -> None:
        """Initialize the executor with access to all indexes."""
        pass

    def execute(self, plan: PhysicalPlan) -> List[SearchResult]:
        """Execute a physical plan. Any top-k/limit should be encoded in the plan."""
        pass
