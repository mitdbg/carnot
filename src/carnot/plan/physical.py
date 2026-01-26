from __future__ import annotations

from carnot.plan.physical import PhysicalOperator


class PhysicalPlan:
    """
    Represents a physical plan for query execution.
    """
    def __init__(self, operator: PhysicalOperator, op_desc: str, subplans: list[PhysicalPlan] | None = None):
        self.operator = operator
        self.op_desc = op_desc
        self.subplans = subplans or []
