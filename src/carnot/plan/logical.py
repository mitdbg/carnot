from __future__ import annotations

from carnot.operators.merge import Merge
from carnot.plan.logical import LogicalOperator


class LogicalPlan:
    """
    Represents a logical plan for query execution.
    """
    def __init__(self, operator: LogicalOperator, op_desc: str, subplans: list[LogicalPlan] | None = None):
        self.operator = operator
        self.op_desc = op_desc
        self.subplans = subplans or []

    @staticmethod
    def from_dict(plan_dict: dict) -> LogicalPlan:
        operator = None
        if plan_dict["operator"] == "Merge":
            left_parent = LogicalPlan.from_dict(plan_dict["parents"][0])
            right_parent = LogicalPlan.from_dict(plan_dict["parents"][1])
            operator = Merge(left_parent, right_parent)

            LogicalOperator[plan_dict["operator"]]
        op_desc = plan_dict["op_desc"]
        subplans = [LogicalPlan.from_dict(sp) for sp in plan_dict.get("subplans", [])]
        return LogicalPlan(operator, op_desc, subplans)