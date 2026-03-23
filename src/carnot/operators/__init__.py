from carnot.operators.logical import (
    Aggregate,
    Code,
    FilteredScan,
    JoinOp,
    Limit,
    LogicalOperator,
    MapScan,
    TopK,
)

LOGICAL_OPERATORS: list[LogicalOperator] = [
    Aggregate,
    Code,
    FilteredScan,
    JoinOp,
    Limit,
    MapScan,
    TopK,
]

__all__ = [
    "LOGICAL_OPERATORS",
]
