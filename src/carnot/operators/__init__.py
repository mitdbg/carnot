from carnot.operators.logical import (
    Aggregate,
    ComputeOperator,
    ConvertScan,
    FilteredScan,
    JoinOp,
    LimitScan,
)

LOGICAL_OPERATORS = [
    Aggregate,
    ComputeOperator,
    ConvertScan,
    FilteredScan,
    JoinOp,
    LimitScan,
]

__all__ = [
    "LOGICAL_OPERATORS",
]
