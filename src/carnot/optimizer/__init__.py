from carnot.optimizer.rules import (
    BasicSubstitutionRule,
    CodeRule,
    ImplementationRule,
    PushDownFilter,
    ReasoningRule,
    Rule,
    ScanRule,
    SemAggRule,
    SemFilterRule,
    SemFlatMapRule,
    SemGroupByRule,
    SemJoinRule,
    SemMapRule,
    SemTopKRule,
    TransformationRule,
)

ALL_RULES = [
    BasicSubstitutionRule,
    CodeRule,
    ImplementationRule,
    PushDownFilter,
    ReasoningRule,
    Rule,
    ScanRule,
    SemAggRule,
    SemFilterRule,
    SemFlatMapRule,
    SemGroupByRule,
    SemJoinRule,
    SemMapRule,
    SemTopKRule,
    TransformationRule,
]

IMPLEMENTATION_RULES = [
    rule
    for rule in ALL_RULES
    if issubclass(rule, ImplementationRule)
    and rule not in [ImplementationRule]
]

TRANSFORMATION_RULES = [
    rule
    for rule in ALL_RULES
    if issubclass(rule, TransformationRule)
    and rule not in [TransformationRule]
]

__all__ = [
    "ALL_RULES",
    "IMPLEMENTATION_RULES",
    "TRANSFORMATION_RULES",
]
