"""Unit tests for :mod:`carnot.plan.feedback`.

Phase 3b of the Carnot library refactor (see
``docs/CARNOT_LIBRARY_REFACTOR.md``).  These are Tier 1 (no LLM) tests
that validate the dataclass contracts of ``NodeEdit``,
``StructuralChange``, and ``PlanFeedback``.
"""

from __future__ import annotations

from carnot.plan.feedback import (
    NodeEdit,
    PlanFeedback,
    StructuralChange,
)

# ═══════════════════════════════════════════════════════════════════════
# NodeEdit
# ═══════════════════════════════════════════════════════════════════════


class TestNodeEdit:
    """Verify ``NodeEdit`` construction and field access."""

    def test_construction(self):
        edit = NodeEdit(node_id="n1", new_params={"condition": "x > 5"})
        assert edit.node_id == "n1"
        assert edit.new_params == {"condition": "x > 5"}

    def test_equality(self):
        a = NodeEdit(node_id="n1", new_params={"k": "v"})
        b = NodeEdit(node_id="n1", new_params={"k": "v"})
        assert a == b

    def test_inequality_different_params(self):
        a = NodeEdit(node_id="n1", new_params={"k": "v"})
        b = NodeEdit(node_id="n1", new_params={"k": "w"})
        assert a != b


# ═══════════════════════════════════════════════════════════════════════
# StructuralChange
# ═══════════════════════════════════════════════════════════════════════


class TestStructuralChange:
    """Verify ``StructuralChange`` construction and field access."""

    def test_delete_construction(self):
        sc = StructuralChange(change_type="delete", node_id="n2")
        assert sc.change_type == "delete"
        assert sc.node_id == "n2"
        assert sc.after_node_id is None
        assert sc.new_node_params is None

    def test_insert_construction(self):
        sc = StructuralChange(
            change_type="insert",
            after_node_id="n1",
            new_node_params={"operator": "Limit", "n": 10},
        )
        assert sc.change_type == "insert"
        assert sc.after_node_id == "n1"
        assert sc.new_node_params == {"operator": "Limit", "n": 10}

    def test_equality(self):
        a = StructuralChange(change_type="delete", node_id="n2")
        b = StructuralChange(change_type="delete", node_id="n2")
        assert a == b


# ═══════════════════════════════════════════════════════════════════════
# PlanFeedback
# ═══════════════════════════════════════════════════════════════════════


class TestPlanFeedback:
    """Verify ``PlanFeedback`` construction and field defaults."""

    def test_defaults(self):
        fb = PlanFeedback()
        assert fb.chat_message is None
        assert fb.node_edits == []
        assert fb.structural_changes == []
        assert fb.cost_budget is None
        assert fb.previous_plan is None

    def test_with_chat_message(self):
        fb = PlanFeedback(chat_message="change the filter")
        assert fb.chat_message == "change the filter"

    def test_with_node_edits(self):
        edit = NodeEdit(node_id="n1", new_params={"condition": "x > 5"})
        fb = PlanFeedback(node_edits=[edit])
        assert len(fb.node_edits) == 1
        assert fb.node_edits[0].node_id == "n1"

    def test_with_structural_changes(self):
        sc = StructuralChange(change_type="delete", node_id="n2")
        fb = PlanFeedback(structural_changes=[sc])
        assert len(fb.structural_changes) == 1
        assert fb.structural_changes[0].change_type == "delete"

    def test_with_cost_budget(self):
        fb = PlanFeedback(cost_budget=0.50)
        assert fb.cost_budget == 0.50

    def test_combined(self):
        edit = NodeEdit(node_id="n1", new_params={"x": 1})
        sc = StructuralChange(change_type="delete", node_id="n2")
        fb = PlanFeedback(
            chat_message="refine plan",
            node_edits=[edit],
            structural_changes=[sc],
            cost_budget=1.0,
        )
        assert fb.chat_message == "refine plan"
        assert len(fb.node_edits) == 1
        assert len(fb.structural_changes) == 1
        assert fb.cost_budget == 1.0

    def test_mutable_default_lists_independent(self):
        """Each instance gets its own list (dataclass field(default_factory))."""
        fb1 = PlanFeedback()
        fb2 = PlanFeedback()
        fb1.node_edits.append(NodeEdit(node_id="x", new_params={"a": 1}))
        assert len(fb2.node_edits) == 0
