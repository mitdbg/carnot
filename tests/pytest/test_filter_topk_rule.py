"""Tests for the FilterToTopKFilter transformation rule and model ID auto-detection.

**Tier 1** (pure unit) tests — no network calls or LLM mocks needed.
"""

from __future__ import annotations

from carnot.operators.logical import Filter, Map, Scan, TopK
from carnot.optimizer.model_ids import get_available_model_ids
from carnot.optimizer.primitives import Group, LogicalExpression
from carnot.optimizer.rules import FilterToTopKFilter

# ── Helpers ───────────────────────────────────────────────────────────


def _make_scan_group(
    group_id: int,
    num_items: int,
    groups: dict[int, Group],
    expressions: dict[int, LogicalExpression],
) -> Group:
    """Create a Scan group."""
    scan_op = Scan(
        name="ds",
        dataset_id="ds",
        num_items=num_items,
        est_tokens_per_item=50.0,
    )
    scan_expr = LogicalExpression(scan_op, input_group_ids=[], group_id=group_id)
    group = Group(logical_expressions=[scan_expr], group_id=group_id)
    groups[group_id] = group
    expressions[scan_expr.expr_id] = scan_expr
    return group


def _make_filter_expr(
    group_id: int,
    input_group_id: int,
    groups: dict[int, Group],
    expressions: dict[int, LogicalExpression],
    filter_text: str = "the review is positive",
) -> LogicalExpression:
    """Create a Filter logical expression in the given group."""
    filter_op = Filter(name="filter_op", filter=filter_text)
    filter_expr = LogicalExpression(filter_op, input_group_ids=[input_group_id], group_id=group_id)
    group = Group(logical_expressions=[filter_expr], group_id=group_id)
    groups[group_id] = group
    expressions[filter_expr.expr_id] = filter_expr
    return filter_expr


# ══════════════════════════════════════════════════════════════════════
#  FilterToTopKFilter
# ══════════════════════════════════════════════════════════════════════


class TestFilterToTopKFilterMatchesPattern:
    """Tests for ``FilterToTopKFilter.matches_pattern``."""

    def test_matches_filter(self):
        """Should match on Filter expressions."""
        filter_op = Filter(name="f", filter="keep mammals")
        expr = LogicalExpression(filter_op, input_group_ids=[0], group_id=1)
        assert FilterToTopKFilter.matches_pattern(expr)

    def test_does_not_match_map(self):
        """Should not match on non-Filter expressions."""
        map_op = Map(name="m", fields=[{"name": "x", "type": "str", "description": "test"}])
        expr = LogicalExpression(map_op, input_group_ids=[0], group_id=1)
        assert not FilterToTopKFilter.matches_pattern(expr)

    def test_does_not_match_topk(self):
        """Should not match on TopK expressions."""
        topk_op = TopK(name="t", task="find best", k=5)
        expr = LogicalExpression(topk_op, input_group_ids=[0], group_id=1)
        assert not FilterToTopKFilter.matches_pattern(expr)


class TestFilterToTopKFilterSubstitute:
    """Tests for ``FilterToTopKFilter.substitute``."""

    def test_creates_topk_alternatives_for_each_k(self):
        """Substitute creates one TopK group + one Filter per k value."""
        groups: dict[int, Group] = {}
        expressions: dict[int, LogicalExpression] = {}

        _make_scan_group(group_id=0, num_items=100, groups=groups, expressions=expressions)
        filter_expr = _make_filter_expr(group_id=1, input_group_id=0, groups=groups, expressions=expressions)

        new_exprs, new_groups, next_gid = FilterToTopKFilter.substitute(
            filter_expr, groups, expressions, next_group_id=2
        )

        num_k = len(FilterToTopKFilter.k_values)
        assert len(new_groups) == num_k  # one TopK group per k value
        assert len(new_exprs) == 2 * num_k  # TopK expr + Filter expr per k
        assert next_gid == 2 + num_k

    def test_topk_uses_filter_text_as_task(self):
        """The TopK operator should use the filter text as its semantic search query."""
        groups: dict[int, Group] = {}
        expressions: dict[int, LogicalExpression] = {}

        _make_scan_group(group_id=0, num_items=100, groups=groups, expressions=expressions)
        filter_expr = _make_filter_expr(
            group_id=1, input_group_id=0, groups=groups, expressions=expressions,
            filter_text="the review mentions shipping problems",
        )

        new_exprs, _, _ = FilterToTopKFilter.substitute(
            filter_expr, groups, expressions, next_group_id=2
        )

        topk_exprs = [e for e in new_exprs if isinstance(e.operator, TopK)]
        assert len(topk_exprs) > 0
        for te in topk_exprs:
            assert te.operator.task == "the review mentions shipping problems"

    def test_topk_k_values_match_class_attribute(self):
        """Each TopK's k should be one of the fixed k_values."""
        groups: dict[int, Group] = {}
        expressions: dict[int, LogicalExpression] = {}

        _make_scan_group(group_id=0, num_items=200, groups=groups, expressions=expressions)
        filter_expr = _make_filter_expr(group_id=1, input_group_id=0, groups=groups, expressions=expressions)

        new_exprs, _, _ = FilterToTopKFilter.substitute(
            filter_expr, groups, expressions, next_group_id=2
        )

        topk_exprs = [e for e in new_exprs if isinstance(e.operator, TopK)]
        k_values = sorted(te.operator.k for te in topk_exprs)
        assert k_values == sorted(FilterToTopKFilter.k_values)

    def test_topk_names_include_k(self):
        """Each TopK operator name should embed the k value for traceability."""
        groups: dict[int, Group] = {}
        expressions: dict[int, LogicalExpression] = {}

        _make_scan_group(group_id=0, num_items=100, groups=groups, expressions=expressions)
        filter_expr = _make_filter_expr(group_id=1, input_group_id=0, groups=groups, expressions=expressions)

        new_exprs, _, _ = FilterToTopKFilter.substitute(
            filter_expr, groups, expressions, next_group_id=2
        )

        topk_exprs = [e for e in new_exprs if isinstance(e.operator, TopK)]
        for te in topk_exprs:
            assert str(te.operator.k) in te.operator.name

    def test_new_filter_reads_from_topk_group(self):
        """The new Filter expressions should read from the new TopK groups, not the original."""
        groups: dict[int, Group] = {}
        expressions: dict[int, LogicalExpression] = {}

        _make_scan_group(group_id=0, num_items=100, groups=groups, expressions=expressions)
        filter_expr = _make_filter_expr(group_id=1, input_group_id=0, groups=groups, expressions=expressions)

        new_exprs, new_groups, _ = FilterToTopKFilter.substitute(
            filter_expr, groups, expressions, next_group_id=2
        )

        filter_exprs = [e for e in new_exprs if isinstance(e.operator, Filter)]
        topk_group_ids = {g.group_id for g in new_groups}
        for fe in filter_exprs:
            # Each new filter should read from exactly one TopK group
            assert len(fe.input_group_ids) == 1
            assert fe.input_group_ids[0] in topk_group_ids

    def test_new_filter_stays_in_original_group(self):
        """The new Filter expressions should target the same group as the original Filter."""
        groups: dict[int, Group] = {}
        expressions: dict[int, LogicalExpression] = {}

        _make_scan_group(group_id=0, num_items=100, groups=groups, expressions=expressions)
        filter_expr = _make_filter_expr(group_id=1, input_group_id=0, groups=groups, expressions=expressions)

        new_exprs, _, _ = FilterToTopKFilter.substitute(
            filter_expr, groups, expressions, next_group_id=2
        )

        filter_exprs = [e for e in new_exprs if isinstance(e.operator, Filter)]
        for fe in filter_exprs:
            assert fe.group_id == 1  # same group as original filter

    def test_topk_groups_registered_in_groups_dict(self):
        """New TopK groups should be registered in the shared groups dict."""
        groups: dict[int, Group] = {}
        expressions: dict[int, LogicalExpression] = {}

        _make_scan_group(group_id=0, num_items=100, groups=groups, expressions=expressions)
        filter_expr = _make_filter_expr(group_id=1, input_group_id=0, groups=groups, expressions=expressions)

        _, new_groups, _ = FilterToTopKFilter.substitute(
            filter_expr, groups, expressions, next_group_id=2
        )

        for g in new_groups:
            assert g.group_id in groups
            assert groups[g.group_id] is g

    def test_each_topk_group_has_one_logical_expression(self):
        """Each new TopK group should contain exactly one logical expression."""
        groups: dict[int, Group] = {}
        expressions: dict[int, LogicalExpression] = {}

        _make_scan_group(group_id=0, num_items=100, groups=groups, expressions=expressions)
        filter_expr = _make_filter_expr(group_id=1, input_group_id=0, groups=groups, expressions=expressions)

        _, new_groups, _ = FilterToTopKFilter.substitute(
            filter_expr, groups, expressions, next_group_id=2
        )

        for g in new_groups:
            assert len(g.logical_expressions) == 1
            topk_expr = next(iter(g.logical_expressions))
            assert isinstance(topk_expr.operator, TopK)


# ══════════════════════════════════════════════════════════════════════
#  get_available_model_ids
# ══════════════════════════════════════════════════════════════════════


class TestGetAvailableModelIds:
    """Tests for ``get_available_model_ids``."""

    def test_openai_key_returns_openai_models(self):
        """With only OPENAI_API_KEY, returns OpenAI tiered models."""
        config = {"OPENAI_API_KEY": "sk-test"}
        model_ids = get_available_model_ids(config)
        assert len(model_ids) >= 2
        assert any("gpt" in m for m in model_ids)

    def test_anthropic_key_returns_anthropic_models(self):
        """With only ANTHROPIC_API_KEY, returns Anthropic tiered models."""
        config = {"ANTHROPIC_API_KEY": "sk-ant-test"}
        model_ids = get_available_model_ids(config)
        assert len(model_ids) >= 2
        assert any("claude" in m for m in model_ids)

    def test_multiple_keys_returns_union(self):
        """With multiple API keys, returns models from all providers."""
        config = {
            "OPENAI_API_KEY": "sk-test",
            "ANTHROPIC_API_KEY": "sk-ant-test",
        }
        model_ids = get_available_model_ids(config)
        assert any("gpt" in m for m in model_ids)
        assert any("claude" in m for m in model_ids)

    def test_empty_config_returns_empty(self):
        """With no API keys, returns empty list."""
        model_ids = get_available_model_ids({})
        assert model_ids == []

    def test_empty_key_value_is_skipped(self):
        """An API key with empty string value is treated as absent."""
        config = {"OPENAI_API_KEY": ""}
        model_ids = get_available_model_ids(config)
        assert model_ids == []

    def test_no_duplicates(self):
        """GEMINI_API_KEY and GOOGLE_API_KEY map to same models — no duplicates."""
        config = {
            "GEMINI_API_KEY": "AIza-test",
            "GOOGLE_API_KEY": "AIza-test2",
        }
        model_ids = get_available_model_ids(config)
        assert len(model_ids) == len(set(model_ids))

    def test_together_ai_key(self):
        """TOGETHER_API_KEY returns Together AI models."""
        config = {"TOGETHER_API_KEY": "tok-test"}
        model_ids = get_available_model_ids(config)
        assert len(model_ids) >= 1
        assert any("together_ai" in m for m in model_ids)

    def test_ordering_preserved_within_provider(self):
        """Within a provider, the order from the tiered list is preserved."""
        from carnot.optimizer.model_ids import _OPENAI_MODELS

        config = {"OPENAI_API_KEY": "sk-test"}
        model_ids = get_available_model_ids(config)
        # All OpenAI models should appear in their original order
        openai_in_result = [m for m in model_ids if m in _OPENAI_MODELS]
        assert openai_in_result == _OPENAI_MODELS
