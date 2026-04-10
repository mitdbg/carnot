"""Tier 1 unit tests for the stats model hierarchy in ``carnot.core.models``.

Tests cover:
1. ``LLMCallStats`` — construction, defaults, derived properties,
   call_type discriminator, multimodal token fields.
2. ``OperatorStats`` — aggregation over ``llm_calls``, completion vs
   embedding call counts.
3. ``PhaseStats`` — summation across operators.
4. ``ExecutionStats`` — two-phase rollup and ``to_summary_dict()``.

All tests are Tier 1 (no mocks, no LLM, < 1 s).
"""

from __future__ import annotations

import pytest

from carnot.core.models import (
    ExecutionStats,
    LLMCallStats,
    OperatorStats,
    PhaseStats,
)

# ── LLMCallStats ────────────────────────────────────────────────────────────


class TestLLMCallStats:
    """Unit tests for ``LLMCallStats``."""

    def test_defaults(self):
        """All numeric fields default to zero; call_type defaults to 'completion'."""
        stats = LLMCallStats(model_id="gpt-4o")
        assert stats.call_type == "completion"
        assert stats.input_text_tokens == 0
        assert stats.output_text_tokens == 0
        assert stats.input_audio_tokens == 0
        assert stats.output_audio_tokens == 0
        assert stats.input_image_tokens == 0
        assert stats.cache_read_tokens == 0
        assert stats.cache_creation_tokens == 0
        assert stats.embedding_input_tokens == 0
        assert stats.cost_usd == 0.0
        assert stats.duration_secs == 0.0

    def test_is_completion(self):
        """call_type='completion' makes is_completion True, is_embedding False."""
        stats = LLMCallStats(model_id="gpt-4o", call_type="completion")
        assert stats.is_completion is True
        assert stats.is_embedding is False

    def test_is_embedding(self):
        """call_type='embedding' makes is_embedding True, is_completion False."""
        stats = LLMCallStats(model_id="text-embedding-3-small", call_type="embedding")
        assert stats.is_embedding is True
        assert stats.is_completion is False

    def test_total_input_tokens_text_only(self):
        """total_input_tokens sums text + audio + image + embedding."""
        stats = LLMCallStats(model_id="gpt-4o", input_text_tokens=100)
        assert stats.total_input_tokens == 100

    def test_total_input_tokens_multimodal(self):
        """Multimodal input tokens contribute to total_input_tokens."""
        stats = LLMCallStats(
            model_id="gpt-4o",
            input_text_tokens=100,
            input_audio_tokens=50,
            input_image_tokens=30,
        )
        assert stats.total_input_tokens == 180

    def test_total_input_tokens_embedding(self):
        """Embedding input tokens contribute to total_input_tokens."""
        stats = LLMCallStats(
            model_id="text-embedding-3-small",
            call_type="embedding",
            embedding_input_tokens=500,
        )
        assert stats.total_input_tokens == 500

    def test_total_output_tokens(self):
        """total_output_tokens sums text + audio output."""
        stats = LLMCallStats(
            model_id="gpt-4o",
            output_text_tokens=200,
            output_audio_tokens=75,
        )
        assert stats.total_output_tokens == 275

    def test_total_output_tokens_text_only(self):
        """Output tokens with no audio."""
        stats = LLMCallStats(model_id="gpt-4o", output_text_tokens=50)
        assert stats.total_output_tokens == 50

    def test_serialization_round_trip(self):
        """model_dump / model_validate round trip preserves all fields."""
        stats = LLMCallStats(
            model_id="gpt-4o",
            call_type="completion",
            input_text_tokens=100,
            output_text_tokens=50,
            input_audio_tokens=10,
            output_audio_tokens=5,
            input_image_tokens=20,
            cache_read_tokens=15,
            cache_creation_tokens=3,
            cost_usd=0.005,
            duration_secs=1.23,
        )
        dumped = stats.model_dump()
        restored = LLMCallStats.model_validate(dumped)
        assert restored == stats

    def test_json_serialization(self):
        """model_dump_json / model_validate_json round trip."""
        stats = LLMCallStats(model_id="gpt-4o", cost_usd=0.01)
        json_str = stats.model_dump_json()
        restored = LLMCallStats.model_validate_json(json_str)
        assert restored == stats


# ── OperatorStats ────────────────────────────────────────────────────────────


class TestOperatorStats:
    """Unit tests for ``OperatorStats``."""

    @pytest.fixture
    def mixed_calls(self) -> list[LLMCallStats]:
        """Two completion calls and one embedding call."""
        return [
            LLMCallStats(
                model_id="gpt-4o",
                call_type="completion",
                input_text_tokens=100,
                output_text_tokens=50,
                input_audio_tokens=10,
                input_image_tokens=5,
                cost_usd=0.003,
                duration_secs=0.5,
            ),
            LLMCallStats(
                model_id="gpt-4o",
                call_type="completion",
                input_text_tokens=200,
                output_text_tokens=80,
                output_audio_tokens=20,
                cost_usd=0.005,
                duration_secs=0.8,
            ),
            LLMCallStats(
                model_id="text-embedding-3-small",
                call_type="embedding",
                embedding_input_tokens=500,
                cost_usd=0.0001,
                duration_secs=0.2,
            ),
        ]

    def test_empty_calls(self):
        """Operator with no LLM calls has zero totals."""
        op = OperatorStats(operator_name="Limit")
        assert op.total_input_tokens == 0
        assert op.total_output_tokens == 0
        assert op.total_cost_usd == 0.0
        assert op.total_llm_calls == 0
        assert op.total_embedding_calls == 0
        assert op.total_llm_duration_secs == 0.0

    def test_total_input_tokens(self, mixed_calls):
        """total_input_tokens sums across all calls."""
        op = OperatorStats(operator_name="SemFilter", llm_calls=mixed_calls)
        # call1: 100+10+5=115, call2: 200=200, call3: 500 embed
        assert op.total_input_tokens == 115 + 200 + 500

    def test_total_output_tokens(self, mixed_calls):
        """total_output_tokens sums across all calls."""
        op = OperatorStats(operator_name="SemFilter", llm_calls=mixed_calls)
        # call1: 50, call2: 80+20=100, call3: 0
        assert op.total_output_tokens == 50 + 100 + 0

    def test_total_input_text_tokens(self, mixed_calls):
        """Sums only input_text_tokens fields."""
        op = OperatorStats(operator_name="SemFilter", llm_calls=mixed_calls)
        assert op.total_input_text_tokens == 100 + 200

    def test_total_input_audio_tokens(self, mixed_calls):
        """Sums only input_audio_tokens fields."""
        op = OperatorStats(operator_name="SemFilter", llm_calls=mixed_calls)
        assert op.total_input_audio_tokens == 10

    def test_total_input_image_tokens(self, mixed_calls):
        """Sums only input_image_tokens fields."""
        op = OperatorStats(operator_name="SemFilter", llm_calls=mixed_calls)
        assert op.total_input_image_tokens == 5

    def test_total_embedding_input_tokens(self, mixed_calls):
        """Sums only embedding_input_tokens fields."""
        op = OperatorStats(operator_name="SemFilter", llm_calls=mixed_calls)
        assert op.total_embedding_input_tokens == 500

    def test_total_cost_usd(self, mixed_calls):
        """Sums cost across all calls."""
        op = OperatorStats(operator_name="SemFilter", llm_calls=mixed_calls)
        assert op.total_cost_usd == pytest.approx(0.003 + 0.005 + 0.0001)

    def test_total_llm_calls(self, mixed_calls):
        """Counts only completion (non-embedding) calls."""
        op = OperatorStats(operator_name="SemFilter", llm_calls=mixed_calls)
        assert op.total_llm_calls == 2

    def test_total_embedding_calls(self, mixed_calls):
        """Counts only embedding calls."""
        op = OperatorStats(operator_name="SemFilter", llm_calls=mixed_calls)
        assert op.total_embedding_calls == 1

    def test_total_llm_duration_secs(self, mixed_calls):
        """Sums duration across all calls."""
        op = OperatorStats(operator_name="SemFilter", llm_calls=mixed_calls)
        assert op.total_llm_duration_secs == pytest.approx(0.5 + 0.8 + 0.2)

    def test_items_in_out(self):
        """items_in and items_out are tracked."""
        op = OperatorStats(
            operator_name="SemFilter",
            items_in=10,
            items_out=4,
        )
        assert op.items_in == 10
        assert op.items_out == 4


# ── PhaseStats ───────────────────────────────────────────────────────────────


class TestPhaseStats:
    """Unit tests for ``PhaseStats``."""

    def test_empty_phase(self):
        """Empty phase has zero totals."""
        phase = PhaseStats(phase="planning")
        assert phase.total_cost_usd == 0.0
        assert phase.total_input_tokens == 0
        assert phase.total_output_tokens == 0

    def test_aggregation_across_operators(self):
        """PhaseStats aggregates cost and tokens across multiple operators."""
        op1 = OperatorStats(
            operator_name="DataDiscovery",
            llm_calls=[
                LLMCallStats(model_id="gpt-4o", input_text_tokens=100, output_text_tokens=20, cost_usd=0.01),
            ],
        )
        op2 = OperatorStats(
            operator_name="Planner",
            llm_calls=[
                LLMCallStats(model_id="gpt-4o", input_text_tokens=200, output_text_tokens=40, cost_usd=0.02),
                LLMCallStats(model_id="gpt-4o", input_text_tokens=150, output_text_tokens=30, cost_usd=0.015),
            ],
        )

        phase = PhaseStats(phase="planning", wall_clock_secs=3.5, operator_stats=[op1, op2])

        assert phase.total_cost_usd == pytest.approx(0.01 + 0.02 + 0.015)
        assert phase.total_input_tokens == 100 + 200 + 150
        assert phase.total_output_tokens == 20 + 40 + 30
        assert phase.wall_clock_secs == 3.5


# ── ExecutionStats ───────────────────────────────────────────────────────────


class TestExecutionStats:
    """Unit tests for ``ExecutionStats``."""

    @pytest.fixture
    def full_stats(self) -> ExecutionStats:
        """An ExecutionStats with planning and execution phases populated."""
        planning = PhaseStats(
            phase="planning",
            wall_clock_secs=2.0,
            operator_stats=[
                OperatorStats(
                    operator_name="Planner",
                    llm_calls=[
                        LLMCallStats(model_id="gpt-4o", input_text_tokens=500, output_text_tokens=100, cost_usd=0.05),
                    ],
                ),
            ],
        )
        execution = PhaseStats(
            phase="execution",
            wall_clock_secs=5.0,
            operator_stats=[
                OperatorStats(
                    operator_name="SemFilter",
                    llm_calls=[
                        LLMCallStats(model_id="gpt-4o", input_text_tokens=300, output_text_tokens=50, cost_usd=0.03),
                    ],
                    items_in=10,
                    items_out=4,
                ),
                OperatorStats(
                    operator_name="SemTopK",
                    llm_calls=[
                        LLMCallStats(
                            model_id="text-embedding-3-small",
                            call_type="embedding",
                            embedding_input_tokens=1000,
                            cost_usd=0.001,
                        ),
                    ],
                    items_in=100,
                    items_out=10,
                ),
            ],
        )
        return ExecutionStats(
            execution_id="test-123",
            query="Find papers about transformers",
            planning=planning,
            execution=execution,
        )

    def test_defaults(self):
        """Default ExecutionStats has empty phases with zero totals."""
        stats = ExecutionStats()
        assert stats.total_cost_usd == 0.0
        assert stats.total_wall_clock_secs == 0.0
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0

    def test_total_cost_usd(self, full_stats):
        """Total cost combines planning and execution phases."""
        assert full_stats.total_cost_usd == pytest.approx(0.05 + 0.03 + 0.001)

    def test_total_wall_clock_secs(self, full_stats):
        """Total wall clock combines both phases."""
        assert full_stats.total_wall_clock_secs == pytest.approx(2.0 + 5.0)

    def test_total_input_tokens(self, full_stats):
        """Total input tokens across both phases."""
        # planning: 500, execution: 300 + 1000
        assert full_stats.total_input_tokens == 500 + 300 + 1000

    def test_total_output_tokens(self, full_stats):
        """Total output tokens across both phases."""
        # planning: 100, execution: 50 + 0
        assert full_stats.total_output_tokens == 100 + 50

    def test_to_summary_dict_keys(self, full_stats):
        """to_summary_dict returns expected top-level keys."""
        d = full_stats.to_summary_dict()
        expected_keys = {
            "execution_id",
            "query",
            "total_cost_usd",
            "total_wall_clock_secs",
            "total_input_tokens",
            "total_output_tokens",
            "planning",
            "execution",
        }
        assert set(d.keys()) == expected_keys

    def test_to_summary_dict_values(self, full_stats):
        """to_summary_dict values match the computed properties."""
        d = full_stats.to_summary_dict()
        assert d["execution_id"] == "test-123"
        assert d["query"] == "Find papers about transformers"
        assert d["total_cost_usd"] == pytest.approx(full_stats.total_cost_usd)
        assert d["total_wall_clock_secs"] == pytest.approx(full_stats.total_wall_clock_secs)

    def test_to_summary_dict_operator_breakdown(self, full_stats):
        """to_summary_dict includes per-operator breakdown."""
        d = full_stats.to_summary_dict()

        planning_ops = d["planning"]["operator_stats"]
        assert len(planning_ops) == 1
        assert planning_ops[0]["operator_name"] == "Planner"

        execution_ops = d["execution"]["operator_stats"]
        assert len(execution_ops) == 2
        assert execution_ops[0]["operator_name"] == "SemFilter"
        assert execution_ops[0]["items_in"] == 10
        assert execution_ops[0]["items_out"] == 4
        assert execution_ops[1]["operator_name"] == "SemTopK"
        assert execution_ops[1]["total_embedding_calls"] == 1
        assert execution_ops[1]["total_embedding_input_tokens"] == 1000

    def test_to_summary_dict_multimodal_fields(self):
        """to_summary_dict includes multimodal token breakdowns per operator."""
        op = OperatorStats(
            operator_name="SemFilter",
            llm_calls=[
                LLMCallStats(
                    model_id="gpt-4o",
                    input_text_tokens=100,
                    input_audio_tokens=50,
                    input_image_tokens=30,
                    output_text_tokens=20,
                    output_audio_tokens=10,
                ),
            ],
        )
        stats = ExecutionStats(
            execution=PhaseStats(phase="execution", operator_stats=[op]),
        )
        d = stats.to_summary_dict()
        op_dict = d["execution"]["operator_stats"][0]
        assert op_dict["total_input_text_tokens"] == 100
        assert op_dict["total_input_audio_tokens"] == 50
        assert op_dict["total_input_image_tokens"] == 30
