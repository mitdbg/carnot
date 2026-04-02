from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMCallStats(BaseModel):
    """Statistics from a single LLM API call.

    Representation invariant:
        - All token counts are >= 0.
        - ``cost_usd >= 0.0``.
        - ``duration_secs >= 0.0``.
        - ``model_id`` is a non-empty string.
        - Exactly one of ``is_embedding`` or ``is_completion`` is True
          (determined by ``call_type``).
        - For embedding calls: ``output_text_tokens == 0`` and
          ``embedding_input_tokens > 0``.
        - For completion calls: ``embedding_input_tokens == 0``.

    Abstraction function:
        Represents the resource consumption of exactly one call to
        ``litellm.completion()`` or ``litellm.embedding()``.
    """

    model_id: str
    call_type: str = "completion"  # "completion" or "embedding"

    # --- text tokens (standard for all completion calls) ---
    input_text_tokens: int = 0
    output_text_tokens: int = 0

    # --- multimodal tokens ---
    input_audio_tokens: int = 0
    output_audio_tokens: int = 0
    input_image_tokens: int = 0

    # --- prompt caching tokens ---
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    # --- embedding tokens ---
    embedding_input_tokens: int = 0

    # --- cost & timing ---
    cost_usd: float = 0.0
    duration_secs: float = 0.0

    # --- derived helpers ---
    @property
    def is_embedding(self) -> bool:
        """True when this stats object represents an embedding call."""
        return self.call_type == "embedding"

    @property
    def is_completion(self) -> bool:
        """True when this stats object represents a completion call."""
        return self.call_type == "completion"

    @property
    def total_input_tokens(self) -> int:
        """Sum of all input token types (text + audio + image + embedding).

        Requires:
            None.

        Returns:
            Non-negative integer sum of ``input_text_tokens``,
            ``input_audio_tokens``, ``input_image_tokens``, and
            ``embedding_input_tokens``.

        Raises:
            None.
        """
        return (
            self.input_text_tokens
            + self.input_audio_tokens
            + self.input_image_tokens
            + self.embedding_input_tokens
        )

    @property
    def total_output_tokens(self) -> int:
        """Sum of all output token types (text + audio).

        Requires:
            None.

        Returns:
            Non-negative integer sum of ``output_text_tokens`` and
            ``output_audio_tokens``.

        Raises:
            None.
        """
        return self.output_text_tokens + self.output_audio_tokens


class OperatorStats(BaseModel):
    """Statistics from the execution of a single operator.

    Representation invariant:
        - ``operator_name`` is a non-empty string.
        - ``wall_clock_secs >= 0.0``.
        - ``llm_calls`` is a list of ``LLMCallStats`` (possibly empty
          for non-LLM operators like ``LimitOperator``).

    Abstraction function:
        Represents the total resource consumption of one operator
        invocation across all items it processed (including retries).
    """

    operator_name: str  # e.g. "SemFilter", "Planner", "DataDiscovery"
    operator_id: str = ""  # the dataset_id or agent name
    wall_clock_secs: float = 0.0
    llm_calls: list[LLMCallStats] = Field(default_factory=list)
    items_in: int = 0
    items_out: int = 0

    # ---- derived (computed) properties ----
    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all LLM calls in this operator.

        Requires:
            None.

        Returns:
            Non-negative integer sum of ``total_input_tokens`` for every
            call in ``llm_calls``.

        Raises:
            None.
        """
        return sum(c.total_input_tokens for c in self.llm_calls)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all LLM calls in this operator.

        Requires:
            None.

        Returns:
            Non-negative integer sum of ``total_output_tokens`` for every
            call in ``llm_calls``.

        Raises:
            None.
        """
        return sum(c.total_output_tokens for c in self.llm_calls)

    @property
    def total_input_text_tokens(self) -> int:
        """Total input text tokens across all LLM calls.

        Requires:
            None.

        Returns:
            Non-negative integer.

        Raises:
            None.
        """
        return sum(c.input_text_tokens for c in self.llm_calls)

    @property
    def total_input_audio_tokens(self) -> int:
        """Total input audio tokens across all LLM calls.

        Requires:
            None.

        Returns:
            Non-negative integer.

        Raises:
            None.
        """
        return sum(c.input_audio_tokens for c in self.llm_calls)

    @property
    def total_input_image_tokens(self) -> int:
        """Total input image tokens across all LLM calls.

        Requires:
            None.

        Returns:
            Non-negative integer.

        Raises:
            None.
        """
        return sum(c.input_image_tokens for c in self.llm_calls)

    @property
    def total_embedding_input_tokens(self) -> int:
        """Total embedding input tokens across all LLM calls.

        Requires:
            None.

        Returns:
            Non-negative integer.

        Raises:
            None.
        """
        return sum(c.embedding_input_tokens for c in self.llm_calls)

    @property
    def total_cost_usd(self) -> float:
        """Total dollar cost across all LLM calls.

        Requires:
            None.

        Returns:
            Non-negative float.

        Raises:
            None.
        """
        return sum(c.cost_usd for c in self.llm_calls)

    @property
    def total_llm_calls(self) -> int:
        """Number of completion (non-embedding) calls.

        Requires:
            None.

        Returns:
            Non-negative integer.

        Raises:
            None.
        """
        return len([c for c in self.llm_calls if not c.is_embedding])

    @property
    def total_embedding_calls(self) -> int:
        """Number of embedding calls.

        Requires:
            None.

        Returns:
            Non-negative integer.

        Raises:
            None.
        """
        return len([c for c in self.llm_calls if c.is_embedding])

    @property
    def total_llm_duration_secs(self) -> float:
        """Total wall-clock time spent in LLM API calls.

        Requires:
            None.

        Returns:
            Non-negative float.

        Raises:
            None.
        """
        return sum(c.duration_secs for c in self.llm_calls)


class PhaseStats(BaseModel):
    """Statistics from one execution phase (planning or execution).

    Representation invariant:
        - ``phase`` is one of ``"planning"`` or ``"execution"``.
        - ``operator_stats`` is a list of ``OperatorStats``.

    Abstraction function:
        Represents the total resource consumption of one major
        phase of query processing.
    """

    phase: str  # "planning" or "execution"
    wall_clock_secs: float = 0.0
    operator_stats: list[OperatorStats] = Field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        """Total dollar cost across all operators in this phase.

        Requires:
            None.

        Returns:
            Non-negative float.

        Raises:
            None.
        """
        return sum(op.total_cost_usd for op in self.operator_stats)

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all operators in this phase.

        Requires:
            None.

        Returns:
            Non-negative integer.

        Raises:
            None.
        """
        return sum(op.total_input_tokens for op in self.operator_stats)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all operators in this phase.

        Requires:
            None.

        Returns:
            Non-negative integer.

        Raises:
            None.
        """
        return sum(op.total_output_tokens for op in self.operator_stats)


class ExecutionStats(BaseModel):
    """Statistics for an entire Carnot query execution.

    Representation invariant:
        - ``planning`` and ``execution`` are ``PhaseStats`` objects
          (possibly with empty operator lists if that phase was skipped).

    Abstraction function:
        Represents the full cost/latency breakdown of a Carnot query,
        from planning through final answer generation.
    """

    execution_id: str = ""
    query: str = ""
    planning: PhaseStats = Field(default_factory=lambda: PhaseStats(phase="planning"))
    execution: PhaseStats = Field(default_factory=lambda: PhaseStats(phase="execution"))

    @property
    def total_cost_usd(self) -> float:
        """Total dollar cost across both phases.

        Requires:
            None.

        Returns:
            Non-negative float.

        Raises:
            None.
        """
        return self.planning.total_cost_usd + self.execution.total_cost_usd

    @property
    def total_wall_clock_secs(self) -> float:
        """Total wall-clock time across both phases.

        Requires:
            None.

        Returns:
            Non-negative float.

        Raises:
            None.
        """
        return self.planning.wall_clock_secs + self.execution.wall_clock_secs

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across both phases.

        Requires:
            None.

        Returns:
            Non-negative integer.

        Raises:
            None.
        """
        return self.planning.total_input_tokens + self.execution.total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across both phases.

        Requires:
            None.

        Returns:
            Non-negative integer.

        Raises:
            None.
        """
        return self.planning.total_output_tokens + self.execution.total_output_tokens

    def to_summary_dict(self) -> dict:
        """Flat summary suitable for logging or API responses.

        Requires:
            None.

        Returns:
            A dictionary with top-level cost/token/timing totals and
            nested ``planning`` and ``execution`` sub-dicts, each
            containing per-operator breakdowns.

        Raises:
            None.
        """
        def _phase_dict(phase: PhaseStats) -> dict:
            return {
                "phase": phase.phase,
                "wall_clock_secs": phase.wall_clock_secs,
                "total_cost_usd": phase.total_cost_usd,
                "total_input_tokens": phase.total_input_tokens,
                "total_output_tokens": phase.total_output_tokens,
                "operator_stats": [
                    {
                        "operator_name": op.operator_name,
                        "operator_id": op.operator_id,
                        "wall_clock_secs": op.wall_clock_secs,
                        "total_cost_usd": op.total_cost_usd,
                        "total_llm_calls": op.total_llm_calls,
                        "total_embedding_calls": op.total_embedding_calls,
                        "total_input_tokens": op.total_input_tokens,
                        "total_output_tokens": op.total_output_tokens,
                        "total_input_text_tokens": op.total_input_text_tokens,
                        "total_input_audio_tokens": op.total_input_audio_tokens,
                        "total_input_image_tokens": op.total_input_image_tokens,
                        "total_embedding_input_tokens": op.total_embedding_input_tokens,
                        "items_in": op.items_in,
                        "items_out": op.items_out,
                    }
                    for op in phase.operator_stats
                ],
            }

        return {
            "execution_id": self.execution_id,
            "query": self.query,
            "total_cost_usd": self.total_cost_usd,
            "total_wall_clock_secs": self.total_wall_clock_secs,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "planning": _phase_dict(self.planning),
            "execution": _phase_dict(self.execution),
        }


class PlanCost(BaseModel):
    """Unified cost estimate for a (sub-)plan.

    Combines cumulative plan-level metrics (``cost``, ``time``,
    ``total_input_tokens``, ``total_scanned_input_tokens``) with
    per-operator metrics (``cardinality``, ``cost_per_record``,
    ``time_per_record``).  This replaces the previous two-class design
    where ``OperatorCostEstimates`` was a separate model embedded inside
    ``PlanCost`` via an ``op_estimates`` field.

    Representation invariant:
        - All numeric fields are non-negative.
        - ``total_scanned_input_tokens <= total_input_tokens``.

    Abstraction function:
        Represents a point in the (cost, time, quality) space for a
        plan or sub-plan, where *quality* is the ratio of scanned
        input tokens to total input tokens.
    """

    # cumulative plan metrics
    cost: float
    time: float
    total_input_tokens: float
    total_scanned_input_tokens: float

    # per-operator metrics (set by the cost model for the current operator)
    cardinality: float = 0.0
    cost_per_record: float = 0.0
    time_per_record: float = 0.0

    @property
    def quality(self) -> float:
        """Ratio of scanned input tokens to total input tokens.

        Requires:
            None.

        Returns:
            ``total_scanned_input_tokens / total_input_tokens`` when
            ``total_input_tokens > 0``, otherwise ``1.0``.

        Raises:
            None.
        """
        if self.total_input_tokens == 0:
            return 1.0
        return self.total_scanned_input_tokens / self.total_input_tokens

    def __hash__(self):
        return hash(f"{self.cost}-{self.time}-{self.quality}")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PlanCost):
            return False
        return (
            self.cost == other.cost
            and self.time == other.time
            and self.quality == other.quality
        )

    def join_add(self, left_plan_cost: PlanCost, right_plan_cost: PlanCost) -> PlanCost:
        """Combine this operator's cost with two joined input plan costs.

        Sums ``cost``, ``time``, ``total_input_tokens`, and ``total_scanned_input_tokens``
        across all three ``PlanCost`` objects (operator + left + right).

        Per-operator fields (``cardinality``, ``cost_per_record``,
        ``time_per_record``) are taken from *self* (the operator cost).

        Requires:
            - *left_plan_cost* and *right_plan_cost* are valid ``PlanCost`` objects.

        Returns:
            A new ``PlanCost`` combining all three.

        Raises:
            None.
        """
        return PlanCost(
            cost=self.cost + left_plan_cost.cost + right_plan_cost.cost,
            time=self.time + left_plan_cost.time + right_plan_cost.time,
            total_input_tokens=(
                self.total_input_tokens + left_plan_cost.total_input_tokens + right_plan_cost.total_input_tokens
            ),
            total_scanned_input_tokens=(
                self.total_scanned_input_tokens + left_plan_cost.total_scanned_input_tokens + right_plan_cost.total_scanned_input_tokens
            ),
            cardinality=self.cardinality,
            cost_per_record=self.cost_per_record,
            time_per_record=self.time_per_record,
        )

    def __iadd__(self, other: PlanCost) -> PlanCost:
        """In-place addition of another ``PlanCost`` (non-join).

        Sums the cumulative fields.  Per-operator fields are left
        unchanged (caller should set them explicitly if needed).

        Requires:
            - *other* is a valid ``PlanCost``.

        Returns:
            *self*, mutated in place.

        Raises:
            None.
        """
        self.cost += other.cost
        self.time += other.time
        self.total_input_tokens += other.total_input_tokens
        self.total_scanned_input_tokens += other.total_scanned_input_tokens

        return self

    def __add__(self, other: PlanCost) -> PlanCost:
        """Add two ``PlanCost`` objects (non-join).

        Sums the cumulative fields.  Per-operator fields are taken from
        *self*.

        Requires:
            - *other* is a valid ``PlanCost``.

        Returns:
            A new ``PlanCost`` with summed cumulative fields.

        Raises:
            None.
        """
        return PlanCost(
            cost=self.cost + other.cost,
            time=self.time + other.time,
            total_input_tokens=self.total_input_tokens + other.total_input_tokens,
            total_scanned_input_tokens=self.total_scanned_input_tokens + other.total_scanned_input_tokens,
            cardinality=self.cardinality,
            cost_per_record=self.cost_per_record,
            time_per_record=self.time_per_record,
        )


# Backward-compatibility alias — deprecated.
OperatorCostEstimates = PlanCost
