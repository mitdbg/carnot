"""
Created on May 28, 2025

@author: Andi Zimmerer, Jiale Lao

Carnot system runner implementation based on generic_runner.
"""

import re
import time
import traceback

import litellm
import pandas as pd
from overrides import override
from runner.generic_runner import GenericQueryMetric, GenericRunner

litellm.drop_params = True


class GenericCarnotRunner(GenericRunner):
    """GenericRunner for Carnot system."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        """
        Initialize Carnot runner.

        Args:
            use_case: The use case to run
            model_name: LLM model to use
            concurrent_llm_worker: Number of concurrent workers
            skip_setup: Whether to skip scenario setup
            config_file: Optional path to JSON configuration file
        """
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )
        self.config_file = None
        self.config_data = None

    @override
    def get_system_name(self) -> str:
        return "carnot"

    def execute_query(self, query_id: int) -> GenericQueryMetric:
        """
        Execute a specific query using Carnot and return metric with
        results.

        Args:
            query_id: ID of the query (e.g., 1 for Q1, 5 for Q5)

        Returns:
            QueryMetric object containing results DataFrame and metrics
        """
        # Create appropriate metric object
        metric = GenericQueryMetric(query_id=query_id, status="pending")

        try:
            query_fn = self._discover_query_impl(query_id)
            start_time = time.time()
            results, execution_stats = query_fn()
            execution_time = time.time() - start_time

            # Store results in metric
            metric.execution_time = execution_time
            metric.status = "success"
            metric.results = pd.DataFrame(results)
            self._update_token_usage(metric, execution_stats)

        except Exception as e:
            # Handle failure
            metric.status = "failed"
            metric.error = str(e)
            metric.results = self._get_empty_results_dataframe(query_id)
            print(f"  Error in Q{query_id} execution: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

        return metric

    # TODO: ensure execution stats has the relevant keys
    def _update_token_usage(self, metric: GenericQueryMetric, exec_stats: dict):
        """Update metric with token usage and cost information."""
        try:
            # Get usage stats from Carnot execution stats
            metric.token_usage = exec_stats["total_tokens"]
            metric.money_cost = exec_stats["total_execution_cost"]

            # Print usage for debugging
            print(
                f"  Token usage: {metric.token_usage} tokens, Cost: ${metric.money_cost:.4f}"  # noqa: E501
            )

        except Exception as e:
            print(f"  Warning: Could not get token usage: {e}")
            metric.token_usage = 0
            metric.money_cost = 0.0

    def _get_empty_results_dataframe(self, query_id: int) -> pd.DataFrame:
        """
        Get empty DataFrame with correct columns for a query.

        Args:
            query_id: ID of the query

        Returns:
            Empty DataFrame with correct columns
        """
        if query_id == 1 or query_id == 2:
            return pd.DataFrame(columns=["reviewId", "id", "reviewText"])
        elif query_id == 3:
            return pd.DataFrame(columns=["count"])
        elif query_id == 4:
            return pd.DataFrame(columns=["average"])
        elif query_id == 5:
            return pd.DataFrame(
                columns=[
                    "id",
                    "reviewId",
                    "reviewText",
                    "reviewId_right",
                    "reviewText_right",
                ]
            )
        else:
            return pd.DataFrame()

    def _discover_queries(self) -> list[int]:
        """
        Discover available queries for Carnot.

        Any method named ``_execute_q<i>`` (where <i> is an integer ≥1) is
        treated as an implemented query.  The function returns the list of
        those integer IDs in ascending order.

        Returns:
            List of available query IDs
        """
        pattern = re.compile(r"_execute_q(\d+)$")
        query_ids: list[int] = []

        # `dir(self)` lists all attribute names visible on the instance;
        # we then pick out callables whose names match our pattern.
        for attr_name in dir(self):
            match = pattern.match(attr_name)
            if match:
                attr = getattr(self, attr_name, None)
                if callable(attr):
                    query_ids.append(int(match.group(1)))

        return sorted(query_ids)
