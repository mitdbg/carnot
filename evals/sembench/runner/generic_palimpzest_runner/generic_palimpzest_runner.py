"""
Created on May 28, 2025

@author: Andi Zimmerer, Jiale Lao

Palimpzest system runner implementation based on generic_runner.
"""

import re
import time
import traceback
from overrides import override
from typing import List, Optional

import litellm
import palimpzest as pz
import pandas as pd
from palimpzest.constants import Model
import json
import os

from runner.generic_runner import GenericRunner, GenericQueryMetric

litellm.drop_params = True


class GenericPalimpzestRunner(GenericRunner):
    """GenericRunner for Palimpzest system."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
        config_file: Optional[str] = None,
    ):
        """
        Initialize Palimpzest runner.

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
        # Check for environment variable for evaluation purposes
        env_config_file = os.getenv("PALIMPZEST_CONFIG_FILE")
        self.config_file = config_file or env_config_file
        self.config_data = self._load_config() if self.config_file else None

    @override
    def get_system_name(self) -> str:
        return "palimpzest"

    def _load_config(self) -> dict:
        """
        Load configuration from JSON file.

        Returns:
            Configuration dictionary
        """
        if not self.config_file:
            return {}

        from pathlib import Path

        config_path = (
            Path(__file__).parent.parent.parent.parent
            / "config"
            / "system"
            / "palimpzest"
            / self.config_file
        )

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with open(config_path, "r") as f:
            return json.load(f)

    def _get_policy_from_config(self, policy_name: str):
        """
        Convert policy name string to Palimpzest policy object.

        Args:
            policy_name: Name of the policy ("MaxQuality" or "MinCost")

        Returns:
            Palimpzest policy object
        """
        if policy_name == "MaxQuality":
            return pz.MaxQuality()
        elif policy_name == "MinCost":
            return pz.MinCost()
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

    def _get_models_from_config(self, model_names: List[str]) -> List[Model]:
        """
        Convert model name strings to Palimpzest Model constants.

        Args:
            model_names: List of model name strings

        Returns:
            List of Model constants
        """
        model_mapping = {
            "GEMINI_2_0_FLASH": Model.GEMINI_2_0_FLASH,
            "GEMINI_2_5_FLASH": Model.GEMINI_2_5_FLASH,
            "GPT_4o_MINI": Model.GPT_4o_MINI,
            "GPT_5_MINI": Model.GPT_5_MINI,
        }

        models = []
        for model_name in model_names:
            if model_name in model_mapping:
                models.append(model_mapping[model_name])
            else:
                raise ValueError(f"Unknown model: {model_name}")

        return models

    def _get_model_from_name(self, model_name: str) -> Model:
        """
        Convert self.model_name string to Palimpzest Model constant.

        Args:
            model_name: String name of the model

        Returns:
            Model constant
        """
        # Map from common model name strings to Palimpzest constants
        model_mapping = {
            "gemini-2.0-flash": Model.GEMINI_2_0_FLASH,
            "gemini-2.5-flash": Model.GEMINI_2_5_FLASH,
            "gpt-4o-mini": Model.GPT_4o_MINI,
            "gpt-5-mini": Model.GPT_5_MINI,
        }

        if model_name in model_mapping:
            return model_mapping[model_name]
        else:
            # Try to find a partial match or default to GEMINI_2_5_FLASH
            model_lower = model_name.lower()
            if "gemini-2.0" in model_lower or "gemini_2_0" in model_lower:
                return Model.GEMINI_2_0_FLASH
            elif "gemini-2.5" in model_lower or "gemini_2_5" in model_lower:
                return Model.GEMINI_2_5_FLASH
            elif "gpt-4o-mini" in model_lower or "gpt_4o_mini" in model_lower:
                return Model.GPT_4o_MINI
            elif "gpt-5-mini" in model_lower or "gpt_5_mini" in model_lower:
                return Model.GPT_5_MINI
            else:
                print(
                    f"Warning: Unknown model '{model_name}', defaulting to GEMINI_2_5_FLASH"
                )
                return Model.GEMINI_2_5_FLASH

    def _should_use_reasoning_effort(self, model: Model) -> bool:
        """
        Check if a model supports/benefits from reasoning_effort parameter.

        Args:
            model: Palimpzest Model constant

        Returns:
            True if reasoning_effort should be used
        """
        # Models that support reasoning effort (typically reasoning models)
        reasoning_models = {Model.GPT_5_MINI}
        return model in reasoning_models

    def palimpzest_config(self) -> pz.QueryProcessorConfig:
        """
        Create Palimpzest configuration.

        Returns:
            QueryProcessorConfig for Palimpzest
        """

        # Use configuration data if available, otherwise use defaults
        if self.config_data:
            config_kwargs = {
                "policy": self._get_policy_from_config(
                    self.config_data["policy"]
                ),
                "execution_strategy": self.config_data.get(
                    "execution_strategy", "parallel"
                ),
                "max_workers": self.config_data.get(
                    "max_workers", self.concurrent_llm_worker
                ),
                "join_parallelism": self.config_data.get(
                    "join_parallelism", self.concurrent_llm_worker
                ),
                "verbose": self.config_data.get("verbose", False),
                "progress": self.config_data.get("progress", True),
                "available_models": self._get_models_from_config(
                    self.config_data["available_models"]
                ),
            }

            # Only add reasoning_effort if it's not null in the config
            reasoning_effort = self.config_data.get("reasoning_effort")
            if reasoning_effort is not None:
                config_kwargs["reasoning_effort"] = reasoning_effort

            return pz.QueryProcessorConfig(**config_kwargs)
        else:
            # Use self.model_name to determine the model when config_data is not provided
            selected_model = self._get_model_from_name(self.model_name)

            config_kwargs = {
                "policy": pz.MaxQuality(),
                "execution_strategy": "parallel",
                "max_workers": self.concurrent_llm_worker,
                "join_parallelism": self.concurrent_llm_worker,
                "verbose": False,
                "progress": True,
                "available_models": [selected_model],
            }

            # Add reasoning_effort for compatible models
            if self._should_use_reasoning_effort(selected_model):
                config_kwargs["reasoning_effort"] = (
                    "minimal"  # Use minimal reasoning effort
                )

            return pz.QueryProcessorConfig(**config_kwargs)

    def execute_query(self, query_id: int) -> GenericQueryMetric:
        """
        Execute a specific query using Palimpzest and return metric with
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
            results = query_fn()
            execution_time = time.time() - start_time

            # Store results in metric
            metric.execution_time = execution_time
            metric.status = "success"

            # tianji: In case we need to do post-processing on PZ results, we
            # allow return type to be dictionary
            if isinstance(results, dict):
                metric.results = results["results"]
                self._update_token_usage(metric, results["execution_stats"])
            else:  # old logic
                metric.results = (
                    results.to_df()
                    if not isinstance(results, pd.DataFrame)
                    else results
                )
                # Get token usage and cost from execution stats
                self._update_token_usage(metric, results.execution_stats)

        except Exception as e:
            # Handle failure
            metric.status = "failed"
            metric.error = str(e)
            metric.results = self._get_empty_results_dataframe(query_id)
            print(f"  Error in Q{query_id} execution: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

        return metric

    def _update_token_usage(self, metric: GenericQueryMetric, exec_stats):
        """Update metric with token usage and cost information."""
        try:
            # Get usage stats from Palimpzest execution stats
            metric.token_usage = exec_stats.total_tokens
            metric.money_cost = exec_stats.total_execution_cost

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
        if query_id == 1:
            return pd.DataFrame(columns=["reviewId", "movieId", "reviewText"])
        elif query_id == 2:
            return pd.DataFrame(columns=["reviewId", "movieId", "reviewText"])
        elif query_id == 3:
            return pd.DataFrame(columns=["count"])
        elif query_id == 4:
            return pd.DataFrame(columns=["average"])
        elif query_id == 5:
            return pd.DataFrame(
                columns=[
                    "movieId",
                    "reviewId_left",
                    "reviewText_left",
                    "reviewId_right",
                    "reviewText_right",
                ]
            )
        else:
            return pd.DataFrame()

    def _discover_queries(self) -> List[int]:
        """
        Discover available queries for Palimpzest.

        Any method named ``_execute_q<i>`` (where <i> is an integer ≥1) is
        treated as an implemented query.  The function returns the list of
        those integer IDs in ascending order.

        Returns:
            List of available query IDs
        """
        pattern = re.compile(r"_execute_q(\d+)$")
        query_ids: List[int] = []

        # `dir(self)` lists all attribute names visible on the instance;
        # we then pick out callables whose names match our pattern.
        for attr_name in dir(self):
            match = pattern.match(attr_name)
            if match:
                attr = getattr(self, attr_name, None)
                if callable(attr):
                    query_ids.append(int(match.group(1)))

        return sorted(query_ids)
