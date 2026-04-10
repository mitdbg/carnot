import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from evaluator.generic_evaluator import (
    GenericEvaluator,
    QueryMetricAggregation,
    QueryMetricRetrieval,
    SingleAccuracyScore,
)

sys.path.append(str(Path(__file__).parent.parent))
from ecomm.ecomm_scenario import EcommScenario


class EcommEvaluator(GenericEvaluator):
    def __init__(self, use_case: str, scale_factor: int = None) -> None:
        super().__init__(use_case, scale_factor)
        # Ideally, we would only have one intantiation of the scenario handler in run.py
        self.scenario_handler = EcommScenario(scale_factor=scale_factor)

    def _load_domain_data(self) -> None:
        # Not needed because this is already done by the ecomm scenario handler
        pass

    def _get_ground_truth(self, query_id: int) -> pd.DataFrame:
        return self.scenario_handler.get_ground_truth(query_id)

    def _evaluate_single_query(
        self,
        query_id: int,
        system_results: pd.DataFrame,
        ground_truth: pd.DataFrame,
    ) -> "QueryMetricRetrieval | QueryMetricAggregation | SingleAccuracyScore":
        # The following code is very generic and could be used for any scenario
        eval_measure = self.scenario_handler.get_accuracy_measure_for_query(
            query_id
        )
        return GenericEvaluator.compute_accuracy_score(
            eval_measure, ground_truth, system_results
        )
