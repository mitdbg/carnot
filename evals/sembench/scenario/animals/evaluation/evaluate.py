"""
Created on July 22, 2025

@author: Jiale Lao

AnimalsEvaluator Implementation based on generic_evaluator
Uses DuckDB with gold SQL files to generate ground truth
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from evaluator.generic_evaluator import GenericEvaluator, QueryMetricAggregation, QueryMetricRetrieval


class AnimalsEvaluator(GenericEvaluator):
    """Evaluator for the animals benchmark using the reusable framework."""

    def __init__(self, use_case: str, scale_factor: int = None) -> None:
        super().__init__(use_case, scale_factor)

    def _load_domain_data(self) -> None:
        """Load the animals data CSV files."""
        data_path = self._root / "data" / f"sf_{self.scale_factor}"
        self.image_data_df = pd.read_csv(data_path / "image_data.csv")
        self.audio_data_df = pd.read_csv(data_path / "audio_data.csv")

    def _get_ground_truth(self, query_id: int) -> pd.DataFrame:  
        """Generate ground truth using DuckDB and gold SQL files."""
        sql_path = self._root / "query" / "gold_sql" / f"Q{query_id}.sql"
        
        if not sql_path.exists():
            raise FileNotFoundError(f"Gold SQL file not found: {sql_path}")
        
        # Read the SQL query
        with open(sql_path) as f:
            sql_query = f.read().strip()
        
        # Create DuckDB connection and load data
        conn = duckdb.connect()
        
        # Register DataFrames as tables
        conn.register('ImageData', self.image_data_df)
        conn.register('AudioData', self.audio_data_df)
        
        # Execute the query and get results
        result = conn.execute(sql_query).fetchdf()
        
        # Save ground truth
        gt_path = self._results_path / "ground_truth" / f"Q{query_id}.csv"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(gt_path, index=False)
        
        conn.close()
        return result

    def _evaluate_single_query(self, query_id: int, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> "QueryMetricRetrieval | QueryMetricAggregation":
        """Evaluate a single query based on its type."""
        evaluate_fn = self._discover_evaluate_impl(query_id)
        return evaluate_fn(system_results, ground_truth)

    def _evaluate_q1(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricAggregation:
        """Q1: Count of zebra pictures - aggregation query."""
        return self._generic_aggregation_evaluation(system_results, ground_truth)
    
    def _evaluate_q2(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricAggregation:
        """Q2: Count of elephant audio recordings - aggregation query."""
        return self._generic_aggregation_evaluation(system_results, ground_truth)
    
    def _evaluate_q3(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricRetrieval:
        """Q3: City with most zebra pictures - retrieval query that may return multiple tied cities."""
        if len(ground_truth) == 0:
            return QueryMetricRetrieval(precision=1.0 if len(system_results) == 0 else 0.0)
        if len(system_results) == 0:
            return QueryMetricRetrieval()
        
        # Get the set of all valid cities from ground truth
        gt_cities = set(ground_truth.iloc[:, 0]) if len(ground_truth.columns) > 0 else set()
        
        # For these "most" queries, we expect system to return only one city
        # But that city should be one of the valid tied cities from ground truth
        if len(system_results) == 1:
            sys_city = system_results.iloc[0, 0] if len(system_results.columns) > 0 else None
            
            if sys_city in gt_cities:
                return QueryMetricRetrieval(precision=1.0, recall=1.0, f1_score=1.0)
        
        return QueryMetricRetrieval(precision=0.0, recall=0.0, f1_score=0.0)
    
    def _evaluate_q4(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricRetrieval:
        """Q4: City with most elephant audio recordings - retrieval query that may return multiple tied cities."""
        if len(ground_truth) == 0:
            return QueryMetricRetrieval(precision=1.0 if len(system_results) == 0 else 0.0)
        if len(system_results) == 0:
            return QueryMetricRetrieval()
        
        # Get the set of all valid cities from ground truth
        gt_cities = set(ground_truth.iloc[:, 0]) if len(ground_truth.columns) > 0 else set()
        
        # For these "most" queries, we expect system to return only one city
        # But that city should be one of the valid tied cities from ground truth
        if len(system_results) == 1:
            sys_city = system_results.iloc[0, 0] if len(system_results.columns) > 0 else None
            
            if sys_city in gt_cities:
                return QueryMetricRetrieval(precision=1.0, recall=1.0, f1_score=1.0)
        
        return QueryMetricRetrieval(precision=0.0, recall=0.0, f1_score=0.0)
    
    def _evaluate_q5(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricRetrieval:
        """Q5: Cities with elephant images or audio - retrieval query."""
        return self._generic_retrieval_evaluation(system_results, ground_truth)
    
    def _evaluate_q6(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricRetrieval:
        """Q6: Cities with monkey images but no audio - retrieval query."""
        return self._generic_retrieval_evaluation(system_results, ground_truth)
    
    def _evaluate_q7(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricRetrieval:
        """Q7: Cities where zebras and impala co-occur in images - retrieval query."""
        return self._generic_retrieval_evaluation(system_results, ground_truth)
    
    def _evaluate_q8(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricRetrieval:
        """Q8: Cities with elephants and monkeys in images or audio - retrieval query."""
        return self._generic_retrieval_evaluation(system_results, ground_truth)
    
    def _evaluate_q9(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricRetrieval:
        """Q9: Cities with both monkey images and audio - retrieval query."""
        return self._generic_retrieval_evaluation(system_results, ground_truth)
    
    def _evaluate_q10(self, system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> QueryMetricRetrieval:
        """Q10: City and station with most zebra pictures - retrieval query that may return multiple tied (city, station) pairs."""
        if len(ground_truth) == 0:
            return QueryMetricRetrieval(precision=1.0 if len(system_results) == 0 else 0.0)
        if len(system_results) == 0:
            return QueryMetricRetrieval()
        
        # Create case-insensitive column mapping for system_results
        sys_col_map = {col.lower(): col for col in system_results.columns}
        
        # Create set of valid (city, station) tuples from ground truth
        gt_tuples = set()
        for _, row in ground_truth.iterrows():
            if len(ground_truth.columns) >= 2:
                gt_tuples.add((row.iloc[0], row.iloc[1]))
        
        # For these "most" queries, we expect system to return only one (city, station) pair
        # But that pair should be one of the valid tied pairs from ground truth
        if len(system_results) == 1 and len(ground_truth.columns) >= 2:
            sys_row = system_results.iloc[0]
            
            # Extract city and station from system results (handle case-insensitive column names)
            city_col = None
            station_col = None
            
            for gt_col in ground_truth.columns[:2]:  # First two columns are city and station
                gt_col_lower = gt_col.lower()
                if gt_col_lower in sys_col_map:
                    if city_col is None:
                        city_col = sys_col_map[gt_col_lower]
                    else:
                        station_col = sys_col_map[gt_col_lower]
            
            if city_col and station_col:
                sys_tuple = (sys_row[city_col], sys_row[station_col])
                if sys_tuple in gt_tuples:
                    return QueryMetricRetrieval(precision=1.0, recall=1.0, f1_score=1.0)
        
        return QueryMetricRetrieval(precision=0.0, recall=0.0, f1_score=0.0)
