from __future__ import annotations

import os
import time
from typing import Any

import litellm
from carnot.core.elements.filters import Filter
from carnot.core.elements.records import DataRecord, DataRecordSet
from carnot.core.models import GenerationStats, OperatorCostEstimates, RecordOpStats
from carnot.operators.physical import PhysicalOperator


class SmolAgentsFilter(PhysicalOperator):
    def __init__(self, filter: Filter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter = filter
        self.model_id = "openai/gpt-4o-mini"
        if os.getenv("ANTHROPIC_API_KEY"):
            self.model_id = "anthropic/claude-3-5-sonnet-20241022"
        elif os.getenv("GEMINI_API_KEY"):
            self.model_id = "vertex_ai/gemini-2.0-flash"
        elif os.getenv("TOGETHER_API_KEY"):
            self.model_id = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"

    def __str__(self):
        op = super().__str__()
        op += f"    Filter: {self.filter}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {
            "filter": self.filter,
            **id_params,
        }

    def get_op_params(self):
        op_params = super().get_op_params()
        return {
            "filter": self.filter,
            **op_params,
        }

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        return OperatorCostEstimates(
            cardinality=source_op_cost_estimates.cardinality * 0.5,  # Assume 50% selectivity
            time_per_record=0.1,
            cost_per_record=0.001,
            quality=1.0,
        )

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        start_time = time.time()
        
        passed = False
        generation_stats = GenerationStats(
            model_name="programmatic",
            total_input_tokens=0,
            total_output_tokens=0,
            total_input_cost=0.0,
            total_output_cost=0.0,
            cost_per_record=0.0,
            llm_call_duration_secs=0.0,
        )

        if self.filter.filter_fn is not None:
            # Programmatic filter
            try:
                # Convert candidate to dict for the filter function
                # TODO: support passing DataRecord directly if the function expects it?
                # For now assume it expects a dict or the schema object
                item = candidate.to_dict(include_bytes=False)
                # We might need to pass the actual schema object if the user defined function expects it
                # But usually pz.Filter works on the object properties.
                # Let's try passing the schema instance.
                schema_instance = self.input_schema(**item)
                passed = self.filter.filter_fn(schema_instance)
            except Exception as e:
                # If it fails, treat as False or log error
                passed = False
        
        elif self.filter.filter_condition is not None:
            # Semantic filter
            prompt = f"""
            Determine if the following record satisfies the condition: "{self.filter.filter_condition}"
            
            Record:
            {candidate.to_dict(include_bytes=False)}
            
            Answer with only "True" or "False".
            """
            
            try:
                response = litellm.completion(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content.strip().lower()
                passed = "true" in content
                
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                # Approximate costs
                cost_per_input_token = 0.15 / 1e6 
                cost_per_output_token = 0.6 / 1e6
                input_cost = input_tokens * cost_per_input_token
                output_cost = output_tokens * cost_per_output_token
                
                generation_stats = GenerationStats(
                    model_name=self.model_id,
                    total_input_tokens=input_tokens,
                    total_output_tokens=output_tokens,
                    total_input_cost=input_cost,
                    total_output_cost=output_cost,
                    cost_per_record=input_cost + output_cost,
                    llm_call_duration_secs=time.time() - start_time,
                )
            except Exception:
                passed = False

        if not passed:
            return DataRecordSet([], [])

        # If passed, return the candidate
        # We need to create a new DataRecord or just return the existing one wrapped?
        # Physical operators usually return new DataRecords with updated lineage.
        
        dr = DataRecord.from_parent(self.output_schema, candidate.to_dict(include_bytes=False), parent_record=candidate)
        
        record_op_stats = RecordOpStats(
            record_id=dr._id,
            record_parent_ids=dr._parent_ids,
            record_source_indices=dr._source_indices,
            record_state=dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=generation_stats.cost_per_record,
            model_name=generation_stats.model_name,
            total_input_tokens=generation_stats.total_input_tokens,
            total_output_tokens=generation_stats.total_output_tokens,
            total_input_cost=generation_stats.total_input_cost,
            total_output_cost=generation_stats.total_output_cost,
            llm_call_duration_secs=generation_stats.llm_call_duration_secs,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        return DataRecordSet([dr], [record_op_stats])
