import logging
from concurrent.futures import ThreadPoolExecutor, wait

from carnot.constants import PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS
from carnot.core.elements.records import DataRecord
from carnot.core.models import PlanStats
from carnot.execution.execution_strategy import ExecutionStrategy
from carnot.operators.scan import ContextScanOp, ScanPhysicalOp
from carnot.optimizer.plan import PhysicalPlan
from carnot.utils.progress import create_progress_manager

logger = logging.getLogger(__name__)


class ParallelExecutionStrategy(ExecutionStrategy):
    """
    A parallel execution strategy that processes data through a pipeline of operators using thread-based parallelism.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _any_queue_not_empty(self, queues: dict[str, list] | dict[str, dict[str, list]]) -> bool:
        """Helper function to check if any queue is not empty."""
        for _, value in queues.items():
            if isinstance(value, dict):
                if any(len(subqueue) > 0 for subqueue in value.values()):
                    return True
            elif len(value) > 0:
                return True
        return False

    def _process_future_results(self, unique_full_op_id: str, future_queues: dict[str, list], plan_stats: PlanStats) -> list[DataRecord]:
        """
        Helper function which takes a full operator id, the future queues, and plan stats, and performs
        the updates to plan stats and progress manager before returning the results from the finished futures.
        """
        # this function is called when the future queue is not empty
        # and the executor is not busy processing other futures
        done_futures, not_done_futures = wait(future_queues[unique_full_op_id], timeout=PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS)

        # add the unfinished futures back to the previous op's future queue
        future_queues[unique_full_op_id] = list(not_done_futures)

        # add the finished futures to the input queue for this operator
        output_records, total_inputs_processed, total_cost = [], 0, 0.0
        for future in done_futures:
            output = future.result()
            record_set, num_inputs_processed = output, 1

            # record set can be empty if one side of join has no input records yet
            if len(record_set) == 0:
                continue

            # otherwise, process records and their stats
            records = record_set.data_records
            record_op_stats = record_set.record_op_stats

            # update the inputs processed and total cost
            total_inputs_processed += num_inputs_processed
            total_cost += record_set.get_total_cost()

            # update plan stats
            plan_stats.add_record_op_stats(unique_full_op_id, record_op_stats)

            # add records which aren't filtered to the output records
            output_records.extend([record for record in records if record._passed_operator])

        # update the progress manager
        if total_inputs_processed > 0:
            num_outputs = len(output_records)
            self.progress_manager.incr(unique_full_op_id, num_inputs=total_inputs_processed, num_outputs=num_outputs, total_cost=total_cost)

        return output_records

    def _execute_plan(
            self,
            plan: PhysicalPlan,
            input_queues: dict[str, dict[str, list]],
            future_queues: dict[str, list],
            plan_stats: PlanStats,
        ) -> tuple[list[DataRecord], PlanStats]:
        # process all of the input records using a thread pool
        output_records = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            logger.debug(f"Created thread pool with {self.max_workers} workers")

            # execute the plan until either:
            # 1. all records have been processed, or
            # 2. the final limit operation has completed (we break out of the loop if this happens)
            final_op = plan.operator
            while self._any_queue_not_empty(input_queues) or self._any_queue_not_empty(future_queues):
                for topo_idx, operator in enumerate(plan):
                    source_unique_full_op_ids = (
                        [f"source_{operator.get_full_op_id()}"]
                        if isinstance(operator, (ContextScanOp, ScanPhysicalOp))
                        else plan.get_source_unique_full_op_ids(topo_idx, operator)
                    )
                    unique_full_op_id = f"{topo_idx}-{operator.get_full_op_id()}"

                    # get any finished futures from the previous operator and add them to the input queue for this operator
                    if not isinstance(operator, (ContextScanOp, ScanPhysicalOp)):
                        for source_unique_full_op_id in source_unique_full_op_ids:
                            records = self._process_future_results(source_unique_full_op_id, future_queues, plan_stats)
                            input_queues[unique_full_op_id][source_unique_full_op_id].extend(records)

                    # for the final operator, add any finished futures to the output_records
                    if unique_full_op_id == f"{topo_idx}-{final_op.get_full_op_id()}":
                        records = self._process_future_results(unique_full_op_id, future_queues, plan_stats)
                        output_records.extend(records)

                    # if this operator does not have enough inputs to execute, then skip it
                    num_inputs = sum(len(inputs) for inputs in input_queues[unique_full_op_id].values())
                    if num_inputs == 0:
                        continue

                    # otherwise, process records according to batch size
                    source_unique_full_op_id = source_unique_full_op_ids[0]
                    input_records = input_queues[unique_full_op_id][source_unique_full_op_id]
                    if self.batch_size is None:
                        for input_record in input_records:
                            future = executor.submit(operator, input_record)
                            future_queues[unique_full_op_id].append(future)
                        input_queues[unique_full_op_id][source_unique_full_op_id].clear()
                    else:
                        batch_size = min(self.batch_size, len(input_records))
                        batch_input_records = input_records[:batch_size]
                        for input_record in batch_input_records:
                            future = executor.submit(operator, input_record)
                            future_queues[unique_full_op_id].append(future)
                        input_queues[unique_full_op_id][source_unique_full_op_id] = input_records[batch_size:]

        # finalize plan stats
        plan_stats.finish()

        return output_records, plan_stats

    def execute_plan(self, plan: PhysicalPlan):
        """Initialize the stats and execute the plan."""
        logger.info(f"Executing plan {plan.plan_id} with {self.max_workers} workers")
        logger.info(f"Plan Details: {plan}")

        # initialize plan stats
        plan_stats = PlanStats.from_plan(plan)
        plan_stats.start()

        # initialize input queues and future queues for each operation
        input_queues = self._create_input_queues(plan)
        future_queues = {f"{topo_idx}-{op.get_full_op_id()}": [] for topo_idx, op in enumerate(plan)}

        # Get session info from config if available
        session_id = getattr(self, 'session_id', None)
        progress_log_file = getattr(self, 'progress_log_file', None)
        
        # initialize and start the progress manager
        self.progress_manager = create_progress_manager(
            plan, 
            num_samples=self.num_samples, 
            progress=self.progress,
            session_id=session_id,
            progress_log_file=progress_log_file,
        )
        self.progress_manager.start()

        # NOTE: we must handle progress manager outside of _execute_plan to ensure that it is shut down correctly;
        #       if we don't have the `finally:` branch, then program crashes can cause future program runs to fail
        #       because the progress manager cannot get a handle to the console 
        try:
            # execute plan
            output_records, plan_stats = self._execute_plan(plan, input_queues, future_queues, plan_stats)

        finally:
            # finish progress tracking
            self.progress_manager.finish()

        logger.info(f"Done executing plan: {plan.plan_id}")
        logger.debug(f"Plan stats: (plan_cost={plan_stats.total_plan_cost}, plan_time={plan_stats.total_plan_time})")

        return output_records, plan_stats
