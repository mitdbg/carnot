import logging
from abc import ABC, abstractmethod

from carnot.core.elements.records import DataRecord
from carnot.core.models import PlanStats
from carnot.operators.scan import ContextScanOp, ScanPhysicalOp
from carnot.optimizer.plan import PhysicalPlan

logger = logging.getLogger(__name__)

class BaseExecutionStrategy:
    def __init__(self,
                 scan_start_idx: int = 0, 
                 max_workers: int | None = None,
                 batch_size: int | None = None,
                 num_samples: int | None = None,
                 verbose: bool = False,
                 progress: bool = True,
                 session_id: str | None = None,
                 progress_log_file: str | None = None,
                 *args,
                 **kwargs):
        self.scan_start_idx = scan_start_idx
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.verbose = verbose
        self.progress = progress
        self.session_id = session_id
        self.progress_log_file = progress_log_file


class ExecutionStrategy(BaseExecutionStrategy, ABC):
    """Base strategy for executing query plans. Defines how to execute a PhysicalPlan.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"Initialized ExecutionStrategy {self.__class__.__name__}")
        logger.debug(f"ExecutionStrategy initialized with config: {self.__dict__}")

    @abstractmethod
    def execute_plan(self, plan: PhysicalPlan) -> tuple[list[DataRecord], PlanStats]:
        """Execute a single plan according to strategy"""
        pass

    def _create_input_queues(self, plan: PhysicalPlan) -> dict[str, dict[str, list]]:
        """Initialize input queues for each operator in the plan."""
        input_queues = {f"{topo_idx}-{op.get_full_op_id()}": {} for topo_idx, op in enumerate(plan)}
        for topo_idx, op in enumerate(plan):
            full_op_id = op.get_full_op_id()
            unique_op_id = f"{topo_idx}-{full_op_id}"
            if isinstance(op, ScanPhysicalOp):
                scan_end_idx = (
                    len(op.datasource)
                    if self.num_samples is None
                    else min(self.scan_start_idx + self.num_samples, len(op.datasource))
                )
                input_queues[unique_op_id][f"source_{full_op_id}"] = [idx for idx in range(self.scan_start_idx, scan_end_idx)]
            elif isinstance(op, ContextScanOp):
                input_queues[unique_op_id][f"source_{full_op_id}"] = [None]
            else:
                for source_unique_full_op_id in plan.get_source_unique_full_op_ids(topo_idx, op):
                    input_queues[unique_op_id][source_unique_full_op_id] = []

        return input_queues
