import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.progress import Progress as RichProgress

from carnot.operators.physical import PhysicalOperator
from carnot.optimizer.plan import PhysicalPlan
from carnot.utils.progress_events import (
    ProgressEvent,
    ProgressLogger,
    ProgressStatus,
)


@dataclass
class ProgressStats:
    """Statistics tracked for progress reporting"""
    start_time: float = 0.0
    total_cost: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    current_operation: str = ""
    memory_usage_mb: float = 0.0
    recent_text: str = ""

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0

# NOTE: right now we only need to support single plan execution; in a multi-plan setting, we will
#       need to modify the semantics of the progress manager to support multiple plans
class ProgressManager(ABC):
    """Abstract base class for progress managers for plan execution"""

    def __init__(self, plan: PhysicalPlan, num_samples: int | None = None):
        """
        Initialize the progress manager for the given plan. This function takes in a plan,
        the number of samples to process (if specified).

        If `num_samples` is None, then the entire Dataset will be scanned.

        For each operator which is not an `AggregateOp` or `LimitScanOp`, we set its task `total`
        to the number of inputs to be processed by the plan. As intermediate operators process
        their inputs, the ProgressManager will update the `total` for their downstream operators.
        """
        # initialize progress object
        self.progress = RichProgress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            #TextColumn("[green]Success: {task.fields[success]}"),
            #TextColumn("[red]Failed: {task.fields[failed]}"),
            #TextColumn("[cyan]Mem: {task.fields[memory]:.1f}MB"),
            TextColumn("[green]Cost: ${task.fields[cost]:.4f}"),
            TextColumn("\n[white]{task.fields[recent]}"),  # Recent text on new line
            refresh_per_second=10,
            expand=True,   # Use full width
        )

        # initialize mapping from unique_full_op_id --> ProgressStats
        self.unique_full_op_id_to_stats: dict[str, ProgressStats] = {}

        # initialize mapping from unique_full_op_id --> task
        self.unique_full_op_id_to_task = {}

        # initialize start time
        self.start_time = None

        # TODO: store plan and use its methods within incr()
        # create mapping from unique_full_op_id --> input unique_full_op_ids
        self.unique_full_op_id_to_input_unique_full_op_ids: dict[str, list[str]] = {}
        for topo_idx, op in enumerate(plan):
            unique_full_op_id = f"{topo_idx}-{op.get_full_op_id()}"
            input_unique_full_op_ids = plan.get_source_unique_full_op_ids(topo_idx, op)
            self.unique_full_op_id_to_input_unique_full_op_ids[unique_full_op_id] = input_unique_full_op_ids

        # create mapping from unique_full_op_id --> next_op
        self.unique_full_op_id_to_next_op_and_id: dict[str, tuple[PhysicalOperator, str]] = {}
        for topo_idx, op in enumerate(plan):
            unique_full_op_id = f"{topo_idx}-{op.get_full_op_id()}"
            next_op, next_unique_full_op_id = plan.get_next_unique_full_op_and_id(topo_idx, op)
            self.unique_full_op_id_to_next_op_and_id[unique_full_op_id] = (next_op, next_unique_full_op_id)

        # add a task to the progress manager for each operator in the plan
        est_total_outputs, _ = plan.get_est_total_outputs(num_samples)
        for topo_idx, op in enumerate(plan):
            # get the op id and a short string representation of the op; (str(op) is too long)
            op_str = f"{op.op_name()} ({op.get_op_id()})"
            unique_full_op_id = f"{topo_idx}-{op.get_full_op_id()}"
            self.add_task(unique_full_op_id, op_str, est_total_outputs[unique_full_op_id])

    def get_task_total(self, unique_full_op_id: str) -> int:
        """Return the current total value for the given task."""
        task = self.unique_full_op_id_to_task[unique_full_op_id]
        return self.progress._tasks[task].total

    def get_task_description(self, unique_full_op_id: str) -> str:
        """Return the current description for the given task."""
        task = self.unique_full_op_id_to_task[unique_full_op_id]
        return self.progress._tasks[task].description

    @abstractmethod
    def add_task(self, unique_full_op_id: str, op_str: str, total: int):
        """Initialize progress tracking for operator execution with total items"""
        pass

    @abstractmethod
    def start(self):
        """Start the progress bar(s)"""
        pass

    @abstractmethod
    def incr(self, unique_full_op_id: str, num_inputs: int = 1, num_outputs: int = 1, display_text: str | None = None, **kwargs):
        """
        Advance the progress bar for the given operator. Modify the downstream operators'
        progress bar `total` to reflect the number of outputs produced by this operator.

        NOTE: `num_outputs` specifies how many outputs were generated by the operator when processing
        the `num_inputs` inputs for which `incr()` was called. E.g. a filter which filters one input record
        will advance its progress bar by 1, but the next operator will now have 1 fewer inputs to process.
        Alternatively, a convert which generates 3 `num_outputs` for 2 `num_inputs` will increase the inputs
        for the next operator by `delta = num_outputs - num_inputs = 3 - 2 = 1`.
        """
        pass

    @abstractmethod
    def finish(self):
        """Clean up and finalize progress tracking"""
        pass


class MockProgressManager(ProgressManager):
    """Mock progress manager for testing purposes"""

    def __init__(self, plan: PhysicalPlan, num_samples: int | None = None):
        pass

    def add_task(self, unique_full_op_id: str, op_str: str, total: int):
        pass

    def start(self):
        pass

    def incr(self, unique_full_op_id: str, num_inputs: int = 1, num_outputs: int = 1, display_text: str | None = None, **kwargs):
        pass

    def finish(self):
        pass

class PZProgressManager(ProgressManager):
    """Progress manager for command line interface using rich"""
    
    def __init__(self, plan: PhysicalPlan, num_samples: int | None = None,
                 session_id: str | None = None, progress_log_file: str | None = None,
                 level: str = "outer"):
        self.console = Console()
        self.session_id = session_id
        self.level = level
        # Initialize progress logger
        self.progress_logger = ProgressLogger(progress_log_file) if progress_log_file else None
        
        super().__init__(plan, num_samples)

    def add_task(self, unique_full_op_id: str, op_str: str, total: int):
        """Add a new task to the progress bar"""
        task = self.progress.add_task(
            f"[blue]{op_str}", 
            total=total,
            cost=0.0,
            success=0,
            failed=0,
            memory=0.0,
            recent="",
        )

        # store the mapping of operator ID to task ID
        self.unique_full_op_id_to_task[unique_full_op_id] = task

        # initialize the stats for this operation
        self.unique_full_op_id_to_stats[unique_full_op_id] = ProgressStats(start_time=time.time())
        
        # Log started event
        if self.progress_logger and self.session_id:
            event = ProgressEvent(
                timestamp=datetime.now(timezone.utc).isoformat() + 'Z',
                session_id=self.session_id,
                level=self.level,
                operator_id=unique_full_op_id,
                operator_name=op_str,
                status=ProgressStatus.STARTED,
                current=0,
                total=total,
                percentage=0.0,
                cost=0.0,
                message="",
            )
            self.progress_logger.log_event(event)

    def start(self):
        # print a newline before starting to separate from previous output
        print()

        # set start time
        self.start_time = time.time()

        # start progress bar
        self.progress.start()

    def incr(self, unique_full_op_id: str, num_inputs: int = 1, num_outputs: int = 1, display_text: str | None = None, **kwargs):
        # get the task for the given operation
        task = self.unique_full_op_id_to_task.get(unique_full_op_id)

        # update statistics with any additional keyword arguments
        if kwargs != {}:
            self.update_stats(unique_full_op_id, **kwargs)

        # update progress bar and recent text in one update
        if display_text is not None:
            self.unique_full_op_id_to_stats[unique_full_op_id].recent_text = display_text

        # update the downstream operators' progress bar total for any operator which is not an AggregateOp or LimitScanOp
        delta = num_outputs - num_inputs
        if delta != 0:
            next_op, next_unique_full_op_id = self.unique_full_op_id_to_next_op_and_id[unique_full_op_id]
            while next_op is not None:
                next_task = self.unique_full_op_id_to_task[next_unique_full_op_id]
                self.progress.update(next_task, total=self.get_task_total(next_unique_full_op_id) + delta)

                # move to the next operator in the plan
                next_op, next_unique_full_op_id = self.unique_full_op_id_to_next_op_and_id[next_unique_full_op_id]

        # advance the progress bar for this task
        self.progress.update(
            task,
            advance=num_inputs,
            description=f"[bold blue]{self.get_task_description(unique_full_op_id)}",
            cost=self.unique_full_op_id_to_stats[unique_full_op_id].total_cost,
            success=self.unique_full_op_id_to_stats[unique_full_op_id].success_count,
            failed=self.unique_full_op_id_to_stats[unique_full_op_id].failure_count,
            memory=get_memory_usage(),
            recent=f"{self.unique_full_op_id_to_stats[unique_full_op_id].recent_text}" if display_text is not None else "",
            refresh=True,
        )
        
        # Log progress event
        if self.progress_logger and self.session_id:
            task_obj = self.progress._tasks[task]
            current = task_obj.completed
            total = task_obj.total
            percentage = (current / total * 100) if total > 0 else 0
            
            event = ProgressEvent(
                timestamp=datetime.now(timezone.utc).isoformat() + 'Z',
                session_id=self.session_id,
                level=self.level,
                operator_id=unique_full_op_id,
                operator_name=self.get_task_description(unique_full_op_id).replace("[bold blue]", ""),
                status=ProgressStatus.PROGRESS,
                current=int(current),
                total=int(total),
                percentage=percentage,
                cost=self.unique_full_op_id_to_stats[unique_full_op_id].total_cost,
                message=display_text or "",
            )
            self.progress_logger.log_event(event)

    def finish(self):
        # Log completion events for all operators
        if self.progress_logger and self.session_id:
            for unique_full_op_id, task_id in self.unique_full_op_id_to_task.items():
                task_obj = self.progress._tasks[task_id]
                event = ProgressEvent(
                    timestamp=datetime.now(timezone.utc).isoformat() + 'Z',
                    session_id=self.session_id,
                    level=self.level,
                    operator_id=unique_full_op_id,
                    operator_name=self.get_task_description(unique_full_op_id).replace("[bold blue]", ""),
                    status=ProgressStatus.COMPLETED,
                    current=int(task_obj.total),
                    total=int(task_obj.total),
                    percentage=100.0,
                    cost=self.unique_full_op_id_to_stats[unique_full_op_id].total_cost,
                    message="",
                )
                self.progress_logger.log_event(event)
        
        self.progress.stop()
        if self.progress_logger:
            self.progress_logger.close()

        # compute total cost, success, and failure
        total_cost = sum(stats.total_cost for stats in self.unique_full_op_id_to_stats.values())
        # success_count = sum(stats.success_count for stats in self.unique_full_op_id_to_stats.values())
        # failure_count = sum(stats.failure_count for stats in self.unique_full_op_id_to_stats.values())

        # Print final stats on new lines after progress display
        print(f"Total time: {time.time() - self.start_time:.2f}s")
        print(f"Total cost: ${total_cost:.4f}")
        # print(f"Success rate: {success_count}/{success_count + failure_count}")

    def update_stats(self, unique_full_op_id: str, **kwargs):
        """Update progress statistics"""
        for key, value in kwargs.items():
            if hasattr(self.unique_full_op_id_to_stats[unique_full_op_id], key):
                if key != "total_cost":
                    setattr(self.unique_full_op_id_to_stats[unique_full_op_id], key, value)
                else:
                    self.unique_full_op_id_to_stats[unique_full_op_id].total_cost += value
        self.unique_full_op_id_to_stats[unique_full_op_id].memory_usage_mb = get_memory_usage()


def create_progress_manager(
    plan: PhysicalPlan,
    num_samples: int | None = None,
    progress: bool = True,
    session_id: str | None = None,
    progress_log_file: str | None = None,
    level: str = "outer",
) -> ProgressManager:
    """Factory function to create appropriate progress manager based on environment"""
    # If we have a log file, we need a real progress manager even if console progress is disabled
    if not progress and not progress_log_file:
        return MockProgressManager(plan, num_samples)

    return PZProgressManager(plan, num_samples, session_id, progress_log_file, level)
