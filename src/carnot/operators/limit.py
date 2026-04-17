from __future__ import annotations

import time
from collections.abc import Callable

from carnot.core.models import OperatorStats
from carnot.data.dataset import Dataset
from carnot.operators.physical import PhysicalOperator


class LimitOperator(PhysicalOperator):
    """Limit operator — truncates a dataset to the first *n* items.

    This is a purely deterministic operator with no LLM involvement.

    Representation invariant:
        - ``n >= 0``.

    Abstraction function:
        An instance of this class is a callable that, given a dataset, returns a new dataset
        containing at most ``n`` items (the first *n* in order).
    """
    def __init__(self, n: int, dataset_id: str, logical_op_id: str | None = None, logical_op_class_name: str | None = None):
        super().__init__(logical_op_id=logical_op_id, logical_op_class_name=logical_op_class_name)
        self.n = n
        self.dataset_id = dataset_id

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "n": self.n,
            "dataset_id": self.dataset_id,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "n": self.n,
            "dataset_id": self.dataset_id,
            **op_params,
        }

        return op_params

    def __call__(self, dataset_id: str, input_datasets: dict[str, Dataset], on_item_complete: Callable[[], None] | None = None) -> tuple[dict[str, Dataset], OperatorStats]:
        """Truncate the input dataset to the first *n* items.

        Requires:
            - *dataset_id* is a key in *input_datasets*.

        Returns:
            A tuple ``(output_datasets, stats)`` where *output_datasets*
            is a new ``dict[str, Dataset]`` with an additional entry
            keyed by ``self.dataset_id`` containing at most
            ``self.n`` items, and *stats* is an :class:`OperatorStats`
            with an empty ``llm_calls`` list (no LLM involvement).

        Raises:
            KeyError: If *dataset_id* is not in *input_datasets*.
        """
        op_start = time.perf_counter()

        # retrieve the input dataset
        input_dataset = input_datasets[dataset_id]
        
        # apply the limit operation to the dataset items
        results = input_dataset.items[:self.n]

        # create new dataset and return it with the input datasets
        output_dataset = Dataset(name=self.dataset_id, annotation=f"Limit operator output for n: {self.n}", items=results)
        output_datasets = {**input_datasets, output_dataset.name: output_dataset}

        op_stats = OperatorStats(
            operator_name="Limit",
            operator_id=self.dataset_id,
            wall_clock_secs=time.perf_counter() - op_start,
            llm_calls=[],
            items_in=len(input_dataset.items),
            items_out=len(results),
        )

        return output_datasets, op_stats
