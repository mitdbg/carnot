from __future__ import annotations

from carnot.operators.physical import PhysicalOperator


class ScanOp(PhysicalOperator):
    """Physical scan operator — provides token-count metadata without materializing the full dataset.

    Unlike Palimpzest's scan, this operator does NOT eagerly read data.
    It exists solely to give the cost model a base-case estimate of the
    total tokens available in a dataset.  Actual data reading is deferred
    to execution time (handled by the existing Dataset / DataItem /
    TieredStorageManager pipeline).

    ``ScanOp`` has no ``__call__`` override — it is a metadata-only
    operator and should never be invoked during plan execution.

    Representation invariant:
        - ``num_items >= 0``.
        - ``est_tokens_per_item >= 0``.

    Abstraction function:
        Represents a zero-cost, zero-latency read of dataset
        ``dataset_id`` whose estimated total token footprint is
        ``num_items * est_tokens_per_item``.
    """

    def __init__(
        self,
        dataset_id: str,
        num_items: int,
        est_tokens_per_item: float,
        logical_op_id: str | None = None,
        logical_op_class_name: str | None = None,
    ):
        super().__init__(logical_op_id=logical_op_id, logical_op_class_name=logical_op_class_name)
        self.dataset_id = dataset_id
        self.num_items = num_items
        self.est_tokens_per_item = est_tokens_per_item

    @property
    def est_total_tokens(self) -> float:
        """Estimated total token count for the entire dataset.

        Requires:
            None.

        Returns:
            ``num_items * est_tokens_per_item``.

        Raises:
            None.
        """
        return self.num_items * self.est_tokens_per_item

    def get_id_params(self) -> dict:
        """Return parameters used to compute the operator id.

        Requires:
            None.

        Returns:
            A dict with ``dataset_id``.

        Raises:
            None.
        """
        return {"dataset_id": self.dataset_id}

    def get_op_params(self) -> dict:
        """Return all parameters needed to reconstruct this operator.

        Requires:
            None.

        Returns:
            A dict containing the base-class params merged with scan-
            specific params.

        Raises:
            None.
        """
        params = super().get_op_params()
        params = {
            **params,
            "dataset_id": self.dataset_id,
            "num_items": self.num_items,
            "est_tokens_per_item": self.est_tokens_per_item,
        }
        return params
