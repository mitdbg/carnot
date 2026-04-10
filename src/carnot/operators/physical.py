from __future__ import annotations

import json

from carnot.core.models import OperatorStats
from carnot.data.dataset import Dataset
from carnot.utils.hash_helpers import hash_for_id


class PhysicalOperator:
    """
    All implemented physical operators should inherit from this class.
    In order for the Optimizer to consider using a physical operator for a
    given logical operation, the user must also write an ImplementationRule.
    """

    def __init__(
        self,
        logical_op_id: str | None = None,
        logical_op_class_name: str | None = None,
    ) -> None:
        self.logical_op_id = logical_op_id
        self.logical_op_class_name = logical_op_class_name
        self.op_id = None

        # sets __hash__() for each child Operator to be the base class' __hash__() method;
        # by default, if a subclass defines __eq__() but not __hash__() Python will set that
        # class' __hash__ to None
        self.__class__.__hash__ = PhysicalOperator.__hash__

    def __str__(self):
        return f"{self.op_name()}\n"

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.get_full_op_id() == other.get_full_op_id()

    def copy(self) -> PhysicalOperator:
        return self.__class__(**self.get_op_params())

    def op_name(self) -> str:
        """Name of the physical operator."""
        return str(self.__class__.__name__)

    def get_id_params(self) -> dict:
        """
        Returns a dictionary mapping of physical operator parameters which are relevant
        for computing the physical operator id.

        NOTE: Should be overriden by subclasses to include class-specific parameters.
        NOTE: input_schema and output_schema are not included in the id params by default,
              because they may depend on the order of operations chosen by the Optimizer.
              This is particularly true for convert operations, where the output schema
              is now the union of the input and output schemas of the logical operator.
        """
        return {}

    def get_op_params(self) -> dict:
        """
        Returns a dictionary mapping of physical operator parameters which may be used to
        create a copy of this physical operation.

        NOTE: Should be overriden by subclasses to include class-specific parameters.
        """
        return {
            "logical_op_id": self.logical_op_id,
            "logical_op_class_name": self.logical_op_class_name,
        }

    def get_op_id(self):
        """
        NOTE: We do not call this in the __init__() method as subclasses may set parameters
              returned by self.get_id_params() after they call to super().__init__().

        NOTE: This is NOT a universal ID.

        Two different PhysicalOperator instances with the identical returned values
        from the call to self.get_id_params() will have equivalent op_ids.
        """
        # return self.op_id if we've computed it before
        if self.op_id is not None:
            return self.op_id

        # get op name and op parameters which are relevant for computing the id
        op_name = self.op_name()
        id_params = self.get_id_params()
        id_params = {k: str(v) for k, v in id_params.items()}

        # compute, set, and return the op_id
        hash_str = json.dumps({"op_name": op_name, **id_params}, sort_keys=True)
        self.op_id = hash_for_id(hash_str)

        return self.op_id

    def get_logical_op_id(self) -> str:
        return self.logical_op_id

    def get_full_op_id(self):
        return f"{self.get_logical_op_id()}-{self.get_op_id()}"

    def __hash__(self):
        return int(hash_for_id(self.get_full_op_id()), 16)

    def __call__(self, dataset_id: str, input_datasets: dict[str, Dataset]) -> tuple[dict[str, Dataset], OperatorStats]:
        raise NotImplementedError("Calling __call__ from abstract method")
