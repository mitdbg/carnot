
class BaseCostModel:
    """
    This base class contains the interface/abstraction that every CostModel must implement
    in order to work with the Optimizer. In brief, the Optimizer expects the CostModel to
    make a prediction about the runtime, cost, and quality of a physical operator.
    """
    def __init__(self):
        """
        CostModel constructor; the arguments for individual CostModels may vary depending
        on the assumptions they make about the prevalance of historical execution data
        and online vs. batch execution settings.
        """
        pass

    def __call__(self, plan):
        """
        The interface exposed by the CostModel to the Optimizer. Subclasses may require
        additional arguments in order to make their predictions.
        """
        raise NotImplementedError("Calling __call__ from abstract method")
