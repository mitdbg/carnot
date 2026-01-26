import logging

from carnot.config import QueryProcessorConfig
from carnot.constants import Cardinality, Model
from carnot.core.data.context import Context, TextFileContext
from carnot.core.data.iter_dataset import (
    AudioFileDataset,
    HTMLFileDataset,
    ImageFileDataset,
    IterDataset,
    MemoryDataset,
    PDFFileDataset,
    TextFileDataset,
    XLSFileDataset,
)
from carnot.core.lib.schemas import AudioBase64, AudioFilepath, ImageBase64, ImageFilepath, ImageURL
from carnot.data.dataset import Dataset
from carnot.data.logical_dataset import LogicalDataset
from carnot.execution.execution import Execution
from carnot.policy import (
    MaxQuality,
    MaxQualityAtFixedCost,
    MaxQualityAtFixedTime,
    MinCost,
    MinCostAtFixedQuality,
    MinTime,
    MinTimeAtFixedQuality,
    PlanCost,
    Policy,
)

# Initialize the root logger
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    # constants
    "Cardinality",
    "Model",
    # core
    "Context",
    "TextFileContext",
    "IterDataset",
    "AudioFileDataset",
    "MemoryDataset",
    "HTMLFileDataset",
    "ImageFileDataset",
    "PDFFileDataset",
    "TextFileDataset",
    "XLSFileDataset",
    # data
    "Dataset",
    "LogicalDataset",
    # execution
    "Execution",
    # schemas
    "AudioBase64",
    "AudioFilepath",
    "ImageBase64",
    "ImageFilepath",
    "ImageURL",
    # policy
    "MaxQuality",
    "MaxQualityAtFixedCost",
    "MaxQualityAtFixedTime",
    "MinCost",
    "MinCostAtFixedQuality",
    "MinTime",
    "MinTimeAtFixedQuality",
    "PlanCost",
    "Policy",
    # query
    "QueryProcessorConfig",
]
