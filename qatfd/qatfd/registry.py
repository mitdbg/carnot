"""Name -> class registries for benchmarks and systems (CLI selection).

Adding a benchmark/system is one import + one dict entry here.
"""

from __future__ import annotations

from qatfd.benchmarks.base import Benchmark
from qatfd.benchmarks.browsecomp_plus import BrowseCompPlusBenchmark
from qatfd.benchmarks.financebench import FinanceBenchBenchmark
from qatfd.benchmarks.freshstack import FreshstackBenchmark
from qatfd.benchmarks.officeqa import OfficeQABenchmark
from qatfd.benchmarks.qampari import QampariBenchmark
from qatfd.benchmarks.trec_biogen import TrecBiogenBenchmark
from qatfd.config import BenchmarkConfig
from qatfd.systems.ablation_search_agent import AblationSearchAgentSystem
from qatfd.systems.base import System
from qatfd.systems.qatfd_search_agent import QATFDSearchAgentSystem
from qatfd.systems.rag_llm import RAGLLMSystem
from qatfd.systems.search_agent import SearchAgentSystem

from skunk.config import SystemConfig

BENCHMARKS: dict[str, type[Benchmark]] = {
    OfficeQABenchmark.name: OfficeQABenchmark,
    BrowseCompPlusBenchmark.name: BrowseCompPlusBenchmark,
    TrecBiogenBenchmark.name: TrecBiogenBenchmark,
    FinanceBenchBenchmark.name: FinanceBenchBenchmark,
    QampariBenchmark.name: QampariBenchmark,
    FreshstackBenchmark.name: FreshstackBenchmark,
}

SYSTEMS: dict[str, type[System]] = {
    RAGLLMSystem.name: RAGLLMSystem,
    SearchAgentSystem.name: SearchAgentSystem,
    QATFDSearchAgentSystem.name: QATFDSearchAgentSystem,
    AblationSearchAgentSystem.name: AblationSearchAgentSystem,
}


def build_benchmark(config: BenchmarkConfig) -> Benchmark:
    name = config.name
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; available: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name](config)


def build_system(config: SystemConfig) -> System:
    name = config.name
    if name not in SYSTEMS:
        raise KeyError(f"unknown system {name!r}; available: {sorted(SYSTEMS)}")
    return SYSTEMS[name](config)
