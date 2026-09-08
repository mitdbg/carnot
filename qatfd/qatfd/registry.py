"""Name -> class registries for benchmarks and systems (CLI selection).

Adding a benchmark/system is one import + one dict entry here.
"""

from __future__ import annotations

from qatfd.benchmarks.base import Benchmark
from qatfd.benchmarks.browsecomp_plus import BrowseCompPlusBenchmark
from qatfd.benchmarks.financebench import FinanceBenchBenchmark
from qatfd.benchmarks.freshstack import FreshstackBenchmark
from qatfd.benchmarks.officeqa import OfficeQABenchmark
from qatfd.benchmarks.officeqa_synth import OfficeQASynthBenchmark
from qatfd.benchmarks.qampari import QampariBenchmark
from qatfd.benchmarks.trec_biogen import TrecBiogenBenchmark
from qatfd.config import BenchmarkConfig
from qatfd.systems.base import System
from qatfd.systems.codex import CodexSystem
from qatfd.systems.rag_llm import RAGLLMSystem
from qatfd.systems.search_agent import SearchAgentSystem

from skunk.config import AgentConfig, InferenceConfig

BENCHMARKS: dict[str, type[Benchmark]] = {
    OfficeQABenchmark.name: OfficeQABenchmark,
    OfficeQASynthBenchmark.name: OfficeQASynthBenchmark,
    BrowseCompPlusBenchmark.name: BrowseCompPlusBenchmark,
    TrecBiogenBenchmark.name: TrecBiogenBenchmark,
    FinanceBenchBenchmark.name: FinanceBenchBenchmark,
    QampariBenchmark.name: QampariBenchmark,
    FreshstackBenchmark.name: FreshstackBenchmark,
}

SYSTEMS: list[type[System]] = [RAGLLMSystem, SearchAgentSystem, CodexSystem]


def build_benchmark(config: BenchmarkConfig) -> Benchmark:
    name = config.name
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; available: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name](config)


def build_system(
    inference: InferenceConfig,
    retrieve_config: AgentConfig | None = None,
    compute_config: AgentConfig | None = None,
    codex_config: AgentConfig | None = None,
) -> System:
    if codex_config is not None:
        return CodexSystem(codex_config, inference)

    assert retrieve_config is not None and compute_config is not None
    name = retrieve_config.name
    if name == RAGLLMSystem.name:
        return RAGLLMSystem(retrieve_config, compute_config, inference)
    elif name == SearchAgentSystem.name:
        return SearchAgentSystem(retrieve_config, compute_config, inference)
    else:
        available = [system.name for system in SYSTEMS]
        raise KeyError(f"unknown system {name!r}; available: {available}")
