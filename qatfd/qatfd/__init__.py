"""qatfd — evaluation harness running a matrix of systems x QA benchmarks.

Two subclassable abstractions: `Benchmark` (loads questions, owns the corpus/index,
scores answers) and `System` (answers a question using the benchmark's resources).
A shared `runner` pairs them, runs questions in parallel, and writes a report.
"""