# Carnot Test Suite

> **Status**: Living document. Last updated: 2026-03-02.
>
> This document describes the design principles, organization, and conventions
> of the Carnot test suite. It should be updated as the suite evolves.

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [Contract-Based Testing Workflow](#contract-based-testing-workflow)
3. [Directory Layout](#directory-layout)
4. [Test Tiers](#test-tiers)
5. [LLM & Mock Strategy](#llm--mock-strategy)
6. [Storage Layer Mocking](#storage-layer-mocking)
7. [Datasets & Test Data](#datasets--test-data)
8. [Docstring & Specification Conventions](#docstring--specification-conventions)
9. [Fixture Organization](#fixture-organization)
10. [Pytest Markers & Configuration](#pytest-markers--configuration)
11. [Writing a New Test — Checklist](#writing-a-new-test--checklist)
12. [Current Coverage Map](#current-coverage-map)
13. [Tier 4 — Docker Compose Integration Tests](#tier-4--docker-compose-integration-tests)
14. [Roadmap](#roadmap)

---

## Design Principles

These principles, in priority order, guide every decision about what and how
we test.

1. **Targeted units, few high-quality E2E tests.**
   Every semantic operator, data structure, and utility gets a focused unit
   test that exercises its contract in isolation. System-level correctness is
   validated by a *small* number of end-to-end tests that cover the full
   query lifecycle (plan → optimize → execute → answer). We prefer one
   well-designed E2E test over twenty overlapping ones.

2. **Dual-mode execution: mocked and live.**
   Any test that calls an LLM or an external service must be runnable in two
   modes, controlled by the environment variable `RUN_TESTS_WITH_LLM`:
   - **Mocked (default):** deterministic, fast, CI-friendly. LLM calls
     return canned responses via `unittest.mock` or lightweight fakes.
   - **Live:** exercises the real LLM path. Run manually or in a nightly CI
     job where API keys are available.

   Similarly, storage-layer tests run against real local/S3 backends when
   testing the storage layer itself, but other subsystems use an
   in-memory mock storage to avoid I/O overhead.

3. **Retry tolerance for non-deterministic assertions.**
   When a test asserts on LLM output (live mode only), it may use
   `pytest-rerunfailures` or a manual retry wrapper to re-run **once** on
   failure. If it fails twice, the test is genuinely broken. Mocked tests
   must *never* require retries.

4. **Datasets designed for high-confidence LLM assertions.**
   Test datasets should present tasks where a capable LLM is overwhelmingly
   likely to succeed (e.g., classifying "giraffe" as a mammal, extracting the
   title from the first page of a paper). If we cannot write a deterministic
   assertion against the expected output, we use accuracy thresholds
   (e.g., `≥ 0.8`) and document why.

5. **Specification-first development.**
   Docstrings are the *specification*. Before writing a test, ensure the
   function or class under test has a docstring that declares its contract
   using the conventions in [Docstring & Specification Conventions](#docstring--specification-conventions).
   Tests then verify that contract. If the docstring is missing or vague,
   improve it first.

6. **Functional design where practical.**
   Prefer pure functions, immutable data, and explicit inputs/outputs.
   Side-effects (file I/O, LLM calls) should be isolated behind narrow
   interfaces so that the bulk of logic is testable with plain values.

---

## Contract-Based Testing Workflow

Every test in the Carnot suite must verify the **documented contract** of
the class or function under test — not its implementation internals.  The
contract is defined by the docstring, following the conventions in
[Docstring & Specification Conventions](#docstring--specification-conventions).

### Why this matters

A test that depends on an implementation detail will break when the
implementation is refactored, even though the observable behavior is
unchanged.  Worse, such a test may *pass* when the contract is violated
(because the test was written against the wrong invariant).

**Canonical example:** `SemTopKOperator`'s constructor internally looks
up `index_name` in a private `index_map` dict whose keys happen to be
`{"chroma", "faiss", "hierarchical", "flat"}`.  A test that used
`index_name="my_index"` was testing the Execution helper, not the
operator, but it failed because it unknowingly relied on the internal
dispatch table.  The operator's docstring said nothing about valid
`index_name` values.  The fix has two parts: (1) the docstring must
document preconditions, and (2) the test must use only values allowed by
those preconditions.

### The workflow

Follow these steps **in order** when writing or modifying a test.

1. **Read the docstring.**  Open the source file and read the docstring of
   the function / class / method under test.  Pay attention to:
   - **Requires** (preconditions): what must the caller provide?
   - **Returns** (postconditions): what can the caller rely on?
   - **Raises**: what exceptions are documented?
   - **Representation invariant** (RI): for classes, what properties
     always hold on internal state?
   - **Abstraction function** (AF): how does internal state map to the
     abstract value?

2. **If the docstring is incomplete, improve it first.**  If writing the
   test requires you to inspect the implementation to understand behavior,
   that behavior is under-documented.  Update the docstring *before*
   writing the test.  Common gaps:
   - Missing `Requires` clause (what values are valid?).
   - Missing `Returns` clause (what structure does the return value have?).
   - Missing RI for classes with internal state the tests need to verify.

3. **Write assertions against the contract only.**
   - ✅ Assert on documented return values, postconditions, and RIs.
   - ✅ Assert on documented exceptions (`Raises`).
   - ✅ Assert on public attributes listed in the class docstring.
   - ❌ Do **not** access private attributes (`_foo`) unless they are
     explicitly listed in the RI or class docstring.
   - ❌ Do **not** call private methods (`_bar()`) unless they are
     documented in the class docstring as part of the testable contract.

4. **If you must test a private method:**  Sometimes a private method
   implements sufficiently complex logic that it deserves a dedicated test
   (e.g., `_get_op_from_plan_dict` in `Execution`).  In that case:
   - Add a comprehensive docstring to the private method, including
     `Requires`, `Returns`, and `Raises`.
   - Note in the test file docstring that the test covers an internal
     method and cite the docstring.

5. **Verify the test locally.**  After writing or modifying:
   ```bash
   pytest tests/pytest/<test_file>.py -v
   ruff check tests/pytest/<test_file>.py
   ```

### Auditing existing tests

When auditing an existing test against its contract, ask for each
assertion:

> *Can I justify this assertion using **only** the docstring of the
> function / class under test?*

- If **yes**: the test is correct.
- If **no, but the behavior is intentional**: update the docstring to
  make the contract explicit, then the test becomes correct.
- If **no, and the test tests an implementation detail that should be
  free to change**: rewrite the test to use only the public API and
  documented postconditions.

### Quick-reference decision tree

```
Want to assert on X?
  │
  ├─ Is X a documented postcondition (Returns / RI / Raises)?
  │    └─ YES → ✅ assert on X.
  │
  ├─ Is X a public attribute / method not mentioned in docstring?
  │    └─ Add it to the docstring → ✅ then assert on X.
  │
  ├─ Is X a private attribute / method (_foo)?
  │    ├─ Is it listed in the class RI or docstring?
  │    │    └─ YES → ✅ assert on X.
  │    └─ NO → ❌ rewrite test to observe X via public API,
  │              or add X to the docstring if it's part of the contract.
  │
  └─ Is X an implementation detail with no user-visible effect?
       └─ ❌ do not test it.
```

---

## Directory Layout

```
tests/pytest/
├── TEST_SUITE.md            # ← this document
├── conftest.py              # root conftest; registers fixture plugins
├── fixtures/
│   ├── config.py            # llm_config, model IDs
│   ├── data.py              # raw data loaders (CSV → DataFrame, file → str)
│   ├── datasets.py          # Dataset objects assembled from raw data
│   ├── mocks.py             # (NEW) mock LLM, mock storage, fake operators
│   └── storage.py           # (NEW) storage backend fixtures (local, S3, in-memory)
├── helpers/
│   ├── assertions.py        # (NEW) reusable assertion helpers (e.g., agent_did_not_hit_max_steps)
│   └── mock_utils.py        # (NEW) msg_text() helper for inspecting litellm-style messages
├── data/
│   ├── movie-reviews/       # Rotten Tomatoes CSVs (movies + reviews)
│   ├── papers/              # plain-text research papers (paper1.txt … paper3.txt)
│   ├── emails/              # Enron email corpus subset
│   └── quest/               # QUeST benchmark data
│
│  ── Tier 3 live LLM tests ───────────────────────────────────
├── test_sem_filter_operator.py
├── test_sem_map_operator.py
├── test_sem_join_operator.py
├── test_sem_agg_operator.py
├── test_sem_topk_operator.py
├── test_sem_flat_map_operator.py
├── test_sem_groupby_operator.py
├── test_code_operator.py
├── test_logical_planning.py
├── test_conversation_integration.py
├── test_conversation_planning.py
│
│  ── Tier 1 unit tests ──────────────────────────────────────
├── test_data_model.py            # (NEW) Dataset, DataItem unit tests
├── test_storage_backends.py      # (NEW) LocalStorageBackend, S3StorageBackend, TieredStorageManager
├── test_index.py                 # (NEW) FlatFileIndex, HierarchicalFileIndex, CarnotIndex
├── test_execution_unit.py        # (NEW) _get_op_from_plan_dict, _get_ops_in_topological_order
│
│  ── Tier 2 mocked operator tests ───────────────────────────
├── test_sem_filter_mocked.py     # (NEW) SemFilterOperator with mocked LLM (7 tests)
├── test_sem_map_mocked.py        # (NEW) SemMapOperator with mocked LLM (7 tests)
├── test_sem_agg_mocked.py        # (NEW) SemAggOperator with mocked LLM (5 tests)
├── test_sem_join_mocked.py       # (NEW) SemJoinOperator with mocked LLM (7 tests)
├── test_sem_flat_map_mocked.py   # (NEW) SemFlatMapOperator with mocked LLM (7 tests)
├── test_sem_groupby_mocked.py    # (NEW) SemGroupByOperator with mocked LLM (7 tests)
├── test_sem_topk_mocked.py       # (NEW) SemTopKOperator with mocked index (6 tests)
├── test_planner_mocked.py        # (NEW) Planner with mocked LLM (12 tests)
│
│  ── End-to-end tests ────────────────────────────────────────
├── test_e2e_sem_topk.py
├── test_e2e_query_lifecycle.py   # (NEW) plan → execute → answer round-trips
│
│  ── Tier 4: Docker Compose integration tests ────────────────
├── integration/
│   ├── conftest.py               # (NEW) docker-compose session fixtures, cleanup
│   ├── test_health.py            # (NEW) backend health + DB connectivity smoke tests
│   ├── test_files_api.py         # (NEW) file upload, browse, delete via REST
│   ├── test_datasets_api.py      # (NEW) dataset CRUD via REST (postgres-backed)
│   ├── test_conversations_api.py # (NEW) conversation + message CRUD via REST
│   └── test_settings_api.py      # (NEW) user settings CRUD via REST
└── test_e2e_query_lifecycle.py
```

### Key changes from the current layout

| Change | Rationale |
|--------|-----------|
| Add `fixtures/mocks.py` | Centralizes mock LLM responses and fake operators for dual-mode tests. |
| Add `fixtures/storage.py` | Provides `tmp_path`-based local backends and an in-memory backend for non-storage tests. |
| Add `helpers/assertions.py` | Eliminates duplicated `assert_agent_did_not_hit_max_steps` across files. |
| Add `test_data_model.py` | `Dataset` and `DataItem` have nontrivial semantics (lazy materialization, URI migration) that are currently untested. |
| Add `test_storage_backends.py` | The storage layer is a critical new subsystem that needs dedicated coverage. |
| Add `test_index.py` | `FlatFileIndex` and `HierarchicalFileIndex` have complex logic (embedding pre-filter, LLM routing, B-tree build) with no dedicated tests. |
| Add `test_e2e_query_lifecycle.py` | A single file for curated end-to-end scenarios, replacing the pattern of scattering E2E logic across operator test files. |

---

## Test Tiers

We organize tests into three tiers. Each tier has different speed, cost, and
reliability characteristics.

### Tier 1 — Pure Unit Tests (no LLM, no I/O)

**Run time:** milliseconds per test.
**When:** every `git push`, local pre-commit, CI on every PR.

These tests exercise deterministic logic: data model construction, plan
serialization, conversation state management, operator parameter validation,
hash computation, storage backend operations (against `tmp_path`), etc.

```python
# Example: verifying Dataset construction and lazy materialization
def test_dataset_from_dicts_is_materialized():
    ds = Dataset(name="T", items=[{"a": 1}])
    assert ds._is_materialized is True
    assert ds.items == [{"a": 1}]
```

### Tier 2 — Mocked LLM Tests

**Run time:** milliseconds per test.
**When:** every CI run (default; `RUN_TESTS_WITH_LLM` unset).

Operator tests with a mock LLM that returns canned, deterministic responses.
This validates the *wiring* — correct prompt construction, output parsing,
dataset threading — without incurring API cost or non-determinism.

```python
# Example: mocked sem_filter
def test_sem_filter_mocked(mock_llm_config, simple_animal_dataset):
    """SemFilterOperator correctly parses a mocked LLM 'yes'/'no' response."""
    op = SemFilterOperator(task="is a mammal", ...)
    result = op("Animals", {"Animals": simple_animal_dataset})
    assert "output" in result
```

### Tier 3 — Live LLM Tests

**Run time:** seconds per test (network-bound).
**When:** on-demand or nightly CI. Requires `RUN_TESTS_WITH_LLM=1` +
valid API key.

These are the existing operator tests (e.g., `test_sem_filter_operator_basic`)
that send real prompts to an LLM and assert on the returned content. They use
accuracy thresholds and are allowed a single retry on failure.

```python
@pytest.mark.llm
@pytest.mark.flaky(reruns=1)
def test_sem_filter_operator_basic(test_model_id, llm_config):
    ...
    assert {"animal": "giraffe"} in output_dataset.items
```

### Tier 4 — Docker Compose Integration Tests

**Run time:** seconds–minutes per test (includes container startup on first
run of session).
**When:** on-demand or pre-release CI. Requires Docker and
`RUN_INTEGRATION_TESTS=1` (or `-m integration`). Skipped by default.

These tests spin up the full `docker-compose.yaml` +
`docker-compose.local.yaml` stack (postgres, backend, frontend) from the
current source tree and exercise the **backend REST API** over HTTP. They
validate that:
- The Docker images build correctly from the latest source.
- Alembic migrations run and create the expected schema.
- API routes for files, datasets, conversations, and settings function
  correctly against a real Postgres instance.
- The local file service correctly reads/writes to the mounted data volume.

Because Auth0 is not available in the local test environment, these tests
bypass JWT auth. The backend in `LOCAL_ENV=true` mode can be configured
to accept a simple header or skip auth entirely for integration testing
(see [Auth Bypass Strategy](#auth-bypass-strategy) in the Tier 4 section
below).

```python
@pytest.mark.integration
def test_backend_health(backend_url):
    """The backend /health endpoint returns 200 after docker-compose up."""
    resp = requests.get(f"{backend_url}/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
```

---

## LLM & Mock Strategy

### Environment gate

```python
import os
import pytest

LIVE_LLM = os.getenv("RUN_TESTS_WITH_LLM", "").lower() in ("1", "true", "yes")

requires_llm = pytest.mark.skipif(
    not LIVE_LLM,
    reason="RUN_TESTS_WITH_LLM not set; skipping live LLM test",
)
```

All existing operator tests that hit a real LLM should be decorated with
`@requires_llm` (or the equivalent `@pytest.mark.llm` marker registered in
`pyproject.toml`).

### Mock LLM fixture

`fixtures/mocks.py` will expose a `mock_litellm` fixture (or similar) that
patches `litellm.completion` and `litellm.embedding` to return
deterministic responses keyed by a combination of the model name and a hash
of the prompt.

```python
# fixtures/mocks.py (sketch)
@pytest.fixture
def mock_litellm(monkeypatch):
    """Patch litellm.completion and litellm.embedding with canned responses."""
    canned = _load_canned_responses()   # from a YAML/JSON file or inline dict

    def fake_completion(model, messages, **kwargs):
        key = _response_key(model, messages)
        return canned["completions"].get(key, _default_completion())

    def fake_embedding(model, input, **kwargs):
        return _deterministic_embedding(input)

    monkeypatch.setattr("litellm.completion", fake_completion)
    monkeypatch.setattr("litellm.embedding", fake_embedding)
```

For embeddings, a simple deterministic function (e.g., hashing text → fixed
vector) is sufficient because mocked tests validate wiring, not semantic
quality.

### Operator test pattern (dual-mode)

Each operator test file should contain both mocked and live tests:

```python
# ── Tier 2: mocked ──────────────────────────────────────────
class TestSemFilterMocked:
    """Mocked tests for SemFilterOperator. Run in every CI build."""

    def test_basic_filter(self, mock_litellm, simple_animal_dataset):
        ...

# ── Tier 3: live ────────────────────────────────────────────
@requires_llm
class TestSemFilterLive:
    """Live LLM tests for SemFilterOperator."""

    @pytest.mark.flaky(reruns=1)
    def test_basic_filter(self, test_model_id, llm_config):
        ...
```

---

## Storage Layer Mocking

### When testing storage directly

Use a real `LocalStorageBackend` backed by `tmp_path`:

```python
@pytest.fixture
def local_backend(tmp_path):
    return LocalStorageBackend(base_dir=tmp_path)
```

For S3 tests, use a real `S3StorageBackend` (requires credentials or
`moto`-mocked S3). Gate these with `@pytest.mark.s3`.

### When testing everything else

Provide an `InMemoryStorageBackend` that stores data in a plain `dict`:

```python
class InMemoryStorageBackend(StorageBackend):
    """In-memory storage backend for fast tests that don't need real I/O."""

    def __init__(self):
        self._store: dict[str, bytes] = {}

    def read(self, uri: str) -> bytes:
        return self._store[uri]

    def write(self, uri: str, data: bytes) -> None:
        self._store[uri] = data

    def exists(self, uri: str) -> bool:
        return uri in self._store

    def delete(self, uri: str) -> None:
        self._store.pop(uri, None)

    def list(self, prefix: str) -> list[str]:
        return sorted(k for k in self._store if k.startswith(prefix))

    def get_uri(self, *path_parts: str) -> str:
        return "/".join(path_parts)
```

This enables operator and execution tests to run without touching the
file system.

---

## Datasets & Test Data

### Guiding principle

Choose tasks where a reasonable LLM has very high probability of success.
This keeps live tests reliable while still exercising the full operator path.

### Current datasets

| Dataset | Location | Good for | Notes |
|---------|----------|----------|-------|
| In-memory animals | inline in tests | `sem_filter`, `sem_map`, `sem_groupby`, `sem_agg` | 5 animals with unambiguous classifications. Near-100% expected accuracy. |
| In-memory fruits | inline in tests | `sem_flat_map` | single text block → structured extraction. |
| Movie reviews (CSV) | `data/movie-reviews/` | `sem_filter`, `sem_map`, `sem_topk`, `code`, planning | 3 movies × up to 10 reviews. Contains `scoreSentiment` for ground-truth. |
| Research papers (TXT) | `data/papers/` | `sem_flat_map`, `sem_join` | 3 papers. Good for author/title extraction. |
| Enron emails (TXT) | `data/emails/` | `sem_topk`, E2E index tests | Real-world unstructured text. Good for search/retrieval tasks. |
| Simple movies | inline fixture | planning, conversation | 4 movies with genre/director/rating/year. Deterministic field lookups. |

### Guidelines for adding new datasets

- Keep datasets **small** (< 50 items for unit tests, < 200 for E2E).
- Include a **ground-truth column or known answer** so assertions are
  objective, not vibes-based.
- Prefer tasks with **categorical or factual answers** (mammal/reptile,
  positive/negative, title extraction) over open-ended generation.
- Store raw data files under `data/`; never commit generated artifacts.

---

## Docstring & Specification Conventions

We adopt the style of Cornell CS 3110 for function and class specifications,
adapted for Python docstrings.

### Functions

```python
def search(self, query: str, k: int = 50) -> list[str]:
    """Top-k file paths most relevant to *query*.

    Uses top-down traversal of the index hierarchy: at each level, nodes
    are ranked by LLM selection (when enabled and the node count fits the
    router context) or by embedding cosine similarity, and the best
    candidates are expanded.

    Requires:
        - `query` is a non-empty string.
        - `k` >= 1.
        - The index has been built (i.e., `_build()` has completed).

    Returns:
        A list of at most *k* file paths, ordered by descending
        relevance. Returns an empty list when the index has no file
        summaries.

    Raises:
        None. Errors during LLM routing are logged and fall back to
        embedding-based ranking.

    Examples:
        >>> idx.search("emails about Raptor investments", k=10)
        ['/data/emails/kaminski-v/123.txt', ...]
    """
```

Key elements:
- **First sentence**: declarative summary — "`search(query, k)` *is* …"
  (or "returns …"). Describes the *result*, not the mechanics.
- **Requires** (preconditions): what the caller must guarantee. Do *not*
  restate type annotations — those are enforced by the type checker.
- **Returns** (postconditions): what the caller can rely on.
- **Raises**: exceptional postconditions. Use "None" to explicitly state
  that the function does not raise.
- **Examples** (optional): concrete input/output pairs.

### Classes

```python
class FlatFileIndex:
    """Single-level index over file summaries for query-time retrieval.

    All file summaries are stored in a flat list. At query time, if the
    number of summaries fits within the LLM context limit, they are sent
    directly to the LLM for top-k selection. Otherwise, embedding cosine
    similarity pre-filters to `max_llm_items` candidates, and the LLM
    selects from that subset.

    Representation invariant:
        - `_embeddings` is None iff `file_summaries` is empty.
        - `_embeddings.shape == (len(file_summaries), embedding_dim)` when
          not None.

    Abstraction function:
        Represents a searchable collection of file summaries where
        `file_summaries[i]` corresponds to row `i` of `_embeddings`.
    """
```

Key elements:
- **First paragraph**: what the class *is* and its high-level behavior.
- **Representation invariant** (RI): properties that must always hold on
  internal state. Tests can assert the RI after construction and mutation.
- **Abstraction function** (AF): how internal state maps to the abstract
  value the class represents.

### When to update docstrings

- **Before writing a test**: if the docstring is missing or imprecise,
  improve it first. The test should verify the spec, not invent one.
- **After changing behavior**: any PR that changes observable behavior must
  update the corresponding docstring(s).

---

## Fixture Organization

### Plugin registration (`conftest.py`)

```python
pytest_plugins = [
    "fixtures.config",
    "fixtures.data",
    "fixtures.datasets",
    "fixtures.mocks",      # NEW
    "fixtures.storage",    # NEW
]
```

### Fixture file responsibilities

| File | Provides | Depends on |
|------|----------|------------|
| `config.py` | `llm_config`, `test_model_id`, `test_embedding_model_id` | env vars |
| `data.py` | `movie_reviews_data`, `research_papers_data`, `enron_emails_data`, `enron_data_items` | file system |
| `datasets.py` | `movie_reviews_datasets`, `simple_movie_dataset` | `data.py` fixtures |
| `mocks.py` | `mock_litellm`, `mock_storage`, `mock_llm_config` | `monkeypatch` |
| `storage.py` | `local_backend`, `tiered_storage`, `in_memory_backend` | `tmp_path` |

### Fixture naming conventions

- `*_data` → raw data (DataFrames, lists of dicts, file contents).
- `*_dataset` / `*_datasets` → assembled `Dataset` objects.
- `*_backend` → `StorageBackend` instances.
- `mock_*` → patched/faked dependencies.
- `*_config` → configuration dicts.

---

## Pytest Markers & Configuration

Add the following to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests/pytest"]
markers = [
    "llm: test requires a live LLM (deselect with '-m not llm')",
    "s3: test requires S3 credentials (deselect with '-m not s3')",
    "slow: test takes > 10 seconds",
    "integration: test requires a running docker-compose stack (deselect with '-m not integration')",
]
filterwarnings = [
    "error",
    "ignore::DeprecationWarning",
    "ignore::ResourceWarning",
    "ignore::UserWarning",
]
```

### Common invocations

```bash
# Fast CI run (Tier 1 + Tier 2 only, no LLM, no S3, no integration)
pytest -m "not llm and not s3 and not integration"

# Full run with live LLM
RUN_TESTS_WITH_LLM=1 pytest

# Only live LLM tests
RUN_TESTS_WITH_LLM=1 pytest -m llm

# Storage-layer tests (local only)
pytest tests/pytest/test_storage_backends.py -m "not s3"

# Single operator, verbose
pytest tests/pytest/test_sem_filter_operator.py -v

# Docker Compose integration tests only
RUN_INTEGRATION_TESTS=1 pytest tests/pytest/integration/ -v

# Everything except integration and LLM (fastest safe default)
pytest -m "not llm and not s3 and not integration"
```

---

## Writing a New Test — Checklist

1. **Identify the contract.** Read (or write) the docstring of the function
   / class under test. Confirm it has Requires/Returns/Raises clauses.

2. **Choose the tier.**
   - Can the test run with no I/O and no LLM? → Tier 1.
   - Does it need an LLM but can use a mock? → Tier 2.
   - Must it hit a real LLM? → Tier 3 (`@requires_llm`).
   - Does it test the deployed web API via docker-compose? → Tier 4
     (`@pytest.mark.integration`). Place in `integration/`.

3. **Pick or create a fixture.** Reuse fixtures from `fixtures/`. If a new
   dataset is needed, add it to the appropriate fixture file with a
   docstring explaining what it's for.

4. **Write the test function.**
   - Name: `test_<unit>_<scenario>` (e.g., `test_sem_filter_empty_dataset`).
   - Group related tests in a class: `TestSemFilterMocked`, `TestSemFilterLive`.
   - For LLM tests, use accuracy thresholds rather than exact equality
     when output is non-deterministic.
   - Keep the body short: setup → act → assert.

5. **Verify locally.**
   ```bash
   pytest tests/pytest/test_<your_file>.py -v
   ```

6. **Update this document** if you introduced a new pattern, fixture, or
   dataset.

---

## Current Coverage Map

This table summarizes what is currently tested and what gaps remain.
"✅" = covered, "⚠️" = partial, "❌" = missing.

| Component | Tier 1 (Unit) | Tier 2 (Mocked) | Tier 3 (Live) | Notes |
|-----------|:---:|:---:|:---:|-------|
| **Semantic Operators** | | | | |
| ∟ `SemFilterOperator` | ❌ | ✅ | ✅ | `test_sem_filter_mocked.py`: boolean parsing, dataset threading, retries, empty input (7 tests) |
| ∟ `SemMapOperator` | ❌ | ✅ | ✅ | `test_sem_map_mocked.py`: JSON enrichment, missing fields → None, original fields preserved (7 tests) |
| ∟ `SemJoinOperator` | ❌ | ✅ | ✅ | `test_sem_join_mocked.py`: cross-product, key-conflict prefixes, call count (7 tests) |
| ∟ `SemAggOperator` | ❌ | ✅ | ✅ | `test_sem_agg_mocked.py`: single-item output, multiple fields, single LLM call (5 tests) |
| ∟ `SemTopKOperator` | ❌ | ✅ | ✅ | `test_sem_topk_mocked.py`: index construction, reuse, catalog registration, k > N (6 tests) |
| ∟ `SemFlatMapOperator` | ❌ | ✅ | ✅ | `test_sem_flat_map_mocked.py`: one-to-many expansion, flattening, empty list (7 tests) |
| ∟ `SemGroupByOperator` | ❌ | ✅ | ✅ | `test_sem_groupby_mocked.py`: count/sum/min/max/mean agg, semantic agg, group-phase call count (7 tests) |
| ∟ `CodeOperator` | ❌ | ❌ | ✅ | Needs mocked tests (complex multi-step code gen) |
| **Data Model** | | | | |
| ∟ `Dataset` | ✅ | — | — | `test_data_model.py`: construction, lazy materialization, serialization, operator chaining, indices, iteration (38 tests) |
| ∟ `DataItem` | ✅ | — | — | `test_data_model.py`: construction, `from_dict`, `materialize`, `update_dict`, path compat (38 tests shared) |
| **Storage Layer** | | | | |
| ∟ `LocalStorageBackend` | ✅ | — | — | `test_storage_backends.py`: full CRUD, streams, config constructor |
| ∟ `S3StorageBackend` | ❌ | — | — | Needs `moto` or real S3 |
| ∟ `TieredStorageManager` | ✅ | — | — | `test_storage_backends.py`: L1 caching, read-through, write-through, invalidation, L2 disk cache, parsed cache |
| ∟ `StorageConfig` | ✅ | — | — | `test_storage_backends.py`: path resolution, CARNOT_HOME, ensure_dirs, repr |
| ∟ `LRUCache` | ✅ | — | — | `test_storage_backends.py`: get/put, eviction, size tracking, clear, contains |
| ∟ `InMemoryStorageBackend` | ✅ | — | — | `test_storage_backends.py`: CRUD, list prefix, len/clear |
| **Index Layer** | | | | |
| ∟ `FlatFileIndex` | ✅ | ❌ | — | `test_index.py`: construction from summaries, empty index, max_llm_items default. Search needs mocked LLM test. |
| ∟ `HierarchicalFileIndex` | ✅ | ❌ | — | `test_index.py`: empty build, flat mode, path-to-summary mapping, max_root_nodes calculation. Full hierarchy needs mocked LLM test. |
| ∟ `FileSummaryCache` | ✅ | — | — | `test_index.py`: save/load, load_many, missing paths, corrupt files |
| ∟ `FileSummaryEntry` | ✅ | — | — | `test_index.py`: dataclass field access |
| ∟ `InternalNode` | ✅ | — | — | `test_index.py`: leaf and internal node construction |
| ∟ `HierarchicalIndexConfig` | ✅ | — | — | `test_index.py`: sensible defaults |
| **Planning** | | | | |
| ∟ `DataDiscoveryAgent` | ❌ | ✅ | ✅ | `test_planner_mocked.py`: exercised as managed agent in multi-step test. Live tests in `test_logical_planning.py`. |
| ∟ `Planner` (logical plan gen) | ❌ | ✅ | ✅ | `test_planner_mocked.py`: leaf plan, sem_filter, sem_map, chained ops, multi-step discovery→plan, max-steps fallback. |
| ∟ `Planner` (paraphrasing) | ❌ | ✅ | ✅ | `test_planner_mocked.py`: paraphrase returns string with plan-tag parsing. |
| ∟ Plan structure validation | ✅ | ✅ | — | `test_planner_mocked.py`: JSON serialisability, dataset-name preservation, recursive node validation. |
| **Conversation** | | | | |
| ∟ `Conversation` basic ops | ✅ | — | — | `test_conversation_integration.py` |
| ∟ `Conversation` → memory steps | ✅ | — | — | |
| ∟ Conversational planning | ❌ | ❌ | ✅ | Needs mocked version |
| **Execution** | | | | |
| ∟ `Execution.plan()` | ❌ | ✅ | ✅ | `test_e2e_query_lifecycle.py`: plan phase exercised in all 3 E2E scenarios |
| ∟ `Execution.run()` | ❌ | ✅ | ⚠️ | `test_e2e_query_lifecycle.py`: filter, join, topk pipelines (3 tests) |
| ∟ `_get_op_from_plan_dict` | ✅ | — | — | `test_execution_unit.py`: all operator types, dataset reference, unknown raises (11 tests) |
| ∟ `_get_ops_in_topological_order` | ✅ | — | — | `test_execution_unit.py`: single leaf, linear chain, two-step chain, join DAG, parent id propagation (5 tests) |
| **Web API (Integration)** | | | | |
| ∟ Health / config endpoints | ❌ | — | — | Tier 4: needs docker-compose stack |
| ∟ Files API (CRUD) | ❌ | — | — | Tier 4: needs docker-compose stack |
| ∟ Datasets API (CRUD) | ❌ | — | — | Tier 4: needs docker-compose stack |
| ∟ Conversations API (CRUD) | ❌ | — | — | Tier 4: needs docker-compose stack |
| ∟ Settings API (CRUD) | ❌ | — | — | Tier 4: needs docker-compose stack |

---

## Tier 4 — Docker Compose Integration Tests

### Overview

Tier 4 tests exercise the **deployed web application** as close to
production as possible. They build Docker images from the current source
tree, bring up the full `docker-compose` stack (Postgres, backend,
frontend), and hit the backend REST API with real HTTP requests using the
`requests` library. No mocking of the backend or database — only Auth0
is bypassed (see below).

These tests answer the question: *"If I deploy this commit, will the web
application actually work?"*

### Gating: when do these tests run?

Integration tests are **skipped by default**. They run only when the
environment variable `RUN_INTEGRATION_TESTS` is set to a truthy value
(`1`, `true`, `yes`) **or** when selected explicitly via pytest marker:

```bash
# Via environment variable
RUN_INTEGRATION_TESTS=1 pytest

# Via marker selection (equivalent)
pytest -m integration

# Combined with other tiers
RUN_INTEGRATION_TESTS=1 pytest -m "not llm and not s3"
```

The marker is registered in `pyproject.toml`:

```toml
markers = [
    ...
    "integration: test requires a running docker-compose stack (deselect with '-m not integration')",
]
```

A `conftest.py` hook auto-applies `pytest.mark.integration` to every
test under `tests/pytest/integration/` so individual test files do not
need to repeat the decorator.

### Session-scoped setup & teardown

All Docker Compose lifecycle management is handled by **session-scoped
fixtures** in `tests/pytest/integration/conftest.py`. The sequence:

#### Setup (once per `pytest` session)

1. **Prepare build context.** Replicate what `deploy/start_local.sh` does:
   copy `app/frontend/`, `app/backend/`, `src/`, `pyproject.toml`, and
   `README.md` into `deploy/compose/` so the Dockerfiles can build from
   the latest source. Use a dedicated `LOCAL_BASE_DIR` under
   `/tmp/carnot-integration-test/` so host state is isolated from any
   real local deployment.

2. **Write secrets files.** Create `deploy/compose/secrets/db_password.txt`,
   `db_user.txt`, `db_name.txt` with test-specific credentials (e.g.,
   `testpassword`, `testuser`, `testdb`).

3. **Set environment variables.** Export the same variables that
   `start_local.sh` exports (`LOCAL_ENV=true`, `VITE_API_BASE_URL`, etc.).
   The Auth0 variables (`VITE_AUTH0_DOMAIN`, `VITE_AUTH0_CLIENT_ID`,
   `VITE_AUTH0_AUDIENCE`, `VITE_AUTH0_ORGANIZATION_ID`) must already be
   set in the developer's shell environment (same requirement as
   `start_local.sh`). The fixture validates their presence and skips
   with a clear message if missing.

4. **`docker compose up --build -d`.** Build images and start all
   services. Poll the backend `/health` endpoint with exponential backoff
   (max ~90 s) to block until services are ready. The `db` service's own
   `healthcheck` already gates backend startup, so the `/health` poll
   is the single readiness signal we need.

5. **Verify health.** Confirm `GET /health` returns
   `{"status": "healthy"}` before yielding control to tests.

6. **Yield `backend_url`.** The fixture yields the base URL
   (`http://localhost:8000/api`) so test functions can make requests.

#### Teardown (once per `pytest` session)

1. **`docker compose down -v`.** Stop containers **and remove volumes**
   (the `-v` flag deletes the Postgres data volume and the mounted
   `LOCAL_BASE_DIR` content). This ensures no state leaks between runs.

2. **Clean up build context.** Remove the files copied into
   `deploy/compose/` (frontend, backend, src, pyproject.toml, README.md),
   mirroring the cleanup in `start_local.sh`.

3. **Remove host temp directory.** Delete the `LOCAL_BASE_DIR` used for
   the test run (`/tmp/carnot-integration-test/`).

#### Per-test cleanup

For tests that create server-side state (files, datasets, conversations),
each test cleans up after itself using API delete calls in a
`yield`-based fixture or `try/finally` block. This avoids coupling
between tests. Example:

```python
@pytest.fixture
def created_dataset(backend_url, auth_headers):
    """Create a dataset for testing and delete it after."""
    resp = requests.post(
        f"{backend_url}/datasets/",
        json={"name": "test-ds", "shared": False, "annotation": "", "files": []},
        headers=auth_headers,
    )
    data = resp.json()
    yield data
    # teardown
    requests.delete(f"{backend_url}/datasets/{data['id']}", headers=auth_headers)
```

### Auth strategy

The backend's `get_current_user` dependency validates a JWT signed by
Auth0. The local docker-compose deployment **does** support real Auth0
authentication (Auth0 is configured with `localhost` as a redirect URI),
so integration tests authenticate with real credentials — no backend
modifications needed.

#### Credentials file: `tests/pytest/.auth`

Auth0 test credentials live in `tests/pytest/.auth` (git-ignored via
`.gitignore`). Format:

```
USERNAME=user@example.com
PASSWORD=hunter2
```

Not all collaborators have access to this file. The `auth_headers`
fixture checks for its existence at session start:
- **File present** → obtain a real JWT and use it for all requests.
- **File missing** → `pytest.skip()` the entire integration session
  with a clear message ("`.auth` credentials file not found; skipping
  integration tests").

This means `RUN_INTEGRATION_TESTS=1` alone is not sufficient — the
`.auth` file must also be present. This is intentional: we don't want
integration tests to silently pass without real auth.

#### Obtaining a JWT programmatically

We use Auth0's **Resource Owner Password Grant** (ROPC) to exchange the
username/password for an access token without a browser. This requires:
- The Auth0 application has the "Password" grant type enabled.
- The Auth0 tenant has a database connection that allows ROPC.

The fixture calls the Auth0 `/oauth/token` endpoint directly:

```python
import requests as http_requests
from pathlib import Path

def _load_auth_credentials() -> dict | None:
    """Load username/password from tests/pytest/.auth, or return None."""
    auth_file = Path(__file__).resolve().parent.parent / ".auth"
    if not auth_file.exists():
        return None
    creds = {}
    for line in auth_file.read_text().strip().splitlines():
        key, _, value = line.partition("=")
        creds[key.strip()] = value.strip()
    return creds

def _get_auth0_token(creds: dict) -> str:
    """Exchange username/password for a JWT via Auth0 ROPC grant."""
    # These env vars must be set (same ones start_local.sh expects)
    domain = os.environ["VITE_AUTH0_DOMAIN"]
    client_id = os.environ["VITE_AUTH0_CLIENT_ID"]
    audience = os.environ.get("VITE_AUTH0_AUDIENCE", "")

    resp = http_requests.post(
        f"https://{domain}/oauth/token",
        json={
            "grant_type": "password",
            "username": creds["USERNAME"],
            "password": creds["PASSWORD"],
            "client_id": client_id,
            "audience": audience,
            "scope": "openid profile offline_access",
        },
    )
    resp.raise_for_status()
    return resp.json()["access_token"]

@pytest.fixture(scope="session")
def auth_headers():
    """Real Auth0 Bearer token for integration tests.

    Requires:
        - ``tests/pytest/.auth`` exists with USERNAME and PASSWORD.
        - Auth0 env vars (``VITE_AUTH0_DOMAIN``, ``VITE_AUTH0_CLIENT_ID``,
          ``VITE_AUTH0_AUDIENCE``) are set in the environment.

    Returns:
        A dict ``{"Authorization": "Bearer <jwt>"}`` usable with
        ``requests.get(..., headers=auth_headers)``.

    Skips:
        The entire integration session if credentials are unavailable.
    """
    creds = _load_auth_credentials()
    if creds is None:
        pytest.skip("tests/pytest/.auth not found; cannot authenticate for integration tests")
    token = _get_auth0_token(creds)
    return {"Authorization": f"Bearer {token}"}
```

#### What if ROPC is not enabled?

If the Auth0 application does not have the Password grant enabled, we
have a fallback path:

1. **Manual token injection.** A developer runs the frontend, logs in
   via the browser, copies the access token from localStorage, and sets
   it as `AUTH0_TEST_TOKEN` in the environment. The fixture checks for
   this env var before attempting ROPC.

2. **Skip gracefully.** If neither ROPC nor manual token is available,
   the fixture skips.

### Test file breakdown

#### `integration/conftest.py` — Session fixtures

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `docker_compose_up` | session | Builds & starts the stack; yields; tears down |
| `backend_url` | session | `http://localhost:8000/api` (depends on `docker_compose_up`) |
| `auth_headers` | session | `{"Authorization": "Bearer <jwt>"}` via Auth0 ROPC grant; skips if `.auth` missing |
| `created_dataset` | function | Creates and cleans up a dataset |
| `uploaded_file` | function | Uploads and cleans up a file |

Auto-marker hook:

```python
def pytest_collection_modifyitems(config, items):
    """Auto-apply @pytest.mark.integration to all tests in this directory."""
    integration_marker = pytest.mark.integration
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(integration_marker)
```

#### `integration/test_health.py` — Smoke tests (4 tests)

| Test | Validates |
|------|-----------|
| `test_backend_health` | `GET /health` → 200, `{"status": "healthy"}` |
| `test_root_endpoint` | `GET /` → 200, `{"message": "Carnot Web API", "status": "running"}` |
| `test_config_endpoint` | `GET /api/config/` returns `base_dir=/carnot/`, `data_dir=/carnot/data/`, `shared_data_dir=/carnot/shared/` |
| `test_frontend_reachable` | `GET http://localhost:80` → 200 (nginx serves the React app) |

#### `integration/test_files_api.py` — File management (5 tests)

| Test | Validates |
|------|-----------|
| `test_browse_root` | `GET /api/files/browse` returns root directory listing |
| `test_upload_file` | `POST /api/files/upload` with a small text file → 200 |
| `test_browse_user_data_dir` | After upload, browsing user's data dir shows the file |
| `test_create_directory` | `POST /api/files/create-directory` creates a subdirectory |
| `test_delete_file` | `POST /api/files/delete` removes a file |

#### `integration/test_datasets_api.py` — Dataset CRUD (6 tests)

| Test | Validates |
|------|-----------|
| `test_create_dataset` | `POST /api/datasets/` → 200 with `id`, `name`, `file_count` |
| `test_list_datasets` | `GET /api/datasets/` includes the created dataset |
| `test_get_dataset_detail` | `GET /api/datasets/{id}` returns files list, annotation |
| `test_update_dataset_annotation` | `PUT /api/datasets/{id}` updates annotation |
| `test_duplicate_name_rejected` | Second `POST` with same name → 400 |
| `test_delete_dataset` | `DELETE /api/datasets/{id}` → 200; list omits it |

#### `integration/test_conversations_api.py` — Conversation CRUD (6 tests)

| Test | Validates |
|------|-----------|
| `test_create_conversation` | `POST /api/conversations/` → 200 |
| `test_list_conversations` | `GET /api/conversations/` includes the created conversation |
| `test_get_conversation_detail` | `GET /api/conversations/{id}` returns messages list |
| `test_add_message` | `POST /api/conversations/message` adds a message |
| `test_update_conversation_title` | `PUT /api/conversations/{id}` updates title |
| `test_delete_conversation` | `DELETE /api/conversations/{id}` → 200 |

#### `integration/test_settings_api.py` — User settings (3 tests)

| Test | Validates |
|------|-----------|
| `test_get_empty_settings` | `GET /api/settings/` for new user → empty / `{}` (LLM-aware: accepts masked values when `RUN_TESTS_WITH_LLM=1`) |
| `test_save_api_keys` | `POST /api/settings/` with API key values → 200 |
| `test_get_masked_settings` | `GET /api/settings/` returns masked versions (`...xxxx`) |

#### `integration/test_query_api.py` — Query execution (4 tests, requires LLM)

These tests are gated with `RUN_TESTS_WITH_LLM=1` in addition to
`RUN_INTEGRATION_TESTS=1`.  They exercise the full query pipeline through
the web API with a live LLM.

**Test data:** `query_test_animals.txt` — 5 animal facts (elephant,
giraffe, blue whale, cheetah, penguin). Query: *"Which animal is the
tallest?"* Expected answer mentions "giraffe".

| Test | Validates |
|------|-----------|
| `test_plan_rejects_empty_query` | `POST /api/query/plan` with `query=""` → error response |
| `test_plan_rejects_no_datasets` | `POST /api/query/plan` with `dataset_ids=[]` → error response |
| `test_plan_query` | Full plan generation (~3–5 min): plan has operators, natural-language summary, session_id |
| `test_execute_query` | Full execution via SSE stream: parses status/result/done events, answer mentions "giraffe" |

**Module-scoped fixtures** (shared across `test_plan_query` and
`test_execute_query` to avoid redundant LLM calls):
- `query_test_dataset` — uploads file, creates dataset, tears down after module.
- `query_plan` — calls `/plan` once with 600 s timeout, returns plan + session_id.

### Implementation phases

See [Phase 6](#phase-6--docker-compose-integration-tests-) in the
Roadmap below.

### Common invocations

```bash
# Run only integration tests (builds & starts docker-compose)
RUN_INTEGRATION_TESTS=1 pytest tests/pytest/integration/ -v

# Run integration + Tier 1/2 (skip LLM and S3)
RUN_INTEGRATION_TESTS=1 pytest -m "not llm and not s3"

# Run everything including integration and live LLM query tests
RUN_INTEGRATION_TESTS=1 RUN_TESTS_WITH_LLM=1 pytest

# Just smoke tests
RUN_INTEGRATION_TESTS=1 pytest tests/pytest/integration/test_health.py -v

# Just CRUD tests (fast, no LLM needed)
RUN_INTEGRATION_TESTS=1 pytest tests/pytest/integration/ -v -k "not query"

# Just query tests (requires LLM key, ~5 min)
RUN_INTEGRATION_TESTS=1 RUN_TESTS_WITH_LLM=1 pytest tests/pytest/integration/test_query_api.py -v
```

---

## Roadmap

### Phase 1 — Foundations (immediate) ✅

- [x] Create `fixtures/mocks.py` with `mock_litellm` and `mock_llm_config`.
- [x] Create `fixtures/storage.py` with `local_backend`, `in_memory_backend`.
- [x] Create `helpers/assertions.py`; refactor duplicated helpers out of
      `test_logical_planning.py` and `test_conversation_planning.py`.
- [x] Register `llm`, `s3`, `slow` markers in `pyproject.toml`.
- [x] Add `@requires_llm` / `@pytest.mark.llm` to all existing live-LLM tests.

### Phase 2 — Tier 1 unit tests ✅

- [x] `test_data_model.py` — `Dataset` construction, lazy materialization,
      `items` property, `serialize/deserialize`, `DataItem.materialize()`.
      (38 tests)
- [x] `test_storage_backends.py` — `LocalStorageBackend` CRUD against
      `tmp_path`; `StorageConfig` path resolution; `LRUCache` eviction;
      `TieredStorageManager` read-through caching.
      (45 tests)
- [x] `test_index.py` — `FlatFileIndex` construction, embedding pre-filter
      logic (with synthetic embeddings), `HierarchicalFileIndex._build()`
      with mock clustering.
      (15 tests)
- [x] `test_execution_unit.py` — `_get_op_from_plan_dict` returns correct
      operator types; `_get_ops_in_topological_order` produces correct order
      for known plan DAGs.
      (16 tests)

### Phase 2b — Contract audit ✅

- [x] Audit all 113 Tier 1 tests against source docstring contracts.
- [x] Update source docstrings to make contracts explicit (DataItem,
      Dataset, StorageConfig, LRUCache, TieredStorageManager, FlatFileIndex,
      HierarchicalFileIndex, FileSummaryCache, Execution helpers,
      hierarchical types).
- [x] Rewrite L2 storage tests to avoid accessing private internals
      (`_l2_key`, `_l1.clear()`).
- [x] Add [Contract-Based Testing Workflow](#contract-based-testing-workflow)
      section to this document.

### Phase 3 — Tier 2 mocked operator tests ✅

- [x] For each `test_sem_*_operator.py`, add a `TestSem*Mocked` class with
      mocked LLM responses. These tests validate prompt construction, output
      parsing, and dataset threading.
      - `test_sem_filter_mocked.py` — 7 tests (boolean parsing, retries, empty input)
      - `test_sem_map_mocked.py` — 7 tests (JSON enrichment, missing fields, preserves originals)
      - `test_sem_agg_mocked.py` — 5 tests (single-item output, multiple fields, single LLM call)
      - `test_sem_join_mocked.py` — 7 tests (cross-product, key-conflict prefixes, call count)
      - `test_sem_flat_map_mocked.py` — 7 tests (one-to-many, flattening, empty list)
      - `test_sem_groupby_mocked.py` — 7 tests (count/sum/min/max/mean, semantic agg)
      - `test_sem_topk_mocked.py` — 6 tests (index mock, construction, reuse, catalog)
      - Supporting infrastructure: `helpers/mock_utils.py` (msg_text helper),
        updated `fixtures/mocks.py` (_make_completion_response with model_dump + usage).
- [x] Mocked planning tests: verify that `Planner.generate_logical_plan`
      produces structurally valid plans without a real LLM.
      - `test_planner_mocked.py` — 12 tests (leaf plan, sem_filter, sem_map,
        chained operators, JSON-serialisability, dataset-name preservation,
        multi-step data-discovery → plan, LLM call count, query-in-messages,
        dataset-info-in-messages, paraphrase, max-steps fallback).

### Phase 4 — Docstring audit (complete ✅)

- [x] Audit and update docstrings for:
      `Dataset`, `DataItem`, `StorageConfig`, `LRUCache`,
      `TieredStorageManager`, `FlatFileIndex`, `HierarchicalFileIndex`,
      `FileSummaryCache`, `FileSummaryEntry`, `InternalNode`,
      `HierarchicalIndexConfig`, `Execution._get_op_from_plan_dict`,
      `Execution._get_ops_in_topological_order`.
- [x] Audit and update docstrings for remaining public methods in:
      `StorageBackend`, `LocalStorageBackend`, `S3StorageBackend`, `Planner`,
      `SemFilterOperator`, `SemMapOperator`, `SemAggOperator`,
      `SemJoinOperator`, `SemFlatMapOperator`, `SemGroupByOperator`,
      `SemTopKOperator`, `CodeOperator`, `ReasoningOperator`,
      `LimitOperator`.
- [x] Ensure each docstring follows the Requires/Returns/Raises convention.
      Classes use RI (Representation Invariant) / AF (Abstraction Function).
      Fixed copy-paste errors in `sem_join`, `sem_flat_map`, and `limit`.

### Phase 5 — Curated E2E tests ✅

- [x] `test_e2e_query_lifecycle.py` — 3 hand-picked queries that exercise
      the full plan → execute → answer pipeline.
      * `TestE2ESingleDatasetFilter` — sem_filter on animals (keeps mammals).
      * `TestE2EMultiDatasetJoin` — sem_join animals × sounds (3 correct pairs).
      * `TestE2EIndexAwareTopK` — sem_topk with mocked chroma index.
- [x] Removed redundant E2E-style assertions from individual operator test
      files (`test_sem_map_operator_movie_reviews` and
      `test_sem_topk_operator_movie_reviews` deleted).

### Phase 6 — Docker Compose integration tests ✅

#### Phase 6a — Scaffolding & smoke tests ✅

- [x] Register `integration` marker in `pyproject.toml`.
- [x] Create `tests/pytest/integration/__init__.py`.
- [x] Create `tests/pytest/integration/conftest.py` with:
  - `docker_compose_up` session fixture (copies source files into build
    context → writes secrets → sets env vars → `docker compose up --build -d`
    → polls `/health` → yield → `docker compose down -v` → cleanup).
  - `backend_url` session fixture (`http://localhost:8000/api`).
  - `backend_root_url` session fixture (`http://localhost:8000`).
  - `auth_headers` session fixture (reads `tests/pytest/.auth`, obtains
    JWT via Auth0 ROPC grant, skips if `.auth` or Auth0 env vars missing).
  - `pytest_collection_modifyitems` hook to auto-mark tests.
  - Skip logic based on `RUN_INTEGRATION_TESTS` env var.
  - Stale PostgreSQL data cleanup (hardcoded `/tmp/pg-data/data` from
    `docker-compose.local.yaml`).
- [x] Create `tests/pytest/integration/test_health.py` with 4 smoke
      tests (`/health`, `/`, `/api/config/`, frontend port 80).
- [x] Verify: `RUN_INTEGRATION_TESTS=1 pytest tests/pytest/integration/ -v`
      — all 4 passed in 21 s.

#### Phase 6b — CRUD API tests ✅

Completed.  All 20 CRUD tests pass (24 total including Phase 6a smoke
tests) in ~24 s.

**Auth strategy change:** Initially used a `TEST_AUTH_BYPASS` env var
for Phase 6b; upgraded to real Auth0 ROPC authentication in Phase 6c
(see below).

**Backend bug fixes discovered by integration tests:**
1. `datetime.UTC` → `timezone.utc` — `from datetime import datetime`
   imports the *class*, which has no `.UTC` attribute.  Fixed in
   `app/database.py` and `app/routes/query.py`.
2. Cross-Base foreign key — `DatasetFile` (app Base) references
   `datasets.id` (catalog Base).  Added metadata merge in
   `app/database.py` mirroring what Alembic env.py already does.

- [x] `tests/pytest/integration/test_files_api.py` — 5 tests (browse root,
      upload, browse user data dir after upload, create directory, delete).
- [x] `tests/pytest/integration/test_datasets_api.py` — 6 tests (create,
      list, detail, update annotation, duplicate-name rejection, delete).
- [x] `tests/pytest/integration/test_conversations_api.py` — 6 tests
      (create, list, detail, add message, update title, delete).
- [x] `tests/pytest/integration/test_settings_api.py` — 3 tests (empty
      settings, save keys, masked retrieval).
- [x] Add function-scoped cleanup fixtures to `integration/conftest.py`.
- [x] `deploy/compose/docker-compose.test.yaml` — test-only compose
      override for `TEST_AUTH_BYPASS`.
- [x] `app/backend/app/auth.py` — conditional bypass when
      `TEST_AUTH_BYPASS=true`.
- [x] Verify: `RUN_INTEGRATION_TESTS=1 pytest tests/pytest/integration/ -v`
      — all 24 passed in ~24 s; 24 skipped in 0.01 s without env var.

#### Phase 6c — Query execution integration tests ✅

End-to-end query execution through the web API with a live LLM.  These
tests upload a file, create a dataset, generate a plan, execute it, and
validate the streamed results.  Gated with both `RUN_INTEGRATION_TESTS=1`
and `RUN_TESTS_WITH_LLM=1`.

**Auth strategy upgrade:** Replaced `TEST_AUTH_BYPASS` with real Auth0
ROPC authentication.  The `auth_headers` fixture in `conftest.py` now
obtains a real JWT via Auth0's `password-realm` grant type using a
dedicated "Carnot Integration Tests" application.  Credentials are
stored in `tests/pytest/.auth` (git-ignored).  The
`docker-compose.test.yaml` override is no longer needed.

**Backend bug fix discovered by integration tests:**
3. `Execution` constructor missing `storage` — The `/plan` and
   `/execute` endpoints in `app/backend/app/routes/query.py` created
   `Execution(...)` without passing a `storage` parameter.  When the
   plan required materializing a dataset (`DataItem.materialize()`),
   it raised `ValueError("No storage provided for materialization")`.
   Fixed by adding a `_build_storage()` helper that creates
   `TieredStorageManager(LocalStorageBackend(base_dir=BASE_DIR))` for
   local deployments (or `S3StorageBackend` for prod) and wiring it
   into both `Execution` constructor calls.

**Upload path resolution fix:**
Tests must upload files to `/carnot/data/` (an absolute path matching
`DATA_DIR`) so that `normalize_path()` triggers the user-specific
directory redirect.  Uploading to a relative `"data/"` path bypasses
the redirect and saves files under the container's CWD (`/code/data/`),
which is outside the volume mount and invisible to
`LocalStorageBackend(base_dir="/carnot/")`.

- [x] `tests/pytest/integration/test_query_api.py` — 4 tests:
  - `test_plan_rejects_empty_query` — `POST /api/query/plan` with empty
    query → 422 / 500.
  - `test_plan_rejects_no_datasets` — `POST /api/query/plan` with empty
    `dataset_ids` → 422 / 500.
  - `test_plan_query` — full plan generation: upload file → create
    dataset → `POST /api/query/plan` with 600 s timeout → validates plan
    has operators, natural-language summary, and session_id (~3–5 min
    with data discovery + LLM planning).
  - `test_execute_query` — full execution: reuses plan from above →
    `POST /api/query/execute` as SSE stream → parses status/result/done
    events → validates answer mentions "giraffe" (the tallest animal in
    the 5-animal test dataset).
- [x] Module-scoped fixtures: `query_test_dataset` (uploads + creates
      dataset, cleans up after module), `query_plan` (calls `/plan` once,
      shared by `test_plan_query` and `test_execute_query`).
- [x] Session-scoped fixtures added to `conftest.py`: `llm_api_key`
      (reads `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` from env),
      `backend_with_llm_key` (saves LLM key to backend settings before
      query tests run).
- [x] Updated `test_settings_api.py` to be LLM-aware: when
      `RUN_TESTS_WITH_LLM=1`, `test_get_empty_settings` accepts masked
      values (from the session-scoped LLM key setup) as well as empty.
- [x] Verify: `RUN_INTEGRATION_TESTS=1 RUN_TESTS_WITH_LLM=1 pytest
      tests/pytest/integration/ -v` — all 28 passed in ~5 min 17 s.

---

## Appendix: Helper Patterns

### `assert_agent_did_not_hit_max_steps`

Currently duplicated in `test_logical_planning.py` and
`test_conversation_planning.py`. Move to `helpers/assertions.py`:

```python
# helpers/assertions.py
from carnot.agents.utils import AgentMaxStepsError

def assert_agent_did_not_hit_max_steps(agent) -> None:
    """Assert that *agent* completed before its step budget.

    Requires:
        - `agent` has a `memory` attribute with a `steps` list.

    Raises:
        AssertionError: if the last step contains an `AgentMaxStepsError`.
    """
    if agent.memory.steps:
        last_step = agent.memory.steps[-1]
        error = getattr(last_step, "error", None)
        assert not isinstance(error, AgentMaxStepsError), (
            f"Agent hit max_steps limit ({agent.max_steps}). "
            "This task should complete in fewer steps."
        )
```

### Accuracy assertion for LLM outputs

```python
def assert_accuracy_above(
    predicted: list[dict],
    ground_truth: list[dict],
    key: str,
    gt_key: str,
    threshold: float = 0.8,
) -> None:
    """Assert that predicted values match ground truth above a threshold.

    Requires:
        - `predicted` and `ground_truth` have the same length.
        - Each dict in both lists contains the specified keys.
        - 0.0 <= threshold <= 1.0.

    Raises:
        AssertionError: if accuracy < threshold.
    """
    assert len(predicted) == len(ground_truth)
    correct = sum(
        1 for p, gt in zip(predicted, ground_truth)
        if p[key].lower() == gt[gt_key].lower()
    )
    accuracy = correct / len(predicted)
    assert accuracy >= threshold, (
        f"Accuracy {accuracy:.2%} below threshold {threshold:.2%}"
    )
```
