# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Carnot is an **optimized system for deep research** from MIT DSG. It uses LLM-powered agents and a semantic query algebra to plan and execute research queries over heterogeneous document datasets.

## Commands

### Core Library (Python package)

```bash
# Install with test + eval extras
pip install -e ".[tests,evals]"

# Run all tests
pytest tests/pytest

# Run a single test file
pytest tests/pytest/test_sem_map_operator.py

# Run a specific test by name
pytest tests/pytest/test_sem_map_operator.py::test_function_name

# Lint & format (run from repo root)
ruff check src/
ruff format src/
```

### Web App (Local Stack)

All deployment is Docker-based. Run from the `deploy/` directory:

```bash
# Start the full local stack (backend + frontend + PostgreSQL)
cd deploy && ./start_local.sh
# Frontend: http://localhost  Backend: http://localhost:8000

# Generate a new Alembic database migration (runs start_local.sh first)
cd deploy && ./generate_migration.sh "Describe schema change here"
```

Database migration files are generated inside the container, then copied to `app/backend/alembic/versions/`.

## Architecture

### Core Library (`src/carnot/`)

The library is structured around a **semantic query algebra** executed via **LLM-powered agents**.

**Public API** (`src/carnot/__init__.py`): `Conversation`, `Dataset`, `DataItem`, `Execution`

**Data model** (`data/`):
- `Dataset`: A named collection of `DataItem` files with an optional annotation and `CarnotIndex` for vector search. Supports chained logical operations (used for plan building).
- `DataItem`: A single file/document in a dataset.

**Query operators** (`operators/`): Two-layer design.
- `operators/logical.py`: Logical operators (`FilteredScan`, `MapScan`, `Aggregate`, `TopK`, `JoinOp`, `Code`, `Limit`) used for plan representation.
- `operators/physical.py` and individual files: Physical implementations (`SemMapOperator`, `SemFilterOperator`, `SemJoinOperator`, `SemAggOperator`, `SemFlatMapOperator`, `SemGroupByOperator`, `SemTopKOperator`, `CodeOperator`, `ReasoningOperator`, etc.).

**Agents** (`agents/`):
- `Planner` (`agents/planner.py`): A `BaseAgent` subclass that generates code-based logical plans using `LOGICAL_OPERATORS`. Manages a `DataDiscoveryAgent` as a sub-agent.
- `DataDiscoveryAgent` (`agents/data_discovery.py`): Inspects dataset schemas, samples items, and identifies relevant datasets for a query. Used by Planner during planning.
- LLM prompt templates are YAML files in `agents/prompts/`.
- LLM access is via `litellm` wrapped in `LiteLLMModel` (`agents/models.py`).

**Planning and execution flow** (`execution/execution.py`):
1. `Execution.__init__()`: Selects planner model based on available API keys (defaults to OpenAI GPT-5, falls back to Anthropic or Gemini).
2. `Execution.plan()`: Two-phase — (a) `Planner` generates a code-based logical plan (calling `DataDiscoveryAgent` as needed), then (b) translates it to natural language.
3. `Execution.execute()`: Runs the physical plan.

**Indices** (`index/index.py`): `ChromaIndex` (ChromaDB) and `FaissIndex` (FAISS) implement `CarnotIndex` for vector similarity search over dataset items.

**Stats tracking** (`core/models.py`): Hierarchical Pydantic models — `RecordOpStats` → `OperatorStats` → `PlanStats` / `SentinelPlanStats` → `ExecutionStats`. `SentinelPlanStats` maps `logical_op_id → {full_op_id → OperatorStats}` while `PlanStats` maps `full_op_id → OperatorStats`.

**Memory** (`memory/memory.py`): Stub `Memory` class for future cross-session context retrieval.

**Optimizer** (`optimizer/`): Cost model and plan optimizer for selecting physical plan variants.

**Conversation** (`conversation/conversation.py`): Manages conversation history; converts messages to `MemoryStep` objects (`ConversationUserStep`, `ConversationAgentStep`) for agent memory.

### Web App (`app/`)

**Backend** (`app/backend/`) — FastAPI + PostgreSQL (async SQLAlchemy + Alembic):

- `app/main.py`: API routes under `/api/` — `config`, `files`, `datasets`, `search`, `query`, `conversations`, `settings`.
- `app/database.py`: DB models — `UserSettings`, `Dataset`, `File`, `DatasetFile`, `Conversation`, `Message`. Database credentials read from Docker secrets at `/run/secrets/`.
- `app/auth.py`: Auth0 JWT validation via RS256 / JWKS. `get_current_user()` is the FastAPI dependency.
- `app/env.py`: `IS_LOCAL_ENV` flag switches storage between local filesystem (`/carnot/`) and S3 (`s3://carnot-research-{env}/`).
- `app/services/file_service.py`: `LocalFileService` / `S3FileService` abstractions.
- `app/services/llm.py`: `get_user_llm_config()` — retrieves per-user API keys from DB for LLM calls.
- `app/routes/query.py`: Streams execution progress to the frontend via SSE (`StreamingResponse`). Long-running queries use a heartbeat (30s) to keep connections alive.

**Frontend** (`app/frontend/`) — React 18 + Vite + Tailwind CSS + Auth0:
- Standard Vite project; `npm run dev` for local dev, `npm run build` for production.
- Auth0 configured via `VITE_AUTH0_*` environment variables at build time.

### Deployment (`deploy/`)

- `compose/docker-compose.yaml` + `compose/docker-compose.local.yaml`: Full stack (backend, frontend, PostgreSQL).
- `compose/Dockerfile-backend` / `compose/Dockerfile-frontend`: Container images.
- `start_local.sh`: Copies source into the compose build context, writes default DB secrets, then runs `docker compose up --build -d`.

### Evaluations (`evals/`)

Contains evaluation pipelines under `evals/quest/` and `evals/sembench/`. Install with `pip install -e ".[evals]"`.

## Key Conventions

- **LLM routing**: `Execution` auto-selects the planner model based on which API key is present in `llm_config` (priority: `OPENAI_API_KEY` > `ANTHROPIC_API_KEY` > `GEMINI_API_KEY`/`GOOGLE_API_KEY`).
- **Line length**: 120 characters (ruff). Target Python 3.12.
- **Tests**: Use pytest fixtures from `tests/pytest/fixtures/` (`config`, `data`, `datasets`); test files follow `test_<operator_name>.py` naming.
- **PRs**: Target the `dev` branch. Reference issues with `Closes #<issue-id>`.
