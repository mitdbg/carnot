# Skunk Console

## Quick Start

```bash
cd /home/gerardo/carnot/skunk/harness_ui
REASONER=dummy_agent:solve ./serve_local_console.sh
```

- Access the UI at `http://localhost:8790`
- The dummy agent simulates 1-5 seconds processing delay per question
- Default reasoner (without override) is `skunk_reasoner:solve`

## Processes

The launcher spawns exactly three long-running processes:

1. **Practice API** - `practice_server.py` on port `8765`
2. **Skunk Server** - `skunk_server.server` on port `8787`
3. **Skunk Client** - `skunk_client.client` on port `8790` (web UI)

## Launcher Variables

All variables are defined inside `serve_local_console.sh` unless overridden via environment:

| Variable        | Purpose                        | Default (script)              |
|-----------------|--------------------------------|-------------------------------|
| `REASONER`      | Reasoner module:function       | `skunk_reasoner:solve`        |
| `AUTO_SUBMIT`   | Enable automatic submission    | `false`                       |
| `PYTHON_BIN`    | Python interpreter path        | `/home/gerardo/.local/share/mamba/envs/carnot/bin/python` |

`ROUND_SECONDS`, `QUESTIONS`, and port numbers are set inside the launcher script and cannot be overridden from the environment.

## Manual Startup (for debugging)

Run each command separately from the `harness_ui` directory.

### 1. Practice Server

```bash
/home/gerardo/.local/share/mamba/envs/carnot/bin/python practice_server.py \
    --host 127.0.0.1 --port 8765 \
    --round-seconds 900 \
    --questions questions/officeqa_3x5_questions.json
```

No `--team-token` flag is used.

### 2. Skunk Server

```bash
/home/gerardo/.local/share/mamba/envs/carnot/bin/python -m skunk_server.server \
    --host 127.0.0.1 --port 8787 \
    --cup-base-url http://127.0.0.1:8765 \
    --team-token anything \
    --reasoner skunk_reasoner:solve \
    --concurrency 3 \
    --no-auto-submit
```

- Use `--auto-submit` instead of `--no-auto-submit` to enable automatic submission when a successful agent task reaches `READY`.

### 3. Skunk Client

```bash
/home/gerardo/.local/share/mamba/envs/carnot/bin/python -m skunk_client.client \
    --host 127.0.0.1 --port 8790 \
    --server-url http://127.0.0.1:8787
```

## Environment Variables (Server & Client)

The server respects:

| Variable              | CLI Equivalent          | Description                           | Default              |
|-----------------------|-------------------------|---------------------------------------|----------------------|
| `CUP_BASE_URL`        | `--cup-base-url`        | Base URL of the Practice API          | (required)           |
| `CUP_TEAM_TOKEN`      | `--team-token`          | Team token for submission             | (required)           |
| `SKUNK_REASONER`      | `--reasoner`            | Reasoner module path                  | `skunk_reasoner:solve` |
| `SKUNK_CONCURRENCY`   | `--concurrency`         | Number of worker threads              | `3`                  |
| `SKUNK_QUEUE_SIZE`    | `--queue-size`          | Maximum task IDs in each bounded queue| `200`                |
| `SKUNK_AUTO_SUBMIT`   | `--auto-submit`         | Enable automatic submission           | `false`              |

The client uses:

| Variable              | CLI Equivalent          | Description               | Default                   |
|-----------------------|-------------------------|---------------------------|---------------------------|
| `SKUNK_SERVER_URL`    | `--server-url`          | URL of the Skunk server   | `http://127.0.0.1:8787`   |

The launcher script reads `REASONER` and `AUTO_SUBMIT` from the environment and passes them to the server.

## Human Workflow and Task Statuses

All data is kept in memory with no persistent database. Multiple human workers may claim the same READY or FAILED task, and the first valid human action wins.

Exact task statuses (no `PENDING` or `IN_PROGRESS`):

| Status         | Description                                                  |
|----------------|--------------------------------------------------------------|
| `RECEIVED`     | Question received                                            |
| `QUEUED`       | Awaiting agent                                               |
| `RETRY_QUEUED` | Retry awaiting agent                                         |
| `PROCESSING`   | Agent running                                                |
| `READY`        | Candidate available                                          |
| `FAILED`       | Attempt failed                                               |
| `SUBMITTING`   | Sending                                                      |
| `SUBMITTED`    | Accepted                                                     |
| `SCORED`       | Score event received                                         |
| `CANCELLED`    | Round closed/cancelled                                       |

## Testing

### Unit Tests (8 pass)

```bash
cd /home/gerardo/carnot/skunk
PYTHONPATH=harness_ui /home/gerardo/.local/share/mamba/envs/carnot/bin/python -m pytest -q harness_ui/tests/test_skunk_server.py
```

The legacy `harness_ui/tests/test_kit_smoke.py` suite could not run in the
current environment because `pytest-asyncio` is missing, although it is
declared in `harness_ui/requirements.txt`.

## Troubleshooting

| Symptom                              | Likely Cause                                     |
|--------------------------------------|--------------------------------------------------|
| Cannot connect to Practice API       | Practice server not running on port 8765         |
| Cannot connect to Skunk server       | Server not started on port 8787                  |
| Web UI not loading                   | Client not started or port 8790 blocked          |
| Python command not found             | `PYTHON_BIN` path incorrect or environment issue |
| Reasoner returns no results          | Data not available or reasoner misconfigured     |

## Notes

- The legacy `competition_console.py` is retained in the repository but is **not launched** by `serve_local_console.sh`.
