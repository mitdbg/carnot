# Skunk Human GUI Architecture (Implemented)

## 1. System Overview

```text
+-----------------------------+  +-----------------------------+  +-----------------------------+
| External Competition Server |  | skunk_client/client.py      |  | skunk_reasoner.py           |
|                             |  |                             |  |                             |
| practice_server.py (local)  |  | Human UI and operator       |  | Creates and invokes the     |
| or competition Cup server   |  | commands                    |  | Orchestrator                |
+--------------+--------------+  +--------------+--------------+  +--------------+--------------+
               ^                                ^                                ^
               | Cup HTTP and WS                | HTTP and WebSocket             | reasoning calls
               v                                v                                v
+------------------------------------------------------------------------------------------------+
| Skunk Server process                                                                            |
|                                                                                                 |
|  +-----------------------------+  +-----------------------------+  +--------------------------+ |
|  | competition_adapter.py      |  | api.py and events.py        |  | agent_worker_pool.py     | |
|  |                             |  |                             |  |                          | |
|  | Receives Cup events and     |  | HTTP commands from clients  |  | Worker threads invoke    | |
|  | sends Cup submissions       |  | and WebSocket updates       |  | the reasoner adapter     | |
|  +--------------+--------------+  +--------------+--------------+  +-------------+------------+ |
|                 |                                |                               |              |
|                 +----------------+---------------+-------------------------------+              |
|                                  | internal commands and results                                |
|                                  v                                                              |
|  +------------------------------------------------------------------------------------------+   |
|  | task_registry.py                                                                         |   |
|  |                                                                                          |   |
|  | Authoritative tasks, attempts, candidates, assignments, submissions, and transitions     |   |
|  +-------------------------------------------+----------------------------------------------+   |
|                                              ^                                                  |
|                                              | task IDs and state transitions                   |
|                                              v                                                  |
|  +------------------------------------------------------------------------------------------+   |
|  | task_queues.py                                                                           |   |
|  |                                                                                          |   |
|  |      Agent Task Queue             Ready Answer Queue            Failed Task Queue        |   |
|  +---------------+----------------------------+----------------------------+----------------+   |
|                  |                            |                            |                    |
|                  |                            +-------------+--------------+                    |
|                  |                                          v                                   |
|                  |                           +-----------------------------+                    |
|                  |                           | human_work_broker.py        |                    |
|                  |                           |                             |                    |
|                  |                           | Assigns ready/failed work   |                    |
|                  |                           | to one or more workers      |                    |
|                  +--------------+------------+--------------+---------------+                    |
|                  |                            |                            |                    |
|                  |                            v                            v                    |
|                  |           +---------------------------+ +-----------------------------+      |
|                  |           | submission_coordinator.py | | human_worker_registry.py    |      |
|                  |           |                           | |                             |      |
|                  |           | Validates manual or auto  | | Worker UUIDs, sessions,     |      |
|                  |           | submissions, delegates to | | display names, presence     |      |
|                  |           | competition_adapter       | +-----------------------------+      |
|                  |           +---------------------------+                                      |
|                  |                                                                             |
|  +------------------------------------------------------------------------------------------+   |
|  | server.py                                                                                |   |
|  | Composition root: creates modules, manages process startup/shutdown, dependency wiring   |   |
|  +------------------------------------------------------------------------------------------+   |
+------------------------------------------------------------------------------------------------+
```

The real competition Server is external to Skunk. During local development,
`practice_server.py` substitutes for it and exposes the same Cup HTTP and
WebSocket surfaces. The existing `cup_kit.client` and `cup_kit.protocol`
modules remain the boundary between the Competition Server and Skunk Server.
Each labeled box corresponds to one actual Python file.

The legacy `harness_ui/competition_console.py` is retained in the repository
but is no longer the active entry point. The `serve_local_console.sh` script
now launches the three-process architecture: practice_server, skunk_server,
and skunk_client.

## 2. Component Responsibilities

### Skunk Server

- Owns the authoritative task state and all queue transitions.
- Receives questions and round events through the Competition Adapter.
- Runs the Agent Worker Pool and injects the configured reasoner into workers.
- Registers human workers and stores their stable IDs and saved display names.
- Assigns ready and failed tasks to one or more human workers.
- Validates and forwards manual or configured automatic submissions.
- Publishes state changes to connected clients.

The server does not render reasoning itself, fabricate missing reasoning, or
allow clients to contact the Competition Server or orchestrators directly.

### Competition Adapter

- Wraps `cup_kit.client.CupClient`.
- Converts Cup round and question events into internal tasks.
- Sends submissions prepared by the Submission Coordinator.
- Converts accepted, rejected, and scored responses into registry transitions.

It contains no queue scheduling or human-workflow policy.

### Agent Workers

- Are server-owned worker threads.
- Atomically claim IDs from the Agent Task Queue.
- Invoke `skunk_reasoner.solve(prompt)` where prompt includes retry feedback
  and available Cup rejection or scoring feedback appended by the server for
  backward compatibility.
- Store an `AnswerCandidate` on success.
- Normalize exceptions into a `FailureRecord`.
- Post completions back to the server event loop via a thread-safe async queue.

Agent workers never submit answers to the Competition Server.

### Skunk Clients / Human Workers

Each new client WebSocket connection creates a new human worker with an
immutable `worker_id` UUID. No authentication or identity recovery is
required. A user may save a natural-language `display_name` through the UI by
sending `PUT /api/workers/{worker_id}` with JSON body
`{"display_name": "Treasury Tables"}`. Display names need not be unique, so
assignments are shown as `<display_name> (<short worker_id>)`, for example
`Treasury Tables (a13f)`.

The client shows an overview of all tasks in the current competition round and
their statuses. Once agent execution reaches `READY` or `FAILED`, a Claim
button lets a human create an assignment for that task. The same task may be
claimed by multiple workers:

- A ready answer may be submitted or retried with free-text feedback.
- A failed task may be retried with free-text feedback or answered directly by
  the human and submitted.

The first valid action atomically transitions the task. Other active
assignments for the old task version become `SUPERSEDED`.

Clients are thin interfaces. They do not own durable task state, Cup
credentials, or reasoning runtimes.

The client process runs as a separate FastAPI application. Its browser-based
UI communicates directly with the Skunk Server via HTTP and WebSocket under
permissive CORS configuration.

### Human Worker Registry

- Allocates a new immutable `worker_id` UUID for every new client connection.
- Stores the optional user-supplied `display_name` when Save is pressed.
- Tracks the connection session and presence for that ephemeral worker.
- Produces mnemonic UI labels such as `Treasury Tables (a13f)`.
- Resolves the full worker UUID for assignment ownership checks.

It does not assign tasks or decide which human action wins; those remain the
responsibility of the Human Work Broker and atomic Task Registry transitions.

## 3. Authoritative Records

```text
QuestionTask
- task_id: str (format: "<round_num>:<question_id>")
- round_num: int
- question_id: str
- prompt: str
- status: TaskStatus enum
- current_attempt_id: uuid | None
- attempts: list[Attempt]
- answer_candidates: list[AnswerCandidate]
- failures: list[FailureRecord]
- submissions: list[SubmissionRecord]
- round_ends_at: datetime | None
- version: int
- created_at: datetime
- updated_at: datetime

Attempt
- attempt_id: uuid
- task_id: str
- attempt_number: int
- worker_id: str
- feedback: str | None
- previous_attempt_ids: list[uuid]
- started_at: datetime
- completed_at: datetime | None

AnswerCandidate
- candidate_id: uuid
- attempt_id: uuid
- answer_text: str
- reasoning: str
- source_docs: list[str]
- submission_type: SubmissionType (AGENT | HUMAN)
- created_at: datetime

FailureRecord
- failure_id: uuid
- attempt_id: uuid
- error_type: str
- error_message: str
- traceback: str | None
- created_at: datetime

HumanWorker
- worker_id: uuid
- display_name: str | None
- created_at: datetime
- updated_at: datetime

HumanWorkerSession
- session_id: uuid
- worker_id: uuid
- connected_at: datetime
- last_seen_at: datetime
- status: SessionStatus (CONNECTED | DISCONNECTED)

HumanAssignment
- assignment_id: uuid
- task_id: str
- worker_id: uuid
- task_version: int
- source_kind: AssignmentSource (READY | FAILED)
- assigned_at: datetime
- completed_at: datetime | None
- status: AssignmentStatus (ACTIVE | RELEASED | COMPLETED | SUPERSEDED)

SubmissionRecord
- local_submission_id: uuid
- task_id: str
- candidate_id: uuid
- submission_type: SubmissionType
- status: SubmissionStatus (PENDING | ACCEPTED | REJECTED | SCORED)
- cup_submission_id: str | None
- rejection_reason: str | None
- correct: bool | None
- points_awarded: float | None
- submitted_at: datetime
```

The Task Registry is authoritative. Queues contain task IDs only, not
independent copies of task state.

## 4. Task Lifecycle

```mermaid
stateDiagram-v2
    [*] --> RECEIVED: Cup question
    RECEIVED --> QUEUED: enqueue
    QUEUED --> PROCESSING: agent claims
    RETRY_QUEUED --> PROCESSING: agent claims
    PROCESSING --> READY: answer candidate
    PROCESSING --> FAILED: execution failure

    READY --> READY: assign human worker
    READY --> SUBMITTING: first valid submit or auto-submit
    READY --> RETRY_QUEUED: first valid retry

    FAILED --> FAILED: assign human worker
    FAILED --> RETRY_QUEUED: first valid retry
    FAILED --> SUBMITTING: first valid direct answer

    SUBMITTING --> SUBMITTED: Cup accepts
    SUBMITTING --> READY: Cup rejects or transport failure
    SUBMITTED --> SCORED: score event

    READY --> CANCELLED: round closes
    FAILED --> CANCELLED: round closes
    QUEUED --> CANCELLED: round closes
    RETRY_QUEUED --> CANCELLED: round closes
```

A Cup business rejection or a transport/network failure during submission
creates a rejected `SubmissionRecord`, preserves the rejection feedback,
restores the task to `READY`, and re-enqueues it on the Ready Answer Queue
while the round remains active.

When auto-submit is enabled, a generated candidate may move directly from
`READY` to `SUBMITTING` without a human assignment. Auto-submit uses the same
validation, idempotency, deadline checks, and Submission Coordinator as manual
submission.

Cup rejection, scoring, and accepted-with-immediate-score feedback are stored
on the task and automatically appended to the context of any later reasoning
attempt. WebSocket score events also update the records.

Human assignments do not change the task from `READY` or `FAILED`. An action
must include its `assignment_id` and observed `task_version`. The first valid
action advances the task and marks other active assignments for that version
as `SUPERSEDED`.

## 5. Normal Submission Flow

```mermaid
sequenceDiagram
    participant Cup as Competition Server
    participant CA as Competition Adapter
    participant TR as Task Registry
    participant AQ as Agent Task Queue
    participant AW as Agent Worker
    participant SR as skunk_reasoner
    participant RQ as Ready Answer Queue
    participant HB as Human Work Broker
    participant HC as Human Client
    participant SC as Submission Coordinator

    Cup->>CA: round/question event
    CA->>TR: create task
    TR->>AQ: enqueue task ID
    AQ->>AW: claim task ID
    AW->>TR: begin attempt
    AW->>SR: solve(prompt with feedback)
    SR-->>AW: answer, reasoning, source docs
    AW->>TR: store AGENT candidate
    AW->>RQ: enqueue ready task ID
    HC->>HB: claim ready task
    HB->>TR: create READY HumanAssignment
    HB-->>HC: assignment and worker mnemonic
    HC->>SC: submit assignment
    SC->>TR: transition to SUBMITTING
    SC->>TR: supersede sibling assignments
    SC->>CA: submit AGENT candidate
    CA->>Cup: POST submission
    Cup-->>CA: accepted (with optional immediate score)
    CA->>TR: transition to SUBMITTED, store immediate correct/points feedback
    Cup-->>CA: score event
    CA->>TR: transition to SCORED, store feedback
```

## 6. Retry and Failure Flows

```mermaid
sequenceDiagram
    participant HC as Human Client
    participant HB as Human Work Broker
    participant TR as Task Registry
    participant AQ as Agent Task Queue
    participant FQ as Failed Task Queue
    participant RQ as Ready Answer Queue
    participant SC as Submission Coordinator
    participant CA as Competition Adapter
    participant Cup as Competition Server

    alt Ready task is retried
        HC->>HB: retry(assignment_id, task_version, feedback)
        HB->>TR: create retry attempt with feedback
        HB->>TR: supersede sibling assignments
        TR->>AQ: enqueue RETRY_QUEUED task
    else Failed task is retried
        HC->>HB: retry(assignment_id, task_version, feedback)
        HB->>TR: create retry attempt with feedback
        HB->>TR: supersede sibling assignments
        TR->>AQ: enqueue RETRY_QUEUED task
    else Failed task receives direct human answer
        HC->>HB: submit_answer(assignment_id, task_version, answer, reasoning, docs)
        HB->>TR: create HUMAN candidate
        HB->>TR: supersede sibling assignments
        HB->>SC: submit HUMAN candidate
        SC->>CA: submit
        CA->>Cup: POST submission
    end

    opt Cup rejects a submission or transport failure occurs
        Cup-->>CA: rejection reason or timeout
        CA->>TR: store rejected submission
        TR->>TR: restore task to READY
        TR->>RQ: re-enqueue ready task
    end
```

## 7. Queue and Concurrency Semantics

- Registry transitions are atomic and version checked.
- The three bounded `queue.Queue` instances provide backpressure and carry
  task IDs only.
- One task may have at most one active agent attempt and one pending
  submission, but may have multiple active human assignments.
- Each processing attempt receives a unique ID. A completion whose attempt ID
  is no longer current is discarded as stale.
- At-least-once execution means a task may be redelivered after recovery or a
  failed handoff. It does not permit simultaneous duplicate active attempts.
- Human assignment does not remove a task from the ready or failed workflow.
  Multiple workers may inspect the same task version concurrently.
- Assignments are created only when a human presses Claim for a `READY` or
  `FAILED` task shown in the current-round overview.
- Human actions include `assignment_id` and `task_version`. The first valid
  action wins atomically; stale actions are rejected and sibling assignments
  are marked `SUPERSEDED`.
- Worker threads use a thread-safe registry boundary. They post completions to
  the async server event loop via an internal async queue, and the event loop
  processes them and broadcasts updates.
- Mutating HTTP commands carry idempotency keys.

## 8. Reasoner Contract

The reasoner boundary is backward-compatible:

```python
# skunk_reasoner.solve() is invoked by agent worker threads
def solve(prompt: str) -> tuple[str, str, list[str]]:
    """
    Returns (answer_text, reasoning, source_docs).

    prompt includes the original question plus any retry feedback and
    Cup rejection/scoring feedback appended by the server.
    """
```

Retry feedback, previous attempts, and available Cup rejection or scoring
feedback are supplied as context. Cup feedback is appended automatically; the
human does not need to repeat it in free text. The worker normalizes the
existing `skunk_reasoner.py` output into `answer_text`, `reasoning`, and
`source_docs`. Exceptions become structured failures rather than escaping the
worker loop. Workers can invoke either a synchronous or async `solve()`
depending on the underlying orchestrator; the interface is compatible with
both.

## 9. Client Protocol

HTTP is authoritative for commands:

```text
GET  /api/state              # full server state snapshot
GET  /api/tasks              # all tasks in current round
GET  /api/tasks/{task_id}    # single task detail
PUT  /api/workers/{worker_id} # save display name
POST /api/assignments        # claim a READY or FAILED task
POST /api/assignments/{assignment_id}/release  # release a claim
POST /api/submissions        # submit an AGENT candidate
POST /api/retries            # request retry with feedback
POST /api/human-answers      # submit a direct human answer
```

The server creates a new worker UUID when the client WebSocket connection is
established and returns it in the initial state snapshot. The UI Save button
sends `PUT /api/workers/{worker_id}` with
`{"display_name": "Treasury Tables"}`. `POST /api/assignments` is called by
the task overview's Claim button and is valid only for `READY` or `FAILED`
tasks. Assignments and task views include both the UUID and the mnemonic label
`Treasury Tables (a13f)`.

The WebSocket is server-push only:

```text
state_snapshot         # full state on connection
task_created           # new task added
task_updated           # task status changed
worker_updated         # display name changed
assignment_created     # new assignment
assignment_updated     # assignment status changed
round_updated          # round info changed
submission_updated     # submission status changed
score_updated          # score event received
```

## 10. Endpoints and Module Responsibilities

### api.py (routes)

- Handles all HTTP endpoints listed above.
- Validates request bodies and dispatches to domain modules.
- Returns appropriate HTTP status codes and JSON responses.
- No domain logic beyond validation and dispatch.

### events.py

- Defines WebSocket event types and serialization.
- Manages connected client sessions for push notifications.
- Broadcasts state changes to all connected clients.

### competition_adapter.py

- Wraps `cup_kit.client.CupClient`.
- Receives Cup round/question events via subscription.
- Converts events to internal tasks via TaskRegistry.
- Sends `SubmitRequest` to Cup server on behalf of SubmissionCoordinator.
- Receives acceptance, rejection, and score responses.
- Stores feedback in TaskRegistry on Cup responses.

### domain.py

- Defines all data classes:
  `QuestionTask`, `Attempt`, `AnswerCandidate`, `FailureRecord`,
  `HumanWorker`, `HumanWorkerSession`, `HumanAssignment`, `SubmissionRecord`
- Defines enums:
  `TaskStatus`, `SubmissionType`, `SessionStatus`, `AssignmentStatus`,
  `SubmissionStatus`
- Provides helper methods for status transitions.

### task_registry.py

- Authoritative in-memory store for all task state.
- Implements atomic state transitions with version checking.
- Creates and queries tasks, attempts, candidates, failures, submissions.
- Manages assignment lifecycle (create, supersede, complete).
- Thread-safe access via locking.

### task_queues.py

- Creates three bounded `queue.Queue` instances:
  - Agent Task Queue (task IDs for agent processing)
  - Ready Answer Queue (task IDs with agent success)
  - Failed Task Queue (task IDs with agent failures)
- Provides thread-safe enqueue and dequeue operations.
- Provides requeue for retries.

### agent_worker_pool.py

- Launches and manages worker threads.
- Worker threads claim task IDs from Agent Task Queue.
- Each worker invokes `skunk_reasoner.solve(prompt)`.
- On success, stores `AnswerCandidate` and enqueues to Ready Answer Queue.
- On failure, stores `FailureRecord` and enqueues to Failed Task Queue.
- Posts completions to async event loop via thread-safe async queue.

### human_worker_registry.py

- Manages in-memory store of `HumanWorker` and `HumanWorkerSession` records.
- Creates new worker UUID for each client WebSocket connection.
- Handles display name updates.
- Tracks connection status.
- Produces mnemonic labels.

### human_work_broker.py

- Handles assignment creation when human claims a task.
- Processes retry requests with feedback.
- Processes direct human answer submissions.
- Validates assignment ownership and task version.
- Coordinates with TaskRegistry for atomic transitions.

### submission_coordinator.py

- Validates submission requests (deadline checks, protocol limits).
- Handles both manual submission and auto-submit configuration.
- Dispatches to CompetitionAdapter for Cup communication.
- Stores submission records on success or failure.

### server.py

- Composition root: creates all modules with dependency injection.
- Manages process startup: initializes queues, pools, adapters.
- Manages shutdown: graceful worker thread termination, cleanup.
- Configures and starts the FastAPI application.

## 11. Submission Rules

Every Cup `SubmitRequest` contains:

```text
answer_text: non-empty string
reasoning: string
source_docs: list[str]
submission_type: "agent" | "human"
```

Generated candidates use `submission_type=agent`. Direct human final answers
use `submission_type=human`. Human reasoning may be empty only if the Cup
protocol permits it. The server validates the protocol limits and never pads
or fabricates reasoning or sources.

## 12. Correctness and Security Invariants

- Cup credentials exist only in the Skunk Server process.
- Human workers require no authentication. Every client WebSocket connection
  creates a distinct ephemeral worker UUID.
- Clients cannot act without an active assignment for the task version.
- Assignments, attempts, and submissions are checked against the current task
  version.
- Submission is blocked after round closure or deadline expiry.
- The Competition Adapter is the only component that calls Cup APIs.
- The Submission Coordinator accepts either an explicit human command or the
  configured auto-submit policy.
- Ready and failed work may be assigned to multiple clients simultaneously.
- Only the first valid action for a task version may transition the task.
- UI assignment labels include the saved display name and a short UUID suffix;
  assignment ownership checks always use the full immutable UUID.
- Internal errors and tracebacks are sanitized before being sent to clients.
- In-memory only: state is lost on server restart. No SQLite or other
  persistence is implemented.

## 13. Implementation Status and Verification

### Passed Tests (8 focused unit/integration tests in harness_ui/tests/test_skunk_server.py)

1. First human action supersedes sibling: concurrent claims on the same READY task; first submit moves task to SUBMITTING and marks other assignments SUPERSEDED.
2. Retry and Cup feedback attach to next attempt: retry creates a new attempt whose feedback field is set; on next agent solve the prompt includes prior feedback.
3. Failed task accepts direct human answer: a FAILED task can be answered directly by a human, moving to SUBMITTING and then SUBMITTED.
4. Worker registry mnemonic names: a worker's display_name is saved via PUT; registry returns mnemonic labels like "Name (abcd)".
5. Agent worker pool success and failure: pool thread processes a task; on success a candidate is stored and task goes READY; on failure a failure record is stored and task goes FAILED.
6. API save-name/claim/retry: HTTP endpoints function correctly: PUT saves name, POST /api/assignments creates assignment, POST /api/retries enqueues retry.
7. Client page controls: the client HTML/JS renders task overview, claim button, submit/retry buttons, and display name form.
8. Auto-submit with immediate score feedback: an auto-submitted candidate transitions READY -> SUBMITTING -> SUBMITTED; the accepted response stores `correct` and `points_awarded` and appends Cup feedback, but the task remains SUBMITTED; only a later `submission_scored` WebSocket event transitions it to SCORED.

### Manual Verification

- Three-process dummy-agent run created five READY tasks: practice_server, skunk_server, and skunk_client all started correctly; the dummy agent produced candidates for each question.
- Manual worker display-name save, claim, and submission reached practice Cup: a human worker saved a display name, claimed a task, and submitted the agent candidate; the practice Cup accepted it and returned a score.
- Auto-submit run submitted all five and recorded immediate wrong-score feedback: with auto-submit enabled, all five tasks transitioned from READY to SUBMITTING to SUBMITTED, and the practice Cup returned scores (all wrong) which were stored.

### Module Layout

```text
harness_ui/
  cup_kit/
    client.py
    protocol.py
    practice_server.py

  skunk_server/
    server.py
    api.py
    events.py
    domain.py
    task_registry.py
    task_queues.py
    competition_adapter.py
    agent_worker_pool.py
    human_work_broker.py
    human_worker_registry.py
    submission_coordinator.py

  skunk_client/
    __init__.py
    client.py

  practice_server.py         # existing local Cup launcher
  skunk_reasoner.py          # existing reasoner adapter
  competition_console.py     # legacy monolithic implementation (retained)
```

The three-process architecture is launched by `serve_local_console.sh`:
`practice_server.py` (Cup substitute),
`python -m skunk_server.server` (new core),
and `python -m skunk_client.client` (UI process).
