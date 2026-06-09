# UI Architecture

This document is the running record for the local OfficeQA Cup console UI.
Update it whenever `competition_console.py`, `skunk_reasoner.py`,
`dummy_agent.py`, `serve_local_console.sh`, or related console behavior
changes.

## Purpose

The console is a local supervised operator UI for the OfficeQA Cup API. It
connects to the practice or live Cup endpoint, receives round/question events,
dispatches questions to a local reasoning function, visualizes execution state,
and lets an operator submit answers manually.

## Files

- `competition_console.py` - FastAPI server, Cup API listener, local UI, and
  submit/retry endpoints.
- `skunk_reasoner.py` - adapter from the console reasoner contract to
  `skunk.Orchestrator`.
- `serve_local_console.sh` - concise Bash launcher for the local practice API
  plus the console UI.
- `dummy_agent.py` - test reasoner. `solve(prompt)` waits 5-10 seconds and
  returns a random number.
- `CONSOLE_README.md` - operator-facing usage instructions.
- `officeqa_3x5_questions.json` - 3 rounds of 5 OfficeQA questions generated
  from the first 15 rows of `skunk/officeqa_full.csv`.

## Runtime Shape

```text
browser UI
  <-> FastAPI console server
      <-> Cup API HTTP endpoints
      <-> Cup API WebSocket events
      -> local reasoner module:function
```

The reasoner is configured as `module:function` and must return an
`AgentAnswer`-compatible object with `answer`, `reasoning`, and `source_docs`.

## UI Layout

The browser UI is a master-detail layout:

- Top summary bar: Cup URL, connection status, round number, round status,
  resubmits left, and aggregate counters.
- Left panel: list of received questions. Each row shows round/question ID,
  prompt preview, and execution status.
- Right panel: selected question details. Shows full prompt, status, elapsed
  time, updated timestamp, answer, reasoning, source docs, scoring feedback,
  rejection/errors, and per-question Submit/Retry buttons.
- Left panel header includes Submit All, enabled only when at least one current
  question has a computed answer ready to submit.

## State Model

`ConsoleState.questions` stores `QuestionState` objects keyed by
`<round_num>:<question_id>`.

Question statuses:

- `queued` - received but not yet dispatched.
- `running` - reasoner is executing.
- `ready` - reasoner returned an answer.
- `failed` - reasoner raised.
- `submitting` - answer is being submitted.
- `submitted` - Cup API accepted the submission.
- `rejected` - Cup API rejected the submission payload or round state.
- `scored` - score event matched the submission.

## Round Lifecycle

When a new round starts, previous questions are cleared and in-flight reasoning
tasks are cancelled before new questions are inserted.

When the Cup API reports a non-active round state, the UI clears the question
list and cancels in-flight reasoning tasks. This prevents stale answers from a
closed round from remaining visible or being submitted after submissions are no
longer allowed.

## Console Endpoints

- `GET /` - embedded browser UI.
- `GET /api/state` - full console state snapshot.
- `POST /api/questions/{round_num}/{question_id}/retry` - rerun the reasoner for
  one question.
- `POST /api/questions/{round_num}/{question_id}/submit` - submit one computed
  answer.
- `POST /api/submit_all` - submit all current questions in `ready` or
  `rejected` state that have non-empty answers.
- `WS /ws` - browser state updates.

## Change Log

- Added `competition_console.py` as an independent local webserver and browser
  UI using the existing `cup_kit` client/protocol.
- Added manual per-question Submit and Retry buttons.
- Added `CONSOLE_README.md` so console usage docs stay separate from the
  official practice kit README.
- Added `officeqa_3x5_questions.json` with 3 rounds of 5 OfficeQA questions.
- Added `dummy_agent.py` for UI testing.
- Added `skunk_reasoner.py` and made `serve_local_console.sh` default to
  `skunk_reasoner:solve`; `dummy_agent:solve` remains available as an override.
- Replaced the initial Python dual-server launcher with concise
  `serve_local_console.sh`.
- Changed the UI from a card grid to a two-pane question list plus detail panel.
- Added round-close/new-round clearing behavior so previous-round questions are
  removed when submissions are no longer allowed.
- Added Submit All for all currently computed answers.
