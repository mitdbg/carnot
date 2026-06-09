# OfficeQA Cup Local Console

`competition_console.py` is an optional local web UI for supervised
competition runs. It connects to the same Cup API as `reference_agent.py`,
starts your reasoning module for each incoming question, shows question and
answer status in the browser, and submits an answer only when you click that
question's Submit button.

## Serve Local Practice + Console

Use `serve_local_console.sh` to start both local processes:

```bash
cd /home/gerardo/carnot/skunk/harness_ui
./serve_local_console.sh
```

Defaults:

- Practice Cup API: `http://127.0.0.1:8765`
- Console UI: `http://127.0.0.1:8787`
- Questions: `officeqa_3x5_questions.json`
- Token: `anything`
- Reasoner: `skunk_reasoner:solve`
- Round length: `900` seconds

Open:

```text
http://127.0.0.1:8787
```

## Useful Launcher Flags

```bash
ROUND_SECONDS=120 ./serve_local_console.sh
QUESTIONS=practice_questions.json ./serve_local_console.sh
REASONER=dummy_agent:solve CONCURRENCY=5 ./serve_local_console.sh
CUP_PORT=9000 CONSOLE_PORT=8788 ./serve_local_console.sh
```

The `--reasoner` value is `module:function`. The function should follow the
same contract as `reference_agent.solve(prompt) -> AgentAnswer`.
`skunk_reasoner:solve` is the Skunk pipeline adapter. `dummy_agent:solve` is
included for UI testing; it waits 5-10 seconds and returns a random number.

## Run Against The Live Cup

For the real competition API, run only the console and point it at the live
endpoint:

```bash
export CUP_BASE_URL=https://<live-cup-url>
export CUP_TEAM_TOKEN=<your-team-token>
python competition_console.py --reasoner skunk_reasoner:solve --port 8787
```

Then open:

```text
http://127.0.0.1:8787
```

## Manual Flow

1. The console receives `round_started` events from the Cup API.
2. Each new question is shown in the browser.
3. The configured reasoner starts computing an answer.
4. Completed answers move to `ready`.
5. Click `Submit` to send the answer to the Cup API.
6. The UI updates when scoring feedback arrives.
