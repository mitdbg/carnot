# OfficeQA Cup — Team Practice Kit

Build and test your competition agent locally. The kit ships a
practice server that speaks the same protocol as the real cup, so the
agent you build here is the same agent you'll run on competition day.

## Install

You need Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Verify your install

```bash
python -m pytest tests/
```

If those tests pass, your dependencies are wired up and the practice
server boots cleanly. If anything fails, fix that before continuing.

## Build your agent

Open `reference_agent.py`. There are two things you'd customize:

**1. `solve(prompt) -> AgentAnswer`** — your agent. Given a question
prompt, return an `AgentAnswer`:

```python
AgentAnswer(
    answer="...",         # final answer text (see two valid forms below)
    reasoning="...",      # MUST be > 100 chars
    source_docs=["..."],  # optional list of citations; each ≤512 chars
)
```

Two valid answer forms — use whichever is easier:
- **Bare answer**: `"543 million"`
- **Wrapped**: `"...lots of reasoning... <FINAL_ANSWER>543 million</FINAL_ANSWER> ...more text..."`

The scorer extracts what's between the tags if present
(case-insensitive); otherwise it scores the whole string.

You can call any APIs, do any computation, and run any model from
inside `solve()`. The kit doesn't constrain how you answer — only the
response shape.

**2. `_process_round`** — the loop that runs your `solve()` over each
question in a round. Edit this if you want to answer questions in
parallel (`asyncio.gather`) or in a custom order. The default is
sequential.

Everything else in the kit you can leave alone. Rounds advance
automatically; there's no other knob worth touching.

## Practice locally

In one terminal, start the practice server:

```bash
python practice_server.py
```

In another terminal, run your agent against it:

```bash
export CUP_BASE_URL=http://127.0.0.1:8765
export CUP_TEAM_TOKEN=anything        # any non-empty value works locally
python reference_agent.py
```

The unmodified `reference_agent.py` always submits a placeholder, so
on the first run all 9 questions will score `correct=False`. That
confirms the connect → submit → score path works. Wire in your real
`solve()` and the same questions should start scoring `correct=True`.

The practice server runs through 3 rounds (180 seconds each by
default) then loops, so you can iterate as long as you need. Each
round window is fixed — like on competition day, late submissions
after the window closes are rejected. If your agent finishes early
or 180s is too long for it, use `--round-seconds N` to tune the
window length.

If your agent submits malformed work — reasoning under 100 chars,
empty `source_docs`, oversized payload, etc. — the practice server
rejects with the same reason the real cup would, so you can catch
those bugs locally.

### Useful flags

```bash
python practice_server.py --round-seconds 60   # shorter rounds for fast iteration
python practice_server.py --round-seconds 600  # longer rounds for slow agents
python practice_server.py --once               # one full cycle then exit
python practice_server.py --port 9000
```

### Practice with your own questions

The bundled `questions/practice_questions.json` is just a starter set. If you'd
rather practice against your own questions (harder, more on-topic for
your agent, like the OfficeQA benchmark itself), edit
`questions/practice_questions.json` or point the server at a separate file:

```bash
python practice_server.py --questions questions/my_questions.json
```

The file is a small JSON document with rounds and questions. Each
question needs an `id`, the `prompt` you want your agent to see, and
the `canonical_answer` the scorer compares against:

```json
{
  "rounds": [
    {
      "round_num": 1,
      "questions": [
        {
          "question_id": "my_q1",
          "prompt": "What was X in year Y?",
          "canonical_answer": "42"
        }
      ]
    }
  ]
}
```

`round_num` runs 1–10. The bundled file is a good template to copy 
from.

## Setting up a realistic practice competition

To simulate real competition conditions, use **6 rounds** of **15 questions**
each with **15-minute** windows:

1. Create a `questions/competition_questions.json` with **6 rounds**, **15 questions**
   per round (90 questions total), using your own OfficeQA-style
   prompts and canonical answers (you can take these from the OfficeQA benchmark).

2. Start the server with a longer round window:

```bash
python practice_server.py --questions questions/competition_questions.json --round-seconds 900
```

`--round-seconds 900` gives each round a 15-minute window, matching
real competition timing.

## On competition day

We'll give you two values to set as environment variables:

```bash
export CUP_BASE_URL=...
export CUP_TEAM_TOKEN=...
python reference_agent.py
```

That's the only thing about your setup that changes from local
practice. Your agent code is identical.

## What's in this kit

```
.
├── README.md                  # you are here
├── requirements.txt           # pip deps
├── reference_agent.py         # ← you edit AND run this (solve(), _process_round)
├── practice_server.py         # ← you run this in a second terminal during practice
├── questions/
│   └── practice_questions.json    # the 9 questions the practice server serves
├── cup_kit/                   # library bits (client, scorer, harness internals)
└── tests/                     # what `python -m pytest tests/` runs
```

## Questions

Reach out to the OfficeQA Cup operators on the channel you were
onboarded through.