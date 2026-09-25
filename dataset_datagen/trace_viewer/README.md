# Datagen Trajectory Viewer

A local, dependency-free viewer for `follow_up_questions.py` runs. It reconstructs, per seed
question, the whole generation trajectory and the retry loops inside it:

```
(1) seed question
 └─ (2) line-of-inquiry rollout ──► (3) unique?  ──✗──► back to (2)   [MAX_INQUIRY_TRIES]
        └─ (4) follow-up rollout (target #docs sampled, corpus searched, QA pair generated)
               ──► (5) solvable? (solver on the groundtruth docs, exact match or equivalence judge)
               ──► (6) unique? (duplicate judge vs. top-k existing QA pairs)
               ──✗──► back to (4)                                     [MAX_QA_PAIR_TRIES]
```

## Run

```bash
python3 dataset_datagen/trace_viewer/serve.py            # serves dataset_datagen/output on http://localhost:7072
python3 dataset_datagen/trace_viewer/serve.py --root /some/other/output --port 7072
```

Deep links work: `http://localhost:7072/#run=officeqa-dev&qid=UID0027`.

## What you see

- **Run overview** – funnel counts (seeds → inquiry rollouts → follow-up rollouts → unsolvable /
  duplicate rejections → accepted pairs), a per-question table (click a row to open it), and the
  duplicate-judge verdict counts.
- **Trajectory map** – one lane per stage, one node per attempt, `↺` between retries with the
  rejection reason. Click a node to jump to that attempt.
- **Attempt cards** – for every attempt: the agent rollout (tool calls with the doc ids each one
  surfaced, observations, LLM call lines), the generated output, the solver's answer side by side
  with the label, the equivalence verdict, the duplicate judge's verdict + reasoning and the top-k
  neighbours it saw. Doc-id chips open the document in a side drawer: for OfficeQA runs (when
  `qatfd/benchmarks/officeqa/treasury_bulletin_pdfs/` is on disk) that is the rendered source PDF
  page — zoom with the buttons, `+`/`-`/`0` keys, ctrl/⌘+wheel or pinch, drag to pan, `⇔` widens
  the drawer, and `open pdf ↗` opens the whole bulletin at that page in a new tab; `text` switches
  to the extracted page text (the only view for other benchmarks). Pages are rendered server-side
  with poppler's `pdftoppm` (falls back to PyMuPDF if installed).
- **Accepted QA pairs** – the chain written to `qa_pairs.json` for that question.

## How it is reconstructed

`trajectory.py` splits the flat per-question event stream (`traces/<qid>.jsonl`) into one segment
per agent run (each starts with a `system_prompt` event), types each segment from its
`call_site` / prompt, and aligns segments in order with the `generation_stats.jsonl` rows for that
question. Rows carry the recorded `solvable` / `unique` flags; segments carry the prompts, tool
calls and final JSON payloads. Questions without stats rows yet (still running, or old smoke
tests) fall back to a trace-only reconstruction that replays the generator's control flow.

Note: the duplicate judge returns the *id* of the duplicate, but `follow_up_questions.py` only
treats `duplicate is True` as a duplicate, so those verdicts never trigger a retry. The viewer
shows both the judge verdict and the recorded flag and marks such cases with ⚠.
