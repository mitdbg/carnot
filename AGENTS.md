## Environment Setup
- At the start of every session, run `source .envrc` to load environment variables.
- IF running Python scripts, ensure the current terminal is activated with the correct environment (e.g., `mamba activate carnot`).
- To make sure the correct Python environment is used in the session, run `which python` and confirm it points to the expected path (e.g., `/home/gerardo/.local/share/mamba/envs/carnot/bin/python`).

## LLM Delegation Tools (Token Saving)

### ask-llm — bulk reading
For reading files >400 lines, or when you'd otherwise read 3+ files:
  ask-llm --paths <file1> <file2>... --question "<question>"
Returns a structured summary. Use that instead of reading files yourself.

### llm-write — boilerplate generation
For tests, config files, docstrings, or repetitive patterns:
  llm-write --spec "<what>" --context <reference> --target <output>
Then review the output and edit only what needs fixing.

### When NOT to delegate
- Tasks under ~2000 tokens (delegation overhead isn't worth it)
- Architectural decisions, debugging, safety-critical code
- Anything requiring careful reasoning
- When exact line numbers are needed for editing

### Documentation workflow (MANDATORY)
**NEVER write documentation directly. Always delegate to llm-write.**

## Worker Delegation Rules

When asked to analyze, summarize, or search across multiple files:
DELEGATE to ask-llm with relevant file paths.

When asked to generate boilerplate, tests, or documentation:
DELEGATE to llm-write with appropriate reference files.

When asked to review session history:
DELEGATE to extract-chat.

DO NOT delegate:
- Architecture decisions
- Debugging complex logic
- Refactoring plans

## Code Generation Style

- No tiny one-use helper functions.
- Scripts are linear, readable narratives. Keep setup, data loading, execution, and output in visible order. 

## Example Script Guidelines
When writing example scripts, smoke tests, or single-use experimental code:

- Write linearly: setup -> data loading -> execution -> output, in visible order.
- Avoid tiny one-use helper functions; inline simple operations.
- Avoid argparse and `if __name__ == "__main__":` wrappers unless building a real reusable CLI.
- Use top-level editable constants (e.g., `DATA_DIR = "data"`) for paths and parameters.
- Prefer plain readable path strings over dense `pathlib` slash syntax when clarity improves.
- For output, just compute and print; no need for CLI-style main entry points.
- Preserve behavior while removing ceremony: fewer abstractions, straight-line logic.

