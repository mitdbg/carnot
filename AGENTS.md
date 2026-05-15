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

## Implementation rules 
- Do not reuse modules and implemented from the main repository src/ folder. It is okay to copy and paste code snippets from the main repository, but do not import or reuse code directly. This is to ensure that the agent's code is self-contained and does not rely on external modules that may change over time.
- Do not write or change code in the main repository src/ folder. All code should be written and live in the 'skunk' folder. This is to ensure that the agent's code is separate and does not interfere with the main repository's codebase.
- Do not stray away from code changes that have been explicitly requested. If a change is requested, only make that change and no more. This is to ensure that the agent's code changes are focused and do not introduce unintended consequences. That includes not adding new features, not refactoring code unless explicitly requested, and not making any changes that are not directly related to the requested change. Always ask for clarification if a change request is unclear or if you are unsure about how to implement it.



## Code Generation Style

- No tiny one-use helper functions.
- No tiny one-use helper functions; inline simple operations instead.
- Code is linear, readable narratives. Keep setup, data loading, execution, and output in visible order.
- Use plain readable path strings for simple local repo paths (e.g., `DATA_DIR = "data"`) rather than `pathlib.Path(...)` when clarity improves. Reserve `pathlib` for cases involving complex path manipulation or cross-platform needs.
- Avoid creating single-use helper functions that obfuscate straightforward logic.
- Prefer top-level editable constants for paths and parameters.
- For scripts, avoid `argparse` and `if __name__ == "__main__":` wrappers unless building a real reusable CLI.
- For notebooks, maintain the same linear, ceremony-light structure: setup -> data loading -> computation -> output/visualization.
- When asked for changes, design a plan first, then ask the user to review the plan before generating code. This ensures alignment and avoids wasted tokens on unwanted code.