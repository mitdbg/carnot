from __future__ import annotations

import os

# ANSI color codes for terminal output
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"


class Tracer:
    """Streams the agent trace to a file and, optionally, to the terminal.

    Use as a context manager so the file is always closed on exit::

        with Tracer("trace.txt", show_output=True) as tracer:
            agent(question, tracer=tracer)
    """

    def __init__(self, trace_filepath: str, show_output: bool = False) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(trace_filepath)), exist_ok=True)
        self._file = open(trace_filepath, "w")  # noqa: SIM115
        self.show_output = show_output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_file(self, text: str) -> None:
        self._file.write(text + "\n\n")
        self._file.flush()

    def _print(self, text: str, ansi: str = "") -> None:
        if self.show_output:
            if ansi:
                print(f"{ansi}{text}{_RESET}", flush=True)
            else:
                print(text, flush=True)

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_system(self, content: str) -> None:
        """Write the system prompt to the file only (never to the terminal)."""
        self._write_file(f"## system\n{content}")

    def log_question(self, content: str) -> None:
        """Initial user question — normal text."""
        self._write_file(f"## user\n{content}")
        self._print(content)

    def log_assistant(self, content: str) -> None:
        """LLM-generated output — green."""
        self._write_file(f"## assistant\n{content}")
        self._print(content, _GREEN)

    def log_observation(self, content: str) -> None:
        """Successful tool output — yellow."""
        self._write_file(f"## user\n{content}")
        self._print(content, _YELLOW)

    def log_error(self, content: str) -> None:
        """Parse / execution / generation error — red."""
        self._write_file(f"## error\n{content}")
        self._print(content, _RED)

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Tracer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
