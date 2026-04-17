"""Rich terminal progress display for :meth:`Execution.run`.

Provides a live-updating table that shows which operators are
executing, their item-level progress, running latency, and cost.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text


def _format_cost(usd: float) -> str:
    if usd < 0.01:
        return f"${usd:.4f}"
    return f"${usd:.2f}"


def _format_duration(secs: float) -> str:
    if secs < 60:
        return f"{secs:.1f}s"
    minutes = int(secs) // 60
    remaining = secs - minutes * 60
    return f"{minutes}m {remaining:.1f}s"


class OperatorProgress:
    """Mutable state for a single operator's progress row."""

    def __init__(self, node_id: str, display_name: str, items_total: int = 0) -> None:
        self.node_id = node_id
        self.display_name = display_name
        self.status: str = "waiting"  # waiting | running | done | skipped
        self.items_total = items_total
        self.items_done = 0
        self.cost_usd: float = 0.0
        self.start_time: float | None = None
        self.end_time: float | None = None

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return end - self.start_time

    def mark_running(self, items_total: int = 0) -> None:
        self.status = "running"
        self.start_time = time.perf_counter()
        if items_total:
            self.items_total = items_total

    def mark_done(self, cost_usd: float = 0.0, items_out: int | None = None) -> None:
        self.status = "done"
        self.end_time = time.perf_counter()
        self.cost_usd = cost_usd
        if items_out is not None:
            self.items_done = items_out

    def mark_skipped(self) -> None:
        self.status = "skipped"

    def increment_item(self) -> None:
        self.items_done += 1


class RichProgressDisplay:
    """Live Rich table showing operator execution progress.

    Representation invariant:
        - ``_operators`` keys are node IDs in plan topological order.
        - ``_global_start`` is set once on first call to :meth:`start`.

    Abstraction function:
        Represents a live terminal display tracking operator-level and
        item-level progress for a Carnot execution run.
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._operators: dict[str, OperatorProgress] = {}
        self._order: list[str] = []
        self._global_start: float = 0.0
        self._global_cost: float = 0.0
        self._live: Live | None = None

    # -- Setup ---------------------------------------------------------------

    def register_node(self, node_id: str, display_name: str) -> None:
        op = OperatorProgress(node_id, display_name)
        self._operators[node_id] = op
        self._order.append(node_id)

    def start(self) -> None:
        self._global_start = time.perf_counter()
        self._live = Live(self._build_table(), console=self._console, refresh_per_second=8)
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._refresh()
            self._live.stop()
            self._live = None

    # -- Updates -------------------------------------------------------------

    def mark_running(self, node_id: str, items_total: int = 0) -> None:
        op = self._operators.get(node_id)
        if op:
            op.mark_running(items_total)
            self._refresh()

    def mark_done(self, node_id: str, cost_usd: float = 0.0, items_out: int | None = None) -> None:
        op = self._operators.get(node_id)
        if op:
            op.mark_done(cost_usd, items_out)
            self._global_cost += cost_usd
            self._refresh()

    def mark_skipped(self, node_id: str) -> None:
        op = self._operators.get(node_id)
        if op:
            op.mark_skipped()
            self._refresh()

    def increment_item(self, node_id: str) -> None:
        op = self._operators.get(node_id)
        if op:
            op.items_done += 1
            self._refresh()

    def make_item_callback(self, node_id: str) -> Callable[[], None]:
        """Return a zero-arg callback that increments the item counter for *node_id*."""
        def _cb():
            self.increment_item(node_id)
        return _cb

    # -- Rendering -----------------------------------------------------------

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._build_table())

    def _build_table(self) -> Table:
        elapsed = time.perf_counter() - self._global_start if self._global_start else 0.0

        table = Table(
            title=f"Carnot Execution  |  Elapsed: {_format_duration(elapsed)}  |  Total Cost: {_format_cost(self._global_cost)}",
            show_lines=True,
            expand=True,
        )
        table.add_column("Step", style="bold", width=4, justify="right")
        table.add_column("Operator", min_width=20)
        table.add_column("Status", width=10, justify="center")
        table.add_column("Progress", width=18, justify="center")
        table.add_column("Duration", width=10, justify="right")
        table.add_column("Cost", width=10, justify="right")

        for i, node_id in enumerate(self._order):
            op = self._operators[node_id]
            step = str(i + 1)
            name = op.display_name

            # Status badge
            if op.status == "waiting":
                status = Text("waiting", style="dim")
            elif op.status == "running":
                status = Text("running", style="bold yellow")
            elif op.status == "done":
                status = Text("done", style="bold green")
            else:
                status = Text("skipped", style="dim italic")

            # Progress bar
            if op.status == "running" and op.items_total > 0:
                pct = op.items_done / op.items_total
                filled = int(pct * 10)
                bar = "█" * filled + "░" * (10 - filled)
                progress = Text(f"{bar} {op.items_done}/{op.items_total}")
            elif op.status == "done" and op.items_total > 0:
                progress = Text(f"{'█' * 10} {op.items_done}/{op.items_total}", style="green")
            elif op.status == "running":
                progress = Text("⣾ running...", style="yellow")
            else:
                progress = Text("-", style="dim")

            # Duration
            duration = _format_duration(op.elapsed) if op.status in ("running", "done") else "-"

            # Cost
            cost = _format_cost(op.cost_usd) if op.status == "done" and op.cost_usd > 0 else "-"

            table.add_row(step, name, status, progress, duration, cost)

        return table
