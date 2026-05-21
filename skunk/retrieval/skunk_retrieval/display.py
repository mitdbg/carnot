import sys
from contextlib import contextmanager


def make_display(enabled: bool):
    if not enabled:
        return NullDisplay()
    try:
        return RichDisplay()
    except ImportError:
        print("Rich is not installed; continuing with JSON output only.", file=sys.stderr)
        return NullDisplay()


class NullDisplay:
    enabled = False

    def build_start(self, kind: str, data_dir: str, sources: str):
        pass

    def build_done(self, payload):
        pass

    def search_start(self, query: str, k: int):
        pass

    def search_done(self, results, elapsed: float):
        pass

    def oracle_done(self, payload, elapsed: float):
        pass

    @contextmanager
    def eval(self, total):
        yield self

    def eval_row(self, index, row, result, elapsed):
        pass

    def eval_done(self, summary, elapsed: float):
        pass


class RichDisplay:
    enabled = True

    def __init__(self):
        from rich.console import Console

        self.console = Console(stderr=True)
        self._eval_rows = []
        self._progress = None
        self._task_id = None

    def build_start(self, kind: str, data_dir: str, sources: str):
        from rich.panel import Panel

        self.console.print(
            Panel.fit(
                "[bold]{}[/bold]\ndata: {}\nsources: {}".format(kind, data_dir, sources),
                title="Build",
            )
        )

    def build_done(self, payload):
        table = self._kv_table("Build Summary", payload)
        self.console.print(table)

    def search_start(self, query: str, k: int):
        from rich.panel import Panel

        self.console.print(Panel.fit(query, title="Query", subtitle="top k = {}".format(k)))

    def search_done(self, results, elapsed: float):
        from rich.table import Table

        table = Table(title="Search Results ({:.3f}s)".format(elapsed))
        table.add_column("Rank", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Type")
        table.add_column("Source File")
        table.add_column("Page", justify="right")
        table.add_column("Snippet")

        for result in results[:20]:
            record = result["record"] if isinstance(result, dict) else result.record.to_dict()
            score = result["score"] if isinstance(result, dict) else result.score
            rank = result["rank"] if isinstance(result, dict) else result.rank
            table.add_row(
                str(rank),
                "{:.4f}".format(score),
                str(record.get("record_type") or ""),
                str(record.get("source_file") or ""),
                "" if record.get("page_id") is None else str(record.get("page_id")),
                self._clip(record.get("text") or "", 120),
            )

        self.console.print(table)

    def oracle_done(self, payload, elapsed: float):
        from rich.table import Table

        table = Table(title="Oracle ({:.3f}s)".format(elapsed))
        table.add_column("Source File")
        table.add_column("Pages")
        for source_file in payload.get("source_files", []):
            pages = payload.get("pages_by_source_file", {}).get(source_file, [])
            table.add_row(source_file, ", ".join(str(page) for page in pages))
        self.console.print(table)

    @contextmanager
    def eval(self, total):
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        self._eval_rows = []
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ]
        if total is None:
            columns.append(TextColumn("{task.completed} rows"))
        else:
            columns.extend([BarColumn(), TextColumn("{task.completed}/{task.total}")])
        columns.append(TimeElapsedColumn())
        progress = Progress(
            *columns,
            console=self.console,
        )
        self._progress = progress
        with progress:
            self._task_id = progress.add_task("Evaluating retrieval", total=total or None)
            yield self
        self._progress = None
        self._task_id = None

    def eval_row(self, index, row, result, elapsed):
        if self._progress and self._task_id is not None:
            description = "{} {}".format(row.uid or index, self._clip(row.question, 60))
            self._progress.update(self._task_id, advance=1, description=description)
        self._eval_rows.append((index, row, result, elapsed))

    def eval_done(self, summary, elapsed: float):
        from rich.table import Table

        overview = self._kv_table(
            "Eval Summary ({:.3f}s)".format(elapsed),
            {
                "total": summary.get("total"),
                "file_recall": summary.get("file_recall"),
                "page_total": summary.get("page_total"),
                "page_recall": summary.get("page_recall"),
                "k": summary.get("k"),
                "index_load_seconds": summary.get("index_load_seconds"),
                "elapsed_seconds": summary.get("elapsed_seconds"),
            },
        )
        self.console.print(overview)

        table = Table(title="Eval Rows")
        table.add_column("#", justify="right")
        table.add_column("UID")
        table.add_column("Oracle Files")
        table.add_column("Oracle Pages")
        table.add_column("File Rank", justify="right")
        table.add_column("Page Rank", justify="right")
        table.add_column("Latency", justify="right")
        table.add_column("Question")

        for index, row, result, row_elapsed in self._eval_rows[:50]:
            table.add_row(
                str(index + 1),
                row.uid,
                self._clip(", ".join(result.gold_files), 34),
                self._format_pages(result.gold_pages),
                "" if result.best_file_rank is None else str(result.best_file_rank),
                "" if result.best_page_rank is None else str(result.best_page_rank),
                "{:.3f}s".format(row_elapsed),
                self._clip(row.question, 80),
            )

        self.console.print(table)

    def _kv_table(self, title, payload):
        from rich.table import Table

        table = Table(title=title)
        table.add_column("Key")
        table.add_column("Value")
        for key, value in payload.items():
            table.add_row(str(key), str(value))
        return table

    def _format_pages(self, pages_by_source_file):
        parts = []
        for source_file, pages in pages_by_source_file.items():
            parts.append("{}:{}".format(source_file, ",".join(str(page) for page in pages)))
        return self._clip("; ".join(parts), 42)

    def _clip(self, text, width):
        text = " ".join(str(text).split())
        if len(text) <= width:
            return text
        return text[: max(0, width - 3)] + "..."
