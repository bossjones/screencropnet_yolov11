"""Textual ``top`` dashboard: a live view of jobs in flight.

The data layer (:class:`TopSnapshot` + :func:`build_snapshot`) is pure — it turns
a :class:`StatusSummary` and a page of :class:`JobView`s into render-ready rows —
so it is unit-testable without mounting a terminal. :class:`TopApp` is the thin
Textual shell: it polls the existing async client on an interval, survives a down
server by showing an error banner, and redraws a summary line plus a job table.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header, Static

from screencropnet_yolo.server.schemas import JobView, StatusSummary

if TYPE_CHECKING:
    from screencropnet_yolo.client.api_client import ScreenCropClient


@dataclass(frozen=True)
class JobRow:
    """One render-ready row of the job table."""

    job_id: str
    status: str
    twitter: str
    pred_class: str
    path: str


@dataclass(frozen=True)
class TopSnapshot:
    """Everything the dashboard needs to render one frame."""

    batch: str
    total: int
    twitter_count: int
    throughput_per_sec: float
    status_counts: list[tuple[str, int]]
    jobs: list[JobRow]
    truncated: int


def build_snapshot(
    summary: StatusSummary, jobs: list[JobView], *, max_rows: int = 20
) -> TopSnapshot:
    """Fold a status summary and a page of jobs into a render-ready snapshot."""
    rows = [
        JobRow(
            job_id=job.job_id[:8],
            status=job.status,
            twitter="" if job.is_twitter is None else str(job.is_twitter),
            pred_class=job.pred_class or "",
            path=job.original_path,
        )
        for job in jobs[:max_rows]
    ]
    return TopSnapshot(
        batch=summary.batch_id or "all",
        total=summary.total,
        twitter_count=summary.twitter_count,
        throughput_per_sec=summary.throughput_per_sec,
        status_counts=sorted(summary.counts.items()),
        jobs=rows,
        truncated=max(0, len(jobs) - max_rows),
    )


class TopApp(App[None]):
    """Live job dashboard that refreshes ``client`` data every ``refresh_seconds``."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh now"),
    ]
    CSS = "#summary { height: auto; padding: 1 2; }"

    def __init__(
        self,
        client: ScreenCropClient,
        batch_id: str | None = None,
        refresh_seconds: float = 5.0,
        max_rows: int = 20,
    ) -> None:
        super().__init__()
        self.client = client
        self.batch_id = batch_id
        self.refresh_seconds = refresh_seconds
        self.max_rows = max_rows

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="summary")
        yield DataTable(id="jobs")
        yield Footer()

    async def on_mount(self) -> None:
        table = self.query_one("#jobs", DataTable)
        table.add_columns("job", "status", "twitter", "pred_class", "path")
        table.zebra_stripes = True
        await self.refresh_data()
        self.set_interval(self.refresh_seconds, self.refresh_data)

    def action_refresh(self) -> None:
        self.run_worker(self.refresh_data())

    async def _fetch_snapshot(self) -> TopSnapshot | str:
        """Pull status + jobs concurrently; return a snapshot, or an error string."""
        try:
            summary, jobs = await asyncio.gather(
                self.client.status(self.batch_id),
                self.client.list_jobs(batch_id=self.batch_id),
            )
        except Exception as exc:  # a down server must not kill the loop
            return f"{type(exc).__name__}: {exc}"
        return build_snapshot(summary, jobs, max_rows=self.max_rows)

    async def refresh_data(self) -> None:
        self._apply(await self._fetch_snapshot())

    def _apply(self, result: TopSnapshot | str) -> None:
        summary_widget = self.query_one("#summary", Static)
        table = self.query_one("#jobs", DataTable)
        if isinstance(result, str):
            summary_widget.update(
                f"[red]server unreachable: {result}[/red]  "
                f"(retrying every {self.refresh_seconds:g}s)"
            )
            return
        table.clear()
        for row in result.jobs:
            table.add_row(row.job_id, row.status, row.twitter, row.pred_class, row.path)
        counts = "  ".join(f"{name}={count}" for name, count in result.status_counts)
        more = f"  (+{result.truncated} more)" if result.truncated else ""
        summary_widget.update(
            f"batch=[bold]{result.batch}[/bold]  total={result.total}  "
            f"twitter={result.twitter_count}  "
            f"throughput={result.throughput_per_sec:.2f}/s\n{counts}{more}"
        )
