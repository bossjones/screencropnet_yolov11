"""Typer + rich CLI for the ingest/classify service.

``submit`` is fire-and-forget: it launches a detached subprocess that does the
uploading, prints the ``batch_id``, and returns immediately. All progress is
reconstructable from Postgres, so a killed CLI never loses state.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

from screencropnet_yolo.client.api_client import ScreenCropClient
from screencropnet_yolo.server.config import Settings, get_settings
from screencropnet_yolo.server.export import export_originals
from screencropnet_yolo.server.schemas import JobView, StatusSummary

app = typer.Typer(help="Submit screenshots for twitter/not classification and export results.")
console = Console()


def build_client(settings: Settings) -> ScreenCropClient:
    base_url = f"http://{settings.api_host}:{settings.api_port}"
    return ScreenCropClient(
        base_url=base_url, settings=settings, concurrency=settings.client_concurrency
    )


def _jobs_table(title: str, jobs: list[JobView]) -> Table:
    table = Table(title=title)
    table.add_column("job_id", overflow="fold")
    table.add_column("status")
    table.add_column("twitter")
    table.add_column("pred_class")
    table.add_column("original_path", overflow="fold")
    for job in jobs:
        table.add_row(
            job.job_id,
            job.status,
            "" if job.is_twitter is None else str(job.is_twitter),
            job.pred_class or "",
            job.original_path,
        )
    return table


def _render_status(summary: StatusSummary) -> None:
    table = Table(title=f"status (batch={summary.batch_id or 'all'})")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("total", str(summary.total))
    for status, count in summary.counts.items():
        table.add_row(status, str(count))
    table.add_row("twitter", str(summary.twitter_count))
    table.add_row("throughput/s", f"{summary.throughput_per_sec:.2f}")
    console.print(table)


@app.command()
def submit(
    folder: str = typer.Argument(..., help="Folder of screenshots to classify."),
    batch_id: str | None = typer.Option(None, "--batch-id"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive"),
) -> None:
    """Fire-and-forget: spawn a detached uploader and return the batch_id."""
    batch = batch_id or uuid4().hex
    cmd = [sys.argv[0], "_submit-worker", "--folder", folder, "--batch-id", batch]
    cmd.append("--recursive" if recursive else "--no-recursive")
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    console.print(f"submitted batch [bold]{batch}[/bold]; poll with: status --batch-id {batch}")


@app.command("_submit-worker", hidden=True)
def submit_worker(
    folder: str = typer.Option(..., "--folder"),
    batch_id: str = typer.Option(..., "--batch-id"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive"),
) -> None:
    """Internal: perform the actual uploading (run detached by ``submit``)."""
    client = build_client(get_settings())

    async def _go() -> None:
        try:
            await client.submit_folder(folder, batch_id, recursive=recursive)
        finally:
            await client.aclose()

    asyncio.run(_go())


@app.command()
def submitted(batch_id: str | None = typer.Option(None, "--batch-id")) -> None:
    """List submitted jobs."""
    client = build_client(get_settings())

    async def _go() -> None:
        try:
            jobs = await client.list_jobs(batch_id=batch_id)
        finally:
            await client.aclose()
        console.print(_jobs_table("submitted", jobs))

    asyncio.run(_go())


@app.command()
def results(batch_id: str | None = typer.Option(None, "--batch-id")) -> None:
    """Show processing results for jobs."""
    client = build_client(get_settings())

    async def _go() -> None:
        try:
            jobs = await client.list_jobs(batch_id=batch_id)
        finally:
            await client.aclose()
        console.print(_jobs_table("results", jobs))

    asyncio.run(_go())


@app.command()
def twitter(batch_id: str | None = typer.Option(None, "--batch-id")) -> None:
    """List twitter-positive results."""
    client = build_client(get_settings())

    async def _go() -> None:
        try:
            jobs = await client.list_twitter(batch_id=batch_id)
        finally:
            await client.aclose()
        console.print(_jobs_table("twitter-positive", jobs))

    asyncio.run(_go())


@app.command()
def status(
    batch_id: str | None = typer.Option(None, "--batch-id"),
    watch: bool = typer.Option(False, "--watch"),
) -> None:
    """Show status; with --watch, refresh until every job is done or failed."""
    client = build_client(get_settings())

    async def _go() -> None:
        try:
            while True:
                summary = await client.status(batch_id)
                _render_status(summary)
                if not watch or (summary.total and summary.done + summary.failed >= summary.total):
                    break
                await asyncio.sleep(1.0)
        finally:
            await client.aclose()

    asyncio.run(_go())


@app.command()
def export(
    batch_id: str | None = typer.Option(None, "--batch-id"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Copy twitter-positive originals into the raw dataset, continuing the sequence."""
    settings = get_settings()
    client = build_client(settings)

    async def _go() -> list[JobView]:
        try:
            return await client.list_twitter(batch_id=batch_id)
        finally:
            await client.aclose()

    jobs = asyncio.run(_go())
    records = export_originals(
        jobs,
        settings.raw_dataset_dir,
        label=settings.export_label,
        pad=settings.export_index_pad,
        dry_run=dry_run,
    )

    table = Table(title="export" + (" (dry run)" if dry_run else ""))
    table.add_column("index", justify="right")
    table.add_column("copied")
    table.add_column("reason")
    table.add_column("dest_path", overflow="fold")
    for record in records:
        table.add_row(str(record.index), str(record.copied), record.reason, record.dest_path)
    console.print(table)


def main() -> None:
    app()
