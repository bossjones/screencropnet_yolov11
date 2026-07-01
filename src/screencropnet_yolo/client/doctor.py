"""Concurrent health checks for the ingest/classify stack.

``doctor`` probes every moving part — Postgres, RabbitMQ, Prometheus, Grafana,
the FastAPI API, and the worker(s) — at once. Each probe is an injectable async
callable returning a :class:`CheckResult`; they run under a single
``asyncio.gather`` with a per-probe timeout, so one hung service never blocks the
rest and the whole sweep finishes in roughly ``doctor_timeout`` seconds.

The concrete probes do the only real IO in this module; everything else
(aggregation, exit code, rendering) is pure and unit-tested directly.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass

import aio_pika
import httpx
from rich.table import Table
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from screencropnet_yolo.server.config import Settings


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one service probe."""

    name: str
    ok: bool
    detail: str
    latency_ms: float | None = None


# A probe does IO and returns a human-readable detail on success, or raises.
Probe = Callable[[], Awaitable[str]]


async def _run_check(name: str, probe: Probe, *, timeout: float) -> CheckResult:
    """Run ``probe`` under ``timeout``, turning any failure into a ``CheckResult``."""
    start = time.perf_counter()
    try:
        detail = await asyncio.wait_for(probe(), timeout)
        ok = True
    except TimeoutError:
        detail = f"timed out after {timeout:.1f}s"
        ok = False
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        ok = False
    latency_ms = (time.perf_counter() - start) * 1000
    return CheckResult(name=name, ok=ok, detail=detail, latency_ms=latency_ms)


async def check_http(
    name: str,
    url: str,
    *,
    timeout: float,
    expect_json_ok: bool = False,
    client: httpx.AsyncClient | None = None,
) -> CheckResult:
    """GET ``url`` and treat a 2xx (optionally ``{"ok": true}``) as healthy."""

    async def probe() -> str:
        owns = client is None
        c = client or httpx.AsyncClient()
        try:
            resp = await c.get(url)
            resp.raise_for_status()
            if expect_json_ok and not resp.json().get("ok"):
                raise RuntimeError(f"{url} did not report ok=true")
            return f"HTTP {resp.status_code}"
        finally:
            if owns:
                await c.aclose()

    return await _run_check(name, probe, timeout=timeout)


async def check_postgres(dsn: str, *, timeout: float) -> CheckResult:
    """Open an async engine and run ``SELECT 1``."""

    async def probe() -> str:
        engine = create_async_engine(dsn)
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return "SELECT 1 ok"
        finally:
            await engine.dispose()

    return await _run_check("postgres", probe, timeout=timeout)


async def check_rabbitmq(url: str, *, timeout: float) -> CheckResult:
    """Open (and immediately close) an AMQP connection."""

    async def probe() -> str:
        conn = await aio_pika.connect(url)
        await conn.close()
        return "connected"

    return await _run_check("rabbitmq", probe, timeout=timeout)


def default_checks(settings: Settings) -> list[Awaitable[CheckResult]]:
    """Build the standard probe set from ``settings`` (one coroutine per service)."""
    t = settings.doctor_timeout
    api_url = f"http://{settings.api_host}:{settings.api_port}/healthz"
    return [
        check_postgres(settings.postgres_dsn, timeout=t),
        check_rabbitmq(settings.rabbit_url, timeout=t),
        check_http("api", api_url, timeout=t, expect_json_ok=True),
        check_http("worker", settings.worker_metrics_url, timeout=t),
        check_http("prometheus", settings.prometheus_url, timeout=t),
        check_http("grafana", settings.grafana_url, timeout=t),
    ]


async def run_doctor(
    settings: Settings, *, checks: Sequence[Awaitable[CheckResult]] | None = None
) -> list[CheckResult]:
    """Run every check concurrently and collect the results."""
    coros = default_checks(settings) if checks is None else checks
    return list(await asyncio.gather(*coros))


def exit_code(results: list[CheckResult]) -> int:
    """0 when every check passed, 1 otherwise."""
    return 0 if all(r.ok for r in results) else 1


def render_table(results: list[CheckResult]) -> Table:
    """A rich table of ✔︎/✘ status, latency, and detail per service."""
    table = Table(title="doctor")
    table.add_column("service")
    table.add_column("status", justify="center")
    table.add_column("latency", justify="right")
    table.add_column("detail", overflow="fold")
    for r in results:
        glyph = "[green]✔︎[/green]" if r.ok else "[red]✘[/red]"
        latency = "" if r.latency_ms is None else f"{r.latency_ms:.0f} ms"
        table.add_row(r.name, glyph, latency, r.detail)
    return table


def render_json(results: list[CheckResult]) -> str:
    """Serialize results as indented JSON (for ``doctor --json``)."""
    return json.dumps([asdict(r) for r in results], indent=2)
