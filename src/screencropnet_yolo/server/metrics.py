"""Prometheus metrics for the API and worker.

Metric objects are module-level singletons so that constructing the FastAPI app
multiple times (as the test suite does) never re-registers a collector. Exact
job counts come from Postgres via ``/status``; these metrics are advisory.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app, start_http_server
from starlette.types import ASGIApp

JOBS_SUBMITTED = Counter(
    "screencrop_jobs_submitted_total", "Jobs accepted by the API", ["batch_id"]
)
JOBS_PROCESSED = Counter(
    "screencrop_jobs_processed_total", "Jobs processed by the worker", ["status"]
)
TWITTER_POSITIVE = Counter("screencrop_twitter_positive_total", "Twitter-positive classifications")
JOBS_IN_PROGRESS = Gauge("screencrop_jobs_in_progress", "Jobs currently being processed")
JOBS_BY_STATUS = Gauge("screencrop_jobs_by_status", "Jobs grouped by status", ["status"])
PRED_LATENCY = Histogram(
    "screencrop_pred_latency_seconds",
    "Classifier prediction latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)


def metrics_asgi_app() -> ASGIApp:
    """ASGI app exposing the Prometheus exposition; mount at ``/metrics``."""
    return make_asgi_app()


def start_worker_metrics_server(port: int) -> None:
    """Start a standalone exposition HTTP server for the worker process."""
    start_http_server(port)
