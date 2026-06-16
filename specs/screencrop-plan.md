# Plan: Local screenshot ingest/classify pipeline (CLI + async API + worker)

## Task Description

Build a local-only system that ingests a folder of 500+ mixed screenshots, classifies each as
"twitter screenshot or not" via an async FastAPI service backed by a RabbitMQ queue and a separate
worker process, tracks every submission as a job in Postgres, and lets a rich-colored Typer CLI:

1. see what was submitted,
2. see the status of submissions (pending → processing → done/failed),
3. see processing results,
4. pull back all twitter-positive results,
5. copy twitter-positive **originals** into the existing raw dataset using its naming scheme,
6. run everything backgrounded so the CLI never blocks,
7. write logs to disk and expose status at a FastAPI endpoint the CLI (and Claude) can poll,
8. compress images **losslessly** for fast upload (to `/tmp`), while copying the **real originals**
   (never the compressed files) to the destination.

Supporting services (Postgres, RabbitMQ, Prometheus, Grafana) run in Docker via `docker-compose`.
The Python API and worker run on the host. Metrics are exposed so progress on a 500-image run can be
monitored in Grafana and via a `/status` endpoint. All work is strict TDD.

This is the **ingest/classify front-end** of an existing dataset-modernization loop. The repo trains a
YOLO `tweet_region` detector from a raw set at
`scratch/datasets/twitter_screenshots_raw/train_images/` (named `NNNNN_twitter.EXT`, currently up to
`01494`). That set is mostly iPhone-X-era screenshots. We now have 500+ new screenshots from iPads,
iPhone 15s, etc. and want to fold the twitter ones in so the dataset (and model) generalizes across
modern device sizes. The full loop:

```
new screenshots → THIS SYSTEM classifies → twitter-positive originals copied into the raw dataset
(NNNNN_twitter naming) → Label Studio labeling → YOLO export → make train
```

Task type: **feature**. Complexity: **complex**.

## Objective

When complete, a developer can: `make services-up && make migrate`, run `make api` and `make worker`
on the host, then `screencrop-cli submit <folder>` to fire-and-forget classify a directory of
screenshots. They can watch progress with `screencrop-cli status --watch <batch_id>` (and in Grafana),
list twitter-positives, and `screencrop-cli export <batch_id>` to copy the twitter-positive originals
into `scratch/datasets/twitter_screenshots_raw/train_images/` as `01495_twitter.*`, `01496_twitter.*`,
… continuing the sequence. `make lint` and `make test` are green with no network, no real model
weights, and no GPU.

## Problem Statement

The naive approach (and the original draft prompt) ran model inference **synchronously inside
`POST /classify`** guarded only by a semaphore. With 500+ images the client blocks on every request —
this is exactly the "waiting on the API is too slow" problem. It also used SQLite as the only store
with no job lifecycle, lossy JPEG compression, and no observability. And it assumed a twitter/not
classifier, while this repo only ships a `tweet_region` **detector** plus a localization checkpoint —
neither directly answers "is this a twitter screenshot?".

We need:
- non-blocking submission (enqueue, return immediately, poll for results),
- an explicit job lifecycle persisted as the source of truth,
- lossless compression for fast transfer while preserving the true originals for the dataset,
- metrics/observability so a long run can be monitored,
- a real twitter/not signal from a classifier.

## Solution Approach

Adopt the decoupled pattern from the sibling repo `fastapi_pytorch_postgresql_sandbox`, modernized:

- **API** (`POST /classify`) validates + enforces max upload size, writes a `pending` job row to
  Postgres, publishes a small JSON message to RabbitMQ, and returns `202 {job_id}` immediately.
- **Worker** consumes from RabbitMQ, marks the job `processing`, runs the classifier off the event
  loop (`anyio.to_thread.run_sync`), and writes `done`/`failed` with the prediction back to Postgres.
- **Postgres is the source of truth** for submissions, status, and results (SQLAlchemy 2.0 async +
  asyncpg + Alembic). The CLI and Claude poll a `GET /status` aggregate endpoint that reflects exact
  Postgres counts; Prometheus/Grafana provide advisory live metrics.
- **Model**: replicate the reference **ScreenNet** classifier — EfficientNet-B0, classes
  `[facebook, tiktok, twitter]`, `infer(PIL) -> [{pred_prob, pred_class, time_for_pred}]`, device
  `mps>cuda>cpu`, weights at `~/Documents/my_models/ScreenNetV1.pth` (configurable, downloadable).
  `is_twitter = (pred_class == "twitter")`. Torch is imported only inside `load_model`/`infer`, behind
  a `Classifier` Protocol, so the API and the entire test suite run without torch, weights, or a GPU.
- **Compression**: lossless full-resolution WebP written to `/tmp/screencropnet_uploads/` before
  upload; the client uploads the WebP; the server stores the `original_path` for later export. The
  **export step copies the real original file**, never the WebP.
- **Export**: continue the dataset's `NNNNN_twitter.EXT` sequence from the max parsed index (note the
  set has gaps, so index is derived from `max(parsed NNNNN)`, not file count), preserving original
  extension/case, idempotent on `original_path`, collision-safe.
- **CLI**: Typer + rich; `submit` is fire-and-forget (a detached subprocess does the uploading; the
  CLI prints the `batch_id` and returns). Progress is always reconstructable from Postgres, so a
  killed CLI never loses state.

## Relevant Files

Existing files to read/respect (do not modify behavior):

- `pyproject.toml` — Python `>=3.11,<4.0`, hatchling, uv, ruff (line 100), basedpyright `recommended`,
  pytest with coverage + markers (`e2e/fast/integration/slow/unittest`); `addopts` already injects
  `-m "not integration"`, so `make test` is unit-only; `testpaths=[src,tests]`. Add new
  `[project.scripts]` and deps here.
- `Makefile` — existing targets (`install/lint/test/check/...`); append the new service/run targets.
- `.cursor/rules/python.mdc` and `.claude/rules/python-scripts.md` — conventions: absolute imports,
  `from __future__ import annotations`, modern unions (`str | None`, no `Optional`), `pathlib.Path`,
  `@override` from `typing_extensions`, `StrEnum`, pytest-mock `mocker` (never `unittest.mock`), tests
  in `tests/test_*.py`, inline tests under `## Tests` must not import pytest, `strif.atomic_output_file`
  for writes.
- `scratch/datasets/twitter_screenshots_raw/train_images/` — export destination; verified naming
  `NNNNN_twitter.EXT`, current max `01494` (374 files, sequence has gaps), mixed extensions/case.
- `src/screencropnet_yolo/inference.py` — `InferencePipeline.predict_image` (YOLO detector; NOT used by
  this feature, but the device-selection / cv2 `# pyright: ignore` patterns are worth mirroring).

Reference files in the sibling repo (read-only anchors for the design):

- `fastapi_pytorch_postgresql_sandbox/.../deeplearning/architecture/screennet/ml_model.py` —
  `ImageClassifier.infer()` returns `[{"pred_prob": float, "pred_class": str, "time_for_pred": float}]`;
  EfficientNet-B0 head `Dropout(0.2) -> Linear(1280, 3)`; `class_names = ["facebook","tiktok","twitter"]`.
- `fastapi_pytorch_postgresql_sandbox/.../worker.py` — `prefetch_count=1`, durable queue,
  `async with message.process()` auto-ack, RGBA→RGB. (We replace pickle with a JSON message.)
- `fastapi_pytorch_postgresql_sandbox/.../settings.py` — pydantic **v1** `BaseSettings` with inner
  `class Config: env_prefix`. (We migrate to pydantic-settings **v2** `SettingsConfigDict`.)
- `fastapi_pytorch_postgresql_sandbox/contrib/download-model.sh` — downloads `ScreenNetV1.pth`.

### New Files

Source (under `src/screencropnet_yolo/`):

- `server/__init__.py`
- `server/config.py` — `Settings` (pydantic-settings v2), `get_settings()`, `pick_device()`.
- `server/db.py` — SQLAlchemy 2.0 async engine/session, `ClassificationJob` model, `JobStatus`,
  repository functions.
- `server/schemas.py` — Pydantic v2 DTOs (`ClassifyAccepted`, `JobView`, `StatusSummary`,
  `QueueMessage`, `ExportRecord`).
- `server/classifier.py` — `Classifier` Protocol, `ScreenNetClassifier`, `FakeClassifier`,
  `is_twitter()`.
- `server/compression.py` — `compress_lossless_webp()`, `enforce_max_size()`.
- `server/export.py` — `current_max_index()`, `next_index()`, `export_originals()`.
- `server/queue.py` — `Publisher` Protocol, `RabbitPublisher`, `FakePublisher`.
- `server/metrics.py` — Prometheus metric objects + ASGI mount helper.
- `server/api.py` — `create_app()` factory, dependency providers, routes.
- `server/worker.py` — `handle_message()` (pure), `on_message()` (aio-pika), `run_worker()`, `main()`.
- `client/__init__.py`
- `client/api_client.py` — `ScreenCropClient`, `discover_images()`.
- `client/cli.py` — Typer app + `main()`.

Tests:

- `tests/conftest.py` — async fixtures (`sqlite_engine`, `session_factory`, `app`, `async_client`).
- `tests/server/test_config.py`, `test_compression.py`, `test_export.py`, `test_db.py`,
  `test_classifier.py`, `test_api.py`, `test_worker.py`, `test_discover.py`.
- `tests/client/test_api_client.py`, `test_cli.py`.
- `tests/integration/test_pipeline_integration.py`, `test_db_postgres.py` (marked `integration`).

Ops / infra:

- `docker-compose.yml` (repo root).
- `ops/prometheus.yml`.
- `ops/grafana/provisioning/datasources/prometheus.yml`.
- `ops/grafana/provisioning/dashboards/screencrop.yml` + `screencrop.json`.
- `alembic.ini` + `alembic/env.py` + `alembic/versions/0001_init.py`.
- `docs/development.md` — run order documentation.

## Implementation Phases

### Phase 1: Foundation
Add dependencies and tooling config; build the pure, dependency-light layers first (config,
compression, export, db against sqlite, classifier with mocked torch). No services required.

### Phase 2: Core Implementation
Wire the async API (enqueue → 202), the queue publisher, and the worker's pure `handle_message`. All
unit-tested with ASGITransport + sqlite + fakes. Add the Python client library and the Typer/rich CLI.

### Phase 3: Integration & Polish
Add `docker-compose` services, Alembic migrations, Prometheus metrics + Grafana provisioning, Makefile
targets, structured logging, and the integration test suite that exercises real Postgres + RabbitMQ.
Document the run order.

## Step by Step Tasks

IMPORTANT: Execute every step in order, top to bottom. Each implementation step is a TDD cycle:
write the failing test first, run it (confirm it fails for the right reason), write the minimum code to
pass, run the full unit suite, refactor on green, commit.

### 1. Dependencies & tooling config
- `uv add fastapi "uvicorn[standard]" "pydantic>=2" "pydantic-settings>=2" aio-pika "sqlalchemy[asyncio]>=2" asyncpg alembic httpx typer pillow anyio prometheus-client strif`
- `uv add --dev pytest-asyncio aiosqlite` (pytest-mock, pytest-timeout, rich already present).
- Make torch/torchvision an **optional extra** (worker-runtime only) so API + tests install without them:
  add a `[project.optional-dependencies] worker = ["torch", "torchvision"]` group.
- In `pyproject.toml` `[tool.basedpyright]`, append `aio_pika`, `asyncpg`, `prometheus_client` to
  `allowedUntypedLibraries` (mirror the existing `ultralytics`/`wandb` entries).
- In `[tool.pytest.ini_options]`, add `asyncio_mode = "auto"`.
- Add to `[project.scripts]`: `screencrop-cli = "screencropnet_yolo.client.cli:main"` and
  `screencrop-worker = "screencropnet_yolo.server.worker:main"` (keep existing
  `screencropnet_yolo = "screencropnet_yolo:main"`).

### 2. `server/config.py` (TDD: `tests/server/test_config.py`)
- Behavior to prove: `pick_device(["mps","cuda","cpu"])` returns `"cpu"` when torch is unavailable;
  env var `SCREENCROPNET_WEIGHTS_PATH` / `SCREENCROPNET_POSTGRES_DSN` override defaults.
- Implement `Settings(BaseSettings)`:
  ```python
  from pydantic_settings import BaseSettings, SettingsConfigDict
  class Settings(BaseSettings):
      model_config = SettingsConfigDict(env_prefix="SCREENCROPNET_", env_file=".env",
                                        env_file_encoding="utf-8", extra="ignore")
      postgres_dsn: str = "postgresql+asyncpg://screencrop:screencrop@localhost:5432/screencrop"
      rabbit_url: str = "amqp://guest:guest@localhost:5672/"
      worker_queue_name: str = "screennet_inference_queue"
      rabbit_prefetch_count: int = 8
      class_names: list[str] = ["facebook", "tiktok", "twitter"]
      arch: str = "efficientnet_b0"
      weights_path: Path = Path("~/Documents/my_models/ScreenNetV1.pth").expanduser()
      device_preference: list[str] = ["mps", "cuda", "cpu"]
      max_upload_bytes: int = 25 * 1024 * 1024
      compress_tmp_dir: Path = Path("/tmp/screencropnet_uploads")
      client_concurrency: int = 8
      raw_dataset_dir: Path = Path("scratch/datasets/twitter_screenshots_raw/train_images")
      export_label: str = "twitter"
      export_index_pad: int = 5
      api_host: str = "127.0.0.1"
      api_port: int = 8000
      worker_metrics_port: int = 8001
      logs_dir: Path = Path("logs")
  ```
- `get_settings()` is `lru_cache`-d. `pick_device(pref)` imports torch lazily and falls back to `"cpu"`.

### 3. `server/compression.py` (TDD: `tests/server/test_compression.py`)
- Behavior: `compress_lossless_webp(src, dst_dir)` opens with Pillow, writes a full-resolution lossless
  `.webp` into `dst_dir`, returns the new path, leaves the original file's bytes unchanged; RGBA is
  preserved (WebP supports alpha). `enforce_max_size(path, max_bytes)` raises `UploadTooLarge` over limit.
- Tests generate tiny images with `PIL.Image.new(...)` (no on-disk fixtures); verify the output is a
  valid re-openable WebP and the original is byte-identical before/after.

### 4. `server/export.py` (TDD: `tests/server/test_export.py`)
- Behavior: `current_max_index(dataset_dir, label="twitter", pad=5)` returns `1494` for a fake set with
  gaps; `next_index` returns `1495`. `export_originals(jobs, dataset_dir, ...)` copies each
  `job.original_path` to `dataset_dir/NNNNN_twitter.<orig-ext>` (preserving extension/case), is
  idempotent (skips originals already exported), collision-safe (probes next free index, never
  overwrites), and copies the **real original** (not the WebP). Uses `strif.atomic_output_file`.
  Returns `list[ExportRecord]`. Supports `dry_run=True`.
- Tests use `tmp_path` with synthetic `NNNNN_twitter.PNG` files (including gaps) and synthetic originals.

### 5. `server/db.py` (TDD: `tests/server/test_db.py`, on aiosqlite in-memory)
- Define `JobStatus(StrEnum)` = pending/processing/done/failed and the `ClassificationJob` model with
  portable column types only (`String`, `Float`, `Boolean`, `SAEnum(JobStatus)`,
  `DateTime(timezone=True)`, `func.now()`):
  ```python
  job_id (PK str/uuid4), batch_id (idx), original_path, status (idx, default pending),
  is_twitter (bool|None), pred_class (str|None), pred_prob (float|None),
  time_for_pred (float|None), error (str|None), created_at (server_default now),
  updated_at (server_default now, onupdate now)
  ```
- Implement `make_engine(dsn)`, `make_sessionmaker(engine)`, `async create_all(engine)` (tests only;
  prod uses Alembic), and repository helpers (all take an `AsyncSession`): `create_job`,
  `mark_processing`, `mark_done`, `mark_failed`, `get_job`, `list_jobs(batch_id, status)`,
  `list_twitter_positive(batch_id)`, `status_summary(batch_id) -> StatusSummary`.
- Behavior to prove: round-trip create→pending; each transition; `list_twitter_positive` filters to
  `status=done AND is_twitter=True`; `status_summary` returns correct counts + twitter_count.

### 6. `server/schemas.py` (no standalone cycle; introduced as needed by db/api/worker)
- `ClassifyAccepted{job_id, batch_id}`; `JobView` with `ConfigDict(from_attributes=True)` (build with
  `JobView.model_validate(row)` **inside** the session scope to avoid `DetachedInstanceError`);
  `StatusSummary{batch_id, total, counts: dict[str,int], twitter_count, done, failed,
  throughput_per_sec}`; `QueueMessage{job_id, batch_id, compressed_path, original_path}` (JSON body);
  `ExportRecord{original_path, dest_path, index, copied, reason}`.

### 7. `server/classifier.py` (TDD: `tests/server/test_classifier.py`, torch mocked)
- `Classifier` Protocol: `def infer(self, image: PIL.Image.Image) -> list[dict[str, object]]`.
- `is_twitter(result)` pure helper: `bool(result) and result[0]["pred_class"] == "twitter"`.
- `ScreenNetClassifier(settings)` — does NOT load weights in `__init__`; `load_model()` imports torch +
  torchvision and builds EfficientNet-B0 with head `Dropout(0.2) -> Linear(1280, len(class_names))`,
  `torch.load(weights_path, map_location=device)`, `model.eval()`, `torch.set_num_threads(1)`. `infer`
  applies the official EfficientNet-B0 transforms (resize/center-crop 224, ImageNet normalize), runs
  `torch.inference_mode()`, softmax+argmax, returns `[{pred_prob, pred_class, time_for_pred}]`;
  converts RGBA→RGB.
- `FakeClassifier(pred_class="twitter", pred_prob=0.99)` — deterministic, torch-free, for tests.
- Behavior to prove: `infer` returns the right dict shape and converts RGBA→RGB with all torch calls
  mocked; assert `torch.load` is NOT called unless `load_model()` was invoked.

### 8. `server/queue.py` (covered via api/worker tests; `FakePublisher` unit-tested inline)
- `Publisher` Protocol: `async def publish(self, msg: QueueMessage) -> None`.
- `RabbitPublisher(url, queue_name)` — `connect_robust`, declare durable queue, publish persistent JSON
  body with header `job_id`. `FakePublisher` — appends to an in-memory `published: list[QueueMessage]`.

### 9. `server/metrics.py` (introduced in api/worker cycles)
- `JOBS_SUBMITTED = Counter("screencrop_jobs_submitted_total", ..., ["batch_id"])`,
  `JOBS_PROCESSED = Counter("screencrop_jobs_processed_total", ..., ["status"])`,
  `TWITTER_POSITIVE = Counter("screencrop_twitter_positive_total", ...)`,
  `JOBS_IN_PROGRESS = Gauge("screencrop_jobs_in_progress", ...)`,
  `JOBS_BY_STATUS = Gauge("screencrop_jobs_by_status", ..., ["status"])`,
  `PRED_LATENCY = Histogram("screencrop_pred_latency_seconds", ..., buckets=(.05,.1,.25,.5,1,2,5))`.
- Helper to mount/expose Prometheus exposition on the API (`/metrics`) and a standalone exposition on
  the worker's `worker_metrics_port`.

### 10. `server/api.py` (TDD: `tests/server/test_api.py`, ASGITransport + overrides)
- `create_app(settings=None) -> FastAPI` factory; dependency providers `get_db_session` (async
  generator), `get_publisher`, `get_settings_dep` — all overridable via `app.dependency_overrides`.
  Mount `/metrics`; configure structured logging to `settings.logs_dir`.
- Routes:
  - `POST /classify` — multipart `file: UploadFile`, `original_path: Form[str]`,
    `batch_id: Form[str | None]`: `enforce_max_size`; `create_job(pending)`; publish `QueueMessage`;
    return `202 ClassifyAccepted`. (Server stores `original_path`; client already compressed.)
  - `GET /jobs/{job_id}` → `JobView` (404 if missing).
  - `GET /jobs?batch_id&status` → `list[JobView]`.
  - `GET /twitter?batch_id` → twitter-positive done jobs.
  - `GET /status?batch_id` → `StatusSummary` (exact Postgres aggregate).
  - `GET /healthz` → `{"ok": true}`.
- No torch import anywhere in this module; the classifier never runs here.
- Behavior to prove: `POST /classify` → 202 with `job_id`, exactly one `pending` row persisted, exactly
  one message published; oversize upload → 413; `/status` returns the documented JSON shape;
  `/jobs/{id}` 404 for unknown id.

### 11. `server/worker.py` (TDD: `tests/server/test_worker.py`, no aio-pika)
- `handle_message(body: bytes, *, classifier: Classifier, session_factory)` — pure core: parse
  `QueueMessage` → `mark_processing` → open `compressed_path` with Pillow → run
  `anyio.to_thread.run_sync(classifier.infer, image)` → on success `mark_done(is_twitter=...,
  pred_class, pred_prob, time_for_pred)` + increment metrics; on exception `mark_failed(error=...)` +
  `JOBS_PROCESSED{status="failed"}`.
- `on_message(message, *, classifier, session_factory)` — thin aio-pika wrapper:
  `async with message.process(): await handle_message(message.body, ...)`.
- `run_worker(settings)` — `connect_robust`, declare durable queue, `set_qos(prefetch_count=...)`,
  build `ScreenNetClassifier` + `load_model()`, start the worker `/metrics` exposition,
  `queue.consume(on_message)`, `await asyncio.Future()` to stay alive.
- `main()` = `asyncio.run(run_worker(get_settings()))`.
- Behavior to prove: `handle_message` writes a `done` row with correct `is_twitter` and increments
  metrics (use `FakeClassifier` + sqlite `session_factory`; spy on `anyio.to_thread.run_sync`); a
  classifier exception yields a `failed` row with the error string.

### 12. `client/api_client.py` + `discover_images` (TDD: `tests/client/test_api_client.py`, `test_discover.py`)
- `discover_images(folder, recursive)` — case-insensitive match of `IMAGE_EXTS` (png/jpg/jpeg/webp/
  bmp/gif/tiff), recursive vs flat, ignores non-images.
- `ScreenCropClient(base_url, client=None, concurrency=8, settings=None)`:
  `submit_image(original_path, batch_id)` (compress via `compress_lossless_webp` then POST multipart
  with the WebP + `original_path`), `submit_folder(folder, batch_id, recursive=True)` (discover +
  bounded by `asyncio.Semaphore(concurrency)`), `get_job`, `list_jobs`, `list_twitter`, `status`.
- Behavior to prove: `submit_image` uploads the **compressed** bytes (assert multipart payload is the
  webp and carries `original_path`); `submit_folder` finds images recursively and respects the
  semaphore; `status()` parses into `StatusSummary`. Tests inject
  `httpx.AsyncClient(transport=ASGITransport(create_app(...)))` with sqlite + `FakePublisher` overrides.

### 13. `client/cli.py` (TDD: `tests/client/test_cli.py`, Typer CliRunner + mocked client)
- Typer commands using rich tables/spinners:
  - `submit(folder, batch_id=None, recursive=True)` — fire-and-forget: launch a detached subprocess
    (`screencrop-cli _submit-worker --batch-id ... --folder ...`) that does the uploading, print the
    `batch_id`, and return immediately.
  - `submitted(batch_id=None)`, `results(batch_id=None)`, `twitter(batch_id=None)` — rich tables.
  - `status(batch_id=None, watch=False)` — rich live summary from `GET /status`.
  - `export(batch_id=None, dry_run=False)` — call `list_twitter` then `export_originals` into the raw
    dataset dir; print an `ExportRecord` table.
- `main()` = `app()`. Behavior to prove: each command calls the (mocked) client with the right args and
  renders output; `export` invokes `export_originals` with the resolved jobs + dataset dir.

### 14. docker-compose + ops config
- `docker-compose.yml` (services only, no Python containers):
  - `postgres:16-alpine` (5432; env `POSTGRES_USER/PASSWORD/DB=screencrop`; volume `pgdata`;
    healthcheck `pg_isready -U screencrop`).
  - `rabbitmq:3.13-management` (5672 + 15672 mgmt UI; volume `rabbitdata`; healthcheck
    `rabbitmq-diagnostics -q ping`).
  - `prom/prometheus:latest` (9090; mount `./ops/prometheus.yml`;
    `extra_hosts: ["host.docker.internal:host-gateway"]`).
  - `grafana/grafana:latest` (3000; anonymous admin; mount `./ops/grafana/provisioning`; volume
    `grafanadata`; `depends_on: [prometheus]`).
- `ops/prometheus.yml` — scrape `host.docker.internal:8000` (api) and `:8001` (worker) every 5s.
- `ops/grafana/provisioning/datasources/prometheus.yml` — datasource → `http://prometheus:9090`.
- `ops/grafana/provisioning/dashboards/screencrop.{yml,json}` — panels: `jobs_by_status` stacked,
  `twitter_positive_total`, processed rate `rate(screencrop_jobs_processed_total[1m])`, pred-latency
  p95 from the histogram.

### 15. Alembic migrations
- `alembic.ini` + async `alembic/env.py` reading `Settings.postgres_dsn` (use `run_sync` for the
  offline/online migration context). Generate initial migration `0001_init.py` from `Base.metadata`
  (the `db.py` model). Tests never run Alembic; they call `create_all` against sqlite.

### 16. Makefile targets
- `services-up` (`docker compose up -d` + `docker compose ps`), `services-down`
  (`docker compose down`), `services-logs` (`docker compose logs -f`), `migrate`
  (`uv run alembic upgrade head`), `api`
  (`uv run uvicorn screencropnet_yolo.server.api:create_app --factory --host 127.0.0.1 --port 8000`),
  `worker` (`uv run screencrop-worker`), `test-integration` (`uv run pytest -m integration`).
- `make test` stays unit-only (addopts already excludes `integration`).

### 17. Integration tests (marked `integration`)
- `tests/integration/test_pipeline_integration.py` — submit 50+ generated images through real
  Postgres + RabbitMQ + a worker (with `FakeClassifier` injected — still no GPU/weights); assert all
  reach `done`, `twitter_count` is correct, throughput > 0.
- `tests/integration/test_db_postgres.py` — re-run the repository suite against asyncpg/Postgres to
  catch enum / `server_default` divergence from sqlite.

### 18. Docs + structured logging
- `docs/development.md` — run order: `services-up → migrate → (api & worker) → submit → status/watch →
  export → test-integration`. Document env knobs (`SCREENCROPNET_*`), the ScreenNetV1.pth download, and
  the metrics/`/status` shapes.
- Confirm the API and worker write structured logs to `settings.logs_dir`.

### 19. Final validation
- Run the Validation Commands below; confirm `make lint` and `make test` are green; spin up services and
  smoke-test the end-to-end flow including a dry-run export.

## Testing Strategy

- **Unit (default `make test`, no services):** sqlite (aiosqlite) for db; `httpx.AsyncClient` +
  `ASGITransport` for the API; `app.dependency_overrides` to inject sqlite session + `FakePublisher` +
  `FakeClassifier`; `mocker` (pytest-mock) to patch torch in classifier tests; Typer `CliRunner` with a
  mocked `ScreenCropClient` for the CLI; `tmp_path` for compression/export filesystem behavior. No
  network, no real weights, no GPU.
- **Worker without RabbitMQ:** test the pure `handle_message(body, ...)` directly with bytes; the
  aio-pika `on_message` wrapper is exercised only in integration.
- **Integration (`@pytest.mark.integration`, excluded from `make test`):** real Postgres + RabbitMQ via
  `make services-up`; still injects `FakeClassifier` so no weights/GPU are needed. Postgres repository
  parity suite catches enum/server_default differences from sqlite.
- **Edge cases:** RGBA/PNG inputs (alpha preserved in WebP, RGBA→RGB in classifier); oversize upload →
  413; export sequence gaps (index from `max` not count), extension/case preservation, idempotent
  re-export, collision avoidance; classifier exception → `failed` job with error; `JobView`
  serialization inside session scope (no `DetachedInstanceError`); concurrency cap respected on
  `submit_folder` and worker prefetch; killed CLI → progress still reconstructable from Postgres.

## Acceptance Criteria

- `make lint` and `make test` are green with no skips/xfails; unit suite needs no network, no real
  weights, no GPU.
- `POST /classify` returns `202 {job_id}` immediately, persists exactly one `pending` job, and publishes
  exactly one queue message; oversize uploads return 413.
- The worker runs the classifier off the event loop and writes `done`/`failed` results to Postgres;
  `is_twitter = (pred_class == "twitter")`.
- The CLI can: submit a folder (fire-and-forget, prints `batch_id` and returns), list submitted jobs,
  show status (incl. `--watch`), show results, list twitter-positives, and export.
- `screencrop-cli export <batch_id>` copies the **real original** twitter-positive files into
  `scratch/datasets/twitter_screenshots_raw/train_images/` as `01495_twitter.*`, `01496_twitter.*`, …
  (continuing from the max parsed index), preserving extension/case, idempotently and collision-safely;
  the compressed `/tmp` WebPs are never copied to the dataset; originals are unmodified.
- `GET /status?batch_id` returns exact Postgres counts; Prometheus metrics are exposed by both API and
  worker; the Grafana dashboard shows status counts, twitter count, processed rate, and pred latency.
- `docker-compose` brings up Postgres, RabbitMQ, Prometheus, and Grafana (no Python containers); the
  integration suite passes against the real services.

## Validation Commands

Execute these to validate the task is complete:

- `uv sync --all-extras` — install all deps (incl. dev + worker extra).
- `uv run python -m py_compile $(git ls-files 'src/screencropnet_yolo/server/*.py' 'src/screencropnet_yolo/client/*.py')` — sanity compile new modules.
- `make lint` — ruff format + lint + basedpyright clean.
- `make test` — full unit suite green (integration auto-excluded via addopts).
- `make check` — `uv run ty check` clean.
- `make services-up && make migrate` — services healthy, schema applied.
- `make api` (terminal 1) and `make worker` (terminal 2) — both start without error.
- `uv run pytest -m integration` (or `make test-integration`) — integration suite green against real services.
- `screencrop-cli submit ./some_folder` then `screencrop-cli status --watch <batch_id>` — counts progress to done.
- `screencrop-cli export <batch_id> --dry-run` then real run — new `NNNNN_twitter.*` files appear in the raw dataset; re-running is idempotent.
- `curl -s localhost:8000/status` and `curl -s localhost:8000/metrics` — return the documented JSON / Prometheus exposition.

## Notes

- New libraries (install with uv):
  `uv add fastapi "uvicorn[standard]" "pydantic>=2" "pydantic-settings>=2" aio-pika "sqlalchemy[asyncio]>=2" asyncpg alembic httpx typer pillow anyio prometheus-client strif`
  and `uv add --dev pytest-asyncio aiosqlite`. Make `torch`/`torchvision` an optional `worker` extra.
- The real model needs `~/Documents/my_models/ScreenNetV1.pth` (EfficientNet-B0, 3-class). Provide a
  `contrib/download-model.sh` analog or document the download; tests never require it.
- Risks & mitigations:
  - Async SQLAlchemy without Postgres → aiosqlite + portable column types for units; parallel asyncpg
    integration suite for parity.
  - RabbitMQ in units → split pure `handle_message` from the aio-pika `on_message`; JSON body, not
    pickle (avoids version/security coupling and makes test construction trivial).
  - No weights/GPU in CI → torch imported lazily, `Classifier` Protocol seam, inject `FakeClassifier`.
  - pydantic v1→v2 → use `SettingsConfigDict` + `ConfigDict(from_attributes=True)`; do not copy the
    reference `class Config` blocks.
  - basedpyright strictness on async libs → extend `allowedUntypedLibraries`; rely on Protocol seams;
    use targeted `# pyright: ignore[...]` sparingly (as `inference.py` does for cv2).
  - Prometheus multiprocess → api:8000 and worker:8001 are separate processes/registries; pin the API to
    a single uvicorn worker; exact counts come from `/status` (Postgres), Prometheus is advisory.
  - Export correctness → index from `max(parsed NNNNN)` not file count; copy original not WebP;
    idempotent + collision-safe via `strif.atomic_output_file`.
  - `DetachedInstanceError` → build `JobView` with `model_validate(row)` inside the session scope.
  - Fire-and-forget submit must outlive the CLI → detached subprocess; state always rebuildable from
    Postgres.
- Conventions: absolute imports, `from __future__ import annotations`, modern unions, `pathlib.Path`,
  `StrEnum`, pytest-mock `mocker`, `strif.atomic_output_file`. Do NOT add backward-compat shims.
- Implement later via `/build` once this spec is confirmed.
