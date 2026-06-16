# Screenshot ingest/classify pipeline

A local-only system that classifies a folder of screenshots as "twitter or not",
tracks every submission as a Postgres job, and exports the twitter-positive
**originals** into the raw YOLO dataset (`scratch/datasets/twitter_screenshots_raw/train_images/`,
continuing the `NNNNN_twitter.EXT` sequence).

Architecture: the CLI compresses each image to a lossless WebP and uploads it to
the FastAPI service, which writes a `pending` job to Postgres and publishes a
small JSON message to RabbitMQ, returning `202` immediately. A separate worker
consumes the queue, runs the classifier off the event loop, and writes
`done`/`failed` back to Postgres. Postgres is the source of truth; Prometheus +
Grafana provide advisory live metrics.

## Run order

```bash
# 1. Supporting services (Postgres, RabbitMQ, Prometheus :9091, Grafana :3001).
make services-up

# 2. Apply the database schema.
make migrate

# 3. In separate terminals, run the host processes.
make api          # FastAPI on http://127.0.0.1:8000  (metrics at /metrics)
make worker       # RabbitMQ consumer (needs the worker deps + weights, see below)

# 4. Fire-and-forget submit a folder; the CLI prints a batch_id and returns.
uv run screencrop-cli submit ./some_folder

# 5. Watch progress (also visible in Grafana at http://localhost:3001).
uv run screencrop-cli status --watch <batch_id>
uv run screencrop-cli submitted --batch-id <batch_id>
uv run screencrop-cli twitter --batch-id <batch_id>

# 6. Export twitter-positive originals into the raw dataset (dry-run first).
uv run screencrop-cli export --batch-id <batch_id> --dry-run
uv run screencrop-cli export --batch-id <batch_id>

# 7. Tear down services when done.
make services-down
```

## Worker dependencies and weights

The classifier (torch/torchvision) lives in an opt-in dependency group so the
API and the test suite stay lean:

```bash
uv sync --group worker
```

The worker loads `ScreenNetV1.pth` (EfficientNet-B0, classes
`[facebook, tiktok, twitter]`) from `~/Documents/my_models/ScreenNetV1.pth` by
default. Download it (or set `SCREENCROPNET_WEIGHTS_PATH`):

```bash
mkdir -p ~/Documents/my_models
curl -L 'https://www.dropbox.com/s/im6ytahqgbpyjvw/ScreenNetV1.pth?dl=1' \
  > ~/Documents/my_models/ScreenNetV1.pth
```

`is_twitter = (pred_class == "twitter")`. The unit tests never need the weights
or a GPU (torch is imported lazily behind a `Classifier` protocol, and
`FakeClassifier` is injected).

## Configuration (env knobs)

All settings use the `SCREENCROPNET_` prefix (pydantic-settings v2; also read
from `.env`). Common overrides:

| Variable | Default |
| --- | --- |
| `SCREENCROPNET_POSTGRES_DSN` | `postgresql+asyncpg://screencrop:screencrop@localhost:5432/screencrop` |
| `SCREENCROPNET_RABBIT_URL` | `amqp://guest:guest@localhost:5672/` |
| `SCREENCROPNET_WORKER_QUEUE_NAME` | `screennet_inference_queue` |
| `SCREENCROPNET_WEIGHTS_PATH` | `~/Documents/my_models/ScreenNetV1.pth` |
| `SCREENCROPNET_MAX_UPLOAD_BYTES` | `26214400` (25 MiB) |
| `SCREENCROPNET_COMPRESS_TMP_DIR` | `/tmp/screencropnet_uploads` |
| `SCREENCROPNET_RAW_DATASET_DIR` | `scratch/datasets/twitter_screenshots_raw/train_images` |
| `SCREENCROPNET_CLIENT_CONCURRENCY` | `8` |
| `SCREENCROPNET_WORKER_METRICS_PORT` | `8001` |
| `SCREENCROPNET_LOGS_DIR` | `logs` |

The API and worker write structured logs to `SCREENCROPNET_LOGS_DIR`
(`api.log`, `worker.log`).

## Endpoints and metrics

- `POST /classify` — multipart `file` (the WebP) + `original_path` form field
  (+ optional `batch_id`); returns `202 {job_id, batch_id}`. Oversize → `413`.
- `GET /jobs/{job_id}` → `JobView` (404 if unknown).
- `GET /jobs?batch_id&status` → `list[JobView]`.
- `GET /twitter?batch_id` → twitter-positive done jobs.
- `GET /status?batch_id` → `StatusSummary`:
  `{batch_id, total, counts: {pending, processing, done, failed}, twitter_count,
  done, failed, throughput_per_sec}` (exact Postgres counts).
- `GET /healthz` → `{"ok": true}`.
- `GET /metrics` (API) and the worker's `:8001` exposition — Prometheus metrics:
  `screencrop_jobs_submitted_total`, `screencrop_jobs_processed_total{status}`,
  `screencrop_twitter_positive_total`, `screencrop_jobs_in_progress`,
  `screencrop_jobs_by_status{status}`, `screencrop_pred_latency_seconds`.

The provisioned Grafana dashboard (**ScreenCrop ingest/classify**) shows jobs by
status, twitter-positive total, processed rate, and prediction-latency p95.

## Export semantics

`export` copies the **real original** twitter-positive files (never the `/tmp`
WebP) into the raw dataset as `01495_twitter.*`, `01496_twitter.*`, … continuing
from the max parsed index (the set has gaps, so the next index comes from
`max(parsed)`, not the file count). Extension/case is preserved, and the export
is idempotent (a sidecar `.export_manifest.json` records what was copied) and
collision-safe (it probes the next free index, never overwriting).

## Tests

- `make test` — unit suite (no services, no torch, no GPU); integration is
  auto-excluded.
- `make test-integration` — runs against the live Postgres + RabbitMQ from
  `make services-up` (still injects `FakeClassifier`, so no weights/GPU needed).
