# Quickstart: the screenshot classify pipeline

Get the async ingest/classify pipeline running end to end — from a clean clone to
exporting twitter-positive screenshots — in about ten minutes.

This is the **classify** half of the repo (FastAPI + RabbitMQ worker + Postgres +
CLI). For the **training** half (the YOLO 26 tweet-region detector), see the main
[README](../README.md).

## What you'll build

You point the CLI at a folder of screenshots. Each image is compressed and uploaded
to a local FastAPI service, which records a job in Postgres and queues it on
RabbitMQ. A separate worker runs the classifier (EfficientNet-B0) off the event
loop and writes `done`/`failed` back to Postgres. You then export the
twitter-positive **originals** into the raw YOLO dataset. Postgres is the source of
truth; Prometheus + Grafana provide live metrics.

For the full architecture, endpoints, metrics, and export semantics, see
[screencrop-pipeline.md](screencrop-pipeline.md).

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — dependency management (this project does not
  use `pip`/`poetry`). See [installation.md](installation.md) if you don't have it.
- **Docker Desktop, running** — Postgres, RabbitMQ, Prometheus, and Grafana all
  come from `docker-compose.yml`.
- Python 3.11–3.13 (uv will fetch one if needed).

## Step 1 — Install (including the worker deps)

```bash
git clone https://github.com/bossjones/screencropnet_yolov11.git
cd screencropnet_yolov11
make install        # uv sync --all-extras
```

The classifier (torch/torchvision) lives in an opt-in `worker` dependency group so
the API and test suite stay lean. `make install` uses `--all-extras`, so the worker
deps are already included — you don't need a separate step.

## Step 2 — Download the model weights

```bash
make download-weights
```

This fetches `ScreenNetV1.pth` (EfficientNet-B0, classes
`[facebook, tiktok, twitter]`) into `scratch/models/ScreenNetV1.pth` (gitignored).
The download is idempotent and creates parent directories automatically. Override
the destination with `SCREENCROPNET_WEIGHTS_PATH`, and pass `ARGS=--force` to
re-download:

```bash
SCREENCROPNET_WEIGHTS_PATH=~/models/ScreenNetV1.pth make download-weights
```

## Step 3 — Start services and apply the schema

```bash
make services-up    # Postgres, RabbitMQ, Prometheus, Grafana
make migrate        # apply Alembic migrations to Postgres
```

Ports you'll use:

| Service | URL |
| --- | --- |
| FastAPI (started in step 4) | <http://127.0.0.1:8000> (metrics at `/metrics`) |
| RabbitMQ management UI | <http://localhost:15672> |
| Prometheus | <http://localhost:9091> |
| Grafana | <http://localhost:3001> |

## Step 4 — Run the API and worker

These are two separate processes — the HTTP layer is torch-free, while the worker
loads the model and runs inference. Run each in its own terminal:

```bash
make api            # terminal A: FastAPI on http://127.0.0.1:8000
make worker         # terminal B: RabbitMQ consumer (needs the weights from step 2)
```

## Step 5 — Submit, monitor, and export

`submit` is fire-and-forget: it spawns a detached uploader, prints a `batch_id`, and
returns immediately. All progress is reconstructable from Postgres, so a killed CLI
never loses state.

```bash
# Submit a folder; note the printed batch_id.
uv run screencrop-cli submit ./some_folder

# Watch progress until every job is done or failed (also visible in Grafana).
uv run screencrop-cli status --watch --batch-id <batch_id>

# List the twitter-positive results.
uv run screencrop-cli twitter --batch-id <batch_id>

# Export twitter-positive originals into the raw dataset (preview first).
uv run screencrop-cli export --batch-id <batch_id> --dry-run
uv run screencrop-cli export --batch-id <batch_id>
```

Other commands: `submitted` (list every submitted job) and `results` (show
processing results) — both accept `--batch-id`.

## Step 6 — Tear down

```bash
make services-down
```

## Try it without your own folder

`make demo` runs the whole thing non-interactively against a fresh stack — services
up, migrate, API + worker, CLI submit/status/twitter/export(dry-run), then teardown.
It runs a Docker preflight first, so a stopped daemon prints an actionable message
rather than a raw socket error.

```bash
make download-weights   # if you skipped step 2
make demo               # ARGS=--keep leaves services up; ARGS=--images N changes the count
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `make services-up` / `make demo` complains the Docker daemon is down | Start Docker Desktop, then retry. |
| `make worker` fails to load the model | Run `make download-weights` (step 2); confirm `scratch/models/ScreenNetV1.pth` exists or that `SCREENCROPNET_WEIGHTS_PATH` points at it. |
| Port 8000 (API) already in use | Set `SCREENCROPNET_API_PORT` to a free port before `make api`. |
| Worker metrics port 8001 already in use | Set `SCREENCROPNET_WORKER_METRICS_PORT` before `make worker`. |

## Where to go next

- [screencrop-pipeline.md](screencrop-pipeline.md) — deep dive: endpoints, Prometheus
  metrics, export semantics, the full `SCREENCROPNET_` config table, and tests.
- [README](../README.md) — the YOLO 26 training/evaluation/inference pipeline.
