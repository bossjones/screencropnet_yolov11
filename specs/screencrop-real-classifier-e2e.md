# Plan: End-to-end CLI demo + real-classifier e2e tests for ScreenNet

## Task Description

The ingest/classify pipeline (Phases 1–3) is complete and verified with `FakeClassifier`, but nothing
yet exercises the **real** ScreenNet model. This task adds an end-to-end CLI demo and pytest e2e tests
that load the actual `ScreenNetV1.pth` weights and classify real Twitter screenshots through the worker
code path. It also makes the weights easy to obtain (a download script + `make` target), moves the
default weights location from `~/Documents/my_models/` to a **repo-local, overridable** path that
creates parent dirs, and bakes in the operational lessons from the earlier session where Docker wasn't
running.

Task type: **feature**. Complexity: **medium–complex**.

## Objective

A developer can run `make download-weights` to fetch `ScreenNetV1.pth` into a local (gitignored) path,
then `make test-e2e` to prove the real model classifies known Twitter screenshots as `twitter`, and
`make demo` to watch the full live stack (docker services → migrate → api+worker → CLI
submit/status/twitter/export) run end-to-end against the real classifier. `make lint` and `make test`
stay green with no weights, torch, GPU, or Docker.

## Problem Statement

- The real `ScreenNetClassifier` path is untested — `load_model()`/`infer()` have only been exercised
  with **mocked torch**. We don't actually know the downloaded checkpoint loads under **torch 2.9.1**,
  which now defaults `torch.load(..., weights_only=True)` (a breaking change vs. the reference repo).
- Weights live at a hard-coded `~/Documents/my_models/` path; the user wants a repo-local default that
  also honours a user-provided location and creates parent dirs.
- There's no one-command way to download the weights or to see the whole pipeline work with a real model.
- The previous session hit avoidable friction: Docker daemon down with a cryptic error, a stray
  `yolo26n.pt` artifact, and a `/metrics` trailing-slash redirect. These should be designed out.

## Solution Approach

1. **Repo-local, overridable weights path.** Default `Settings.weights_path` →
   `scratch/models/ScreenNetV1.pth` (`scratch/` is already gitignored). Add a `field_validator` that
   `expanduser()`s the value so `SCREENCROPNET_WEIGHTS_PATH=~/foo` works. Add `Settings.weights_url`
   (the Dropbox `dl=1` direct link) so the downloader is configurable. Parent-dir creation happens in
   the download script (`dest.parent.mkdir(parents=True, exist_ok=True)`).
2. **Download script + make target.** `scripts/download_screennet_weights.py` (PEP 723, `uv run`):
   streams the weights to `--dest` (default `get_settings().weights_path`), follows redirects, is
   idempotent, validates the file is a real `.pth` (size sanity, not an HTML error page), optional
   `--sha256`. `make download-weights` wraps it.
3. **Make `load_model()` torch-2.9-safe.** Load with `weights_only=False` (trusted local file) and
   tolerate a checkpoint that wraps the `state_dict` (`state_dict`/`model_state_dict` keys), with a
   clear error if `load_state_dict` rejects the keys. Confirm the actual structure on first download.
4. **Lightweight pytest e2e (no Docker).** Real classifier through the real worker code path on sqlite:
   prove `infer()` and `handle_message()` classify known Twitter screenshots as `twitter`. Marked
   `integration` + `e2e`, with skip-guards (`importorskip("torch")`, skip if weights absent) so
   `make test` and CI stay green.
5. **Full live-stack demo script.** `scripts/e2e_demo.py` orchestrates docker up → health-poll →
   migrate → background api+worker → CLI submit/status/twitter/export(dry-run) → teardown, with a
   **Docker preflight** and robust cleanup.
6. **Assertion strictness (chosen): assert correctness.** The classifier e2e runs the real model on K
   known Twitter screenshots and asserts **≥ 80%** classify as `twitter` (proves the model works, with
   a margin so one odd image doesn't flake).

## Relevant Files

Existing files to modify:

- `src/screencropnet_yolo/server/config.py` — change `weights_path` default to
  `scratch/models/ScreenNetV1.pth`; add `weights_url`; add a `field_validator` to `expanduser()` the
  path. (Line 29 today: `weights_path: Path = Path("~/Documents/my_models/ScreenNetV1.pth").expanduser()`.)
- `src/screencropnet_yolo/server/classifier.py` — `load_model()` (lines 38–54): make `torch.load`
  torch-2.9-safe (`weights_only=False`, tolerate wrapped state_dict, clear error on key mismatch).
- `Makefile` — append `download-weights`, `test-e2e`, `demo`; add a Docker preflight to `services-up`.
- `docs/screencrop-pipeline.md` — replace the old `~/Documents/my_models` + old-URL download block with
  the new local path + `make download-weights`; add an "End-to-end demo" section.
- `pyproject.toml` — (only if needed) keep `addopts -m "not integration"`; `e2e` marker already exists.
  No change expected — `-m e2e` on the CLI overrides the addopts marker filter.

Existing files/patterns to reuse:

- `scripts/setup_ls_project.py` — PEP 723 header + argparse pattern to mirror for the new scripts.
- `src/screencropnet_yolo/server/worker.py::handle_message` — the e2e worker test drives this directly
  with a real `ScreenNetClassifier` (no broker).
- `src/screencropnet_yolo/server/db.py` (`make_engine`, `make_sessionmaker`, `create_all`, repo helpers)
  and `tests/conftest.py` (`sqlite_engine`, `session_factory`) — reuse for the sqlite e2e.
- `src/screencropnet_yolo/server/classifier.py::is_twitter`, `ScreenNetClassifier` — the unit under test.
- `src/screencropnet_yolo/server/compression.py::compress_lossless_webp` — to stage a real screenshot as
  the worker's `compressed_path`.
- Health-poll + teardown pattern proven last session:
  `docker inspect --format '{{.State.Health.Status}}' <container>`.
- Dataset inputs: `scratch/datasets/twitter_screenshots_raw/train_images/00000_twitter.PNG …` (all
  twitter screenshots; use the first K as e2e/demo inputs).

### New Files

- `scripts/download_screennet_weights.py` — PEP 723; download weights to a local/overridable path.
- `scripts/e2e_demo.py` — PEP 723; orchestrate the full live-stack demo with preflight + teardown.
- `tests/integration/test_classifier_e2e.py` — real `ScreenNetClassifier` on K real screenshots.
- `tests/integration/test_worker_e2e_real.py` — `handle_message` with the real classifier on sqlite.

## Implementation Phases

### Phase 1: Weights plumbing (no model required)
Config default move + validator + `weights_url`; the download script; `make download-weights`. Unit
test the config changes (default path, env override, `~` expansion) — torch-free, stays in `make test`.

### Phase 2: Real-model loading + e2e tests
Harden `load_model()` for torch 2.9; add the two e2e tests (classifier + worker path, sqlite, skip-
guarded). Run them with weights present to prove the real model works.

### Phase 3: Live-stack demo + docs + lessons-learned hardening
`scripts/e2e_demo.py` (docker preflight, health-poll, background api/worker, CLI flow, teardown);
`make demo`; `services-up` Docker preflight; docs.

## Step by Step Tasks

IMPORTANT: Execute every step in order. TDD where unit-testable (config); the model-dependent pieces are
proven by the e2e tests + the live demo.

### 1. Config: local weights path + url + validator (TDD)
- In `server/config.py`: `weights_path: Path = Path("scratch/models/ScreenNetV1.pth")`;
  add `weights_url: str = "https://www.dropbox.com/scl/fi/8a5cc7e1ngcnm78kcqnga/ScreenNetV1.pth?rlkey=sbxats642fui9gpuwj8susha0&dl=1"`;
  add `@field_validator("weights_path", mode="after")` returning `v.expanduser()`.
- Update `tests/server/test_config.py`: assert new default is `Path("scratch/models/ScreenNetV1.pth")`,
  that `SCREENCROPNET_WEIGHTS_PATH=~/x.pth` expands to an absolute path, and `weights_url` ends `dl=1`.
- `make test` stays green (torch-free).

### 2. Download script + make target
- `scripts/download_screennet_weights.py` (PEP 723, deps `httpx`): args `--dest` (default
  `get_settings().weights_path`), `--url` (default `get_settings().weights_url`), `--force`,
  `--sha256`. `dest.parent.mkdir(parents=True, exist_ok=True)`; stream with
  `httpx.stream("GET", url, follow_redirects=True)`; write via temp file + atomic rename; skip if exists
  and `> 1 MiB` unless `--force`; **reject HTML** (content-type/`<html` sniff or size `< 1 MiB`) so a
  Dropbox error page never masquerades as weights; optional sha256 verify; print final path + size.
- `Makefile`: `download-weights: @uv run scripts/download_screennet_weights.py` (honours
  `SCREENCROPNET_WEIGHTS_PATH`). Document `--dest`/`--force` passthrough.

### 3. Harden `load_model()` for torch 2.9
- Replace `torch.load(self._weights_path, map_location=self._device)` with
  `torch.load(self._weights_path, map_location=self._device, weights_only=False)` (trusted local file;
  comment the why). If the result is a `dict` that wraps the params, select
  `state.get("state_dict") or state.get("model_state_dict") or state`. Wrap `model.load_state_dict` so a
  key mismatch raises a clear, actionable error (which keys were missing/unexpected → "checkpoint format
  mismatch; expected an EfficientNet-B0 3-class state_dict").
- **Verify on first real download**: inspect `type(checkpoint)` and keys, confirm the minimal correct
  load (adjust the wrapping logic only if the real file needs it).

### 4. Classifier e2e test (real model, correctness assertion)
- `tests/integration/test_classifier_e2e.py`, `@pytest.mark.integration` + `@pytest.mark.e2e`.
- Guards: `pytest.importorskip("torch")`; `settings = get_settings()`; skip if
  `not settings.weights_path.exists()` ("run `make download-weights`"); skip if the dataset dir is
  absent.
- Load `ScreenNetClassifier(settings)`, `load_model()`; run `infer()` on the first **K=10** real
  `*_twitter.PNG` screenshots. Assert every result has the 3 keys, `pred_class in class_names`,
  `0 ≤ pred_prob ≤ 1`, and **≥ 80%** have `is_twitter(...) == True`.

### 5. Worker-path e2e test (real classifier, sqlite, no Docker)
- `tests/integration/test_worker_e2e_real.py`, `@pytest.mark.integration` + `@pytest.mark.e2e`, same
  guards.
- Build an in-memory sqlite `session_factory` (reuse `make_engine(":memory:")` + `create_all`);
  `create_job(pending)` for a real screenshot; stage it as the worker input via
  `compress_lossless_webp(real_png, tmp)`; call
  `handle_message(QueueMessage(...).model_dump_json().encode(), classifier=ScreenNetClassifier(settings)
  (loaded), session_factory=...)`. Assert the job is `done` with `is_twitter is True` and a populated
  `pred_prob`/`time_for_pred`. (Proves the real model through the actual worker code path, no broker/PG.)

### 6. Live-stack demo script + make target
- `scripts/e2e_demo.py` (PEP 723): flags `--images N` (default 8), `--keep` (don't tear down).
  - **Preflight (lessons learned):** `docker info` reachable? else print "Start Docker Desktop (e.g.
    `open -a Docker`)" and exit 2. Check `torch` importable; check weights exist (offer
    `make download-weights`); check ports 8000/8001 free.
  - `docker compose up -d`; **poll** Postgres+RabbitMQ health via `docker inspect … Health.Status` until
    healthy or timeout; `uv run alembic upgrade head`.
  - Start `uvicorn …:create_app --factory` and `screencrop-worker` as background `subprocess.Popen`
    (worker loads the real model); wait for `/healthz`.
  - Stage K real screenshots into a temp dir; `screencrop-cli submit`; poll `screencrop-cli status`
    until done; `screencrop-cli twitter`; `screencrop-cli export --dry-run`; print a summary.
  - `try/finally`: terminate api/worker subprocesses; `docker compose down` unless `--keep`.
- `Makefile`: `demo: @uv run scripts/e2e_demo.py`.

### 7. Make `services-up` fail fast when Docker is down (lessons learned)
- Prepend a Docker preflight to `services-up` (and reuse in the demo): if `docker info` fails, echo a
  clear "Docker daemon not running — start Docker Desktop / `open -a Docker`" and exit non-zero, instead
  of the raw socket error seen last session.

### 8. Docs
- `docs/screencrop-pipeline.md`: replace the `~/Documents/my_models` + old-URL `curl` block with the new
  default (`scratch/models/ScreenNetV1.pth`), `make download-weights`, and `SCREENCROPNET_WEIGHTS_PATH`
  override; add an **End-to-end demo** section (`make download-weights && make demo`) and a
  `make test-e2e` note.

### 9. Make target + final validation
- `Makefile`: `test-e2e: @uv run pytest -m e2e` (CLI `-m e2e` overrides addopts' `-m "not integration"`).
- Run the Validation Commands; confirm `make lint`/`make test` green, then the weighted path:
  `make download-weights` → `make test-e2e` green → `make demo` completes.

## Testing Strategy

- **Unit (default `make test`, unchanged):** config changes are torch-free and covered in
  `test_config.py`; the model/demo code is import-guarded so collection never requires torch/weights.
  Suite stays green with no network/weights/GPU/Docker.
- **E2E (`-m e2e`, opt-in):** two tests, both skip-guarded (`importorskip("torch")` + weights-exist
  skip). `test_classifier_e2e` asserts the real model classifies ≥ 80% of K known Twitter screenshots as
  `twitter`. `test_worker_e2e_real` proves the real model through `handle_message` on sqlite — **no
  Docker required**. Marked `integration` too, so they never run in `make test`.
- **Live demo (`make demo`):** human-facing smoke of the whole stack with the real worker; not a pytest
  test but the truest end-to-end check; idempotent and self-cleaning.
- **Edge cases:** torch-2.9 `weights_only` checkpoint load; checkpoint that wraps `state_dict`; weights
  absent → friendly skip/instruction (never a hard failure); Docker down → friendly preflight; Dropbox
  returning an HTML error page → size/type guard; RGBA screenshots → already handled by `infer()`.

## Acceptance Criteria

- `make download-weights` fetches `ScreenNetV1.pth` into `scratch/models/` (or `--dest`/
  `SCREENCROPNET_WEIGHTS_PATH`), creating parent dirs, idempotently, and refuses non-`.pth` payloads.
- Default `Settings.weights_path` is repo-local and gitignored; `~`-based overrides expand correctly.
- `ScreenNetClassifier.load_model()` loads the real checkpoint under torch 2.9.1.
- `make test-e2e` is green with weights present: the real model classifies ≥ 80% of K Twitter
  screenshots as `twitter`, and the worker path produces a `done` + `is_twitter=True` job on sqlite.
- `make demo` runs the full live stack end-to-end (real classifier) and tears down cleanly; Docker-down
  yields a clear instruction, not a raw socket error.
- `make lint` and `make test` remain green with no weights/torch/GPU/Docker; no stray artifacts left in
  the working tree.

## Validation Commands

- `make lint` — ruff + basedpyright clean.
- `make test` — full unit suite green (e2e + integration auto-excluded).
- `uv run python -m py_compile scripts/download_screennet_weights.py scripts/e2e_demo.py` — scripts compile.
- `make download-weights` — weights land in `scratch/models/ScreenNetV1.pth` (re-run = idempotent skip).
- `SCREENCROPNET_WEIGHTS_PATH=./tmp/weights/ScreenNetV1.pth make download-weights` — parent dirs created.
- `make test-e2e` — real-classifier e2e green (≥ 80% twitter; worker path done+twitter).
- `make demo` — full live stack runs and tears down; `make demo --keep` leaves services up.
- (Docker down) `make services-up` — prints a clear "start Docker" message and exits non-zero.

## Notes

- **Weights URL:** the share link `…ScreenNetV1.pth?rlkey=…&st=…&dl=0` → use `…?rlkey=…&dl=1` (drop the
  ephemeral `st` token, set `dl=1`) and **follow redirects** (Dropbox 302s to `dl.dropboxusercontent.com`).
- **torch is already present** via ultralytics' main deps (torch 2.9.1 / torchvision 0.24.1), so e2e
  doesn't strictly need `uv sync --group worker`; the `importorskip` guard still protects lean installs.
- **Lessons baked in from the Docker-down session:** Docker preflight with actionable message; health-
  poll loop; artifact hygiene (weights under gitignored `scratch/`, demo temp under tmp; `yolo*.pt`
  already ignored); `/metrics` trailing-slash already fixed in `ops/prometheus.yml`; port checks before
  starting api/worker.
- New deps: none required (`httpx` already a dependency; PEP 723 scripts pin their own).
- Conventions: absolute imports, `from __future__ import annotations`, `pathlib.Path`, pytest-mock
  `mocker`, `strif.atomic_output_file` for the downloader's atomic write; no backward-compat shims.
