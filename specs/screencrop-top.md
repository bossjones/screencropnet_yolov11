# Plan: `top` TUI, `doctor` health check, fuzzy-selected `serve` (+ pyinstrument)

## Task Description
Extend the screencrop ingest/classify stack (from `specs/screencrop-plan.md`, already
implemented under `src/screencropnet_yolo/server/` and `client/`) with operator tooling:

- a live async **Textual** TUI (`screencrop-cli top`) that shows jobs in flight and refreshes on an interval,
- a fully concurrent service-health command (`screencrop-cli doctor`),
- a `serve` launcher that **fuzzy-picks the model weights** before booting the API (borrowing the
  `discover_models`/`select_model`/`_fzf_select` logic from `specs/fuzzy-model-select-demo.md`, already
  shipped in `demo.py`),
- and, as a reach goal, [`pyinstrument`](https://github.com/joerick/pyinstrument) profiling with Makefile targets.

Everything is async and maximally concurrent, and is built **test-first (TDD)**.

## Objective
When this plan is complete:
- `screencrop-cli serve --select` fzf-picks a weights file and boots the API (and optionally the worker) against it.
- `screencrop-cli top --refresh 5` shows a live Textual dashboard of job status, refreshing on the given interval (default 5 s).
- `screencrop-cli doctor` concurrently reports ✔︎/✘ for postgres, rabbitmq, prometheus, grafana, the FastAPI server, and the worker(s), exiting non-zero if any critical check fails.
- `make profile-*` targets produce openable pyinstrument reports.
- `make lint`, `make test`, and `make check` are clean.

## Problem Statement
Operators currently poll `status --watch` (a plain reprinting loop with no interactivity), have no single
command to confirm the whole stack is up, and must hand-edit `SCREENCROPNET_WEIGHTS_PATH` to switch which
model the worker loads. There is also no profiling entry point for the hot paths (inference, IO fan-out).

## Solution Approach
- **Reuse, don't duplicate.** The model-discovery/fzf helpers already exist in `demo.py`. Extract them into a
  shared, torch-free `model_select.py`; `demo.py` re-imports them (no behavior change). `serve` calls the same
  `select_model` over broader search roots and the `.pth` extension used by the ScreenNet classifier.
- **`top`** is a Textual `App` driven by `set_interval(refresh_seconds, …)`, whose async callback calls the
  existing `ScreenCropClient`. A pure `build_snapshot()` function converts `StatusSummary` + `list[JobView]`
  into render-ready rows, so the data layer is unit-testable without a running terminal.
- **`doctor`** models each check as an injectable async coroutine returning a `CheckResult`; all checks run
  under `asyncio.gather` with per-check `asyncio.wait_for` timeouts. Pure aggregation/rendering is unit-tested;
  the probes are mocked.
- All new IO uses the async stack already present: `httpx.AsyncClient`, `aio_pika`, SQLAlchemy async.

### Key facts this design relies on (verified against the current code)
- CLI is **Typer** at `src/screencropnet_yolo/client/cli.py`; `main()` → `app()`. `build_client(settings)`
  returns a `ScreenCropClient`; `console = Console()` (rich). Existing commands: `submit`, `_submit-worker`
  (hidden), `submitted`, `results`, `twitter`, `status` (`--watch`), `export`.
- `ScreenCropClient` (`client/api_client.py`) already exposes `status()`, `list_jobs()`, `list_twitter()`,
  `get_job()` — `top` reuses these, **no new client methods required**.
- DTOs (`server/schemas.py`):
  `StatusSummary{batch_id, total, counts: dict[str,int], twitter_count, done, failed, throughput_per_sec}`
  and `JobView{job_id, batch_id, original_path, status, is_twitter, pred_class, pred_prob, time_for_pred, error}`.
- Fuzzy helpers currently in `demo.py`: `MODEL_EXTS = {".pt", ".onnx"}`,
  `ModelSelector = Callable[[list[str]], list[str]]`, `discover_models`, `format_model_choice`, `select_model`,
  `_fzf_select` (lazy `pyfzf` import); plus `human_size` in `output.py`.
- Weights are loaded by the **worker's** `ScreenNetClassifier(settings)` from `settings.weights_path` (a `.pth`);
  the API only enqueues. `serve` drives the model via the `SCREENCROPNET_WEIGHTS_PATH` env var so both the API
  and a co-launched worker see it. `get_settings()` is `@lru_cache`, so set the env **before** the first
  `Settings()` construction (or call `get_settings.cache_clear()`).
- Service host ports (`docker-compose.yml`): postgres `5432`, rabbitmq `5672` + management `15672`,
  prometheus **`9091`**→9090, grafana **`3001`**→3000. API exposes `/healthz`→`{ok}` and `/metrics`; the worker
  exposes metrics on `worker_metrics_port` (default `8001`).
- Tests: pytest + `pytest-mock` (`mocker`) + `pytest-asyncio` (`asyncio_mode="auto"`) + `aiosqlite`;
  `tests/client/` and `tests/server/` subdirs exist. Pure helpers get inline `## Tests` blocks (no pytest
  import); interactive/IO bits get files under `tests/`.

## Relevant Files
Existing (read/respect; reuse their helpers — do not re-implement):
- `src/screencropnet_yolo/demo.py` — source of `discover_models`/`format_model_choice`/`select_model`/`_fzf_select`/`MODEL_EXTS`/`ModelSelector`.
- `src/screencropnet_yolo/output.py` — `human_size`, `colorize`, `Color` for formatting.
- `src/screencropnet_yolo/client/api_client.py` — `ScreenCropClient` (`status`, `list_jobs`, `list_twitter`).
- `src/screencropnet_yolo/client/cli.py` — Typer app to extend; `build_client`, `_jobs_table`, `_render_status` patterns.
- `src/screencropnet_yolo/server/config.py` — `Settings`, `get_settings` (`@lru_cache`).
- `src/screencropnet_yolo/server/schemas.py` — `StatusSummary`, `JobView`.
- `src/screencropnet_yolo/server/worker.py`, `src/screencropnet_yolo/server/api.py` — endpoints/ports referenced by `doctor`.
- `docker-compose.yml` — authoritative host ports (prometheus 9091, grafana 3001).
- `Makefile` — target style (`@echo "🚀 ..."` + `@uv run ...`, `## help` comments).

### New Files
- `src/screencropnet_yolo/model_select.py` — extracted shared fuzzy-model helpers (torch-free; lazy `pyfzf`).
- `src/screencropnet_yolo/client/tui.py` — Textual `TopApp` + pure `build_snapshot()` + `TopSnapshot`.
- `src/screencropnet_yolo/client/doctor.py` — async `CheckResult`, per-service checks, `run_doctor()`, renderer.
- `tests/test_model_select.py` — extraction + `.pth`/multi-root behavior.
- `tests/client/test_tui.py` — `build_snapshot()` + mocked `TopApp` refresh.
- `tests/client/test_doctor.py` — each check (mocked IO), aggregation, exit-code summarizer.
- `tests/client/test_serve.py` — weights resolution + env export (process launch mocked).
- `docs/top.md`, `docs/doctor.md`, `docs/serve.md` — usage docs.

## Implementation Phases
### Phase 1: Foundation
Extract `model_select.py` (keep `demo.py` importing from it — full suite green before moving on). Add the new
`Settings` fields. Add dependencies: `uv add textual`, `uv add --dev pyinstrument`.

### Phase 2: Core Implementation
TDD `doctor` (pure checks → concurrent runner), then `top` (`build_snapshot` → `TopApp`), then `serve`
(weights resolution → env export → launcher).

### Phase 3: Integration & Polish
Wire all three into the Typer app, add Makefile targets (`serve`, `top`, `doctor`, `profile-*`), the optional
pyinstrument FastAPI middleware gated by `SCREENCROPNET_PROFILE`, docs, and final validation.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom. Each feature is test-first: write the failing test, run
it red, implement, run green, refactor.

### 1. Dependencies & config
- `uv add textual`; `uv add --dev pyinstrument`. (`pyfzf`, `httpx`, `aio-pika` are already present.)
- Add to `Settings` (`server/config.py`):
  - `model_search_roots: list[Path] = [Path("runs"), Path("scratch/models")]`
  - `prometheus_url: str = "http://127.0.0.1:9091/-/healthy"`
  - `grafana_url: str = "http://127.0.0.1:3001/api/health"`
  - `rabbit_mgmt_url: str = "http://127.0.0.1:15672/"`
  - `worker_metrics_url: str = "http://127.0.0.1:8001/"` (or derive the default from `worker_metrics_port` in a validator)
  - `doctor_timeout: float = 2.0`

### 2. Extract `model_select.py` (reuse, no dup)
- Write `tests/test_model_select.py` first: `discover_models` finds `.pt/.onnx/.pth`, newest-first, `[]` on a
  missing dir; `select_model` maps display→Path and returns `None` on cancel (empty selection);
  `format_model_choice` line shape (`name : /abs/path  [size]`).
- Move `MODEL_EXTS`, `ModelSelector`, `discover_models`, `format_model_choice`, `select_model`, `_fzf_select`
  into `model_select.py`. Parametrize `discover_models(search_root, exts=MODEL_EXTS)` and add
  `SERVER_MODEL_EXTS = {".pt", ".onnx", ".pth"}`.
- Update `demo.py` to `from screencropnet_yolo.model_select import ...` (behavior unchanged; its inline
  `## Tests` still pass). Run the full suite green before continuing.

### 3. `doctor` — pure checks (TDD)
- Write `tests/client/test_doctor.py` first. Define `CheckResult(name: str, ok: bool, detail: str, latency_ms: float | None)`.
- Implement async checks in `doctor.py`, each injectable for testing:
  - `check_http(name, url, *, expect_json_ok=False)` using `httpx.AsyncClient` → API `/healthz` (`{ok: true}`),
    prometheus, grafana, worker metrics.
  - `check_postgres(dsn)` — SQLAlchemy async `SELECT 1` (fallback: `asyncio.open_connection` TCP to host:5432).
  - `check_rabbitmq(url)` — `aio_pika.connect_robust` then close (fallback: TCP `5672`).
  - `check_worker()` — HTTP GET `worker_metrics_url` (chosen worker-detection method: scrape the metrics port).
- Wrap each in `asyncio.wait_for(..., settings.doctor_timeout)`; convert timeout/exception into a
  `CheckResult(ok=False, detail=...)` rather than raising. Record `latency_ms`.

### 4. `doctor` — concurrent runner + render (TDD)
- `run_doctor(settings, *, checks=None) -> list[CheckResult]`: build the check coroutines and
  `await asyncio.gather(*checks)` so they all run concurrently. Allow injecting fake checks in tests.
- Renderer: a rich `Table` with ✔︎ (green) / ✘ (red) per the project glyph convention plus latency; a `--json`
  branch dumps the `CheckResult`s. Provide a pure summarizer `exit_code(results) -> int` (non-zero if any
  critical check failed) and unit-test it directly rather than patching `SystemExit`.

### 5. `top` — pure snapshot (TDD)
- Write `tests/client/test_tui.py` first for
  `build_snapshot(summary: StatusSummary, jobs: list[JobView], *, max_rows=20) -> TopSnapshot`.
- `TopSnapshot` holds: header fields (batch, totals, throughput, twitter_count), status-count rows, and
  truncated job rows (short job_id, status, twitter, pred_class, path). Pure — no Textual import.

### 6. `top` — Textual app (TDD-light)
- `TopApp(App)` in `tui.py`: constructor takes `client`, `batch_id`, `refresh_seconds`. `on_mount` calls
  `refresh_data()` once, then `self.set_interval(self.refresh_seconds, self.refresh_data)`. `refresh_data`
  (async) calls `client.status()` + `client.list_jobs()` concurrently (`asyncio.gather`), builds a snapshot,
  and updates a `DataTable` + a summary `Static`. Bindings: `q` quit, `r` refresh-now. A down server → error
  banner, keep polling.
- Test with a mocked client (`mocker`): assert `refresh_data` pulls data and populates the widgets; never spin
  a real terminal.

### 7. `serve` — weights resolution (TDD)
- Write `tests/client/test_serve.py` first. `resolve_serve_weights(*, select: bool, settings, selector=None) -> Path`:
  if `select`, `discover_models` across `settings.model_search_roots` with `SERVER_MODEL_EXTS` → `select_model`;
  else return `settings.weights_path`. Raise on empty candidates / cancel (mirror `demo.resolve_model`).
- `apply_weights_env(path)`: set `os.environ["SCREENCROPNET_WEIGHTS_PATH"] = str(path)` and call
  `get_settings.cache_clear()` so a fresh `Settings()` reads it.

### 8. `serve` — launcher
- `serve` command: resolve weights, export env, then
  `uvicorn.run("screencropnet_yolo.server.api:create_app", factory=True, host=..., port=...)`.
  `--with-worker` first spawns `screencrop-worker` as a detached subprocess in the same environment.
  Keep the launch call thin; tests patch `uvicorn.run` / `subprocess.Popen`.

### 9. Wire Typer commands
- Add to `cli.py`:
  - `serve(--select/--fuzzy, --host, --port, --with-worker)`
  - `top(--batch-id, --refresh: float = 5.0)` — build a client, construct `TopApp(client, batch_id, refresh)`,
    `app.run()`, and close the client in a `finally`.
  - `doctor(--json)` — `asyncio.run(run_doctor(...))`, render, then `raise typer.Exit(exit_code(results))`.

### 10. pyinstrument (reach goal)
- Makefile: `profile-demo` (`uv run pyinstrument -r html -o profile_demo.html -m screencropnet_yolo.demo ...`),
  `profile-doctor`, and a generic `profile SCRIPT=...`. Add `profile-open` (open the HTML report).
- Optional: in `api.create_app`, if `os.environ.get("SCREENCROPNET_PROFILE")`, add pyinstrument's async-safe
  `ProfilerMiddleware` (`?profile=1` returns an HTML flamegraph). Gated so production is unaffected; add
  `make profile-api` documenting the env toggle.

### 11. Makefile ergonomics
- Add `serve`, `top`, `doctor` targets mirroring the existing `@uv run screencrop-cli ...` style with a `🚀`
  echo and a `## help` comment.

### 12. Docs
- `docs/top.md`, `docs/doctor.md`, `docs/serve.md`: usage, flags, and the `doctor` exit-code contract.
  Cross-link from `docs/demo.md` where fuzzy selection is described.

### 13. Final validation
- Run the Validation Commands below; ensure `make lint`, `make test`, and `make check` are clean.

## Testing Strategy
- **Unit (no live services)**: `model_select` discovery/format/select; `build_snapshot`; each `doctor` check with
  mocked `httpx`/`aio_pika`/engine (success, failure, timeout); `run_doctor` aggregation + `exit_code`
  summarizer; `resolve_serve_weights` + `apply_weights_env` (env set + `cache_clear`); `TopApp.refresh_data`
  with a mocked client.
- **Interactive bits never really run**: `fzf`, a live terminal, uvicorn, and subprocesses are all patched.
- **Integration (`-m integration`, opt-in)**: `doctor` against a real `make services-up` stack; a
  `serve --with-worker` smoke test.
- **Edge cases**: empty model roots, fzf cancel (ESC), server down during `top`, one service down in `doctor`,
  `--refresh` < 1 s, malformed `postgres_dsn`.

## Acceptance Criteria
- `model_select.py` is the single source of the fuzzy helpers; `demo.py` imports them and all prior `demo`
  tests pass unchanged.
- `screencrop-cli serve --select` shows an fzf list of `.pt/.onnx/.pth` under the configured roots and boots
  the API against the pick.
- `screencrop-cli top --refresh N` renders a live Textual dashboard refreshing every N seconds; `q` quits,
  `r` refreshes now; it survives a down server.
- `screencrop-cli doctor` runs all six checks concurrently, prints ✔︎/✘ + latency, supports `--json`, and
  exits non-zero on any critical failure.
- `make profile-demo` writes an openable pyinstrument HTML report.
- `make lint`, `make test`, `make check` are clean.

## Validation Commands
- `uv run pytest tests/test_model_select.py tests/client/test_doctor.py tests/client/test_tui.py tests/client/test_serve.py` — the new suites.
- `uv run pytest` — full suite (unit; integration deselected by default).
- `make lint` — codespell + ruff + basedpyright.
- `make check` — `ty` type check.
- `uv run screencrop-cli doctor --json` — after `make services-up && make api && make worker`.
- `uv run screencrop-cli top --refresh 2 --batch-id <id>` — live dashboard.
- `uv run screencrop-cli serve --select --with-worker` — fzf pick + boot.
- `make profile-demo && make profile-open` — profiling report.

## Notes
- New deps: `uv add textual`; `uv add --dev pyinstrument`.
- `get_settings()` is `@lru_cache` — call `get_settings.cache_clear()` after mutating
  `SCREENCROPNET_WEIGHTS_PATH` in `serve`.
- Ports come from `docker-compose.yml` (prometheus **9091**, grafana **3001**); expose them as `Settings`
  fields so non-default deployments can override via `SCREENCROPNET_*` env vars.
- `pyfzf` needs the `fzf` binary (`brew install fzf`) — a system prerequisite, imported lazily so non-`serve`
  code paths never require it.
- Follow project conventions: absolute imports, `from __future__ import annotations`, modern unions
  (`str | None`), `pathlib.Path`, `@override` from `typing_extensions`, `StrEnum`, `pytest-mock` (never
  `unittest.mock`), and inline `## Tests` blocks for pure torch/IO-free helpers.
