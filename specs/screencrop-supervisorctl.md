# Spec: `screencrop-supervisor-worker` / `screencrop-supervisorctl` CLI

> Implementation is **TDD** — drive every unit with the
> `superpowers:test-driven-development` skill (RED → GREEN → refactor). Write the failing
> test first, then the minimal implementation.

## Context

The repo already runs a client/server ingest→classify pipeline: `screencrop-worker`
(`server/worker.py`) is an **async RabbitMQ consumer** that pulls jobs off a durable
queue (`screennet_inference_queue`, `prefetch_count=8`), classifies each image off the
event loop via `anyio.to_thread.run_sync`, and writes results to Postgres. Because
RabbitMQ workers are **competing consumers on one shared queue**, running several of them
already gives the exact behavior we want: *"depending on if a given worker is currently
completing a job, another one will accept the work till all work is completed."* The broker
load-balances; whichever worker is idle grabs the next message.

What is missing is a way to **operate a fleet** of those workers: spin up N of them against a
chosen model, stop/restart/inspect them, and read their logs. Today the only launcher is
`serve.py:_launch_worker`, which spawns exactly one worker as a side effect of `serve
--with-worker`. This spec generalizes that into a supervisor CLI.

**Outcome:** a new command (two binary names, same Typer app) —
`screencrop-supervisor-worker` and its alias `screencrop-supervisorctl` — that can
`start`, `stop`, `restart`, `status`, and tail `logs` for a pool of `screencrop-worker`
processes, with fuzzy model selection (`--select`/`--fuzzy`) and colored or plain log output.

## Objective

Ship `screencrop-supervisorctl` (alias of `screencrop-supervisor-worker`) that:

1. `start --workers N` — resolve a model (optionally via fzf), spawn N **detached**
   worker processes as competing RabbitMQ consumers, each with its own metrics port,
   log file, and PID/state record.
2. `stop [--all|NAME]` — **warm** shutdown (SIGTERM, let in-flight jobs finish up to a
   timeout) then **cold** kill (SIGKILL), Celery-style.
3. `restart [--all|NAME]` — stop then re-spawn the same fleet.
4. `status [--json] [--watch]` — liveness + per-worker metrics probe, rendered as a rich
   table or JSON.
5. `logs [NAME] [-f/--follow] [-n N] [--color/--plain]` — tail per-worker logs.
6. `--select`/`--fuzzy` model picking, reusing the existing `model_select` + `serve` seam.

## Problem Statement

There is no operational surface for running **multiple** inference workers in parallel or
for observing/controlling them once launched. A single `serve --with-worker` fires one
worker and forgets it (no PID, no clean stop, no status, no per-worker logs). Scaling
throughput and managing worker lifecycle requires a supervisor.

## Solution Approach

**Confirmed decisions:**

| Fork | Choice |
|------|--------|
| Supervision model | **Broker-backed process supervisor** — manage N real `screencrop-worker` OS processes as RabbitMQ competing consumers. |
| Lifecycle/control | **Detached workers + PID/state files + Unix signals** (SIGTERM warm → SIGKILL cold). No long-lived daemon. |
| Packaging | **Two new binaries**, both pointing at `client/supervisor.py:main`. |

**Parallelism / concurrency model:**

- **True parallelism = separate OS processes.** Each `screencrop-worker` is its own
  process, so N workers run inference on CPU/GPU genuinely in parallel — the GIL never
  binds across processes. This is Celery's "prefork"-style concurrency; `--workers N`
  mirrors Celery's `--concurrency`.
- **Per-worker internal concurrency = `prefetch_count` + thread offload.** Inside a worker,
  `anyio.to_thread.run_sync` runs the blocking classify off the event loop (torch releases
  the GIL during native ops), and RabbitMQ `prefetch_count` bounds how many messages it
  buffers. Exposed as `--prefetch`.
- **Load balancing = competing consumers.** No custom dispatch/queue is written — the
  durable RabbitMQ queue is the shared work source; idle workers pull the next job. The
  *coordination* is I/O-bound (perfect for `asyncio`); the *compute* is CPU/GPU-bound (needs
  processes, not threads/coroutines).
- **Supervisor coordination = `asyncio`.** `status` probes all workers concurrently with
  `asyncio.gather` (reusing `doctor.check_http`); `logs --follow` is an async tail;
  `--watch`/`--follow` install `loop.add_signal_handler` for clean Ctrl-C.

**Design pattern:** follow the repo's **pure-core + thin-IO-wrapper seam** (as in
`doctor.py`, `serve.py`, `worker.py`). All planning/formatting/state logic is pure and
unit-tested directly; the three IO edges (spawn process, send signal, HTTP probe) are thin
and patched in tests via `pytest-subprocess` / `pytest-mock`.

## Relevant Files

Reuse (do **not** reinvent):

- `src/screencropnet_yolo/model_select.py` — `discover_models`, `select_model`,
  `SERVER_MODEL_EXTS`, `ModelSelector`. The fzf picker.
- `src/screencropnet_yolo/client/serve.py` — `resolve_serve_weights(select=…, settings,
  selector)`, `apply_weights_env(path)`, `WEIGHTS_ENV` (`SCREENCROPNET_WEIGHTS_PATH`), and
  the `_launch_worker` subprocess pattern (`Popen([...], env=…, start_new_session=True)`).
- `src/screencropnet_yolo/client/doctor.py` — `check_http`, `CheckResult`, `run_doctor`
  pattern (concurrent `asyncio.gather` probes with per-probe timeout). Reused to probe each
  worker's metrics port for `status`.
- `src/screencropnet_yolo/output.py` — `colorize`, `Color`, `human_size` for colored/plain
  log lines; respects `NO_COLOR` + TTY.
- `src/screencropnet_yolo/server/config.py` — `Settings` (`env_prefix=SCREENCROPNET_`),
  `logs_dir`, `worker_metrics_port`, `model_search_roots`, `get_settings` (`@lru_cache`).
- `src/screencropnet_yolo/client/cli.py` — Typer app conventions (`typer.Typer`, rich
  `Console`, `Table`, command signatures).
- `strif.atomic_output_file` — crash-safe writes for state files (used in `server/export.py`).

Modify:

- `pyproject.toml` `[project.scripts]` — add the two entries.
- `src/screencropnet_yolo/server/worker.py` — (a) per-worker log path via settings/env in
  `_configure_logging`; (b) **warm-shutdown** SIGTERM handler in `run_worker` (cancel the
  consumer, await in-flight handlers, close). Metrics port is already env-overridable — no
  code change, just set the env per child.
- `src/screencropnet_yolo/server/config.py` — add `worker_log_path: Path | None = None` and
  `supervisor_state_dir: Path = Path("logs/supervisor")` (plus optional
  `supervisor_workers: int`, `supervisor_metrics_base_port: int` defaults).

### New Files

- `src/screencropnet_yolo/client/supervisor.py` — the Typer app + pure core + IO wrappers.
- `tests/client/test_supervisor.py` — the unit suite (primary TDD target).
- `docs/supervisor.md` — user docs, mirroring `docs/demo.md` (Phase 3).

## Implementation Phases

### Phase 1: Foundation — state model + pure core (no real processes)
Pure, torch-free, broker-free helpers with inline/unit tests: worker spec computation,
state file read/write, liveness classification, status/log rendering. Fastest to TDD.

### Phase 2: Core Implementation — lifecycle + CLI wiring
Thin IO wrappers (spawn/signal/probe), the five Typer commands, and worker.py changes
(per-worker log path, warm-shutdown signal handling). Entry points wired in `pyproject.toml`.

### Phase 3: Integration & Polish
`--select`/`--fuzzy` wiring, colored `logs --follow`, `docs/supervisor.md`, end-to-end
smoke against a live stack, `make lint` + `make test` green.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom. Each code step is TDD: write the
failing test first, then the minimal implementation.

### 1. Settings additions (TDD)
- In `tests/server/test_config.py` (or inline `## Tests` in `config.py`), assert the new
  fields default correctly and honor `SCREENCROPNET_*` env overrides:
  `worker_log_path: Path | None = None`, `supervisor_state_dir: Path = Path("logs/supervisor")`,
  `supervisor_metrics_base_port: int = 8001`, `supervisor_workers: int = 2`.
- Add the fields to `server/config.py:Settings`.

### 2. Worker spec + state model (pure core, TDD)
Create `client/supervisor.py`. Test-drive these pure functions in `test_supervisor.py`:
- `@dataclass(frozen=True) WorkerSpec`: `name`, `index`, `metrics_port`, `log_path`,
  `state_path`, `env: dict[str,str]`.
- `worker_specs(count, *, weights, state_dir, log_dir, base_metrics_port) -> list[WorkerSpec]`
  — pure: names `worker@1..N`, ports `base+i`, log `state_dir/worker-<i>.log`, and per-spec
  env carrying `SCREENCROPNET_WEIGHTS_PATH`, `SCREENCROPNET_WORKER_METRICS_PORT`,
  `SCREENCROPNET_WORKER_LOG_PATH` (merged onto `os.environ`).
- `@dataclass(frozen=True) WorkerState`: `name`, `pid`, `metrics_port`, `weights_path`,
  `log_path`, `started_at`.
- `write_state(spec, pid, *, started_at)` using `strif.atomic_output_file` → JSON;
  `read_states(state_dir) -> list[WorkerState]`; `remove_state(state_path)`.
- Tests use `tmp_path`; assert round-trip and newest-first / stable ordering.

### 3. Liveness + status rendering (pure core, TDD)
- `is_alive(pid) -> bool` (wrap `os.kill(pid, 0)`; treat `ProcessLookupError`→dead,
  `PermissionError`→alive). Patch `os.kill` in tests.
- `classify_status(*, alive, metrics_ok) -> str` → `running` | `dead` | `unreachable`.
- `build_status_rows(states, liveness, probes)` — pure merge of state + liveness +
  `CheckResult`.
- `render_status_table(rows) -> rich.Table` and `render_status_json(rows) -> str`
  (mirror `doctor.render_table` / `render_json`, reuse `✔︎`/`✘` glyphs).

### 4. Log tailing + colorization (pure core, TDD)
- `tail_lines(path, n) -> list[str]` (pure, `tmp_path`-tested; missing file → `[]`).
- `colorize_log_line(line, *, enabled) -> str` — map level tokens (`ERROR`/`WARNING`/
  `INFO`) to `Color.*` via `output.colorize`; `enabled = --color AND isatty AND not NO_COLOR`.
- Assert plain passthrough when disabled and correct ANSI when enabled.

### 5. Thin IO wrappers (TDD with `pytest-subprocess` / `pytest-mock`)
- `spawn_worker(spec) -> int`: `subprocess.Popen(["screencrop-worker"], env=spec.env,
  stdout=DEVNULL, stderr=DEVNULL, stdin=DEVNULL, start_new_session=True)` → return `pid`.
  Test with the `fake_process` fixture; assert argv + env keys.
- `signal_worker(pid, sig)`: wrap `os.kill`; patched in tests.
- `async probe_worker(state, *, timeout) -> CheckResult`: reuse
  `doctor.check_http("worker@i", f"http://127.0.0.1:{state.metrics_port}/", timeout=…)`.
- `async follow_log(path)` async generator yielding appended lines (poll + `asyncio.sleep`;
  test with a file written incrementally under `tmp_path`).

### 6. Worker per-worker log path + warm shutdown (TDD, `server/worker.py`)
- `_configure_logging`: use `settings.worker_log_path or settings.logs_dir/"worker.log"`.
  Test that a set `worker_log_path` routes the FileHandler to that path.
- `run_worker`: replace `await asyncio.Future()` with a warm-shutdown path — install a
  SIGTERM handler (`loop.add_signal_handler`) that `await queue.cancel(consumer_tag)` to
  stop new deliveries, waits for outstanding `on_message` tasks to drain (bounded by a
  timeout), then closes the connection and returns. Unit-test the drain logic against a
  fake queue/consumer (no live broker); keep the aio-pika wiring thin.

### 7. Lifecycle orchestration (TDD, pure over injected IO)
Write these to take the IO wrappers as injectable params (default to the real ones) so they
test without processes:
- `start_fleet(count, *, weights, settings, spawn=spawn_worker, now=time.time)` — compute
  specs, ensure `state_dir`, spawn each, `write_state`. Assert N spawns + N state files.
- `stop_fleet(states, *, timeout, signal=signal_worker, alive=is_alive, sleep)` — send
  SIGTERM to each; poll `alive` until gone or `timeout`; then SIGKILL survivors; remove
  state files. Assert warm-then-cold escalation via a fake that "dies" after k polls, and
  the cold path when it never dies.
- `restart_fleet(...)` — `read_states` → `stop_fleet` → `start_fleet` reconstructed from
  the prior states (count + weights).
- `collect_status(states, *, probe, alive)` — `asyncio.gather` liveness + probes → rows.

### 8. Typer CLI + `--select`/`--fuzzy` (TDD with `CliRunner`)
- Build `app = typer.Typer(...)` with commands `start`, `stop`, `restart`, `status`, `logs`.
- `start(--workers/-w, --select/--fuzzy, --model, --prefetch, --metrics-base-port)`:
  resolve weights via `serve.resolve_serve_weights(select=…, settings, selector)` (reuse the
  fzf seam and its cancelled/empty error semantics), then `start_fleet`.
- `stop(name, --all, --timeout, --cold)`; `restart(name, --all)`;
  `status(--json, --watch, --refresh)`; `logs(name, -f/--follow, -n/--lines, --color/--plain)`.
- Tests: `CliRunner().invoke(app, [...])` with `spawn`/`signal`/`selector` patched; assert
  exit codes, spawn counts, and that `--fuzzy` routes through the selector.
- `def main() -> None: app()`.

### 9. Wire entry points
- `pyproject.toml` `[project.scripts]`:
  ```toml
  screencrop-supervisor-worker = "screencropnet_yolo.client.supervisor:main"
  screencrop-supervisorctl     = "screencropnet_yolo.client.supervisor:main"
  ```
- Add a `test_packaging.py`-style assertion that both console scripts resolve.

### 10. Docs + validation
- Write `docs/supervisor.md` (mirror `docs/demo.md`): commands, the competing-consumer
  model, warm/cold shutdown, `--select`, per-worker ports/logs, env overrides.
- Cross-link from `CLAUDE.md` Architecture section (one bullet for `supervisor.py`).
- Run `make lint` and `make test`; fix until green. Manual smoke (below).

## Testing Strategy

- **Framework:** pytest + `asyncio_mode=auto` (no `@pytest.mark.asyncio` needed) +
  `pytest-mock` (`mocker`) + `pytest-subprocess` (`fake_process`) + `typer.testing.CliRunner`.
  Long tests in `tests/client/test_supervisor.py`; a few pure helpers may carry inline
  `## Tests` blocks (collected from `src/`, no pytest import).
- **Pure core** (specs, state I/O, liveness classification, status/log rendering, tail):
  tested directly with `tmp_path` — no processes, no broker, no torch.
- **IO edges** patched: `spawn_worker` via `fake_process`; `signal_worker`/`is_alive` via
  `mocker.patch("...os.kill")`; metrics probe via a fake `check_http`/httpx client (as in
  `test_doctor.py`); fzf via an injected `selector` (as in `test_serve.py`).
- **Edge cases:** empty/missing state dir; stale PID whose process is gone
  (`ProcessLookupError`); a worker that ignores SIGTERM (cold-kill escalation after timeout);
  port collision avoided by `base+i` allocation; `NO_COLOR`/non-TTY forces plain logs;
  cancelled or empty fzf pick raises (mirrors `serve.resolve_serve_weights`); `logs` on a
  never-started worker returns `[]`; `--all` vs single-name targeting.
- **Assertions:** follow repo convention — `if not cond: raise AssertionError(msg)`, never
  bare `assert False`; no trivial assertions.

## Acceptance Criteria

- Both `screencrop-supervisor-worker` and `screencrop-supervisorctl` are installed console
  scripts resolving to the same Typer app.
- `start -w N` spawns N detached workers, each with a distinct metrics port and log file,
  and writes N state files under `supervisor_state_dir`.
- `status` shows per-worker liveness + metrics reachability (table and `--json`).
- `stop --all` performs warm shutdown then cold-kills stragglers after `--timeout`, and
  clears state files; `restart --all` reconstructs the same fleet.
- `logs NAME -f` tails a worker's log; `--color` colorizes, `--plain`/`NO_COLOR`/non-TTY
  does not.
- `start --select`/`--fuzzy` fuzzy-picks weights and propagates them to every worker via
  `SCREENCROPNET_WEIGHTS_PATH`.
- Workers finish in-flight jobs on SIGTERM (warm shutdown) rather than dropping them.
- `make lint` and `make test` are clean.

## Validation Commands

- `uv run pytest tests/client/test_supervisor.py -q` — the new unit suite passes.
- `uv run pytest src/screencropnet_yolo/client/supervisor.py -q` — inline `## Tests` pass.
- `make lint` — codespell + ruff + basedpyright clean.
- `make check` — `ty` type check clean.
- `make test` — full suite + coverage clean.
- `uv run python -c "import screencropnet_yolo.client.supervisor"` — imports torch-free.
- Manual smoke against a live stack (`docker compose up` for rabbit/postgres):
  - `uv run screencrop-supervisorctl start -w 3` → 3 workers up.
  - `uv run screencrop-supervisorctl status` → 3 running rows.
  - `uv run screencrop-cli submit <folder>` then `screencrop-supervisorctl status --watch`.
  - `uv run screencrop-supervisorctl logs worker@1 -f --color`.
  - `uv run screencrop-supervisorctl restart --all` then `stop --all` → state files gone.

## Notes

- **No new runtime dependencies.** `typer`, `rich`, `httpx`, `aio-pika`, `pyfzf`, `strif`,
  `anyio` are already declared; `pytest-subprocess` is already a dev dep. If a test ever
  needs a package, add it with `uv add --dev <pkg>` — never `pip`.
- **`fzf` is a system prerequisite** (`brew install fzf`) only for `--select`, imported
  lazily inside `model_select._fzf_select`; importing `supervisor.py` never requires it.
- **Metrics-port collision** is the key gotcha: `worker.py` binds a single
  `worker_metrics_port`; the supervisor must assign `base+i` per child via env — covered in
  Task 2.
- **No daemon / no custom queue:** control is stateless (PID/state files + signals) and
  load-balancing is delegated to RabbitMQ's competing consumers. A future enhancement could
  add Celery-style `max-tasks-per-child` (recycle a worker after N jobs to bound memory) and
  an optional watchdog auto-restart loop.
