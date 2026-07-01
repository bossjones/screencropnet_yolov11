# Spec: `screencrop-supervisor-worker` / `screencrop-supervisorctl` CLI (trimmed)

> Implementation is **TDD** — drive every unit with `superpowers:test-driven-development`
> (RED → GREEN → refactor). Write the failing test first, then the minimal implementation.

## Context

`screencrop-worker` (`server/worker.py`) is an **async RabbitMQ consumer**: pulls jobs off a
durable queue (`screennet_inference_queue`), classifies off the event loop via
`anyio.to_thread.run_sync`, and writes results to **Postgres** (the job-status source of
truth read by `status`/`list_jobs`/`twitter`/`export`/TUI). Multiple workers are **competing
consumers on one shared queue**, so running N of them already gives "idle worker grabs the
next job." `docker-compose.yml` runs **infra only** (postgres, rabbitmq, prometheus, grafana);
its comment states *workers run on the host, not in containers* — because they need MPS/CUDA
GPU access. So workers are **host processes** and compose cannot scale them; a host-side
manager is the right tool.

What's missing: (a) a clean way to operate a **fleet** of host workers (start N, stop,
restart, status) with fuzzy model selection, and (b) **warm shutdown** — today `run_worker`
does `await asyncio.Future()` with no signal handling, so SIGTERM **drops in-flight jobs**.

**Outcome:** one command (two binary names) — `screencrop-supervisor-worker` and its alias
`screencrop-supervisorctl` — to `start` / `stop` / `restart` / `status` / `logs` a pool of
host workers, plus a real warm-shutdown fix in the worker.

## Design rationale: why NOT Celery (and why the spec was trimmed)

Celery was considered and rejected. It fits greenfield systems, multi-node fan-out, or
workflow primitives (chords/ETA/retries) — none of which apply. Adopting it here fights the
existing design:

- **Async mismatch.** Celery task bodies are **synchronous** (prefork). Our worker + DB layer
  are async (`aio-pika`, asyncpg, `async_sessionmaker`, `anyio.to_thread`). Adoption ⇒
  de-async rewrite of `worker.py`/`db.py` (new sync DB path or fragile `asyncio.run()` per
  task) + swap the FastAPI `RabbitPublisher` for Celery's producer.
- **Two sources of truth.** Postgres already *is* job-status truth; Celery brings its own
  result backend ⇒ either divergent state (Flower vs our CLI) or rewrite all reporting/export/
  TUI onto Celery's backend.
- **GPU + fork footgun.** Prefork forks children holding CUDA/MPS contexts (classic
  breakage). Celery's `worker_process_init` "load model per child" is exactly what each
  standalone `screencrop-worker` process already does — no gain, added risk.
- **Testability regresses.** Our suite is broker-free (pure-core + `FakePublisher`/
  `FakeClassifier` + sqlite, `asyncio_mode=auto`). Celery's docs say `task_always_eager` is
  *"not suitable for unit tests,"* and its `celery_worker` fixture spins a real threaded
  worker (≤10s waits). We'd trade a fast suite for a slow one.
- **Monitoring is already covered.** Prometheus + Grafana (in compose) + `doctor` give the
  `celery inspect`/Flower equivalent without the rewrite.

**Trim of the original spec (the over-engineered ~30%):**
- **Drop bespoke log tailing** (`tail_lines`/`follow_log`/`colorize_log_line`). `logs` shells
  out to `tail`/`less -R` — deletes a whole pure-core section + its tests.
- **Drop per-worker metrics HTTP probing for `status`.** Use plain **PID liveness**; leave
  metrics to Prometheus/Grafana/`doctor`. (An optional `--probe` flag can reuse
  `doctor.check_http` for reachability, off by default.)
- **Keep** warm shutdown (a real bug fix), thin `start/stop/restart/status` over PID/state
  files, `--select`/`--fuzzy` model pick, and per-worker metrics-port + log-path env.

## Objective

Ship `screencrop-supervisorctl` (alias of `screencrop-supervisor-worker`):

1. `start --workers N [--select/--fuzzy] [--model] [--prefetch] [--metrics-base-port]` —
   resolve a model, spawn N **detached** host workers (competing consumers), each with its
   own metrics port + log file + PID/state record.
2. `stop [--all|NAME] [--timeout] [--cold]` — **warm** shutdown (SIGTERM, let in-flight jobs
   finish up to timeout) then **cold** kill (SIGKILL). Clears state files.
3. `restart [--all|NAME]` — stop then re-spawn the same fleet (reconstructed from state).
4. `status [--json] [--probe]` — **PID-liveness** + state metadata (name/pid/port/weights/
   uptime) as a rich table or JSON; `--probe` optionally adds metrics reachability via
   `doctor.check_http`.
5. `logs NAME [-f/--follow] [-n N] [--color/--plain]` — **shell out** to `tail -n N [-f]`
   (with `less -R` when `--color`) on the resolved per-worker log path.

## Relevant Files

Reuse (do **not** reinvent):

- `src/screencropnet_yolo/client/serve.py` — `resolve_serve_weights(select=…, settings,
  selector)`, `apply_weights_env`, `WEIGHTS_ENV`, and the `_launch_worker` Popen pattern
  (`Popen([...], env=…, start_new_session=True)`) — the seed of `spawn_worker`.
- `src/screencropnet_yolo/model_select.py` — `discover_models`, `select_model`,
  `SERVER_MODEL_EXTS`, `ModelSelector` (the fzf picker; lazily imported).
- `src/screencropnet_yolo/client/doctor.py` — `check_http`, `CheckResult`, `render_table`/
  `render_json` idiom; reused only for the optional `status --probe`.
- `src/screencropnet_yolo/output.py` — `colorize`, `Color`, `human_size` (for the status
  table; log color is delegated to `less -R`).
- `src/screencropnet_yolo/server/config.py` — `Settings` (`env_prefix=SCREENCROPNET_`),
  `logs_dir`, `worker_metrics_port`, `model_search_roots`, `get_settings`.
- `src/screencropnet_yolo/client/cli.py` — Typer + rich conventions.
- `strif.atomic_output_file` — crash-safe state-file writes (as in `server/export.py`).

Modify:

- `pyproject.toml` `[project.scripts]` — two entries → `client/supervisor.py:main`.
- `src/screencropnet_yolo/server/worker.py` — (a) per-worker log path in
  `_configure_logging` (`settings.worker_log_path or logs_dir/"worker.log"`); (b)
  **warm-shutdown** SIGTERM handler in `run_worker` (cancel the consumer tag, await
  in-flight handlers up to a timeout, close). Metrics port is already env-overridable.
- `src/screencropnet_yolo/server/config.py` — add `worker_log_path: Path | None = None`,
  `supervisor_state_dir: Path = Path("logs/supervisor")`, `supervisor_metrics_base_port:
  int = 8001`, `supervisor_workers: int = 2`.

### New Files

- `src/screencropnet_yolo/client/supervisor.py` — Typer app + pure core + thin IO wrappers.
- `tests/client/test_supervisor.py` — primary TDD suite.
- `docs/supervisor.md` — user docs (mirror `docs/demo.md`), Phase 3.

## Step by Step Tasks
Execute in order. Each code step is TDD: failing test first, then minimal implementation.

### 1. Settings additions (TDD)
- Assert defaults + `SCREENCROPNET_*` env overrides for the four new fields (test in
  `tests/server/test_config.py` or inline `## Tests`), then add them to `Settings`.

### 2. Worker spec + state model (pure core, TDD)
Create `client/supervisor.py`. Test-drive:
- `@dataclass(frozen=True) WorkerSpec`: `name`, `index`, `metrics_port`, `log_path`,
  `state_path`, `env: dict[str,str]`.
- `worker_specs(count, *, weights, state_dir, log_dir, base_metrics_port) -> list[WorkerSpec]`
  — pure: names `worker@1..N`, ports `base+i`, log `state_dir/worker-<i>.log`, per-spec env
  carrying `SCREENCROPNET_WEIGHTS_PATH`, `SCREENCROPNET_WORKER_METRICS_PORT`,
  `SCREENCROPNET_WORKER_LOG_PATH` (merged onto `os.environ`).
- `@dataclass(frozen=True) WorkerState`: `name`, `pid`, `metrics_port`, `weights_path`,
  `log_path`, `started_at`.
- `write_state` (via `strif.atomic_output_file` → JSON), `read_states(state_dir)`,
  `remove_state`. Tested round-trip under `tmp_path`.

### 3. Liveness + status rendering (pure core, TDD)
- `is_alive(pid) -> bool` (wrap `os.kill(pid, 0)`; `ProcessLookupError`→dead,
  `PermissionError`→alive; patch `os.kill` in tests).
- `build_status_rows(states, liveness, *, probes=None)` — pure merge of state + liveness
  (+ optional `CheckResult`), with uptime from `started_at`.
- `render_status_table(rows)` / `render_status_json(rows)` (mirror `doctor` idiom, reuse
  `✔︎`/`✘`). No per-worker HTTP probe in the default path.

### 4. Thin IO wrappers (TDD with `pytest-subprocess` / `pytest-mock`)
- `spawn_worker(spec) -> int`: `Popen(["screencrop-worker"], env=spec.env, stdout=DEVNULL,
  stderr=DEVNULL, stdin=DEVNULL, start_new_session=True)`; return pid. Test via
  `fake_process`; assert argv + env keys.
- `signal_worker(pid, sig)`: wrap `os.kill`; patched in tests.
- `tail_command(state, *, follow, lines, color) -> list[str]` (**pure**): build the argv,
  e.g. `["tail", "-n", str(lines), "-f", str(log_path)]`, optionally piped to `less -R`.
  `run_logs(...)` execs it (thin; patched in tests — assert the argv, don't tail).
- Optional `async probe_worker(state, *, timeout)` reusing `doctor.check_http` — only for
  `status --probe`.

### 5. Worker per-worker log path + warm shutdown (TDD, `server/worker.py`)
- `_configure_logging`: route the FileHandler to `settings.worker_log_path or
  logs_dir/"worker.log"`. Test the override.
- `run_worker`: replace `await asyncio.Future()` with a warm-shutdown path — install a
  SIGTERM handler (`loop.add_signal_handler`) that `await queue.cancel(consumer_tag)`, waits
  for outstanding `on_message` tasks to drain (bounded by a timeout), then closes and
  returns. Unit-test the drain against a fake queue/consumer (no live broker).

### 6. Lifecycle orchestration (TDD, pure over injected IO)
IO wrappers injectable (default to the real ones) so these test without processes:
- `start_fleet(count, *, weights, settings, spawn=spawn_worker, now=time.time)` — compute
  specs, ensure `state_dir`, spawn, `write_state`. Assert N spawns + N state files.
- `stop_fleet(states, *, timeout, signal=signal_worker, alive=is_alive, sleep)` — SIGTERM
  each; poll `alive` until gone or timeout; SIGKILL survivors; remove state. Assert
  warm-then-cold escalation (fake dies after k polls) and the cold path (never dies).
- `restart_fleet(...)` — `read_states` → `stop_fleet` → `start_fleet` from prior states.

### 7. Typer CLI + `--select`/`--fuzzy` (TDD with `CliRunner`)
- `app = typer.Typer(...)` with `start`, `stop`, `restart`, `status`, `logs`.
- `start`: resolve weights via `serve.resolve_serve_weights(select=…, settings, selector)`
  (reuse its cancelled/empty error semantics), then `start_fleet`.
- `status(--json, --probe)`; `logs(name, -f, -n, --color/--plain)` → `run_logs`.
- Tests: `CliRunner().invoke(...)` with `spawn`/`signal`/`selector`/`run_logs` patched;
  assert exit codes, spawn counts, `--fuzzy` routes through the selector, and the `logs`
  argv. `def main() -> None: app()`.

### 8. Wire entry points
- `pyproject.toml`:
  ```toml
  screencrop-supervisor-worker = "screencropnet_yolo.client.supervisor:main"
  screencrop-supervisorctl     = "screencropnet_yolo.client.supervisor:main"
  ```
- Add a `test_packaging.py`-style assertion both console scripts resolve.

### 9. Docs + validation
- `docs/supervisor.md`: commands, competing-consumer model, warm/cold shutdown, `--select`,
  per-worker ports/logs, env overrides, and a one-line "why not Celery" pointer.
- One `CLAUDE.md` Architecture bullet for `supervisor.py`.
- `make lint` + `make test` green; manual smoke (below).

## Testing Strategy

- pytest + `asyncio_mode=auto` + `pytest-mock` + `pytest-subprocess` (`fake_process`) +
  `typer.testing.CliRunner`. Long tests in `tests/client/test_supervisor.py`; a few pure
  helpers may carry inline `## Tests`.
- **Pure core** (specs, state I/O, liveness, status rendering, `tail_command` argv) tested
  directly under `tmp_path` — no processes, broker, or torch.
- **IO edges patched:** `spawn_worker` via `fake_process`; `signal_worker`/`is_alive` via
  `mocker.patch("...os.kill")`; `run_logs` asserted by argv (no real `tail`); fzf via an
  injected `selector` (as in `test_serve.py`); optional probe via a fake `check_http`.
- **Edge cases:** empty/missing state dir; stale PID (`ProcessLookupError`); worker ignoring
  SIGTERM (cold-kill escalation); `base+i` avoids metrics-port collision; cancelled/empty fzf
  pick raises; `logs` on a never-started worker errors cleanly; `--all` vs single-name.
- Repo convention: `if not cond: raise AssertionError(msg)`; no trivial assertions.

## Acceptance Criteria

- Both console scripts resolve to the same Typer app.
- `start -w N` spawns N detached workers with distinct metrics ports + log files and writes
  N state files under `supervisor_state_dir`.
- `status` shows per-worker PID liveness + metadata (table + `--json`); `--probe` adds
  reachability.
- `stop --all` warm-shuts-down then cold-kills stragglers after `--timeout` and clears state;
  `restart --all` reconstructs the fleet.
- `logs NAME -f` tails via `tail`/`less -R`; no bespoke tailing code.
- `start --select`/`--fuzzy` fuzzy-picks weights and propagates via
  `SCREENCROPNET_WEIGHTS_PATH` to every worker.
- Workers finish in-flight jobs on SIGTERM (warm shutdown).
- `make lint` + `make test` clean.

## Validation Commands

- `uv run pytest tests/client/test_supervisor.py -q`
- `uv run pytest src/screencropnet_yolo/client/supervisor.py -q` (inline tests)
- `make lint` · `make check` · `make test`
- `uv run python -c "import screencropnet_yolo.client.supervisor"` (torch-free import)
- Live smoke (`docker compose up` for infra):
  - `uv run screencrop-supervisorctl start -w 3` → 3 workers; `status` → 3 running rows.
  - `uv run screencrop-cli submit <folder>` then `screencrop-supervisorctl status --probe`.
  - `uv run screencrop-supervisorctl logs worker@1 -f --color`.
  - `uv run screencrop-supervisorctl restart --all` then `stop --all` → state files gone.

## Notes

- **No new runtime deps.** `typer`, `rich`, `httpx`, `aio-pika`, `pyfzf`, `strif`, `anyio`
  already declared; `pytest-subprocess` already dev. Add via `uv add --dev` only if needed.
- **`fzf`** is a system prerequisite for `--select` only (lazy-imported); `tail`/`less` are
  standard system tools for `logs`.
- **Metrics-port collision** is the key gotcha — supervisor assigns `base+i` per child via
  env (Task 2).
- **Future (not now):** Celery-style `max-tasks-per-child` recycle and a watchdog
  auto-restart loop, if worker memory growth or crash-looping ever becomes a real problem.
