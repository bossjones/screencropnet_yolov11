"""``screencrop-supervisorctl`` — operate a fleet of host ``screencrop-worker`` processes.

Workers are competing consumers on one shared durable RabbitMQ queue and must run
on the host (they need MPS/CUDA), so ``docker-compose`` cannot scale them. This
module is the host-side manager: it computes per-worker specs (distinct metrics
port + log file + PID/state file), spawns detached workers, and does warm/cold
shutdown, status, and log tailing.

The module is split into a **pure core** (specs, state I/O, liveness, status
rendering, ``tail`` argv) that is unit-tested directly under ``tmp_path`` with no
processes, and **thin IO wrappers** (``spawn_worker``/``signal_worker``/
``run_logs``) that are patched in tests. It is deliberately torch-free so
importing it costs nothing.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from signal import SIGKILL, SIGTERM

import typer
from rich.console import Console
from rich.table import Table
from strif import atomic_output_file

from screencropnet_yolo.client.doctor import CheckResult, check_http
from screencropnet_yolo.client.serve import apply_weights_env, resolve_serve_weights
from screencropnet_yolo.server.config import Settings, get_settings

WEIGHTS_ENV = "SCREENCROPNET_WEIGHTS_PATH"
METRICS_PORT_ENV = "SCREENCROPNET_WORKER_METRICS_PORT"
LOG_PATH_ENV = "SCREENCROPNET_WORKER_LOG_PATH"


@dataclass(frozen=True)
class WorkerSpec:
    """Everything needed to launch and track one worker (pure, no process yet)."""

    name: str
    index: int
    metrics_port: int
    log_path: Path
    state_path: Path
    env: dict[str, str]


def _state_filename(name: str) -> str:
    """Map a worker name (``worker@3``) to its state file (``worker-3.json``)."""
    return f"{name.replace('@', '-')}.json"


def worker_specs(
    count: int,
    *,
    weights: Path,
    state_dir: Path,
    log_dir: Path,
    base_metrics_port: int,
) -> list[WorkerSpec]:
    """Build ``count`` worker specs with collision-free metrics ports (``base+i``).

    Each spec's ``env`` is ``os.environ`` merged with this worker's weights,
    metrics port, and log path, so a spawned ``screencrop-worker`` reads the right
    model and writes to its own port/file.
    """
    specs: list[WorkerSpec] = []
    for i in range(count):
        n = i + 1
        name = f"worker@{n}"
        metrics_port = base_metrics_port + i
        log_path = log_dir / f"worker-{n}.log"
        env = {
            **os.environ,
            WEIGHTS_ENV: str(weights),
            METRICS_PORT_ENV: str(metrics_port),
            LOG_PATH_ENV: str(log_path),
        }
        specs.append(
            WorkerSpec(
                name=name,
                index=i,
                metrics_port=metrics_port,
                log_path=log_path,
                state_path=state_dir / _state_filename(name),
                env=env,
            )
        )
    return specs


@dataclass(frozen=True)
class WorkerState:
    """Persisted record of a spawned worker (the source of truth for stop/status)."""

    name: str
    pid: int
    metrics_port: int
    weights_path: str
    log_path: str
    started_at: float


def write_state(state_dir: Path, state: WorkerState) -> Path:
    """Atomically write ``state`` as JSON under ``state_dir``; return its path."""
    path = state_dir / _state_filename(state.name)
    with atomic_output_file(path, make_parents=True) as tmp:
        tmp.write_text(json.dumps(asdict(state), indent=2, sort_keys=True))
    return path


def read_states(state_dir: Path) -> list[WorkerState]:
    """All persisted worker states under ``state_dir``, ordered by metrics port."""
    if not state_dir.is_dir():
        return []
    states = [
        WorkerState(**json.loads(path.read_text())) for path in state_dir.glob("worker-*.json")
    ]
    return sorted(states, key=lambda s: s.metrics_port)


def remove_state(state_dir: Path, state: WorkerState) -> None:
    """Delete ``state``'s file (no error if it is already gone)."""
    (state_dir / _state_filename(state.name)).unlink(missing_ok=True)


def is_alive(pid: int) -> bool:
    """Whether ``pid`` names a live process, via ``kill(pid, 0)``.

    ``ProcessLookupError`` means the pid is gone; ``PermissionError`` means the
    process exists but we may not signal it (still alive).
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass(frozen=True)
class StatusRow:
    """One rendered fleet row: state + liveness (+ optional probe)."""

    name: str
    pid: int
    alive: bool
    metrics_port: int
    weights_path: str
    uptime_s: float
    probe_ok: bool | None = None
    probe_detail: str = ""


def build_status_rows(
    states: list[WorkerState],
    liveness: dict[str, bool],
    *,
    now: float,
    probes: dict[str, CheckResult] | None = None,
) -> list[StatusRow]:
    """Merge persisted state with liveness (and optional probe results) into rows."""
    rows: list[StatusRow] = []
    for state in states:
        probe = (probes or {}).get(state.name)
        rows.append(
            StatusRow(
                name=state.name,
                pid=state.pid,
                alive=liveness.get(state.name, False),
                metrics_port=state.metrics_port,
                weights_path=state.weights_path,
                uptime_s=now - state.started_at,
                probe_ok=None if probe is None else probe.ok,
                probe_detail="" if probe is None else probe.detail,
            )
        )
    return rows


def _uptime_str(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def render_status_table(rows: list[StatusRow]) -> Table:
    """A rich table of ✔︎/✘ liveness, pid, port, uptime, weights (+ probe if present)."""
    show_probe = any(r.probe_ok is not None for r in rows)
    table = Table(title="supervisor")
    table.add_column("worker")
    table.add_column("alive", justify="center")
    table.add_column("pid", justify="right")
    table.add_column("port", justify="right")
    table.add_column("uptime", justify="right")
    table.add_column("weights", overflow="fold")
    if show_probe:
        table.add_column("probe", justify="center")
    for r in rows:
        glyph = "[green]✔︎[/green]" if r.alive else "[red]✘[/red]"
        cells = [
            r.name,
            glyph,
            str(r.pid),
            str(r.metrics_port),
            _uptime_str(r.uptime_s),
            r.weights_path,
        ]
        if show_probe:
            probe_glyph = (
                "" if r.probe_ok is None else "[green]✔︎[/green]" if r.probe_ok else "[red]✘[/red]"
            )
            cells.append(probe_glyph)
        table.add_row(*cells)
    return table


def render_status_json(rows: list[StatusRow]) -> str:
    """Serialize status rows as indented JSON (for ``status --json``)."""
    return json.dumps([asdict(r) for r in rows], indent=2)


# ---- Thin IO wrappers (patched in tests) -----------------------------------


def spawn_worker(spec: WorkerSpec) -> int:
    """Launch a detached ``screencrop-worker`` in ``spec.env``; return its pid.

    Stdio is sent to ``/dev/null`` (the worker logs to its own file) and the child
    is detached with ``start_new_session`` so it outlives the supervisor.
    """
    proc = subprocess.Popen(
        ["screencrop-worker"],
        env=spec.env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc.pid


def signal_worker(pid: int, sig: int) -> None:
    """Send ``sig`` to ``pid`` (SIGTERM for warm, SIGKILL for cold shutdown)."""
    os.kill(pid, sig)


def tail_command(state: WorkerState, *, follow: bool, lines: int, color: bool) -> list[str]:
    """Build the argv that shows ``state``'s log — ``tail``, piped to ``less -R`` for color."""
    tail = ["tail", "-n", str(lines)]
    if follow:
        tail.append("-f")
    tail.append(state.log_path)
    if color:
        return ["sh", "-c", f"{shlex.join(tail)} | less -R"]
    return tail


def run_logs(state: WorkerState, *, follow: bool, lines: int, color: bool) -> int:
    """Exec ``tail_command`` for ``state``; raise if the log file does not exist."""
    if not Path(state.log_path).exists():
        raise FileNotFoundError(f"no log file for {state.name}: {state.log_path}")
    cmd = tail_command(state, follow=follow, lines=lines, color=color)
    return subprocess.run(cmd, check=False).returncode


async def probe_worker(state: WorkerState, *, timeout: float) -> CheckResult:
    """Optional metrics-port reachability probe (only for ``status --probe``)."""
    url = f"http://127.0.0.1:{state.metrics_port}/"
    return await check_http(state.name, url, timeout=timeout)


# ---- Lifecycle orchestration (pure over injected IO) -----------------------

SpawnFn = Callable[[WorkerSpec], int]
SignalFn = Callable[[int, int], None]
AliveFn = Callable[[int], bool]


def start_fleet(
    count: int,
    *,
    weights: Path,
    settings: Settings,
    spawn: SpawnFn = spawn_worker,
    now: Callable[[], float] = time.time,
) -> list[WorkerState]:
    """Spawn ``count`` workers (logs + state under ``supervisor_state_dir``) and persist state."""
    state_dir = settings.supervisor_state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    specs = worker_specs(
        count,
        weights=weights,
        state_dir=state_dir,
        log_dir=state_dir,
        base_metrics_port=settings.supervisor_metrics_base_port,
    )
    states: list[WorkerState] = []
    for spec in specs:
        pid = spawn(spec)
        state = WorkerState(
            name=spec.name,
            pid=pid,
            metrics_port=spec.metrics_port,
            weights_path=str(weights),
            log_path=str(spec.log_path),
            started_at=now(),
        )
        write_state(state_dir, state)
        states.append(state)
    return states


def stop_fleet(
    states: list[WorkerState],
    *,
    state_dir: Path,
    timeout: float,
    signal: SignalFn = signal_worker,
    alive: AliveFn = is_alive,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = 0.2,
) -> None:
    """Warm-shutdown ``states`` (SIGTERM + drain up to ``timeout``), cold-kill stragglers, clear state.

    SIGTERM everyone, poll ``alive`` until all exit or the timeout elapses, then
    SIGKILL any survivor. State files are removed unconditionally at the end.
    """
    for state in states:
        signal(state.pid, SIGTERM)

    remaining: list[WorkerState] = list(states)
    waited = 0.0
    while remaining and waited < timeout:
        remaining = [s for s in remaining if alive(s.pid)]
        if not remaining:
            break
        sleep(poll_interval)
        waited += poll_interval

    for state in remaining:
        if alive(state.pid):
            signal(state.pid, SIGKILL)

    for state in states:
        remove_state(state_dir, state)


def restart_fleet(
    settings: Settings,
    *,
    timeout: float = 30.0,
    spawn: SpawnFn = spawn_worker,
    signal: SignalFn = signal_worker,
    alive: AliveFn = is_alive,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.time,
) -> list[WorkerState]:
    """Reconstruct the fleet from persisted state: stop it, then re-spawn the same count/weights."""
    state_dir = settings.supervisor_state_dir
    states = read_states(state_dir)
    if not states:
        return []
    weights = Path(states[0].weights_path)
    stop_fleet(
        states, state_dir=state_dir, timeout=timeout, signal=signal, alive=alive, sleep=sleep
    )
    return start_fleet(len(states), weights=weights, settings=settings, spawn=spawn, now=now)


# ---- Typer CLI --------------------------------------------------------------

app = typer.Typer(help="Operate a fleet of host screencrop-worker processes (competing consumers).")
console = Console()


async def _probe_all(states: list[WorkerState], *, timeout: float) -> list[CheckResult]:
    return list(await asyncio.gather(*(probe_worker(s, timeout=timeout) for s in states)))


def _resolve_targets(settings: Settings, *, name: str | None, all_: bool) -> list[WorkerState]:
    """Select the workers a stop/logs command applies to, or exit with a clear error."""
    states = read_states(settings.supervisor_state_dir)
    if all_:
        return states
    if name is None:
        console.print("[red]✘[/red] specify a worker NAME or --all")
        raise typer.Exit(code=2)
    matched = [s for s in states if s.name == name]
    if not matched:
        console.print(f"[red]✘[/red] no such worker: {name}")
        raise typer.Exit(code=1)
    return matched


@app.command()
def start(
    workers: int | None = typer.Option(
        None, "--workers", "-w", help="Number of workers (default: settings.supervisor_workers)."
    ),
    select: bool = typer.Option(False, "--select", "--fuzzy", help="Fuzzy-pick weights via fzf."),
    model: str | None = typer.Option(
        None, "--model", help="Explicit weights path (overrides --select and the default)."
    ),
    prefetch: int | None = typer.Option(
        None,
        "--prefetch",
        help="Per-worker RabbitMQ prefetch (SCREENCROPNET_RABBIT_PREFETCH_COUNT).",
    ),
    metrics_base_port: int | None = typer.Option(
        None, "--metrics-base-port", help="Base metrics port; worker i listens on base+i."
    ),
) -> None:
    """Spawn N detached workers (each with its own metrics port + log + state file)."""
    settings = get_settings()
    count = workers if workers is not None else settings.supervisor_workers
    if metrics_base_port is not None:
        settings = settings.model_copy(update={"supervisor_metrics_base_port": metrics_base_port})

    try:
        weights = (
            Path(model)
            if model is not None
            else resolve_serve_weights(select=select, settings=settings)
        )
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]✘[/red] {exc}")
        raise typer.Exit(code=1) from exc

    apply_weights_env(weights)
    if prefetch is not None:
        os.environ["SCREENCROPNET_RABBIT_PREFETCH_COUNT"] = str(prefetch)

    states = start_fleet(count, weights=weights, settings=settings, spawn=spawn_worker)
    console.print(f"started {len(states)} worker(s) from {weights}")
    for s in states:
        console.print(f"  {s.name}  pid={s.pid}  port={s.metrics_port}")


@app.command()
def stop(
    name: str | None = typer.Argument(None, help="Worker name (e.g. worker@1); omit with --all."),
    all_: bool = typer.Option(False, "--all", help="Stop the entire fleet."),
    timeout: float = typer.Option(30.0, "--timeout", help="Warm-shutdown grace before cold kill."),
    cold: bool = typer.Option(False, "--cold", help="Skip the grace period; kill immediately."),
) -> None:
    """Warm-shutdown workers (SIGTERM, drain up to --timeout) then cold-kill stragglers."""
    settings = get_settings()
    states = _resolve_targets(settings, name=name, all_=all_)
    stop_fleet(
        states,
        state_dir=settings.supervisor_state_dir,
        timeout=0.0 if cold else timeout,
        signal=signal_worker,
        alive=is_alive,
    )
    console.print(f"stopped {len(states)} worker(s)")


@app.command()
def restart(
    name: str | None = typer.Argument(
        None, help="Accepted for symmetry; restart reconstructs the whole fleet."
    ),
    all_: bool = typer.Option(False, "--all", help="Restart the entire fleet."),
    timeout: float = typer.Option(30.0, "--timeout", help="Warm-shutdown grace before cold kill."),
) -> None:
    """Stop the fleet and re-spawn it from persisted state (same count + weights)."""
    settings = get_settings()
    if not all_ and name is None:
        console.print("[red]✘[/red] specify a worker NAME or --all")
        raise typer.Exit(code=2)
    states = restart_fleet(
        settings, timeout=timeout, spawn=spawn_worker, signal=signal_worker, alive=is_alive
    )
    console.print(f"restarted {len(states)} worker(s)")


@app.command()
def status(
    json_: bool = typer.Option(False, "--json", help="Emit raw JSON instead of a table."),
    probe: bool = typer.Option(False, "--probe", help="Also probe each worker's metrics port."),
) -> None:
    """Show per-worker PID liveness + metadata (table or JSON); --probe adds reachability."""
    settings = get_settings()
    states = read_states(settings.supervisor_state_dir)
    liveness = {s.name: is_alive(s.pid) for s in states}
    probes: dict[str, CheckResult] | None = None
    if probe:
        results = asyncio.run(_probe_all(states, timeout=settings.doctor_timeout))
        probes = {r.name: r for r in results}
    rows = build_status_rows(states, liveness, now=time.time(), probes=probes)
    if json_:
        typer.echo(render_status_json(rows))
    else:
        console.print(render_status_table(rows))


@app.command()
def logs(
    name: str = typer.Argument(..., help="Worker name, e.g. worker@1."),
    follow: bool = typer.Option(False, "-f", "--follow", help="Keep streaming new lines."),
    lines: int = typer.Option(200, "-n", "--lines", help="How many trailing lines to show."),
    color: bool = typer.Option(False, "--color/--plain", help="Pipe through less -R for color."),
) -> None:
    """Tail a worker's log via ``tail`` (piped to ``less -R`` when --color)."""
    settings = get_settings()
    matched = [s for s in read_states(settings.supervisor_state_dir) if s.name == name]
    if not matched:
        console.print(f"[red]✘[/red] no such worker: {name}")
        raise typer.Exit(code=1)
    try:
        run_logs(matched[0], follow=follow, lines=lines, color=color)
    except FileNotFoundError as exc:
        console.print(f"[red]✘[/red] {exc}")
        raise typer.Exit(code=1) from exc


def main() -> None:
    app()
