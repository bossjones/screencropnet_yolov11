from __future__ import annotations

import json
import signal
from pathlib import Path

import pytest
from pytest_mock import MockerFixture
from typer.testing import CliRunner

from screencropnet_yolo.client import supervisor
from screencropnet_yolo.client.doctor import CheckResult
from screencropnet_yolo.server.config import Settings

runner = CliRunner()


def test_worker_specs_assigns_distinct_ports_and_paths(tmp_path: Path) -> None:
    specs = supervisor.worker_specs(
        3,
        weights=Path("/w/model.pth"),
        state_dir=tmp_path / "state",
        log_dir=tmp_path / "logs",
        base_metrics_port=8001,
    )
    if [s.name for s in specs] != ["worker@1", "worker@2", "worker@3"]:
        raise AssertionError("workers must be named worker@1..N")
    ports = [s.metrics_port for s in specs]
    if ports != [8001, 8002, 8003]:
        raise AssertionError(f"ports must be base+i (collision-free), got {ports}")
    if len(set(ports)) != len(ports):
        raise AssertionError("metrics ports must be unique across the fleet")
    if specs[2].log_path != tmp_path / "logs" / "worker-3.log":
        raise AssertionError("per-worker log path must be log_dir/worker-<n>.log")
    if specs[2].state_path != tmp_path / "state" / "worker-3.json":
        raise AssertionError("per-worker state path must be state_dir/worker-<n>.json")


def test_worker_specs_env_carries_overrides_merged_onto_environ(tmp_path: Path) -> None:
    specs = supervisor.worker_specs(
        1,
        weights=Path("/w/m.pth"),
        state_dir=tmp_path / "s",
        log_dir=tmp_path / "l",
        base_metrics_port=9000,
    )
    env = specs[0].env
    if env["SCREENCROPNET_WEIGHTS_PATH"] != "/w/m.pth":
        raise AssertionError("spec env must export the resolved weights path")
    if env["SCREENCROPNET_WORKER_METRICS_PORT"] != "9000":
        raise AssertionError("spec env must export this worker's metrics port")
    if env["SCREENCROPNET_WORKER_LOG_PATH"] != str(tmp_path / "l" / "worker-1.log"):
        raise AssertionError("spec env must export this worker's log path")
    if "PATH" not in env:
        raise AssertionError("spec env must be merged onto os.environ (PATH should survive)")


def _sample_state(tmp_path: Path) -> supervisor.WorkerState:
    return supervisor.WorkerState(
        name="worker@1",
        pid=4242,
        metrics_port=8001,
        weights_path="/w/m.pth",
        log_path=str(tmp_path / "l" / "worker-1.log"),
        started_at=1000.0,
    )


def test_state_write_read_remove_round_trip(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state = _sample_state(tmp_path)

    path = supervisor.write_state(state_dir, state)
    if not path.is_file():
        raise AssertionError("write_state must create the state file")
    if supervisor.read_states(state_dir) != [state]:
        raise AssertionError("read_states must round-trip the written state")

    supervisor.remove_state(state_dir, state)
    if supervisor.read_states(state_dir) != []:
        raise AssertionError("remove_state must delete the state file")


def test_read_states_missing_dir_returns_empty(tmp_path: Path) -> None:
    if supervisor.read_states(tmp_path / "never-created") != []:
        raise AssertionError("a missing state dir must read as an empty fleet")


def test_read_states_sorted_by_port(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    for i in (2, 0, 1):
        supervisor.write_state(
            state_dir,
            supervisor.WorkerState(
                name=f"worker@{i + 1}",
                pid=100 + i,
                metrics_port=8001 + i,
                weights_path="/w/m.pth",
                log_path=f"/l/worker-{i + 1}.log",
                started_at=float(i),
            ),
        )
    ports = [s.metrics_port for s in supervisor.read_states(state_dir)]
    if ports != [8001, 8002, 8003]:
        raise AssertionError(f"read_states must be ordered by metrics port, got {ports}")


def test_is_alive_true_when_kill_succeeds(mocker: MockerFixture) -> None:
    mocker.patch("screencropnet_yolo.client.supervisor.os.kill")
    if not supervisor.is_alive(123):
        raise AssertionError("a signalable pid must read as alive")


def test_is_alive_false_on_process_lookup(mocker: MockerFixture) -> None:
    mocker.patch("screencropnet_yolo.client.supervisor.os.kill", side_effect=ProcessLookupError)
    if supervisor.is_alive(123):
        raise AssertionError("a stale pid (ProcessLookupError) must read as dead")


def test_is_alive_true_on_permission_error(mocker: MockerFixture) -> None:
    mocker.patch("screencropnet_yolo.client.supervisor.os.kill", side_effect=PermissionError)
    if not supervisor.is_alive(123):
        raise AssertionError("a pid we may not signal (PermissionError) is still alive")


def _state(
    name: str = "worker@1", *, port: int = 8001, started: float = 1000.0
) -> supervisor.WorkerState:
    return supervisor.WorkerState(
        name=name,
        pid=42,
        metrics_port=port,
        weights_path="/w/m.pth",
        log_path="/l/w1.log",
        started_at=started,
    )


def test_build_status_rows_merges_liveness_and_uptime() -> None:
    rows = supervisor.build_status_rows([_state()], {"worker@1": True}, now=1060.0)
    row = rows[0]
    if row.name != "worker@1" or not row.alive:
        raise AssertionError("row must reflect the state name and liveness")
    if row.uptime_s != 60.0:
        raise AssertionError(f"uptime must be now - started_at, got {row.uptime_s}")
    if row.probe_ok is not None:
        raise AssertionError("probe_ok must be None when no probes are supplied")


def test_build_status_rows_includes_probe_when_supplied() -> None:
    probes = {"worker@1": CheckResult("worker@1", True, "HTTP 200", latency_ms=3.0)}
    rows = supervisor.build_status_rows([_state()], {"worker@1": False}, now=1000.0, probes=probes)
    if rows[0].probe_ok is not True:
        raise AssertionError("probe_ok must reflect the CheckResult")
    if "HTTP 200" not in rows[0].probe_detail:
        raise AssertionError("probe_detail must carry the CheckResult detail")


def test_render_status_json_is_valid_and_complete() -> None:
    rows = supervisor.build_status_rows([_state()], {"worker@1": True}, now=1000.0)
    data = json.loads(supervisor.render_status_json(rows))
    if data[0]["name"] != "worker@1" or data[0]["alive"] is not True:
        raise AssertionError("render_status_json must emit each row's fields")


def test_render_status_table_has_one_row_per_worker() -> None:
    from rich.table import Table

    rows = supervisor.build_status_rows(
        [_state("worker@1", port=8001), _state("worker@2", port=8002)],
        {"worker@1": True, "worker@2": False},
        now=1000.0,
    )
    table = supervisor.render_status_table(rows)
    if not isinstance(table, Table):
        raise AssertionError("render_status_table must return a rich Table")
    if table.row_count != 2:
        raise AssertionError("the table must have one row per worker")


def test_spawn_worker_launches_detached_with_spec_env(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    specs = supervisor.worker_specs(
        1,
        weights=Path("/w/m.pth"),
        state_dir=tmp_path / "s",
        log_dir=tmp_path / "l",
        base_metrics_port=8001,
    )
    popen = mocker.patch("screencropnet_yolo.client.supervisor.subprocess.Popen")
    popen.return_value = mocker.Mock(pid=4321)

    pid = supervisor.spawn_worker(specs[0])

    if pid != 4321:
        raise AssertionError("spawn_worker must return the child pid")
    if popen.call_args.args[0] != ["screencrop-worker"]:
        raise AssertionError("spawn_worker must exec the screencrop-worker console script")
    kwargs = popen.call_args.kwargs
    if kwargs.get("start_new_session") is not True:
        raise AssertionError("workers must be detached into a new session")
    if kwargs["env"]["SCREENCROPNET_WORKER_METRICS_PORT"] != "8001":
        raise AssertionError("the child must inherit this worker's spec env")


def test_signal_worker_delegates_to_os_kill(mocker: MockerFixture) -> None:
    kill = mocker.patch("screencropnet_yolo.client.supervisor.os.kill")
    supervisor.signal_worker(999, signal.SIGTERM)
    kill.assert_called_once_with(999, signal.SIGTERM)


def test_tail_command_plain() -> None:
    cmd = supervisor.tail_command(_state(), follow=False, lines=50, color=False)
    if cmd != ["tail", "-n", "50", "/l/w1.log"]:
        raise AssertionError(f"plain tail argv is wrong: {cmd}")


def test_tail_command_follow() -> None:
    cmd = supervisor.tail_command(_state(), follow=True, lines=100, color=False)
    if cmd != ["tail", "-n", "100", "-f", "/l/w1.log"]:
        raise AssertionError(f"follow tail argv is wrong: {cmd}")


def test_tail_command_color_pipes_through_less() -> None:
    cmd = supervisor.tail_command(_state(), follow=True, lines=20, color=True)
    if cmd[:2] != ["sh", "-c"]:
        raise AssertionError("color output must run through a shell pipeline")
    if "less -R" not in cmd[2] or "tail -n 20 -f" not in cmd[2]:
        raise AssertionError(f"color pipeline must tail into less -R: {cmd[2]!r}")


def test_run_logs_execs_tail_argv(mocker: MockerFixture, tmp_path: Path) -> None:
    log = tmp_path / "w.log"
    log.write_text("hi")
    state = supervisor.WorkerState(
        name="worker@1",
        pid=1,
        metrics_port=8001,
        weights_path="/w/m.pth",
        log_path=str(log),
        started_at=0.0,
    )
    run = mocker.patch("screencropnet_yolo.client.supervisor.subprocess.run")
    run.return_value = mocker.Mock(returncode=0)

    supervisor.run_logs(state, follow=True, lines=10, color=False)

    if run.call_args.args[0] != ["tail", "-n", "10", "-f", str(log)]:
        raise AssertionError(f"run_logs must exec the tail argv: {run.call_args.args[0]}")


def test_run_logs_missing_log_errors_cleanly(tmp_path: Path) -> None:
    state = supervisor.WorkerState(
        name="worker@1",
        pid=1,
        metrics_port=8001,
        weights_path="/w/m.pth",
        log_path=str(tmp_path / "never.log"),
        started_at=0.0,
    )
    with pytest.raises(FileNotFoundError):
        supervisor.run_logs(state, follow=False, lines=10, color=False)


async def test_probe_worker_reuses_check_http(mocker: MockerFixture) -> None:
    fake = mocker.patch(
        "screencropnet_yolo.client.supervisor.check_http",
        new=mocker.AsyncMock(return_value=CheckResult("worker@1", True, "HTTP 200")),
    )
    result = await supervisor.probe_worker(_state(port=8007), timeout=1.0)
    if result.ok is not True:
        raise AssertionError("probe_worker must return the CheckResult from check_http")
    fake.assert_awaited_once()
    if "8007" not in fake.call_args.args[1]:
        raise AssertionError("probe_worker must target this worker's metrics port")


def _fleet_settings(tmp_path: Path, *, base: int = 8001) -> Settings:
    return Settings(supervisor_state_dir=tmp_path / "state", supervisor_metrics_base_port=base)


def test_start_fleet_spawns_and_writes_state(tmp_path: Path) -> None:
    settings = _fleet_settings(tmp_path)
    pids = iter([11, 12, 13])
    states = supervisor.start_fleet(
        3,
        weights=Path("/w/m.pth"),
        settings=settings,
        spawn=lambda _spec: next(pids),
        now=lambda: 1000.0,
    )
    if [s.pid for s in states] != [11, 12, 13]:
        raise AssertionError("start_fleet must record each spawned pid")
    written = supervisor.read_states(tmp_path / "state")
    if len(written) != 3:
        raise AssertionError("start_fleet must persist one state file per worker")
    if {s.metrics_port for s in written} != {8001, 8002, 8003}:
        raise AssertionError("each worker must get a distinct base+i metrics port")


def test_stop_fleet_warm_shutdown_does_not_cold_kill(tmp_path: Path) -> None:
    sent: list[int] = []
    checks = {"n": 0}

    def alive(_pid: int) -> bool:
        checks["n"] += 1
        return checks["n"] <= 1  # alive on the first poll, gone after

    state = _state()
    supervisor.write_state(tmp_path / "state", state)
    supervisor.stop_fleet(
        [state],
        state_dir=tmp_path / "state",
        timeout=1.0,
        signal=lambda _pid, sig: sent.append(sig),
        alive=alive,
        sleep=lambda _s: None,
    )
    if signal.SIGKILL in sent:
        raise AssertionError("a worker that exits under timeout must not be cold-killed")
    if signal.SIGTERM not in sent:
        raise AssertionError("warm shutdown must send SIGTERM")


def test_stop_fleet_cold_kills_stragglers_and_clears_state(tmp_path: Path) -> None:
    sent: list[int] = []
    state = _state()
    supervisor.write_state(tmp_path / "state", state)
    supervisor.stop_fleet(
        [state],
        state_dir=tmp_path / "state",
        timeout=0.4,
        signal=lambda _pid, sig: sent.append(sig),
        alive=lambda _pid: True,  # never dies
        sleep=lambda _s: None,
        poll_interval=0.2,
    )
    if signal.SIGTERM not in sent or signal.SIGKILL not in sent:
        raise AssertionError("a straggler must be SIGTERM'd then SIGKILL'd")
    if supervisor.read_states(tmp_path / "state") != []:
        raise AssertionError("stop_fleet must clear state files")


def test_restart_fleet_stops_then_respawns_same_count(tmp_path: Path) -> None:
    settings = _fleet_settings(tmp_path)
    supervisor.start_fleet(
        2, weights=Path("/w/m.pth"), settings=settings, spawn=lambda _s: 100, now=lambda: 1.0
    )
    stopped: list[int] = []
    spawned: list[str] = []
    states = supervisor.restart_fleet(
        settings,
        spawn=lambda spec: (spawned.append(spec.name), 200)[1],
        signal=lambda pid, _sig: stopped.append(pid),
        alive=lambda _pid: False,
        sleep=lambda _s: None,
        now=lambda: 2.0,
    )
    if len(stopped) != 2:
        raise AssertionError("restart must SIGTERM the 2 prior workers")
    if len(spawned) != 2 or len(states) != 2:
        raise AssertionError("restart must respawn the same number of workers")
    if [s.pid for s in states] != [200, 200]:
        raise AssertionError("restart must record the freshly spawned pids")


def test_restart_fleet_empty_when_no_prior_state(tmp_path: Path) -> None:
    settings = _fleet_settings(tmp_path)
    states = supervisor.restart_fleet(
        settings,
        spawn=lambda _s: 1,
        signal=lambda _p, _s: None,
        alive=lambda _p: False,
        sleep=lambda _s: None,
    )
    if states != []:
        raise AssertionError("restart with no prior state must be a no-op returning []")


def _seed_states(state_dir: Path, count: int) -> None:
    for i in range(count):
        supervisor.write_state(
            state_dir,
            supervisor.WorkerState(
                name=f"worker@{i + 1}",
                pid=100 + i,
                metrics_port=8001 + i,
                weights_path="/w/m.pth",
                log_path=str(state_dir / f"worker-{i + 1}.log"),
                started_at=0.0,
            ),
        )


def test_cli_start_spawns_n_workers(mocker: MockerFixture, tmp_path: Path) -> None:
    settings = _fleet_settings(tmp_path)
    weights = tmp_path / "m.pth"
    weights.write_bytes(b"x")
    mocker.patch("screencropnet_yolo.client.supervisor.get_settings", return_value=settings)
    mocker.patch("screencropnet_yolo.client.supervisor.resolve_serve_weights", return_value=weights)
    mocker.patch("screencropnet_yolo.client.supervisor.apply_weights_env")
    spawn = mocker.patch(
        "screencropnet_yolo.client.supervisor.spawn_worker", side_effect=[101, 102]
    )

    result = runner.invoke(supervisor.app, ["start", "-w", "2"])

    if result.exit_code != 0:
        raise AssertionError(f"start failed: {result.output}")
    if spawn.call_count != 2:
        raise AssertionError("start -w 2 must spawn two workers")
    if len(supervisor.read_states(tmp_path / "state")) != 2:
        raise AssertionError("start must persist a state file per worker")


def test_cli_start_fuzzy_resolves_with_select(mocker: MockerFixture, tmp_path: Path) -> None:
    settings = _fleet_settings(tmp_path)
    mocker.patch("screencropnet_yolo.client.supervisor.get_settings", return_value=settings)
    resolve = mocker.patch(
        "screencropnet_yolo.client.supervisor.resolve_serve_weights",
        return_value=tmp_path / "m.pth",
    )
    mocker.patch("screencropnet_yolo.client.supervisor.apply_weights_env")
    mocker.patch("screencropnet_yolo.client.supervisor.spawn_worker", return_value=1)

    result = runner.invoke(supervisor.app, ["start", "-w", "1", "--fuzzy"])

    if result.exit_code != 0:
        raise AssertionError(f"start --fuzzy failed: {result.output}")
    if resolve.call_args.kwargs.get("select") is not True:
        raise AssertionError("--fuzzy must resolve weights with select=True")


def test_cli_start_cancelled_pick_exits_nonzero(mocker: MockerFixture, tmp_path: Path) -> None:
    settings = _fleet_settings(tmp_path)
    mocker.patch("screencropnet_yolo.client.supervisor.get_settings", return_value=settings)
    mocker.patch(
        "screencropnet_yolo.client.supervisor.resolve_serve_weights",
        side_effect=RuntimeError("Model selection cancelled"),
    )
    spawn = mocker.patch("screencropnet_yolo.client.supervisor.spawn_worker")

    result = runner.invoke(supervisor.app, ["start", "--select"])

    if result.exit_code == 0:
        raise AssertionError("a cancelled fuzzy pick must exit non-zero")
    if spawn.called:
        raise AssertionError("no worker may spawn when weight selection is cancelled")


def test_cli_stop_all_signals_and_clears_state(mocker: MockerFixture, tmp_path: Path) -> None:
    settings = _fleet_settings(tmp_path)
    mocker.patch("screencropnet_yolo.client.supervisor.get_settings", return_value=settings)
    _seed_states(tmp_path / "state", 2)
    sig = mocker.patch("screencropnet_yolo.client.supervisor.signal_worker")
    mocker.patch("screencropnet_yolo.client.supervisor.is_alive", return_value=False)

    result = runner.invoke(supervisor.app, ["stop", "--all"])

    if result.exit_code != 0:
        raise AssertionError(f"stop --all failed: {result.output}")
    if sig.call_count < 2:
        raise AssertionError("stop --all must signal every worker")
    if supervisor.read_states(tmp_path / "state") != []:
        raise AssertionError("stop --all must clear all state files")


def test_cli_stop_requires_name_or_all(mocker: MockerFixture, tmp_path: Path) -> None:
    settings = _fleet_settings(tmp_path)
    mocker.patch("screencropnet_yolo.client.supervisor.get_settings", return_value=settings)
    result = runner.invoke(supervisor.app, ["stop"])
    if result.exit_code == 0:
        raise AssertionError("stop with neither NAME nor --all must be an error")


def test_cli_status_json_lists_workers(mocker: MockerFixture, tmp_path: Path) -> None:
    settings = _fleet_settings(tmp_path)
    mocker.patch("screencropnet_yolo.client.supervisor.get_settings", return_value=settings)
    _seed_states(tmp_path / "state", 1)
    mocker.patch("screencropnet_yolo.client.supervisor.is_alive", return_value=True)

    result = runner.invoke(supervisor.app, ["status", "--json"])

    if result.exit_code != 0:
        raise AssertionError(f"status --json failed: {result.output}")
    if '"name": "worker@1"' not in result.output:
        raise AssertionError("status --json must emit raw JSON with each worker")


def test_cli_logs_execs_run_logs_with_flags(mocker: MockerFixture, tmp_path: Path) -> None:
    settings = _fleet_settings(tmp_path)
    mocker.patch("screencropnet_yolo.client.supervisor.get_settings", return_value=settings)
    _seed_states(tmp_path / "state", 1)
    run = mocker.patch("screencropnet_yolo.client.supervisor.run_logs", return_value=0)

    result = runner.invoke(supervisor.app, ["logs", "worker@1", "-n", "5", "--follow"])

    if result.exit_code != 0:
        raise AssertionError(f"logs failed: {result.output}")
    kwargs = run.call_args.kwargs
    if kwargs.get("follow") is not True or kwargs.get("lines") != 5:
        raise AssertionError(f"logs must thread -n/-f into run_logs: {kwargs}")


def test_cli_logs_unknown_worker_errors_cleanly(mocker: MockerFixture, tmp_path: Path) -> None:
    settings = _fleet_settings(tmp_path)
    mocker.patch("screencropnet_yolo.client.supervisor.get_settings", return_value=settings)
    result = runner.invoke(supervisor.app, ["logs", "worker@9"])
    if result.exit_code == 0:
        raise AssertionError("logs on a never-started worker must error cleanly")


def test_cli_restart_all_reconstructs_fleet(mocker: MockerFixture, tmp_path: Path) -> None:
    settings = _fleet_settings(tmp_path)
    mocker.patch("screencropnet_yolo.client.supervisor.get_settings", return_value=settings)
    _seed_states(tmp_path / "state", 2)
    mocker.patch("screencropnet_yolo.client.supervisor.signal_worker")
    mocker.patch("screencropnet_yolo.client.supervisor.is_alive", return_value=False)
    spawn = mocker.patch(
        "screencropnet_yolo.client.supervisor.spawn_worker", side_effect=[201, 202]
    )

    result = runner.invoke(supervisor.app, ["restart", "--all"])

    if result.exit_code != 0:
        raise AssertionError(f"restart --all failed: {result.output}")
    if spawn.call_count != 2:
        raise AssertionError("restart --all must respawn the same number of workers")
