from __future__ import annotations

from pytest_mock import MockerFixture
from typer.testing import CliRunner

from screencropnet_yolo.client import cli
from screencropnet_yolo.server.schemas import ExportRecord, JobView, StatusSummary

runner = CliRunner()


def _mock_client(mocker: MockerFixture):
    client = mocker.MagicMock()
    client.aclose = mocker.AsyncMock()
    return client


def test_submit_is_fire_and_forget(mocker: MockerFixture) -> None:
    popen = mocker.patch("screencropnet_yolo.client.cli.subprocess.Popen")
    result = runner.invoke(cli.app, ["submit", "/tmp/folder", "--batch-id", "b1"])
    if result.exit_code != 0:
        raise AssertionError(f"submit failed: {result.output}")
    if "b1" not in result.output:
        raise AssertionError("submit must print the batch_id")
    if popen.call_count != 1:
        raise AssertionError("submit must launch exactly one detached subprocess")
    cmd = popen.call_args.args[0]
    if "_submit-worker" not in cmd or "/tmp/folder" not in cmd or "b1" not in cmd:
        raise AssertionError(f"detached command missing required args: {cmd}")


def test_submit_worker_runs_submit_folder(mocker: MockerFixture) -> None:
    client = _mock_client(mocker)
    client.submit_folder = mocker.AsyncMock(return_value=[])
    mocker.patch("screencropnet_yolo.client.cli.build_client", return_value=client)
    result = runner.invoke(cli.app, ["_submit-worker", "--folder", "/tmp/f", "--batch-id", "b1"])
    if result.exit_code != 0:
        raise AssertionError(f"_submit-worker failed: {result.output}")
    client.submit_folder.assert_awaited_once_with("/tmp/f", "b1", recursive=True)


def test_status_renders_summary(mocker: MockerFixture) -> None:
    client = _mock_client(mocker)
    client.status = mocker.AsyncMock(
        return_value=StatusSummary(
            batch_id="b1",
            total=3,
            counts={"pending": 1, "processing": 0, "done": 2, "failed": 0},
            twitter_count=2,
            done=2,
            failed=0,
            throughput_per_sec=1.5,
        )
    )
    mocker.patch("screencropnet_yolo.client.cli.build_client", return_value=client)
    result = runner.invoke(cli.app, ["status", "--batch-id", "b1"])
    if result.exit_code != 0:
        raise AssertionError(f"status failed: {result.output}")
    client.status.assert_awaited_once_with("b1")
    if "3" not in result.output:
        raise AssertionError("status output must include the total count")


def test_submitted_lists_jobs(mocker: MockerFixture) -> None:
    client = _mock_client(mocker)
    client.list_jobs = mocker.AsyncMock(
        return_value=[
            JobView(job_id="j1", batch_id="b1", original_path="/o/x.png", status="pending")
        ]
    )
    mocker.patch("screencropnet_yolo.client.cli.build_client", return_value=client)
    result = runner.invoke(cli.app, ["submitted", "--batch-id", "b1"])
    if result.exit_code != 0:
        raise AssertionError(f"submitted failed: {result.output}")
    client.list_jobs.assert_awaited_once_with(batch_id="b1")
    if "j1" not in result.output:
        raise AssertionError("submitted must render the job id")


def test_twitter_lists_positives(mocker: MockerFixture) -> None:
    client = _mock_client(mocker)
    client.list_twitter = mocker.AsyncMock(
        return_value=[
            JobView(
                job_id="j2",
                batch_id="b1",
                original_path="/o/y.png",
                status="done",
                is_twitter=True,
                pred_class="twitter",
                pred_prob=0.95,
            )
        ]
    )
    mocker.patch("screencropnet_yolo.client.cli.build_client", return_value=client)
    result = runner.invoke(cli.app, ["twitter", "--batch-id", "b1"])
    if result.exit_code != 0:
        raise AssertionError(f"twitter failed: {result.output}")
    client.list_twitter.assert_awaited_once_with(batch_id="b1")


def test_export_invokes_export_originals(mocker: MockerFixture) -> None:
    client = _mock_client(mocker)
    jobs = [
        JobView(
            job_id="j2",
            batch_id="b1",
            original_path="/o/y.png",
            status="done",
            is_twitter=True,
            pred_class="twitter",
            pred_prob=0.95,
        )
    ]
    client.list_twitter = mocker.AsyncMock(return_value=jobs)
    mocker.patch("screencropnet_yolo.client.cli.build_client", return_value=client)
    export_mock = mocker.patch(
        "screencropnet_yolo.client.cli.export_originals",
        return_value=[
            ExportRecord(
                original_path="/o/y.png",
                dest_path="/ds/01495_twitter.png",
                index=1495,
                copied=False,
                reason="dry_run",
            )
        ],
    )
    result = runner.invoke(cli.app, ["export", "--batch-id", "b1", "--dry-run"])
    if result.exit_code != 0:
        raise AssertionError(f"export failed: {result.output}")
    if export_mock.call_count != 1:
        raise AssertionError("export must call export_originals once")
    passed_jobs = export_mock.call_args.args[0]
    if passed_jobs != jobs:
        raise AssertionError("export must pass the resolved twitter-positive jobs")
    if export_mock.call_args.kwargs.get("dry_run") is not True:
        raise AssertionError("--dry-run must be threaded through to export_originals")
