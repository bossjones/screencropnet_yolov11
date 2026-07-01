from __future__ import annotations

from pytest_mock import MockerFixture

from screencropnet_yolo.client import tui
from screencropnet_yolo.client.tui import TopSnapshot, build_snapshot
from screencropnet_yolo.server.schemas import JobView, StatusSummary


def _summary() -> StatusSummary:
    return StatusSummary(
        batch_id="b1",
        total=3,
        counts={"pending": 1, "processing": 0, "done": 2, "failed": 0},
        twitter_count=2,
        done=2,
        failed=0,
        throughput_per_sec=1.5,
    )


def _jobs(n: int) -> list[JobView]:
    return [
        JobView(
            job_id=f"job-{i:04d}-abcdef",
            batch_id="b1",
            original_path=f"/o/x{i}.png",
            status="done",
            is_twitter=(i % 2 == 0),
            pred_class="twitter",
        )
        for i in range(n)
    ]


def test_build_snapshot_maps_summary_and_jobs() -> None:
    snap = build_snapshot(_summary(), _jobs(2))
    if snap.batch != "b1" or snap.total != 3 or snap.twitter_count != 2:
        raise AssertionError(f"header fields wrong: {snap}")
    if ("done", 2) not in snap.status_counts:
        raise AssertionError("status counts must include done=2")
    if len(snap.jobs) != 2:
        raise AssertionError("all jobs (under max) must be present")
    if snap.jobs[0].twitter != "True":
        raise AssertionError("is_twitter must render as a string flag")


def test_build_snapshot_truncates_and_shortens_ids() -> None:
    snap = build_snapshot(_summary(), _jobs(25), max_rows=10)
    if len(snap.jobs) != 10:
        raise AssertionError("jobs must be capped at max_rows")
    if snap.truncated != 15:
        raise AssertionError(f"truncated count wrong: {snap.truncated}")
    if len(snap.jobs[0].job_id) >= len("job-0000-abcdef"):
        raise AssertionError("job id should be shortened for display")


def test_build_snapshot_batch_none_renders_all() -> None:
    summary = _summary().model_copy(update={"batch_id": None})
    snap = build_snapshot(summary, [])
    if snap.batch != "all":
        raise AssertionError("a None batch should display as 'all'")


async def test_fetch_snapshot_pulls_both_endpoints(mocker: MockerFixture) -> None:
    client = mocker.MagicMock()
    client.status = mocker.AsyncMock(return_value=_summary())
    client.list_jobs = mocker.AsyncMock(return_value=_jobs(1))
    app = tui.TopApp(client=client, batch_id="b1", refresh_seconds=5.0)

    result = await app._fetch_snapshot()

    client.status.assert_awaited_once_with("b1")
    client.list_jobs.assert_awaited_once_with(batch_id="b1")
    if not isinstance(result, TopSnapshot):
        raise AssertionError(f"expected a snapshot, got {result!r}")


async def test_fetch_snapshot_reports_server_down(mocker: MockerFixture) -> None:
    client = mocker.MagicMock()
    client.status = mocker.AsyncMock(side_effect=ConnectionError("refused"))
    client.list_jobs = mocker.AsyncMock(return_value=[])
    app = tui.TopApp(client=client, batch_id=None, refresh_seconds=5.0)

    result = await app._fetch_snapshot()

    if not isinstance(result, str) or "refused" not in result:
        raise AssertionError(f"a down server must yield an error string, got {result!r}")


async def test_app_populates_table(mocker: MockerFixture) -> None:
    from textual.widgets import DataTable

    client = mocker.MagicMock()
    client.status = mocker.AsyncMock(return_value=_summary())
    client.list_jobs = mocker.AsyncMock(return_value=_jobs(3))
    app = tui.TopApp(client=client, batch_id="b1", refresh_seconds=999.0)

    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#jobs", DataTable)
        if table.row_count != 3:
            raise AssertionError(f"table should hold 3 job rows, got {table.row_count}")
