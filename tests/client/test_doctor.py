from __future__ import annotations

import asyncio
from typing import cast

import httpx
from pytest_mock import MockerFixture

from screencropnet_yolo.client import doctor
from screencropnet_yolo.client.doctor import CheckResult
from screencropnet_yolo.server.config import Settings


def _resp(mocker: MockerFixture, *, status: int = 200, json_body: object = None):  # noqa: ANN202
    resp = mocker.MagicMock()
    resp.status_code = status
    resp.raise_for_status = mocker.MagicMock()
    resp.json = mocker.MagicMock(return_value=json_body)
    return resp


def _client(mocker: MockerFixture, resp: object) -> httpx.AsyncClient:
    client = mocker.MagicMock()
    client.get = mocker.AsyncMock(return_value=resp)
    client.aclose = mocker.AsyncMock()
    return cast("httpx.AsyncClient", client)


async def test_check_http_ok(mocker: MockerFixture) -> None:
    client = _client(mocker, _resp(mocker, json_body={"ok": True}))
    result = await doctor.check_http("api", "http://x/healthz", timeout=1.0, client=client)
    if not result.ok:
        raise AssertionError(f"expected ok, got {result}")
    if result.latency_ms is None:
        raise AssertionError("latency must be recorded")


async def test_check_http_expect_json_ok_false(mocker: MockerFixture) -> None:
    client = _client(mocker, _resp(mocker, json_body={"ok": False}))
    result = await doctor.check_http(
        "api", "http://x/healthz", timeout=1.0, expect_json_ok=True, client=client
    )
    if result.ok:
        raise AssertionError("ok=false in body must fail the check")


async def test_check_http_bad_status(mocker: MockerFixture) -> None:
    resp = _resp(mocker, status=500)
    resp.raise_for_status.side_effect = RuntimeError("500 server error")
    client = _client(mocker, resp)
    result = await doctor.check_http("prometheus", "http://x", timeout=1.0, client=client)
    if result.ok:
        raise AssertionError("a raised status must fail the check")
    if "500" not in result.detail:
        raise AssertionError(f"detail should carry the error, got {result.detail!r}")


async def test_check_times_out(mocker: MockerFixture) -> None:
    async def _slow() -> str:
        await asyncio.sleep(5)
        return "never"

    result = await doctor._run_check("slow", _slow, timeout=0.02)
    if result.ok:
        raise AssertionError("a slow probe must fail under the timeout")
    if "tim" not in result.detail.lower():
        raise AssertionError(f"detail should mention timeout, got {result.detail!r}")


async def test_check_postgres_ok(mocker: MockerFixture) -> None:
    conn = mocker.AsyncMock()
    engine = mocker.MagicMock()
    engine.connect.return_value.__aenter__ = mocker.AsyncMock(return_value=conn)
    engine.connect.return_value.__aexit__ = mocker.AsyncMock(return_value=False)
    engine.dispose = mocker.AsyncMock()
    mocker.patch("screencropnet_yolo.client.doctor.create_async_engine", return_value=engine)

    result = await doctor.check_postgres("postgresql+asyncpg://x", timeout=1.0)
    if not result.ok:
        raise AssertionError(f"expected ok, got {result}")
    conn.execute.assert_awaited_once()
    engine.dispose.assert_awaited_once()


async def test_check_rabbitmq_failure(mocker: MockerFixture) -> None:
    mocker.patch(
        "screencropnet_yolo.client.doctor.aio_pika.connect",
        new=mocker.AsyncMock(side_effect=ConnectionError("refused")),
    )
    result = await doctor.check_rabbitmq("amqp://x", timeout=1.0)
    if result.ok:
        raise AssertionError("a refused broker connection must fail")
    if "refused" not in result.detail:
        raise AssertionError(f"detail should carry the cause, got {result.detail!r}")


async def test_run_doctor_runs_all_concurrently(mocker: MockerFixture) -> None:
    order: list[str] = []

    async def _check(name: str, delay: float) -> CheckResult:
        await asyncio.sleep(delay)
        order.append(name)
        return CheckResult(name=name, ok=True, detail="ok", latency_ms=1.0)

    checks = [_check("a", 0.03), _check("b", 0.01)]
    results = await doctor.run_doctor(Settings(), checks=checks)
    # Concurrent: the shorter-delay check finishes first regardless of list order.
    if order != ["b", "a"]:
        raise AssertionError(f"checks did not run concurrently: {order}")
    if {r.name for r in results} != {"a", "b"}:
        raise AssertionError("run_doctor must return every result")


def test_exit_code() -> None:
    ok = [CheckResult("a", True, "ok"), CheckResult("b", True, "ok")]
    bad = [CheckResult("a", True, "ok"), CheckResult("b", False, "down")]
    if doctor.exit_code(ok) != 0:
        raise AssertionError("all-ok must exit 0")
    if doctor.exit_code(bad) == 0:
        raise AssertionError("any failure must exit non-zero")


def test_render_json_roundtrips() -> None:
    import json

    results = [CheckResult("api", True, "HTTP 200", latency_ms=12.3)]
    payload = json.loads(doctor.render_json(results))
    if payload[0]["name"] != "api" or payload[0]["ok"] is not True:
        raise AssertionError(f"unexpected json payload: {payload}")
