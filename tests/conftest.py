from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from screencropnet_yolo.server.api import create_app, get_db_session, get_publisher
from screencropnet_yolo.server.config import Settings
from screencropnet_yolo.server.db import create_all, make_engine, make_sessionmaker
from screencropnet_yolo.server.queue import FakePublisher


@pytest_asyncio.fixture
async def sqlite_engine() -> AsyncIterator[AsyncEngine]:
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    await create_all(engine)
    yield engine
    await engine.dispose()


@pytest.fixture
def session_factory(sqlite_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return make_sessionmaker(sqlite_engine)


@pytest.fixture
def fake_publisher() -> FakePublisher:
    return FakePublisher()


@pytest.fixture
def build_app(
    session_factory: async_sessionmaker[AsyncSession],
    fake_publisher: FakePublisher,
) -> Callable[[Settings], FastAPI]:
    """Build a fully wired app with sqlite + FakePublisher dependency overrides."""

    def _build(settings: Settings) -> FastAPI:
        app = create_app(settings)

        async def _session_override() -> AsyncIterator[AsyncSession]:
            async with session_factory() as session:
                yield session

        app.dependency_overrides[get_db_session] = _session_override
        app.dependency_overrides[get_publisher] = lambda: fake_publisher
        return app

    return _build


@pytest.fixture
def app(build_app: Callable[[Settings], FastAPI], tmp_path: Path) -> FastAPI:
    settings = Settings(logs_dir=tmp_path / "logs", compress_tmp_dir=tmp_path / "uploads")
    return build_app(settings)


@pytest_asyncio.fixture
async def async_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture(autouse=True)
def _pytorch_profiler(request: pytest.FixtureRequest) -> Iterator[None]:
    """Wrap ``@pytest.mark.pytorch_profile`` tests in ``torch.profiler``.

    Inert unless the test carries the marker *and* the ``perf`` skill's runner
    has set ``SCREENCROPNET_TORCH_PROFILE_DIR`` — so marking a test never
    perturbs a normal ``make test`` run. When active, it exports the
    ``key_averages`` events (as JSON the renderer consumes) plus a Chrome trace,
    and, only when CUDA is present, a memory-history snapshot (the
    *Understanding GPU Memory* workflow is CUDA-only). torch is imported lazily
    so the default (torch-free) environment can still collect this conftest.
    """
    out_dir = os.environ.get("SCREENCROPNET_TORCH_PROFILE_DIR")
    if request.node.get_closest_marker("pytorch_profile") is None or not out_dir:
        yield
        return

    import torch
    from torch.profiler import ProfilerActivity, profile

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    slug = request.node.name.replace("/", "-").replace("::", "-")
    cuda = torch.cuda.is_available()
    activities = [ProfilerActivity.CPU]
    if cuda:
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.memory._record_memory_history(max_entries=100_000)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        yield

    events = [
        {
            "name": e.key,
            "count": e.count,
            "self_cpu_time_total": float(e.self_cpu_time_total),
            "cpu_time_total": float(e.cpu_time_total),
            "self_cuda_time_total": float(getattr(e, "self_device_time_total", 0.0)),
            "cpu_memory_usage": int(e.cpu_memory_usage),
            "cuda_memory_usage": int(getattr(e, "device_memory_usage", 0)),
        }
        for e in prof.key_averages()
    ]
    (dest / f"{slug}.events.json").write_text(json.dumps({"events": events, "cuda": cuda}))
    prof.export_chrome_trace(str(dest / f"{slug}.trace.json"))
    if cuda:
        torch.cuda.memory._dump_snapshot(str(dest / f"{slug}.memory-snapshot.pickle"))
        torch.cuda.memory._record_memory_history(enabled=None)
