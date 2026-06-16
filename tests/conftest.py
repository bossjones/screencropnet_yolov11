from __future__ import annotations

from collections.abc import AsyncIterator, Callable
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
