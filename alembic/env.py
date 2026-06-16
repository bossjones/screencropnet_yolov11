"""Async Alembic environment driven by ``Settings.postgres_dsn``.

Online migrations run against the asyncpg engine via ``connection.run_sync``.
"""

from __future__ import annotations

import asyncio

from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy.pool import NullPool

from screencropnet_yolo.server.config import get_settings
from screencropnet_yolo.server.db import Base

config = context.config
target_metadata = Base.metadata

_dsn = get_settings().postgres_dsn
config.set_main_option("sqlalchemy.url", _dsn)


def run_migrations_offline() -> None:
    context.configure(
        url=_dsn,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
