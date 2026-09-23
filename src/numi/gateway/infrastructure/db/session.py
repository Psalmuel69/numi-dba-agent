"""Async SQLAlchemy engine/session factory for the control-plane database."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from numi.gateway.infrastructure.db.models import Base


def make_engine(database_url: str) -> AsyncEngine:
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    return create_async_engine(database_url, connect_args=connect_args, future=True)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_models(engine: AsyncEngine) -> None:
    """Create all tables. Used for local dev/tests only — production schema
    changes go through Alembic migrations (migrations/)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


class Database:
    """Small convenience wrapper bundling an engine + session factory so
    services can depend on a single object via FastAPI dependency injection."""

    def __init__(self, database_url: str):
        self.engine = make_engine(database_url)
        self.session_factory = make_session_factory(self.engine)

    async def create_all(self) -> None:
        await init_models(self.engine)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session

    async def dispose(self) -> None:
        await self.engine.dispose()
