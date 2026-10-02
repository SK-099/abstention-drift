"""SQLAlchemy 2 engines and sessions: async for the API, sync for the Celery worker and Alembic."""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .settings import get_settings


class Base(DeclarativeBase):
    pass


@lru_cache
def async_engine():
    s = get_settings()
    kw = {} if s.database_url.startswith("sqlite") else {"pool_pre_ping": True, "pool_size": 10}
    url = s.database_url
    if "+asyncpg" in url:  # asyncpg spells libpq's sslmode=require as ssl=require
        url = url.replace("sslmode=", "ssl=")
    return create_async_engine(url, **kw)


@lru_cache
def async_sessionmaker_():
    return async_sessionmaker(async_engine(), expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with async_sessionmaker_()() as session:
        yield session


@lru_cache
def sync_engine():
    return create_engine(get_settings().sync_database_url, pool_pre_ping=True)


@lru_cache
def sync_sessionmaker():
    return sessionmaker(sync_engine(), expire_on_commit=False)
