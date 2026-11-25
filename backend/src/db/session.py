"""Async SQLite engine + session factory (WAL, FK enforcement) and startup initialization."""
from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from src.config import get_settings
from src.db import models  # noqa: F401  (register tables)
from src.db.migrations import apply_all_column_migrations, run_startup_migrations_and_seeds
from src.db.seed_data import DEFAULT_DOMAINS, DEFAULT_SUBDOMAINS, STARTER_LEXICON  # noqa: F401
from src.db.settings_bootstrap import parse_settings_json_dict as _parse_settings_json_dict  # noqa: F401

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_engine_url: str | None = None
_factory: async_sessionmaker[AsyncSession] | None = None


def create_engine(url: str | None = None) -> AsyncEngine:
    url = url or get_settings().sqlite_url
    if url.startswith("sqlite") and ":memory:" not in url:
        db_path = url.split(":///", 1)[1]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(url, echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def _pragmas(dbapi_conn, _record):
        journal_mode = os.environ.get("SQLITE_JOURNAL_MODE", "WAL").strip().upper()
        cur = dbapi_conn.cursor()
        cur.execute(f"PRAGMA journal_mode={journal_mode}")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()

    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


def set_engine_state(
    engine: AsyncEngine | None,
    engine_url: str | None,
    factory: async_sessionmaker[AsyncSession] | None,
) -> None:
    """Explicitly update module-level engine and session factory singletons."""
    global _engine, _engine_url, _factory
    _engine = engine
    _engine_url = engine_url
    _factory = factory


async def init_db(engine: AsyncEngine) -> None:
    """Create tables, run column migrations, seed taxonomy/lexicon, and backfill legacy rows."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.run_sync(apply_all_column_migrations)

    session_factory = create_session_factory(engine)
    async with session_factory() as session:
        async with session.begin():
            await run_startup_migrations_and_seeds(session)


def get_engine() -> AsyncEngine:
    global _engine, _engine_url, _factory
    target_url = get_settings().sqlite_url
    if _engine is not None and _engine_url != target_url:
        try:
            _engine.sync_engine.dispose()
        except Exception:
            pass
        _engine = None
        _factory = None
    if _engine is None:
        _engine = create_engine(target_url)
        _engine_url = target_url
        _factory = create_session_factory(_engine)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    get_engine()
    assert _factory is not None
    return _factory


async def close_engine() -> None:
    """Flush WAL checkpoint to TRUNCATE, dispose SQLAlchemy engine, and reset singletons."""
    global _engine, _engine_url, _factory
    if _engine is not None:
        try:
            async with _engine.begin() as conn:
                await conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
        except Exception:
            pass
        try:
            await _engine.dispose()
        except Exception:
            pass
        _engine = None
        _engine_url = None
        _factory = None


async def vacuum_sqlite_db(engine: AsyncEngine) -> None:
    """Execute VACUUM and PRAGMA wal_checkpoint(TRUNCATE) outside of any transaction with autocommit.

    Handles both AsyncEngine and sync Engine. Outside transactions prevents
    sqlite3.OperationalError: cannot VACUUM from within a transaction.
    """
    if hasattr(engine, "sync_engine"):
        async with engine.connect() as conn:
            autocommit_conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
            await autocommit_conn.execute(text("VACUUM"))
            await autocommit_conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
    else:
        with engine.connect() as conn:
            autocommit_conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            autocommit_conn.execute(text("VACUUM"))
            autocommit_conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))


async def get_session() -> AsyncIterator[AsyncSession]:
    async with get_session_factory()() as session:
        yield session
