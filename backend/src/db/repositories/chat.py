"""Relational `ChatThread` repository scoped strictly to `second_brain.db`."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.db.models import ChatThread
from src.db.session import get_engine, get_session_factory, init_db


async def upsert_chat_thread(
    thread_id: str,
    user_message: str,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> ChatThread:
    """Insert or update a `ChatThread` row in `second_brain.db` on every conversation turn."""
    if session_factory is None:
        engine = get_engine()
        await init_db(engine)
        session_factory = get_session_factory()

    clean_title = " ".join((user_message or "").strip().split())
    if len(clean_title) > 80:
        clean_title = clean_title[:77] + "..."
    if not clean_title:
        clean_title = "New chat"

    now = datetime.now(timezone.utc)
    async with session_factory() as session:
        existing = await session.get(ChatThread, thread_id)
        if existing is None:
            row = ChatThread(
                thread_id=thread_id,
                title=clean_title,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
        else:
            if (not existing.title or existing.title == "New chat") and clean_title != "New chat":
                existing.title = clean_title
            existing.updated_at = now
            row = existing
        await session.commit()
        await session.refresh(row)
        return row


async def list_chat_threads(
    session: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[ChatThread], int]:
    """List `ChatThread` rows ordered newest first alongside total row count."""
    total = int((await session.execute(select(func.count()).select_from(ChatThread))).scalar() or 0)
    rows = list(
        (
            await session.execute(
                select(ChatThread).order_by(ChatThread.updated_at.desc()).offset(offset).limit(limit)
            )
        ).scalars().all()
    )
    return rows, total


async def get_chat_thread_row(session: AsyncSession, thread_id: str) -> ChatThread | None:
    """Fetch a single `ChatThread` row by thread_id."""
    return await session.get(ChatThread, thread_id)


async def delete_chat_thread_row(session: AsyncSession, thread_id: str) -> bool:
    """Delete a `ChatThread` row by thread_id. Returns True if deleted, False if absent."""
    row = await session.get(ChatThread, thread_id)
    if row is None:
        return False
    await session.delete(row)
    await session.commit()
    return True
