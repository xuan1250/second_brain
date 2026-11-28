"""Persistent AsyncSqliteSaver checkpointer management, compaction, and thread state APIs."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import aiosqlite
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import get_settings
from src.db.repositories.chat import delete_chat_thread_row
from src.db.session import get_engine, get_session_factory, init_db
from src.rag.builder import build_rag_graph

logger = logging.getLogger("uvicorn.error")

# Persistent AsyncSqliteSaver management (`data/db/chat_history.db`)
_checkpointer_singleton: BaseCheckpointSaver | None = None
_checkpointer_conn: aiosqlite.Connection | None = None
_checkpointer_db_path: Path | None = None


def set_checkpointer(checkpointer: BaseCheckpointSaver | None) -> None:
    """Override or reset the shared LangGraph checkpointer (useful for tests or FastAPI lifespan)."""
    global _checkpointer_singleton
    _checkpointer_singleton = checkpointer


async def get_checkpointer(db_path: Path | str | None = None) -> BaseCheckpointSaver:
    """Return or initialize the persistent `AsyncSqliteSaver` at `data/db/chat_history.db`."""
    global _checkpointer_singleton, _checkpointer_conn, _checkpointer_db_path
    if _checkpointer_singleton is not None and db_path is None:
        return _checkpointer_singleton

    settings = get_settings()
    target_path = Path(db_path) if db_path is not None else (settings.db_dir / "chat_history.db")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if (
        _checkpointer_singleton is not None
        and _checkpointer_db_path is not None
        and _checkpointer_db_path == target_path
    ):
        return _checkpointer_singleton

    await close_checkpointer()
    conn = await aiosqlite.connect(str(target_path))
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    _checkpointer_conn = conn
    _checkpointer_singleton = saver
    _checkpointer_db_path = target_path
    return saver


async def close_checkpointer() -> None:
    """Close the underlying aiosqlite connection for `chat_history.db` if managed here."""
    global _checkpointer_singleton, _checkpointer_conn, _checkpointer_db_path
    if _checkpointer_conn is not None:
        try:
            await _checkpointer_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        try:
            await _checkpointer_conn.close()
        except Exception:
            pass
    _checkpointer_conn = None
    _checkpointer_singleton = None
    _checkpointer_db_path = None


async def compact_chat_history_db(
    thread_id: str | None = None,
    run_vacuum: bool = False,
    checkpointer: BaseCheckpointSaver | None = None,
) -> dict[str, int]:
    """Prune superseded intermediate checkpoints and clear transient chunk blobs.

    Args:
        thread_id: Optional specific thread_id to compact. If None, compacts all threads.
        run_vacuum: If True, executes VACUUM and PRAGMA wal_checkpoint(TRUNCATE).
        checkpointer: Active checkpointer instance. If provided and not AsyncSqliteSaver,
                      returns early without touching disk (e.g. for MemorySaver in tests).

    Returns:
        Dict with count of deleted checkpoints, deleted writes, and compacted blobs.
    """
    if checkpointer is not None and not isinstance(checkpointer, AsyncSqliteSaver):
        return {
            "superseded_checkpoints_deleted": 0,
            "writes_deleted": 0,
            "blobs_compacted": 0,
        }

    saver = checkpointer if isinstance(checkpointer, AsyncSqliteSaver) else await get_checkpointer()
    if not isinstance(saver, AsyncSqliteSaver):
        return {
            "superseded_checkpoints_deleted": 0,
            "writes_deleted": 0,
            "blobs_compacted": 0,
        }

    conn: aiosqlite.Connection = saver.conn
    superseded_checkpoints_deleted = 0
    writes_deleted = 0
    blobs_compacted = 0

    thread_filter = "thread_id = ? AND " if thread_id else ""
    select_where = " WHERE thread_id = ?" if thread_id else ""
    params: tuple[Any, ...] = (thread_id,) if thread_id else ()

    async with saver.lock:
        chk_del_sql = f"""
            DELETE FROM checkpoints
            WHERE {thread_filter}checkpoint_id != (
                SELECT MAX(c2.checkpoint_id)
                FROM checkpoints c2
                WHERE c2.thread_id = checkpoints.thread_id
                  AND c2.checkpoint_ns = checkpoints.checkpoint_ns
            )
        """
        cursor = await conn.execute(chk_del_sql, params)
        superseded_checkpoints_deleted = cursor.rowcount if cursor.rowcount != -1 else 0

        writes_del_sql = f"""
            DELETE FROM writes
            WHERE {thread_filter}NOT EXISTS (
                SELECT 1 FROM checkpoints c
                WHERE c.thread_id = writes.thread_id
                  AND c.checkpoint_ns = writes.checkpoint_ns
                  AND c.checkpoint_id = writes.checkpoint_id
            )
        """
        cursor = await conn.execute(writes_del_sql, params)
        writes_deleted = cursor.rowcount if cursor.rowcount != -1 else 0

        select_sql = (
            "SELECT thread_id, checkpoint_ns, checkpoint_id, type, checkpoint "
            f"FROM checkpoints{select_where}"
        )

        async with conn.execute(select_sql, params) as cur:
            rows = await cur.fetchall()

        for t_id, ns, c_id, chk_type, blob in rows:
            if not blob:
                continue
            try:
                data = saver.serde.loads_typed((chk_type, blob))
                if isinstance(data, dict) and "channel_values" in data:
                    cv = data["channel_values"]
                    modified = False
                    for k in ("retrieved_chunks", "reranked_chunks", "graded_chunks"):
                        if k in cv and cv[k]:
                            cv[k] = []
                            modified = True
                    if modified:
                        new_type, new_blob = saver.serde.dumps_typed(data)
                        await conn.execute(
                            "UPDATE checkpoints SET type = ?, checkpoint = ? "
                            "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                            (new_type, new_blob, t_id, ns, c_id),
                        )
                        blobs_compacted += 1
            except Exception as err:
                logger.warning(
                    "[CRAG] Failed to compact checkpoint blob for thread=%s id=%s: %s",
                    t_id,
                    c_id,
                    err,
                )

        await conn.commit()

        if run_vacuum:
            await conn.execute("VACUUM")
            await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    return {
        "superseded_checkpoints_deleted": max(0, superseded_checkpoints_deleted),
        "writes_deleted": max(0, writes_deleted),
        "blobs_compacted": blobs_compacted,
    }


async def get_thread_state(
    thread_id: str,
    checkpointer: BaseCheckpointSaver | None = None,
) -> dict[str, Any]:
    """Retrieve persisted messages and citations for `thread_id` from the checkpointer."""
    saver = checkpointer or await get_checkpointer()
    graph = build_rag_graph(checkpointer=saver)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)
    values = dict(snapshot.values or {}) if snapshot else {}
    return {
        "thread_id": thread_id,
        "messages": list(values.get("messages") or []),
        "citations": list(values.get("citations") or []),
    }


async def delete_thread_state(
    thread_id: str,
    checkpointer: BaseCheckpointSaver | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> bool:
    """Delete thread checkpoints from `chat_history.db` and remove the `ChatThread` row."""
    saver = checkpointer or await get_checkpointer()
    if hasattr(saver, "adelete_thread"):
        await saver.adelete_thread(thread_id)

    if session_factory is None:
        engine = get_engine()
        await init_db(engine)
        session_factory = get_session_factory()

    async with session_factory() as session:
        return await delete_chat_thread_row(session, thread_id)
