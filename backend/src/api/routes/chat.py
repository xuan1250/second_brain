"""SSE streaming chat completions and conversation thread history routes."""
from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

logger = logging.getLogger("uvicorn.error")
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from qdrant_client import AsyncQdrantClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.dependencies import (
    get_active_checkpointer,
    get_app_settings,
    get_db_session,
    get_db_session_factory,
    get_qdrant,
    get_shared_embedder,
    get_shared_reranker,
    get_shared_sparse_encoder,
)
from src.api.schemas import (
    ChatMessagePayload,
    ChatRequest,
    CitationPayload,
    DomainFilterSpec,
    ThreadDeleteResponse,
    ThreadDetailResponse,
    ThreadListResponse,
    ThreadSummary,
)
from src.config import Settings
from src.db.models import ChatThread
from src.ingestion.indexer import BaseEmbedder, SparseEncoder
from src.rag.graph import delete_thread_state, get_thread_state, stream_chat_response
from src.rag.llm import record_key_validation, sanitize_llm_error
from src.rag.reranker import FlashRankReranker

router = APIRouter(prefix="/chat", tags=["chat"])


def format_sse_frame(event: str, data: dict[str, Any]) -> str:
    """Serialize an event name and JSON payload into a standard SSE wire frame."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _serialize_messages(
    messages: list[BaseMessage],
    latest_citations: list[CitationPayload],
) -> list[ChatMessagePayload]:
    serialized: list[ChatMessagePayload] = []
    last_ai_idx = -1
    for idx, msg in enumerate(messages):
        if isinstance(msg, AIMessage):
            last_ai_idx = idx

    for idx, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            role = str(getattr(msg, "type", "assistant"))

        content = str(getattr(msg, "content", "") or "")
        msg_citations: list[CitationPayload] = []
        domain_filter: DomainFilterSpec | None = None
        steps: list[dict[str, Any]] = []

        extra = getattr(msg, "additional_kwargs", None) or {}
        if isinstance(extra, dict):
            if "citations" in extra and isinstance(extra["citations"], list):
                msg_citations = [
                    CitationPayload.model_validate(c)
                    for c in extra["citations"]
                    if isinstance(c, dict)
                ]
            if "domain_filter" in extra and extra["domain_filter"]:
                df_raw = extra["domain_filter"]
                domain_filter = (
                    DomainFilterSpec.model_validate(df_raw)
                    if isinstance(df_raw, dict)
                    else (df_raw if isinstance(df_raw, DomainFilterSpec) else None)
                )
            if "steps" in extra and isinstance(extra["steps"], list):
                steps = [s for s in extra["steps"] if isinstance(s, dict)]

        if idx == last_ai_idx and latest_citations and not msg_citations:
            msg_citations = latest_citations

        serialized.append(
            ChatMessagePayload(
                role=role,
                content=content,
                citations=msg_citations,
                domain_filter=domain_filter,
                steps=steps,
            )
        )
    return serialized


@router.post("/completions")
async def chat_completions(
    payload: ChatRequest,
    request: Request,
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    embedder: BaseEmbedder = Depends(get_shared_embedder),
    sparse_encoder: SparseEncoder = Depends(get_shared_sparse_encoder),
    reranker: FlashRankReranker = Depends(get_shared_reranker),
    checkpointer: BaseCheckpointSaver = Depends(get_active_checkpointer),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_db_session_factory),
    settings: Settings = Depends(get_app_settings),
) -> StreamingResponse:
    """Stream Corrective RAG reasoning steps, citations, and synthesized answer tokens via SSE."""
    active_thread_id = (payload.thread_id or "").strip() or uuid.uuid4().hex

    logger.info(
        "[CHAT-REQUEST] Received chat query: thread_id=%s, provider=%s, model=%s, query=%r",
        active_thread_id,
        payload.provider,
        payload.model,
        (payload.message or "")[:60],
    )

    async def _sse_generator() -> AsyncIterator[str]:
        try:
            async for item in stream_chat_response(
                query=payload.message,
                thread_id=active_thread_id,
                filter_doc_ids=payload.filter_doc_ids,
                domain_filter=payload.domain_filter,
                provider=payload.provider,
                model=payload.model,
                qdrant=qdrant,
                embedder=embedder,
                sparse_encoder=sparse_encoder,
                reranker=reranker,
                checkpointer=checkpointer,
                session_factory=session_factory,
                collection=settings.qdrant_collection,
            ):
                if await request.is_disconnected():
                    break
                event_type = str(item.get("event") or "message")
                data_obj = item.get("data") if isinstance(item.get("data"), dict) else {"value": item.get("data")}
                yield format_sse_frame(event_type, data_obj)
        except Exception as exc:
            clean_err, is_auth = sanitize_llm_error(exc, provider=payload.provider)
            if is_auth and payload.provider:
                try:
                    record_key_validation(payload.provider, status="invalid", error=clean_err, settings=settings)
                except Exception:
                    pass
            yield format_sse_frame(
                "error",
                {
                    "thread_id": active_thread_id,
                    "message": clean_err,
                    "error": clean_err,
                },
            )

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers=headers,
    )


@router.get("/threads", response_model=ThreadListResponse)
async def list_chat_threads(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> ThreadListResponse:
    """List saved conversation threads from `chat_threads`, ordered newest first with pagination."""
    total = int((await session.execute(select(func.count()).select_from(ChatThread))).scalar() or 0)
    rows = (
        await session.execute(
            select(ChatThread).order_by(ChatThread.updated_at.desc()).offset(offset).limit(limit)
        )
    ).scalars().all()
    summaries = [ThreadSummary.model_validate(row) for row in rows]
    return ThreadListResponse(
        threads=summaries,
        items=summaries,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/threads/{thread_id}", response_model=ThreadDetailResponse)
async def get_chat_thread(
    thread_id: str,
    session: AsyncSession = Depends(get_db_session),
    checkpointer: BaseCheckpointSaver = Depends(get_active_checkpointer),
) -> ThreadDetailResponse:
    """Return full message and citation history for a conversation thread."""
    row = await session.get(ChatThread, thread_id)
    state = await get_thread_state(thread_id=thread_id, checkpointer=checkpointer)
    raw_messages: list[BaseMessage] = list(state.get("messages") or [])
    raw_citations: list[dict[str, Any]] = list(state.get("citations") or [])

    if row is None and not raw_messages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread '{thread_id}' not found.",
        )

    citations = [
        CitationPayload.model_validate(c)
        for c in raw_citations
        if isinstance(c, dict)
    ]
    messages = _serialize_messages(raw_messages, citations)

    title = row.title if row is not None else "New chat"
    if title == "New chat" and messages:
        for m in messages:
            if m.role == "user" and m.content.strip():
                title = m.content.strip()[:80]
                break

    return ThreadDetailResponse(
        thread_id=thread_id,
        title=title,
        created_at=row.created_at if row is not None else None,
        updated_at=row.updated_at if row is not None else None,
        messages=messages,
        citations=citations,
    )


@router.delete("/threads/{thread_id}", response_model=ThreadDeleteResponse)
async def delete_chat_thread(
    thread_id: str,
    session: AsyncSession = Depends(get_db_session),
    checkpointer: BaseCheckpointSaver = Depends(get_active_checkpointer),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_db_session_factory),
) -> ThreadDeleteResponse:
    """Clear thread checkpoint state from `chat_history.db` and delete the `chat_threads` row."""
    row = await session.get(ChatThread, thread_id)
    state = await get_thread_state(thread_id=thread_id, checkpointer=checkpointer)
    has_messages = bool(state.get("messages"))

    deleted_row = await delete_thread_state(
        thread_id=thread_id,
        checkpointer=checkpointer,
        session_factory=session_factory,
    )

    if not deleted_row and row is None and not has_messages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread '{thread_id}' not found.",
        )

    return ThreadDeleteResponse(
        thread_id=thread_id,
        deleted=True,
        status="deleted",
    )

