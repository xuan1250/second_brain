"""CRAG SSE streaming execution, token chunking, and shared error event framing."""
from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import Settings, get_settings
from src.db.repository import upsert_chat_thread
from src.ingestion.indexer import BaseEmbedder, SparseEncoder
from src.rag.builder import build_rag_graph
from src.rag.checkpointer import compact_chat_history_db, get_checkpointer
from src.rag.grader import ContextGrader
from src.rag.llm import record_key_validation, resolve_provider_and_model, sanitize_llm_error
from src.rag.nodes import NODE_STEP_LABELS
from src.rag.reranker import FlashRankReranker
from src.rag.retriever import HybridRetriever
from src.rag.state import AgentState, Citation

logger = logging.getLogger("uvicorn.error")


def _chunk_text_for_token_stream(text: str, chunk_size: int = 24) -> list[str]:
    """Split a completed assistant message into token-sized chunks for SSE streaming."""
    if not text:
        return []
    tokens = re.findall(r"\S+\s*|\n+", text)
    if not tokens:
        return [text]
    grouped: list[str] = []
    buf = ""
    for tok in tokens:
        if len(buf) + len(tok) > chunk_size and buf:
            grouped.append(buf)
            buf = tok
        else:
            buf += tok
    if buf:
        grouped.append(buf)
    return grouped


def format_sse_error_event(
    exc: Exception | str,
    *,
    thread_id: str,
    provider: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Sanitize an exception, record invalid auth state if applicable, and build an SSE error event."""
    clean_err, is_auth = sanitize_llm_error(exc, provider=provider)
    if is_auth and provider:
        try:
            record_key_validation(
                provider,
                status="invalid",
                error=clean_err,
                settings=settings or get_settings(),
            )
        except Exception:
            pass
    return {
        "event": "error",
        "data": {
            "thread_id": thread_id,
            "message": clean_err,
            "error": clean_err,
        },
    }


async def stream_chat_response(
    query: str,
    thread_id: str | None = None,
    filter_doc_ids: list[str] | None = None,
    domain_filter: Any | None = None,
    provider: str | None = None,
    model: str | None = None,
    *,
    qdrant: AsyncQdrantClient | None = None,
    embedder: BaseEmbedder | None = None,
    sparse_encoder: SparseEncoder | None = None,
    reranker: FlashRankReranker | None = None,
    grader: ContextGrader | None = None,
    llm: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    collection: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Execute the 7-node CRAG workflow and yield structured SSE events:

    Event types emitted:
    - `step`: `{"node": str, "label": str, "status": "completed", ...}`
    - `routing`: `{"routing_mode": str, "target_domains": list, ...}`
    - `citations`: `{"citations": list[Citation]}`
    - `token`: `{"content": str, "token": str}`
    - `done`: `{"thread_id": str, "message": str, "citations": list[Citation]}`
    - `error`: `{"thread_id": str, "message": str, "error": str}`
    """
    active_thread_id = (thread_id or "").strip() or uuid.uuid4().hex
    clean_query = (query or "").strip()
    active_provider: str | None = provider

    try:
        if not clean_query:
            raise ValueError("Query must not be empty.")

        await upsert_chat_thread(
            thread_id=active_thread_id,
            user_message=clean_query,
            session_factory=session_factory,
        )

        saver = checkpointer or await get_checkpointer()
        active_retriever = (
            HybridRetriever(
                qdrant=qdrant,
                embedder=embedder,
                sparse_encoder=sparse_encoder,
                collection=collection,
            )
            if (
                qdrant is not None
                or embedder is not None
                or sparse_encoder is not None
                or collection is not None
            )
            else None
        )

        active_provider, active_model = resolve_provider_and_model(
            provider_override=provider,
            model_override=model,
        )
        logger.info(
            "[CRAG] Executing workflow: thread_id=%s, provider=%s, model=%s",
            active_thread_id,
            active_provider,
            active_model,
        )

        graph = build_rag_graph(
            retriever=active_retriever,
            reranker=reranker,
            grader=grader,
            llm=llm,
            provider_override=provider,
            model_override=model,
            checkpointer=saver,
            session_factory=session_factory,
            collection=collection,
        )

        domain_filter_dict = (
            domain_filter.model_dump()
            if hasattr(domain_filter, "model_dump")
            else (dict(domain_filter) if isinstance(domain_filter, dict) else None)
        )

        initial_input: AgentState = {
            "messages": [
                HumanMessage(
                    content=clean_query,
                    additional_kwargs={"domain_filter": domain_filter_dict} if domain_filter_dict else {},
                )
            ],
            "original_query": clean_query,
            "filter_doc_ids": list(filter_doc_ids) if filter_doc_ids else None,
            "domain_filter": domain_filter_dict,
            "citations": [],
            "steps": [],
        }
        config: RunnableConfig = {"configurable": {"thread_id": active_thread_id}}

        final_message = ""
        final_citations: list[Citation] = []
        citations_emitted = False
        last_routing_info: dict[str, Any] = {}

        async for update in graph.astream(initial_input, config=config, stream_mode="updates"):
            if not isinstance(update, dict):
                continue
            for node_name, node_output in update.items():
                if node_name not in NODE_STEP_LABELS:
                    continue

                step_data: dict[str, Any] = {
                    "node": node_name,
                    "label": NODE_STEP_LABELS[node_name],
                    "status": "completed",
                }

                if isinstance(node_output, dict):
                    if node_name == "rewrite_query":
                        r_mode = node_output.get("routing_mode", "single")
                        t_domains = node_output.get("target_domains", [])
                        p_collisions = node_output.get("polysemy_collisions", [])
                        d_queries = node_output.get("domain_queries", {})
                        last_routing_info = {
                            "routing_mode": r_mode,
                            "target_domains": t_domains,
                            "polysemy_collisions": p_collisions,
                            "domain_queries": d_queries,
                        }
                        step_data.update(last_routing_info)
                        yield {
                            "event": "step",
                            "data": step_data,
                        }
                        yield {
                            "event": "routing",
                            "data": last_routing_info,
                        }
                    else:
                        yield {
                            "event": "step",
                            "data": step_data,
                        }

                    if "citations" in node_output:
                        final_citations = list(node_output.get("citations") or [])
                    if node_name in ("synthesize", "fallback_notice", "direct_reply"):
                        yield {
                            "event": "citations",
                            "data": {"citations": final_citations},
                        }
                        citations_emitted = True

                        msgs = node_output.get("messages") or []
                        if msgs:
                            last_ai = msgs[-1]
                            final_message = str(getattr(last_ai, "content", "") or "")
                            for token_piece in _chunk_text_for_token_stream(final_message):
                                yield {
                                    "event": "token",
                                    "data": {
                                        "content": token_piece,
                                        "token": token_piece,
                                    },
                                }

        if not citations_emitted:
            yield {
                "event": "citations",
                "data": {"citations": final_citations},
            }

        # Refresh ChatThread updated_at timestamp on completion
        await upsert_chat_thread(
            thread_id=active_thread_id,
            user_message=clean_query,
            session_factory=session_factory,
        )

        try:
            await compact_chat_history_db(
                thread_id=active_thread_id,
                run_vacuum=False,
                checkpointer=saver,
            )
        except Exception as comp_err:
            logger.warning(
                "[CRAG] Checkpointer compaction failed for thread %s: %s",
                active_thread_id,
                comp_err,
            )

        logger.info(
            "[CRAG] Workflow complete: thread_id=%s, provider=%s, model=%s",
            active_thread_id,
            active_provider,
            active_model,
        )
        done_payload: dict[str, Any] = {
            "thread_id": active_thread_id,
            "message": final_message,
            "citations": final_citations,
            "provider": active_provider,
            "model": active_model,
        }
        if last_routing_info:
            done_payload.update(last_routing_info)

        yield {
            "event": "done",
            "data": done_payload,
        }
    except Exception as exc:
        yield format_sse_error_event(
            exc,
            thread_id=active_thread_id,
            provider=active_provider,
        )
