"""7-node LangGraph Corrective RAG (CRAG) conversational facade."""
from __future__ import annotations

from src.db.repository import upsert_chat_thread
from src.rag.builder import build_rag_graph
from src.rag.checkpointer import (
    close_checkpointer,
    compact_chat_history_db,
    delete_thread_state,
    get_checkpointer,
    get_thread_state,
    set_checkpointer,
)
from src.rag.citations import (
    build_citations_from_chunks,
    format_synthesis_context,
    validate_and_sanitize_citations,
)
from src.rag.nodes import NODE_STEP_LABELS
from src.rag.routing import (
    QueryRewriteResult,
    _detect_engineering_query_intent,
    _fallback_parse_rewrite,
    _route_after_grade,
    _route_after_rewrite,
)
from src.rag.streaming import (
    _chunk_text_for_token_stream,
    format_sse_error_event,
    stream_chat_response,
)

__all__ = [
    "NODE_STEP_LABELS",
    "QueryRewriteResult",
    "build_rag_graph",
    "close_checkpointer",
    "compact_chat_history_db",
    "get_checkpointer",
    "get_thread_state",
    "delete_thread_state",
    "set_checkpointer",
    "stream_chat_response",
    "upsert_chat_thread",
    "validate_and_sanitize_citations",
]
