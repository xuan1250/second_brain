"""Conversational Corrective RAG (CRAG) package."""
from __future__ import annotations

from src.rag.grader import BatchGradeResult, ContextGrader, format_documents_xml, grade_chunks
from src.rag.graph import (
    NODE_STEP_LABELS,
    QueryRewriteResult,
    build_rag_graph,
    close_checkpointer,
    delete_thread_state,
    get_checkpointer,
    get_thread_state,
    set_checkpointer,
    stream_chat_response,
    upsert_chat_thread,
    validate_and_sanitize_citations,
)
from src.rag.llm import get_chat_model, invoke_structured_output, resolve_provider_and_model, set_chat_model
from src.rag.reranker import FlashRankReranker, deduplicate_parent_chunks, get_reranker, set_reranker
from src.rag.retriever import (
    BgeM3SparseEncoder,
    HybridRetriever,
    build_hybrid_query_kwargs,
    build_qdrant_filter,
    get_active_qdrant_client,
    get_embedder,
    get_sparse_encoder,
    resolve_compute_device,
    resolve_sparse_mode,
    set_active_qdrant_client,
    set_embedder,
    set_sparse_encoder,
)
from src.rag.state import AgentState, Citation, CitationModel, format_citation_source

__all__ = [
    "AgentState",
    "BatchGradeResult",
    "BgeM3SparseEncoder",
    "Citation",
    "CitationModel",
    "ContextGrader",
    "FlashRankReranker",
    "HybridRetriever",
    "NODE_STEP_LABELS",
    "QueryRewriteResult",
    "build_hybrid_query_kwargs",
    "build_qdrant_filter",
    "build_rag_graph",
    "close_checkpointer",
    "deduplicate_parent_chunks",
    "delete_thread_state",
    "format_citation_source",
    "format_documents_xml",
    "get_active_qdrant_client",
    "get_chat_model",
    "get_checkpointer",
    "get_embedder",
    "get_reranker",
    "get_sparse_encoder",
    "get_thread_state",
    "grade_chunks",
    "invoke_structured_output",
    "resolve_compute_device",
    "resolve_provider_and_model",
    "resolve_sparse_mode",
    "set_active_qdrant_client",
    "set_chat_model",
    "set_checkpointer",
    "set_embedder",
    "set_reranker",
    "set_sparse_encoder",
    "stream_chat_response",
    "upsert_chat_thread",
    "validate_and_sanitize_citations",
]

