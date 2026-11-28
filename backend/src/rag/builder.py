"""LangGraph StateGraph builder wiring the 7-node Corrective RAG workflow."""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.rag.grader import ContextGrader
from src.rag.llm import get_chat_model
from src.rag.nodes import (
    make_direct_reply_node,
    make_fallback_notice_node,
    make_grade_context_node,
    make_rerank_node,
    make_retrieve_node,
    make_rewrite_query_node,
    make_synthesize_node,
)
from src.rag.reranker import FlashRankReranker, get_reranker
from src.rag.retriever import HybridRetriever
from src.rag.routing import _route_after_grade, _route_after_rewrite
from src.rag.state import AgentState


def build_rag_graph(
    *,
    retriever: HybridRetriever | None = None,
    reranker: FlashRankReranker | None = None,
    grader: ContextGrader | None = None,
    llm: BaseChatModel | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    collection: str | None = None,
):
    """Build and compile the 7-node LangGraph Corrective RAG `StateGraph(AgentState)`."""

    def _resolve_llm() -> BaseChatModel:
        if llm is not None:
            return llm
        return get_chat_model(
            provider_override=provider_override,
            model_override=model_override,
        )

    active_retriever = retriever or HybridRetriever(collection=collection)
    active_reranker = reranker or get_reranker()

    workflow = StateGraph(AgentState)
    workflow.add_node("rewrite_query", make_rewrite_query_node(_resolve_llm, session_factory=session_factory))
    workflow.add_node("direct_reply", make_direct_reply_node(_resolve_llm))
    workflow.add_node("retrieve", make_retrieve_node(active_retriever))
    workflow.add_node("rerank", make_rerank_node(active_reranker))
    workflow.add_node(
        "grade_context",
        make_grade_context_node(
            grader=grader,
            llm=llm,
            provider_override=provider_override,
            model_override=model_override,
        ),
    )
    workflow.add_node("synthesize", make_synthesize_node(_resolve_llm))
    workflow.add_node("fallback_notice", make_fallback_notice_node())

    workflow.add_edge(START, "rewrite_query")
    workflow.add_conditional_edges(
        "rewrite_query",
        _route_after_rewrite,
        {
            "retrieve": "retrieve",
            "direct_reply": "direct_reply",
        },
    )
    workflow.add_edge("direct_reply", END)
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "grade_context")
    workflow.add_conditional_edges(
        "grade_context",
        _route_after_grade,
        {
            "synthesize": "synthesize",
            "fallback_notice": "fallback_notice",
        },
    )
    workflow.add_edge("synthesize", END)
    workflow.add_edge("fallback_notice", END)

    return workflow.compile(checkpointer=checkpointer)
