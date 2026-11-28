"""LangGraph generation/rerank/grading/fallback node factories and step labels."""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.rag.citations import (
    build_citations_from_chunks,
    format_synthesis_context,
    validate_and_sanitize_citations,
)
from src.rag.grader import ContextGrader
from src.rag.prompts import build_synthesis_system_prompt
from src.rag.reranker import FlashRankReranker
from src.rag.state import AgentState

NODE_STEP_LABELS: dict[str, str] = {
    "rewrite_query": "Rewriting query",
    "direct_reply": "Replying directly",
    "retrieve": "Hybrid search",
    "rerank": "Reranking (FlashRank)",
    "grade_context": "Grading context",
    "synthesize": "Synthesizing",
    "fallback_notice": "Generating fallback notice",
}

NodeCallable = Callable[[AgentState], Awaitable[dict[str, Any]]]


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """Return True if callable `fn` declares parameter `name`."""
    try:
        return name in inspect.signature(fn).parameters
    except Exception:
        return False


def make_rerank_node(active_reranker: FlashRankReranker) -> NodeCallable:
    """Create the `rerank` node callable."""

    async def rerank_node(state: AgentState) -> dict[str, Any]:
        original_query = str(state.get("original_query") or "")
        candidates = list(state.get("retrieved_chunks") or [])
        if not candidates:
            return {
                "retrieved_chunks": [],
                "reranked_chunks": [],
                "steps": list(state.get("steps") or []),
            }
        rerank_kwargs: dict[str, Any] = {
            "query": original_query,
            "candidates": candidates,
        }
        eng_top_n = state.get("engineering_top_n")
        if eng_top_n is not None and _accepts_kwarg(active_reranker.rerank, "top_n"):
            rerank_kwargs["top_n"] = eng_top_n
        reranked = await active_reranker.rerank(**rerank_kwargs)
        step_record = {
            "node": "rerank",
            "label": NODE_STEP_LABELS["rerank"],
            "status": "completed",
            "reranked_count": len(reranked),
        }
        steps = list(state.get("steps") or []) + [step_record]
        return {
            "retrieved_chunks": [],
            "reranked_chunks": reranked,
            "steps": steps,
        }

    return rerank_node


def make_grade_context_node(
    *,
    grader: ContextGrader | None = None,
    llm: BaseChatModel | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> NodeCallable:
    """Create the `grade_context` node callable."""

    async def grade_context_node(state: AgentState) -> dict[str, Any]:
        original_query = str(state.get("original_query") or "")
        reranked = list(state.get("reranked_chunks") or [])
        if not reranked:
            return {
                "reranked_chunks": [],
                "graded_chunks": [],
                "steps": list(state.get("steps") or []),
            }
        active_grader = grader or ContextGrader(
            llm=llm,
            provider_override=provider_override,
            model_override=model_override,
        )
        graded = await active_grader.grade(query=original_query, chunks=reranked)
        step_record = {
            "node": "grade_context",
            "label": NODE_STEP_LABELS["grade_context"],
            "status": "completed",
            "graded_count": len(graded),
        }
        steps = list(state.get("steps") or []) + [step_record]
        return {
            "reranked_chunks": [],
            "graded_chunks": graded,
            "steps": steps,
        }

    return grade_context_node


def make_synthesize_node(resolve_llm: Callable[[], BaseChatModel]) -> NodeCallable:
    """Create the `synthesize` node callable."""

    async def synthesize_node(state: AgentState) -> dict[str, Any]:
        original_query = str(state.get("original_query") or "")
        graded = list(state.get("graded_chunks") or [])
        citations = build_citations_from_chunks(graded)
        xml_context = format_synthesis_context(graded, citations)

        messages = list(state.get("messages") or [])
        prior_context: list[BaseMessage] = messages[:-1][-4:] if len(messages) > 1 else []
        system_prompt = build_synthesis_system_prompt(graded, original_query)

        prompt_messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            *prior_context,
            HumanMessage(
                content=(
                    f"Retrieved Knowledge Context:\n{xml_context}\n\n"
                    f"Question:\n{original_query}\n\n"
                    "Provide a well-structured Markdown answer with inline `[1]`..`[N]` citations."
                )
            ),
        ]

        model_instance = resolve_llm()
        ai_msg = await model_instance.ainvoke(prompt_messages)
        raw_answer = str(ai_msg.content or "").strip()
        sanitized_answer = validate_and_sanitize_citations(raw_answer, citations)

        step_record = {
            "node": "synthesize",
            "label": NODE_STEP_LABELS["synthesize"],
            "status": "completed",
        }
        steps = list(state.get("steps") or []) + [step_record]

        return {
            "messages": [
                AIMessage(
                    content=sanitized_answer,
                    additional_kwargs={
                        "citations": [dict(c) for c in citations],
                        "steps": steps,
                        "routing_mode": state.get("routing_mode", "auto"),
                        "target_domains": list(state.get("target_domains") or []),
                        "polysemy_collisions": list(state.get("polysemy_collisions") or []),
                        "domain_queries": dict(state.get("domain_queries") or {}),
                    },
                )
            ],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "graded_chunks": [],
            "citations": citations,
            "steps": steps,
        }

    return synthesize_node


def make_fallback_notice_node() -> NodeCallable:
    """Create the `fallback_notice` node callable."""

    async def fallback_notice_node(state: AgentState) -> dict[str, Any]:
        original_query = str(state.get("original_query") or "").strip()
        notice = (
            "I could not find sufficient relevant evidence in the indexed knowledge base "
            f"to answer your question{f' about \"{original_query}\"' if original_query else ''}. "
            "Please upload the relevant document(s) or rephrase your question with more specific terms."
        )
        step_record = {
            "node": "fallback_notice",
            "label": NODE_STEP_LABELS["fallback_notice"],
            "status": "completed",
        }
        steps = list(state.get("steps") or []) + [step_record]
        return {
            "messages": [
                AIMessage(
                    content=notice,
                    additional_kwargs={
                        "citations": [],
                        "steps": steps,
                        "routing_mode": state.get("routing_mode", "auto"),
                        "target_domains": list(state.get("target_domains") or []),
                        "polysemy_collisions": list(state.get("polysemy_collisions") or []),
                        "domain_queries": dict(state.get("domain_queries") or {}),
                    },
                )
            ],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "graded_chunks": [],
            "citations": [],
            "steps": steps,
        }

    return fallback_notice_node
