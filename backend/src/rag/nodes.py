"""LangGraph node factories and step metadata for the 7-node Corrective RAG workflow."""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.db.repository import get_cross_domain_entity_matches, get_lexicon_matches
from src.db.session import get_session_factory
from src.ingestion.classifier import (
    LayaDomainClassifier,
    StubLayaClassifier,
    get_classifier,
    get_taxonomy_catalog,
    refresh_taxonomy_catalog,
)
from src.ingestion.entity_extractor import EntityExtractor
from src.rag.generation_nodes import (
    NODE_STEP_LABELS,
    NodeCallable,
    _accepts_kwarg,
    make_fallback_notice_node,
    make_grade_context_node,
    make_rerank_node,
    make_synthesize_node,
)
from src.rag.llm import invoke_structured_output
from src.rag.prompts import _DIRECT_REPLY_SYSTEM_PROMPT, build_rewrite_messages
from src.rag.retriever import HybridRetriever
from src.rag.routing import (
    QueryRewriteResult,
    _detect_engineering_query_intent,
    _fallback_parse_rewrite,
    build_polysemy_collisions,
    resolve_target_domains_and_routing,
)
from src.rag.state import AgentState

logger = logging.getLogger("uvicorn.error")


def make_rewrite_query_node(
    resolve_llm: Callable[[], BaseChatModel],
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> NodeCallable:
    """Create the `rewrite_query` node callable."""

    async def rewrite_query_node(state: AgentState) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        original_query = str(state.get("original_query") or "").strip()
        if not original_query and messages:
            last_msg = messages[-1]
            original_query = str(getattr(last_msg, "content", "") or "").strip()

        catalog = get_taxonomy_catalog()
        lexicon_matches = []
        entity_matches = []
        try:
            db_factory = session_factory or get_session_factory()
            async with db_factory() as session:
                catalog = await refresh_taxonomy_catalog(session)
                lexicon_matches = await get_lexicon_matches(session, original_query)
                entity_matches = await get_cross_domain_entity_matches(session, original_query)
        except Exception as exc:
            logger.debug("[CRAG-REWRITE] Lexicon/entity lookup skipped or failed: %s", exc)

        polysemy_collisions = build_polysemy_collisions(lexicon_matches)

        db_matched_entities = [e.canonical_name for e in entity_matches if getattr(e, "canonical_name", None)]
        eng_tags = EntityExtractor().extract_query_tags_and_stems(original_query)
        matched_entities = list(
            dict.fromkeys(
                clean_ent
                for clean_ent in (str(ent).strip() for ent in (*eng_tags, *db_matched_entities))
                if clean_ent
            )
        )
        query_intent, engineering_top_n = _detect_engineering_query_intent(original_query, eng_tags)

        primary_domain = "general"
        confidence = 0.5
        try:
            classifier = get_classifier()
            if isinstance(classifier, LayaDomainClassifier) and not classifier.is_resident:
                try:
                    classifier._ensure_model_loaded()
                    cls_res = await classifier.classify_document(original_query)
                except Exception:
                    stub = StubLayaClassifier()
                    cls_res = await stub.classify_document(original_query)
            else:
                cls_res = await classifier.classify_document(original_query)
            primary_domain = cls_res.recommended.primary_domain
            confidence = cls_res.recommended.confidence
        except Exception as exc:
            logger.debug("[CRAG-REWRITE] Classifier pre-scoring failed: %s", exc)

        domain_filter = state.get("domain_filter") or {}
        prompt_messages = build_rewrite_messages(
            messages=messages,
            original_query=original_query,
            catalog=catalog,
            polysemy_collisions=polysemy_collisions,
            matched_entities=matched_entities,
            primary_domain=primary_domain,
            confidence=confidence,
            domain_filter=domain_filter,
        )

        model_instance = resolve_llm()
        rewrite_res = await invoke_structured_output(
            llm=model_instance,
            schema=QueryRewriteResult,
            messages=prompt_messages,
            fallback_parser=lambda raw: _fallback_parse_rewrite(raw, original_query),
        )

        (
            needs_retrieval,
            target_domains,
            domain_queries,
            domain_weights,
            routing_mode,
            rewritten,
        ) = resolve_target_domains_and_routing(
            rewrite_res=rewrite_res,
            original_query=original_query,
            domain_filter=domain_filter,
            catalog=catalog,
            polysemy_collisions=polysemy_collisions,
            primary_domain=primary_domain,
        )

        step_record = {
            "node": "rewrite_query",
            "label": NODE_STEP_LABELS["rewrite_query"],
            "status": "completed",
            "routing_mode": routing_mode,
            "target_domains": target_domains,
            "domain_queries": domain_queries,
            "polysemy_collisions": polysemy_collisions,
            "query_intent": query_intent,
            "engineering_top_n": engineering_top_n,
        }
        steps = list(state.get("steps") or []) + [step_record]

        return {
            "original_query": original_query,
            "needs_retrieval": needs_retrieval,
            "rewritten_queries": rewritten,
            "routing_mode": routing_mode,
            "target_domains": target_domains,
            "domain_queries": domain_queries,
            "domain_weights": domain_weights,
            "polysemy_collisions": polysemy_collisions,
            "matched_entities": matched_entities,
            "query_intent": query_intent,
            "engineering_top_n": engineering_top_n,
            "steps": steps,
        }

    return rewrite_query_node


def make_direct_reply_node(resolve_llm: Callable[[], BaseChatModel]) -> NodeCallable:
    """Create the `direct_reply` node callable."""

    async def direct_reply_node(state: AgentState) -> dict[str, Any]:
        original_query = str(state.get("original_query") or "").strip()
        messages = list(state.get("messages") or [])
        prompt_messages: list[BaseMessage] = [
            SystemMessage(content=_DIRECT_REPLY_SYSTEM_PROMPT),
            *messages[-6:],
        ]
        if not messages or not isinstance(messages[-1], HumanMessage):
            prompt_messages.append(HumanMessage(content=original_query))

        model_instance = resolve_llm()
        ai_msg = await model_instance.ainvoke(prompt_messages)
        reply_text = str(ai_msg.content or "").strip()
        step_record = {
            "node": "direct_reply",
            "label": NODE_STEP_LABELS["direct_reply"],
            "status": "completed",
        }
        steps = list(state.get("steps") or []) + [step_record]
        return {
            "messages": [
                AIMessage(
                    content=reply_text,
                    additional_kwargs={
                        "citations": [],
                        "steps": steps,
                        "routing_mode": state.get("routing_mode", "direct"),
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

    return direct_reply_node


def make_retrieve_node(active_retriever: HybridRetriever) -> NodeCallable:
    """Create the `retrieve` node callable."""

    async def retrieve_node(state: AgentState) -> dict[str, Any]:
        domain_queries = dict(state.get("domain_queries") or {})
        filter_doc_ids = state.get("filter_doc_ids")
        domain_filter = state.get("domain_filter") or {}
        mode = str((domain_filter.get("mode") if domain_filter else None) or "auto")
        subdomains = domain_filter.get("subdomains")
        max_authority_tier = domain_filter.get("max_authority_tier")
        only_current = domain_filter.get("only_current", True)
        matched_entities = state.get("matched_entities")

        if domain_queries and hasattr(active_retriever, "retrieve_federated"):
            candidates = await active_retriever.retrieve_federated(
                domain_queries=domain_queries,
                mode=mode,
                domain_weights=state.get("domain_weights"),
                doc_ids=filter_doc_ids,
                subdomains=subdomains,
                max_authority_tier=max_authority_tier,
                only_current=only_current,
                associated_entities=matched_entities,
            )
        else:
            queries = list(state.get("rewritten_queries") or [])
            if not queries:
                orig = str(state.get("original_query") or "").strip()
                if orig:
                    queries = [orig]
            retrieve_kwargs: dict[str, Any] = {
                "queries": queries,
                "doc_ids": filter_doc_ids,
            }
            if _accepts_kwarg(active_retriever.retrieve, "associated_entities"):
                retrieve_kwargs["associated_entities"] = matched_entities
            candidates = await active_retriever.retrieve(**retrieve_kwargs)

        candidates = [dict(c) for c in (candidates or [])]

        if (
            state.get("query_intent", "prose") != "prose"
            and candidates
            and hasattr(active_retriever, "fetch_document_legends")
        ):
            doc_ids_for_legends = list(
                dict.fromkeys(
                    d_id
                    for d_id in (str(cand.get("doc_id") or "").strip() for cand in candidates)
                    if d_id
                )
            )
            if doc_ids_for_legends:
                try:
                    legend_chunks = await active_retriever.fetch_document_legends(doc_ids_for_legends)
                except Exception as exc:
                    logger.debug("[CRAG-RETRIEVE] Legend context fetch skipped or failed: %s", exc)
                    legend_chunks = []

                by_chunk_id: dict[str, dict[str, Any]] = {
                    str(c.get("chunk_id") or ""): c
                    for c in candidates
                    if c.get("chunk_id")
                }
                for leg in legend_chunks or []:
                    if not isinstance(leg, dict):
                        continue
                    leg_id = str(leg.get("chunk_id") or "")
                    if leg_id and leg_id in by_chunk_id:
                        by_chunk_id[leg_id]["is_legend_context"] = True
                    else:
                        leg_copy = dict(leg)
                        leg_copy["is_legend_context"] = True
                        candidates.append(leg_copy)
                        if leg_id:
                            by_chunk_id[leg_id] = leg_copy

        step_record = {
            "node": "retrieve",
            "label": NODE_STEP_LABELS["retrieve"],
            "status": "completed",
            "retrieved_count": len(candidates),
        }
        steps = list(state.get("steps") or []) + [step_record]
        return {
            "retrieved_chunks": candidates,
            "steps": steps,
        }

    return retrieve_node


__all__ = [
    "NODE_STEP_LABELS",
    "NodeCallable",
    "_accepts_kwarg",
    "make_direct_reply_node",
    "make_fallback_notice_node",
    "make_grade_context_node",
    "make_rerank_node",
    "make_retrieve_node",
    "make_rewrite_query_node",
    "make_synthesize_node",
]
