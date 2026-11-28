"""CRAG query routing schemas, intent classification, polysemy grouping, and graph edge predicates."""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.rag.prompts import _GREETING_RE
from src.rag.state import AgentState


class QueryRewriteResult(BaseModel):
    """Structured output from Node 1 (`rewrite_query`)."""

    needs_retrieval: bool = Field(
        default=True,
        description="False ONLY for pure greetings, pleasantries, or meta assistant questions that do not need knowledge base lookup.",
    )
    rewritten_queries: list[str] = Field(
        default_factory=list,
        description="1 to 3 standalone search queries resolving pronouns from conversation history.",
    )
    routing_mode: str = Field(
        default="single",
        description="'single' for single-domain, 'multi' for cross-domain / polysemy, or 'direct' for no lookup.",
    )
    target_domains: list[str] = Field(
        default_factory=list,
        description="Identified target domains, e.g. ['engineering', 'economics', 'law', 'general'].",
    )
    domain_queries: dict[str, str] = Field(
        default_factory=dict,
        description="Domain-tailored sub-queries mapped by domain key, preserving user query language and identifiers.",
    )
    domain_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Domain importance weights for RRF calibration.",
    )


def _fallback_parse_rewrite(raw_text: str, original_query: str) -> QueryRewriteResult:
    """Fallback parser when a fake/test model returns plain text instead of JSON."""
    text = (raw_text or "").strip()
    lower = text.lower()
    if lower in ("direct_reply", "no_retrieval", "false") or '"needs_retrieval": false' in lower:
        return QueryRewriteResult(needs_retrieval=False, rewritten_queries=[], routing_mode="direct")
    if _GREETING_RE.match(original_query) and "needs_retrieval" not in lower:
        return QueryRewriteResult(needs_retrieval=False, rewritten_queries=[], routing_mode="direct")
    queries = [line.strip("- *•\t ") for line in text.splitlines() if line.strip()]
    if not queries:
        queries = [original_query]
    return QueryRewriteResult(
        needs_retrieval=True,
        rewritten_queries=queries[:3],
        routing_mode="single",
        target_domains=["general"],
        domain_queries={"general": queries[0]},
    )


def _route_after_rewrite(state: AgentState) -> Literal["retrieve", "direct_reply"]:
    if state.get("needs_retrieval", True):
        return "retrieve"
    return "direct_reply"


def _route_after_grade(state: AgentState) -> Literal["synthesize", "fallback_notice"]:
    graded = state.get("graded_chunks") or []
    if len(graded) > 0:
        return "synthesize"
    return "fallback_notice"


_MATRIX_TRACE_INTENT_RE = re.compile(
    r"(?:\bcauses?\b|\beffects?\b|\btrigger(?:s|ed|ing)?\b|\binterlock(?:s|ed|ing)?\b|"
    r"\btrip(?:s|ped|ping)?\b|\bshutdowns?\b|\besd\b|\bmatrices\b|\bmatrix\b|"
    r"(?<!\w)c\s*&\s*e(?!\w)|\bvoting\b|\b2oo3\b|\b2\s+out\s+of\s+3\b|"
    r"\bactivat(?:e|es|ed|ing)\b|\bactuate(?:s|d|ing)?\b|\bnguyen\s+nhan\b|\bnguyên\s+nhân\b|"
    r"\bket\s+qua\b|\bkết\s+quả\b|\blien\s+dong\b|\bliên\s+động\b)",
    flags=re.IGNORECASE,
)

_ENUM_WIRING_INTENT_RE = re.compile(
    r"(?:\blist\s+all\b|\ball\s+terminals?\b|\ball\s+causes?\b|\ball\s+effects?\b|"
    r"\bwiring\b|\bterminals?\b|\bcabinets?\b|\bio\s+list\b|(?<!\w)i/o(?!\w)|"
    r"\bchannels?\b|\bmodules?\b|\bschedules?\b|"
    r"\btat\s+ca\b|\btất\s+cả\b|\bliet\s+ke\b|\bliệt\s+kê\b|\bdau\s+noi\b|\bđấu\s+nối\b)",
    flags=re.IGNORECASE,
)


def _detect_engineering_query_intent(
    query: str,
    extracted_tags: Sequence[str],
) -> tuple[str, int | None]:
    """Classify query intent and return `(query_intent, engineering_top_n)`."""
    text = (query or "").strip()
    if _MATRIX_TRACE_INTENT_RE.search(text):
        return "matrix_trace", 12
    if _ENUM_WIRING_INTENT_RE.search(text):
        return "enumeration", 12
    if extracted_tags:
        return "tag_lookup", 12
    return "prose", None


def build_polysemy_collisions(
    lexicon_matches: Sequence[Any],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Group matched lexicon entries by normalized term and return top polysemic collisions."""
    terms_grouped: dict[str, list[Any]] = {}
    for entry in lexicon_matches:
        key = entry.term.strip().lower()
        terms_grouped.setdefault(key, []).append(entry)

    collisions: list[dict[str, Any]] = []
    for entries in terms_grouped.values():
        doms = sorted({e.domain_id for e in entries if e.domain_id})
        senses = {e.domain_id: e.specific_sense for e in entries if e.domain_id and e.specific_sense}
        warnings = "; ".join(e.contrastive_warning for e in entries if e.contrastive_warning)
        if len(doms) >= 2 or warnings:
            collisions.append(
                {
                    "term": entries[0].term,
                    "domains": doms,
                    "senses": senses,
                    "contrastive_warning": warnings,
                }
            )

    collisions.sort(key=lambda x: len(x["term"]), reverse=True)
    return collisions[:limit]


def resolve_target_domains_and_routing(
    *,
    rewrite_res: QueryRewriteResult,
    original_query: str,
    domain_filter: dict[str, Any],
    catalog: Any,
    polysemy_collisions: Sequence[dict[str, Any]],
    primary_domain: str,
) -> tuple[bool, list[str], dict[str, str], dict[str, float], str, list[str]]:
    """Normalize target domains, weights, sub-queries, and routing mode with explicit-filter precedence."""
    needs_retrieval = bool(rewrite_res.needs_retrieval)
    target_domains = list(rewrite_res.target_domains or [])
    domain_queries = dict(rewrite_res.domain_queries or {})
    domain_weights = dict(rewrite_res.domain_weights or {})
    routing_mode = rewrite_res.routing_mode or ("multi" if len(target_domains) > 1 else "single")

    user_mode = str(domain_filter.get("mode") or "").lower()
    raw_filter_domains = domain_filter.get("domains") or []
    valid_filter_domains = [d for d in raw_filter_domains if d in catalog.domains]
    base_q = rewrite_res.rewritten_queries[0] if rewrite_res.rewritten_queries else original_query

    if user_mode in ("explicit", "single", "federated") and valid_filter_domains:
        target_domains = valid_filter_domains[:5]
        routing_mode = "explicit"
        for dom in target_domains:
            if dom not in domain_queries:
                domain_queries[dom] = base_q
        domain_queries = {d: q for d, q in domain_queries.items() if d in target_domains}
        domain_weights = {d: domain_weights.get(d, 1.0) for d in target_domains}
    elif needs_retrieval:
        valid_targets = [d for d in target_domains if d in catalog.domains]
        if not valid_targets:
            if polysemy_collisions:
                valid_targets = [
                    d
                    for col in polysemy_collisions
                    for d in col.get("domains", [])
                    if d in catalog.domains
                ]
            if not valid_targets and primary_domain in catalog.domains and primary_domain != "general":
                valid_targets = [primary_domain]
            if not valid_targets:
                valid_targets = ["general"]

        # In non-explicit mode: sort specialized domains (primary first, then weights), cap to 3, prune
        specialized = [d for d in valid_targets if d != "general"]
        specialized.sort(key=lambda d: (0 if d == primary_domain else 1, -domain_weights.get(d, 1.0)))
        capped_specialized = specialized[:3]
        if "general" in valid_targets and not capped_specialized:
            target_domains = ["general"]
        else:
            target_domains = capped_specialized

        domain_queries = {d: domain_queries.get(d, base_q) for d in target_domains}
        domain_weights = {d: domain_weights.get(d, 1.0) for d in target_domains}

        if len(target_domains) > 1 and routing_mode == "single":
            routing_mode = "multi"

    rewritten = (
        list(domain_queries.values())
        if domain_queries
        else (rewrite_res.rewritten_queries or [original_query])
    )
    return needs_retrieval, target_domains, domain_queries, domain_weights, routing_mode, rewritten
