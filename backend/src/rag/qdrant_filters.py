"""Qdrant filter construction, hybrid RRF query kwargs, and candidate payload normalization."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from qdrant_client import models

from src.db.qdrant import DENSE, SPARSE


def normalize_page_numbers(raw_pages: Any) -> list[int]:
    """Normalize a collection of page values into sorted, deduplicated non-negative integers."""
    return sorted(
        {
            int(p)
            for p in (raw_pages or [])
            if isinstance(p, (int, str)) and not isinstance(p, bool) and str(p).isdigit()
        }
    )


def _add_match_condition(
    conditions: list[models.Condition],
    key: str,
    values: Sequence[Any] | None,
    *,
    normalize: Callable[[str], str] | None = None,
) -> None:
    """Append a Qdrant MatchValue or MatchAny FieldCondition for cleaned non-empty values."""
    cleaned: list[str] = []
    for raw in values or []:
        val = str(raw).strip()
        if not val:
            continue
        if normalize is not None:
            val = normalize(val)
        if val:
            cleaned.append(val)
    if not cleaned:
        return
    if len(cleaned) == 1:
        conditions.append(
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=cleaned[0]),
            )
        )
    else:
        conditions.append(
            models.FieldCondition(
                key=key,
                match=models.MatchAny(any=cleaned),
            )
        )


def build_qdrant_filter(
    doc_ids: Sequence[str] | None = None,
    file_types: Sequence[str] | None = None,
    *,
    domains: Sequence[str] | None = None,
    subdomains: Sequence[str] | None = None,
    max_authority_tier: int | None = None,
    only_current: bool = False,
    associated_entities: Sequence[str] | None = None,
) -> models.Filter | None:
    """Construct a Qdrant payload Filter with optional domain, authority, and entity constraints."""
    must_conditions: list[models.Condition] = []

    _add_match_condition(must_conditions, "doc_id", doc_ids)
    _add_match_condition(
        must_conditions,
        "file_type",
        file_types,
        normalize=lambda ft: ft.lower().lstrip("."),
    )
    _add_match_condition(must_conditions, "domain", domains)
    _add_match_condition(must_conditions, "subdomain", subdomains)

    if max_authority_tier is not None:
        must_conditions.append(
            models.FieldCondition(
                key="authority_tier",
                range=models.Range(lte=max_authority_tier),
            )
        )

    if only_current:
        must_conditions.append(
            models.FieldCondition(
                key="is_current",
                match=models.MatchValue(value=True),
            )
        )

    _add_match_condition(must_conditions, "associated_entities", associated_entities)

    if not must_conditions:
        return None
    return models.Filter(must=must_conditions)


def build_hybrid_query_kwargs(
    collection_name: str,
    dense_vector: list[float],
    sparse_vector: models.SparseVector,
    limit: int = 20,
    prefetch_limit: int | None = None,
    doc_ids: Sequence[str] | None = None,
    file_types: Sequence[str] | None = None,
    *,
    domains: Sequence[str] | None = None,
    subdomains: Sequence[str] | None = None,
    max_authority_tier: int | None = None,
    only_current: bool = False,
    associated_entities: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build exact keyword arguments for `AsyncQdrantClient.query_points` with RRF fusion."""
    pf_limit = max(limit, prefetch_limit or limit)
    qfilter = build_qdrant_filter(
        doc_ids=doc_ids,
        file_types=file_types,
        domains=domains,
        subdomains=subdomains,
        max_authority_tier=max_authority_tier,
        only_current=only_current,
        associated_entities=associated_entities,
    )

    prefetch = [
        models.Prefetch(
            query=dense_vector,
            using=DENSE,
            limit=pf_limit,
            filter=qfilter,
        ),
        models.Prefetch(
            query=sparse_vector,
            using=SPARSE,
            limit=pf_limit,
            filter=qfilter,
        ),
    ]
    return {
        "collection_name": collection_name,
        "prefetch": prefetch,
        "query": models.FusionQuery(fusion=models.Fusion.RRF),
        "query_filter": qfilter,
        "limit": limit,
        "with_payload": True,
    }


def _point_payload_to_candidate(
    point_id: Any,
    payload: Mapping[str, Any] | None,
    *,
    score: float = 0.0,
    fallback_domain: str = "general",
) -> dict[str, Any]:
    """Convert a Qdrant point ID and payload dict into a normalized RAG candidate dict."""
    payload_dict = dict(payload or {})
    chunk_id = str(payload_dict.get("chunk_id") or point_id)
    child_text = str(payload_dict.get("child_text") or "")
    parent_content = str(payload_dict.get("parent_content") or child_text)
    page_numbers = normalize_page_numbers(payload_dict.get("page_numbers"))
    raw_tier = payload_dict.get("authority_tier")
    authority_tier = int(raw_tier) if raw_tier is not None else 3
    return {
        "chunk_id": chunk_id,
        "point_id": str(point_id),
        "doc_id": str(payload_dict.get("doc_id") or ""),
        "source_filename": str(payload_dict.get("source_filename") or ""),
        "file_type": str(payload_dict.get("file_type") or ""),
        "heading_path": str(payload_dict.get("heading_path") or ""),
        "page_numbers": page_numbers,
        "token_count": int(payload_dict.get("token_count") or 0),
        "child_text": child_text,
        "parent_content": parent_content,
        "contextual_prefix": str(payload_dict.get("contextual_prefix") or ""),
        "domain": str(payload_dict.get("domain") or fallback_domain),
        "subdomain": str(payload_dict.get("subdomain") or "general"),
        "epistemic_type": str(payload_dict.get("epistemic_type") or "general"),
        "authority_tier": authority_tier,
        "jurisdiction": str(payload_dict.get("jurisdiction") or ""),
        "is_current": bool(payload_dict.get("is_current", True)),
        "associated_entities": list(payload_dict.get("associated_entities") or []),
        "record_type": payload_dict.get("record_type"),
        "score": float(score),
    }
