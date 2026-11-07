"""Qdrant hybrid dense + BM25 sparse retrieval with Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from src.config import get_settings
from src.db.qdrant import DENSE, SPARSE, get_qdrant_client
from src.ingestion.indexer import (
    BaseEmbedder,
    BgeM3SparseEncoder,
    SparseEncoder,
    get_embedder,
    get_sparse_encoder,
    resolve_compute_device,
    resolve_sparse_mode,
    set_embedder,
    set_sparse_encoder,
)

__all__ = [
    "HybridRetriever",
    "BgeM3SparseEncoder",
    "build_qdrant_filter",
    "build_hybrid_query_kwargs",
    "get_embedder",
    "set_embedder",
    "get_sparse_encoder",
    "set_sparse_encoder",
    "get_active_qdrant_client",
    "set_active_qdrant_client",
    "resolve_compute_device",
    "resolve_sparse_mode",
]

_qdrant_client_override: AsyncQdrantClient | None = None


def set_active_qdrant_client(client: AsyncQdrantClient | None) -> None:
    """Register or clear a shared AsyncQdrantClient (used by FastAPI lifespan and tests)."""
    global _qdrant_client_override
    _qdrant_client_override = client


def get_active_qdrant_client() -> AsyncQdrantClient:
    """Return the active AsyncQdrantClient singleton or create a default client."""
    global _qdrant_client_override
    if _qdrant_client_override is None:
        _qdrant_client_override = get_qdrant_client()
    return _qdrant_client_override


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

    cleaned_doc_ids = [str(d).strip() for d in (doc_ids or []) if str(d).strip()]
    if cleaned_doc_ids:
        if len(cleaned_doc_ids) == 1:
            must_conditions.append(
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=cleaned_doc_ids[0]),
                )
            )
        else:
            must_conditions.append(
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchAny(any=cleaned_doc_ids),
                )
            )

    cleaned_file_types = [str(ft).strip().lower().lstrip(".") for ft in (file_types or []) if str(ft).strip()]
    if cleaned_file_types:
        if len(cleaned_file_types) == 1:
            must_conditions.append(
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value=cleaned_file_types[0]),
                )
            )
        else:
            must_conditions.append(
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchAny(any=cleaned_file_types),
                )
            )

    cleaned_domains = [str(d).strip() for d in (domains or []) if str(d).strip()]
    if cleaned_domains:
        if len(cleaned_domains) == 1:
            must_conditions.append(
                models.FieldCondition(
                    key="domain",
                    match=models.MatchValue(value=cleaned_domains[0]),
                )
            )
        else:
            must_conditions.append(
                models.FieldCondition(
                    key="domain",
                    match=models.MatchAny(any=cleaned_domains),
                )
            )

    cleaned_subdomains = [str(s).strip() for s in (subdomains or []) if str(s).strip()]
    if cleaned_subdomains:
        if len(cleaned_subdomains) == 1:
            must_conditions.append(
                models.FieldCondition(
                    key="subdomain",
                    match=models.MatchValue(value=cleaned_subdomains[0]),
                )
            )
        else:
            must_conditions.append(
                models.FieldCondition(
                    key="subdomain",
                    match=models.MatchAny(any=cleaned_subdomains),
                )
            )

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

    cleaned_entities = [str(e).strip() for e in (associated_entities or []) if str(e).strip()]
    if cleaned_entities:
        if len(cleaned_entities) == 1:
            must_conditions.append(
                models.FieldCondition(
                    key="associated_entities",
                    match=models.MatchValue(value=cleaned_entities[0]),
                )
            )
        else:
            must_conditions.append(
                models.FieldCondition(
                    key="associated_entities",
                    match=models.MatchAny(any=cleaned_entities),
                )
            )

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


class HybridRetriever:
    """Executes Qdrant hybrid (dense bge-m3 + sparse BM25) search fused via RRF."""

    def __init__(
        self,
        qdrant: AsyncQdrantClient | None = None,
        embedder: BaseEmbedder | None = None,
        sparse_encoder: SparseEncoder | None = None,
        collection: str | None = None,
        limit: int = 20,
    ) -> None:
        settings = get_settings()
        self.qdrant = qdrant or get_active_qdrant_client()
        self.embedder = embedder or get_embedder()
        self.sparse_encoder = sparse_encoder or get_sparse_encoder()
        self.collection = collection or settings.qdrant_collection
        self.limit = limit

    async def retrieve(
        self,
        queries: str | Sequence[str],
        *,
        doc_ids: Sequence[str] | None = None,
        file_types: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Embed query/queries and execute hybrid RRF retrieval against Qdrant.

        Returns candidate dicts sorted by descending RRF score.
        """
        if isinstance(queries, str):
            query_list = [queries.strip()] if queries.strip() else []
        else:
            query_list = [q.strip() for q in queries if q and q.strip()]

        if not query_list:
            return []

        if not await self.qdrant.collection_exists(self.collection):
            return []

        top_limit = limit or self.limit
        if isinstance(self.sparse_encoder, BgeM3SparseEncoder) and getattr(
            self.embedder, "supports_native_sparse", False
        ):
            dense_vectors, sparse_vectors = await self.embedder.embed_documents_with_sparse(query_list)
        else:
            dense_vectors = await self.embedder.embed_documents(query_list)
            sparse_vectors = await self.sparse_encoder.encode(query_list)

        merged_by_chunk: dict[str, dict[str, Any]] = {}

        for dense_vec, sparse_vec in zip(dense_vectors, sparse_vectors):
            query_kwargs = build_hybrid_query_kwargs(
                collection_name=self.collection,
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                limit=top_limit,
                doc_ids=doc_ids,
                file_types=file_types,
            )
            response = await self.qdrant.query_points(**query_kwargs)
            for point in response.points:
                payload = dict(point.payload or {})
                chunk_id = str(payload.get("chunk_id") or point.id)
                score = float(point.score or 0.0)
                page_numbers = [
                    int(p)
                    for p in (payload.get("page_numbers") or [])
                    if isinstance(p, (int, str)) and str(p).isdigit()
                ]
                candidate: dict[str, Any] = {
                    "chunk_id": chunk_id,
                    "point_id": str(point.id),
                    "doc_id": str(payload.get("doc_id") or ""),
                    "source_filename": str(payload.get("source_filename") or ""),
                    "file_type": str(payload.get("file_type") or ""),
                    "heading_path": str(payload.get("heading_path") or ""),
                    "page_numbers": page_numbers,
                    "token_count": int(payload.get("token_count") or 0),
                    "child_text": str(payload.get("child_text") or ""),
                    "parent_content": str(payload.get("parent_content") or payload.get("child_text") or ""),
                    "contextual_prefix": str(payload.get("contextual_prefix") or ""),
                    "domain": str(payload.get("domain") or "general"),
                    "subdomain": str(payload.get("subdomain") or "general"),
                    "epistemic_type": str(payload.get("epistemic_type") or "general"),
                    "authority_tier": int(payload.get("authority_tier", 3)),
                    "jurisdiction": str(payload.get("jurisdiction") or ""),
                    "is_current": bool(payload.get("is_current", True)),
                    "associated_entities": list(payload.get("associated_entities") or []),
                    "score": score,
                }
                existing = merged_by_chunk.get(chunk_id)
                if existing is None or score > float(existing.get("score", 0.0)):
                    merged_by_chunk[chunk_id] = candidate

        sorted_candidates = sorted(
            merged_by_chunk.values(),
            key=lambda c: float(c.get("score", 0.0)),
            reverse=True,
        )
        return sorted_candidates[:top_limit]

    async def retrieve_federated(
        self,
        domain_queries: dict[str, str],
        *,
        mode: str = "auto",
        domain_weights: dict[str, float] | None = None,
        doc_ids: Sequence[str] | None = None,
        file_types: Sequence[str] | None = None,
        subdomains: Sequence[str] | None = None,
        max_authority_tier: int | None = None,
        only_current: bool = True,
        associated_entities: Sequence[str] | None = None,
        limit: int | None = None,
        min_per_domain: int = 2,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """Execute parallel isolated per-domain hybrid searches and fuse via Domain-Calibrated RRF."""
        if not domain_queries:
            return []

        if not await self.qdrant.collection_exists(self.collection):
            return []

        top_limit = limit or self.limit
        active_queries = {k.strip(): v.strip() for k, v in domain_queries.items() if k and v and v.strip()}
        if not active_queries:
            return []

        # In auto/adaptive mode (anything other than explicit user lock),
        # include "general" domain fallback with 0.5x discount if not present.
        # In explicit mode, strictly exclude "general" unless caller explicitly asked for it in domains.
        if mode != "explicit":
            if "general" not in active_queries:
                primary_q = next(iter(active_queries.values()))
                active_queries["general"] = primary_q
        else:
            if "general" in active_queries and len(active_queries) > 1:
                active_queries.pop("general", None)

        domains_to_query = list(active_queries.keys())
        query_texts = [active_queries[d] for d in domains_to_query]

        # Batch-encode all sub-queries in a single call
        if isinstance(self.sparse_encoder, BgeM3SparseEncoder) and getattr(
            self.embedder, "supports_native_sparse", False
        ):
            dense_vectors, sparse_vectors = await self.embedder.embed_documents_with_sparse(query_texts)
        else:
            dense_vectors = await self.embedder.embed_documents(query_texts)
            sparse_vectors = await self.sparse_encoder.encode(query_texts)

        # Isolated per-domain Qdrant searches in parallel via asyncio.gather
        async def _search_single_domain(
            dom: str,
            d_vec: list[float],
            s_vec: Any,
        ) -> tuple[str, list[models.ScoredPoint]]:
            kwargs = build_hybrid_query_kwargs(
                collection_name=self.collection,
                dense_vector=d_vec,
                sparse_vector=s_vec,
                limit=top_limit,
                doc_ids=doc_ids,
                file_types=file_types,
                domains=[dom],
                subdomains=subdomains,
                max_authority_tier=max_authority_tier,
                only_current=only_current,
                associated_entities=associated_entities,
            )
            resp = await self.qdrant.query_points(**kwargs)
            return dom, list(resp.points)

        tasks = [
            _search_single_domain(dom, d_vec, s_vec)
            for dom, d_vec, s_vec in zip(domains_to_query, dense_vectors, sparse_vectors)
        ]
        results = await asyncio.gather(*tasks)

        weights = dict(domain_weights or {})
        cleaned_entities = {str(e).strip().lower() for e in (associated_entities or []) if str(e).strip()}

        candidates_by_id: dict[str, dict[str, Any]] = {}
        rrf_scores_by_id: dict[str, float] = {}
        domain_chunk_lists: dict[str, list[dict[str, Any]]] = {d: [] for d in domains_to_query}

        for dom, points in results:
            dom_weight = float(weights.get(dom, 1.0))
            if dom == "general" and mode == "auto":
                dom_weight *= 0.5  # 0.5x RRF discount for general fallback

            for rank_0, point in enumerate(points):
                rank = rank_0 + 1
                payload = dict(point.payload or {})
                chunk_id = str(payload.get("chunk_id") or point.id)

                # Entity match check
                chunk_entities = {
                    str(e).strip().lower()
                    for e in (payload.get("associated_entities") or [])
                    if str(e).strip()
                }
                child_text = str(payload.get("child_text") or "")
                parent_content = str(payload.get("parent_content") or child_text)
                has_entity_match = False
                if cleaned_entities:
                    if chunk_entities & cleaned_entities:
                        has_entity_match = True
                    elif any(e in child_text.lower() or e in parent_content.lower() for e in cleaned_entities):
                        has_entity_match = True

                # Calibrated RRF contribution
                rrf_contrib = dom_weight / (float(rrf_k) + float(rank))
                if has_entity_match:
                    rrf_contrib += (1.0 / float(rrf_k))  # Entity-match rank bonus

                rrf_scores_by_id[chunk_id] = rrf_scores_by_id.get(chunk_id, 0.0) + rrf_contrib

                page_numbers = [
                    int(p)
                    for p in (payload.get("page_numbers") or [])
                    if isinstance(p, (int, str)) and str(p).isdigit()
                ]
                candidate: dict[str, Any] = {
                    "chunk_id": chunk_id,
                    "point_id": str(point.id),
                    "doc_id": str(payload.get("doc_id") or ""),
                    "source_filename": str(payload.get("source_filename") or ""),
                    "file_type": str(payload.get("file_type") or ""),
                    "heading_path": str(payload.get("heading_path") or ""),
                    "page_numbers": page_numbers,
                    "token_count": int(payload.get("token_count") or 0),
                    "child_text": child_text,
                    "parent_content": parent_content,
                    "contextual_prefix": str(payload.get("contextual_prefix") or ""),
                    "domain": str(payload.get("domain") or dom),
                    "subdomain": str(payload.get("subdomain") or "general"),
                    "epistemic_type": str(payload.get("epistemic_type") or "general"),
                    "authority_tier": int(payload.get("authority_tier", 3)),
                    "jurisdiction": str(payload.get("jurisdiction") or ""),
                    "is_current": bool(payload.get("is_current", True)),
                    "associated_entities": list(payload.get("associated_entities") or []),
                    "score": 0.0,
                }
                candidates_by_id[chunk_id] = candidate
                domain_chunk_lists[dom].append(candidate)

        # Assign final RRF scores to all candidates
        for c_id, score in rrf_scores_by_id.items():
            candidates_by_id[c_id]["score"] = round(score, 6)

        # Stratified Per-Domain Quotas:
        # Guarantee min_per_domain chunks from each active domain if available
        guaranteed_ids: set[str] = set()
        for dom, c_list in domain_chunk_lists.items():
            sorted_dom = sorted(c_list, key=lambda c: float(c.get("score", 0.0)), reverse=True)
            for c in sorted_dom[:min_per_domain]:
                guaranteed_ids.add(c["chunk_id"])

        selected_candidates: list[dict[str, Any]] = [candidates_by_id[cid] for cid in guaranteed_ids]

        # Fill remaining capacity up to top_limit by highest RRF score
        remaining = [c for cid, c in candidates_by_id.items() if cid not in guaranteed_ids]
        remaining.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)

        slots_left = max(0, top_limit - len(selected_candidates))
        selected_candidates.extend(remaining[:slots_left])

        # Final sort by descending RRF score
        selected_candidates.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
        return selected_candidates[:top_limit]

