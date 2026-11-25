"""Qdrant hybrid dense + BM25 sparse retrieval with Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from src.config import get_settings
from src.db.qdrant import get_qdrant_client
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
from src.rag.entity_matching import (
    _accumulate_federated_rrf,
    _chunk_matches_entities,
    _is_primary_subject_match,
    _prioritize_scrolled_points,
    _select_stratified_candidates,
    _upsert_exact_tag_candidate,
)
from src.rag.qdrant_filters import (
    _add_match_condition,
    _point_payload_to_candidate,
    build_hybrid_query_kwargs,
    build_qdrant_filter,
    normalize_page_numbers,
)

logger = logging.getLogger(__name__)

__all__ = [
    "HybridRetriever",
    "BgeM3SparseEncoder",
    "build_qdrant_filter",
    "build_hybrid_query_kwargs",
    "normalize_page_numbers",
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

    async def _encode_queries(
        self,
        query_texts: Sequence[str],
    ) -> tuple[list[list[float]], list[Any]]:
        """Encode queries into paired dense and sparse vectors."""
        if isinstance(self.sparse_encoder, BgeM3SparseEncoder) and getattr(
            self.embedder, "supports_native_sparse", False
        ):
            return await self.embedder.embed_documents_with_sparse(query_texts)
        dense_vectors = await self.embedder.embed_documents(query_texts)
        sparse_vectors = await self.sparse_encoder.encode(query_texts)
        return dense_vectors, sparse_vectors

    async def _scroll_exact_entity_points(
        self,
        *,
        associated_entities: Sequence[str] | None,
        doc_ids: Sequence[str] | None = None,
        file_types: Sequence[str] | None = None,
        domains: Sequence[str] | None = None,
        subdomains: Sequence[str] | None = None,
        max_authority_tier: int | None = None,
        only_current: bool = False,
        limit: int = 24,
    ) -> list[Any]:
        """Scroll Qdrant for exact payload matches on `associated_entities`."""
        cleaned = [str(e).strip() for e in (associated_entities or []) if str(e).strip()]
        if not cleaned:
            return []

        qfilter = build_qdrant_filter(
            doc_ids=doc_ids,
            file_types=file_types,
            domains=domains,
            subdomains=subdomains,
            max_authority_tier=max_authority_tier,
            only_current=only_current,
            associated_entities=cleaned,
        )
        try:
            points, _ = await self.qdrant.scroll(
                collection_name=self.collection,
                scroll_filter=qfilter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return list(points or [])
        except Exception as exc:
            logger.debug("Exact-entity scroll failed on collection '%s': %s", self.collection, exc)
            return []

    async def fetch_document_legends(
        self,
        doc_ids: Sequence[str],
        *,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        """Fetch document-level legend/notes chunks (`record_type='document_legend_notes'`) for given docs."""
        cleaned_doc_ids = list(dict.fromkeys(str(d).strip() for d in (doc_ids or []) if str(d).strip()))
        if not cleaned_doc_ids:
            return []

        if not await self.qdrant.collection_exists(self.collection):
            return []

        legend_conditions: list[models.Condition] = []
        _add_match_condition(legend_conditions, "doc_id", cleaned_doc_ids)
        _add_match_condition(legend_conditions, "record_type", ["document_legend_notes"])
        qfilter = models.Filter(must=legend_conditions)
        try:
            points, _ = await self.qdrant.scroll(
                collection_name=self.collection,
                scroll_filter=qfilter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.debug("Legend scroll failed on collection '%s': %s", self.collection, exc)
            return []

        legend_candidates: list[dict[str, Any]] = []
        for point in points or []:
            payload = dict(point.payload or {})
            cand = _point_payload_to_candidate(
                point.id,
                payload,
                score=0.72,
                fallback_domain=str(payload.get("domain") or "general"),
            )
            cand["is_legend_context"] = True
            cand["record_type"] = "document_legend_notes"
            legend_candidates.append(cand)
        return legend_candidates

    async def retrieve(
        self,
        queries: str | Sequence[str],
        *,
        doc_ids: Sequence[str] | None = None,
        file_types: Sequence[str] | None = None,
        associated_entities: Sequence[str] | None = None,
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
        dense_vectors, sparse_vectors = await self._encode_queries(query_list)

        cleaned_entities = [str(e).strip() for e in (associated_entities or []) if str(e).strip()]
        cleaned_entities_lower = {e.lower() for e in cleaned_entities}
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
                score = float(point.score or 0.0)
                candidate = _point_payload_to_candidate(
                    point.id,
                    payload,
                    score=score,
                    fallback_domain="general",
                )
                if _chunk_matches_entities(candidate, cleaned_entities_lower):
                    candidate["exact_tag_match"] = True
                    if _is_primary_subject_match(candidate, cleaned_entities_lower):
                        candidate["exact_subject_match"] = True
                chunk_id = candidate["chunk_id"]
                existing = merged_by_chunk.get(chunk_id)
                if existing is None or score > float(existing.get("score", 0.0)):
                    if existing is not None and existing.get("exact_tag_match"):
                        candidate["exact_tag_match"] = True
                    if existing is not None and existing.get("exact_subject_match"):
                        candidate["exact_subject_match"] = True
                    merged_by_chunk[chunk_id] = candidate

        if cleaned_entities:
            supp_points = await self._scroll_exact_entity_points(
                associated_entities=cleaned_entities,
                doc_ids=doc_ids,
                file_types=file_types,
                limit=max(top_limit * 5, 120),
            )
            supp_points = _prioritize_scrolled_points(
                supp_points,
                cleaned_entities_lower,
                cap=max(top_limit * 2, 24),
            )
            for point in supp_points:
                pl = dict(point.payload or {})
                is_subj = _is_primary_subject_match(pl, cleaned_entities_lower)
                _upsert_exact_tag_candidate(
                    merged_by_chunk,
                    point,
                    min_score=0.88 if is_subj else 0.85,
                    is_subject=is_subj,
                )

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
        if not domain_queries or not await self.qdrant.collection_exists(self.collection):
            return []

        top_limit = limit or self.limit
        active_queries = {k.strip(): v.strip() for k, v in domain_queries.items() if k and v and v.strip()}
        if not active_queries:
            return []

        if mode != "explicit":
            if "general" not in active_queries:
                active_queries["general"] = next(iter(active_queries.values()))
        elif "general" in active_queries and len(active_queries) > 1:
            active_queries.pop("general", None)

        domains_to_query = list(active_queries.keys())
        dense_vectors, sparse_vectors = await self._encode_queries([active_queries[d] for d in domains_to_query])

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
                associated_entities=None,
            )
            resp = await self.qdrant.query_points(**kwargs)
            return dom, list(resp.points)

        results = await asyncio.gather(
            *[
                _search_single_domain(dom, d_vec, s_vec)
                for dom, d_vec, s_vec in zip(domains_to_query, dense_vectors, sparse_vectors)
            ]
        )

        raw_cleaned_entities = [str(e).strip() for e in (associated_entities or []) if str(e).strip()]
        cleaned_entities_lower = {e.lower() for e in raw_cleaned_entities}
        candidates_by_id, rrf_scores_by_id, domain_chunk_lists = _accumulate_federated_rrf(
            results,
            domains_to_query,
            weights=dict(domain_weights or {}),
            mode=mode,
            rrf_k=rrf_k,
            cleaned_entities_lower=cleaned_entities_lower,
        )

        guaranteed_ids: set[str] = set()
        if raw_cleaned_entities:
            scroll_limit = max(top_limit * 5, 120)
            supp_points = await self._scroll_exact_entity_points(
                associated_entities=raw_cleaned_entities,
                doc_ids=doc_ids,
                file_types=file_types,
                domains=domains_to_query,
                subdomains=subdomains,
                max_authority_tier=max_authority_tier,
                only_current=only_current,
                limit=scroll_limit,
            )
            if not supp_points and mode != "explicit":
                supp_points = await self._scroll_exact_entity_points(
                    associated_entities=raw_cleaned_entities,
                    doc_ids=doc_ids,
                    file_types=file_types,
                    domains=None,
                    subdomains=subdomains,
                    max_authority_tier=max_authority_tier,
                    only_current=only_current,
                    limit=scroll_limit,
                )

            supp_points = _prioritize_scrolled_points(
                supp_points,
                cleaned_entities_lower,
                cap=max(top_limit * 2, 32),
            )
            for point in supp_points:
                pl = dict(point.payload or {})
                is_subj = _is_primary_subject_match(pl, cleaned_entities_lower)
                chunk_id = _upsert_exact_tag_candidate(candidates_by_id, point, is_subject=is_subj)
                rrf_scores_by_id[chunk_id] = rrf_scores_by_id.get(chunk_id, 0.0) + (2.0 / float(rrf_k))
                guaranteed_ids.add(chunk_id)

        return _select_stratified_candidates(
            candidates_by_id,
            rrf_scores_by_id,
            domain_chunk_lists,
            guaranteed_ids,
            top_limit=top_limit,
            min_per_domain=min_per_domain,
        )
