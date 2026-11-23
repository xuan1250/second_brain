"""Qdrant hybrid dense + BM25 sparse retrieval with Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
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

logger = logging.getLogger(__name__)

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
    page_numbers = [
        int(p)
        for p in (payload_dict.get("page_numbers") or [])
        if isinstance(p, (int, str)) and str(p).isdigit()
    ]
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


def _chunk_matches_entities(
    candidate: Mapping[str, Any],
    cleaned_entities_lower: set[str],
) -> bool:
    """Return True if candidate matches extracted entities by payload tag or text substring."""
    if not cleaned_entities_lower:
        return False
    chunk_entities = {
        str(e).strip().lower()
        for e in (candidate.get("associated_entities") or [])
        if str(e).strip()
    }
    if chunk_entities & cleaned_entities_lower:
        return True
    child_lower = str(candidate.get("child_text") or "").lower()
    parent_lower = str(candidate.get("parent_content") or "").lower()
    return any(e in child_lower or e in parent_lower for e in cleaned_entities_lower)


def _is_primary_subject_match(
    candidate: Mapping[str, Any],
    cleaned_entities_lower: set[str],
) -> bool:
    """Return True when queried entity matches the primary subject header rather than an incidental body/group mention."""
    if not cleaned_entities_lower:
        return False

    record_type = str(candidate.get("record_type") or "")
    child_lines = str(candidate.get("child_text") or "").splitlines()
    heading_lower = str(candidate.get("heading_path") or "").lower()

    if record_type == "matrix_effect_col":
        # Check Effect column header banner/tag/description lines (not Triggering Causes line)
        subject_lines = "\n".join(
            ln for ln in child_lines[:3]
            if not ln.lstrip().lower().startswith("triggering causes")
        ).lower()
        subject_text = f"{heading_lower}\n{subject_lines}"
    elif record_type == "matrix_cause_row":
        # Check ONLY CAUSE TAG field (before '| GROUP:') so zone-grouped rows don't crowd Effect queries
        cause_line = next(
            (ln for ln in child_lines[:3] if "cause tag:" in ln.lower()),
            child_lines[0] if child_lines else "",
        )
        subject_text = cause_line.split("|", 1)[0].lower()
    else:
        first_line = child_lines[0].lower() if child_lines else ""
        subject_text = f"{heading_lower}\n{first_line}"

    compact_subject = subject_text.replace("-", "").replace("_", "")
    return any(
        e in subject_text or e.replace("-", "").replace("_", "") in compact_subject
        for e in cleaned_entities_lower
    )


def _prioritize_scrolled_points(
    points: Sequence[Any],
    cleaned_entities_lower: set[str],
    *,
    cap: int,
) -> list[Any]:
    """Order scrolled exact-entity points so primary subject and Effect-column chunks survive culling."""
    if not points:
        return []

    def _priority_key(pt: Any) -> tuple[int, int]:
        payload = dict(getattr(pt, "payload", None) or {})
        cand = {
            "record_type": payload.get("record_type"),
            "heading_path": payload.get("heading_path") or "",
            "child_text": payload.get("child_text") or "",
        }
        is_subj = 1 if _is_primary_subject_match(cand, cleaned_entities_lower) else 0
        rtype = str(payload.get("record_type") or "")
        type_rank = 2 if rtype == "matrix_effect_col" else (1 if rtype in ("cad_logic_diagram", "schedule_row") else 0)
        return (-is_subj, -type_rank)

    return sorted(points, key=_priority_key)[:cap]


def _upsert_exact_tag_candidate(
    candidates_by_id: dict[str, dict[str, Any]],
    point: Any,
    *,
    min_score: float | None = None,
    is_subject: bool = False,
) -> str:
    """Insert or upgrade a Qdrant point as an exact_tag_match candidate and return its chunk_id."""
    payload = dict(point.payload or {})
    chunk_id = str(payload.get("chunk_id") or point.id)
    existing = candidates_by_id.get(chunk_id)
    if existing is not None:
        existing["exact_tag_match"] = True
        if is_subject:
            existing["exact_subject_match"] = True
        if min_score is not None:
            existing["score"] = max(float(existing.get("score", 0.0)), min_score)
    else:
        cand = _point_payload_to_candidate(
            point.id,
            payload,
            score=min_score if min_score is not None else 0.0,
            fallback_domain=str(payload.get("domain") or "general"),
        )
        cand["exact_tag_match"] = True
        if is_subject:
            cand["exact_subject_match"] = True
        candidates_by_id[chunk_id] = cand
    return chunk_id


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

        doc_match: models.Match = (
            models.MatchValue(value=cleaned_doc_ids[0])
            if len(cleaned_doc_ids) == 1
            else models.MatchAny(any=cleaned_doc_ids)
        )
        qfilter = models.Filter(
            must=[
                models.FieldCondition(key="doc_id", match=doc_match),
                models.FieldCondition(
                    key="record_type",
                    match=models.MatchValue(value="document_legend_notes"),
                ),
            ]
        )
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
        dense_vectors, sparse_vectors = await self._encode_queries(query_texts)

        # Isolated per-domain Qdrant searches in parallel via asyncio.gather.
        # Primary RRF MUST NOT hard-filter by associated_entities so non-tagged
        # contextual chunks and Legend/Notes pages remain retrievable.
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

        tasks = [
            _search_single_domain(dom, d_vec, s_vec)
            for dom, d_vec, s_vec in zip(domains_to_query, dense_vectors, sparse_vectors)
        ]
        results = await asyncio.gather(*tasks)

        weights = dict(domain_weights or {})
        raw_cleaned_entities = [str(e).strip() for e in (associated_entities or []) if str(e).strip()]
        cleaned_entities_lower = {e.lower() for e in raw_cleaned_entities}

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
                candidate = _point_payload_to_candidate(
                    point.id,
                    payload,
                    score=0.0,
                    fallback_domain=dom,
                )
                chunk_id = candidate["chunk_id"]
                has_entity_match = _chunk_matches_entities(candidate, cleaned_entities_lower)

                # Calibrated RRF contribution
                rrf_contrib = dom_weight / (float(rrf_k) + float(rank))
                if has_entity_match:
                    rrf_contrib += 1.0 / float(rrf_k)  # Entity-match rank bonus

                rrf_scores_by_id[chunk_id] = rrf_scores_by_id.get(chunk_id, 0.0) + rrf_contrib

                target = candidates_by_id.setdefault(chunk_id, candidate)
                if has_entity_match:
                    target["exact_tag_match"] = True
                    if _is_primary_subject_match(candidate, cleaned_entities_lower):
                        target["exact_subject_match"] = True
                domain_chunk_lists[dom].append(target)

        guaranteed_ids: set[str] = set()

        # Supplementary exact-entity pre-fetch ensures exact tag matches surface even if
        # crowded out of dense/sparse semantic top_limit by similar prose.
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
                rrf_scores_by_id[chunk_id] = rrf_scores_by_id.get(chunk_id, 0.0) + (2.0 / float(rrf_k))
                guaranteed_ids.add(chunk_id)

        # Assign final RRF scores to all candidates
        for c_id, score in rrf_scores_by_id.items():
            if c_id in candidates_by_id:
                candidates_by_id[c_id]["score"] = round(score, 6)

        # Stratified Per-Domain Quotas:
        # Guarantee min_per_domain chunks from each active domain if available
        for dom, c_list in domain_chunk_lists.items():
            sorted_dom = sorted(c_list, key=lambda c: float(c.get("score", 0.0)), reverse=True)
            for c in sorted_dom[:min_per_domain]:
                guaranteed_ids.add(c["chunk_id"])

        selected_candidates: list[dict[str, Any]] = [candidates_by_id[cid] for cid in guaranteed_ids]
        effective_limit = max(top_limit, len(guaranteed_ids))

        # Fill remaining capacity up to effective_limit by highest RRF score
        remaining = [c for cid, c in candidates_by_id.items() if cid not in guaranteed_ids]
        remaining.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)

        slots_left = max(0, effective_limit - len(selected_candidates))
        selected_candidates.extend(remaining[:slots_left])

        # Final sort by descending RRF score
        selected_candidates.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
        return selected_candidates[:effective_limit]

