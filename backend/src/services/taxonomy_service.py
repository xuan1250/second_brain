"""Domain taxonomy, subdomain, and polysemy lexicon business logic service."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, status
from qdrant_client import AsyncQdrantClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings
from src.db.models import Document, DomainRegistry, SubdomainRegistry
from src.db.repository import (
    create_domain,
    create_subdomain,
    delete_domain_and_reassign,
    delete_subdomain,
    get_domain_by_slug,
    list_subdomains,
    list_subdomains_by_domain_ids,
    update_domain,
    update_subdomain,
)
from src.ingestion.worker import IngestionOrchestrator
from src.services.lexicon_service import (
    create_lexicon_entry,
    delete_lexicon_entry,
    list_lexicon_entries,
)
from src.services.reindex_service import find_reindexable_doc_ids, queue_documents_for_reindex

if TYPE_CHECKING:
    from src.api.schemas import (
        DomainCreateRequest,
        DomainDeleteResponse,
        DomainItemResponse,
        DomainUpdateRequest,
        SubdomainCreateRequest,
        SubdomainItemResponse,
        SubdomainUpdateRequest,
    )


def _invalidate_taxonomy_cache() -> None:
    """Invalidate in-memory taxonomy cache across runtime components if present."""
    try:
        from src.ingestion.classifier import invalidate_taxonomy_catalog

        invalidate_taxonomy_catalog()
    except (ImportError, AttributeError):
        pass


async def _require_domain(session: AsyncSession, slug: str, label: str = "Domain") -> DomainRegistry:
    """Fetch a domain by slug or raise HTTP 404."""
    domain = await get_domain_by_slug(session, slug)
    if domain is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{label} '{slug}' not found",
        )
    return domain


def build_domain_item_response(
    domain: DomainRegistry,
    *,
    document_count: int = 0,
    subdomains: Sequence[SubdomainRegistry] | None = None,
) -> DomainItemResponse:
    """Serialize a DomainRegistry ORM object with nested SubdomainItemResponse items."""
    from src.api.schemas import DomainItemResponse, SubdomainItemResponse

    sub_items = [SubdomainItemResponse.model_validate(s) for s in (subdomains or ())]
    return DomainItemResponse(
        id=domain.id,
        slug=domain.slug,
        name=domain.name,
        epistemic_type=domain.epistemic_type,
        authority_priority=domain.authority_priority,
        description=domain.description,
        icon=domain.icon,
        color_token=domain.color_token,
        chunking_strategy=domain.chunking_strategy,
        classifier_keywords=domain.classifier_keywords,
        is_system=domain.is_system,
        is_active=domain.is_active,
        document_count=document_count,
        subdomains=sub_items,
    )


async def list_domains_with_counts(
    session: AsyncSession,
    *,
    include_inactive: bool = False,
) -> list[DomainItemResponse]:
    """List domains with batched subdomains and per-domain document counts (no N+1 queries)."""
    stmt = select(DomainRegistry)
    if not include_inactive:
        stmt = stmt.where(DomainRegistry.is_active == True)  # noqa: E712
    stmt = stmt.order_by(DomainRegistry.authority_priority, DomainRegistry.slug)
    domains = list((await session.execute(stmt)).scalars().all())
    if not domains:
        return []

    count_stmt = select(Document.primary_domain_id, func.count(Document.id)).group_by(
        Document.primary_domain_id
    )
    counts = {
        row[0]: int(row[1])
        for row in (await session.execute(count_stmt)).all()
        if row[0]
    }
    grouped_subs = await list_subdomains_by_domain_ids(session, [d.id for d in domains])

    return [
        build_domain_item_response(
            d,
            document_count=counts.get(d.id, 0),
            subdomains=grouped_subs.get(d.id, []),
        )
        for d in domains
    ]


async def create_custom_domain(
    session: AsyncSession,
    payload: DomainCreateRequest,
) -> DomainItemResponse:
    """Create a custom domain, enforcing the 10-active-domain cap and slug uniqueness."""
    active_count = (
        await session.execute(
            select(func.count(DomainRegistry.id)).where(DomainRegistry.is_active == True)  # noqa: E712
        )
    ).scalar() or 0
    if active_count >= 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum limit of 10 active top-level domains reached",
        )

    existing = await get_domain_by_slug(session, payload.slug)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Domain with slug '{payload.slug}' already exists",
        )

    domain = await create_domain(
        session=session,
        slug=payload.slug,
        name=payload.name,
        epistemic_type=payload.epistemic_type,
        authority_priority=payload.authority_priority,
        description=payload.description,
        icon=payload.icon,
        color_token=payload.color_token,
        chunking_strategy=payload.chunking_strategy,
        classifier_keywords=payload.classifier_keywords,
        is_system=False,
        is_active=payload.is_active,
        auto_create_default_subdomain=True,
    )
    _invalidate_taxonomy_cache()

    subs = await list_subdomains(session, domain_id=domain.id)
    return build_domain_item_response(domain, document_count=0, subdomains=subs)


async def update_domain_and_schedule_reindex(
    *,
    slug: str,
    payload: DomainUpdateRequest,
    app_state: Any,
    session: AsyncSession,
    qdrant: AsyncQdrantClient,
    orchestrator: IngestionOrchestrator,
    settings: Settings,
) -> DomainItemResponse:
    """Update domain attributes, enforce system-domain locks, and queue optional re-indexing."""
    domain = await _require_domain(session, slug)

    if domain.is_system:
        locked_fields = (
            ("name", payload.name, domain.name),
            ("epistemic_type", payload.epistemic_type, domain.epistemic_type),
            ("authority_priority", payload.authority_priority, domain.authority_priority),
            ("chunking_strategy", payload.chunking_strategy, domain.chunking_strategy),
        )
        for field_name, incoming, current in locked_fields:
            if incoming is not None and incoming != current:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot modify {field_name} of system protected domain",
                )

    updated = await update_domain(
        session=session,
        qdrant=qdrant,
        collection_name=settings.qdrant_collection,
        slug=slug,
        name=payload.name,
        epistemic_type=payload.epistemic_type,
        authority_priority=payload.authority_priority,
        description=payload.description,
        icon=payload.icon,
        color_token=payload.color_token,
        chunking_strategy=payload.chunking_strategy,
        classifier_keywords=payload.classifier_keywords,
        is_active=payload.is_active,
    )

    if payload.re_index:
        await queue_documents_for_reindex(
            app_state=app_state,
            session=session,
            orchestrator=orchestrator,
            settings=settings,
            domain_id=domain.id,
            allow_raw_fallback=False,
        )

    _invalidate_taxonomy_cache()

    subs = await list_subdomains(session, domain_id=updated.id)
    doc_count = (
        await session.execute(
            select(func.count(Document.id)).where(Document.primary_domain_id == updated.id)
        )
    ).scalar() or 0
    return build_domain_item_response(updated, document_count=int(doc_count), subdomains=subs)


async def delete_domain_and_schedule_reindex(
    *,
    slug: str,
    target_domain: str,
    target_subdomain: str | None,
    re_index: bool,
    app_state: Any,
    session: AsyncSession,
    qdrant: AsyncQdrantClient,
    orchestrator: IngestionOrchestrator,
    settings: Settings,
) -> DomainDeleteResponse:
    """Delete a custom domain, reassign records atomically, and optionally queue re-indexing."""
    domain = await _require_domain(session, slug)
    if domain.is_system:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete system protected domain '{slug}'",
        )
    if target_domain == slug:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Target domain must be different from domain being deleted",
        )

    target_dom_obj = await _require_domain(session, target_domain, label="Target domain")

    in_flight_res = await session.execute(
        select(Document).where(
            Document.primary_domain_id == domain.id,
            Document.status.in_(["queued", "parsing", "classifying", "chunking", "indexing"]),
        )
    )
    if in_flight_res.scalars().first() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot delete domain '{slug}': active documents are currently in-flight",
        )

    eligible_doc_ids = (
        await find_reindexable_doc_ids(session, domain_id=domain.id, data_dir=settings.data_dir)
        if re_index
        else []
    )

    try:
        reassign_res = await delete_domain_and_reassign(
            session=session,
            qdrant=qdrant,
            collection_name=settings.qdrant_collection,
            slug=slug,
            target_domain_slug=target_domain,
            target_subdomain_slug=target_subdomain,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    queued_count = 0
    if re_index and eligible_doc_ids:
        queued_count, _ = await queue_documents_for_reindex(
            app_state=app_state,
            session=session,
            orchestrator=orchestrator,
            settings=settings,
            doc_ids=eligible_doc_ids,
            allow_raw_fallback=False,
        )

    _invalidate_taxonomy_cache()
    from src.api.schemas import DomainDeleteResponse

    return DomainDeleteResponse(
        deleted_slug=slug,
        target_domain=target_dom_obj.slug,
        target_subdomain=reassign_res["target_subdomain"],
        reassigned_documents=reassign_res["reassigned_documents"],
        reassigned_chunks=reassign_res["reassigned_chunks"],
        re_index_queued=queued_count if re_index else 0,
    )


async def create_domain_subdomain(
    session: AsyncSession,
    slug: str,
    payload: SubdomainCreateRequest,
) -> SubdomainItemResponse:
    """Create a new subdomain under `slug`, normalizing prefix and checking uniqueness."""
    domain = await _require_domain(session, slug)

    sub_slug = payload.slug.strip()
    prefix = f"{domain.slug}."
    if not sub_slug.startswith(prefix):
        sub_slug = f"{prefix}{sub_slug}"

    existing = await session.execute(
        select(SubdomainRegistry).where(SubdomainRegistry.slug == sub_slug)
    )
    if existing.scalars().first() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Subdomain with slug '{sub_slug}' already exists",
        )

    sub = await create_subdomain(
        session=session,
        slug=sub_slug,
        name=payload.name,
        domain_id=domain.id,
        standard_bodies=payload.standard_bodies,
    )
    _invalidate_taxonomy_cache()
    from src.api.schemas import SubdomainItemResponse

    return SubdomainItemResponse.model_validate(sub)


async def update_domain_subdomain(
    session: AsyncSession,
    slug: str,
    sub_slug: str,
    payload: SubdomainUpdateRequest,
) -> SubdomainItemResponse:
    """Update subdomain metadata and invalidate the taxonomy cache."""
    await _require_domain(session, slug)

    try:
        sub = await update_subdomain(
            session=session,
            domain_slug=slug,
            subdomain_slug=sub_slug,
            name=payload.name,
            standard_bodies=payload.standard_bodies,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    _invalidate_taxonomy_cache()
    from src.api.schemas import SubdomainItemResponse

    return SubdomainItemResponse.model_validate(sub)


async def delete_domain_subdomain(
    *,
    session: AsyncSession,
    qdrant: AsyncQdrantClient,
    settings: Settings,
    slug: str,
    sub_slug: str,
    target_subdomain: str | None,
) -> dict[str, Any]:
    """Delete a subdomain while enforcing root/last-subdomain guards via repository validation."""
    if sub_slug == "general.general":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete root general.general subdomain",
        )

    domain = await _require_domain(session, slug)

    subs = await list_subdomains(session, domain_id=domain.id)
    if not any(s.slug == sub_slug for s in subs):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Subdomain '{sub_slug}' not found in domain '{slug}'",
        )

    try:
        reassigned_to = await delete_subdomain(
            session=session,
            qdrant=qdrant,
            collection_name=settings.qdrant_collection,
            domain_slug=slug,
            subdomain_slug=sub_slug,
            target_subdomain_slug=target_subdomain,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    _invalidate_taxonomy_cache()
    return {
        "status": "deleted",
        "deleted_slug": sub_slug,
        "target_subdomain": reassigned_to,
    }


__all__ = [
    "_invalidate_taxonomy_cache",
    "build_domain_item_response",
    "create_custom_domain",
    "create_domain_subdomain",
    "create_lexicon_entry",
    "delete_domain_and_schedule_reindex",
    "delete_domain_subdomain",
    "delete_lexicon_entry",
    "list_domains_with_counts",
    "list_lexicon_entries",
    "update_domain_and_schedule_reindex",
    "update_domain_subdomain",
]

