"""FastAPI endpoints for Domain Taxonomy, Subdomain, and Lexicon management."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request, status
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import (
    get_app_settings,
    get_db_session,
    get_orchestrator,
    get_qdrant,
)
from src.api.routes.settings import verify_key_endpoint  # noqa: F401
from src.api.schemas import (
    DomainCreateRequest,
    DomainDeleteResponse,
    DomainItemResponse,
    DomainUpdateRequest,
    LexiconCreateRequest,
    LexiconItemResponse,
    SubdomainCreateRequest,
    SubdomainItemResponse,
    SubdomainUpdateRequest,
)
from src.config import Settings
from src.ingestion.worker import IngestionOrchestrator
from src.services.taxonomy_service import (
    create_custom_domain,
    create_domain_subdomain,
    create_lexicon_entry,
    delete_domain_and_schedule_reindex,
    delete_domain_subdomain,
    delete_lexicon_entry,
    list_domains_with_counts,
    list_lexicon_entries,
    update_domain_and_schedule_reindex,
    update_domain_subdomain,
)

router = APIRouter(tags=["domains"])


@router.get("/domains", response_model=list[DomainItemResponse])
async def list_domains_endpoint(
    include_inactive: bool = Query(default=False, description="Include deactivated domains"),
    session: AsyncSession = Depends(get_db_session),
) -> list[DomainItemResponse]:
    """List active domains with nested subdomains and per-domain document counts."""
    return await list_domains_with_counts(session, include_inactive=include_inactive)


@router.post(
    "/domains",
    response_model=DomainItemResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_domain_endpoint(
    payload: DomainCreateRequest,
    session: AsyncSession = Depends(get_db_session),
) -> DomainItemResponse:
    """Create a new custom domain, enforcing a maximum limit of 10 active domains."""
    return await create_custom_domain(session, payload)


@router.patch("/domains/{slug}", response_model=DomainItemResponse)
async def update_domain_endpoint(
    slug: str,
    payload: DomainUpdateRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    orchestrator: IngestionOrchestrator = Depends(get_orchestrator),
    settings: Settings = Depends(get_app_settings),
) -> DomainItemResponse:
    """Update domain attributes, locking system starter domain invariants and scheduling optional re-indexing."""
    return await update_domain_and_schedule_reindex(
        slug=slug,
        payload=payload,
        app_state=request.app.state,
        session=session,
        qdrant=qdrant,
        orchestrator=orchestrator,
        settings=settings,
    )


@router.delete("/domains/{slug}", response_model=DomainDeleteResponse)
async def delete_domain_endpoint(
    slug: str,
    request: Request,
    target_domain: str = Query(default="general", description="Target domain slug to reassign records to"),
    target_subdomain: str | None = Query(default=None, description="Optional target subdomain slug"),
    re_index: bool = Query(default=False, description="Queue background re-indexing for ready documents"),
    session: AsyncSession = Depends(get_db_session),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    orchestrator: IngestionOrchestrator = Depends(get_orchestrator),
    settings: Settings = Depends(get_app_settings),
) -> DomainDeleteResponse:
    """Delete a custom domain and atomically reassign documents and vectors, checking in-flight guards."""
    return await delete_domain_and_schedule_reindex(
        slug=slug,
        target_domain=target_domain,
        target_subdomain=target_subdomain,
        re_index=re_index,
        app_state=request.app.state,
        session=session,
        qdrant=qdrant,
        orchestrator=orchestrator,
        settings=settings,
    )


@router.post(
    "/domains/{slug}/subdomains",
    response_model=SubdomainItemResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_subdomain_endpoint(
    slug: str,
    payload: SubdomainCreateRequest,
    session: AsyncSession = Depends(get_db_session),
) -> SubdomainItemResponse:
    """Create a new subdomain, auto-prefixing with domain slug if necessary."""
    return await create_domain_subdomain(session, slug, payload)


@router.patch(
    "/domains/{slug}/subdomains/{sub_slug}",
    response_model=SubdomainItemResponse,
)
async def update_subdomain_endpoint(
    slug: str,
    sub_slug: str,
    payload: SubdomainUpdateRequest,
    session: AsyncSession = Depends(get_db_session),
) -> SubdomainItemResponse:
    """Update subdomain name or standard bodies."""
    return await update_domain_subdomain(session, slug, sub_slug, payload)


@router.delete("/domains/{slug}/subdomains/{sub_slug}")
async def delete_subdomain_endpoint(
    slug: str,
    sub_slug: str,
    target_subdomain: str | None = Query(default=None, description="Subdomain to reassign records to"),
    session: AsyncSession = Depends(get_db_session),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    settings: Settings = Depends(get_app_settings),
) -> dict[str, Any]:
    """Delete a subdomain, enforcing root general.general and last-subdomain guards."""
    return await delete_domain_subdomain(
        session=session,
        qdrant=qdrant,
        settings=settings,
        slug=slug,
        sub_slug=sub_slug,
        target_subdomain=target_subdomain,
    )


@router.get("/lexicon", response_model=list[LexiconItemResponse])
async def list_lexicon_endpoint(
    domain_id: str | None = Query(default=None, description="Filter by domain ID or slug"),
    search: str | None = Query(default=None, description="Search term in term or definitions"),
    session: AsyncSession = Depends(get_db_session),
) -> list[LexiconItemResponse]:
    """List polysemy lexicon definitions with optional domain filter and search query."""
    return await list_lexicon_entries(session, domain_id=domain_id, search=search)


@router.post(
    "/lexicon",
    response_model=LexiconItemResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_lexicon_endpoint(
    payload: LexiconCreateRequest,
    session: AsyncSession = Depends(get_db_session),
) -> LexiconItemResponse:
    """Add or update a polysemic term definition in the domain lexicon, enforcing workspace and domain caps."""
    return await create_lexicon_entry(session, payload)


@router.delete("/lexicon/{entry_id}")
async def delete_lexicon_endpoint(
    entry_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Delete a domain lexicon entry by ID."""
    return await delete_lexicon_entry(session, entry_id)
