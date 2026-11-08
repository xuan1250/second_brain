"""FastAPI endpoints for Domain Taxonomy, Lexicon, and Workspace Settings management."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from langchain_core.messages import HumanMessage
from qdrant_client import AsyncQdrantClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import (
    get_app_settings,
    get_db_session,
    get_orchestrator,
    get_qdrant,
)
from src.api.routes.system import _build_provider_options
from src.api.schemas import (
    DomainCreateRequest,
    DomainDeleteResponse,
    DomainItemResponse,
    DomainUpdateRequest,
    LexiconCreateRequest,
    LexiconItemResponse,
    LexiconUpdateRequest,
    ProviderOption,
    SubdomainCreateRequest,
    SubdomainItemResponse,
    SubdomainUpdateRequest,
    VerifyKeyRequest,
    VerifyKeyResponse,
    WorkspaceSettingsResponse,
    WorkspaceSettingsUpdateRequest,
    _validate_safe_base_url,
)
from src.config import Settings
from src.rag.llm import (
    _key_validations,
    compute_key_hash,
    get_chat_model,
    is_valid_api_key,
    mask_api_key,
    normalize_api_key,
    resolve_provider_credential,
    sanitize_llm_error,
)
from src.db.models import (
    Document,
    DomainLexicon,
    DomainRegistry,
    IngestionJob,
    SubdomainRegistry,
    WorkspaceSettings,
    _now,
)
from src.db.repository import (
    add_lexicon_term,
    create_domain,
    create_subdomain,
    delete_domain_and_reassign,
    delete_lexicon_term,
    delete_subdomain,
    get_domain_by_slug,
    get_workspace_settings,
    list_domains,
    list_lexicon_terms,
    list_subdomains,
    update_domain,
    update_subdomain,
    update_workspace_settings,
)
from src.ingestion.parser import load_parsed_checkpoint
from src.ingestion.worker import IngestionOrchestrator

router = APIRouter(tags=["domains", "settings"])


def _invalidate_taxonomy_cache() -> None:
    """Invalidate in-memory taxonomy cache across runtime components if present."""
    try:
        from src.ingestion.classifier import invalidate_taxonomy_catalog

        invalidate_taxonomy_catalog()
    except (ImportError, AttributeError):
        pass


# ============================================================================
# Domains CRUD
# ============================================================================


@router.get("/domains", response_model=list[DomainItemResponse])
async def list_domains_endpoint(
    include_inactive: bool = Query(default=False, description="Include deactivated domains"),
    session: AsyncSession = Depends(get_db_session),
) -> list[DomainItemResponse]:
    """List active domains with nested subdomains and per-domain document counts."""
    stmt = select(DomainRegistry)
    if not include_inactive:
        stmt = stmt.where(DomainRegistry.is_active == True)  # noqa: E712
    stmt = stmt.order_by(DomainRegistry.authority_priority, DomainRegistry.slug)
    res = await session.execute(stmt)
    domains = list(res.scalars().all())

    # Count documents per primary domain
    count_stmt = select(Document.primary_domain_id, func.count(Document.id)).group_by(
        Document.primary_domain_id
    )
    count_res = await session.execute(count_stmt)
    counts = {row[0]: row[1] for row in count_res.all() if row[0]}

    # Batch or per-domain subdomains
    results: list[DomainItemResponse] = []
    for d in domains:
        subs = await list_subdomains(session, domain_id=d.id)
        sub_items = [SubdomainItemResponse.model_validate(s) for s in subs]
        results.append(
            DomainItemResponse(
                id=d.id,
                slug=d.slug,
                name=d.name,
                epistemic_type=d.epistemic_type,
                authority_priority=d.authority_priority,
                description=d.description,
                icon=d.icon,
                color_token=d.color_token,
                chunking_strategy=d.chunking_strategy,
                classifier_keywords=d.classifier_keywords,
                is_system=d.is_system,
                is_active=d.is_active,
                document_count=counts.get(d.id, 0),
                subdomains=sub_items,
            )
        )
    return results


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
    # Check max 10 active top-level domains
    active_count_res = await session.execute(
        select(func.count(DomainRegistry.id)).where(DomainRegistry.is_active == True)  # noqa: E712
    )
    active_count = active_count_res.scalar() or 0
    if active_count >= 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum limit of 10 active top-level domains reached",
        )

    # Check slug uniqueness
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
        document_count=0,
        subdomains=[SubdomainItemResponse.model_validate(s) for s in subs],
    )


@router.patch("/domains/{slug}", response_model=DomainItemResponse)
async def update_domain_endpoint(
    slug: str,
    payload: DomainUpdateRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    orchestrator: IngestionOrchestrator = Depends(get_orchestrator),
    settings: Settings = Depends(get_app_settings),
) -> DomainItemResponse:
    """Update domain attributes, locking system starter domain invariants and scheduling optional re-indexing."""
    domain = await get_domain_by_slug(session, slug)
    if domain is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain '{slug}' not found",
        )

    # System domain field locking
    if domain.is_system:
        if payload.name is not None and payload.name != domain.name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot modify name of system protected domain",
            )
        if payload.epistemic_type is not None and payload.epistemic_type != domain.epistemic_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot modify epistemic_type of system protected domain",
            )
        if payload.authority_priority is not None and payload.authority_priority != domain.authority_priority:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot modify authority_priority of system protected domain",
            )
        if payload.chunking_strategy is not None and payload.chunking_strategy != domain.chunking_strategy:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot modify chunking_strategy of system protected domain",
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

    # Optional background re-indexing
    if payload.re_index:
        doc_stmt = select(Document).where(
            Document.primary_domain_id == domain.id,
            Document.status == "ready",
        )
        ready_docs = (await session.execute(doc_stmt)).scalars().all()
        for d in ready_docs:
            cached = load_parsed_checkpoint(
                data_dir=settings.data_dir,
                sha=d.sha256,
                filename=d.filename,
                file_type=d.file_type.lstrip("."),
            )
            if cached is not None:
                d.status = "queued"
                d.updated_at = _now()
                session.add(d)

                new_job = IngestionJob(
                    doc_id=d.id,
                    status="queued",
                    stage="queued",
                    progress=0.45,
                    stage_timings=json.dumps({}),
                    created_at=_now(),
                    updated_at=_now(),
                )
                session.add(new_job)
                await session.commit()
                await session.refresh(new_job)

                background_tasks.add_task(
                    orchestrator.complete_approved_ingestion,
                    doc_id=d.id,
                    job_id=new_job.id,
                )

    _invalidate_taxonomy_cache()

    subs = await list_subdomains(session, domain_id=updated.id)
    doc_count_res = await session.execute(
        select(func.count(Document.id)).where(Document.primary_domain_id == updated.id)
    )
    doc_count = doc_count_res.scalar() or 0

    return DomainItemResponse(
        id=updated.id,
        slug=updated.slug,
        name=updated.name,
        epistemic_type=updated.epistemic_type,
        authority_priority=updated.authority_priority,
        description=updated.description,
        icon=updated.icon,
        color_token=updated.color_token,
        chunking_strategy=updated.chunking_strategy,
        classifier_keywords=updated.classifier_keywords,
        is_system=updated.is_system,
        is_active=updated.is_active,
        document_count=doc_count,
        subdomains=[SubdomainItemResponse.model_validate(s) for s in subs],
    )


@router.delete("/domains/{slug}", response_model=DomainDeleteResponse)
async def delete_domain_endpoint(
    slug: str,
    background_tasks: BackgroundTasks,
    target_domain: str = Query(default="general", description="Target domain slug to reassign records to"),
    target_subdomain: str | None = Query(default=None, description="Optional target subdomain slug"),
    re_index: bool = Query(default=False, description="Queue background re-indexing for ready documents"),
    session: AsyncSession = Depends(get_db_session),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    orchestrator: IngestionOrchestrator = Depends(get_orchestrator),
    settings: Settings = Depends(get_app_settings),
) -> DomainDeleteResponse:
    """Delete a custom domain and atomically reassign documents and vectors, checking in-flight guards."""
    domain = await get_domain_by_slug(session, slug)
    if domain is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain '{slug}' not found",
        )
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

    target_dom_obj = await get_domain_by_slug(session, target_domain)
    if target_dom_obj is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target domain '{target_domain}' not found",
        )

    # In-flight documents check
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

    # Collect eligible docs for re-indexing before reassignment
    eligible_docs_to_reindex: list[str] = []
    if re_index:
        ready_docs_res = await session.execute(
            select(Document).where(
                Document.primary_domain_id == domain.id,
                Document.status == "ready",
            )
        )
        for d in ready_docs_res.scalars().all():
            cached = load_parsed_checkpoint(
                data_dir=settings.data_dir,
                sha=d.sha256,
                filename=d.filename,
                file_type=d.file_type.lstrip("."),
            )
            if cached is not None:
                eligible_docs_to_reindex.append(d.id)

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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    # Schedule re-indexing for eligible documents
    if re_index and eligible_docs_to_reindex:
        for doc_id in eligible_docs_to_reindex:
            doc = await session.get(Document, doc_id)
            if doc and doc.status == "ready":
                doc.status = "queued"
                doc.updated_at = _now()
                session.add(doc)

                new_job = IngestionJob(
                    doc_id=doc.id,
                    status="queued",
                    stage="queued",
                    progress=0.45,
                    stage_timings=json.dumps({}),
                    created_at=_now(),
                    updated_at=_now(),
                )
                session.add(new_job)
                await session.commit()
                await session.refresh(new_job)

                background_tasks.add_task(
                    orchestrator.complete_approved_ingestion,
                    doc_id=doc.id,
                    job_id=new_job.id,
                )

    _invalidate_taxonomy_cache()

    return DomainDeleteResponse(
        deleted_slug=slug,
        target_domain=target_dom_obj.slug,
        target_subdomain=reassign_res["target_subdomain"],
        reassigned_documents=reassign_res["reassigned_documents"],
        reassigned_chunks=reassign_res["reassigned_chunks"],
        re_index_queued=len(eligible_docs_to_reindex) if re_index else 0,
    )


# ============================================================================
# Subdomains CRUD
# ============================================================================


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
    domain = await get_domain_by_slug(session, slug)
    if domain is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain '{slug}' not found",
        )

    # Normalize subdomain slug with domain prefix
    sub_slug = payload.slug.strip()
    prefix = f"{domain.slug}."
    if not sub_slug.startswith(prefix):
        sub_slug = f"{prefix}{sub_slug}"

    # Check global uniqueness of subdomain slug
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
    return SubdomainItemResponse.model_validate(sub)


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
    domain = await get_domain_by_slug(session, slug)
    if domain is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain '{slug}' not found",
        )

    try:
        sub = await update_subdomain(
            session=session,
            domain_slug=slug,
            subdomain_slug=sub_slug,
            name=payload.name,
            standard_bodies=payload.standard_bodies,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    _invalidate_taxonomy_cache()
    return SubdomainItemResponse.model_validate(sub)


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
    if sub_slug == "general.general":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete root general.general subdomain",
        )

    domain = await get_domain_by_slug(session, slug)
    if domain is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain '{slug}' not found",
        )

    subs = await list_subdomains(session, domain_id=domain.id)
    if len(subs) <= 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete the only remaining subdomain for domain '{slug}'",
        )

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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    _invalidate_taxonomy_cache()
    return {
        "status": "deleted",
        "deleted_slug": sub_slug,
        "target_subdomain": reassigned_to,
    }


# ============================================================================
# Lexicon CRUD
# ============================================================================


@router.get("/lexicon", response_model=list[LexiconItemResponse])
async def list_lexicon_endpoint(
    domain_id: str | None = Query(default=None, description="Filter by domain ID or slug"),
    search: str | None = Query(default=None, description="Search term in term or definitions"),
    session: AsyncSession = Depends(get_db_session),
) -> list[LexiconItemResponse]:
    """List polysemy lexicon definitions with optional domain filter and search query."""
    resolved_domain_id = domain_id
    if domain_id:
        d = await get_domain_by_slug(session, domain_id)
        if d:
            resolved_domain_id = d.id

    entries = await list_lexicon_terms(session, domain_id=resolved_domain_id, search=search)
    return [LexiconItemResponse.model_validate(e) for e in entries]


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
    domain = await session.get(DomainRegistry, payload.domain_id)
    if domain is None:
        domain = await get_domain_by_slug(session, payload.domain_id)
    if domain is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain '{payload.domain_id}' not found",
        )

    norm_term = payload.term.strip()

    # Check if this term already exists under this domain
    existing_stmt = select(DomainLexicon).where(
        DomainLexicon.term == norm_term,
        DomainLexicon.domain_id == domain.id,
    )
    existing_res = await session.execute(existing_stmt)
    is_new = existing_res.scalars().first() is None

    if is_new:
        # Enforce workspace cap: 500 total lexicon entries
        total_count_res = await session.execute(select(func.count(DomainLexicon.id)))
        total_count = total_count_res.scalar() or 0
        if total_count >= 500:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum limit of 500 lexicon entries across workspace reached",
            )

        # Enforce per-domain cap: 100 entries per domain
        domain_count_res = await session.execute(
            select(func.count(DomainLexicon.id)).where(DomainLexicon.domain_id == domain.id)
        )
        domain_count = domain_count_res.scalar() or 0
        if domain_count >= 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Maximum limit of 100 lexicon entries for domain '{domain.slug}' reached",
            )

    entry = await add_lexicon_term(
        session=session,
        term=norm_term,
        domain_id=domain.id,
        specific_sense=payload.specific_sense,
        contrastive_warning=payload.contrastive_warning,
    )
    return LexiconItemResponse.model_validate(entry)


@router.delete("/lexicon/{entry_id}")
async def delete_lexicon_endpoint(
    entry_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Delete a domain lexicon entry by ID."""
    success = await delete_lexicon_term(session, lexicon_id=entry_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lexicon entry '{entry_id}' not found",
        )
    return {"deleted": True, "id": entry_id}


# ============================================================================
# Workspace Settings Endpoints
# ============================================================================


@router.get("/settings", response_model=WorkspaceSettingsResponse)
async def get_settings_endpoint(
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
) -> WorkspaceSettingsResponse:
    """Get the current workspace settings merged with available LLM provider options."""
    ws = await get_workspace_settings(session)
    _, available_providers = _build_provider_options(settings)
    return WorkspaceSettingsResponse(
        id=ws.id,
        default_upload_mode=ws.default_upload_mode,
        default_approval_mode=ws.default_approval_mode,
        high_confidence_threshold=ws.high_confidence_threshold,
        default_llm_provider=ws.default_llm_provider,
        default_llm_model=ws.default_llm_model,
        llm_compat_base_url=ws.llm_compat_base_url,
        available_providers=available_providers,
        updated_at=ws.updated_at,
    )


@router.patch("/settings", response_model=WorkspaceSettingsResponse)
async def update_settings_endpoint(
    payload: WorkspaceSettingsUpdateRequest,
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
) -> WorkspaceSettingsResponse:
    """Update workspace settings and sync runtime configuration."""
    ws = await update_workspace_settings(
        session=session,
        default_upload_mode=payload.default_upload_mode,
        default_approval_mode=payload.default_approval_mode,
        high_confidence_threshold=payload.high_confidence_threshold,
        default_llm_provider=payload.default_llm_provider,
        default_llm_model=payload.default_llm_model,
        api_keys=payload.api_keys,
        llm_compat_base_url=payload.llm_compat_base_url,
    )

    # Sync runtime settings
    if payload.high_confidence_threshold is not None:
        settings.ingestion.high_confidence_threshold = payload.high_confidence_threshold
    if payload.default_llm_provider is not None:
        settings.llm.provider = payload.default_llm_provider
    if payload.default_llm_model is not None:
        settings.llm.model_name = payload.default_llm_model

    _, available_providers = _build_provider_options(settings)
    return WorkspaceSettingsResponse(
        id=ws.id,
        default_upload_mode=ws.default_upload_mode,
        default_approval_mode=ws.default_approval_mode,
        high_confidence_threshold=ws.high_confidence_threshold,
        default_llm_provider=ws.default_llm_provider,
        default_llm_model=ws.default_llm_model,
        llm_compat_base_url=ws.llm_compat_base_url,
        available_providers=available_providers,
        updated_at=ws.updated_at,
    )


@router.post("/settings/verify-key", response_model=VerifyKeyResponse)
async def verify_key_endpoint(
    payload: VerifyKeyRequest,
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
) -> VerifyKeyResponse:
    """Verify an API key live via bounded LLM ping and optionally persist to WorkspaceSettings."""
    # 1. Ignore payload.base_url unless provider == "openai_compatible"
    raw_base_url = payload.base_url if payload.provider == "openai_compatible" else None

    # 2. Resolve current credentials
    resolved_key, resolved_source, resolved_base_url = resolve_provider_credential(payload.provider, settings)

    # 3. For openai_compatible, resolve candidate_base_url & validate safety
    candidate_base_url: str | None = None
    if payload.provider == "openai_compatible":
        candidate_base_url = (
            raw_base_url.strip() if raw_base_url and raw_base_url.strip() else resolved_base_url
        )
        if candidate_base_url:
            try:
                candidate_base_url = _validate_safe_base_url(candidate_base_url)
            except ValueError as ve:
                _, all_opts = _build_provider_options(settings)
                opt = next((o for o in all_opts if o.provider == payload.provider), None)
                return VerifyKeyResponse(
                    provider=payload.provider,
                    valid=False,
                    validation_status="invalid",
                    validation_error=str(ve),
                    key_source=resolved_source,
                    masked_key=mask_api_key(resolved_key),
                    tested_model="",
                    provider_option=opt or ProviderOption(
                        provider=payload.provider,
                        label=payload.provider,
                        configured=False,
                        validation_status="invalid",
                        validation_error=str(ve),
                    ),
                )

        # Anti-exfiltration check:
        # If payload.base_url is a new unsaved URL (candidate_base_url != resolved_base_url)
        # and not (payload.api_key and payload.api_key.strip())
        if candidate_base_url != resolved_base_url and not (payload.api_key and payload.api_key.strip()):
            _, all_opts = _build_provider_options(settings)
            opt = next((o for o in all_opts if o.provider == payload.provider), None)
            err_msg = "Please enter the API key when verifying a new Base URL."
            return VerifyKeyResponse(
                provider=payload.provider,
                valid=False,
                validation_status="invalid",
                validation_error=err_msg,
                key_source=resolved_source,
                masked_key=mask_api_key(resolved_key),
                tested_model="",
                provider_option=opt or ProviderOption(
                    provider=payload.provider,
                    label=payload.provider,
                    configured=False,
                    validation_status="invalid",
                    validation_error=err_msg,
                ),
            )

    # 4. Determine candidate_key
    has_explicit_key = bool(payload.api_key and payload.api_key.strip())
    candidate_key = normalize_api_key(payload.api_key) if has_explicit_key else resolved_key
    candidate_source: Literal["ui", "env", "none"] = "ui" if has_explicit_key else resolved_source

    # Check missing key
    if not candidate_key:
        _, all_opts = _build_provider_options(settings)
        opt = next((o for o in all_opts if o.provider == payload.provider), None)
        err_msg = "No API key configured in Workspace Settings or backend/.env."
        return VerifyKeyResponse(
            provider=payload.provider,
            valid=False,
            validation_status="missing",
            validation_error=err_msg,
            key_source="none",
            masked_key=None,
            tested_model="",
            provider_option=opt or ProviderOption(
                provider=payload.provider,
                label=payload.provider,
                configured=False,
                validation_status="missing",
                validation_error=err_msg,
            ),
        )

    # Check format validity
    if not is_valid_api_key(candidate_key):
        _, all_opts = _build_provider_options(settings)
        opt = next((o for o in all_opts if o.provider == payload.provider), None)
        err_msg = "Invalid API key format (placeholders and whitespace/control characters are not allowed)."
        return VerifyKeyResponse(
            provider=payload.provider,
            valid=False,
            validation_status="invalid",
            validation_error=err_msg,
            key_source=candidate_source,
            masked_key=mask_api_key(candidate_key),
            tested_model="",
            provider_option=opt or ProviderOption(
                provider=payload.provider,
                label=payload.provider,
                configured=False,
                validation_status="invalid",
                validation_error=err_msg,
            ),
        )

    # 5. Resolve model for ping
    test_model = (payload.model.strip() if payload.model and payload.model.strip() else "")
    if not test_model:
        prov_entry = settings.llm.providers.get(payload.provider)
        test_model = (prov_entry.model_name.strip() if prov_entry and prov_entry.model_name else "")
    if not test_model:
        fallback_models = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-5-20250929",
            "deepseek": "deepseek-chat",
            "openai_compatible": "glm-4-plus",
        }
        test_model = fallback_models.get(payload.provider, "gpt-4o")

    # 6. Live ping (5.0s bounded)
    k_hash = compute_key_hash(
        candidate_key,
        candidate_base_url if payload.provider == "openai_compatible" else None,
    )

    llm_kwargs: dict[str, Any] = {
        "provider_override": payload.provider,
        "model_override": test_model,
        "settings": settings,
        "api_key": candidate_key,
        "max_tokens": 1,
        "timeout": 5.0,
        "max_retries": 0,
    }
    if payload.provider == "openai_compatible" and candidate_base_url:
        llm_kwargs["base_url"] = candidate_base_url

    valid = False
    validation_status: Literal["valid", "invalid"] = "invalid"
    validation_error: str | None = None

    try:
        llm = get_chat_model(**llm_kwargs)
        await asyncio.wait_for(llm.ainvoke([HumanMessage(content="ping")]), timeout=5.0)
        valid = True
        validation_status = "valid"
        validation_error = None
    except Exception as exc:
        clean_err, _ = sanitize_llm_error(exc, provider=payload.provider, candidate_key=candidate_key)
        valid = False
        validation_status = "invalid"
        validation_error = clean_err

    # Update in-memory / persistent state
    now_iso = datetime.now(timezone.utc).isoformat()
    val_entry = {
        "status": validation_status,
        "error": validation_error,
        "key_hash": k_hash,
        "checked_at": now_iso,
    }

    if payload.save_if_valid:
        api_keys_update = (
            {payload.provider: candidate_key}
            if has_explicit_key
            else None
        )
        compat_url_update = (
            candidate_base_url
            if (payload.provider == "openai_compatible" and raw_base_url is not None)
            else None
        )
        await update_workspace_settings(
            session=session,
            api_keys=api_keys_update,
            key_validations={payload.provider: val_entry},
            llm_compat_base_url=compat_url_update,
        )
    else:
        _key_validations[payload.provider] = val_entry

    _, all_opts = _build_provider_options(settings)
    matching_option = next(
        (opt for opt in all_opts if opt.provider == payload.provider),
        ProviderOption(
            provider=payload.provider,
            label=payload.provider,
            configured=bool(candidate_key),
            default_model=test_model,
            key_source=candidate_source,
            masked_key=mask_api_key(candidate_key),
            validation_status=validation_status,
            validation_error=validation_error,
            base_url=candidate_base_url if payload.provider == "openai_compatible" else None,
        ),
    )

    if not payload.save_if_valid:
        matching_option = matching_option.model_copy(
            update={
                "validation_status": validation_status,
                "validation_error": validation_error,
                "masked_key": mask_api_key(candidate_key) or matching_option.masked_key,
                "key_source": candidate_source if candidate_key else matching_option.key_source,
                "configured": (
                    bool(candidate_key and candidate_base_url)
                    if payload.provider == "openai_compatible"
                    else bool(candidate_key)
                ),
                "base_url": candidate_base_url if payload.provider == "openai_compatible" else matching_option.base_url,
            }
        )

    return VerifyKeyResponse(
        provider=payload.provider,
        valid=valid,
        validation_status=validation_status,
        validation_error=validation_error,
        key_source=matching_option.key_source,
        masked_key=matching_option.masked_key,
        tested_model=test_model,
        provider_option=matching_option,
    )
