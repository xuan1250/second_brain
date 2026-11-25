"""Document query, artifact download, deletion, metadata update, and lifecycle facade service."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, status
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings
from src.db.models import Document, DomainRegistry, SubdomainRegistry
from src.db.repository import (
    delete_document_atomic,
    get_domain_by_slug,
    get_latest_job_for_document,
    get_latest_jobs_for_documents,
    list_documents_paginated,
    sync_document_metadata_across_stores,
)
from src.ingestion.worker import IngestionOrchestrator
from src.services.document_common import (
    MEDIA_TYPES,
    _HEADING_LINE_RE,
    _now,
    _require_document,
    build_document_response,
    extract_markdown_headings,
    resolve_document_artifact_path,
    sanitize_filename,
)
from src.services.document_lifecycle_service import (
    _finalize_raw_destination,
    approve_document_classification,
    approve_documents_batch,
    create_queued_ingestion_job,
    handle_document_uploads,
    resolve_upload_defaults,
    retry_document_ingestion,
    run_background_ingestion,
    stream_upload_to_temp,
)

if TYPE_CHECKING:
    from src.api.schemas import (
        DocumentDeleteResponse,
        DocumentListResponse,
        DocumentMarkdownResponse,
        DocumentResponse,
        DocumentStatusResponse,
        HeadingOutlineItem,
        UpdateDocumentMetadataRequest,
    )


async def list_documents_view(
    session: AsyncSession,
    *,
    status_filter: str | None = None,
    file_type: str | None = None,
    search: str | None = None,
    domain: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> DocumentListResponse:
    """List paginated documents with optional filters and batched latest job hydration."""
    from src.api.schemas import DocumentListResponse

    domain_id_to_filter: str | None = None
    if domain and domain.strip():
        clean_domain = domain.strip()
        d_obj = await get_domain_by_slug(session, clean_domain)
        domain_id_to_filter = d_obj.id if d_obj else clean_domain

    rows, total = await list_documents_paginated(
        session,
        status_filter=status_filter,
        file_type=file_type,
        search=search,
        domain_id=domain_id_to_filter,
        limit=limit,
        offset=offset,
    )
    latest_jobs = await get_latest_jobs_for_documents(session, [d.id for d in rows]) if rows else {}
    items = [build_document_response(doc, latest_jobs.get(doc.id)) for doc in rows]
    return DocumentListResponse(
        items=items,
        documents=items,
        total=total,
        limit=limit,
        offset=offset,
    )


async def get_document_detail(session: AsyncSession, doc_id: str) -> DocumentResponse:
    """Return DocumentResponse for `doc_id` or raise HTTP 404."""
    doc = await _require_document(session, doc_id)
    latest_job = await get_latest_job_for_document(session, doc.id)
    return build_document_response(doc, latest_job)


async def get_document_status_view(session: AsyncSession, doc_id: str) -> DocumentStatusResponse:
    """Return DocumentStatusResponse for `doc_id` or raise HTTP 404."""
    from src.api.schemas import DocumentStatusResponse

    base = await get_document_detail(session, doc_id)
    return DocumentStatusResponse(**base.model_dump())


async def get_document_raw_download(
    session: AsyncSession,
    doc_id: str,
    data_dir: Path,
) -> tuple[Path, str, str]:
    """Return `(raw_path, media_type, safe_filename)` for inline FileResponse streaming."""
    doc = await _require_document(session, doc_id)
    raw_path = resolve_document_artifact_path(doc, data_dir, "raw")
    ext = (doc.file_type or "").lower().lstrip(".")
    safe_name = sanitize_filename(doc.filename)
    media_type = MEDIA_TYPES.get(ext, "application/octet-stream")
    return raw_path, media_type, safe_name


async def get_document_markdown_view(
    session: AsyncSession,
    doc_id: str,
    data_dir: Path,
) -> DocumentMarkdownResponse:
    """Read Markdown mirror and extract heading outline for `doc_id`."""
    from src.api.schemas import DocumentMarkdownResponse

    doc = await _require_document(session, doc_id)
    md_path = resolve_document_artifact_path(doc, data_dir, "markdown")

    def _read_markdown_and_headings(path: Path) -> tuple[str, list[HeadingOutlineItem]]:
        text = path.read_text(encoding="utf-8")
        return text, extract_markdown_headings(text)

    md_text, headings = await asyncio.to_thread(_read_markdown_and_headings, md_path)
    return DocumentMarkdownResponse(
        id=doc.id,
        doc_id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        page_count=doc.page_count,
        markdown=md_text,
        headings=headings,
    )


async def delete_document_and_cancel(
    *,
    doc_id: str,
    orchestrator: IngestionOrchestrator | None,
    session: AsyncSession,
    qdrant: AsyncQdrantClient,
    settings: Settings,
) -> DocumentDeleteResponse:
    """Cancel any in-flight ingestion for `doc_id` and delete across Qdrant, SQLite, and disk."""
    from src.api.schemas import DocumentDeleteResponse

    if orchestrator is not None:
        orchestrator.cancel_document(doc_id)

    deleted = await delete_document_atomic(
        session=session,
        qdrant=qdrant,
        doc_id=doc_id,
        data_dir=settings.data_dir,
        collection=settings.qdrant_collection,
        orchestrator=orchestrator,
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )
    return DocumentDeleteResponse(id=doc_id, deleted=True, status="deleted")


async def _resolve_metadata_domain_and_subdomain(
    session: AsyncSession,
    doc: Document,
    payload: UpdateDocumentMetadataRequest,
) -> tuple[str | None, str | None]:
    """Resolve and validate target domain/subdomain IDs for metadata updates."""
    d_obj: DomainRegistry | None = None
    target_domain_id = payload.primary_domain_id
    if payload.domain:
        d_obj = await get_domain_by_slug(session, payload.domain)
        if d_obj is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Domain '{payload.domain}' not found",
            )
        target_domain_id = d_obj.id
    elif target_domain_id:
        d_obj = await session.get(DomainRegistry, target_domain_id)
        if d_obj is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Domain id '{target_domain_id}' not found",
            )
    elif doc.primary_domain_id:
        d_obj = await session.get(DomainRegistry, doc.primary_domain_id)

    s_obj: SubdomainRegistry | None = None
    target_subdomain_id = payload.primary_subdomain_id
    if payload.subdomain:
        res_s = await session.execute(
            select(SubdomainRegistry).where(SubdomainRegistry.slug == payload.subdomain)
        )
        s_obj = res_s.scalars().first()
        if s_obj is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Subdomain '{payload.subdomain}' not found",
            )
        target_subdomain_id = s_obj.id
    elif target_subdomain_id:
        s_obj = await session.get(SubdomainRegistry, target_subdomain_id)
        if s_obj is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Subdomain id '{target_subdomain_id}' not found",
            )

    if d_obj is not None and s_obj is not None:
        if s_obj.domain_id != d_obj.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Subdomain '{s_obj.slug}' does not belong to domain '{d_obj.slug}'",
            )
    elif d_obj is not None and s_obj is None and (payload.domain or payload.primary_domain_id):
        curr_sub = (
            await session.get(SubdomainRegistry, doc.primary_subdomain_id)
            if doc.primary_subdomain_id
            else None
        )
        if curr_sub and curr_sub.domain_id == d_obj.id:
            target_subdomain_id = curr_sub.id
        else:
            res_s_fallback = await session.execute(
                select(SubdomainRegistry).where(
                    SubdomainRegistry.domain_id == d_obj.id,
                    SubdomainRegistry.slug == f"{d_obj.slug}.general",
                )
            )
            s_obj = res_s_fallback.scalars().first()
            if s_obj is None:
                res_s_any = await session.execute(
                    select(SubdomainRegistry)
                    .where(SubdomainRegistry.domain_id == d_obj.id)
                    .order_by(SubdomainRegistry.slug)
                )
                s_obj = res_s_any.scalars().first()
            if s_obj is not None:
                target_subdomain_id = s_obj.id

    return target_domain_id, target_subdomain_id


async def update_document_metadata_view(
    *,
    doc_id: str,
    payload: UpdateDocumentMetadataRequest,
    session: AsyncSession,
    qdrant: AsyncQdrantClient,
    settings: Settings,
) -> DocumentResponse:
    """Update document metadata in SQLite and synchronize chunks & Qdrant vector payloads."""
    doc = await session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found",
        )

    target_domain_id, target_subdomain_id = await _resolve_metadata_domain_and_subdomain(
        session, doc, payload
    )
    if target_domain_id is not None:
        doc.primary_domain_id = target_domain_id
    if target_subdomain_id is not None:
        doc.primary_subdomain_id = target_subdomain_id

    if payload.authority_tier is not None:
        doc.authority_tier = payload.authority_tier
    if payload.jurisdiction is not None:
        doc.jurisdiction = payload.jurisdiction
    if payload.effective_year is not None:
        doc.effective_year = payload.effective_year
    if payload.is_current is not None:
        doc.is_current = payload.is_current

    if payload.section_overrides is not None:
        parsed_prof: dict[str, Any] = {}
        if doc.classification_result:
            try:
                parsed_prof = json.loads(doc.classification_result)
            except Exception:
                parsed_prof = {}
        parsed_prof["section_overrides"] = [ov.model_dump() for ov in payload.section_overrides]
        parsed_prof["is_hybrid"] = bool(payload.section_overrides)
        doc.classification_result = json.dumps(parsed_prof)

    await sync_document_metadata_across_stores(
        session=session,
        qdrant=qdrant,
        collection_name=settings.qdrant_collection,
        doc=doc,
    )

    latest_job = await get_latest_job_for_document(session, doc.id)
    return build_document_response(doc, latest_job)


__all__ = [
    "MEDIA_TYPES",
    "_HEADING_LINE_RE",
    "_finalize_raw_destination",
    "_now",
    "_require_document",
    "_resolve_metadata_domain_and_subdomain",
    "approve_document_classification",
    "approve_documents_batch",
    "build_document_response",
    "create_queued_ingestion_job",
    "delete_document_and_cancel",
    "extract_markdown_headings",
    "get_document_detail",
    "get_document_markdown_view",
    "get_document_raw_download",
    "get_document_status_view",
    "handle_document_uploads",
    "list_documents_view",
    "resolve_document_artifact_path",
    "resolve_upload_defaults",
    "retry_document_ingestion",
    "run_background_ingestion",
    "sanitize_filename",
    "stream_upload_to_temp",
    "update_document_metadata_view",
]
