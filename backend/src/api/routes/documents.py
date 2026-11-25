"""Document upload, paginated listing, status tracking, raw/Markdown preview, and atomic deletion routes."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.datastructures import UploadFile as StarletteUploadFile

from src.api.dependencies import (
    get_app_settings,
    get_db_session,
    get_orchestrator,
    get_qdrant,
)
from src.api.schemas import (
    ApproveBatchRequest,
    ApproveBatchResponse,
    ApprovedClassificationRequest,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentMarkdownResponse,
    DocumentResponse,
    DocumentStatusResponse,
    UpdateDocumentMetadataRequest,
)
from src.config import Settings
from src.db.repository import get_document_by_sha
from src.ingestion.worker import IngestionOrchestrator
from src.services.document_service import (
    approve_document_classification,
    approve_documents_batch,
    build_document_response as _build_document_response,
    delete_document_and_cancel,
    extract_markdown_headings,
    get_document_detail,
    get_document_markdown_view,
    get_document_raw_download,
    get_document_status_view,
    handle_document_uploads,
    list_documents_view,
    retry_document_ingestion,
    run_background_ingestion,
    sanitize_filename,
    update_document_metadata_view,
)
from src.services.reindex_service import schedule_background_task

router = APIRouter(prefix="/documents", tags=["documents"])

__all__ = [
    "_build_document_response",
    "_schedule_ingestion_task",
    "extract_markdown_headings",
    "get_document_by_sha",
    "router",
    "sanitize_filename",
]


def _schedule_ingestion_task(
    request: Request,
    orchestrator: IngestionOrchestrator,
    doc_id: str,
    job_id: str,
    raw_path: Path,
    force_reparse: bool = False,
    prefer_pymupdf4llm: bool | None = None,
    approval_mode: str = "review_all",
) -> asyncio.Task[None]:
    """Schedule background document ingestion on `app.state.background_tasks`."""
    return schedule_background_task(
        request.app.state,
        run_background_ingestion(
            orchestrator=orchestrator,
            doc_id=doc_id,
            job_id=job_id,
            raw_path=raw_path,
            force_reparse=force_reparse,
            prefer_pymupdf4llm=prefer_pymupdf4llm,
            approval_mode=approval_mode,
        ),
    )


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_documents(
    request: Request,
    response: Response,
    file: UploadFile | None = File(default=None),
    files: list[UploadFile] | None = File(default=None),
    prefer_fast_path: bool | None = Query(
        default=None,
        description="Override parser engine: None=auto heuristic, True=force PyMuPDF CPU, False=force Docling CUDA",
    ),
    approval_mode: Literal["review_all", "auto_high_confidence", "auto_all"] | None = Query(
        default=None,
        description="Approval mode: review_all (default for UI), auto_high_confidence, or auto_all",
    ),
    session: AsyncSession = Depends(get_db_session),
    orchestrator: IngestionOrchestrator = Depends(get_orchestrator),
    settings: Settings = Depends(get_app_settings),
) -> DocumentResponse:
    """Upload one or more documents, deduplicate via SHA-256, and dispatch background ingestion."""
    upload_list: list[UploadFile] = []
    if file is not None and isinstance(file, StarletteUploadFile):
        upload_list.append(file)
    if files and isinstance(files, list):
        upload_list.extend(files)

    if not upload_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided. Submit multipart field 'file' or 'files'.",
        )

    primary, status_code = await handle_document_uploads(
        request=request,
        upload_list=upload_list,
        prefer_fast_path=prefer_fast_path,
        approval_mode=approval_mode,
        session=session,
        orchestrator=orchestrator,
        settings=settings,
        get_doc_by_sha_fn=get_document_by_sha,
        schedule_fn=_schedule_ingestion_task,
    )
    response.status_code = status_code
    return primary


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    status_filter: str | None = Query(default=None, alias="status"),
    file_type: str | None = Query(default=None),
    search: str | None = Query(default=None),
    domain: str | None = Query(default=None, description="Filter documents by primary domain slug or ID"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> DocumentListResponse:
    """Return a paginated list of documents with optional `status`, `file_type`, `search`, and `domain` filters."""
    return await list_documents_view(
        session,
        status_filter=status_filter,
        file_type=file_type,
        search=search,
        domain=domain,
        limit=limit,
        offset=offset,
    )


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> DocumentResponse:
    """Return document metadata, chunk count, and latest ingestion job progress."""
    return await get_document_detail(session, doc_id)


@router.get("/{doc_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    doc_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> DocumentStatusResponse:
    """Return current ingestion status, progress, and stage timings for a document."""
    return await get_document_status_view(session, doc_id)


@router.get("/{doc_id}/raw")
async def get_document_raw(
    doc_id: str,
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
) -> FileResponse:
    """Stream the original immutable uploaded file from `data/raw/{sha256}.{ext}`."""
    raw_path, media_type, safe_name = await get_document_raw_download(
        session, doc_id, settings.data_dir
    )
    return FileResponse(
        path=raw_path,
        media_type=media_type,
        filename=safe_name,
        content_disposition_type="inline",
    )


@router.get("/{doc_id}/markdown", response_model=DocumentMarkdownResponse)
async def get_document_markdown(
    doc_id: str,
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
) -> DocumentMarkdownResponse:
    """Return the converted Markdown mirror and extracted heading outline for a document."""
    return await get_document_markdown_view(session, doc_id, settings.data_dir)


@router.post("/{doc_id}/retry", response_model=DocumentResponse)
async def retry_document(
    doc_id: str,
    request: Request,
    force_reparse: bool = Query(default=False),
    prefer_fast_path: bool | None = Query(default=None),
    session: AsyncSession = Depends(get_db_session),
    orchestrator: IngestionOrchestrator = Depends(get_orchestrator),
    settings: Settings = Depends(get_app_settings),
) -> DocumentResponse:
    """Retry ingestion for a failed, cancelled, or ready document, with optional checkpoint bypass and fast-path override."""
    return await retry_document_ingestion(
        doc_id=doc_id,
        request=request,
        force_reparse=force_reparse,
        prefer_fast_path=prefer_fast_path,
        session=session,
        orchestrator=orchestrator,
        settings=settings,
        schedule_fn=_schedule_ingestion_task,
    )


@router.delete("/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    doc_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    settings: Settings = Depends(get_app_settings),
) -> DocumentDeleteResponse:
    """Atomically delete a document across Qdrant vectors, SQLite tables, and `data/` artifacts."""
    orchestrator: IngestionOrchestrator | None = getattr(request.app.state, "orchestrator", None)
    return await delete_document_and_cancel(
        doc_id=doc_id,
        orchestrator=orchestrator,
        session=session,
        qdrant=qdrant,
        settings=settings,
    )


@router.post(
    "/{doc_id}/approve-classification",
    response_model=DocumentResponse,
    summary="Approve or adjust document classification and proceed with chunking & indexing",
)
async def approve_classification(
    doc_id: str,
    payload: ApprovedClassificationRequest | None = None,
    orchestrator: IngestionOrchestrator = Depends(get_orchestrator),
    session: AsyncSession = Depends(get_db_session),
) -> DocumentResponse:
    """Approve classification and resume Stage 2 (chunking) and Stage 3 (indexing) from cache."""
    return await approve_document_classification(
        doc_id=doc_id,
        payload=payload,
        orchestrator=orchestrator,
        session=session,
    )


@router.post(
    "/approve-batch",
    response_model=ApproveBatchResponse,
    summary="Batch approve all documents in awaiting_approval state",
)
async def approve_batch(
    payload: ApproveBatchRequest,
    orchestrator: IngestionOrchestrator = Depends(get_orchestrator),
    session: AsyncSession = Depends(get_db_session),
) -> ApproveBatchResponse:
    """Resume Stage 2/3 for all specified documents in awaiting_approval state."""
    return await approve_documents_batch(
        doc_ids=payload.doc_ids,
        orchestrator=orchestrator,
        session=session,
    )


@router.patch(
    "/{doc_id}/metadata",
    response_model=DocumentResponse,
    summary="Update document domain, authority, or temporal metadata without re-parsing",
)
async def update_document_metadata(
    doc_id: str,
    payload: UpdateDocumentMetadataRequest,
    session: AsyncSession = Depends(get_db_session),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    settings: Settings = Depends(get_app_settings),
) -> DocumentResponse:
    """Update document metadata fields in SQLite and synchronize chunks & Qdrant points."""
    return await update_document_metadata_view(
        doc_id=doc_id,
        payload=payload,
        session=session,
        qdrant=qdrant,
        settings=settings,
    )
