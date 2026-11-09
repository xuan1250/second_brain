"""Document upload, paginated listing, status tracking, raw/Markdown preview, and atomic deletion routes."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile as StarletteUploadFile
from qdrant_client import AsyncQdrantClient
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

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
    HeadingOutlineItem,
    UpdateDocumentMetadataRequest,
)
from src.db.models import DomainRegistry, SubdomainRegistry
from src.config import Settings
from src.db.models import Document, IngestionJob
from src.db.repository import (
    delete_document_atomic,
    get_document_by_sha,
    get_domain_by_slug,
    get_workspace_settings,
    sync_document_metadata_across_stores,
)
from src.ingestion.parser import SUPPORTED_EXTENSIONS
from src.ingestion.worker import IngestionOrchestrator

router = APIRouter(prefix="/documents", tags=["documents"])

_MEDIA_TYPES: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def sanitize_filename(filename: str) -> str:
    """Sanitize a user-supplied filename for safe storage metadata and Content-Disposition headers."""
    base = Path(filename or "document").name
    cleaned = re.sub(r'[\x00-\x1f\x7f"\\/:<>|?*]+', "_", base).strip()
    return cleaned or "document"


def extract_markdown_headings(markdown_text: str) -> list[HeadingOutlineItem]:
    """Extract hierarchical heading outline (`level`, `text`, `heading_path`, `line`) from Markdown."""
    headings: list[HeadingOutlineItem] = []
    stack: dict[int, str] = {}
    for line_no, raw_line in enumerate((markdown_text or "").splitlines(), start=1):
        match = _HEADING_LINE_RE.match(raw_line)
        if not match:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        if not title:
            continue
        for k in list(stack.keys()):
            if k >= level:
                del stack[k]
        stack[level] = title
        breadcrumb = " > ".join(stack[k] for k in sorted(stack.keys()))
        headings.append(
            HeadingOutlineItem(
                level=level,
                text=title,
                heading_path=breadcrumb,
                line=line_no,
            )
        )
    return headings


def _build_document_response(
    doc: Document,
    job: IngestionJob | None = None,
    *,
    duplicate: bool = False,
) -> DocumentResponse:
    timings: dict[str, int] = {}
    if job is not None and job.stage_timings:
        try:
            parsed = json.loads(job.stage_timings)
            if isinstance(parsed, dict):
                timings = {str(k): int(v) for k, v in parsed.items() if isinstance(v, (int, float))}
        except Exception:
            timings = {}

    if job is not None:
        stage = job.stage or doc.status
        progress = float(job.progress)
        duration_ms = job.duration_ms
        job_id = job.id
        err = doc.error or job.error
    else:
        stage = doc.status
        if doc.status == "awaiting_approval":
            progress = 0.45
        elif doc.status == "classifying":
            progress = 0.35
        elif doc.status in ("ready", "failed"):
            progress = 1.0
        else:
            progress = 0.0
        duration_ms = None
        job_id = None
        err = doc.error

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        sha256=doc.sha256,
        size_bytes=doc.size_bytes,
        page_count=doc.page_count,
        chunk_count=doc.chunk_count,
        status=doc.status,
        error=err,
        error_message=err,
        job_id=job_id,
        stage=stage,
        progress=progress,
        duration_ms=duration_ms,
        stage_timings=timings,
        duplicate=duplicate,
        parsing_engine=doc.parsing_engine,
        engine_fallback_reason=doc.engine_fallback_reason,
        primary_domain_id=doc.primary_domain_id,
        primary_subdomain_id=doc.primary_subdomain_id,
        jurisdiction=doc.jurisdiction,
        authority_tier=doc.authority_tier,
        effective_year=doc.effective_year,
        is_current=doc.is_current,
        classification_result=doc.classification_result,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


async def _get_latest_job(session: AsyncSession, doc_id: str) -> IngestionJob | None:
    res = await session.execute(
        select(IngestionJob)
        .where(IngestionJob.doc_id == doc_id)
        .order_by(IngestionJob.created_at.desc())
    )
    return res.scalars().first()


async def _run_background_ingestion(
    orchestrator: IngestionOrchestrator,
    doc_id: str,
    job_id: str,
    raw_path: Path,
    force_reparse: bool = False,
    prefer_pymupdf4llm: bool | None = None,
    approval_mode: str = "review_all",
) -> None:
    """Execute document ingestion in the background; errors are recorded on the Document/Job rows."""
    try:
        await orchestrator.process_document(
            doc_id=doc_id,
            job_id=job_id,
            raw_path=raw_path,
            force_reparse=force_reparse,
            prefer_pymupdf4llm=prefer_pymupdf4llm,
            approval_mode=approval_mode,
        )
    except Exception:
        # `process_document` already updates Document.status='failed' and IngestionJob.error
        pass


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
    task = asyncio.create_task(
        _run_background_ingestion(
            orchestrator=orchestrator,
            doc_id=doc_id,
            job_id=job_id,
            raw_path=raw_path,
            force_reparse=force_reparse,
            prefer_pymupdf4llm=prefer_pymupdf4llm,
            approval_mode=approval_mode,
        )
    )
    bg_tasks: set[asyncio.Task[Any]] | None = getattr(request.app.state, "background_tasks", None)
    if bg_tasks is None:
        bg_tasks = set()
        request.app.state.background_tasks = bg_tasks
    bg_tasks.add(task)
    task.add_done_callback(bg_tasks.discard)
    return task


async def _stream_upload_to_temp(
    upload: UploadFile,
    raw_dir: Path,
    max_bytes: int,
    max_upload_size_mb: int,
) -> tuple[str, int, Path]:
    """Stream an UploadFile in 64KB chunks while computing SHA-256 and enforcing size limits."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = raw_dir / f".upload_{uuid.uuid4().hex}.tmp"
    hasher = hashlib.sha256()
    size_bytes = 0

    try:
        with tmp_path.open("wb") as f:
            while True:
                chunk = await upload.read(65536)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {max_upload_size_mb} MB.",
                    )
                hasher.update(chunk)
                f.write(chunk)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    if size_bytes == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    return hasher.hexdigest(), size_bytes, tmp_path


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
    # Fallback to WorkspaceSettings when query parameters are omitted
    if prefer_fast_path is None or approval_mode is None:
        ws = await get_workspace_settings(session)
        if prefer_fast_path is None:
            if ws.default_upload_mode == "fast":
                prefer_fast_path = True
            elif ws.default_upload_mode == "gpu":
                prefer_fast_path = False
            else:
                prefer_fast_path = None
        if approval_mode is None:
            if ws.default_approval_mode in ("review_all", "auto_high_confidence", "auto_all"):
                approval_mode = ws.default_approval_mode  # type: ignore[assignment]
            else:
                approval_mode = "review_all"

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

    raw_dir = settings.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    max_upload_mb = settings.ingestion.max_upload_size_mb
    max_bytes = max_upload_mb * 1024 * 1024

    results: list[DocumentResponse] = []
    all_duplicates = True

    for upload in upload_list:
        orig_filename = sanitize_filename(upload.filename or "")
        ext_with_dot = Path(orig_filename).suffix.lower()
        if ext_with_dot not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported file extension '{ext_with_dot}'. "
                    f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
                ),
            )
        ext = ext_with_dot.lstrip(".")

        sha256, size_bytes, tmp_path = await _stream_upload_to_temp(
            upload=upload,
            raw_dir=raw_dir,
            max_bytes=max_bytes,
            max_upload_size_mb=max_upload_mb,
        )
        raw_dest = raw_dir / f"{sha256}.{ext}"

        existing = await get_document_by_sha(session, sha256)
        if existing is not None and existing.status != "failed":
            requested_engine = (
                "pymupdf4llm"
                if prefer_fast_path is True
                else ("docling" if prefer_fast_path is False else None)
            )
            # Auto-upgrade on duplicate upload or allow awaiting_approval to re-enter
            is_mode_upgrade = requested_engine == "docling" and existing.parsing_engine != "docling"
            is_awaiting_reenter = existing.status == "awaiting_approval" and approval_mode in ("auto_all", "auto_high_confidence")
            if not is_mode_upgrade and not is_awaiting_reenter:
                # Idempotent hit: ensure raw file exists on disk, remove temp file, and return existing record
                if not raw_dest.is_file():
                    os.replace(tmp_path, raw_dest)
                else:
                    tmp_path.unlink(missing_ok=True)
                latest_job = await _get_latest_job(session, existing.id)
                results.append(_build_document_response(existing, latest_job, duplicate=True))
                continue

            # User explicitly requested upgrade or awaiting_approval re-entry
            if not raw_dest.is_file():
                os.replace(tmp_path, raw_dest)
            else:
                tmp_path.unlink(missing_ok=True)

            now = _now()
            stmt = (
                update(Document)
                .where(Document.id == existing.id)
                .where(Document.status.in_(["ready", "failed", "awaiting_approval"]))
                .values(status="queued", error=None, updated_at=now)
            )
            res = await session.execute(stmt)
            await session.commit()

            if res.rowcount == 0:
                # Document is already in-flight (queued, parsing, chunking, indexing)
                # Safely coalesce the request without scheduling a duplicate worker
                await session.refresh(existing)
                latest_job = await _get_latest_job(session, existing.id)
                results.append(_build_document_response(existing, latest_job, duplicate=True))
                continue

            all_duplicates = False
            await session.refresh(existing)

            job = IngestionJob(
                doc_id=existing.id,
                status="queued",
                stage="queued",
                progress=0.0,
                stage_timings=json.dumps({}),
                created_at=now,
                updated_at=now,
            )
            session.add(job)
            await session.commit()
            await session.refresh(job)

            _schedule_ingestion_task(
                request=request,
                orchestrator=orchestrator,
                doc_id=existing.id,
                job_id=job.id,
                raw_path=raw_dest,
                force_reparse=True,
                prefer_pymupdf4llm=prefer_fast_path,
                approval_mode=approval_mode,
            )
            results.append(_build_document_response(existing, job, duplicate=False))
            continue

        all_duplicates = False
        os.replace(tmp_path, raw_dest)

        now = _now()
        if existing is None:
            doc = Document(
                filename=orig_filename,
                file_type=ext,
                sha256=sha256,
                size_bytes=size_bytes,
                status="queued",
                created_at=now,
                updated_at=now,
            )
            session.add(doc)
            try:
                await session.commit()
                await session.refresh(doc)
            except IntegrityError:
                await session.rollback()
                existing = await get_document_by_sha(session, sha256)
                if existing is not None:
                    latest_job = await _get_latest_job(session, existing.id)
                    results.append(_build_document_response(existing, latest_job, duplicate=True))
                    continue
                raise
        else:
            doc = existing
            doc.filename = orig_filename
            doc.size_bytes = size_bytes
            doc.status = "queued"
            doc.error = None
            doc.updated_at = now
            session.add(doc)
            await session.commit()
            await session.refresh(doc)

        job = IngestionJob(
            doc_id=doc.id,
            status="queued",
            stage="queued",
            progress=0.0,
            stage_timings=json.dumps({}),
            created_at=now,
            updated_at=now,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)

        _schedule_ingestion_task(
            request=request,
            orchestrator=orchestrator,
            doc_id=doc.id,
            job_id=job.id,
            raw_path=raw_dest,
            prefer_pymupdf4llm=prefer_fast_path,
            approval_mode=approval_mode,
        )
        results.append(_build_document_response(doc, job, duplicate=False))

    all_duplicates = all(r.duplicate for r in results)
    response.status_code = status.HTTP_200_OK if all_duplicates else status.HTTP_202_ACCEPTED
    primary = results[0]
    primary.documents = [r.model_copy(update={"documents": None}) for r in results]
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
    stmt = select(Document)
    count_stmt = select(func.count()).select_from(Document)

    if status_filter and status_filter.strip():
        clean_status = status_filter.strip().lower()
        stmt = stmt.where(Document.status == clean_status)
        count_stmt = count_stmt.where(Document.status == clean_status)

    if file_type and file_type.strip():
        clean_ft = file_type.strip().lower().lstrip(".")
        stmt = stmt.where(Document.file_type == clean_ft)
        count_stmt = count_stmt.where(Document.file_type == clean_ft)

    if search and search.strip():
        pattern = f"%{search.strip()}%"
        stmt = stmt.where(Document.filename.ilike(pattern))
        count_stmt = count_stmt.where(Document.filename.ilike(pattern))

    if domain and domain.strip():
        clean_domain = domain.strip()
        d_obj = await get_domain_by_slug(session, clean_domain)
        domain_id_to_filter = d_obj.id if d_obj else clean_domain
        stmt = stmt.where(Document.primary_domain_id == domain_id_to_filter)
        count_stmt = count_stmt.where(Document.primary_domain_id == domain_id_to_filter)

    total = int((await session.execute(count_stmt)).scalar() or 0)
    rows = (
        await session.execute(
            stmt.order_by(Document.created_at.desc()).offset(offset).limit(limit)
        )
    ).scalars().all()

    latest_jobs: dict[str, IngestionJob] = {}
    if rows:
        doc_ids = [d.id for d in rows]
        job_stmt = (
            select(IngestionJob)
            .where(IngestionJob.doc_id.in_(doc_ids))
            .order_by(IngestionJob.created_at.desc())
        )
        job_rows = (await session.execute(job_stmt)).scalars().all()
        for j in job_rows:
            if j.doc_id not in latest_jobs:
                latest_jobs[j.doc_id] = j

    items: list[DocumentResponse] = []
    for doc in rows:
        latest_job = latest_jobs.get(doc.id)
        items.append(_build_document_response(doc, latest_job))

    return DocumentListResponse(
        items=items,
        documents=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> DocumentResponse:
    """Return document metadata, chunk count, and latest ingestion job progress."""
    doc = await session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )
    latest_job = await _get_latest_job(session, doc.id)
    return _build_document_response(doc, latest_job)


@router.get("/{doc_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    doc_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> DocumentStatusResponse:
    """Return current ingestion status, progress, and stage timings for a document."""
    doc = await session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )
    latest_job = await _get_latest_job(session, doc.id)
    base = _build_document_response(doc, latest_job)
    return DocumentStatusResponse(**base.model_dump())


@router.get("/{doc_id}/raw")
async def get_document_raw(
    doc_id: str,
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
) -> FileResponse:
    """Stream the original immutable uploaded file from `data/raw/{sha256}.{ext}`."""
    doc = await session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    sha = (doc.sha256 or "").lower()
    ext = (doc.file_type or "").lower().lstrip(".")
    if not _HEX64_RE.match(sha) or not ext:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid document storage reference.",
        )

    raw_path = settings.data_dir / "raw" / f"{sha}.{ext}"
    if not raw_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Raw file for document '{doc_id}' not found on disk.",
        )

    safe_name = sanitize_filename(doc.filename)
    media_type = _MEDIA_TYPES.get(ext, "application/octet-stream")
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
    doc = await session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    sha = (doc.sha256 or "").lower()
    if not _HEX64_RE.match(sha):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid document storage reference.",
        )

    md_path = settings.data_dir / "markdown" / f"{sha}.md"
    if not md_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Markdown mirror for document '{doc_id}' is not available yet.",
        )

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
    doc = await session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    if doc.status == "ready" and not force_reparse:
        latest_job = await _get_latest_job(session, doc.id)
        return _build_document_response(doc, latest_job)

    sha = (doc.sha256 or "").lower()
    ext = (doc.file_type or "").lower().lstrip(".")
    if not _HEX64_RE.match(sha) or not ext:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid document storage reference.",
        )

    raw_dest = settings.data_dir / "raw" / f"{sha}.{ext}"
    if not raw_dest.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Raw file for document '{doc_id}' not found on disk.",
        )

    now = _now()
    stmt = (
        update(Document)
        .where(
            Document.id == doc_id,
            Document.status.not_in(["parsing", "classifying", "chunking", "indexing", "queued"]),
        )
        .values(status="queued", error=None, updated_at=now)
    )
    res = await session.execute(stmt)
    if res.rowcount == 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document is currently processing or already queued",
        )
    await session.commit()
    await session.refresh(doc)

    job = IngestionJob(
        doc_id=doc.id,
        status="queued",
        stage="queued",
        progress=0.0,
        stage_timings=json.dumps({}),
        created_at=now,
        updated_at=now,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    _schedule_ingestion_task(
        request=request,
        orchestrator=orchestrator,
        doc_id=doc.id,
        job_id=job.id,
        raw_path=raw_dest,
        force_reparse=force_reparse,
        prefer_pymupdf4llm=prefer_fast_path,
    )

    return _build_document_response(doc, job)


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
    doc = await session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found",
        )

    if doc.status not in ("awaiting_approval", "ready", "failed"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document is currently processing (status='{doc.status}')",
        )

    latest_job = await _get_latest_job(session, doc.id)
    job_id = latest_job.id if latest_job else None

    completed_doc = await orchestrator.complete_approved_ingestion(
        doc_id=doc.id,
        job_id=job_id,
        approved_profile=payload,
    )

    latest_job = await _get_latest_job(session, doc.id)
    return _build_document_response(completed_doc, latest_job)


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
    approved_docs: list[DocumentResponse] = []
    for d_id in payload.doc_ids:
        doc = await session.get(Document, d_id)
        if doc is None or doc.status != "awaiting_approval":
            continue
        latest_job = await _get_latest_job(session, doc.id)
        job_id = latest_job.id if latest_job else None
        completed_doc = await orchestrator.complete_approved_ingestion(
            doc_id=doc.id,
            job_id=job_id,
        )
        latest_job = await _get_latest_job(session, doc.id)
        approved_docs.append(_build_document_response(completed_doc, latest_job))

    return ApproveBatchResponse(
        approved_count=len(approved_docs),
        documents=approved_docs,
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
    doc = await session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found",
        )

    # 1. Resolve domain
    d_obj = None
    target_domain_id = payload.primary_domain_id
    if payload.domain:
        res_d = await session.execute(
            select(DomainRegistry).where(DomainRegistry.slug == payload.domain)
        )
        d_obj = res_d.scalars().first()
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

    # 2. Resolve subdomain
    s_obj = None
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

    # Validate pair if both are resolved
    if d_obj is not None and s_obj is not None:
        if s_obj.domain_id != d_obj.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Subdomain '{s_obj.slug}' does not belong to domain '{d_obj.slug}'",
            )
    elif d_obj is not None and s_obj is None and (payload.domain or payload.primary_domain_id):
        res_curr_sub = await session.get(SubdomainRegistry, doc.primary_subdomain_id) if doc.primary_subdomain_id else None
        if res_curr_sub and res_curr_sub.domain_id == d_obj.id:
            s_obj = res_curr_sub
            target_subdomain_id = s_obj.id
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
            if s_obj:
                target_subdomain_id = s_obj.id

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

    # Process section_overrides if provided
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

    latest_job = await _get_latest_job(session, doc.id)
    return _build_document_response(doc, latest_job)
