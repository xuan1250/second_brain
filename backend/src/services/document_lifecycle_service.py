"""Document upload, deduplication, retry, and approval lifecycle service."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from fastapi import HTTPException, Request, UploadFile, status
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings
from src.db.models import Document, IngestionJob
from src.db.repository import (
    get_document_by_sha,
    get_latest_job_for_document,
    get_workspace_settings,
)
from src.ingestion.constants import SUPPORTED_EXTENSIONS, is_engine_mode_upgrade
from src.ingestion.worker import IngestionOrchestrator
from src.services.document_common import (
    _now,
    _require_document,
    build_document_response,
    resolve_document_artifact_path,
    sanitize_filename,
)

if TYPE_CHECKING:
    from src.api.schemas import (
        ApproveBatchResponse,
        ApprovedClassificationRequest,
        DocumentResponse,
    )


async def run_background_ingestion(
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
        # `process_document` already updates Document.status='failed' and IngestionJob.error.
        pass


async def stream_upload_to_temp(
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


async def create_queued_ingestion_job(
    session: AsyncSession,
    doc_id: str,
    *,
    now: datetime | None = None,
    progress: float = 0.0,
) -> IngestionJob:
    """Create, commit, and refresh a queued IngestionJob row."""
    ts = now or _now()
    job = IngestionJob(
        doc_id=doc_id,
        status="queued",
        stage="queued",
        progress=progress,
        stage_timings=json.dumps({}),
        created_at=ts,
        updated_at=ts,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def resolve_upload_defaults(
    session: AsyncSession,
    prefer_fast_path: bool | None,
    approval_mode: Literal["review_all", "auto_high_confidence", "auto_all"] | None,
) -> tuple[bool | None, Literal["review_all", "auto_high_confidence", "auto_all"]]:
    """Resolve omitted upload query flags from WorkspaceSettings defaults."""
    if prefer_fast_path is not None and approval_mode is not None:
        return prefer_fast_path, approval_mode

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

    return prefer_fast_path, approval_mode


def _finalize_raw_destination(tmp_path: Path, raw_dest: Path) -> None:
    if not raw_dest.is_file():
        os.replace(tmp_path, raw_dest)
    else:
        tmp_path.unlink(missing_ok=True)


async def handle_document_uploads(
    *,
    request: Request,
    upload_list: list[UploadFile],
    prefer_fast_path: bool | None,
    approval_mode: Literal["review_all", "auto_high_confidence", "auto_all"] | None,
    session: AsyncSession,
    orchestrator: IngestionOrchestrator,
    settings: Settings,
    get_doc_by_sha_fn: Callable[[AsyncSession, str], Awaitable[Document | None]] = get_document_by_sha,
    schedule_fn: Callable[..., asyncio.Task[None]],
) -> tuple[DocumentResponse, int]:
    """Process one or more uploaded documents with SHA-256 deduplication and mode upgrades."""
    prefer_fast_path, approval_mode = await resolve_upload_defaults(
        session, prefer_fast_path, approval_mode
    )

    raw_dir = settings.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    max_upload_mb = settings.ingestion.max_upload_size_mb
    max_bytes = max_upload_mb * 1024 * 1024

    results: list[DocumentResponse] = []

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

        sha256, size_bytes, tmp_path = await stream_upload_to_temp(
            upload=upload,
            raw_dir=raw_dir,
            max_bytes=max_bytes,
            max_upload_size_mb=max_upload_mb,
        )
        raw_dest = raw_dir / f"{sha256}.{ext}"

        existing = await get_doc_by_sha_fn(session, sha256)
        if existing is not None and existing.status != "failed":
            is_mode_upgrade = is_engine_mode_upgrade(existing.parsing_engine, prefer_fast_path)
            is_awaiting_reenter = (
                existing.status == "awaiting_approval"
                and approval_mode in ("auto_all", "auto_high_confidence")
            )

            _finalize_raw_destination(tmp_path, raw_dest)

            if not is_mode_upgrade and not is_awaiting_reenter:
                latest_job = await get_latest_job_for_document(session, existing.id)
                results.append(build_document_response(existing, latest_job, duplicate=True))
                continue

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
                await session.refresh(existing)
                latest_job = await get_latest_job_for_document(session, existing.id)
                results.append(build_document_response(existing, latest_job, duplicate=True))
                continue

            await session.refresh(existing)
            job = await create_queued_ingestion_job(session, existing.id, now=now)
            schedule_fn(
                request=request,
                orchestrator=orchestrator,
                doc_id=existing.id,
                job_id=job.id,
                raw_path=raw_dest,
                force_reparse=True,
                prefer_pymupdf4llm=prefer_fast_path,
                approval_mode=approval_mode,
            )
            results.append(build_document_response(existing, job, duplicate=False))
            continue

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
                existing_raced = await get_doc_by_sha_fn(session, sha256)
                if existing_raced is not None:
                    latest_job = await get_latest_job_for_document(session, existing_raced.id)
                    results.append(
                        build_document_response(existing_raced, latest_job, duplicate=True)
                    )
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

        job = await create_queued_ingestion_job(session, doc.id, now=now)
        schedule_fn(
            request=request,
            orchestrator=orchestrator,
            doc_id=doc.id,
            job_id=job.id,
            raw_path=raw_dest,
            prefer_pymupdf4llm=prefer_fast_path,
            approval_mode=approval_mode,
        )
        results.append(build_document_response(doc, job, duplicate=False))

    all_duplicates = all(r.duplicate for r in results)
    status_code = status.HTTP_200_OK if all_duplicates else status.HTTP_202_ACCEPTED
    primary = results[0]
    primary.documents = [r.model_copy(update={"documents": None}) for r in results]
    return primary, status_code


async def retry_document_ingestion(
    *,
    doc_id: str,
    request: Request,
    force_reparse: bool,
    prefer_fast_path: bool | None,
    session: AsyncSession,
    orchestrator: IngestionOrchestrator,
    settings: Settings,
    schedule_fn: Callable[..., asyncio.Task[None]],
) -> DocumentResponse:
    """Retry ingestion for a failed, cancelled, or ready document with atomic CAS state guard."""
    doc = await _require_document(session, doc_id)

    if doc.status == "ready" and not force_reparse:
        latest_job = await get_latest_job_for_document(session, doc.id)
        return build_document_response(doc, latest_job)

    raw_dest = resolve_document_artifact_path(doc, settings.data_dir, "raw")
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

    job = await create_queued_ingestion_job(session, doc.id, now=now)
    schedule_fn(
        request=request,
        orchestrator=orchestrator,
        doc_id=doc.id,
        job_id=job.id,
        raw_path=raw_dest,
        force_reparse=force_reparse,
        prefer_pymupdf4llm=prefer_fast_path,
    )
    return build_document_response(doc, job)


async def approve_document_classification(
    *,
    doc_id: str,
    payload: ApprovedClassificationRequest | None,
    orchestrator: IngestionOrchestrator,
    session: AsyncSession,
) -> DocumentResponse:
    """Approve classification and resume chunking & indexing from parsed cache."""
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

    latest_job = await get_latest_job_for_document(session, doc.id)
    job_id = latest_job.id if latest_job else None
    completed_doc = await orchestrator.complete_approved_ingestion(
        doc_id=doc.id,
        job_id=job_id,
        approved_profile=payload,
    )
    latest_job = await get_latest_job_for_document(session, doc.id)
    return build_document_response(completed_doc, latest_job)


async def approve_documents_batch(
    *,
    doc_ids: list[str],
    orchestrator: IngestionOrchestrator,
    session: AsyncSession,
) -> ApproveBatchResponse:
    """Batch-approve all specified documents currently in `awaiting_approval` state."""
    from src.api.schemas import ApproveBatchResponse

    approved_docs: list[DocumentResponse] = []
    for d_id in doc_ids:
        doc = await session.get(Document, d_id)
        if doc is None or doc.status != "awaiting_approval":
            continue
        latest_job = await get_latest_job_for_document(session, doc.id)
        job_id = latest_job.id if latest_job else None
        completed_doc = await orchestrator.complete_approved_ingestion(
            doc_id=doc.id,
            job_id=job_id,
        )
        latest_job = await get_latest_job_for_document(session, doc.id)
        approved_docs.append(build_document_response(completed_doc, latest_job))

    return ApproveBatchResponse(
        approved_count=len(approved_docs),
        documents=approved_docs,
    )
