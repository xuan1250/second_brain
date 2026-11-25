"""Shared document response serializers, filename sanitization, and artifact path helpers."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Document, IngestionJob
from src.db.repository import is_valid_sha256_hex

if TYPE_CHECKING:
    from src.api.schemas import DocumentResponse, HeadingOutlineItem

MEDIA_TYPES: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def sanitize_filename(filename: str) -> str:
    """Sanitize a user-supplied filename for safe storage metadata and Content-Disposition headers."""
    base = Path(filename or "document").name
    cleaned = re.sub(r'[\x00-\x1f\x7f"\\/:<>|?*]+', "_", base).strip()
    return cleaned or "document"


def extract_markdown_headings(markdown_text: str) -> list[HeadingOutlineItem]:
    """Extract hierarchical heading outline (`level`, `text`, `heading_path`, `line`) from Markdown."""
    from src.api.schemas import HeadingOutlineItem

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


def build_document_response(
    doc: Document,
    job: IngestionJob | None = None,
    *,
    duplicate: bool = False,
) -> DocumentResponse:
    """Serialize a Document and optional IngestionJob without touching disk artifacts."""
    from src.api.schemas import DocumentResponse

    timings: dict[str, int] = {}
    if job is not None and job.stage_timings:
        try:
            parsed = json.loads(job.stage_timings)
            if isinstance(parsed, dict):
                timings = {
                    str(k): int(v)
                    for k, v in parsed.items()
                    if isinstance(v, (int, float))
                }
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


def resolve_document_artifact_path(
    doc: Document,
    data_dir: Path,
    kind: str,
) -> Path:
    """Validate SHA-256 storage reference and verify that the requested artifact exists on disk."""
    sha = (doc.sha256 or "").lower()
    if kind == "raw":
        ext = (doc.file_type or "").lower().lstrip(".")
        if not is_valid_sha256_hex(sha) or not ext:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invalid document storage reference.",
            )
        artifact_path = data_dir / "raw" / f"{sha}.{ext}"
        if not artifact_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Raw file for document '{doc.id}' not found on disk.",
            )
        return artifact_path

    if not is_valid_sha256_hex(sha):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid document storage reference.",
        )
    artifact_path = data_dir / "markdown" / f"{sha}.md"
    if not artifact_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Markdown mirror for document '{doc.id}' is not available yet.",
        )
    return artifact_path


async def _require_document(session: AsyncSession, doc_id: str) -> Document:
    doc = await session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )
    return doc
