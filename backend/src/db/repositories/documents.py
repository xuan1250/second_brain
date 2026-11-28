"""Document repository queries, cross-store metadata synchronization, and 3-tier atomic deletion."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.db.models import (
    Document,
    DocumentChunk,
    DomainRegistry,
    IngestionJob,
    SubdomainRegistry,
    _now,
)
from src.db.repositories.common import is_valid_sha256_hex


class DocumentCancellable(Protocol):
    """Protocol for components that can cancel in-flight document processing."""

    def cancel_document(self, doc_id: str) -> None:
        ...


async def get_document_by_sha(session: AsyncSession, sha256: str) -> Document | None:
    """Fetch a Document by its SHA-256 hash."""
    res = await session.execute(select(Document).where(Document.sha256 == sha256))
    return res.scalars().first()


async def get_latest_job_for_document(session: AsyncSession, doc_id: str) -> IngestionJob | None:
    """Fetch the most recent IngestionJob for a single document."""
    res = await session.execute(
        select(IngestionJob)
        .where(IngestionJob.doc_id == doc_id)
        .order_by(IngestionJob.created_at.desc())
    )
    return res.scalars().first()


async def get_latest_jobs_for_documents(
    session: AsyncSession,
    doc_ids: Sequence[str],
) -> dict[str, IngestionJob]:
    """Batch-fetch the latest IngestionJob per document ID."""
    ids = list(dict.fromkeys(doc_ids))
    if not ids:
        return {}
    job_stmt = (
        select(IngestionJob)
        .where(IngestionJob.doc_id.in_(ids))
        .order_by(IngestionJob.created_at.desc())
    )
    job_rows = (await session.execute(job_stmt)).scalars().all()
    latest_jobs: dict[str, IngestionJob] = {}
    for job in job_rows:
        if job.doc_id not in latest_jobs:
            latest_jobs[job.doc_id] = job
    return latest_jobs


async def list_documents_paginated(
    session: AsyncSession,
    *,
    status_filter: str | None = None,
    file_type: str | None = None,
    search: str | None = None,
    domain_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[Document], int]:
    """Return paginated Document rows matching optional filters along with total count."""
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

    if domain_id and domain_id.strip():
        clean_domain_id = domain_id.strip()
        stmt = stmt.where(Document.primary_domain_id == clean_domain_id)
        count_stmt = count_stmt.where(Document.primary_domain_id == clean_domain_id)

    total = int((await session.execute(count_stmt)).scalar() or 0)
    rows = list(
        (
            await session.execute(
                stmt.order_by(Document.created_at.desc()).offset(offset).limit(limit)
            )
        ).scalars().all()
    )
    return rows, total


async def delete_document_atomic(
    session: AsyncSession,
    qdrant: AsyncQdrantClient,
    doc_id: str,
    data_dir: Path | None = None,
    collection: str | None = None,
    orchestrator: DocumentCancellable | None = None,
) -> bool:
    """Delete a document across all tiers. Returns False if it does not exist.

    Order: In-flight cancellation signal -> Qdrant purge first (a failure aborts
    before touching SQLite, so the document stays retryable), then SQLite cascade,
    then artifact cleanup.
    """
    if orchestrator is not None:
        orchestrator.cancel_document(doc_id)

    settings = get_settings()
    collection = collection or settings.qdrant_collection
    data_dir = Path(data_dir) if data_dir else settings.data_dir

    doc = await session.get(Document, doc_id)
    if doc is None:
        return False
    sha, ext = doc.sha256, doc.file_type.lstrip(".")

    # Tier 1: vectors
    await qdrant.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            )
        ),
        wait=True,
    )
    # Tier 2: relational (FK ON DELETE CASCADE removes chunks/jobs)
    await session.delete(doc)
    await session.commit()

    # Tier 3: artifacts (sha256-derived names only; no user-controlled paths)
    if is_valid_sha256_hex(sha):
        for sub, suffix in (("raw", f".{ext}"), ("parsed", ".json"), ("markdown", ".md")):
            (data_dir / sub / f"{sha}{suffix}").unlink(missing_ok=True)
    return True


async def sync_document_metadata_across_stores(
    session: AsyncSession,
    qdrant: AsyncQdrantClient | None,
    collection_name: str | None,
    doc: Document,
) -> None:
    """Synchronize document metadata, chunks, and Qdrant points across stores.

    Order: SQLite updates -> session.flush() -> Qdrant set_payload(wait=True) -> session.commit().
    """
    primary_domain_slug = "general"
    primary_subdomain_slug = "general.general"
    primary_epistemic = "general"

    if doc.primary_domain_id:
        d_obj = await session.get(DomainRegistry, doc.primary_domain_id)
        if d_obj:
            primary_domain_slug = d_obj.slug
            primary_epistemic = d_obj.epistemic_type
    if doc.primary_subdomain_id:
        s_obj = await session.get(SubdomainRegistry, doc.primary_subdomain_id)
        if s_obj:
            primary_subdomain_slug = s_obj.slug

    # Parse and synchronize doc.classification_result
    parsed_profile: dict[str, Any] = {}
    if doc.classification_result:
        try:
            parsed_profile = json.loads(doc.classification_result)
        except Exception:
            parsed_profile = {}

    parsed_profile["primary_domain"] = primary_domain_slug
    parsed_profile["primary_subdomain"] = primary_subdomain_slug
    parsed_profile["epistemic_modality"] = primary_epistemic
    parsed_profile["authority_tier"] = doc.authority_tier
    parsed_profile["jurisdiction"] = doc.jurisdiction
    parsed_profile["effective_year"] = doc.effective_year
    parsed_profile["is_current"] = doc.is_current

    section_overrides = parsed_profile.get("section_overrides") or []
    doc.classification_result = json.dumps(parsed_profile)
    doc.updated_at = _now()
    session.add(doc)

    # Fetch all chunks for this document
    chunks_res = await session.execute(
        select(DocumentChunk).where(DocumentChunk.doc_id == doc.id)
    )
    chunks = list(chunks_res.scalars().all())

    # Map chunks to effective domain, subdomain, epistemic_modality
    grouped_points: dict[tuple[str, str, str], list[str]] = {}

    for chunk in chunks:
        matched_override = None
        if chunk.heading_path and section_overrides:
            h_clean = chunk.heading_path.strip().lower()
            for ov in section_overrides:
                ov_h = (
                    ov.get("heading_path") if isinstance(ov, dict)
                    else getattr(ov, "heading_path", "")
                ).strip().lower()
                if ov_h and (ov_h == h_clean or ov_h in h_clean or h_clean in ov_h):
                    matched_override = ov
                    break

        if matched_override:
            c_domain = matched_override.get("domain") if isinstance(matched_override, dict) else matched_override.domain
            c_subdomain = matched_override.get("subdomain") if isinstance(matched_override, dict) else matched_override.subdomain
            c_epistemic = (
                matched_override.get("epistemic_modality") if isinstance(matched_override, dict)
                else getattr(matched_override, "epistemic_modality", None)
            ) or "general"
        else:
            c_domain = primary_domain_slug
            c_subdomain = primary_subdomain_slug
            c_epistemic = primary_epistemic

        chunk.domain_key = c_domain
        chunk.subdomain_key = c_subdomain
        chunk.epistemic_modality = c_epistemic
        session.add(chunk)

        if chunk.point_id:
            key = (c_domain, c_subdomain, c_epistemic)
            grouped_points.setdefault(key, []).append(chunk.point_id)

    # Invariant: session.flush() -> qdrant.set_payload(..., wait=True) -> session.commit()
    await session.flush()

    if qdrant is not None and collection_name:
        for (g_domain, g_subdomain, g_epistemic), point_ids in grouped_points.items():
            if not point_ids:
                continue
            payload_to_set = {
                "domain": g_domain,
                "subdomain": g_subdomain,
                "epistemic_type": g_epistemic,
                "authority_tier": int(doc.authority_tier),
                "jurisdiction": doc.jurisdiction,
                "effective_year": doc.effective_year,
                "is_current": bool(doc.is_current),
            }
            await qdrant.set_payload(
                collection_name=collection_name,
                payload=payload_to_set,
                points=models.PointIdsList(points=point_ids),
                wait=True,
            )

    await session.commit()
    await session.refresh(doc)
