"""SQLite state helpers for ingestion jobs, domain resolution, and approval profile overrides."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.db.models import Document, DomainRegistry, IngestionJob, SubdomainRegistry
from src.ingestion.constants import DEFAULT_DOMAIN


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _extract_sections_from_markdown(markdown: str) -> list[dict[str, str]]:
    """Extract section headings and body text from markdown for classification."""
    if not markdown:
        return []
    lines = markdown.splitlines()
    sections: list[dict[str, str]] = []
    current_heading = "Introduction"
    current_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("#"):
            if current_lines:
                sec_text = "\n".join(current_lines).strip()
                if sec_text:
                    sections.append({"heading_path": current_heading, "text": sec_text})
                current_lines = []
            current_heading = line.lstrip("#").strip()
        else:
            current_lines.append(line)
    if current_lines:
        sec_text = "\n".join(current_lines).strip()
        if sec_text:
            sections.append({"heading_path": current_heading, "text": sec_text})
    return sections


async def get_or_create_latest_job(
    session: AsyncSession,
    doc_id: str,
    *,
    job_id: str | None = None,
    default_progress: float = 0.0,
) -> IngestionJob:
    """Fetch explicit job_id or latest IngestionJob for doc_id, creating one if absent."""
    if job_id is not None:
        job = await session.get(IngestionJob, job_id) if job_id else None
        if job is not None:
            return job
    res = await session.execute(
        select(IngestionJob)
        .where(IngestionJob.doc_id == doc_id)
        .order_by(IngestionJob.created_at.desc())
    )
    latest_job = res.scalars().first()
    if latest_job is None:
        latest_job = IngestionJob(
            doc_id=doc_id,
            status="queued",
            stage="queued",
            progress=default_progress,
        )
        session.add(latest_job)
        await session.commit()
        await session.refresh(latest_job)
    return latest_job


async def resolve_domain_and_subdomain(
    session: AsyncSession,
    domain_slug: str | None,
    subdomain_slug: str | None,
    *,
    fallback_domain_id: str | None = None,
) -> tuple[DomainRegistry | None, SubdomainRegistry | None]:
    """Resolve DomainRegistry and SubdomainRegistry rows with safe general/first-child fallbacks."""
    d_obj: DomainRegistry | None = None
    if domain_slug:
        res_d = await session.execute(
            select(DomainRegistry).where(DomainRegistry.slug == domain_slug)
        )
        d_obj = res_d.scalars().first()
    elif fallback_domain_id:
        d_obj = await session.get(DomainRegistry, fallback_domain_id)

    if d_obj is None:
        res_d = await session.execute(
            select(DomainRegistry).where(DomainRegistry.slug == DEFAULT_DOMAIN)
        )
        d_obj = res_d.scalars().first()

    if d_obj is None:
        return None, None

    s_obj: SubdomainRegistry | None = None
    if subdomain_slug:
        res_s = await session.execute(
            select(SubdomainRegistry).where(
                SubdomainRegistry.slug == subdomain_slug,
                SubdomainRegistry.domain_id == d_obj.id,
            )
        )
        s_obj = res_s.scalars().first()

    if s_obj is None:
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

    return d_obj, s_obj


async def apply_approved_profile_overrides(
    session: AsyncSession,
    doc: Document,
    approved_profile: Any,
) -> None:
    """Persist human-approved classification overrides onto Document and classification_result JSON."""
    if hasattr(approved_profile, "model_dump"):
        prof_data = approved_profile.model_dump(exclude_unset=True)
    elif isinstance(approved_profile, dict):
        prof_data = approved_profile
    else:
        prof_data = {}

    d_obj, s_obj = await resolve_domain_and_subdomain(
        session,
        prof_data.get("domain"),
        prof_data.get("subdomain"),
        fallback_domain_id=doc.primary_domain_id,
    )
    if d_obj:
        doc.primary_domain_id = d_obj.id
    if s_obj:
        doc.primary_subdomain_id = s_obj.id

    for field_name in ("authority_tier", "jurisdiction", "effective_year", "is_current"):
        if field_name in prof_data and prof_data[field_name] is not None:
            setattr(doc, field_name, prof_data[field_name])

    parsed_prof: dict[str, Any] = {}
    if doc.classification_result:
        try:
            parsed_prof = json.loads(doc.classification_result)
        except Exception:
            parsed_prof = {}

    if d_obj:
        parsed_prof["primary_domain"] = d_obj.slug
        parsed_prof["epistemic_modality"] = d_obj.epistemic_type
        if getattr(d_obj, "chunking_strategy", None):
            parsed_prof["chunking_strategy"] = d_obj.chunking_strategy
    if s_obj:
        parsed_prof["primary_subdomain"] = s_obj.slug

    for field_name in ("authority_tier", "jurisdiction", "effective_year", "is_current"):
        if field_name in prof_data and prof_data[field_name] is not None:
            parsed_prof[field_name] = prof_data[field_name]

    if "section_overrides" in prof_data and prof_data["section_overrides"] is not None:
        parsed_prof["section_overrides"] = prof_data["section_overrides"]

    doc.classification_result = json.dumps(parsed_prof)
    doc.updated_at = _now()
    session.add(doc)
    await session.commit()
    await session.refresh(doc)


async def update_ingestion_state(
    session_factory: async_sessionmaker[AsyncSession],
    doc_id: str,
    job_id: str,
    *,
    status: str,
    stage: str,
    progress: float,
    timings: dict[str, int],
    duration_ms: int | None = None,
    chunk_count: int | None = None,
    page_count: int | None = None,
    parsing_engine: str | None = None,
    engine_fallback_reason: str | None = None,
    primary_domain_id: str | None = None,
    primary_subdomain_id: str | None = None,
    authority_tier: int | None = None,
    jurisdiction: str | None = None,
    effective_year: int | None = None,
    is_current: bool | None = None,
    classification_result: str | None = None,
    error: str | None = None,
) -> None:
    """Persist Document and IngestionJob stage transitions transactionally."""
    now = _now()
    async with session_factory() as session:
        doc = await session.get(Document, doc_id)
        if doc is not None:
            doc.status = status
            doc.updated_at = now
            if chunk_count is not None:
                doc.chunk_count = chunk_count
            if page_count is not None:
                doc.page_count = page_count
            if error is not None:
                doc.error = error
            if parsing_engine is not None:
                doc.parsing_engine = parsing_engine
                doc.engine_fallback_reason = engine_fallback_reason
            if primary_domain_id is not None:
                doc.primary_domain_id = primary_domain_id
            if primary_subdomain_id is not None:
                doc.primary_subdomain_id = primary_subdomain_id
            if authority_tier is not None:
                doc.authority_tier = authority_tier
            if jurisdiction is not None:
                doc.jurisdiction = jurisdiction
            if effective_year is not None:
                doc.effective_year = effective_year
            if is_current is not None:
                doc.is_current = is_current
            if classification_result is not None:
                doc.classification_result = classification_result
            session.add(doc)

        job = await session.get(IngestionJob, job_id) if job_id else None
        if job is not None:
            job.status = status
            job.stage = stage
            job.progress = progress
            job.stage_timings = json.dumps(timings)
            if duration_ms is not None:
                job.duration_ms = duration_ms
            if error is not None:
                job.error = error
            job.updated_at = now
            session.add(job)

        await session.commit()
