"""Tracked background task scheduling and unified corpus/domain re-indexing service."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Coroutine, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings
from src.db.models import Document, IngestionJob
from src.ingestion.parser import load_parsed_checkpoint
from src.ingestion.worker import IngestionOrchestrator

logger = logging.getLogger(__name__)


def schedule_background_task(
    app_state: Any,
    coro: Coroutine[Any, Any, Any],
) -> asyncio.Task[Any]:
    """Create an asyncio task and track it on `app_state.background_tasks` for graceful shutdown."""
    task = asyncio.create_task(coro)
    if app_state is not None:
        bg_tasks: set[asyncio.Task[Any]] | None = getattr(app_state, "background_tasks", None)
        if bg_tasks is None:
            bg_tasks = set()
            app_state.background_tasks = bg_tasks
        bg_tasks.add(task)
        task.add_done_callback(bg_tasks.discard)
    return task


async def _run_reindex_job(
    orchestrator: IngestionOrchestrator,
    *,
    doc_id: str,
    job_id: str,
    raw_path: Path | None,
    has_checkpoint: bool,
) -> None:
    """Execute background re-chunking/re-indexing for a single document with exception logging."""
    try:
        if has_checkpoint:
            await orchestrator.complete_approved_ingestion(doc_id=doc_id, job_id=job_id)
        elif raw_path is not None:
            await orchestrator.process_document(
                doc_id=doc_id,
                job_id=job_id,
                raw_path=raw_path,
                force_reparse=False,
                approval_mode="auto_all",
            )
    except Exception:
        logger.exception("Background re-index task failed for doc_id=%s job_id=%s", doc_id, job_id)


async def find_reindexable_doc_ids(
    session: AsyncSession,
    *,
    domain_id: str,
    data_dir: Path,
) -> list[str]:
    """Return IDs of ready documents in `domain_id` that have a cached parsed checkpoint on disk."""
    res = await session.execute(
        select(Document).where(
            Document.primary_domain_id == domain_id,
            Document.status == "ready",
        )
    )
    eligible: list[str] = []
    for doc in res.scalars().all():
        cached = load_parsed_checkpoint(
            data_dir=data_dir,
            sha=doc.sha256,
            filename=doc.filename,
            file_type=(doc.file_type or "").lstrip("."),
        )
        if cached is not None:
            eligible.append(doc.id)
    return eligible


async def queue_documents_for_reindex(
    *,
    app_state: Any,
    session: AsyncSession,
    orchestrator: IngestionOrchestrator,
    settings: Settings,
    doc_ids: Sequence[str] | None = None,
    domain_id: str | None = None,
    allow_raw_fallback: bool = True,
) -> tuple[int, int]:
    """Queue ready documents for tracked background re-indexing.

    Returns `(queued_count, skipped_count)`.
    """
    if doc_ids is not None:
        unique_ids = list(dict.fromkeys(doc_ids))
        if not unique_ids:
            return 0, 0
        stmt = select(Document).where(
            Document.id.in_(unique_ids),
            Document.status == "ready",
        )
    elif domain_id is not None:
        stmt = select(Document).where(
            Document.primary_domain_id == domain_id,
            Document.status == "ready",
        )
    else:
        stmt = select(Document).where(Document.status == "ready")

    ready_docs = (await session.execute(stmt)).scalars().all()

    queued_count = 0
    skipped_count = 0

    for doc in ready_docs:
        ext = (doc.file_type or "").lower().lstrip(".")
        sha = (doc.sha256 or "").lower()
        cached = load_parsed_checkpoint(
            data_dir=settings.data_dir,
            sha=doc.sha256,
            filename=doc.filename,
            file_type=ext,
        )
        raw_dest = settings.data_dir / "raw" / f"{sha}.{ext}" if sha and ext else None
        has_raw = allow_raw_fallback and raw_dest is not None and raw_dest.is_file()

        if cached is None and not has_raw:
            skipped_count += 1
            continue

        now = datetime.now(timezone.utc)
        doc.status = "queued"
        doc.error = None
        doc.updated_at = now
        session.add(doc)

        new_job = IngestionJob(
            doc_id=doc.id,
            status="queued",
            stage="queued",
            progress=0.45 if cached is not None else 0.0,
            stage_timings=json.dumps({}),
            created_at=now,
            updated_at=now,
        )
        session.add(new_job)
        await session.commit()
        await session.refresh(new_job)

        schedule_background_task(
            app_state,
            _run_reindex_job(
                orchestrator,
                doc_id=doc.id,
                job_id=new_job.id,
                raw_path=raw_dest if has_raw else None,
                has_checkpoint=cached is not None,
            ),
        )
        queued_count += 1

    return queued_count, skipped_count
