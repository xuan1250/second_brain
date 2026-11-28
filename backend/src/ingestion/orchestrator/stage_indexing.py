"""Stage 2 (chunking) and Stage 3 (indexing) orchestrator runners."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Optional

from src.db.models import Document
from src.ingestion.classifier import (
    RecommendedDomainProfile,
    get_taxonomy_catalog,
    refresh_taxonomy_catalog,
)
from src.ingestion.constants import normalize_chunking_strategy
from src.ingestion.indexer import HybridIndexer
from src.ingestion.orchestrator.db_state import (
    _now,
    apply_approved_profile_overrides,
    get_or_create_latest_job,
)
from src.ingestion.parser import load_parsed_checkpoint

logger = logging.getLogger("src.ingestion.worker")


async def run_complete_approved_ingestion(
    orchestrator: Any,
    doc_id: str,
    job_id: str | None = None,
    *,
    approved_profile: Optional[Any] = None,
) -> Document:
    """Resume Stage 2 (chunking) and Stage 3 (indexing) from cached data/parsed/{sha}.json."""
    worker_mod = sys.modules["src.ingestion.worker"]
    cancel_event = orchestrator._get_or_create_cancel_event(doc_id)
    orchestrator._queued_tasks += 1

    if cancel_event.is_set():
        logger.info("Document %s was cancelled before acquiring semaphore", doc_id)
        async with orchestrator.session_factory() as session:
            doc = await session.get(Document, doc_id)
            if doc:
                doc.status = "failed"
                doc.error = "Ingestion cancelled"
                doc.updated_at = _now()
                session.add(doc)
                await session.commit()
                await session.refresh(doc)
                return doc
        return Document(id=doc_id, status="failed", error="Ingestion cancelled")

    try:
        if orchestrator._is_paused:
            await orchestrator._resume_event.wait()

        async with orchestrator._doc_semaphore:
            if orchestrator._is_paused:
                await orchestrator._resume_event.wait()

            timings: dict[str, int] = {}
            t_start = time.perf_counter()

            if (
                c_doc := await orchestrator._run_cancel_checkpoint(
                    cancel_event, doc_id, job_id or "", 0.45, "chunking", timings
                )
            ) is not None:
                return c_doc

            async with orchestrator.session_factory() as session:
                try:
                    await refresh_taxonomy_catalog(session)
                except Exception as exc:
                    logger.debug("Failed refreshing taxonomy catalog in complete_approved_ingestion: %s", exc)
                doc = await session.get(Document, doc_id)
                if doc is None:
                    raise ValueError(f"Document {doc_id} not found")
                latest_job = await get_or_create_latest_job(
                    session, doc_id, job_id=job_id, default_progress=0.45
                )
                job_id = latest_job.id

                if latest_job.stage_timings:
                    try:
                        parsed_t = json.loads(latest_job.stage_timings)
                        if isinstance(parsed_t, dict):
                            timings = {k: int(v) for k, v in parsed_t.items()}
                    except Exception:
                        timings = {}

                sha = doc.sha256
                ext = doc.file_type.lstrip(".")
                orig_filename = doc.filename

                if approved_profile is not None:
                    await apply_approved_profile_overrides(session, doc, approved_profile)

            cached = orchestrator._parsed_cache.pop(doc_id, None) or load_parsed_checkpoint(
                orchestrator.data_dir, sha, orig_filename, ext
            )
            if cached is None:
                raw_path = orchestrator.raw_dir / f"{sha}.{ext}"
                parse_task_fn = getattr(worker_mod, "_worker_parse_task")
                cached, p_ms = await asyncio.to_thread(
                    parse_task_fn,
                    str(raw_path),
                    str(orchestrator.data_dir),
                    sha,
                    orig_filename,
                    False,
                    orchestrator.prefer_pymupdf4llm,
                    orchestrator.excel_window_rows,
                    orchestrator.max_upload_size_mb,
                    orchestrator.device,
                )
                timings["parsing_ms"] = p_ms

            profile_obj: Optional[RecommendedDomainProfile] = None
            if doc.classification_result:
                try:
                    profile_obj = RecommendedDomainProfile.model_validate_json(doc.classification_result)
                except Exception:
                    profile_obj = None

            if profile_obj is not None:
                catalog = get_taxonomy_catalog()
                profile_obj.chunking_strategy = normalize_chunking_strategy(
                    getattr(profile_obj, "chunking_strategy", None),
                    catalog.get_chunking_strategy(profile_obj.primary_domain),
                )
                for sec_ov in (profile_obj.section_overrides or []):
                    sec_ov.chunking_strategy = normalize_chunking_strategy(
                        getattr(sec_ov, "chunking_strategy", None),
                        catalog.get_chunking_strategy(sec_ov.domain),
                    )

            try:
                if (
                    c_doc := await orchestrator._run_cancel_checkpoint(
                        cancel_event, doc_id, job_id, 0.55, "chunking", timings
                    )
                ) is not None:
                    return c_doc

                orchestrator._acquire_worker_slot()
                try:
                    await orchestrator._update_state(
                        doc_id,
                        job_id,
                        status="chunking",
                        stage="chunking",
                        progress=0.55,
                        timings=timings,
                        page_count=cached.page_count,
                        parsing_engine=cached.extra_meta.get("engine") if cached else None,
                        engine_fallback_reason=(
                            cached.extra_meta.get("engine_fallback_reason") if cached else None
                        ),
                    )
                    chunk_task_fn = getattr(worker_mod, "_worker_chunk_task")
                    loop = asyncio.get_running_loop()
                    if orchestrator.use_process_pool:
                        try:
                            chunks, chunk_ms = await loop.run_in_executor(
                                orchestrator.get_pool(),
                                chunk_task_fn,
                                cached,
                                orchestrator.child_max_tokens,
                                orchestrator.parent_max_tokens,
                                profile_obj,
                            )
                        except BrokenProcessPool:
                            return await orchestrator._handle_broken_pool(
                                doc_id, job_id, "chunking", t_start, timings
                            )
                    else:
                        chunks, chunk_ms = await asyncio.to_thread(
                            chunk_task_fn,
                            cached,
                            orchestrator.child_max_tokens,
                            orchestrator.parent_max_tokens,
                            profile_obj,
                        )
                    timings["chunking_ms"] = chunk_ms

                    if orchestrator.use_process_pool:
                        orchestrator._pool_completed_docs += 1
                finally:
                    orchestrator._release_worker_slot()

                if (
                    c_doc := await orchestrator._run_cancel_checkpoint(
                        cancel_event, doc_id, job_id, 0.8, "indexing", timings
                    )
                ) is not None:
                    return c_doc

                await orchestrator._update_state(
                    doc_id,
                    job_id,
                    status="indexing",
                    stage="indexing",
                    progress=0.8,
                    timings=timings,
                )
                t_index = time.perf_counter()
                if orchestrator.qdrant is None:
                    from src.db.qdrant import get_qdrant_client

                    qdrant_client = get_qdrant_client()
                    close_qdrant = True
                else:
                    qdrant_client = orchestrator.qdrant
                    close_qdrant = False

                orchestrator._active_indexing_jobs += 1
                try:
                    try:
                        indexer = HybridIndexer(
                            qdrant=qdrant_client,
                            embedder=orchestrator.embedder,
                            sparse_encoder=orchestrator.sparse_encoder,
                            collection=orchestrator.collection,
                            clear_cuda_cache=False,
                        )
                        async with orchestrator.session_factory() as session:
                            indexed_count = await indexer.index_chunks(session, doc_id=doc_id, chunks=chunks)
                    finally:
                        if close_qdrant:
                            await qdrant_client.close()
                finally:
                    orchestrator._active_indexing_jobs = max(0, orchestrator._active_indexing_jobs - 1)
                    if orchestrator.clear_cuda_cache and orchestrator._active_indexing_jobs == 0:
                        release_fn = getattr(worker_mod, "release_cuda_cache")
                        release_fn()

                timings["indexing_ms"] = max(1, int((time.perf_counter() - t_index) * 1000))
                total_ms = max(1, int((time.perf_counter() - t_start) * 1000))
                timings["total_ms"] = total_ms

                await orchestrator._update_state(
                    doc_id,
                    job_id,
                    status="ready",
                    stage="ready",
                    progress=1.0,
                    timings=timings,
                    duration_ms=total_ms,
                    chunk_count=indexed_count,
                    page_count=cached.page_count,
                )

                async with orchestrator.session_factory() as session:
                    final_doc = await session.get(Document, doc_id)
                    assert final_doc is not None
                    return final_doc

            except Exception as exc:
                logger.exception("complete_approved_ingestion failed for doc %s", doc_id)
                try:
                    await orchestrator._update_state(
                        doc_id,
                        job_id,
                        status="failed",
                        stage="failed",
                        progress=0.0,
                        timings=timings,
                        error=str(exc),
                    )
                except Exception:
                    pass
                raise

    finally:
        orchestrator._queued_tasks = max(0, orchestrator._queued_tasks - 1)
        orchestrator._cancellations.pop(doc_id, None)


async def run_process_document(
    orchestrator: Any,
    doc_id: str,
    job_id: str | None = None,
    raw_path: Path | None = None,
    *,
    force_reparse: bool = False,
    prefer_pymupdf4llm: bool | None = None,
    approval_mode: str = "auto_all",
) -> Document:
    """Backward-compatible wrapper executing parsing, classification, and auto-completing indexing when auto-approved."""
    orchestrator._acquire_worker_slot()
    try:
        doc, auto_approved = await orchestrator.process_to_classification(
            doc_id=doc_id,
            job_id=job_id,
            raw_path=raw_path,
            force_reparse=force_reparse,
            prefer_pymupdf4llm=prefer_pymupdf4llm,
            approval_mode=approval_mode,
        )
        if auto_approved:
            return await orchestrator.complete_approved_ingestion(doc_id=doc_id, job_id=job_id)
        return doc
    finally:
        orchestrator._release_worker_slot()
