"""Stage 1 (parsing) and Stage 1b (classification) orchestrator runner."""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.db.models import Document
from src.ingestion.classifier import get_classifier, refresh_taxonomy_catalog
from src.ingestion.constants import DEFAULT_DOMAIN, is_cache_engine_mismatch
from src.ingestion.orchestrator.db_state import (
    _extract_sections_from_markdown,
    get_or_create_latest_job,
    resolve_domain_and_subdomain,
)
from src.ingestion.parser import ParseResult, load_parsed_checkpoint

logger = logging.getLogger("src.ingestion.worker")


async def run_process_to_classification(
    orchestrator: Any,
    doc_id: str,
    job_id: str | None = None,
    raw_path: Path | None = None,
    *,
    force_reparse: bool = False,
    prefer_pymupdf4llm: bool | None = None,
    approval_mode: str = "auto_all",
) -> tuple[Document, bool]:
    """Execute Stage 1 (parsing) and Stage 1b (classifying), pausing at awaiting_approval if not auto-approved."""
    worker_mod = sys.modules["src.ingestion.worker"]
    cancel_event = orchestrator._get_or_create_cancel_event(doc_id)
    orchestrator._queued_tasks += 1

    try:
        if orchestrator._is_paused:
            logger.info("Doc %s waiting at semaphore barrier due to ingestion pause", doc_id)
            await orchestrator._resume_event.wait()

        async with orchestrator._doc_semaphore:
            if orchestrator._is_paused:
                logger.info("Doc %s waiting inside semaphore barrier due to ingestion pause", doc_id)
                await orchestrator._resume_event.wait()

            timings: dict[str, int] = {}
            t_start = time.perf_counter()
            active_job_id = job_id or ""

            try:
                async with orchestrator.session_factory() as session:
                    doc = await session.get(Document, doc_id)
                    if doc is None:
                        raise ValueError(f"Document {doc_id} not found")
                    latest_job = await get_or_create_latest_job(
                        session, doc_id, job_id=job_id, default_progress=0.0
                    )
                    job_id = latest_job.id
                    active_job_id = latest_job.id
                    sha = doc.sha256
                    ext = doc.file_type.lstrip(".")
                    orig_filename = doc.filename

                target_path = raw_path or (orchestrator.raw_dir / f"{sha}.{ext}")

                if (
                    c_doc := await orchestrator._run_cancel_checkpoint(
                        cancel_event, doc_id, active_job_id, 0.0, "parsing", timings
                    )
                ) is not None:
                    return c_doc, False

                parsed_path = orchestrator.data_dir / "parsed" / f"{sha}.json"
                markdown_path = orchestrator.data_dir / "markdown" / f"{sha}.md"

                parse_result: ParseResult | None = None
                if not force_reparse and parsed_path.is_file() and markdown_path.is_file():
                    cached = load_parsed_checkpoint(orchestrator.data_dir, sha, orig_filename, ext)
                    if cached is not None:
                        cached_engine = cached.extra_meta.get("engine")
                        effective_prefer = (
                            orchestrator.prefer_pymupdf4llm
                            if prefer_pymupdf4llm is None
                            else prefer_pymupdf4llm
                        )
                        if is_cache_engine_mismatch(cached_engine, effective_prefer):
                            parse_result = None
                            force_reparse = True
                        else:
                            parse_result = cached

                if parse_result is not None:
                    timings["parsing_ms"] = 1
                else:
                    orchestrator._acquire_worker_slot()
                    try:
                        await orchestrator._update_state(
                            doc_id, active_job_id, status="parsing", stage="parsing", progress=0.2, timings=timings
                        )
                        effective_prefer_pymupdf = (
                            orchestrator.prefer_pymupdf4llm
                            if prefer_pymupdf4llm is None
                            else prefer_pymupdf4llm
                        )
                        parse_task_fn = getattr(worker_mod, "_worker_parse_task")
                        loop = asyncio.get_running_loop()
                        if orchestrator.use_process_pool:
                            try:
                                parse_result, parse_ms = await loop.run_in_executor(
                                    orchestrator.get_pool(),
                                    parse_task_fn,
                                    str(target_path),
                                    str(orchestrator.data_dir),
                                    sha,
                                    orig_filename,
                                    force_reparse,
                                    effective_prefer_pymupdf,
                                    orchestrator.excel_window_rows,
                                    orchestrator.max_upload_size_mb,
                                    orchestrator.device,
                                )
                            except BrokenProcessPool:
                                broken_doc = await orchestrator._handle_broken_pool(
                                    doc_id, active_job_id, "parsing", t_start, timings
                                )
                                return broken_doc, False
                        else:
                            parse_result, parse_ms = await asyncio.to_thread(
                                parse_task_fn,
                                str(target_path),
                                str(orchestrator.data_dir),
                                sha,
                                orig_filename,
                                force_reparse,
                                effective_prefer_pymupdf,
                                orchestrator.excel_window_rows,
                                orchestrator.max_upload_size_mb,
                                orchestrator.device,
                            )
                        timings["parsing_ms"] = parse_ms
                    finally:
                        orchestrator._release_worker_slot()

                if parse_result is not None:
                    orchestrator._parsed_cache[doc_id] = parse_result

                if (
                    c_doc := await orchestrator._run_cancel_checkpoint(
                        cancel_event, doc_id, active_job_id, 0.2, "classifying", timings
                    )
                ) is not None:
                    return c_doc, False

                await orchestrator._update_state(
                    doc_id,
                    active_job_id,
                    status="classifying",
                    stage="classifying",
                    progress=0.35,
                    timings=timings,
                    page_count=parse_result.page_count,
                    parsing_engine=parse_result.extra_meta.get("engine") if parse_result else None,
                    engine_fallback_reason=(
                        parse_result.extra_meta.get("engine_fallback_reason") if parse_result else None
                    ),
                )

                t_classify = time.perf_counter()
                sections = _extract_sections_from_markdown(parse_result.markdown if parse_result else "")
                try:
                    async with orchestrator.session_factory() as s_cat:
                        await refresh_taxonomy_catalog(s_cat)
                except Exception as exc:
                    logger.debug("Failed refreshing taxonomy catalog before classification: %s", exc)
                classifier = get_classifier()
                classification_res = await classifier.classify_document(
                    parse_result.markdown if parse_result else "",
                    sections=sections,
                    filename=orig_filename,
                )
                rec_profile = classification_res.recommended
                timings["classifying_ms"] = max(1, int((time.perf_counter() - t_classify) * 1000))

                dom_id = None
                sub_id = None
                async with orchestrator.session_factory() as s_meta:
                    d_obj, s_obj = await resolve_domain_and_subdomain(
                        s_meta, rec_profile.primary_domain, rec_profile.primary_subdomain
                    )
                    if d_obj:
                        dom_id = d_obj.id
                        if d_obj.slug == DEFAULT_DOMAIN and rec_profile.primary_domain != DEFAULT_DOMAIN:
                            rec_profile.primary_domain = DEFAULT_DOMAIN
                    if s_obj:
                        sub_id = s_obj.id
                        rec_profile.primary_subdomain = s_obj.slug

                await orchestrator._update_state(
                    doc_id,
                    active_job_id,
                    status="classifying",
                    stage="classifying",
                    progress=0.35,
                    timings=timings,
                    primary_domain_id=dom_id,
                    primary_subdomain_id=sub_id,
                    authority_tier=rec_profile.authority_tier,
                    jurisdiction=rec_profile.jurisdiction,
                    effective_year=rec_profile.effective_year,
                    is_current=rec_profile.is_current,
                    classification_result=rec_profile.model_dump_json(),
                )

                settings = get_settings()
                high_thresh = settings.ingestion.high_confidence_threshold
                is_auto_approved = (
                    approval_mode == "auto_all"
                    or (
                        approval_mode == "auto_high_confidence"
                        and rec_profile.confidence >= high_thresh
                        and not rec_profile.is_hybrid
                    )
                )

                if not is_auto_approved:
                    await orchestrator._update_state(
                        doc_id,
                        active_job_id,
                        status="awaiting_approval",
                        stage="awaiting_approval",
                        progress=0.45,
                        timings=timings,
                    )
                    async with orchestrator.session_factory() as s_final:
                        doc = await s_final.get(Document, doc_id)
                        assert doc is not None
                        return doc, False

                async with orchestrator.session_factory() as s_final:
                    doc = await s_final.get(Document, doc_id)
                    assert doc is not None
                    return doc, True

            except Exception as exc:
                logger.exception("process_to_classification failed for doc %s", doc_id)
                try:
                    await orchestrator._update_state(
                        doc_id,
                        active_job_id,
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
