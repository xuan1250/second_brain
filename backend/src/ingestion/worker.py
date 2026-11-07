"""Background ingestion orchestrator: ProcessPoolExecutor (spawn) for parse+chunk, main-process indexing, and Laya classification gate."""
from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import get_settings
from src.db.models import Document, DomainRegistry, IngestionJob, SubdomainRegistry
from src.db.repository import get_document_by_sha
from src.db.session import get_session_factory
from src.ingestion.chunker import ChunkPayload, HierarchicalChunker
from src.ingestion.classifier import (
    DomainClassificationResult,
    RecommendedDomainProfile,
    SectionOverride,
    get_classifier,
    get_taxonomy_catalog,
    refresh_taxonomy_catalog,
)
from src.ingestion.indexer import BaseEmbedder, HybridIndexer, SparseEncoder, release_cuda_cache
from src.ingestion.parser import (
    DocumentParserService,
    ParseResult,
    compute_sha256,
    load_parsed_checkpoint,
)

logger = logging.getLogger(__name__)

# Lazy process-local singletons inside spawned worker process
_WORKER_PARSER: DocumentParserService | None = None
_WORKER_CHUNKER: HierarchicalChunker | None = None


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


def _worker_parse_task(
    file_path_str: str,
    data_dir_str: str,
    sha256: str,
    filename: str,
    force_reparse: bool = False,
    prefer_pymupdf4llm: bool | None = None,
    excel_window_rows: int = 50,
    max_upload_size_mb: int = 100,
    device: str = "auto",
) -> tuple[ParseResult, int]:
    """Top-level picklable worker function for document parsing in spawned process."""
    global _WORKER_PARSER
    t0 = time.perf_counter()
    if (
        _WORKER_PARSER is None
        or str(_WORKER_PARSER.data_dir) != data_dir_str
        or _WORKER_PARSER.excel_window_rows != excel_window_rows
        or _WORKER_PARSER.prefer_pymupdf4llm != prefer_pymupdf4llm
        or _WORKER_PARSER.device != device
    ):
        _WORKER_PARSER = DocumentParserService(
            data_dir=Path(data_dir_str),
            prefer_pymupdf4llm=prefer_pymupdf4llm,
            excel_window_rows=excel_window_rows,
            max_upload_size_mb=max_upload_size_mb,
            device=device,
        )
    result = _WORKER_PARSER.parse(
        Path(file_path_str),
        sha256=sha256,
        filename=filename,
        force_reparse=force_reparse,
        use_pymupdf_fallback=prefer_pymupdf4llm,
    )
    elapsed_ms = max(1, int((time.perf_counter() - t0) * 1000))
    return result, elapsed_ms


def _worker_chunk_task(
    parse_result: ParseResult,
    child_max_tokens: int,
    parent_max_tokens: int,
    domain_profile: Optional[RecommendedDomainProfile] = None,
) -> tuple[list[ChunkPayload], int]:
    """Top-level picklable worker function for hierarchical chunking in spawned process."""
    global _WORKER_CHUNKER
    t0 = time.perf_counter()
    if (
        _WORKER_CHUNKER is None
        or _WORKER_CHUNKER.child_max_tokens != child_max_tokens
        or _WORKER_CHUNKER.parent_max_tokens != parent_max_tokens
    ):
        _WORKER_CHUNKER = HierarchicalChunker(
            child_max_tokens=child_max_tokens,
            parent_max_tokens=parent_max_tokens,
        )
    chunks = _WORKER_CHUNKER.chunk(parse_result, domain_profile=domain_profile)
    elapsed_ms = max(1, int((time.perf_counter() - t0) * 1000))
    return chunks, elapsed_ms


class IngestionOrchestrator:
    """Coordinates idempotent ingestion: queued -> parsing -> classifying -> awaiting_approval -> chunking -> indexing -> ready | failed."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        qdrant: AsyncQdrantClient | None = None,
        embedder: BaseEmbedder | None = None,
        sparse_encoder: SparseEncoder | None = None,
        data_dir: Path | None = None,
        collection: str | None = None,
        max_workers: int | None = None,
        use_process_pool: bool = True,
        prefer_pymupdf4llm: bool | None = None,
        excel_window_rows: int = 50,
        child_max_tokens: int | None = None,
        parent_max_tokens: int | None = None,
        device: str | None = None,
        worker_idle_timeout_sec: float | None = None,
        worker_max_tasks: int | None = None,
        clear_cuda_cache: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.session_factory = session_factory or get_session_factory()
        self.qdrant = qdrant
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.data_dir = Path(data_dir) if data_dir else settings.data_dir
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.collection = collection or settings.qdrant_collection
        self.max_workers = max_workers or settings.ingestion.max_workers
        self.max_upload_size_mb = settings.ingestion.max_upload_size_mb
        self.use_process_pool = use_process_pool
        self.device = (device or settings.ingestion.device or "auto").strip().lower()
        if prefer_pymupdf4llm is None:
            env_val = os.environ.get("PREFER_PYMUPDF4LLM")
            if env_val is not None and env_val.strip():
                self.prefer_pymupdf4llm: bool | None = env_val.lower() in (
                    "1",
                    "true",
                    "yes",
                )
            else:
                self.prefer_pymupdf4llm = None
        else:
            self.prefer_pymupdf4llm = prefer_pymupdf4llm
        self.excel_window_rows = excel_window_rows
        self.child_max_tokens = child_max_tokens or settings.chunking.child_max_tokens
        self.parent_max_tokens = parent_max_tokens or settings.chunking.parent_max_tokens

        # Worker pool lifecycle management
        self.worker_idle_timeout_sec = (
            worker_idle_timeout_sec
            if worker_idle_timeout_sec is not None
            else settings.ingestion.worker_idle_timeout_sec
        )
        self.worker_max_tasks = (
            worker_max_tasks
            if worker_max_tasks is not None
            else settings.ingestion.worker_max_tasks
        )
        self.clear_cuda_cache = (
            clear_cuda_cache
            if clear_cuda_cache is not None
            else settings.ingestion.clear_cuda_cache_after_indexing
        )

        self._pool: ProcessPoolExecutor | None = None
        self._pool_completed_docs = 0
        self._pool_reclaim_count = 0
        self._executing_tasks = 0
        self._active_worker_jobs = 0
        self._active_indexing_jobs = 0
        self._last_worker_activity_at = time.time()
        self._idle_timer_handle: asyncio.TimerHandle | None = None
        self._reclaim_on_idle = False

        # Ingestion concurrency & flow control
        self._doc_semaphore = asyncio.Semaphore(self.max_workers)
        self._queued_tasks = 0
        self._cancellations: dict[str, asyncio.Event] = {}
        self._parsed_cache: dict[str, ParseResult] = {}
        self._is_paused = False
        self._resume_event = asyncio.Event()
        self._resume_event.set()

    def cancel_document(self, doc_id: str) -> None:
        """Signal in-flight cancellation for a queued or executing document."""
        self._parsed_cache.pop(doc_id, None)
        if doc_id in self._cancellations:
            logger.info("Signalling cancellation for doc_id=%s", doc_id)
            self._cancellations[doc_id].set()
        else:
            ev = asyncio.Event()
            ev.set()
            self._cancellations[doc_id] = ev

    def pause(self) -> None:
        self._is_paused = True
        self._resume_event.clear()
        logger.info("Ingestion orchestrator paused")

    def resume(self) -> None:
        self._is_paused = False
        self._resume_event.set()
        logger.info("Ingestion orchestrator resumed")

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    def get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            ctx = multiprocessing.get_context("spawn")
            self._pool = ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx)
        return self._pool

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer_handle is not None:
            self._idle_timer_handle.cancel()
            self._idle_timer_handle = None

    def _acquire_worker_slot(self) -> None:
        self._executing_tasks += 1
        self._active_worker_jobs = self._executing_tasks
        self._cancel_idle_timer()

    def _release_worker_slot(self) -> None:
        self._executing_tasks = max(0, self._executing_tasks - 1)
        self._active_worker_jobs = self._executing_tasks
        self._last_worker_activity_at = time.time()
        if self._executing_tasks == 0 and self._pool is not None:
            if self._reclaim_on_idle or (
                self.worker_max_tasks > 0 and self._pool_completed_docs >= self.worker_max_tasks
            ):
                self._reclaim_on_idle = False
                self.reclaim_worker_pool(wait=False, force=False)
            elif self.worker_idle_timeout_sec > 0:
                try:
                    loop = asyncio.get_running_loop()
                    self._idle_timer_handle = loop.call_later(
                        self.worker_idle_timeout_sec, self._on_idle_timeout
                    )
                except RuntimeError:
                    pass

    def _on_idle_timeout(self) -> None:
        logger.info(
            "Worker pool idle timeout (%ss) reached; reclaiming worker pool",
            self.worker_idle_timeout_sec,
        )
        self.reclaim_worker_pool(wait=False, force=False)

    def reclaim_worker_pool(self, *, wait: bool = False, force: bool = False) -> bool:
        self._cancel_idle_timer()
        if (self._executing_tasks > 0 or self._active_worker_jobs > 0) and not force:
            if self._pool is not None:
                self._reclaim_on_idle = True
            return False
        if self._pool is not None:
            self._reclaim_on_idle = False
            self._pool_completed_docs = 0
            self._pool.shutdown(wait=wait, cancel_futures=False)
            self._pool = None
            self._pool_reclaim_count += 1
            logger.info("Worker pool successfully reclaimed (reclaim_count=%d)", self._pool_reclaim_count)
            return True
        return False

    def get_worker_state(self) -> dict[str, Any]:
        return {
            "worker_pool_active": self._pool is not None,
            "ingestion_paused": self._is_paused,
            "queued_tasks": self._queued_tasks,
            "executing_tasks": self._executing_tasks,
            "active_worker_jobs": self._active_worker_jobs,
            "active_indexing_jobs": self._active_indexing_jobs,
            "worker_reclaim_count": self._pool_reclaim_count,
            "reclaim_deferred": self._reclaim_on_idle,
            "worker_idle_timeout_sec": self.worker_idle_timeout_sec,
            "worker_max_tasks": self.worker_max_tasks,
        }

    def shutdown(self, wait: bool = True) -> None:
        self._cancel_idle_timer()
        self._reclaim_on_idle = False
        self.reclaim_worker_pool(wait=wait, force=True)

    async def __aenter__(self) -> IngestionOrchestrator:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.shutdown(wait=True)

    async def ingest_file(
        self,
        file_path: Path | str,
        *,
        filename: str | None = None,
        force_reparse: bool = False,
        prefer_pymupdf4llm: bool | None = None,
        approval_mode: str = "auto_all",
    ) -> Document:
        """Ingest a file end-to-end with SHA-256 idempotency check and transactional stage tracking."""
        src_path = Path(file_path)
        if not src_path.is_file():
            raise FileNotFoundError(f"Source file not found: {src_path}")

        orig_filename = filename or src_path.name
        ext = src_path.suffix.lower().lstrip(".")
        sha = compute_sha256(src_path)
        size_bytes = src_path.stat().st_size

        # 1. Idempotency check in SQLite
        async with self.session_factory() as session:
            existing = await get_document_by_sha(session, sha)
            if existing is not None and existing.status == "ready" and not force_reparse:
                requested_engine = (
                    "pymupdf4llm"
                    if prefer_pymupdf4llm is True
                    else ("docling" if prefer_pymupdf4llm is False else None)
                )
                is_mode_upgrade = requested_engine == "docling" and existing.parsing_engine != "docling"
                if not is_mode_upgrade:
                    return existing
                force_reparse = True

            if existing is None:
                doc = Document(
                    filename=orig_filename,
                    file_type=ext,
                    sha256=sha,
                    size_bytes=size_bytes,
                    status="queued",
                )
                session.add(doc)
                await session.commit()
                await session.refresh(doc)
            else:
                doc = existing
                doc.status = "queued"
                doc.error = None
                doc.updated_at = _now()
                session.add(doc)
                await session.commit()
                await session.refresh(doc)

            job = IngestionJob(
                doc_id=doc.id,
                status="queued",
                stage="queued",
                progress=0.0,
                stage_timings=json.dumps({}),
            )
            session.add(job)
            await session.commit()
            await session.refresh(job)
            doc_id = doc.id
            job_id = job.id

        # 2. Copy immutable raw file to data/raw/{sha256}.{ext}
        raw_dest = self.raw_dir / f"{sha}.{ext}"
        if not raw_dest.is_file() or raw_dest.resolve() != src_path.resolve():
            if not raw_dest.is_file():
                shutil.copy2(src_path, raw_dest)

        return await self.process_document(
            doc_id=doc_id,
            job_id=job_id,
            raw_path=raw_dest,
            force_reparse=force_reparse,
            prefer_pymupdf4llm=prefer_pymupdf4llm,
            approval_mode=approval_mode,
        )

    async def process_to_classification(
        self,
        doc_id: str,
        job_id: str | None = None,
        raw_path: Path | None = None,
        *,
        force_reparse: bool = False,
        prefer_pymupdf4llm: bool | None = None,
        approval_mode: str = "auto_all",
    ) -> tuple[Document, bool]:
        """Execute Stage 1 (parsing) and Stage 1b (classifying), pausing at awaiting_approval if not auto-approved."""
        cancel_event = asyncio.Event()
        self._cancellations[doc_id] = cancel_event
        self._queued_tasks += 1

        try:
            if self._is_paused:
                logger.info("Doc %s waiting at semaphore barrier due to ingestion pause", doc_id)
                await self._resume_event.wait()

            async with self._doc_semaphore:
                if self._is_paused:
                    logger.info("Doc %s waiting inside semaphore barrier due to ingestion pause", doc_id)
                    await self._resume_event.wait()

                timings: dict[str, int] = {}
                t_start = time.perf_counter()

                async def _handle_cancellation(progress: float, stage_name: str) -> Document | None:
                    if cancel_event.is_set():
                        logger.info("Document %s was cancelled before %s", doc_id, stage_name)
                        try:
                            await self._update_state(
                                doc_id,
                                job_id,
                                status="failed",
                                stage="cancelled",
                                progress=progress,
                                timings=timings,
                                error="Ingestion cancelled",
                            )
                        except Exception:
                            pass
                        async with self.session_factory() as s:
                            final_doc = await s.get(Document, doc_id)
                            return final_doc or Document(id=doc_id, status="failed", error="Ingestion cancelled")
                    return None

                async with self.session_factory() as session:
                    try:
                        await refresh_taxonomy_catalog(session)
                    except Exception as exc:
                        logger.debug("Failed refreshing taxonomy catalog in complete_approved_ingestion: %s", exc)
                    doc = await session.get(Document, doc_id)
                    if doc is None:
                        raise ValueError(f"Document {doc_id} not found")
                    if job_id is None:
                        res = await session.execute(
                            select(IngestionJob)
                            .where(IngestionJob.doc_id == doc_id)
                            .order_by(IngestionJob.created_at.desc())
                        )
                        latest_job = res.scalars().first()
                        if latest_job is None:
                            latest_job = IngestionJob(doc_id=doc_id, status="queued", stage="queued", progress=0.0)
                            session.add(latest_job)
                            await session.commit()
                            await session.refresh(latest_job)
                        job_id = latest_job.id

                    sha = doc.sha256
                    ext = doc.file_type.lstrip(".")
                    orig_filename = doc.filename

                target_path = raw_path or (self.raw_dir / f"{sha}.{ext}")

                if (c_doc := await _handle_cancellation(0.0, "parsing")) is not None:
                    return c_doc, False

                # Stage 1: Checkpoint or Parsing
                parsed_path = self.data_dir / "parsed" / f"{sha}.json"
                markdown_path = self.data_dir / "markdown" / f"{sha}.md"

                parse_result: ParseResult | None = None
                if not force_reparse and parsed_path.is_file() and markdown_path.is_file():
                    cached = load_parsed_checkpoint(self.data_dir, sha, orig_filename, ext)
                    if cached is not None:
                        cached_engine = cached.extra_meta.get("engine")
                        effective_prefer = (
                            self.prefer_pymupdf4llm
                            if prefer_pymupdf4llm is None
                            else prefer_pymupdf4llm
                        )
                        mismatch = (
                            (effective_prefer is True and cached_engine != "pymupdf4llm")
                            or (effective_prefer is False and cached_engine != "docling")
                        )
                        if mismatch:
                            parse_result = None
                            force_reparse = True
                        else:
                            parse_result = cached

                if parse_result is not None:
                    timings["parsing_ms"] = 1
                else:
                    self._acquire_worker_slot()
                    try:
                        await self._update_state(
                            doc_id, job_id, status="parsing", stage="parsing", progress=0.2, timings=timings
                        )
                        effective_prefer_pymupdf = (
                            self.prefer_pymupdf4llm if prefer_pymupdf4llm is None else prefer_pymupdf4llm
                        )
                        loop = asyncio.get_running_loop()
                        if self.use_process_pool:
                            try:
                                parse_result, parse_ms = await loop.run_in_executor(
                                    self.get_pool(),
                                    _worker_parse_task,
                                    str(target_path),
                                    str(self.data_dir),
                                    sha,
                                    orig_filename,
                                    force_reparse,
                                    effective_prefer_pymupdf,
                                    self.excel_window_rows,
                                    self.max_upload_size_mb,
                                    self.device,
                                )
                            except BrokenProcessPool:
                                broken_doc = await self._handle_broken_pool(
                                    doc_id, job_id, "parsing", t_start, timings
                                )
                                return broken_doc, False
                        else:
                            parse_result, parse_ms = await asyncio.to_thread(
                                _worker_parse_task,
                                str(target_path),
                                str(self.data_dir),
                                sha,
                                orig_filename,
                                force_reparse,
                                effective_prefer_pymupdf,
                                self.excel_window_rows,
                                self.max_upload_size_mb,
                                self.device,
                            )
                        timings["parsing_ms"] = parse_ms
                    finally:
                        # Release worker pool slot immediately after parsing!
                        self._release_worker_slot()

                if parse_result is not None:
                    self._parsed_cache[doc_id] = parse_result

                if (c_doc := await _handle_cancellation(0.2, "classifying")) is not None:
                    return c_doc, False

                # Stage 1b: Classifying via Laya classifier
                await self._update_state(
                    doc_id,
                    job_id,
                    status="classifying",
                    stage="classifying",
                    progress=0.35,
                    timings=timings,
                    page_count=parse_result.page_count,
                    parsing_engine=parse_result.extra_meta.get("engine") if parse_result else None,
                    engine_fallback_reason=parse_result.extra_meta.get("engine_fallback_reason") if parse_result else None,
                )

                t_classify = time.perf_counter()
                sections = _extract_sections_from_markdown(parse_result.markdown if parse_result else "")
                try:
                    async with self.session_factory() as s_cat:
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

                # Resolve domain IDs in SQLite
                dom_id = None
                sub_id = None
                async with self.session_factory() as s_meta:
                    res_dom = await s_meta.execute(
                        select(DomainRegistry).where(DomainRegistry.slug == rec_profile.primary_domain)
                    )
                    d_obj = res_dom.scalars().first()
                    if d_obj is None:
                        res_dom_gen = await s_meta.execute(
                            select(DomainRegistry).where(DomainRegistry.slug == "general")
                        )
                        d_obj = res_dom_gen.scalars().first()
                        rec_profile.primary_domain = "general"

                    if d_obj:
                        dom_id = d_obj.id
                        res_sub = await s_meta.execute(
                            select(SubdomainRegistry).where(
                                SubdomainRegistry.slug == rec_profile.primary_subdomain,
                                SubdomainRegistry.domain_id == d_obj.id,
                            )
                        )
                        s_obj = res_sub.scalars().first()
                        if s_obj is None:
                            res_sub_fallback = await s_meta.execute(
                                select(SubdomainRegistry).where(
                                    SubdomainRegistry.domain_id == d_obj.id,
                                    SubdomainRegistry.slug == f"{d_obj.slug}.general",
                                )
                            )
                            s_obj = res_sub_fallback.scalars().first()
                            if s_obj is None:
                                res_sub_any = await s_meta.execute(
                                    select(SubdomainRegistry)
                                    .where(SubdomainRegistry.domain_id == d_obj.id)
                                    .order_by(SubdomainRegistry.slug)
                                )
                                s_obj = res_sub_any.scalars().first()
                            if s_obj:
                                rec_profile.primary_subdomain = s_obj.slug
                        if s_obj:
                            sub_id = s_obj.id

                # Save classification metadata
                await self._update_state(
                    doc_id,
                    job_id,
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

                # Evaluate auto-approval criteria
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
                    await self._update_state(
                        doc_id,
                        job_id,
                        status="awaiting_approval",
                        stage="awaiting_approval",
                        progress=0.45,
                        timings=timings,
                    )
                    async with self.session_factory() as s_final:
                        doc = await s_final.get(Document, doc_id)
                        assert doc is not None
                        return doc, False

                async with self.session_factory() as s_final:
                    doc = await s_final.get(Document, doc_id)
                    assert doc is not None
                    return doc, True

        finally:
            self._queued_tasks = max(0, self._queued_tasks - 1)
            self._cancellations.pop(doc_id, None)

    async def complete_approved_ingestion(
        self,
        doc_id: str,
        job_id: str | None = None,
        *,
        approved_profile: Optional[Any] = None,
    ) -> Document:
        """Resume Stage 2 (chunking) and Stage 3 (indexing) from cached data/parsed/{sha}.json."""
        if doc_id in self._cancellations and self._cancellations[doc_id].is_set():
            cancel_event = self._cancellations[doc_id]
        else:
            cancel_event = asyncio.Event()
            self._cancellations[doc_id] = cancel_event
        self._queued_tasks += 1

        if cancel_event.is_set():
            logger.info("Document %s was cancelled before acquiring semaphore", doc_id)
            async with self.session_factory() as session:
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
            if self._is_paused:
                await self._resume_event.wait()

            async with self._doc_semaphore:
                if self._is_paused:
                    await self._resume_event.wait()

                timings: dict[str, int] = {}
                t_start = time.perf_counter()

                async def _handle_cancellation(progress: float, stage_name: str) -> Document | None:
                    if cancel_event.is_set():
                        logger.info("Document %s was cancelled before %s in complete_approved_ingestion", doc_id, stage_name)
                        try:
                            await self._update_state(
                                doc_id,
                                job_id,
                                status="failed",
                                stage="cancelled",
                                progress=progress,
                                timings=timings,
                                error="Ingestion cancelled",
                            )
                        except Exception:
                            pass
                        async with self.session_factory() as s:
                            final_doc = await s.get(Document, doc_id)
                            return final_doc or Document(id=doc_id, status="failed", error="Ingestion cancelled")
                    return None

                if (c_doc := await _handle_cancellation(0.45, "chunking")) is not None:
                    return c_doc

                async with self.session_factory() as session:
                    try:
                        await refresh_taxonomy_catalog(session)
                    except Exception as exc:
                        logger.debug("Failed refreshing taxonomy catalog in complete_approved_ingestion: %s", exc)
                    doc = await session.get(Document, doc_id)
                    if doc is None:
                        raise ValueError(f"Document {doc_id} not found")
                    if job_id is None:
                        res = await session.execute(
                            select(IngestionJob)
                            .where(IngestionJob.doc_id == doc_id)
                            .order_by(IngestionJob.created_at.desc())
                        )
                        latest_job = res.scalars().first()
                        if latest_job is None:
                            latest_job = IngestionJob(doc_id=doc_id, status="queued", stage="queued", progress=0.45)
                            session.add(latest_job)
                            await session.commit()
                            await session.refresh(latest_job)
                        job_id = latest_job.id
                    else:
                        latest_job = await session.get(IngestionJob, job_id)

                    if latest_job and latest_job.stage_timings:
                        try:
                            parsed_t = json.loads(latest_job.stage_timings)
                            if isinstance(parsed_t, dict):
                                timings = {k: int(v) for k, v in parsed_t.items()}
                        except Exception:
                            timings = {}

                    sha = doc.sha256
                    ext = doc.file_type.lstrip(".")
                    orig_filename = doc.filename

                    # Process approved_profile overrides if provided
                    if approved_profile is not None:
                        if hasattr(approved_profile, "model_dump"):
                            prof_data = approved_profile.model_dump(exclude_unset=True)
                        elif isinstance(approved_profile, dict):
                            prof_data = approved_profile
                        else:
                            prof_data = {}

                        d_obj = None
                        if "domain" in prof_data and prof_data["domain"]:
                            res_d = await session.execute(
                                select(DomainRegistry).where(DomainRegistry.slug == prof_data["domain"])
                            )
                            d_obj = res_d.scalars().first()
                            if d_obj:
                                doc.primary_domain_id = d_obj.id
                        elif doc.primary_domain_id:
                            d_obj = await session.get(DomainRegistry, doc.primary_domain_id)

                        if d_obj is None:
                            res_d = await session.execute(
                                select(DomainRegistry).where(DomainRegistry.slug == "general")
                            )
                            d_obj = res_d.scalars().first()
                            if d_obj:
                                doc.primary_domain_id = d_obj.id

                        s_obj = None
                        sub_slug = prof_data.get("subdomain")
                        if d_obj and sub_slug:
                            res_s = await session.execute(
                                select(SubdomainRegistry).where(
                                    SubdomainRegistry.slug == sub_slug,
                                    SubdomainRegistry.domain_id == d_obj.id,
                                )
                            )
                            s_obj = res_s.scalars().first()

                        if d_obj and s_obj is None:
                            res_s = await session.execute(
                                select(SubdomainRegistry).where(
                                    SubdomainRegistry.domain_id == d_obj.id,
                                    SubdomainRegistry.slug == f"{d_obj.slug}.general",
                                )
                            )
                            s_obj = res_s.scalars().first()
                            if s_obj is None:
                                res_s = await session.execute(
                                    select(SubdomainRegistry)
                                    .where(SubdomainRegistry.domain_id == d_obj.id)
                                    .order_by(SubdomainRegistry.slug)
                                )
                                s_obj = res_s.scalars().first()

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

                # Load cached parsed checkpoint (prefer in-memory cache from Stage 1, then disk)
                cached = self._parsed_cache.pop(doc_id, None) or load_parsed_checkpoint(
                    self.data_dir, sha, orig_filename, ext
                )
                if cached is None:
                    # Parse if cache unexpectedly missing
                    raw_path = self.raw_dir / f"{sha}.{ext}"
                    parse_res, p_ms = await asyncio.to_thread(
                        _worker_parse_task,
                        str(raw_path),
                        str(self.data_dir),
                        sha,
                        orig_filename,
                        False,
                        self.prefer_pymupdf4llm,
                        self.excel_window_rows,
                        self.max_upload_size_mb,
                        self.device,
                    )
                    cached = parse_res
                    timings["parsing_ms"] = p_ms

                # Reconstruct domain profile
                profile_obj: Optional[RecommendedDomainProfile] = None
                if doc.classification_result:
                    try:
                        profile_obj = RecommendedDomainProfile.model_validate_json(doc.classification_result)
                    except Exception:
                        profile_obj = None

                if profile_obj is not None:
                    catalog = get_taxonomy_catalog()
                    if not getattr(profile_obj, "chunking_strategy", None) or profile_obj.chunking_strategy == "standard_markdown":
                        profile_obj.chunking_strategy = catalog.get_chunking_strategy(profile_obj.primary_domain)
                    for sec_ov in (profile_obj.section_overrides or []):
                        if not getattr(sec_ov, "chunking_strategy", None) or sec_ov.chunking_strategy == "standard_markdown":
                            sec_ov.chunking_strategy = catalog.get_chunking_strategy(sec_ov.domain)

                try:
                    # Stage 2: Chunking
                    if (c_doc := await _handle_cancellation(0.55, "chunking")) is not None:
                        return c_doc

                    self._acquire_worker_slot()
                    try:
                        await self._update_state(
                            doc_id,
                            job_id,
                            status="chunking",
                            stage="chunking",
                            progress=0.55,
                            timings=timings,
                            page_count=cached.page_count,
                            parsing_engine=cached.extra_meta.get("engine") if cached else None,
                            engine_fallback_reason=cached.extra_meta.get("engine_fallback_reason") if cached else None,
                        )
                        loop = asyncio.get_running_loop()
                        if self.use_process_pool:
                            try:
                                chunks, chunk_ms = await loop.run_in_executor(
                                    self.get_pool(),
                                    _worker_chunk_task,
                                    cached,
                                    self.child_max_tokens,
                                    self.parent_max_tokens,
                                    profile_obj,
                                )
                            except BrokenProcessPool:
                                return await self._handle_broken_pool(doc_id, job_id, "chunking", t_start, timings)
                        else:
                            chunks, chunk_ms = await asyncio.to_thread(
                                _worker_chunk_task,
                                cached,
                                self.child_max_tokens,
                                self.parent_max_tokens,
                                profile_obj,
                            )
                        timings["chunking_ms"] = chunk_ms

                        if self.use_process_pool:
                            self._pool_completed_docs += 1
                    finally:
                        self._release_worker_slot()

                    # Stage 3: Indexing in main process
                    if (c_doc := await _handle_cancellation(0.8, "indexing")) is not None:
                        return c_doc

                    await self._update_state(
                        doc_id,
                        job_id,
                        status="indexing",
                        stage="indexing",
                        progress=0.8,
                        timings=timings,
                    )
                    t_index = time.perf_counter()
                    if self.qdrant is None:
                        from src.db.qdrant import get_qdrant_client

                        qdrant_client = get_qdrant_client()
                        close_qdrant = True
                    else:
                        qdrant_client = self.qdrant
                        close_qdrant = False

                    self._active_indexing_jobs += 1
                    try:
                        try:
                            indexer = HybridIndexer(
                                qdrant=qdrant_client,
                                embedder=self.embedder,
                                sparse_encoder=self.sparse_encoder,
                                collection=self.collection,
                                clear_cuda_cache=False,
                            )
                            async with self.session_factory() as session:
                                indexed_count = await indexer.index_chunks(session, doc_id=doc_id, chunks=chunks)
                        finally:
                            if close_qdrant:
                                await qdrant_client.close()
                    finally:
                        self._active_indexing_jobs = max(0, self._active_indexing_jobs - 1)
                        if self.clear_cuda_cache and self._active_indexing_jobs == 0:
                            release_cuda_cache()

                    timings["indexing_ms"] = max(1, int((time.perf_counter() - t_index) * 1000))
                    total_ms = max(1, int((time.perf_counter() - t_start) * 1000))
                    timings["total_ms"] = total_ms

                    # Stage 4: Ready
                    await self._update_state(
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

                    async with self.session_factory() as session:
                        final_doc = await session.get(Document, doc_id)
                        assert final_doc is not None
                        return final_doc

                except Exception as exc:
                    logger.exception("complete_approved_ingestion failed for doc %s", doc_id)
                    try:
                        await self._update_state(
                            doc_id,
                            job_id,
                            status="failed",
                            stage="failed",
                            progress=0.0,
                            error=str(exc),
                        )
                    except Exception:
                        pass
                    raise

        finally:
            self._queued_tasks = max(0, self._queued_tasks - 1)
            self._cancellations.pop(doc_id, None)

    async def process_document(
        self,
        doc_id: str,
        job_id: str | None = None,
        raw_path: Path | None = None,
        *,
        force_reparse: bool = False,
        prefer_pymupdf4llm: bool | None = None,
        approval_mode: str = "auto_all",
    ) -> Document:
        """Backward-compatible wrapper executing parsing, classification, and auto-completing indexing when auto-approved."""
        self._acquire_worker_slot()
        try:
            doc, auto_approved = await self.process_to_classification(
                doc_id=doc_id,
                job_id=job_id,
                raw_path=raw_path,
                force_reparse=force_reparse,
                prefer_pymupdf4llm=prefer_pymupdf4llm,
                approval_mode=approval_mode,
            )
            if auto_approved:
                return await self.complete_approved_ingestion(doc_id=doc_id, job_id=job_id)
            return doc
        finally:
            self._release_worker_slot()

    async def _handle_broken_pool(
        self,
        doc_id: str,
        job_id: str,
        stage: str,
        t_start: float,
        timings: dict[str, int],
    ) -> Document:
        logger.critical(
            "Process pool terminated abruptly during %s (poison document or C-level segfault). Forcing pool recovery for doc_id=%s",
            stage,
            doc_id,
        )
        self.reclaim_worker_pool(wait=False, force=True)
        poison_msg = f"Document {stage} caused fatal worker process termination (poison document)"
        total_ms = max(1, int((time.perf_counter() - t_start) * 1000))
        timings["total_ms"] = total_ms
        await self._update_state(
            doc_id,
            job_id,
            status="failed",
            stage="failed",
            progress=1.0,
            timings=timings,
            duration_ms=total_ms,
            error=poison_msg,
        )
        async with self.session_factory() as session:
            final_doc = await session.get(Document, doc_id)
            return final_doc or Document(id=doc_id, status="failed", error=poison_msg)

    async def _update_state(
        self,
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
        now = _now()
        async with self.session_factory() as session:
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

            job = await session.get(IngestionJob, job_id)
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
