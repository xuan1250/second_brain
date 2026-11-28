"""Background ingestion orchestrator: ProcessPoolExecutor (spawn) for parse+chunk, main-process indexing, and Laya classification gate."""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import get_settings
from src.db.models import Document, IngestionJob
from src.db.repository import get_document_by_sha
from src.db.session import get_session_factory
from src.ingestion.chunker import ChunkPayload, HierarchicalChunker
from src.ingestion.classifier import RecommendedDomainProfile
from src.ingestion.constants import is_engine_mode_upgrade, resolve_prefer_pymupdf4llm
from src.ingestion.indexer import BaseEmbedder, SparseEncoder, release_cuda_cache
from src.ingestion.orchestrator.db_state import _now, update_ingestion_state
from src.ingestion.orchestrator.pool import WorkerPoolMixin
from src.ingestion.orchestrator.stages import (
    run_complete_approved_ingestion,
    run_process_document,
    run_process_to_classification,
)
from src.ingestion.parser import DocumentParserService, ParseResult, compute_sha256

logger = logging.getLogger(__name__)

# Lazy process-local singletons inside spawned worker process
_WORKER_PARSER: DocumentParserService | None = None
_WORKER_CHUNKER: HierarchicalChunker | None = None


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


class IngestionOrchestrator(WorkerPoolMixin):
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
        self.prefer_pymupdf4llm: bool | None = resolve_prefer_pymupdf4llm(prefer_pymupdf4llm)
        self.excel_window_rows = excel_window_rows
        self.child_max_tokens = child_max_tokens or settings.chunking.child_max_tokens
        self.parent_max_tokens = parent_max_tokens or settings.chunking.parent_max_tokens

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

        self._init_pool_state()

    async def __aenter__(self) -> IngestionOrchestrator:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.shutdown(wait=True)

    async def _run_cancel_checkpoint(
        self,
        cancel_event: asyncio.Event,
        doc_id: str,
        job_id: str,
        progress: float,
        stage_name: str,
        timings: dict[str, int],
    ) -> Document | None:
        if not cancel_event.is_set():
            return None
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

        async with self.session_factory() as session:
            existing = await get_document_by_sha(session, sha)
            if existing is not None and existing.status == "ready" and not force_reparse:
                if not is_engine_mode_upgrade(existing.parsing_engine, prefer_pymupdf4llm):
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
        return await run_process_to_classification(
            self,
            doc_id=doc_id,
            job_id=job_id,
            raw_path=raw_path,
            force_reparse=force_reparse,
            prefer_pymupdf4llm=prefer_pymupdf4llm,
            approval_mode=approval_mode,
        )

    async def complete_approved_ingestion(
        self,
        doc_id: str,
        job_id: str | None = None,
        *,
        approved_profile: Optional[Any] = None,
    ) -> Document:
        """Resume Stage 2 (chunking) and Stage 3 (indexing) from cached data/parsed/{sha}.json."""
        return await run_complete_approved_ingestion(
            self,
            doc_id=doc_id,
            job_id=job_id,
            approved_profile=approved_profile,
        )

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
        return await run_process_document(
            self,
            doc_id=doc_id,
            job_id=job_id,
            raw_path=raw_path,
            force_reparse=force_reparse,
            prefer_pymupdf4llm=prefer_pymupdf4llm,
            approval_mode=approval_mode,
        )

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
        await update_ingestion_state(
            self.session_factory,
            doc_id,
            job_id,
            status=status,
            stage=stage,
            progress=progress,
            timings=timings,
            duration_ms=duration_ms,
            chunk_count=chunk_count,
            page_count=page_count,
            parsing_engine=parsing_engine,
            engine_fallback_reason=engine_fallback_reason,
            primary_domain_id=primary_domain_id,
            primary_subdomain_id=primary_subdomain_id,
            authority_tier=authority_tier,
            jurisdiction=jurisdiction,
            effective_year=effective_year,
            is_current=is_current,
            classification_result=classification_result,
            error=error,
        )


__all__ = [
    "IngestionOrchestrator",
    "_worker_chunk_task",
    "_worker_parse_task",
    "release_cuda_cache",
]
