"""Process RSS memory telemetry, CUDA VRAM stats, and on-demand memory reclamation."""
from __future__ import annotations

import ctypes
import logging
import os
import sys
from collections.abc import Callable

from qdrant_client import AsyncQdrantClient
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    MemoryReclaimResponse,
    MemoryStatsResponse,
    SystemHealthResponse,
    SystemStatsResponse,
)
from src.config import Settings
from src.core.credentials import resolve_provider_credential
from src.db.models import ChatThread, Document, DocumentChunk
from src.ingestion.classifier import get_classifier_model_name, is_classifier_resident
from src.ingestion.indexer import release_cuda_cache, resolve_compute_device, resolve_sparse_mode
from src.ingestion.worker import IngestionOrchestrator
from src.services.provider_service import build_provider_options

logger = logging.getLogger(__name__)


def get_process_rss_mb() -> float:
    """Return current process RSS in MiB with Linux /proc, Win32 psapi, and Unix getrusage fallback."""
    # Keep os.path.exists("/proc/self/status") and open(...) so unit mocks intercept cleanly.
    try:
        if os.path.exists("/proc/self/status"):
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            kb = float(parts[1])
                            return round(kb / 1024.0, 2)
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p

            try:
                psapi = ctypes.WinDLL("psapi", use_last_error=True)
                get_mem_info = psapi.GetProcessMemoryInfo
            except Exception:
                get_mem_info = kernel32.K32GetProcessMemoryInfo

            get_mem_info.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_ProcessMemoryCounters),
                ctypes.c_ulong,
            ]
            get_mem_info.restype = ctypes.c_int

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
            proc_handle = kernel32.GetCurrentProcess()
            if get_mem_info(proc_handle, ctypes.byref(counters), counters.cb):
                return round(counters.WorkingSetSize / (1024.0 * 1024.0), 2)
        except Exception:
            pass

    try:
        import resource

        rusage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return round(rusage.ru_maxrss / (1024.0 * 1024.0), 2)
        return round(rusage.ru_maxrss / 1024.0, 2)
    except Exception:
        pass

    return 0.0


def collect_memory_stats(
    orchestrator: IngestionOrchestrator | None = None,
    settings: Settings | None = None,
    *,
    rss_fn: Callable[[], float] = get_process_rss_mb,
) -> MemoryStatsResponse:
    """Collect host process RSS, PyTorch CUDA memory metrics, and orchestrator worker pool telemetry."""
    rss_mb = rss_fn()
    cuda_avail = False
    cuda_alloc_mb = 0.0
    cuda_res_mb = 0.0
    cuda_driver_free_mb: float | None = None
    cuda_driver_total_mb: float | None = None

    try:
        import torch

        if torch.cuda.is_available():
            cuda_avail = True
            cuda_alloc_mb = round(torch.cuda.memory_allocated() / (1024.0 * 1024.0), 2)
            cuda_res_mb = round(torch.cuda.memory_reserved() / (1024.0 * 1024.0), 2)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                cuda_driver_free_mb = round(free_bytes / (1024.0 * 1024.0), 2)
                cuda_driver_total_mb = round(total_bytes / (1024.0 * 1024.0), 2)
            except Exception:
                pass
    except Exception:
        pass

    worker_pool_active = False
    active_worker_jobs = 0
    active_indexing_jobs = 0
    worker_reclaim_count = 0
    reclaim_deferred = False
    worker_idle_timeout_sec = 60.0
    worker_max_tasks = 0

    if orchestrator is not None:
        state = orchestrator.get_worker_state()
        worker_pool_active = bool(state.get("worker_pool_active", False))
        active_worker_jobs = int(state.get("active_worker_jobs", 0))
        active_indexing_jobs = int(state.get("active_indexing_jobs", 0))
        worker_reclaim_count = int(state.get("worker_reclaim_count", 0))
        reclaim_deferred = bool(state.get("reclaim_deferred", False))
        worker_idle_timeout_sec = float(state.get("worker_idle_timeout_sec", 60.0))
        worker_max_tasks = int(state.get("worker_max_tasks", 0))
    elif settings is not None and hasattr(settings, "ingestion"):
        worker_idle_timeout_sec = float(settings.ingestion.worker_idle_timeout_sec)
        worker_max_tasks = int(settings.ingestion.worker_max_tasks)

    classifier_res = False
    classifier_mname = None
    try:
        classifier_res = is_classifier_resident()
        classifier_mname = get_classifier_model_name()
    except Exception:
        pass

    return MemoryStatsResponse(
        process_rss_mb=rss_mb,
        cuda_available=cuda_avail,
        cuda_allocated_mb=cuda_alloc_mb,
        cuda_reserved_mb=cuda_res_mb,
        cuda_driver_free_mb=cuda_driver_free_mb,
        cuda_driver_total_mb=cuda_driver_total_mb,
        worker_pool_active=worker_pool_active,
        active_worker_jobs=active_worker_jobs,
        active_indexing_jobs=active_indexing_jobs,
        worker_reclaim_count=worker_reclaim_count,
        reclaim_deferred=reclaim_deferred,
        worker_idle_timeout_sec=worker_idle_timeout_sec,
        worker_max_tasks=worker_max_tasks,
        classifier_resident=classifier_res,
        classifier_model_name=classifier_mname,
    )


async def check_system_health(
    session: AsyncSession,
    qdrant: AsyncQdrantClient,
    settings: Settings,
) -> SystemHealthResponse:
    """Verify reachability of both SQLite and Qdrant."""
    db_status = "ok"
    qdrant_status = "ok"

    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"error: {exc}"

    try:
        await qdrant.collection_exists(settings.qdrant_collection)
    except Exception as exc:
        qdrant_status = f"error: {exc}"

    overall = "ok" if (db_status == "ok" and qdrant_status == "ok") else "degraded"
    return SystemHealthResponse(
        status=overall,
        database=db_status,
        qdrant=qdrant_status,
        collection=settings.qdrant_collection,
    )


async def reclaim_system_memory(
    orchestrator: IngestionOrchestrator,
    settings: Settings,
    *,
    release_cuda_fn: Callable[[], bool] = release_cuda_cache,
    rss_fn: Callable[[], float] = get_process_rss_mb,
) -> MemoryReclaimResponse:
    """Shut down idle worker pool, flush CUDA cache when safe, and compact SQLite."""
    reclaimed = orchestrator.reclaim_worker_pool(wait=False, force=False)
    worker_state = orchestrator.get_worker_state()
    active_worker_jobs = worker_state.get("active_worker_jobs", 0)
    active_indexing_jobs = worker_state.get("active_indexing_jobs", 0)
    cuda_cleared = release_cuda_fn() if active_indexing_jobs == 0 else False

    if active_worker_jobs == 0 and active_indexing_jobs == 0:
        try:
            from src.db.session import get_engine, vacuum_sqlite_db
            from src.rag.graph import compact_chat_history_db

            await compact_chat_history_db(run_vacuum=True)
            await vacuum_sqlite_db(get_engine())
        except Exception as exc:
            logger.warning("Failed to compact/vacuum SQLite during memory reclaim: %s", exc)

    stats = collect_memory_stats(orchestrator, settings, rss_fn=rss_fn)
    return MemoryReclaimResponse(
        status="ok",
        worker_pool_reclaimed=reclaimed,
        deferred=worker_state.get("reclaim_deferred", False),
        cuda_cache_cleared=cuda_cleared,
        active_worker_jobs=active_worker_jobs,
        memory=stats,
    )


async def build_system_stats(
    session: AsyncSession,
    qdrant: AsyncQdrantClient,
    orchestrator: IngestionOrchestrator,
    settings: Settings,
    *,
    memory_fn: Callable[[IngestionOrchestrator | None, Settings | None], MemoryStatsResponse] | None = None,
) -> SystemStatsResponse:
    """Collect document/chunk counts, Qdrant vector count, active model settings, and memory stats."""
    total_docs = int((await session.execute(select(func.count()).select_from(Document))).scalar() or 0)
    ready_docs = int(
        (
            await session.execute(
                select(func.count()).select_from(Document).where(Document.status == "ready")
            )
        ).scalar()
        or 0
    )
    total_chunks = int((await session.execute(select(func.count()).select_from(DocumentChunk))).scalar() or 0)
    total_threads = int((await session.execute(select(func.count()).select_from(ChatThread))).scalar() or 0)

    status_rows = (
        await session.execute(select(Document.status, func.count()).group_by(Document.status))
    ).all()
    docs_by_status = {str(st): int(cnt) for st, cnt in status_rows if st}

    vector_count = 0
    try:
        if await qdrant.collection_exists(settings.qdrant_collection):
            cnt_res = await qdrant.count(collection_name=settings.qdrant_collection, exact=True)
            vector_count = int(cnt_res.count or 0)
    except Exception:
        vector_count = 0

    configured_providers, available_providers = build_provider_options(settings)

    active_provider = (os.environ.get("LLM_PROVIDER") or settings.llm.provider or "openai").strip().lower()
    active_model = (os.environ.get("LLM_MODEL_NAME") or settings.llm.model_name or "").strip()

    active_device = (os.environ.get("EMBEDDING_DEVICE") or settings.embedding.device or "auto").strip().lower()
    resolved_device = resolve_compute_device(active_device)
    active_sparse_mode = (
        os.environ.get("EMBEDDING_SPARSE_MODE") or settings.embedding.sparse_mode or "auto"
    ).strip().lower()
    resolved_sparse = resolve_sparse_mode(active_sparse_mode, resolved_device, settings.embedding.provider)

    ingestion_device = (
        os.environ.get("INGESTION_DEVICE") or settings.ingestion.device or "auto"
    ).strip().lower()
    resolved_ingestion_device = resolve_compute_device(ingestion_device)

    stats_collector = memory_fn or collect_memory_stats
    return SystemStatsResponse(
        total_documents=total_docs,
        ready_documents=ready_docs,
        total_chunks=total_chunks,
        total_threads=total_threads,
        qdrant_vectors_count=vector_count,
        vector_count=vector_count,
        ingestion_paused=orchestrator.is_paused,
        documents_by_status=docs_by_status,
        llm={
            "provider": active_provider,
            "model_name": active_model,
            "temperature": settings.llm.temperature,
            "base_url": resolve_provider_credential(active_provider, settings)[2],
        },
        embedding={
            "provider": settings.embedding.provider,
            "model_name": settings.embedding.model_name,
            "dimensions": settings.embedding.dimensions,
            "device": active_device,
            "resolved_device": resolved_device,
            "fp16": settings.embedding.fp16,
            "sparse_mode": active_sparse_mode,
            "resolved_sparse_mode": resolved_sparse,
            "max_length": settings.embedding.max_length,
        },
        reranker={
            "model_name": settings.reranker.model_name,
            "top_n": settings.reranker.top_n,
            "score_threshold": settings.reranker.score_threshold,
        },
        ingestion={
            "device": ingestion_device,
            "resolved_device": resolved_ingestion_device,
            "max_workers": settings.ingestion.max_workers,
            "worker_idle_timeout_sec": settings.ingestion.worker_idle_timeout_sec,
            "worker_max_tasks": settings.ingestion.worker_max_tasks,
            "clear_cuda_cache_after_indexing": settings.ingestion.clear_cuda_cache_after_indexing,
        },
        classifier={
            "model_name": os.environ.get("LAYA_MODEL_NAME") or settings.ingestion.laya_model_name,
            "resident": is_classifier_resident(),
            "enable_llm_classification_fallback": settings.ingestion.enable_llm_classification_fallback,
            "high_confidence_threshold": settings.ingestion.high_confidence_threshold,
        },
        memory=stats_collector(orchestrator, settings),
        configured_providers=configured_providers,
        available_providers=available_providers,
    )
