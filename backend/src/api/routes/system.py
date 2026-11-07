"""System health check, Qdrant/SQLite statistics, memory telemetry, and read-only provider configuration routes."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends
from qdrant_client import AsyncQdrantClient
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_app_settings, get_db_session, get_orchestrator, get_qdrant
from src.api.schemas import (
    IngestionPauseResumeResponse,
    MemoryReclaimResponse,
    MemoryStatsResponse,
    ProviderOption,
    SystemHealthResponse,
    SystemStatsResponse,
)
from src.config import Settings
from src.db.models import ChatThread, Document, DocumentChunk
from src.ingestion.indexer import release_cuda_cache, resolve_compute_device, resolve_sparse_mode
from src.ingestion.classifier import is_classifier_resident, get_classifier_model_name
from src.ingestion.worker import IngestionOrchestrator
from src.rag.llm import _load_dotenv_if_present, is_valid_api_key

router = APIRouter(prefix="/system", tags=["system"])


def _get_process_rss_mb() -> float:
    """Return current process RSS in MiB with Linux /proc and Unix getrusage fallback."""
    # 1. Linux /proc/self/status (accurate current RSS inside Docker / Linux)
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

    # 2. resource.getrusage fallback (Unix max RSS)
    try:
        import resource
        import sys

        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # On macOS ru_maxrss is in bytes, on Linux in KiB
        if sys.platform == "darwin":
            return round(rusage.ru_maxrss / (1024.0 * 1024.0), 2)
        return round(rusage.ru_maxrss / 1024.0, 2)
    except Exception:
        pass

    return 0.0


def collect_memory_stats(
    orchestrator: IngestionOrchestrator | None = None,
    settings: Settings | None = None,
) -> MemoryStatsResponse:
    """Collect host process RSS, PyTorch CUDA memory metrics, and orchestrator worker pool telemetry."""
    rss_mb = _get_process_rss_mb()
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
        from src.ingestion.classifier import is_classifier_resident, get_classifier_model_name
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


def _build_provider_options(settings: Settings) -> tuple[list[ProviderOption], list[ProviderOption]]:
    """Inspect environment and settings (without exposing secrets) to report provider availability."""
    _load_dotenv_if_present()
    active_provider = (os.environ.get("LLM_PROVIDER") or settings.llm.provider or "openai").strip().lower()
    active_model = (os.environ.get("LLM_MODEL_NAME") or settings.llm.model_name or "").strip()
    prov_compat = settings.llm.providers.get("openai_compatible")
    has_base_url = bool(
        os.environ.get("LLM_BASE_URL")
        or settings.llm.base_url
        or (prov_compat and prov_compat.base_url)
    )
    has_compat_key = any(
        is_valid_api_key(os.environ.get(k))
        for k in ("LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
    )

    definitions = [
        (
            "openai",
            "OpenAI",
            is_valid_api_key(os.environ.get("OPENAI_API_KEY")),
        ),
        (
            "anthropic",
            "Anthropic",
            is_valid_api_key(os.environ.get("ANTHROPIC_API_KEY")),
        ),
        (
            "deepseek",
            "DeepSeek",
            is_valid_api_key(os.environ.get("DEEPSEEK_API_KEY")),
        ),
        (
            "openai_compatible",
            "OpenAI-Compatible (Custom / GLM / vLLM)",
            bool(has_base_url and has_compat_key),
        ),
    ]

    all_options: list[ProviderOption] = []
    for prov_id, label, is_configured in definitions:
        prov_entry = settings.llm.providers.get(prov_id)
        entry_model = (prov_entry.model_name if prov_entry else "").strip()
        default_model = entry_model or (active_model if active_provider == prov_id else "")
        all_options.append(
            ProviderOption(
                provider=prov_id,
                label=label,
                configured=is_configured,
                default_model=default_model,
            )
        )

    configured_only = [opt for opt in all_options if opt.configured]
    return configured_only, all_options


@router.get("/health", response_model=SystemHealthResponse)
async def system_health(
    session: AsyncSession = Depends(get_db_session),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    settings: Settings = Depends(get_app_settings),
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


@router.get("/memory", response_model=MemoryStatsResponse)
async def system_memory(
    orchestrator: IngestionOrchestrator = Depends(get_orchestrator),
    settings: Settings = Depends(get_app_settings),
) -> MemoryStatsResponse:
    """Return live host process RSS, PyTorch CUDA memory telemetry, and worker pool state."""
    return collect_memory_stats(orchestrator, settings)


@router.post("/memory/reclaim", response_model=MemoryReclaimResponse)
async def system_memory_reclaim(
    orchestrator: IngestionOrchestrator = Depends(get_orchestrator),
    settings: Settings = Depends(get_app_settings),
) -> MemoryReclaimResponse:
    """Explicit on-demand memory reclamation: shut down idle worker pool and flush CUDA cache."""
    reclaimed = orchestrator.reclaim_worker_pool(wait=False, force=False)
    worker_state = orchestrator.get_worker_state()
    active_indexing_jobs = worker_state.get("active_indexing_jobs", 0)
    if active_indexing_jobs == 0:
        cuda_cleared = release_cuda_cache()
    else:
        cuda_cleared = False
    stats = collect_memory_stats(orchestrator, settings)
    return MemoryReclaimResponse(
        status="ok",
        worker_pool_reclaimed=reclaimed,
        deferred=worker_state.get("reclaim_deferred", False),
        cuda_cache_cleared=cuda_cleared,
        active_worker_jobs=worker_state.get("active_worker_jobs", 0),
        memory=stats,
    )


@router.get("/stats", response_model=SystemStatsResponse)
async def system_stats(
    session: AsyncSession = Depends(get_db_session),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    orchestrator: IngestionOrchestrator = Depends(get_orchestrator),
    settings: Settings = Depends(get_app_settings),
) -> SystemStatsResponse:
    """Return document/chunk counts, Qdrant vector count, and read-only active model configuration."""
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

    configured_providers, available_providers = _build_provider_options(settings)

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
            "base_url": os.environ.get("LLM_BASE_URL") or settings.llm.base_url,
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
        memory=collect_memory_stats(orchestrator, settings),
        configured_providers=configured_providers,
        available_providers=available_providers,
    )


@router.post("/ingestion/pause", response_model=IngestionPauseResumeResponse)
async def pause_ingestion(
    orchestrator: IngestionOrchestrator = Depends(get_orchestrator),
) -> IngestionPauseResumeResponse:
    """Pause ingestion dispatch. Running jobs drain cleanly; subsequent queued jobs wait."""
    orchestrator.pause()
    return IngestionPauseResumeResponse(status="ok", ingestion_paused=True)


@router.post("/ingestion/resume", response_model=IngestionPauseResumeResponse)
async def resume_ingestion(
    orchestrator: IngestionOrchestrator = Depends(get_orchestrator),
) -> IngestionPauseResumeResponse:
    """Resume ingestion dispatch for held queued jobs."""
    orchestrator.resume()
    return IngestionPauseResumeResponse(status="ok", ingestion_paused=False)
