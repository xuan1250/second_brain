"""System health check, Qdrant/SQLite statistics, memory telemetry, hardware, and compute profile routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_app_settings, get_db_session, get_orchestrator, get_qdrant
from src.api.schemas import (
    BootstrapStatusResponse,
    ComputeProfileRequest,
    ComputeProfileResponse,
    HardwareReportResponse,
    IngestionPauseResumeResponse,
    MemoryReclaimResponse,
    MemoryStatsResponse,
    ReindexCorpusResponse,
    SystemHealthResponse,
    SystemStatsResponse,
)
from src.config import Settings
from src.desktop.hardware import probe_system_hardware
from src.ingestion.indexer import release_cuda_cache
from src.ingestion.worker import IngestionOrchestrator
from src.services.compute_profile_service import (
    get_bootstrap_status,
    get_hardware_report,
    update_compute_profile_settings,
)
from src.services.reindex_service import queue_documents_for_reindex
from src.services.telemetry_service import (
    build_system_stats,
    check_system_health,
    collect_memory_stats as _collect_memory_stats_impl,
    get_process_rss_mb,
    reclaim_system_memory,
)

router = APIRouter(prefix="/system", tags=["system"])


def _get_process_rss_mb() -> float:
    """Forwarding seam for process RSS inspection."""
    return get_process_rss_mb()


def collect_memory_stats(
    orchestrator: IngestionOrchestrator | None = None,
    settings: Settings | None = None,
) -> MemoryStatsResponse:
    """Forwarding seam for memory statistics collection."""
    return _collect_memory_stats_impl(
        orchestrator=orchestrator,
        settings=settings,
        rss_fn=_get_process_rss_mb,
    )


@router.get("/health", response_model=SystemHealthResponse)
async def system_health(
    session: AsyncSession = Depends(get_db_session),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    settings: Settings = Depends(get_app_settings),
) -> SystemHealthResponse:
    """Verify reachability of both SQLite and Qdrant."""
    return await check_system_health(session, qdrant, settings)


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
    return await reclaim_system_memory(
        orchestrator,
        settings,
        release_cuda_fn=release_cuda_cache,
        rss_fn=_get_process_rss_mb,
    )


@router.get("/stats", response_model=SystemStatsResponse)
async def system_stats(
    session: AsyncSession = Depends(get_db_session),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    orchestrator: IngestionOrchestrator = Depends(get_orchestrator),
    settings: Settings = Depends(get_app_settings),
) -> SystemStatsResponse:
    """Return document/chunk counts, Qdrant vector count, and read-only active model configuration."""
    return await build_system_stats(
        session,
        qdrant,
        orchestrator,
        settings,
        memory_fn=collect_memory_stats,
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


@router.get("/hardware", response_model=HardwareReportResponse)
async def system_hardware(
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    settings: Settings = Depends(get_app_settings),
) -> HardwareReportResponse:
    """Inspect host CPU, RAM, GPUs, disk capacity, and active compute/sparse-mode configuration."""
    return await get_hardware_report(
        qdrant,
        settings,
        probe_fn=probe_system_hardware,
    )


@router.get("/bootstrap-status", response_model=BootstrapStatusResponse)
async def system_bootstrap_status(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> BootstrapStatusResponse:
    """Return offline readiness across all 6 model groups, disk usage, and any active download progress."""
    return get_bootstrap_status(request.app.state, settings)


@router.post("/reindex", response_model=ReindexCorpusResponse)
async def reindex_corpus(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    orchestrator: IngestionOrchestrator = Depends(get_orchestrator),
    settings: Settings = Depends(get_app_settings),
) -> ReindexCorpusResponse:
    """Queue all ready documents for background re-indexing under the active sparse/dense profile."""
    queued_count, skipped_count = await queue_documents_for_reindex(
        app_state=request.app.state,
        session=session,
        orchestrator=orchestrator,
        settings=settings,
    )
    msg = (
        f"Queued {queued_count} document(s) for background re-indexing."
        if queued_count > 0
        else "No ready documents required re-indexing."
    )
    if skipped_count > 0:
        msg += f" Skipped {skipped_count} document(s) without cached checkpoint or raw source file."
    return ReindexCorpusResponse(
        status="ok",
        queued_count=queued_count,
        skipped_count=skipped_count,
        message=msg,
    )


@router.post("/compute-profile", response_model=ComputeProfileResponse)
async def update_compute_profile(
    payload: ComputeProfileRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    orchestrator: IngestionOrchestrator = Depends(get_orchestrator),
    settings: Settings = Depends(get_app_settings),
) -> ComputeProfileResponse:
    """Validate and persist CPU/CUDA compute profile selection with sparse-mode corpus protection."""
    return await update_compute_profile_settings(
        payload,
        app_state=request.app.state,
        session=session,
        qdrant=qdrant,
        orchestrator=orchestrator,
        settings=settings,
        probe_fn=probe_system_hardware,
    )
