"""System health check, Qdrant/SQLite statistics, memory telemetry, and read-only provider configuration routes."""
from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from qdrant_client import AsyncQdrantClient
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_app_settings, get_db_session, get_orchestrator, get_qdrant
from src.api.schemas import (
    BootstrapStatusResponse,
    ComputeProfileRequest,
    ComputeProfileResponse,
    GpuAdapterSchema,
    HardwareReportResponse,
    IngestionPauseResumeResponse,
    MemoryReclaimResponse,
    MemoryStatsResponse,
    ModelStatusSchema,
    ProviderOption,
    ReindexCorpusResponse,
    SystemHealthResponse,
    SystemStatsResponse,
)
from src.config import Settings
from src.db.models import ChatThread, Document, DocumentChunk, IngestionJob
from src.desktop.bootstrap import RuntimeBootstrapService, detect_collection_sparse_mode
from src.desktop.config import DesktopConfig, DesktopPaths, load_desktop_config, save_desktop_config
from src.desktop.hardware import probe_system_hardware
from src.ingestion.indexer import release_cuda_cache, resolve_compute_device, resolve_sparse_mode
from src.ingestion.classifier import is_classifier_resident, get_classifier_model_name
from src.ingestion.parser import load_parsed_checkpoint
from src.ingestion.worker import IngestionOrchestrator
from src.rag.llm import (
    _load_dotenv_if_present,
    get_provider_validation_state,
    is_valid_api_key,
    mask_api_key,
    resolve_provider_credential,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["system"])


def _get_process_rss_mb() -> float:
    """Return current process RSS in MiB with Linux /proc, Win32 psapi, and Unix getrusage fallback."""
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

    # 2. Windows Win32 GetProcessMemoryInfo (WorkingSetSize)
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

            get_mem_info = None
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

    # 3. resource.getrusage fallback (Unix max RSS)
    try:
        import resource

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

    definitions = [
        ("openai", "OpenAI"),
        ("anthropic", "Anthropic"),
        ("deepseek", "DeepSeek"),
        ("openai_compatible", "OpenAI-Compatible (Custom / GLM / vLLM)"),
    ]

    all_options: list[ProviderOption] = []
    for prov_id, label in definitions:
        resolved_key, key_source, resolved_base_url = resolve_provider_credential(prov_id, settings)
        if prov_id == "openai_compatible":
            is_configured = bool(resolved_key and resolved_base_url)
        else:
            is_configured = bool(resolved_key)

        masked = mask_api_key(resolved_key)
        val_status, val_err = get_provider_validation_state(
            prov_id, resolved_key, is_configured, resolved_base_url
        )

        prov_entry = settings.llm.providers.get(prov_id)
        entry_model = (prov_entry.model_name if prov_entry else "").strip()
        default_model = entry_model or (active_model if active_provider == prov_id else "")
        all_options.append(
            ProviderOption(
                provider=prov_id,
                label=label,
                configured=is_configured,
                default_model=default_model,
                key_source=key_source,
                masked_key=masked,
                validation_status=val_status,
                validation_error=val_err,
                base_url=resolved_base_url if prov_id == "openai_compatible" else resolved_base_url,
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
    active_worker_jobs = worker_state.get("active_worker_jobs", 0)
    active_indexing_jobs = worker_state.get("active_indexing_jobs", 0)
    if active_indexing_jobs == 0:
        cuda_cleared = release_cuda_cache()
    else:
        cuda_cleared = False

    if active_worker_jobs == 0 and active_indexing_jobs == 0:
        try:
            from src.rag.graph import compact_chat_history_db
            from src.db.session import get_engine, vacuum_sqlite_db

            await compact_chat_history_db(run_vacuum=True)
            await vacuum_sqlite_db(get_engine())
        except Exception as exc:
            logger.warning("Failed to compact/vacuum SQLite during memory reclaim: %s", exc)

    stats = collect_memory_stats(orchestrator, settings)
    return MemoryReclaimResponse(
        status="ok",
        worker_pool_reclaimed=reclaimed,
        deferred=worker_state.get("reclaim_deferred", False),
        cuda_cache_cleared=cuda_cleared,
        active_worker_jobs=active_worker_jobs,
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


def _resolve_desktop_context(settings: Settings) -> tuple[DesktopPaths, DesktopConfig]:
    """Resolve DesktopPaths and DesktopConfig, isolating pytest runs from host `%LOCALAPPDATA%`."""
    desktop_home = (os.environ.get("SECOND_BRAIN_DESKTOP_HOME") or "").strip()
    if desktop_home:
        paths = DesktopPaths()
        return paths, load_desktop_config(paths)

    if "PYTEST_CURRENT_TEST" in os.environ:
        paths = DesktopPaths(
            user_data_dir=settings.data_dir.parent,
            data_dir_override=settings.data_dir,
        )
        return paths, load_desktop_config(paths)

    paths = DesktopPaths()
    config = load_desktop_config(paths)
    if (os.environ.get("DATA_DIR") or "").strip() and not (config.data_dir_override or "").strip():
        paths = DesktopPaths(data_dir_override=settings.data_dir)
    return paths, config


@router.get("/hardware", response_model=HardwareReportResponse)
async def system_hardware(
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    settings: Settings = Depends(get_app_settings),
) -> HardwareReportResponse:
    """Inspect host CPU, RAM, GPUs, disk capacity, and active compute/sparse-mode configuration."""
    paths, config = _resolve_desktop_context(settings)
    report = probe_system_hardware(data_dir=paths.resolve_data_dir(config))
    collection_sparse_mode = await detect_collection_sparse_mode(qdrant, settings.qdrant_collection)
    sparse_mode_pinned = config.sparse_mode_override in ("bm25", "bge_m3")

    payload = report.model_dump()
    payload["gpus"] = [GpuAdapterSchema(**g) for g in payload.get("gpus", [])]
    return HardwareReportResponse(
        **payload,
        active_compute_profile=config.compute_profile,
        cuda_runtime_ready=config.cuda_runtime_ready,
        active_sparse_mode_override=config.sparse_mode_override,
        collection_sparse_mode=collection_sparse_mode,
        sparse_mode_pinned=sparse_mode_pinned,
    )


@router.get("/bootstrap-status", response_model=BootstrapStatusResponse)
async def system_bootstrap_status(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> BootstrapStatusResponse:
    """Return offline readiness across all 6 model groups, disk usage, and any active download progress."""
    paths, config = _resolve_desktop_context(settings)
    bootstrap = RuntimeBootstrapService(paths=paths, config=config)
    status = bootstrap.inspect_models_status()

    raw_progress = getattr(request.app.state, "bootstrap_progress", None)
    progress_dict: dict[str, Any] | None = None
    if raw_progress is not None:
        if hasattr(raw_progress, "model_dump"):
            progress_dict = raw_progress.model_dump()
        elif isinstance(raw_progress, dict):
            progress_dict = dict(raw_progress)

    return BootstrapStatusResponse(
        models_ready=bool(status["models_ready"]),
        cuda_runtime_ready=config.cuda_runtime_ready,
        compute_profile=config.compute_profile,
        total_size_mb=float(status["total_size_mb"]),
        models_dir=str(status["models_dir"]),
        models=[
            ModelStatusSchema(**(m.model_dump() if hasattr(m, "model_dump") else m))
            for m in status["models"]
        ],
        progress=progress_dict,
    )


async def _run_corpus_reindex_job(
    orchestrator: IngestionOrchestrator,
    *,
    doc_id: str,
    job_id: str,
    raw_path: Any | None,
    has_checkpoint: bool,
) -> None:
    """Execute background re-chunking/re-indexing for a single document."""
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
        pass


async def _queue_corpus_reindex(
    request: Request,
    session: AsyncSession,
    orchestrator: IngestionOrchestrator,
    settings: Settings,
) -> tuple[int, int]:
    """Queue all ready documents for background sparse/vector re-indexing."""
    ready_docs = (
        await session.execute(select(Document).where(Document.status == "ready"))
    ).scalars().all()

    queued_count = 0
    skipped_count = 0
    bg_tasks: set[asyncio.Task[Any]] | None = getattr(request.app.state, "background_tasks", None)
    if bg_tasks is None:
        bg_tasks = set()
        request.app.state.background_tasks = bg_tasks

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
        has_raw = raw_dest is not None and raw_dest.is_file()
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

        task = asyncio.create_task(
            _run_corpus_reindex_job(
                orchestrator,
                doc_id=doc.id,
                job_id=new_job.id,
                raw_path=raw_dest if has_raw else None,
                has_checkpoint=cached is not None,
            )
        )
        bg_tasks.add(task)
        task.add_done_callback(bg_tasks.discard)
        queued_count += 1

    return queued_count, skipped_count


@router.post("/reindex", response_model=ReindexCorpusResponse)
async def reindex_corpus(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    orchestrator: IngestionOrchestrator = Depends(get_orchestrator),
    settings: Settings = Depends(get_app_settings),
) -> ReindexCorpusResponse:
    """Queue all ready documents for background re-indexing under the active sparse/dense profile."""
    queued_count, skipped_count = await _queue_corpus_reindex(request, session, orchestrator, settings)
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
    paths, config = _resolve_desktop_context(settings)
    previous_profile = config.compute_profile
    previous_sparse_override = config.sparse_mode_override
    target_profile = payload.profile

    if target_profile == "cuda":
        report = probe_system_hardware(data_dir=paths.resolve_data_dir(config))
        if not report.has_nvidia_cuda_capable_gpu and not payload.force:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Cannot switch to CUDA profile without an NVIDIA GPU (>= 4.0 GB VRAM). "
                    f"{report.recommendation_reason} Pass force=true to override."
                ),
            )
        if not report.cuda_disk_sufficient and not payload.force:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"CUDA 12.4 wheels require at least 12.0 GB free disk space "
                    f"(found {report.free_disk_gb:.1f} GB). Pass force=true to override."
                ),
            )

    collection_sparse_mode = await detect_collection_sparse_mode(qdrant, settings.qdrant_collection)
    reindex_recommended = RuntimeBootstrapService._apply_sparse_mode_policy(
        config,
        target_profile=target_profile,
        existing_sparse_mode=collection_sparse_mode,
        reindex_sparse=payload.reindex_sparse,
    )

    config.compute_profile = target_profile
    save_desktop_config(config, paths)

    wheel_install_required = target_profile == "cuda" and not config.cuda_runtime_ready
    restart_required = (
        target_profile != previous_profile
        or wheel_install_required
        or config.sparse_mode_override != previous_sparse_override
    )

    reindex_queued = 0
    if payload.reindex_sparse and not restart_required:
        reindex_queued, _ = await _queue_corpus_reindex(request, session, orchestrator, settings)

    if wheel_install_required:
        message = (
            "Compute profile set to 'cuda'. CUDA 12.4 runtime wheels must be installed "
            "and the backend sidecar restarted."
        )
    elif restart_required:
        message = (
            f"Compute profile updated from '{previous_profile}' to '{target_profile}'. "
            "Restart the backend sidecar to apply."
        )
        if payload.reindex_sparse:
            message += " Invoke POST /api/v1/system/reindex after restart to re-index existing documents."
    else:
        message = f"Compute profile remains '{target_profile}'."
        if reindex_queued > 0:
            message += f" Queued {reindex_queued} document(s) for background sparse re-indexing."

    if reindex_recommended:
        message += " Existing Qdrant points are pinned to 'bm25'; opt into re-indexing to enable learned BGE-M3 sparse vectors."

    return ComputeProfileResponse(
        status="ok",
        compute_profile=target_profile,
        previous_profile=previous_profile,
        restart_required=restart_required,
        wheel_install_required=wheel_install_required,
        cuda_runtime_ready=config.cuda_runtime_ready,
        sparse_mode_override=config.sparse_mode_override,
        collection_sparse_mode=collection_sparse_mode,
        reindex_recommended=reindex_recommended,
        reindex_queued=reindex_queued,
        message=message,
    )

