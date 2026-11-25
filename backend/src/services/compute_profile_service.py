"""Desktop hardware probing, model bootstrap readiness, and compute profile switching service."""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    BootstrapStatusResponse,
    ComputeProfileRequest,
    ComputeProfileResponse,
    GpuAdapterSchema,
    HardwareReportResponse,
    ModelStatusSchema,
)
from src.config import Settings
from src.desktop.bootstrap import (
    RuntimeBootstrapService,
    apply_sparse_mode_policy,
    detect_collection_sparse_mode,
)
from src.desktop.config import DesktopConfig, DesktopPaths, load_desktop_config, save_desktop_config
from src.desktop.hardware import probe_system_hardware
from src.ingestion.worker import IngestionOrchestrator
from src.services.reindex_service import queue_documents_for_reindex


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


async def get_hardware_report(
    qdrant: AsyncQdrantClient,
    settings: Settings,
    *,
    probe_fn: Callable[..., Any] = probe_system_hardware,
) -> HardwareReportResponse:
    """Inspect host hardware and combine with active desktop compute/sparse-mode settings."""
    paths, config = _resolve_desktop_context(settings)
    report = probe_fn(data_dir=paths.resolve_data_dir(config))
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


def get_bootstrap_status(
    app_state: Any,
    settings: Settings,
) -> BootstrapStatusResponse:
    """Return offline readiness across all 6 model groups, disk usage, and active progress."""
    paths, config = _resolve_desktop_context(settings)
    bootstrap = RuntimeBootstrapService(paths=paths, config=config)
    status_info = bootstrap.inspect_models_status()

    raw_progress = getattr(app_state, "bootstrap_progress", None)
    progress_dict: dict[str, Any] | None = None
    if raw_progress is not None:
        if hasattr(raw_progress, "model_dump"):
            progress_dict = raw_progress.model_dump()
        elif isinstance(raw_progress, dict):
            progress_dict = dict(raw_progress)

    return BootstrapStatusResponse(
        models_ready=bool(status_info["models_ready"]),
        cuda_runtime_ready=config.cuda_runtime_ready,
        compute_profile=config.compute_profile,
        total_size_mb=float(status_info["total_size_mb"]),
        models_dir=str(status_info["models_dir"]),
        models=[
            ModelStatusSchema(**(m.model_dump() if hasattr(m, "model_dump") else m))
            for m in status_info["models"]
        ],
        progress=progress_dict,
    )


async def update_compute_profile_settings(
    payload: ComputeProfileRequest,
    *,
    app_state: Any,
    session: AsyncSession,
    qdrant: AsyncQdrantClient,
    orchestrator: IngestionOrchestrator,
    settings: Settings,
    probe_fn: Callable[..., Any] = probe_system_hardware,
) -> ComputeProfileResponse:
    """Validate and persist CPU/CUDA compute profile selection with sparse-mode corpus protection."""
    paths, config = _resolve_desktop_context(settings)
    previous_profile = config.compute_profile
    previous_sparse_override = config.sparse_mode_override
    target_profile = payload.profile

    if target_profile == "cuda":
        report = probe_fn(data_dir=paths.resolve_data_dir(config))
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
    reindex_recommended = apply_sparse_mode_policy(
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
        reindex_queued, _ = await queue_documents_for_reindex(
            app_state=app_state,
            session=session,
            orchestrator=orchestrator,
            settings=settings,
        )

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
        message += (
            " Existing Qdrant points are pinned to 'bm25'; opt into re-indexing to enable learned BGE-M3 sparse vectors."
        )

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
