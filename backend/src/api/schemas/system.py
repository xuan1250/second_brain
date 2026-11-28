"""System health, memory telemetry, hardware profile, model bootstrap, and reindex schemas."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.api.schemas.settings import ProviderOption


class SystemHealthResponse(BaseModel):
    """Health check status for SQLite and Qdrant."""

    status: str = "ok"
    database: str = "ok"
    qdrant: str = "ok"
    collection: str


class MemoryStatsResponse(BaseModel):
    """Host process RSS, PyTorch CUDA memory telemetry, and orchestrator worker pool metrics."""

    process_rss_mb: float = 0.0
    cuda_available: bool = False
    cuda_allocated_mb: float = 0.0
    cuda_reserved_mb: float = 0.0
    cuda_driver_free_mb: float | None = None
    cuda_driver_total_mb: float | None = None
    worker_pool_active: bool = False
    active_worker_jobs: int = 0
    active_indexing_jobs: int = 0
    worker_reclaim_count: int = 0
    reclaim_deferred: bool = False
    worker_idle_timeout_sec: float = 60.0
    worker_max_tasks: int = 0
    classifier_resident: bool = False
    classifier_model_name: str | None = None


class MemoryReclaimResponse(BaseModel):
    """Outcome of an on-demand memory / VRAM reclamation request."""

    status: str = "ok"
    worker_pool_reclaimed: bool = False
    deferred: bool = False
    cuda_cache_cleared: bool = False
    active_worker_jobs: int = 0
    memory: MemoryStatsResponse


class IngestionPauseResumeResponse(BaseModel):
    """Payload returned by ingestion pause/resume control endpoints."""

    status: str = "ok"
    ingestion_paused: bool


class SystemStatsResponse(BaseModel):
    """System-wide counts, Qdrant vector statistics, and active read-only model config."""

    total_documents: int = 0
    ready_documents: int = 0
    total_chunks: int = 0
    total_threads: int = 0
    qdrant_vectors_count: int = 0
    vector_count: int = 0
    ingestion_paused: bool = False
    documents_by_status: dict[str, int] = Field(default_factory=dict)
    llm: dict[str, Any] = Field(default_factory=dict)
    embedding: dict[str, Any] = Field(default_factory=dict)
    reranker: dict[str, Any] = Field(default_factory=dict)
    ingestion: dict[str, Any] = Field(default_factory=dict)
    classifier: dict[str, Any] = Field(default_factory=dict)
    memory: MemoryStatsResponse | None = None
    configured_providers: list[ProviderOption] = Field(default_factory=list)
    available_providers: list[ProviderOption] = Field(default_factory=list)


class GpuAdapterSchema(BaseModel):
    """Serialized host GPU adapter capabilities."""

    model_config = ConfigDict(from_attributes=True)

    name: str
    vram_gb: float | None = None
    vendor: str = "Unknown"
    driver_version: str | None = None
    is_nvidia: bool = False
    cuda_capable: bool = False


class HardwareReportResponse(BaseModel):
    """Live system hardware report merged with desktop compute profile and sparse-mode state."""

    cpu_name: str
    logical_cores: int
    physical_cores: int | None = None
    total_ram_gb: float
    available_ram_gb: float
    ram_status: Literal["insufficient", "limited", "optimal"]
    gpus: list[GpuAdapterSchema] = Field(default_factory=list)
    has_nvidia_cuda_capable_gpu: bool = False
    gpu_status: Literal["cuda_compatible", "nvidia_low_vram", "cpu_only"]
    target_drive: str
    free_disk_gb: float
    disk_status: Literal["insufficient", "limited", "optimal"]
    cuda_disk_sufficient: bool
    is_onedrive_path: bool = False
    onedrive_warning: str | None = None
    recommended_profile: Literal["cpu", "cuda"]
    recommended_upload_mode: Literal["fast", "auto"]
    recommendation_reason: str
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    active_compute_profile: Literal["cpu", "cuda"] = "cpu"
    cuda_runtime_ready: bool = False
    active_sparse_mode_override: Literal["auto", "bge_m3", "bm25"] | None = None
    collection_sparse_mode: Literal["bm25", "bge_m3"] | None = None
    sparse_mode_pinned: bool = False


class ModelStatusSchema(BaseModel):
    """Per-model group cache readiness and size telemetry."""

    model_config = ConfigDict(from_attributes=True)

    key: str
    display_name: str
    cached: bool
    path: str
    size_mb: float = 0.0
    detail: str = ""


class BootstrapStatusResponse(BaseModel):
    """Offline AI model cache readiness across all 6 model groups plus active progress."""

    models_ready: bool
    cuda_runtime_ready: bool = False
    compute_profile: Literal["cpu", "cuda"] = "cpu"
    total_size_mb: float = 0.0
    models_dir: str
    models: list[ModelStatusSchema] = Field(default_factory=list)
    progress: dict[str, Any] | None = None


class ComputeProfileRequest(BaseModel):
    """Request payload to switch the active CPU / CUDA compute runtime profile."""

    profile: Literal["cpu", "cuda"]
    force: bool = False
    reindex_sparse: bool = False


class ComputeProfileResponse(BaseModel):
    """Result of updating the desktop compute runtime profile and sparse-mode pinning."""

    status: str = "ok"
    compute_profile: Literal["cpu", "cuda"]
    previous_profile: Literal["cpu", "cuda"]
    restart_required: bool
    wheel_install_required: bool
    cuda_runtime_ready: bool
    sparse_mode_override: Literal["auto", "bge_m3", "bm25"] | None = None
    collection_sparse_mode: Literal["bm25", "bge_m3"] | None = None
    reindex_recommended: bool = False
    reindex_queued: int = 0
    message: str


class ReindexCorpusResponse(BaseModel):
    """Result of queuing ready documents for sparse/vector re-indexing."""

    status: str = "ok"
    queued_count: int = 0
    skipped_count: int = 0
    message: str
