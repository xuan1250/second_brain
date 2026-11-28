"""Windows hardware capability probes and CPU vs. NVIDIA CUDA compute recommendation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.desktop.config import DesktopPaths
from src.desktop.hardware_probes import (
    _probe_cpu,
    _probe_disk,
    _probe_memory,
    _probe_nvidia_smi,
    _probe_wmi_gpus,
)


class GpuAdapterInfo(BaseModel):
    """Normalized metadata for a single host GPU adapter."""

    name: str
    vram_gb: float | None = None
    vendor: str = "Unknown"
    driver_version: str | None = None
    is_nvidia: bool = False
    cuda_capable: bool = False


class HardwareRecommendation(BaseModel):
    """Evaluated threshold statuses and recommended runtime profile."""

    ram_status: Literal["insufficient", "limited", "optimal"]
    has_nvidia_cuda_capable_gpu: bool
    gpu_status: Literal["cuda_compatible", "nvidia_low_vram", "cpu_only"]
    disk_status: Literal["insufficient", "limited", "optimal"]
    cuda_disk_sufficient: bool
    recommended_profile: Literal["cpu", "cuda"]
    recommended_upload_mode: Literal["fast", "auto"]
    recommendation_reason: str
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HardwareReport(BaseModel):
    """Full system hardware diagnostics report for installer and desktop UI."""

    cpu_name: str
    logical_cores: int
    physical_cores: int | None = None
    total_ram_gb: float
    available_ram_gb: float
    ram_status: Literal["insufficient", "limited", "optimal"]
    gpus: list[GpuAdapterInfo] = Field(default_factory=list)
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


def _probe_gpus() -> list[GpuAdapterInfo]:
    """Combine `nvidia-smi` fast path with WMI fallback without double-counting NVIDIA GPUs."""
    nvidia_gpus = _probe_nvidia_smi()
    wmi_gpus = _probe_wmi_gpus()

    if not nvidia_gpus:
        return wmi_gpus

    merged = list(nvidia_gpus)
    for adapter in wmi_gpus:
        if adapter.is_nvidia:
            # Skip WMI NVIDIA entries when nvidia-smi already reported authoritative VRAM.
            continue
        merged.append(adapter)
    return merged


def _check_onedrive_path(target_path: Path) -> tuple[bool, str | None]:
    """Detect whether `target_path` resides under a OneDrive-synced folder."""
    abs_target = os.path.abspath(str(target_path))
    norm_target = Path(os.path.normcase(abs_target))

    for env_key in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
        env_val = (os.environ.get(env_key) or "").strip()
        if not env_val:
            continue
        norm_root = Path(os.path.normcase(os.path.abspath(env_val)))
        try:
            if norm_target == norm_root or norm_target.is_relative_to(norm_root):
                return (
                    True,
                    f"Target data directory '{target_path}' is inside OneDrive ('{env_val}'). "
                    "SQLite WAL locks and large Qdrant/model files can suffer sync conflicts; "
                    "a local non-OneDrive directory is recommended.",
                )
        except ValueError:
            continue

    for part in target_path.parts:
        lower_part = part.strip().lower()
        if (
            lower_part == "onedrive"
            or lower_part.startswith("onedrive -")
            or lower_part.startswith("onedrive ")
        ):
            return (
                True,
                f"Target data directory '{target_path}' appears to be inside a OneDrive folder. "
                "SQLite WAL locks and large Qdrant/model files can suffer sync conflicts; "
                "a local non-OneDrive directory is recommended.",
            )

    return False, None


def recommend_compute_profile(
    total_ram_gb: float,
    gpus: list[GpuAdapterInfo],
    free_disk_gb: float,
    *,
    is_onedrive_path: bool = False,
    onedrive_warning: str | None = None,
) -> HardwareRecommendation:
    """Evaluate RAM, GPU/VRAM, disk, and OneDrive thresholds for CPU vs. CUDA recommendation."""
    blockers: list[str] = []
    warnings: list[str] = []

    if total_ram_gb < 8.0:
        ram_status: Literal["insufficient", "limited", "optimal"] = "insufficient"
        recommended_upload_mode: Literal["fast", "auto"] = "fast"
        blockers.append(
            f"System RAM is {total_ram_gb:.1f} GB (< 8.0 GB minimum required for local embeddings and Docling)."
        )
    elif total_ram_gb < 16.0:
        ram_status = "limited"
        recommended_upload_mode = "fast"
        warnings.append(
            f"System RAM is {total_ram_gb:.1f} GB (< 16.0 GB). Fast upload mode is recommended to reduce peak memory."
        )
    else:
        ram_status = "optimal"
        recommended_upload_mode = "auto"

    cuda_disk_sufficient = free_disk_gb >= 12.0
    if free_disk_gb < 8.0:
        disk_status: Literal["insufficient", "limited", "optimal"] = "insufficient"
        blockers.append(
            f"Free disk space is {free_disk_gb:.1f} GB (< 8.0 GB minimum required for CPU mode and models)."
        )
    elif free_disk_gb < 15.0:
        disk_status = "limited"
        if not cuda_disk_sufficient:
            warnings.append(
                f"Free disk space is {free_disk_gb:.1f} GB (< 12.0 GB required for CUDA 12.4 wheels, < 15.0 GB recommended)."
            )
        else:
            warnings.append(
                f"Free disk space is {free_disk_gb:.1f} GB (< 15.0 GB recommended for comfortable document growth)."
            )
    else:
        disk_status = "optimal"

    nvidia_gpus = [g for g in gpus if g.is_nvidia]
    capable_nvidia = [g for g in nvidia_gpus if g.vram_gb is not None and g.vram_gb >= 4.0]

    if capable_nvidia:
        has_nvidia_cuda_capable_gpu = True
        gpu_status: Literal["cuda_compatible", "nvidia_low_vram", "cpu_only"] = "cuda_compatible"
        best_gpu = max(capable_nvidia, key=lambda g: g.vram_gb or 0.0)
        vram_val = best_gpu.vram_gb or 0.0
        if cuda_disk_sufficient:
            recommended_profile: Literal["cpu", "cuda"] = "cuda"
            recommendation_reason = (
                f"Detected NVIDIA GPU '{best_gpu.name}' with {vram_val:.1f} GB VRAM (>= 4.0 GB) "
                f"and {free_disk_gb:.1f} GB free disk space; NVIDIA CUDA 12.4 mode is recommended."
            )
        else:
            recommended_profile = "cpu"
            recommendation_reason = (
                f"Detected CUDA-capable NVIDIA GPU '{best_gpu.name}' ({vram_val:.1f} GB VRAM), "
                f"but only {free_disk_gb:.1f} GB disk space is free (< 12.0 GB required for CUDA 12.4 wheels); "
                "Standard CPU mode is recommended instead."
            )
    elif nvidia_gpus:
        has_nvidia_cuda_capable_gpu = False
        gpu_status = "nvidia_low_vram"
        recommended_profile = "cpu"
        best_gpu = max(nvidia_gpus, key=lambda g: g.vram_gb or 0.0)
        vram_desc = f"{best_gpu.vram_gb:.1f} GB" if best_gpu.vram_gb is not None else "unknown"
        recommendation_reason = (
            f"Detected NVIDIA GPU '{best_gpu.name}' ({vram_desc} VRAM), below the 4.0 GB VRAM threshold "
            "for CUDA acceleration; Standard CPU mode is recommended."
        )
    else:
        has_nvidia_cuda_capable_gpu = False
        gpu_status = "cpu_only"
        recommended_profile = "cpu"
        if gpus:
            names = ", ".join(g.name for g in gpus)
            recommendation_reason = (
                f"No NVIDIA CUDA GPU detected (found: {names}); Standard CPU mode is recommended."
            )
        else:
            recommendation_reason = (
                "No discrete NVIDIA GPU detected; Standard CPU mode is recommended."
            )

    if is_onedrive_path:
        warnings.append(
            onedrive_warning
            or "Target data directory is inside a OneDrive-synced folder, which may cause SQLite or Qdrant lock conflicts."
        )

    return HardwareRecommendation(
        ram_status=ram_status,
        has_nvidia_cuda_capable_gpu=has_nvidia_cuda_capable_gpu,
        gpu_status=gpu_status,
        disk_status=disk_status,
        cuda_disk_sufficient=cuda_disk_sufficient,
        recommended_profile=recommended_profile,
        recommended_upload_mode=recommended_upload_mode,
        recommendation_reason=recommendation_reason,
        blockers=blockers,
        warnings=warnings,
    )


def probe_system_hardware(data_dir: Path | None = None) -> HardwareReport:
    """Inspect host CPU, RAM, GPUs, disk capacity, and OneDrive sync risk."""
    target_dir = Path(data_dir) if data_dir is not None else DesktopPaths().data_dir
    cpu_name, logical_cores, physical_cores = _probe_cpu()
    total_ram_gb, available_ram_gb = _probe_memory()
    gpus = _probe_gpus()
    target_drive, free_disk_gb, is_onedrive_path, onedrive_warning = _probe_disk(
        target_dir,
        onedrive_checker=_check_onedrive_path,
    )

    rec = recommend_compute_profile(
        total_ram_gb=total_ram_gb,
        gpus=gpus,
        free_disk_gb=free_disk_gb,
        is_onedrive_path=is_onedrive_path,
        onedrive_warning=onedrive_warning,
    )

    return HardwareReport(
        cpu_name=cpu_name,
        logical_cores=logical_cores,
        physical_cores=physical_cores,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        ram_status=rec.ram_status,
        gpus=gpus,
        has_nvidia_cuda_capable_gpu=rec.has_nvidia_cuda_capable_gpu,
        gpu_status=rec.gpu_status,
        target_drive=target_drive,
        free_disk_gb=free_disk_gb,
        disk_status=rec.disk_status,
        cuda_disk_sufficient=rec.cuda_disk_sufficient,
        is_onedrive_path=is_onedrive_path,
        onedrive_warning=onedrive_warning,
        recommended_profile=rec.recommended_profile,
        recommended_upload_mode=rec.recommended_upload_mode,
        recommendation_reason=rec.recommendation_reason,
        blockers=rec.blockers,
        warnings=rec.warnings,
    )
