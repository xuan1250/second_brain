"""Windows hardware capability probes and CPU vs. NVIDIA CUDA compute recommendation."""
from __future__ import annotations

import ctypes
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.desktop.config import DesktopPaths

logger = logging.getLogger(__name__)

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_PROBE_TIMEOUT_SEC = 3.0
_BYTES_PER_GIB = float(1024**3)


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


def _probe_physical_cores_windows() -> int | None:
    """Query physical CPU core count via Win32 GetLogicalProcessorInformation."""
    if sys.platform != "win32":
        return None

    class _LogicalProcUnion(ctypes.Union):
        _fields_ = [("Reserved", ctypes.c_uint64 * 2)]

    class _SystemLogicalProcessorInformation(ctypes.Structure):
        _fields_ = [
            ("ProcessorMask", ctypes.c_size_t),
            ("Relationship", ctypes.c_int),
            ("Union", _LogicalProcUnion),
        ]

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        fn = kernel32.GetLogicalProcessorInformation
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        fn.restype = ctypes.c_int

        buf_len = ctypes.c_uint32(0)
        fn(None, ctypes.byref(buf_len))
        if buf_len.value == 0:
            return None

        entry_size = ctypes.sizeof(_SystemLogicalProcessorInformation)
        count = buf_len.value // entry_size
        if count <= 0:
            return None

        buffer = (_SystemLogicalProcessorInformation * count)()
        if not fn(ctypes.byref(buffer), ctypes.byref(buf_len)):
            return None

        actual_count = buf_len.value // entry_size
        # Relationship == 0 corresponds to RelationProcessorCore
        cores = sum(1 for i in range(actual_count) if buffer[i].Relationship == 0)
        return cores if cores > 0 else None
    except Exception as exc:
        logger.debug("Physical core probe failed: %s", exc)
        return None


def _probe_cpu() -> tuple[str, int, int | None]:
    """Return `(cpu_name, logical_cores, physical_cores)`."""
    logical_cores = os.cpu_count() or 1
    cpu_name = ""
    physical_cores: int | None = None

    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                cpu_name = str(value).strip()
        except Exception as exc:
            logger.debug("Registry CPU name probe failed: %s", exc)

        physical_cores = _probe_physical_cores_windows()
    else:
        try:
            cpuinfo = Path("/proc/cpuinfo")
            if cpuinfo.is_file():
                core_ids: set[tuple[str, str]] = set()
                phys_id = "0"
                for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.startswith("model name") and not cpu_name:
                        cpu_name = line.partition(":")[2].strip()
                    elif line.startswith("physical id"):
                        phys_id = line.partition(":")[2].strip()
                    elif line.startswith("core id"):
                        core_ids.add((phys_id, line.partition(":")[2].strip()))
                if core_ids:
                    physical_cores = len(core_ids)
        except Exception:
            pass

    if not cpu_name:
        cpu_name = (platform.processor() or platform.machine() or "Unknown CPU").strip()

    return cpu_name, logical_cores, physical_cores


def _probe_memory() -> tuple[float, float]:
    """Return `(total_ram_gb, available_ram_gb)` rounded to 2 decimal places."""
    if sys.platform == "win32":

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            global_mem = kernel32.GlobalMemoryStatusEx
            global_mem.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
            global_mem.restype = ctypes.c_int

            mem_status = MEMORYSTATUSEX()
            mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if global_mem(ctypes.byref(mem_status)):
                total_gb = round(mem_status.ullTotalPhys / _BYTES_PER_GIB, 2)
                avail_gb = round(mem_status.ullAvailPhys / _BYTES_PER_GIB, 2)
                return total_gb, avail_gb
        except Exception as exc:
            logger.debug("GlobalMemoryStatusEx probe failed: %s", exc)

    # POSIX fallback for non-Windows hosts / CI
    try:
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            vals: dict[str, int] = {}
            for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts and parts[0].isdigit():
                    vals[key.strip()] = int(parts[0]) * 1024
            total_b = vals.get("MemTotal", 0)
            avail_b = vals.get("MemAvailable", vals.get("MemFree", 0))
            if total_b > 0:
                return round(total_b / _BYTES_PER_GIB, 2), round(avail_b / _BYTES_PER_GIB, 2)
    except Exception:
        pass

    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            total_gb = round((pages * page_size) / _BYTES_PER_GIB, 2)
            avail_gb = round((avail_pages * page_size) / _BYTES_PER_GIB, 2)
            return total_gb, avail_gb
    except Exception:
        pass

    return 0.0, 0.0


def _find_nvidia_smi() -> str | None:
    """Locate `nvidia-smi` on PATH or standard Windows NVIDIA driver directories."""
    on_path = shutil.which("nvidia-smi")
    if on_path:
        return on_path

    if sys.platform == "win32":
        for candidate in (
            Path(r"C:\Windows\System32\nvidia-smi.exe"),
            Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
        ):
            if candidate.is_file():
                return str(candidate)
    return None


def _probe_nvidia_smi() -> list[GpuAdapterInfo]:
    """Query NVIDIA GPUs via `nvidia-smi` with strict 3s timeout."""
    exe = _find_nvidia_smi()
    if not exe:
        return []

    try:
        proc = subprocess.run(
            [
                exe,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PROBE_TIMEOUT_SEC,
            creationflags=_CREATE_NO_WINDOW,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("nvidia-smi probe failed/timed out: %s", exc)
        return []

    if proc.returncode != 0 or not proc.stdout.strip():
        return []

    gpus: list[GpuAdapterInfo] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        name = parts[0] if parts else "NVIDIA GPU"
        vram_gb: float | None = None
        if len(parts) >= 2:
            try:
                vram_mb = float(parts[1])
                vram_gb = round(vram_mb / 1024.0, 2)
            except ValueError:
                vram_gb = None
        driver_version = parts[2] if len(parts) >= 3 and parts[2] and parts[2] != "[N/A]" else None
        gpus.append(
            GpuAdapterInfo(
                name=name,
                vram_gb=vram_gb,
                vendor="NVIDIA",
                driver_version=driver_version,
                is_nvidia=True,
                cuda_capable=bool(vram_gb is not None and vram_gb >= 4.0),
            )
        )
    return gpus


def _infer_vendor(name: str, compatibility: str | None) -> str:
    """Normalize GPU vendor label from adapter compatibility/name strings."""
    haystack = f"{compatibility or ''} {name}".lower()
    if "nvidia" in haystack:
        return "NVIDIA"
    if "amd" in haystack or "radeon" in haystack or "ati" in haystack:
        return "AMD"
    if "intel" in haystack:
        return "Intel"
    cleaned = (compatibility or "").strip()
    return cleaned or "Unknown"


def _probe_wmi_gpus() -> list[GpuAdapterInfo]:
    """Query display adapters via PowerShell CIM `Win32_VideoController` with strict 3s timeout."""
    if sys.platform != "win32":
        return []

    ps_exe = shutil.which("powershell") or shutil.which("pwsh")
    if not ps_exe:
        return []

    command = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterCompatibility,AdapterRAM,DriverVersion | "
        "ConvertTo-Json -Compress"
    )
    try:
        proc = subprocess.run(
            [ps_exe, "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PROBE_TIMEOUT_SEC,
            creationflags=_CREATE_NO_WINDOW,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("WMI GPU probe failed/timed out: %s", exc)
        return []

    raw_out = proc.stdout.strip()
    if proc.returncode != 0 or not raw_out:
        return []

    try:
        parsed = json.loads(raw_out)
    except ValueError as exc:
        logger.debug("Failed to parse WMI GPU JSON: %s", exc)
        return []

    items: list[dict[str, Any]] = parsed if isinstance(parsed, list) else [parsed]
    gpus: list[GpuAdapterInfo] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name") or "").strip()
        if not name:
            continue
        compat = str(item.get("AdapterCompatibility") or "").strip()
        vendor = _infer_vendor(name, compat)
        is_nvidia = vendor == "NVIDIA"

        vram_gb: float | None = None
        adapter_ram = item.get("AdapterRAM")
        if isinstance(adapter_ram, (int, float)) and adapter_ram > 0:
            vram_gb = round(float(adapter_ram) / _BYTES_PER_GIB, 2)

        driver_raw = item.get("DriverVersion")
        driver_version = str(driver_raw).strip() if driver_raw else None
        gpus.append(
            GpuAdapterInfo(
                name=name,
                vram_gb=vram_gb,
                vendor=vendor,
                driver_version=driver_version,
                is_nvidia=is_nvidia,
                cuda_capable=bool(is_nvidia and vram_gb is not None and vram_gb >= 4.0),
            )
        )
    return gpus


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


def _probe_disk(target_path: Path) -> tuple[str, float, bool, str | None]:
    """Return `(target_drive, free_disk_gb, is_onedrive_path, onedrive_warning)`."""
    resolved = target_path.expanduser().resolve()
    probe_dir = resolved
    while not probe_dir.exists():
        parent = probe_dir.parent
        if parent == probe_dir:
            break
        probe_dir = parent

    if not probe_dir.exists():
        probe_dir = Path.home()

    usage = shutil.disk_usage(probe_dir)
    free_disk_gb = round(usage.free / _BYTES_PER_GIB, 2)
    target_drive = probe_dir.drive or str(probe_dir.anchor) or "/"
    is_onedrive, onedrive_warning = _check_onedrive_path(resolved)
    return target_drive, free_disk_gb, is_onedrive, onedrive_warning


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
    target_drive, free_disk_gb, is_onedrive_path, onedrive_warning = _probe_disk(target_dir)

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
