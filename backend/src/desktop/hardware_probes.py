"""Low-level OS, WMI, nvidia-smi, memory, CPU, and disk probes for hardware diagnostics."""
from __future__ import annotations

import ctypes
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.desktop.env import run_hidden_subprocess

if TYPE_CHECKING:
    from src.desktop.hardware import GpuAdapterInfo

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_SEC = 3.0
_BYTES_PER_GIB = float(1024**3)


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
    from src.desktop.hardware import GpuAdapterInfo

    exe = _find_nvidia_smi()
    if not exe:
        return []

    try:
        proc = run_hidden_subprocess(
            [
                exe,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            timeout=_PROBE_TIMEOUT_SEC,
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
    from src.desktop.hardware import GpuAdapterInfo

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
        proc = run_hidden_subprocess(
            [ps_exe, "-NoProfile", "-NonInteractive", "-Command", command],
            timeout=_PROBE_TIMEOUT_SEC,
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


def _probe_disk(
    target_path: Path,
    *,
    onedrive_checker: Callable[[Path], tuple[bool, str | None]] | None = None,
) -> tuple[str, float, bool, str | None]:
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

    if onedrive_checker is None:
        from src.desktop.hardware import _check_onedrive_path as default_checker

        is_onedrive, onedrive_warning = default_checker(resolved)
    else:
        is_onedrive, onedrive_warning = onedrive_checker(resolved)

    return target_drive, free_disk_gb, is_onedrive, onedrive_warning
