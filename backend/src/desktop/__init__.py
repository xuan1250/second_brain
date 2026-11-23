"""Windows desktop runtime foundation: hardware probes, config paths, Win32 safety, and sidecar supervisor."""
from __future__ import annotations

from src.desktop.app import DesktopApp, DesktopBridgeApi, main
from src.desktop.bootstrap import (
    MODEL_SPECS,
    BootstrapProgress,
    ComputeRuntimeResult,
    ModelCacheStatus,
    ModelSpec,
    RuntimeBootstrapService,
    RuntimeCheckResult,
    detect_collection_sparse_mode,
)
from src.desktop.config import (
    DesktopConfig,
    DesktopPaths,
    find_free_loopback_port,
    load_desktop_config,
    save_desktop_config,
)
from src.desktop.hardware import (
    GpuAdapterInfo,
    HardwareRecommendation,
    HardwareReport,
    probe_system_hardware,
    recommend_compute_profile,
)
from src.desktop.job_object import (
    ControlChannelServer,
    DEFAULT_MUTEX_NAME,
    SingleInstanceMutex,
    WindowsJobObject,
    is_mutex_held,
    send_control_command,
)
from src.desktop.supervisor import SidecarSupervisor
from src.desktop.tray import SystemTrayController, build_tray_icon_image

__all__ = [
    "BootstrapProgress",
    "ComputeRuntimeResult",
    "ControlChannelServer",
    "DEFAULT_MUTEX_NAME",
    "DesktopApp",
    "DesktopBridgeApi",
    "DesktopConfig",
    "DesktopPaths",
    "GpuAdapterInfo",
    "HardwareRecommendation",
    "HardwareReport",
    "MODEL_SPECS",
    "ModelCacheStatus",
    "ModelSpec",
    "RuntimeBootstrapService",
    "RuntimeCheckResult",
    "SidecarSupervisor",
    "SingleInstanceMutex",
    "SystemTrayController",
    "WindowsJobObject",
    "build_tray_icon_image",
    "detect_collection_sparse_mode",
    "find_free_loopback_port",
    "is_mutex_held",
    "load_desktop_config",
    "main",
    "probe_system_hardware",
    "recommend_compute_profile",
    "save_desktop_config",
    "send_control_command",
]
