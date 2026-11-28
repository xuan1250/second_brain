"""Desktop paths, persistent user configuration, and loopback port allocation."""
from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.desktop.io_utils import atomic_write_json

logger = logging.getLogger(__name__)


def _venv_python_relpath() -> Path:
    """Return the relative path to the Python interpreter inside a virtualenv."""
    if sys.platform == "win32":
        return Path("Scripts") / "python.exe"
    return Path("bin") / "python"


def _venv_pythonw_relpath() -> Path:
    """Return the relative path to the windowless Python interpreter inside a virtualenv."""
    if sys.platform == "win32":
        return Path("Scripts") / "pythonw.exe"
    return Path("bin") / "python"


def _detect_app_dir() -> Path:
    """Auto-detect packaged app root (`<app>/app/backend`) or fall back to repo root."""
    env_app_dir = (os.environ.get("SECOND_BRAIN_APP_DIR") or "").strip()
    if env_app_dir:
        return Path(env_app_dir).expanduser().resolve()

    candidates: list[Path] = []
    module_path = Path(__file__).resolve()
    # In staged install: <app>/app/backend/src/desktop/config.py -> parents[4] == <app>
    if len(module_path.parents) > 4:
        candidates.append(module_path.parents[4])

    if getattr(sys, "frozen", False):
        exe_path = Path(sys.executable).resolve()
        candidates.extend([exe_path.parent, *exe_path.parents[:3]])
    else:
        exe_path = Path(sys.executable).resolve()
        # In staged install: <app>/venv/Scripts/pythonw.exe -> parents[2] == <app>
        candidates.extend(exe_path.parents[:4])

    for candidate in candidates:
        if (candidate / "app" / "backend").is_dir():
            return candidate

    # Development repository root: <repo>/backend/src/desktop/config.py -> parents[3]
    return module_path.parents[3]


def _default_user_data_dir() -> Path:
    """Resolve the mutable per-user data directory for Second Brain Desktop."""
    override = (os.environ.get("SECOND_BRAIN_DESKTOP_HOME") or "").strip()
    if override:
        return Path(override).expanduser().resolve()

    if sys.platform == "win32":
        local_app_data = (os.environ.get("LOCALAPPDATA") or "").strip()
        base = Path(local_app_data) if local_app_data else (Path.home() / "AppData" / "Local")
        return (base / "SecondBrain").resolve()

    return (Path.home() / ".local" / "share" / "SecondBrain").resolve()


class DesktopConfig(BaseModel):
    """Persistent desktop launcher configuration stored in `desktop_config.json`."""

    model_config = ConfigDict(extra="ignore")

    compute_profile: Literal["cpu", "cuda"] = "cpu"
    sparse_mode_override: Literal["auto", "bge_m3", "bm25"] | None = None
    data_dir_override: str | None = None
    models_ready: bool = False
    cuda_runtime_ready: bool = False
    last_installed_version: str = "0.1.0"
    preferred_port: int = 8000
    preferred_qdrant_port: int = 6333
    show_tray_hint: bool = True


class DesktopPaths:
    """Resolved read-only application assets and mutable per-user runtime/data paths."""

    def __init__(
        self,
        app_dir: Path | str | None = None,
        user_data_dir: Path | str | None = None,
        data_dir_override: Path | str | None = None,
    ) -> None:
        self.app_dir: Path = (
            Path(app_dir).expanduser().resolve() if app_dir is not None else _detect_app_dir()
        )
        self.user_data_dir: Path = (
            Path(user_data_dir).expanduser().resolve()
            if user_data_dir is not None
            else _default_user_data_dir()
        )
        self._data_dir_override: Path | None = (
            Path(data_dir_override).expanduser().resolve()
            if data_dir_override is not None and str(data_dir_override).strip()
            else None
        )

    @property
    def is_staged_install(self) -> bool:
        """Return True when `app_dir` matches the packaged `<app>/app/backend` layout."""
        return (self.app_dir / "app" / "backend").is_dir()

    @property
    def config_dir(self) -> Path:
        return self.user_data_dir / "config"

    @property
    def config_path(self) -> Path:
        return self.config_dir / "desktop_config.json"

    @property
    def control_file_path(self) -> Path:
        return self.config_dir / "desktop_control.json"

    @property
    def logs_dir(self) -> Path:
        return self.user_data_dir / "logs"

    @property
    def runtime_dir(self) -> Path:
        return self.user_data_dir / "runtime"

    @property
    def cuda_venv_dir(self) -> Path:
        return self.runtime_dir / "venv"

    @property
    def webview2_cache_dir(self) -> Path:
        return self.user_data_dir / "webview2_cache"

    def resolve_data_dir(self, config: DesktopConfig | None = None) -> Path:
        """Resolve active data root honoring explicit override, config override, or default."""
        if self._data_dir_override is not None:
            return self._data_dir_override

        if config is not None:
            if config.data_dir_override and config.data_dir_override.strip():
                return Path(config.data_dir_override).expanduser().resolve()
            return self.user_data_dir / "data"

        if self.config_path.is_file():
            loaded = load_desktop_config(self)
            if loaded.data_dir_override and loaded.data_dir_override.strip():
                return Path(loaded.data_dir_override).expanduser().resolve()

        return self.user_data_dir / "data"

    @property
    def data_dir(self) -> Path:
        return self.resolve_data_dir(None)

    @property
    def db_dir(self) -> Path:
        return self.data_dir / "db"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def qdrant_storage_dir(self) -> Path:
        return self.data_dir / "qdrant_storage"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def markdown_dir(self) -> Path:
        return self.data_dir / "markdown"

    @property
    def backend_dir(self) -> Path:
        if self.is_staged_install:
            return self.app_dir / "app" / "backend"
        return self.app_dir / "backend"

    @property
    def frontend_dist_dir(self) -> Path:
        if self.is_staged_install:
            return self.app_dir / "app" / "frontend" / "dist"
        return self.app_dir / "frontend" / "dist"

    def _bin_exe(self, exe_stem: str) -> Path:
        exe_name = f"{exe_stem}.exe" if sys.platform == "win32" else exe_stem
        staged_bin = self.app_dir / "bin" / exe_name
        if self.is_staged_install or staged_bin.exists():
            return staged_bin
        alt_exe = self.app_dir / "bin" / f"{exe_stem}.exe"
        if alt_exe.exists():
            return alt_exe
        found = shutil.which(exe_stem)
        return Path(found) if found else staged_bin

    def _venv_python(self, rel: Path) -> Path:
        staged_py = self.app_dir / "venv" / rel
        if self.is_staged_install or staged_py.exists() or (self.app_dir / "venv").is_dir():
            return staged_py
        dev_py = self.backend_dir / ".venv" / rel
        if dev_py.exists():
            return dev_py
        return staged_py

    @property
    def qdrant_exe(self) -> Path:
        return self._bin_exe("qdrant")

    @property
    def uv_exe(self) -> Path:
        return self._bin_exe("uv")

    @property
    def base_venv_python(self) -> Path:
        return self._venv_python(_venv_python_relpath())

    @property
    def base_venv_pythonw(self) -> Path:
        return self._venv_python(_venv_pythonw_relpath())

    @property
    def cuda_venv_python(self) -> Path:
        return self.cuda_venv_dir / _venv_python_relpath()

    def ensure_directories(self, config: DesktopConfig | None = None) -> None:
        """Create all required per-user runtime and data directories."""
        data_root = self.resolve_data_dir(config)
        for sub in ("raw", "parsed", "markdown", "models", "db", "qdrant_storage"):
            (data_root / sub).mkdir(parents=True, exist_ok=True)
        for user_sub in (
            self.runtime_dir,
            self.webview2_cache_dir,
            self.config_dir,
            self.logs_dir,
        ):
            user_sub.mkdir(parents=True, exist_ok=True)

    def resolve_backend_python(
        self,
        compute_profile: str,
        cuda_runtime_ready: bool = True,
    ) -> Path:
        """Select CUDA runtime interpreter when active and ready, else base CPU Python."""
        normalized = (compute_profile or "cpu").strip().lower()
        if normalized == "cuda" and cuda_runtime_ready and self.cuda_venv_python.is_file():
            return self.cuda_venv_python

        staged_py = self.app_dir / "venv" / _venv_python_relpath()
        if self.is_staged_install or staged_py.is_file():
            return staged_py

        dev_py = self.backend_dir / ".venv" / _venv_python_relpath()
        if dev_py.is_file():
            return dev_py

        return Path(sys.executable)

    def iter_uninstall_targets(
        self,
        include_models: bool = True,
        include_user_data: bool = True,
        config: DesktopConfig | None = None,
    ) -> list[Path]:
        """Return deduplicated list of runtime-created directories for clean uninstall."""
        effective_config = config
        if effective_config is None and self.config_path.is_file():
            effective_config = load_desktop_config(self)

        resolved_data = self.resolve_data_dir(effective_config)
        default_data = self.user_data_dir / "data"

        raw_targets: list[Path] = [
            self.runtime_dir,
            self.webview2_cache_dir,
            self.logs_dir,
        ]

        if include_models:
            raw_targets.append(resolved_data / "models")
            if default_data != resolved_data and (default_data / "models").exists():
                raw_targets.append(default_data / "models")

        if include_user_data:
            raw_targets.append(resolved_data)
            if (
                effective_config is not None
                and effective_config.data_dir_override
                and effective_config.data_dir_override.strip()
            ):
                raw_targets.append(
                    Path(effective_config.data_dir_override).expanduser().resolve()
                )
            if default_data != resolved_data:
                raw_targets.append(default_data)
            raw_targets.append(self.config_dir)
            raw_targets.append(self.user_data_dir)

        deduped: list[Path] = []
        seen: set[str] = set()
        for target in raw_targets:
            norm_key = os.path.normcase(os.path.abspath(str(target)))
            if norm_key not in seen:
                seen.add(norm_key)
                deduped.append(target)
        return deduped


def load_desktop_config(paths: DesktopPaths | None = None) -> DesktopConfig:
    """Load `desktop_config.json`, returning defaults when missing or corrupted."""
    resolved_paths = paths or DesktopPaths()
    cfg_path = resolved_paths.config_path
    if not cfg_path.is_file():
        return DesktopConfig()

    try:
        raw_text = cfg_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
        if not isinstance(data, dict):
            raise ValueError("desktop_config.json root must be a JSON object")
        return DesktopConfig.model_validate(data)
    except Exception as exc:
        logger.warning("Failed to parse desktop config at %s (%s); using defaults.", cfg_path, exc)
        return DesktopConfig()


def save_desktop_config(config: DesktopConfig, paths: DesktopPaths | None = None) -> Path:
    """Atomically persist `DesktopConfig` to `desktop_config.json` via temp file + `os.replace`."""
    resolved_paths = paths or DesktopPaths()
    resolved_paths.config_dir.mkdir(parents=True, exist_ok=True)
    return atomic_write_json(resolved_paths.config_path, config, fsync=True)


def find_free_loopback_port(preferred_port: int) -> int:
    """Return `preferred_port` if exclusively bindable on 127.0.0.1, else an ephemeral free port."""

    def _apply_exclusive_bind_opts(sock: socket.socket) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)

    if 1 <= preferred_port <= 65535:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                _apply_exclusive_bind_opts(probe)
                probe.bind(("127.0.0.1", preferred_port))
                return preferred_port
        except OSError:
            pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as fallback:
        _apply_exclusive_bind_opts(fallback)
        fallback.bind(("127.0.0.1", 0))
        return int(fallback.getsockname()[1])
