"""Managed sidecar process supervisor for `qdrant.exe` and FastAPI `uvicorn`."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from src.desktop.config import (
    DesktopConfig,
    DesktopPaths,
    find_free_loopback_port,
    load_desktop_config,
)
from src.desktop.env import CREATE_NO_WINDOW, build_model_cache_env
from src.desktop.job_object import WindowsJobObject
from src.desktop.logging_utils import acquire_log_handler, release_log_handler
from src.desktop.supervisor_process import (
    spawn_pipe_reader,
    spawn_sidecar,
    terminate_process,
    wait_for_http_ready,
)

_REAL_POPEN_TYPE = subprocess.Popen
_LOG_CHANNELS: tuple[tuple[str, str], ...] = (
    ("launcher", "launcher.log"),
    ("qdrant", "qdrant.log"),
    ("backend", "backend.log"),
)


def _kill_process_tree_win32(pid: int) -> None:
    """Terminate a Windows process and all descendants via `taskkill /T /F`."""
    if sys.platform != "win32" or pid <= 0:
        return
    try:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            creationflags=CREATE_NO_WINDOW,
        )
    except Exception:
        pass


class SidecarSupervisor:
    """Manage local `qdrant.exe` and FastAPI `uvicorn` sidecar processes cleanly."""

    def __init__(
        self,
        paths: DesktopPaths | None = None,
        config: DesktopConfig | None = None,
        *,
        api_port: int | None = None,
        qdrant_port: int | None = None,
        startup_timeout_sec: float = 45.0,
        poll_interval_sec: float = 0.2,
    ) -> None:
        self.paths = paths or DesktopPaths()
        self.config = config or load_desktop_config(self.paths)
        self.startup_timeout_sec = startup_timeout_sec
        self.poll_interval_sec = poll_interval_sec

        self.qdrant_port: int = (
            qdrant_port
            if qdrant_port is not None
            else find_free_loopback_port(self.config.preferred_qdrant_port)
        )
        if api_port is not None:
            self.api_port = api_port
        else:
            candidate_api = find_free_loopback_port(self.config.preferred_port)
            if candidate_api == self.qdrant_port:
                candidate_api = find_free_loopback_port(0)
            self.api_port = candidate_api

        self._job_object: WindowsJobObject = WindowsJobObject()
        self._qdrant_proc: subprocess.Popen[str] | None = None
        self._backend_proc: subprocess.Popen[str] | None = None
        self._qdrant_log_thread: threading.Thread | None = None
        self._backend_log_thread: threading.Thread | None = None
        self._loggers: dict[str, logging.Logger] = {}
        self._log_handlers: dict[str, RotatingFileHandler] = {}
        self._log_paths: dict[str, Path] = {}
        self._lock = threading.RLock()

    @property
    def api_url(self) -> str:
        """Return loopback base URL for FastAPI and React SPA."""
        return f"http://127.0.0.1:{self.api_port}"

    @property
    def qdrant_url(self) -> str:
        """Return loopback HTTP URL for the local Qdrant sidecar."""
        return f"http://127.0.0.1:{self.qdrant_port}"

    @property
    def is_qdrant_running(self) -> bool:
        """Return True if the supervised Qdrant process is currently alive."""
        return self._qdrant_proc is not None and self._qdrant_proc.poll() is None

    @property
    def is_backend_running(self) -> bool:
        """Return True if the supervised Uvicorn backend process is currently alive."""
        return self._backend_proc is not None and self._backend_proc.poll() is None

    def _ensure_job_object(self) -> WindowsJobObject:
        if sys.platform == "win32" and not self._job_object.is_active:
            self._job_object = WindowsJobObject()
        return self._job_object

    def _ensure_loggers(self) -> None:
        """Initialize refcounted rotating log files (`qdrant.log`, `backend.log`, `launcher.log`)."""
        if self._log_handlers:
            return

        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        for channel, filename in _LOG_CHANNELS:
            log_path = self.paths.logs_dir / filename
            handler = acquire_log_handler(log_path)

            inst_logger = logging.getLogger(f"second_brain.desktop.{id(self)}.{channel}")
            inst_logger.setLevel(logging.INFO)
            inst_logger.propagate = False
            inst_logger.handlers.clear()
            inst_logger.addHandler(handler)

            self._log_handlers[channel] = handler
            self._log_paths[channel] = log_path
            self._loggers[channel] = inst_logger

    def _close_loggers(self) -> None:
        """Detach and release refcounted log handlers so closed files are unlocked."""
        for channel, inst_logger in list(self._loggers.items()):
            handler = self._log_handlers.get(channel)
            if handler is not None:
                try:
                    inst_logger.removeHandler(handler)
                except Exception:
                    pass
            log_path = self._log_paths.pop(channel, None)
            if log_path is not None:
                release_log_handler(log_path)
        self._log_handlers.clear()
        self._log_paths.clear()
        self._loggers.clear()

    def _log_launcher(self, level: int, message: str, *args: Any) -> None:
        inst_logger = self._loggers.get("launcher")
        if inst_logger is not None:
            inst_logger.log(level, message, *args)

    def _spawn_pipe_reader(
        self,
        proc: subprocess.Popen[str],
        channel: str,
    ) -> threading.Thread | None:
        """Stream child stdout/stderr lines into a rotating file logger in a daemon thread."""
        return spawn_pipe_reader(proc, channel, self._loggers.get(channel))

    def build_qdrant_env(self) -> dict[str, str]:
        """Construct environment variables for the loopback Qdrant sidecar."""
        data_root = self.paths.resolve_data_dir(self.config)
        env = os.environ.copy()
        env.update(
            {
                "QDRANT__SERVICE__HOST": "127.0.0.1",
                "QDRANT__SERVICE__HTTP_PORT": str(self.qdrant_port),
                # Qdrant v1.20+ rejects "null" as an invalid u16 env var; "0" assigns an ephemeral free port avoiding 6334 collisions.
                "QDRANT__SERVICE__GRPC_PORT": "0",
                "QDRANT__STORAGE__STORAGE_PATH": str(data_root / "qdrant_storage"),
                "QDRANT__TELEMETRY_DISABLED": "true",
            }
        )
        return env

    def build_backend_env(self) -> dict[str, str]:
        """Construct environment variables for the FastAPI `uvicorn` backend process."""
        data_root = self.paths.resolve_data_dir(self.config)
        models_dir = data_root / "models"
        db_dir = data_root / "db"
        is_cuda = self.config.compute_profile == "cuda" and self.config.cuda_runtime_ready
        device = "cuda" if is_cuda else "cpu"
        fp16 = "true" if is_cuda else "false"
        sparse_mode = (
            self.config.sparse_mode_override
            if self.config.sparse_mode_override
            else ("auto" if is_cuda else "bm25")
        )

        cache_env = build_model_cache_env(models_dir, for_download=False)
        env = os.environ.copy()
        env.update(
            {
                "SECOND_BRAIN_DESKTOP_HOME": str(self.paths.user_data_dir),
                "SECOND_BRAIN_APP_DIR": str(self.paths.app_dir),
                "DATA_DIR": str(data_root),
                "DB_DIR": str(db_dir),
                "FRONTEND_DIST_DIR": str(self.paths.frontend_dist_dir),
                "QDRANT_URL": self.qdrant_url,
                "NO_PROXY": "127.0.0.1,localhost",
                "CORS_ORIGINS": (
                    f"http://127.0.0.1:{self.api_port},http://localhost:{self.api_port}"
                ),
                **cache_env,
                "EMBEDDING_DEVICE": device,
                "INGESTION_DEVICE": device,
                "EMBEDDING_FP16": fp16,
                "EMBEDDING_SPARSE_MODE": sparse_mode,
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(self.paths.backend_dir),
            }
        )
        if "DOCLING_ARTIFACTS_PATH" not in cache_env:
            env.pop("DOCLING_ARTIFACTS_PATH", None)
        return env

    def _wait_for_http_ready(
        self,
        url: str,
        proc: subprocess.Popen[Any],
        service_name: str,
    ) -> None:
        """Poll `url` until HTTP 200 or fail early if child exits or timeout expires."""
        wait_for_http_ready(
            url,
            proc,
            service_name,
            startup_timeout_sec=self.startup_timeout_sec,
            poll_interval_sec=self.poll_interval_sec,
        )

    def _terminate_process(
        self,
        proc: subprocess.Popen[Any] | None,
        service_name: str,
        timeout_sec: float,
        reader_thread: threading.Thread | None = None,
    ) -> None:
        """Gracefully terminate a child process tree with timeout and hard-kill fallback."""
        terminate_process(
            proc,
            service_name,
            timeout_sec,
            reader_thread=reader_thread,
            log_fn=self._log_launcher,
        )

    def _spawn_sidecar(
        self,
        *,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        channel: str,
        service_name: str,
        health_url: str,
    ) -> tuple[subprocess.Popen[str], threading.Thread | None]:
        return spawn_sidecar(
            cmd=cmd,
            cwd=cwd,
            env=env,
            channel=channel,
            service_name=service_name,
            health_url=health_url,
            job_object=self._ensure_job_object(),
            target_logger=self._loggers.get(channel),
            startup_timeout_sec=self.startup_timeout_sec,
            poll_interval_sec=self.poll_interval_sec,
            terminate_fn=self._terminate_process,
        )

    def start_qdrant(self) -> subprocess.Popen[str]:
        """Launch `qdrant.exe` on `127.0.0.1:<qdrant_port>` and await `/readyz`."""
        with self._lock:
            if self.is_qdrant_running and self._qdrant_proc is not None:
                return self._qdrant_proc

            self.paths.ensure_directories(self.config)
            self._ensure_loggers()

            data_root = self.paths.resolve_data_dir(self.config)
            cmd = [str(self.paths.qdrant_exe)]
            self._log_launcher(logging.INFO, "Starting Qdrant sidecar on %s (%s)", self.qdrant_url, cmd[0])
            proc, reader = self._spawn_sidecar(
                cmd=cmd,
                cwd=data_root,
                env=self.build_qdrant_env(),
                channel="qdrant",
                service_name="Qdrant",
                health_url=f"{self.qdrant_url}/readyz",
            )
            self._qdrant_proc = proc
            self._qdrant_log_thread = reader
            self._log_launcher(logging.INFO, "Qdrant sidecar is ready at %s.", self.qdrant_url)
            return proc

    def start_backend(self) -> subprocess.Popen[str]:
        """Launch FastAPI `uvicorn` on `127.0.0.1:<api_port>` and await `/api/v1/system/health`."""
        with self._lock:
            if self.is_backend_running and self._backend_proc is not None:
                return self._backend_proc

            self.paths.ensure_directories(self.config)
            self._ensure_loggers()

            python_exe = self.paths.resolve_backend_python(
                self.config.compute_profile,
                cuda_runtime_ready=self.config.cuda_runtime_ready,
            )
            cmd = [
                str(python_exe),
                "-m",
                "uvicorn",
                "src.api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.api_port),
            ]
            self._log_launcher(
                logging.INFO,
                "Starting FastAPI backend on %s via %s (profile=%s)",
                self.api_url,
                python_exe,
                self.config.compute_profile,
            )
            proc, reader = self._spawn_sidecar(
                cmd=cmd,
                cwd=self.paths.backend_dir,
                env=self.build_backend_env(),
                channel="backend",
                service_name="FastAPI backend",
                health_url=f"{self.api_url}/api/v1/system/health",
            )
            self._backend_proc = proc
            self._backend_log_thread = reader
            self._log_launcher(logging.INFO, "FastAPI backend is ready at %s.", self.api_url)
            return proc

    def stop_backend(self, timeout_sec: float = 10.0) -> None:
        """Stop the FastAPI `uvicorn` sidecar so Python DLL/PYD locks are released."""
        with self._lock:
            proc = self._backend_proc
            reader = self._backend_log_thread
            self._backend_proc = None
            self._backend_log_thread = None
            self._terminate_process(proc, "FastAPI backend", timeout_sec=timeout_sec, reader_thread=reader)

    def stop_qdrant(self, timeout_sec: float = 10.0) -> None:
        """Stop the Qdrant sidecar process cleanly."""
        with self._lock:
            proc = self._qdrant_proc
            reader = self._qdrant_log_thread
            self._qdrant_proc = None
            self._qdrant_log_thread = None
            self._terminate_process(proc, "Qdrant", timeout_sec=timeout_sec, reader_thread=reader)

    def restart_backend(
        self,
        config: DesktopConfig | None = None,
        timeout_sec: float = 10.0,
    ) -> subprocess.Popen[str]:
        """Stop `uvicorn`, optionally update `DesktopConfig`, and start `uvicorn` again."""
        with self._lock:
            self.stop_backend(timeout_sec=timeout_sec)
            if config is not None:
                self.config = config
            return self.start_backend()

    def start(self) -> None:
        """Start both Qdrant and FastAPI sidecars in dependency order."""
        with self._lock:
            self.start_qdrant()
            self.start_backend()

    def shutdown(self, timeout_sec: float = 10.0) -> None:
        """Perform ordered shutdown: stop FastAPI first, then Qdrant, then Job Object and logs."""
        with self._lock:
            self._log_launcher(logging.INFO, "Initiating ordered sidecar shutdown...")
            try:
                try:
                    self.stop_backend(timeout_sec=timeout_sec)
                finally:
                    self.stop_qdrant(timeout_sec=timeout_sec)
            finally:
                try:
                    self._job_object.close()
                finally:
                    self._log_launcher(logging.INFO, "Sidecar shutdown complete.")
                    self._close_loggers()

    def __enter__(self) -> SidecarSupervisor:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()
