"""Managed sidecar process supervisor for `qdrant.exe` and FastAPI `uvicorn`."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
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
from src.desktop.job_object import WindowsJobObject

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_LOG_MAX_BYTES = 5 * 1024 * 1024
_LOG_BACKUP_COUNT = 3


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
        """Initialize rotating log files (`qdrant.log`, `backend.log`, `launcher.log`)."""
        if self._log_handlers:
            return

        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        for channel, filename in (
            ("launcher", "launcher.log"),
            ("qdrant", "qdrant.log"),
            ("backend", "backend.log"),
        ):
            log_path = self.paths.logs_dir / filename
            handler = RotatingFileHandler(
                log_path,
                maxBytes=_LOG_MAX_BYTES,
                backupCount=_LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            handler.setFormatter(formatter)

            inst_logger = logging.getLogger(f"second_brain.desktop.{id(self)}.{channel}")
            inst_logger.setLevel(logging.INFO)
            inst_logger.propagate = False
            inst_logger.handlers.clear()
            inst_logger.addHandler(handler)

            self._log_handlers[channel] = handler
            self._loggers[channel] = inst_logger

    def _close_loggers(self) -> None:
        """Flush and close rotating file handlers so log files are unlocked."""
        for channel, inst_logger in list(self._loggers.items()):
            handler = self._log_handlers.get(channel)
            if handler is not None:
                try:
                    inst_logger.removeHandler(handler)
                    handler.flush()
                    handler.close()
                except Exception:
                    pass
        self._log_handlers.clear()
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
        pipe = getattr(proc, "stdout", None)
        if pipe is None:
            return None

        target_logger = self._loggers.get(channel)

        def _reader() -> None:
            try:
                for raw_line in pipe:
                    line = str(raw_line).rstrip("\r\n")
                    if line and target_logger is not None:
                        target_logger.info("%s", line)
            except Exception:
                pass
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        thread = threading.Thread(
            target=_reader,
            name=f"desktop-log-{channel}",
            daemon=True,
        )
        thread.start()
        return thread

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
                "HF_HOME": str(models_dir / "huggingface"),
                "HF_HUB_CACHE": str(models_dir / "huggingface" / "hub"),
                "TORCH_HOME": str(models_dir / "torch"),
                "FASTEMBED_CACHE_PATH": str(models_dir / "fastembed"),
                "RAPIDOCR_MODEL_DIR": str(models_dir / "rapidocr"),
                "DOCLING_CACHE_DIR": str(models_dir / "docling"),
                "FLASHRANK_ALLOW_DOWNLOAD": "true",
                "EMBEDDING_DEVICE": device,
                "INGESTION_DEVICE": device,
                "EMBEDDING_FP16": fp16,
                "EMBEDDING_SPARSE_MODE": sparse_mode,
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(self.paths.backend_dir),
            }
        )

        docling_artifacts = models_dir / "docling" / "models"
        if docling_artifacts.is_dir() and any(docling_artifacts.iterdir()):
            env["DOCLING_ARTIFACTS_PATH"] = str(docling_artifacts)
        else:
            env.pop("DOCLING_ARTIFACTS_PATH", None)

        return env

    def _wait_for_http_ready(
        self,
        url: str,
        proc: subprocess.Popen[Any],
        service_name: str,
    ) -> None:
        """Poll `url` until HTTP 200 or fail early if child exits or timeout expires."""
        deadline = time.monotonic() + self.startup_timeout_sec
        last_error: str = "no response yet"

        while time.monotonic() < deadline:
            exit_code = proc.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"{service_name} exited prematurely with exit code {exit_code}"
                )

            try:
                with urllib.request.urlopen(url, timeout=1.5) as resp:
                    status = getattr(resp, "status", None)
                    if status is None:
                        status = resp.getcode()
                    if status == 200:
                        return
                    last_error = f"HTTP {status}"
            except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
                last_error = str(exc)

            time.sleep(self.poll_interval_sec)

        raise TimeoutError(
            f"{service_name} did not become ready at {url} within {self.startup_timeout_sec:.1f}s "
            f"(last error: {last_error})"
        )

    def _terminate_process(
        self,
        proc: subprocess.Popen[Any] | None,
        service_name: str,
        timeout_sec: float,
        reader_thread: threading.Thread | None = None,
    ) -> None:
        """Gracefully terminate a child process with timeout and hard-kill fallback."""
        if proc is None:
            return

        if proc.poll() is None:
            self._log_launcher(logging.INFO, "Stopping %s (pid=%s)...", service_name, getattr(proc, "pid", "?"))
            try:
                proc.terminate()
                proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                self._log_launcher(
                    logging.WARNING,
                    "%s did not exit within %.1fs; forcing kill.",
                    service_name,
                    timeout_sec,
                )
                try:
                    proc.kill()
                    proc.wait(timeout=2.0)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            except OSError:
                pass

        if reader_thread is not None and reader_thread.is_alive():
            reader_thread.join(timeout=1.5)

        pipe = getattr(proc, "stdout", None)
        if pipe is not None:
            try:
                pipe.close()
            except Exception:
                pass

    def start_qdrant(self) -> subprocess.Popen[str]:
        """Launch `qdrant.exe` on `127.0.0.1:<qdrant_port>` and await `/readyz`."""
        with self._lock:
            if self.is_qdrant_running and self._qdrant_proc is not None:
                return self._qdrant_proc

            self.paths.ensure_directories(self.config)
            self._ensure_loggers()
            job = self._ensure_job_object()

            data_root = self.paths.resolve_data_dir(self.config)
            cmd = [str(self.paths.qdrant_exe)]
            self._log_launcher(
                logging.INFO,
                "Starting Qdrant sidecar on %s (%s)",
                self.qdrant_url,
                cmd[0],
            )
            proc = subprocess.Popen(
                cmd,
                cwd=str(data_root),
                env=self.build_qdrant_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=_CREATE_NO_WINDOW,
            )
            job.assign_process(proc)
            reader = self._spawn_pipe_reader(proc, "qdrant")
            self._qdrant_proc = proc
            self._qdrant_log_thread = reader

            try:
                self._wait_for_http_ready(f"{self.qdrant_url}/readyz", proc, "Qdrant")
            except Exception:
                self._terminate_process(proc, "Qdrant", timeout_sec=5.0, reader_thread=reader)
                self._qdrant_proc = None
                self._qdrant_log_thread = None
                raise

            self._log_launcher(logging.INFO, "Qdrant sidecar is ready at %s.", self.qdrant_url)
            return proc

    def start_backend(self) -> subprocess.Popen[str]:
        """Launch FastAPI `uvicorn` on `127.0.0.1:<api_port>` and await `/api/v1/system/health`."""
        with self._lock:
            if self.is_backend_running and self._backend_proc is not None:
                return self._backend_proc

            self.paths.ensure_directories(self.config)
            self._ensure_loggers()
            job = self._ensure_job_object()

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
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.paths.backend_dir),
                env=self.build_backend_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=_CREATE_NO_WINDOW,
            )
            job.assign_process(proc)
            reader = self._spawn_pipe_reader(proc, "backend")
            self._backend_proc = proc
            self._backend_log_thread = reader

            try:
                self._wait_for_http_ready(
                    f"{self.api_url}/api/v1/system/health",
                    proc,
                    "FastAPI backend",
                )
            except Exception:
                self._terminate_process(proc, "FastAPI backend", timeout_sec=5.0, reader_thread=reader)
                self._backend_proc = None
                self._backend_log_thread = None
                raise

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
            self.stop_backend(timeout_sec=timeout_sec)
            self.stop_qdrant(timeout_sec=timeout_sec)
            self._job_object.close()
            self._log_launcher(logging.INFO, "Sidecar shutdown complete.")
            self._close_loggers()

    def __enter__(self) -> SidecarSupervisor:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()
