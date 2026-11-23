"""Native Edge WebView2 application coordinator, JS bridge API, and single-instance CLI entry."""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from collections.abc import Callable, Sequence
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal

from src.desktop.bootstrap import (
    MODEL_SPECS,
    BootstrapProgress,
    RuntimeBootstrapService,
    detect_collection_sparse_mode,
)
from src.desktop.config import (
    DesktopConfig,
    DesktopPaths,
    load_desktop_config,
    save_desktop_config,
)
from src.desktop.hardware import probe_system_hardware
from src.desktop.job_object import (
    DEFAULT_MUTEX_NAME,
    ControlChannelServer,
    SingleInstanceMutex,
    is_mutex_held,
    send_control_command,
)
from src.desktop.supervisor import SidecarSupervisor
from src.desktop.tray import SystemTrayController

logger = logging.getLogger(__name__)

_LOG_MAX_BYTES = 5 * 1024 * 1024
_LOG_BACKUP_COUNT = 3
_SETUP_HTML_PATH = Path(__file__).with_name("setup_ui.html").resolve()
_SW_RESTORE = 9


def _find_msedge_exe() -> Path | None:
    """Locate Microsoft Edge (`msedge.exe`) on PATH or standard Windows install paths."""
    for candidate_name in ("msedge.exe", "msedge"):
        found = shutil.which(candidate_name)
        if found:
            return Path(found)

    if sys.platform == "win32":
        roots = [
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("LOCALAPPDATA", ""),
        ]
        for root in roots:
            if not root:
                continue
            candidate = Path(root) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            if candidate.is_file():
                return candidate
    return None


def _focus_win32_window(title: str = "Second Brain") -> bool:
    """Best-effort Win32 FindWindowW + ShowWindow + SetForegroundWindow fallback."""
    if sys.platform != "win32":
        return False
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        user32.FindWindowW.restype = ctypes.c_void_p
        user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        user32.ShowWindow.restype = ctypes.c_int
        user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
        user32.SetForegroundWindow.restype = ctypes.c_int

        hwnd = user32.FindWindowW(None, title)
        if not hwnd:
            return False
        user32.ShowWindow(ctypes.c_void_p(hwnd), _SW_RESTORE)
        user32.SetForegroundWindow(ctypes.c_void_p(hwnd))
        return True
    except Exception:
        return False


def _get_qdrant_collection_name() -> str:
    """Resolve configured Qdrant collection name without failing in minimal test contexts."""
    try:
        from src.config import get_settings

        return get_settings().qdrant_collection
    except Exception:
        return "second_brain_knowledge"


def _build_external_link_interceptor_js(api_port: int) -> str:
    """Build idempotent JS click/open guard routing non-local http(s) links to system browser."""
    allowed_origins = json.dumps(
        [f"http://127.0.0.1:{api_port}", f"http://localhost:{api_port}"]
    )
    return f"""
    (function() {{
        if (window.__secondBrainExternalLinkHooked) return;
        window.__secondBrainExternalLinkHooked = true;
        var allowed = {allowed_origins};
        function isExternalHttp(rawUrl) {{
            if (!rawUrl) return false;
            try {{
                var u = new URL(String(rawUrl), window.location.href);
                if (u.protocol !== "http:" && u.protocol !== "https:") return false;
                return allowed.indexOf(u.origin) === -1;
            }} catch (_e) {{
                return false;
            }}
        }}
        document.addEventListener("click", function(ev) {{
            var target = ev.target;
            while (target && target.tagName !== "A") {{
                target = target.parentElement;
            }}
            if (!target || !target.getAttribute) return;
            var href = target.getAttribute("href");
            if (!href) return;
            var resolved = target.href || href;
            if (isExternalHttp(resolved) && window.pywebview && window.pywebview.api) {{
                ev.preventDefault();
                ev.stopPropagation();
                window.pywebview.api.open_external_url(resolved);
            }}
        }}, true);
        var origOpen = window.open;
        window.open = function(url, target, features) {{
            if (isExternalHttp(url) && window.pywebview && window.pywebview.api) {{
                window.pywebview.api.open_external_url(String(url));
                return null;
            }}
            return origOpen ? origOpen.apply(window, arguments) : null;
        }};
    }})();
    """


class DesktopBridgeApi:
    """JS-exposed `window.pywebview.api` bridge (keeps all internals on private `_app`)."""

    def __init__(self, app: DesktopApp) -> None:
        self._app = app

    def get_initial_state(self) -> dict[str, Any]:
        """Return hardware diagnostics, compute profile state, and 6-group model readiness."""
        return self._app.get_initial_state()

    def get_progress(self) -> dict[str, Any]:
        """Return live setup/switch progress snapshot for UI polling."""
        return self._app.get_progress()

    def start_setup(
        self,
        compute_profile: str = "cpu",
        reindex_sparse: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        """Start background runtime verification, compute setup, model downloads, and sidecars."""
        return self._app.start_setup(
            compute_profile=compute_profile,
            reindex_sparse=reindex_sparse,
            force=force,
        )

    def switch_compute_profile(
        self,
        compute_profile: str,
        reindex_sparse: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        """Stop backend, switch CPU/CUDA runtime cleanly, restart backend, and reopen workspace."""
        return self._app.switch_compute_profile(
            compute_profile=compute_profile,
            reindex_sparse=reindex_sparse,
            force=force,
        )

    def pick_models_folder(self) -> dict[str, Any]:
        """Prompt user for a pre-downloaded local models directory and import cached weights."""
        return self._app.pick_models_folder()

    def open_workspace(self) -> dict[str, Any]:
        """Ensure sidecars are ready and navigate WebView2 to the workspace SPA."""
        return self._app.open_workspace()

    def open_external_url(self, url: str) -> bool:
        """Open a validated `http://` or `https://` URL in the user's default browser."""
        return self._app.open_external_url(url)


class DesktopApp:
    """Coordinate single-instance mutex, gated bootstrap, WebView2 window, tray, and sidecars."""

    def __init__(
        self,
        paths: DesktopPaths | None = None,
        config: DesktopConfig | None = None,
        *,
        supervisor: SidecarSupervisor | None = None,
        bootstrap: RuntimeBootstrapService | None = None,
        webview_module: Any | None = None,
        tray_controller: SystemTrayController | None = None,
        edge_launcher: Callable[[str, Path], Any] | None = None,
        mutex: SingleInstanceMutex | None = None,
        mutex_name: str = DEFAULT_MUTEX_NAME,
    ) -> None:
        self.paths: DesktopPaths = paths or DesktopPaths()
        self.config: DesktopConfig = config or load_desktop_config(self.paths)
        self.supervisor: SidecarSupervisor = supervisor or SidecarSupervisor(self.paths, self.config)
        self.bootstrap: RuntimeBootstrapService = bootstrap or RuntimeBootstrapService(
            paths=self.paths,
            config=self.config,
        )
        self._webview_module = webview_module
        self._edge_launcher = edge_launcher
        self._mutex: SingleInstanceMutex = mutex or SingleInstanceMutex(name=mutex_name)

        self.is_quitting: bool = False
        self._quit_event = threading.Event()
        self._shutdown_complete = threading.Event()
        self._quit_lock = threading.Lock()
        self._quit_done: bool = False
        self._quit_owner_thread: threading.Thread | None = None

        self._window: Any | None = None
        self._bridge_mode: Literal["setup", "switch"] = "setup"
        self._using_edge_fallback: bool = False
        self._fallback_url: str | None = None

        self._app_logger: logging.Logger | None = None
        self._app_log_handler: RotatingFileHandler | None = None

        self._worker_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None

        self._progress_lock = threading.Lock()
        self._progress_state: dict[str, Any] = self._default_progress_state()

        self.bridge_api = DesktopBridgeApi(self)

        if tray_controller is not None:
            self.tray = tray_controller
            if isinstance(tray_controller, SystemTrayController):
                if tray_controller.on_open is None:
                    tray_controller.on_open = self.focus_window
                if tray_controller.on_open_hardware_mode is None:
                    tray_controller.on_open_hardware_mode = self.open_hardware_mode
                if tray_controller.on_reclaim_memory is None:
                    tray_controller.on_reclaim_memory = self.reclaim_memory_async
                if tray_controller.on_quit is None:
                    tray_controller.on_quit = self.request_quit
        else:
            self.tray = SystemTrayController(
                paths=self.paths,
                get_config=lambda: self.config,
                on_open=self.focus_window,
                on_open_hardware_mode=self.open_hardware_mode,
                on_reclaim_memory=self.reclaim_memory_async,
                on_quit=self.request_quit,
            )

        self._control_server = ControlChannelServer(
            self.paths,
            on_focus=self.focus_window,
            on_quit=self.request_quit,
        )

    @staticmethod
    def _default_progress_state() -> dict[str, Any]:
        return {
            "step": "idle",
            "percent": 0.0,
            "detail": "Ready to verify runtime and offline models.",
            "error": None,
            "remediation": None,
            "error_category": None,
            "is_running": False,
            "completed": False,
            "models": {spec.key: "pending" for spec in MODEL_SPECS},
            "steps": {
                "runtime_check": "pending",
                "compute_runtime": "pending",
            },
            "logs": [],
            "target_url": None,
            "reindex_recommended": False,
        }

    def _ensure_logger(self) -> None:
        """Initialize rotating `launcher.log` handler for desktop UI lifecycle diagnostics."""
        if self._app_log_handler is not None:
            return
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.paths.logs_dir / "launcher.log"
        handler = RotatingFileHandler(
            log_path,
            maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        inst_logger = logging.getLogger(f"second_brain.desktop.app.{id(self)}")
        inst_logger.setLevel(logging.INFO)
        inst_logger.propagate = False
        inst_logger.handlers.clear()
        inst_logger.addHandler(handler)
        self._app_logger = inst_logger
        self._app_log_handler = handler

    def _close_logger(self) -> None:
        if self._app_logger is not None and self._app_log_handler is not None:
            try:
                self._app_logger.removeHandler(self._app_log_handler)
                self._app_log_handler.flush()
                self._app_log_handler.close()
            except Exception:
                pass
        self._app_logger = None
        self._app_log_handler = None

    def _log(self, level: int, message: str, *args: Any) -> None:
        self._ensure_logger()
        if self._app_logger is not None:
            self._app_logger.log(level, message, *args)

    def _log_exception(self, message: str) -> None:
        self._ensure_logger()
        if self._app_logger is not None:
            self._app_logger.exception(message)

    def _update_progress(
        self,
        *,
        step: str | None = None,
        percent: float | None = None,
        detail: str | None = None,
        error: str | None = ...,  # type: ignore[assignment]
        remediation: str | None = ...,  # type: ignore[assignment]
        error_category: str | None = ...,  # type: ignore[assignment]
        is_running: bool | None = None,
        completed: bool | None = None,
        models: dict[str, str] | None = None,
        steps_update: dict[str, str] | None = None,
        target_url: str | None = ...,  # type: ignore[assignment]
        reindex_recommended: bool | None = None,
    ) -> None:
        with self._progress_lock:
            st = self._progress_state
            if step is not None:
                st["step"] = step
            if percent is not None:
                st["percent"] = round(max(0.0, min(100.0, float(percent))), 1)
            if detail is not None:
                st["detail"] = detail
                logs: list[str] = st["logs"]
                if detail and (not logs or logs[-1] != detail):
                    logs.append(detail)
                    if len(logs) > 100:
                        del logs[:-100]
            if error is not ...:
                st["error"] = error
            if remediation is not ...:
                st["remediation"] = remediation
            if error_category is not ...:
                st["error_category"] = error_category
            if is_running is not None:
                st["is_running"] = is_running
            if completed is not None:
                st["completed"] = completed
            if models is not None:
                merged_models = dict(st["models"])
                merged_models.update(models)
                st["models"] = merged_models
            if steps_update:
                merged_steps = dict(st["steps"])
                merged_steps.update(steps_update)
                st["steps"] = merged_steps
            if target_url is not ...:
                st["target_url"] = target_url
            if reindex_recommended is not None:
                st["reindex_recommended"] = reindex_recommended

    def get_progress(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of current bootstrap/switch progress."""
        with self._progress_lock:
            snap = dict(self._progress_state)
            snap["models"] = dict(self._progress_state["models"])
            snap["steps"] = dict(self._progress_state["steps"])
            snap["logs"] = list(self._progress_state["logs"])
            return snap

    def setup_ui_url(self, mode: Literal["setup", "switch"] | None = None) -> str:
        """Return `file://` URI for `setup_ui.html`, optionally tagged `?mode=switch`."""
        base_uri = _SETUP_HTML_PATH.as_uri()
        effective_mode = mode or self._bridge_mode
        if effective_mode == "switch":
            return f"{base_uri}?mode=switch"
        return base_uri

    def needs_bootstrap(self) -> bool:
        """Determine whether first-run runtime/model setup must gate sidecar startup."""
        if not self.config.models_ready:
            return True
        if self.config.compute_profile == "cuda" and (
            not self.config.cuda_runtime_ready or not self.paths.cuda_venv_python.is_file()
        ):
            return True
        status = self.bootstrap.inspect_models_status()
        return not bool(status.get("models_ready"))

    def _inspect_collection_sparse_mode_sync(self) -> Literal["bm25", "bge_m3"] | None:
        """Query Qdrant's active collection `sparse_mode` safely if Qdrant is running."""
        if not self.supervisor.is_qdrant_running:
            return None

        collection = _get_qdrant_collection_name()
        qdrant_url = self.supervisor.qdrant_url

        async def _probe() -> Literal["bm25", "bge_m3"] | None:
            from qdrant_client import AsyncQdrantClient

            client = AsyncQdrantClient(url=qdrant_url, timeout=2.0)
            try:
                return await detect_collection_sparse_mode(client, collection)
            finally:
                await client.close()

        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(_probe())

            result_holder: list[Literal["bm25", "bge_m3"] | None] = [None]

            def _runner() -> None:
                result_holder[0] = asyncio.run(_probe())

            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            t.join(timeout=3.0)
            return result_holder[0]
        except Exception as exc:
            self._log(logging.DEBUG, "Collection sparse_mode probe skipped: %s", exc)
            return None

    def get_initial_state(self) -> dict[str, Any]:
        """Build complete initial state payload consumed by `setup_ui.html`."""
        data_dir = self.paths.resolve_data_dir(self.config)
        hw_report = probe_system_hardware(data_dir=data_dir)
        models_status = self.bootstrap.inspect_models_status()
        collection_sparse_mode = self._inspect_collection_sparse_mode_sync()

        spec_by_key = {spec.key: spec for spec in MODEL_SPECS}
        serialized_models: list[dict[str, Any]] = []
        progress_model_map: dict[str, str] = {}
        for item in models_status.get("models", []):
            item_dict = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            spec = spec_by_key.get(item_dict.get("key", ""))
            if spec is not None:
                item_dict.setdefault("repo_or_id", spec.repo_or_id)
                item_dict.setdefault("approx_size_mb", spec.approx_size_mb)
                item_dict.setdefault("description", spec.description)
            serialized_models.append(item_dict)
            key = str(item_dict.get("key", ""))
            if key:
                progress_model_map[key] = "cached" if item_dict.get("cached") else "pending"

        with self._progress_lock:
            if not self._progress_state["is_running"]:
                self._progress_state["models"].update(progress_model_map)

        return {
            "mode": self._bridge_mode,
            "active_profile": self.config.compute_profile,
            "cuda_runtime_ready": bool(
                self.config.cuda_runtime_ready and self.paths.cuda_venv_python.is_file()
            ),
            "models_ready": bool(models_status.get("models_ready")),
            "sparse_mode_override": self.config.sparse_mode_override,
            "collection_sparse_mode": collection_sparse_mode,
            "backend_running": bool(self.supervisor.is_backend_running),
            "qdrant_running": bool(self.supervisor.is_qdrant_running),
            "api_url": self.supervisor.api_url,
            "data_dir": str(data_dir),
            "cuda_venv_dir": str(self.paths.cuda_venv_dir),
            "models_dir": str(models_status.get("models_dir", self.paths.models_dir)),
            "total_size_mb": float(models_status.get("total_size_mb", 0.0)),
            "hardware": hw_report.model_dump(),
            "models": serialized_models,
            "progress": self.get_progress(),
        }

    def _validate_target_profile(
        self,
        compute_profile: str,
        *,
        force: bool,
    ) -> tuple[Literal["cpu", "cuda"] | None, str | None]:
        normalized = (compute_profile or "cpu").strip().lower()
        if normalized not in ("cpu", "cuda"):
            return None, f"Unsupported compute profile '{compute_profile}'. Expected 'cpu' or 'cuda'."

        if normalized == "cuda" and not force:
            hw = probe_system_hardware(data_dir=self.paths.resolve_data_dir(self.config))
            if not hw.has_nvidia_cuda_capable_gpu:
                return (
                    None,
                    "CUDA mode requires an NVIDIA GPU with at least 4.0 GB VRAM "
                    "(or enable the manual compatibility override).",
                )
            if not hw.cuda_disk_sufficient:
                return (
                    None,
                    f"CUDA 12.4 wheels require at least 12.0 GB free disk space "
                    f"(found {hw.free_disk_gb:.1f} GB; enable override to proceed anyway).",
                )

        return normalized, None  # type: ignore[return-value]

    def _spawn_worker(self, target: Callable[[], None], name: str) -> dict[str, Any]:
        with self._worker_lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return {"ok": False, "error": "Another setup or compute switch task is already running."}
            thread = threading.Thread(target=target, name=name, daemon=True)
            self._worker_thread = thread
            thread.start()
        return {"ok": True, "started": True}

    def _launch_guarded_worker(
        self,
        *,
        compute_profile: str,
        force: bool,
        init_progress: dict[str, Any],
        worker_factory: Callable[[Literal["cpu", "cuda"]], Callable[[], None]],
        thread_name: str,
    ) -> dict[str, Any]:
        """Check active worker lock BEFORE validating or mutating progress state."""
        with self._worker_lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return {"ok": False, "error": "Another setup or compute switch task is already running."}

            target_profile, validation_err = self._validate_target_profile(compute_profile, force=force)
            if target_profile is None:
                self._update_progress(
                    step="error",
                    detail=validation_err,
                    error=validation_err,
                    is_running=False,
                    completed=False,
                )
                return {"ok": False, "error": validation_err}

            self._update_progress(**init_progress)
            thread = threading.Thread(
                target=worker_factory(target_profile),
                name=thread_name,
                daemon=True,
            )
            self._worker_thread = thread
            thread.start()
        return {"ok": True, "started": True}

    def _rollback_compute_config(
        self,
        prev_profile: Literal["cpu", "cuda"],
        prev_cuda_ready: bool,
        prev_sparse_override: Literal["auto", "bge_m3", "bm25"] | None,
    ) -> None:
        """Restore prior compute profile state on failed wheel installation or runtime check."""
        self.config.compute_profile = prev_profile
        self.config.cuda_runtime_ready = prev_cuda_ready
        self.config.sparse_mode_override = prev_sparse_override
        self.bootstrap.config = self.config
        self.supervisor.config = self.config
        try:
            save_desktop_config(self.config, self.paths)
        except Exception as exc:
            self._log(logging.WARNING, "Failed to persist compute config rollback: %s", exc)

    def _trigger_corpus_reindex(self, api_url: str) -> tuple[int, str | None]:
        """Request background re-indexing on the restarted backend after sparse-mode change."""
        url = f"{api_url.rstrip('/')}/api/v1/system/reindex"
        req = urllib.request.Request(url, data=b"", method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            payload = json.loads(raw)
            return int(payload.get("queued_count", 0)), None
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            self._log(logging.WARNING, "Corpus re-index dispatch failed: %s", exc)
            return 0, str(exc)

    def wait_for_worker(self, timeout: float = 10.0) -> bool:
        """Wait for any active background setup/switch worker thread (primarily for tests)."""
        thread = self._worker_thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def _navigate_window(self, url: str) -> None:
        window = self._window
        if window is None:
            return
        try:
            window.load_url(url)
        except Exception as exc:
            self._log(logging.WARNING, "Failed to navigate WebView2 to %s: %s", url, exc)

    def start_setup(
        self,
        compute_profile: str = "cpu",
        reindex_sparse: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        """Start first-run verification, compute runtime setup, model downloads, and sidecars."""
        init_progress = {
            "step": "runtime_check",
            "percent": 3.0,
            "detail": "Running Python C-extension self-test...",
            "error": None,
            "remediation": None,
            "error_category": None,
            "is_running": True,
            "completed": False,
            "steps_update": {"runtime_check": "running", "compute_runtime": "pending"},
            "target_url": None,
            "reindex_recommended": False,
        }

        def _build_worker(target_profile: Literal["cpu", "cuda"]) -> Callable[[], None]:
            def _worker() -> None:
                prev_profile = self.config.compute_profile
                prev_cuda_ready = self.config.cuda_runtime_ready
                prev_sparse_override = self.config.sparse_mode_override
                try:
                    self._log(logging.INFO, "Starting desktop bootstrap (profile=%s)...", target_profile)
                    rt_result = self.bootstrap.verify_python_runtime()
                    if not rt_result.ok:
                        msg = rt_result.error_detail or "Python runtime self-test failed."
                        self._log(logging.ERROR, "Runtime check failed (%s): %s", rt_result.error_category, msg)
                        self._update_progress(
                            step="runtime_check",
                            percent=5.0,
                            detail=msg,
                            error=msg,
                            remediation=rt_result.remediation,
                            error_category=rt_result.error_category,
                            is_running=False,
                            completed=False,
                            steps_update={"runtime_check": "failed"},
                        )
                        return

                    self._update_progress(
                        step="compute_runtime",
                        percent=10.0,
                        detail=f"Preparing compute runtime ({target_profile.upper()})...",
                        steps_update={"runtime_check": "ok", "compute_runtime": "running"},
                    )

                    def _on_compute_progress(prog: BootstrapProgress) -> None:
                        scaled = 10.0 + (prog.percent / 100.0) * 25.0
                        self._update_progress(
                            step=prog.step,
                            percent=scaled,
                            detail=prog.detail,
                            is_running=True,
                            completed=False,
                        )

                    compute_res = self.bootstrap.ensure_compute_runtime(
                        target_profile,
                        is_backend_running=lambda: self.supervisor.is_backend_running,
                        reindex_sparse=reindex_sparse,
                        progress_cb=_on_compute_progress,
                    )
                    if not compute_res.ok:
                        self._rollback_compute_config(prev_profile, prev_cuda_ready, prev_sparse_override)
                        self._log(logging.ERROR, "Compute runtime setup failed: %s", compute_res.detail)
                        self._update_progress(
                            step="compute_runtime",
                            percent=20.0,
                            detail=compute_res.detail,
                            error=compute_res.detail,
                            is_running=False,
                            completed=False,
                            steps_update={"compute_runtime": "failed"},
                        )
                        return

                    if target_profile == "cuda":
                        cuda_rt = self.bootstrap.verify_python_runtime()
                        if not cuda_rt.ok:
                            self._rollback_compute_config(prev_profile, prev_cuda_ready, prev_sparse_override)
                            msg = cuda_rt.error_detail or "CUDA Python runtime verification failed."
                            self._log(logging.ERROR, "CUDA runtime verification failed: %s", msg)
                            self._update_progress(
                                step="compute_runtime",
                                percent=34.0,
                                detail=msg,
                                error=msg,
                                remediation=cuda_rt.remediation,
                                error_category=cuda_rt.error_category,
                                is_running=False,
                                completed=False,
                                steps_update={"compute_runtime": "failed"},
                            )
                            return

                    self._update_progress(
                        step="models",
                        percent=35.0,
                        detail="Checking offline AI model groups...",
                        steps_update={"compute_runtime": "ok"},
                    )

                    def _on_models_progress(prog: BootstrapProgress) -> None:
                        scaled = 35.0 + (prog.percent / 100.0) * 57.0
                        self._update_progress(
                            step=prog.step,
                            percent=scaled,
                            detail=prog.detail,
                            models=prog.models if prog.models else None,
                            is_running=True,
                            completed=False,
                        )

                    self.bootstrap.ensure_models_ready(progress_cb=_on_models_progress)

                    self.config = self.bootstrap.config
                    self.supervisor.config = self.config

                    self._update_progress(
                        step="start_sidecars",
                        percent=94.0,
                        detail="Starting Qdrant & Second Brain API sidecars...",
                        models={spec.key: "cached" for spec in MODEL_SPECS},
                        is_running=True,
                        completed=False,
                    )

                    self.supervisor.start()
                    api_url = self.supervisor.api_url

                    detail_msg = "Setup complete! Opening Second Brain workspace..."
                    if reindex_sparse:
                        queued, reindex_err = self._trigger_corpus_reindex(api_url)
                        if reindex_err is not None:
                            detail_msg = (
                                "Setup complete (warning: background sparse re-index trigger failed; "
                                f"retry from workspace): {reindex_err}"
                            )
                        elif queued > 0:
                            detail_msg = f"Setup complete! Re-indexing {queued} existing document(s) in background..."

                    self._update_progress(
                        step="complete",
                        percent=100.0,
                        detail=detail_msg,
                        error=None,
                        remediation=None,
                        is_running=False,
                        completed=True,
                        target_url=api_url,
                        reindex_recommended=compute_res.reindex_recommended,
                    )
                    self._log(logging.INFO, "Bootstrap complete; navigating to %s", api_url)
                    self._navigate_window(api_url)
                except Exception as exc:
                    self._rollback_compute_config(prev_profile, prev_cuda_ready, prev_sparse_override)
                    self._log_exception("Desktop bootstrap failed")
                    self._update_progress(
                        step="error",
                        detail=str(exc),
                        error=str(exc),
                        is_running=False,
                        completed=False,
                    )

            return _worker

        return self._launch_guarded_worker(
            compute_profile=compute_profile,
            force=force,
            init_progress=init_progress,
            worker_factory=_build_worker,
            thread_name="desktop-bootstrap-setup",
        )

    def switch_compute_profile(
        self,
        compute_profile: str,
        reindex_sparse: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        """Safely swap CPU/CUDA compute runtime with `stop_backend()` before wheel modification."""
        init_progress = {
            "step": "stop_backend",
            "percent": 8.0,
            "detail": "Stopping backend sidecar to release Python DLL/PYD locks...",
            "error": None,
            "remediation": None,
            "error_category": None,
            "is_running": True,
            "completed": False,
            "steps_update": {"runtime_check": "pending", "compute_runtime": "running"},
            "target_url": None,
            "reindex_recommended": False,
        }

        def _build_worker(target_profile: Literal["cpu", "cuda"]) -> Callable[[], None]:
            def _worker() -> None:
                prev_profile = self.config.compute_profile
                prev_cuda_ready = self.config.cuda_runtime_ready
                prev_sparse_override = self.config.sparse_mode_override
                try:
                    self._log(
                        logging.INFO,
                        "Switching compute profile to %s (reindex_sparse=%s)...",
                        target_profile,
                        reindex_sparse,
                    )
                    existing_sparse = self._inspect_collection_sparse_mode_sync()

                    # Stop uvicorn BEFORE modifying runtime wheels to prevent WinError 32 file locks.
                    self.supervisor.stop_backend(timeout_sec=10.0)

                    def _on_compute_progress(prog: BootstrapProgress) -> None:
                        scaled = 15.0 + (prog.percent / 100.0) * 60.0
                        self._update_progress(
                            step=prog.step,
                            percent=scaled,
                            detail=prog.detail,
                            is_running=True,
                            completed=False,
                        )

                    compute_res = self.bootstrap.ensure_compute_runtime(
                        target_profile,
                        is_backend_running=lambda: self.supervisor.is_backend_running,
                        existing_sparse_mode=existing_sparse,
                        reindex_sparse=reindex_sparse,
                        progress_cb=_on_compute_progress,
                    )
                    if not compute_res.ok:
                        self._rollback_compute_config(prev_profile, prev_cuda_ready, prev_sparse_override)
                        self._log(logging.ERROR, "Compute profile switch failed: %s", compute_res.detail)
                        self._update_progress(
                            step="compute_runtime",
                            percent=40.0,
                            detail=compute_res.detail,
                            error=compute_res.detail,
                            is_running=False,
                            completed=False,
                            steps_update={"compute_runtime": "failed"},
                        )
                        return

                    self._update_progress(
                        step="runtime_check",
                        percent=80.0,
                        detail="Verifying target compute Python runtime...",
                        steps_update={"compute_runtime": "ok", "runtime_check": "running"},
                    )

                    rt_result = self.bootstrap.verify_python_runtime()
                    if not rt_result.ok:
                        self._rollback_compute_config(prev_profile, prev_cuda_ready, prev_sparse_override)
                        msg = rt_result.error_detail or "Runtime verification failed after compute switch."
                        self._log(logging.ERROR, "Post-switch runtime check failed: %s", msg)
                        self._update_progress(
                            step="runtime_check",
                            percent=82.0,
                            detail=msg,
                            error=msg,
                            remediation=rt_result.remediation,
                            error_category=rt_result.error_category,
                            is_running=False,
                            completed=False,
                            steps_update={"runtime_check": "failed"},
                        )
                        return

                    self.config = self.bootstrap.config
                    self.supervisor.config = self.config

                    self._update_progress(
                        step="restart_backend",
                        percent=90.0,
                        detail="Restarting Second Brain API backend on new compute profile...",
                        steps_update={"runtime_check": "ok"},
                    )

                    if not self.supervisor.is_qdrant_running:
                        self.supervisor.start_qdrant()
                    self.supervisor.restart_backend(self.config)
                    api_url = self.supervisor.api_url

                    detail_msg = f"Switched to {target_profile.upper()} mode. Returning to workspace..."
                    if reindex_sparse:
                        queued, reindex_err = self._trigger_corpus_reindex(api_url)
                        if reindex_err is not None:
                            detail_msg = (
                                f"Switched to {target_profile.upper()} mode, "
                                f"but background re-index dispatch failed: {reindex_err}"
                            )
                            self.tray.notify(detail_msg, title="Second Brain")
                        elif queued > 0:
                            detail_msg = (
                                f"Switched to {target_profile.upper()} mode. "
                                f"Re-indexing {queued} document(s) in background..."
                            )
                            self.tray.notify(
                                f"Switched to {target_profile.upper()} mode. Re-indexing {queued} document(s) for learned sparse vectors.",
                                title="Second Brain",
                            )
                        else:
                            detail_msg = (
                                f"Switched to {target_profile.upper()} mode (no ready documents needed re-indexing)."
                            )
                    elif compute_res.reindex_recommended:
                        self.tray.notify(
                            "Switched to CUDA mode. Existing collection remains pinned to BM25 until re-indexed.",
                            title="Second Brain",
                        )

                    self._bridge_mode = "setup"
                    self._update_progress(
                        step="complete",
                        percent=100.0,
                        detail=detail_msg,
                        error=None,
                        remediation=None,
                        is_running=False,
                        completed=True,
                        target_url=api_url,
                        reindex_recommended=compute_res.reindex_recommended,
                    )

                    self._navigate_window(api_url)
                except Exception as exc:
                    self._rollback_compute_config(prev_profile, prev_cuda_ready, prev_sparse_override)
                    self._log_exception("Compute profile switch failed")
                    self._update_progress(
                        step="error",
                        detail=str(exc),
                        error=str(exc),
                        is_running=False,
                        completed=False,
                    )

            return _worker

        return self._launch_guarded_worker(
            compute_profile=compute_profile,
            force=force,
            init_progress=init_progress,
            worker_factory=_build_worker,
            thread_name="desktop-compute-switch",
        )

    def pick_models_folder(self) -> dict[str, Any]:
        """Open native folder dialog and import pre-cached models via `import_local_models_dir`."""
        window = self._window
        if window is None:
            return {"ok": False, "cancelled": True, "error": "Desktop window is not active."}

        webview_mod = self._resolve_webview()
        file_dialog_enum = getattr(webview_mod, "FileDialog", None) if webview_mod is not None else None
        dialog_type = (
            getattr(file_dialog_enum, "FOLDER", None)
            if file_dialog_enum is not None
            else getattr(webview_mod, "FOLDER_DIALOG", 20)
        )
        if dialog_type is None:
            dialog_type = getattr(webview_mod, "FOLDER_DIALOG", 20)

        try:
            selected = window.create_file_dialog(dialog_type)
        except Exception as exc:
            self._log(logging.WARNING, "Folder dialog failed: %s", exc)
            return {"ok": False, "cancelled": False, "error": str(exc)}

        folder_path: str | None = None
        if isinstance(selected, (list, tuple)):
            folder_path = str(selected[0]) if selected else None
        elif isinstance(selected, str) and selected.strip():
            folder_path = selected.strip()

        if not folder_path:
            return {"ok": False, "cancelled": True}

        try:
            status = self.bootstrap.import_local_models_dir(folder_path)
            self.config = self.bootstrap.config
            model_map: dict[str, str] = {}
            models_serialized: list[dict[str, Any]] = []
            for item in status.get("models", []):
                item_dict = item.model_dump() if hasattr(item, "model_dump") else dict(item)
                models_serialized.append(item_dict)
                key = str(item_dict.get("key", ""))
                if key:
                    model_map[key] = "cached" if item_dict.get("cached") else "pending"

            self._update_progress(
                detail=f"Imported local models from {folder_path}",
                models=model_map,
            )
            return {
                "ok": True,
                "cancelled": False,
                "selected_path": folder_path,
                "models_ready": bool(status.get("models_ready")),
                "total_size_mb": float(status.get("total_size_mb", 0.0)),
                "models": models_serialized,
            }
        except Exception as exc:
            self._log(logging.WARNING, "Local model import failed for '%s': %s", folder_path, exc)
            return {"ok": False, "cancelled": False, "selected_path": folder_path, "error": str(exc)}

    def open_workspace(self) -> dict[str, Any]:
        """Navigate the WebView2 window back to the Second Brain React workspace."""
        try:
            if not self.supervisor.is_qdrant_running or not self.supervisor.is_backend_running:
                if self.needs_bootstrap():
                    return {
                        "ok": False,
                        "error": "Models and runtime must be verified before opening the workspace.",
                    }
                self.supervisor.start()
            self._bridge_mode = "setup"
            self._navigate_window(self.supervisor.api_url)
            return {"ok": True, "url": self.supervisor.api_url}
        except Exception as exc:
            self._log_exception("Failed to open workspace")
            return {"ok": False, "error": str(exc)}

    def open_external_url(self, url: str) -> bool:
        """Validate `http://` / `https://` scheme and open external link in system browser."""
        try:
            raw_url = (url or "").strip()
            if not raw_url or any(ch in raw_url for ch in ("\r", "\n", "\x00")):
                return False
            parsed = urllib.parse.urlparse(raw_url)
            if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
                return False
            return bool(webbrowser.open(parsed.geturl()))
        except Exception as exc:
            self._log(logging.WARNING, "Failed to open external URL '%s': %s", url, exc)
            return False

    def focus_window(self) -> None:
        """Restore and focus the desktop window, or reopen Edge fallback app window."""
        if self._window is not None:
            try:
                self._window.show()
                if hasattr(self._window, "restore"):
                    self._window.restore()
            except Exception as exc:
                self._log(logging.WARNING, "Failed to restore WebView2 window: %s", exc)
            _focus_win32_window("Second Brain")
            return

        if self._using_edge_fallback and self._fallback_url:
            self._launch_edge_app(self._fallback_url)

    def open_hardware_mode(self) -> None:
        """Switch window to `setup_ui.html?mode=switch` and bring it to foreground."""
        self._bridge_mode = "switch"
        with self._progress_lock:
            if not self._progress_state["is_running"]:
                fresh = self._default_progress_state()
                fresh["detail"] = "Inspect hardware or switch between CPU and CUDA 12.4 mode."
                self._progress_state = fresh
        switch_url = self.setup_ui_url(mode="switch")
        if self._window is not None:
            self._navigate_window(switch_url)
            self.focus_window()
            return

        if self._using_edge_fallback:
            self.tray.notify(
                "Hardware & Compute Mode requires the embedded WebView2 window.",
                title="Second Brain",
            )

    def reclaim_memory_async(self) -> None:
        """Trigger `/api/v1/system/memory/reclaim` in a background thread and notify via tray."""

        def _worker() -> None:
            if not self.supervisor.is_backend_running:
                self.tray.notify("Backend service is not running yet.", title="Second Brain")
                return

            url = f"{self.supervisor.api_url}/api/v1/system/memory/reclaim"
            req = urllib.request.Request(url, data=b"", method="POST")
            try:
                with urllib.request.urlopen(req, timeout=15.0) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                payload = json.loads(raw)
                mem = payload.get("memory") or {}
                rss_mb = mem.get("process_rss_mb")
                rss_text = f"{float(rss_mb):.0f} MB RSS" if rss_mb is not None else "RSS updated"
                if payload.get("deferred"):
                    active_jobs = payload.get("active_worker_jobs", 0)
                    msg = f"Ingestion active ({active_jobs} job(s)); worker reclaim deferred ({rss_text})."
                else:
                    parts: list[str] = []
                    if payload.get("worker_pool_reclaimed"):
                        parts.append("worker pool reclaimed")
                    else:
                        parts.append("worker pool idle")
                    if payload.get("cuda_cache_cleared"):
                        parts.append("VRAM cache cleared")
                    parts.append(rss_text)
                    if mem.get("cuda_available") and mem.get("cuda_allocated_mb") is not None:
                        parts.append(f"CUDA {float(mem['cuda_allocated_mb']):.0f} MB")
                    msg = " | ".join(parts)
                self.tray.notify(msg, title="Second Brain — Memory Reclaim")
            except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
                self._log(logging.WARNING, "Tray memory reclaim failed: %s", exc)
                self.tray.notify(f"Memory reclaim failed: {exc}", title="Second Brain")

        threading.Thread(target=_worker, name="desktop-memory-reclaim", daemon=True).start()

    def _on_window_closing(self) -> bool:
        """Hide window to system tray on close (`X`) unless `is_quitting` is True."""
        if self.is_quitting:
            return True

        if self._window is not None:
            try:
                self._window.hide()
            except Exception as exc:
                self._log(logging.WARNING, "Failed to hide window on close: %s", exc)

        if self.config.show_tray_hint:
            self.tray.notify(
                "Second Brain is still running in the system tray",
                title="Second Brain",
            )
            self.config.show_tray_hint = False
            try:
                save_desktop_config(self.config, self.paths)
            except Exception as exc:
                self._log(logging.WARNING, "Failed to persist show_tray_hint=False: %s", exc)

        return False

    def _on_window_loaded(self) -> None:
        """Attach external-link interception script whenever a document finishes loading."""
        window = self._window
        if window is None:
            return
        try:
            js = _build_external_link_interceptor_js(self.supervisor.api_port)
            window.evaluate_js(js)
        except Exception as exc:
            self._log(logging.DEBUG, "External link hook injection skipped: %s", exc)

    def _resolve_webview(self) -> Any | None:
        if self._webview_module is not None:
            return self._webview_module
        try:
            import webview

            return webview
        except Exception as exc:
            self._log(logging.ERROR, "Failed to import pywebview: %s", exc)
            return None

    def _launch_edge_app(self, target_url: str) -> None:
        """Fallback launcher invoking Microsoft Edge `--app` with isolated user-data dir."""
        self.paths.webview2_cache_dir.mkdir(parents=True, exist_ok=True)
        if self._edge_launcher is not None:
            self._edge_launcher(target_url, self.paths.webview2_cache_dir)
            return

        edge_exe = _find_msedge_exe()
        if edge_exe is not None:
            subprocess.Popen(
                [
                    str(edge_exe),
                    f"--app={target_url}",
                    f"--user-data-dir={self.paths.webview2_cache_dir}",
                ]
            )
            return

        self._log(logging.WARNING, "msedge.exe not found; falling back to default web browser for %s", target_url)
        webbrowser.open(target_url)

    def request_quit(self) -> None:
        """Trigger ordered shutdown asynchronously (safe from control-channel socket thread)."""
        threading.Thread(target=self.quit, name="desktop-quit", daemon=True).start()

    def quit(self) -> None:
        """Perform ordered desktop shutdown: control channel, sidecars, tray, window, mutex."""
        current_thread = threading.current_thread()
        with self._quit_lock:
            self.is_quitting = True
            if self._quit_done:
                wait_for_owner = current_thread is not self._quit_owner_thread
            else:
                self._quit_done = True
                self._quit_owner_thread = current_thread
                wait_for_owner = False

        if wait_for_owner:
            # Ensure callers (e.g. main thread finally block) wait until the active shutdown finishes.
            self._shutdown_complete.wait(timeout=15.0)
            return

        self._log(logging.INFO, "DesktopApp ordered shutdown initiated.")
        try:
            try:
                self._control_server.stop()
            except Exception as exc:
                self._log(logging.WARNING, "Error stopping control server: %s", exc)

            try:
                self.supervisor.shutdown()
            except Exception as exc:
                self._log(logging.WARNING, "Error shutting down sidecars: %s", exc)

            try:
                self.tray.stop()
            except Exception as exc:
                self._log(logging.WARNING, "Error stopping system tray: %s", exc)

            window = self._window
            self._window = None
            if window is not None:
                try:
                    window.destroy()
                except Exception as exc:
                    self._log(logging.DEBUG, "Error destroying WebView2 window: %s", exc)

            self._close_logger()
            try:
                self._mutex.release()
            except Exception:
                pass
        finally:
            self._shutdown_complete.set()
            self._quit_event.set()

    def run(self) -> int:
        """Run the desktop application lifecycle until explicit quit or window termination."""
        self.paths.ensure_directories(self.config)
        self._ensure_logger()

        if not self._mutex.acquire():
            self._log(logging.INFO, "Another Second Brain desktop instance is active; sending FOCUS.")
            if not send_control_command(self.paths, "FOCUS"):
                _focus_win32_window("Second Brain")
            self._close_logger()
            return 0

        try:
            self._control_server.start()
            bootstrap_needed = self.needs_bootstrap()

            if bootstrap_needed:
                self._bridge_mode = "setup"
                initial_url = self.setup_ui_url(mode="setup")
                self._log(logging.INFO, "Bootstrap required; opening setup UI without starting sidecars.")
            else:
                self._bridge_mode = "setup"
                self.supervisor.start()
                initial_url = self.supervisor.api_url
                self._log(logging.INFO, "Warm launch ready; sidecars started at %s", initial_url)

            self.tray.start()

            try:
                webview_mod = self._resolve_webview()
                if webview_mod is None:
                    raise RuntimeError("pywebview runtime could not be imported.")

                if hasattr(webview_mod, "settings") and isinstance(webview_mod.settings, dict):
                    webview_mod.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True

                window = webview_mod.create_window(
                    "Second Brain",
                    url=initial_url,
                    js_api=self.bridge_api,
                    width=1360,
                    height=880,
                    min_size=(960, 640),
                )
                self._window = window

                events = getattr(window, "events", None)
                if events is not None:
                    if hasattr(events, "closing"):
                        events.closing += self._on_window_closing
                    if hasattr(events, "loaded"):
                        events.loaded += self._on_window_loaded

                start_kwargs: dict[str, Any] = {
                    "storage_path": str(self.paths.webview2_cache_dir),
                    "private_mode": False,
                }
                if sys.platform == "win32":
                    start_kwargs["gui"] = "edgechromium"

                webview_mod.start(**start_kwargs)
            except Exception:
                self._log_exception("WebView2 startup failed; falling back to Microsoft Edge --app mode")
                self._window = None
                self._using_edge_fallback = True
                self._fallback_url = initial_url
                self._launch_edge_app(initial_url)
                self._quit_event.wait()

            return 0
        finally:
            self.quit()


def quit_running_instance(
    paths: DesktopPaths | None = None,
    *,
    mutex_name: str = DEFAULT_MUTEX_NAME,
    timeout_sec: float = 10.0,
    poll_interval_sec: float = 0.05,
) -> int:
    """Signal an active `DesktopApp` via `QUIT` and wait until its mutex is released."""
    resolved_paths = paths or DesktopPaths()
    sent = send_control_command(resolved_paths, "QUIT", timeout=2.0)

    if not sent and not is_mutex_held(mutex_name):
        return 0

    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    while is_mutex_held(mutex_name) and time.monotonic() < deadline:
        time.sleep(poll_interval_sec)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Desktop entry point supporting normal single-instance launch and `--quit` uninstaller signal."""
    parser = argparse.ArgumentParser(prog="second-brain-desktop")
    parser.add_argument(
        "--quit",
        action="store_true",
        help="Signal a running Second Brain desktop instance to shut down cleanly and wait for exit.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    paths = DesktopPaths()
    if args.quit:
        return quit_running_instance(paths, mutex_name=DEFAULT_MUTEX_NAME, timeout_sec=10.0)

    mutex = SingleInstanceMutex(name=DEFAULT_MUTEX_NAME)
    if not mutex.acquire():
        if not send_control_command(paths, "FOCUS", timeout=2.0):
            _focus_win32_window("Second Brain")
        return 0

    app = DesktopApp(paths=paths, mutex=mutex)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
