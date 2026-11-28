"""Native Edge WebView2 application coordinator, JS bridge API, and single-instance CLI entry."""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import urllib.request
import webbrowser
from collections.abc import Callable, Sequence
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal

from src.desktop import backend_client, webview_host, workflows
from src.desktop.bootstrap import RuntimeBootstrapService
from src.desktop.bridge import DesktopBridgeApi, ProgressStore
from src.desktop.config import DesktopConfig, DesktopPaths, load_desktop_config
from src.desktop.hardware import probe_system_hardware
from src.desktop.job_object import (
    DEFAULT_MUTEX_NAME,
    ControlChannelServer,
    SingleInstanceMutex,
    is_mutex_held,
    send_control_command,
)
from src.desktop.logging_utils import acquire_log_handler, release_log_handler
from src.desktop.supervisor import SidecarSupervisor
from src.desktop.tray import SystemTrayController
from src.desktop.webview_host import _build_external_link_interceptor_js
from src.desktop.win32_window import (
    DESKTOP_APP_USER_MODEL_ID,
    _apply_win32_window_icon,
    _find_msedge_exe,
    _focus_win32_window,
    _set_app_user_model_id,
    resolve_app_icon_path,
)

logger = logging.getLogger(__name__)

_SETUP_HTML_PATH = Path(__file__).with_name("setup_ui.html").resolve()


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
        self._app_log_path: Path | None = None

        self._worker_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None

        self._progress_store = ProgressStore()
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

    def _ensure_logger(self) -> None:
        """Initialize refcounted `launcher.log` handler for desktop UI lifecycle diagnostics."""
        if self._app_log_handler is not None:
            return
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.paths.logs_dir / "launcher.log"
        handler = acquire_log_handler(log_path)
        inst_logger = logging.getLogger(f"second_brain.desktop.app.{id(self)}")
        inst_logger.setLevel(logging.INFO)
        inst_logger.propagate = False
        inst_logger.handlers.clear()
        inst_logger.addHandler(handler)
        self._app_logger = inst_logger
        self._app_log_handler = handler
        self._app_log_path = log_path

    def _close_logger(self) -> None:
        if self._app_logger is not None and self._app_log_handler is not None:
            try:
                self._app_logger.removeHandler(self._app_log_handler)
            except Exception:
                pass
        if self._app_log_path is not None:
            release_log_handler(self._app_log_path)
        self._app_logger = None
        self._app_log_handler = None
        self._app_log_path = None

    def _log(self, level: int, message: str, *args: Any) -> None:
        self._ensure_logger()
        if self._app_logger is not None:
            self._app_logger.log(level, message, *args)

    def _log_exception(self, message: str) -> None:
        self._ensure_logger()
        if self._app_logger is not None:
            self._app_logger.exception(message)

    def _update_progress(self, **kwargs: Any) -> None:
        self._progress_store.update(**kwargs)

    def get_progress(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of current bootstrap/switch progress."""
        return self._progress_store.snapshot()

    def setup_ui_url(self, mode: Literal["setup", "switch"] | None = None) -> str:
        """Return `file://` URI for `setup_ui.html`, optionally tagged `?mode=switch`."""
        base_uri = _SETUP_HTML_PATH.as_uri()
        effective_mode = mode or self._bridge_mode
        if effective_mode == "switch":
            return f"{base_uri}?mode=switch"
        return base_uri

    def resolve_app_icon_path(self) -> Path | None:
        """Resolve `second-brain.ico` across packaged installer and dev repo layouts."""
        return resolve_app_icon_path(self.paths)

    def needs_bootstrap(self) -> bool:
        """Determine whether first-run runtime/model setup must gate sidecar startup."""
        return workflows.needs_bootstrap(self)

    def _inspect_collection_sparse_mode_sync(self) -> Literal["bm25", "bge_m3"] | None:
        """Query Qdrant's active collection `sparse_mode` safely if Qdrant is running."""
        return workflows.inspect_collection_sparse_mode_sync(self)

    def get_initial_state(self) -> dict[str, Any]:
        """Build complete initial state payload consumed by `setup_ui.html`."""
        return workflows.get_initial_state(self)

    def _trigger_corpus_reindex(self, api_url: str) -> tuple[int, str | None]:
        """Request background re-indexing on the restarted backend after sparse-mode change."""
        return backend_client.trigger_corpus_reindex(api_url, log_fn=self._log)

    def wait_for_worker(self, timeout: float = 10.0) -> bool:
        """Wait for any active background setup/switch worker thread (primarily for tests)."""
        thread = self._worker_thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def _navigate_window(self, url: str) -> None:
        webview_host.navigate_window(self, url)

    def start_setup(
        self,
        compute_profile: str = "cpu",
        reindex_sparse: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        """Start first-run verification, compute runtime setup, model downloads, and sidecars."""
        return workflows.start_setup(
            self,
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
        """Safely swap CPU/CUDA compute runtime with `stop_backend()` before wheel modification."""
        return workflows.switch_compute_profile(
            self,
            compute_profile=compute_profile,
            reindex_sparse=reindex_sparse,
            force=force,
        )

    def pick_models_folder(self) -> dict[str, Any]:
        """Open native folder dialog and import pre-cached models via `import_local_models_dir`."""
        return webview_host.pick_models_folder(self)

    def open_workspace(self) -> dict[str, Any]:
        """Navigate the WebView2 window back to the Second Brain React workspace."""
        return webview_host.open_workspace(self)

    def open_external_url(self, url: str) -> bool:
        """Validate `http://` / `https://` scheme and open external link in system browser."""
        return webview_host.open_external_url(self, url)

    def focus_window(self) -> None:
        """Restore and focus the desktop window, or reopen Edge fallback app window."""
        webview_host.focus_window(self)

    def open_hardware_mode(self) -> None:
        """Switch window to `setup_ui.html?mode=switch` and bring it to foreground."""
        webview_host.open_hardware_mode(self)

    def reclaim_memory_async(self) -> None:
        """Trigger `/api/v1/system/memory/reclaim` in a background thread and notify via tray."""
        backend_client.reclaim_memory_async(self)

    def _on_window_closing(self) -> bool:
        """Hide window to system tray on close (`X`) unless `is_quitting` is True."""
        return webview_host.on_window_closing(self)

    def _on_window_loaded(self) -> None:
        """Attach external-link interception script and Win32 icon whenever a document finishes loading."""
        webview_host.on_window_loaded(self)

    def _resolve_webview(self) -> Any | None:
        return webview_host.resolve_webview(self)

    def _launch_edge_app(self, target_url: str) -> None:
        """Fallback launcher invoking Microsoft Edge `--app` with isolated user-data dir."""
        webview_host.launch_edge_app(self, target_url)

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
        if sys.platform == "win32":
            _set_app_user_model_id()
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
            webview_host.run_webview_or_fallback(self, initial_url)
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


__all__ = [
    "DESKTOP_APP_USER_MODEL_ID",
    "DesktopApp",
    "DesktopBridgeApi",
    "_SETUP_HTML_PATH",
    "_apply_win32_window_icon",
    "_build_external_link_interceptor_js",
    "_find_msedge_exe",
    "_focus_win32_window",
    "_set_app_user_model_id",
    "main",
    "quit_running_instance",
]


if __name__ == "__main__":
    raise SystemExit(main())
