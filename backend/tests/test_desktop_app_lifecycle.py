"""Lifecycle, tray, WebView2 bridge, and CLI tests for Phase 3 desktop application."""
from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.desktop import (
    MODEL_SPECS,
    BootstrapProgress,
    ComputeRuntimeResult,
    DesktopApp,
    DesktopBridgeApi,
    DesktopConfig,
    DesktopPaths,
    GpuAdapterInfo,
    HardwareReport,
    RuntimeBootstrapService,
    RuntimeCheckResult,
    SidecarSupervisor,
    SystemTrayController,
    build_tray_icon_image,
    load_desktop_config,
    main,
    save_desktop_config,
)
from src.desktop.app import _SETUP_HTML_PATH


class _FakeEvent:
    """Minimal pywebview-compatible event supporting `+=` subscription and `.fire()`."""

    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def __iadd__(self, handler: Any) -> _FakeEvent:
        self.handlers.append(handler)
        return self

    def fire(self, *args: Any, **kwargs: Any) -> list[Any]:
        return [h(*args, **kwargs) for h in list(self.handlers)]


class _FakeWindow:
    """Mock pywebview `Window` capturing URL navigations, show/hide/destroy, and events."""

    def __init__(self, title: str, url: str | None = None, js_api: Any = None, **kwargs: Any) -> None:
        self.title = title
        self.url = url
        self.js_api = js_api
        self.kwargs = kwargs
        self.events = types.SimpleNamespace(
            closing=_FakeEvent(),
            loaded=_FakeEvent(),
        )
        self.loaded_urls: list[str] = [url] if url else []
        self.hide = MagicMock()
        self.show = MagicMock()
        self.restore = MagicMock()
        self.destroy = MagicMock()
        self.evaluate_js = MagicMock()
        self.create_file_dialog = MagicMock(return_value=())

    def load_url(self, url: str) -> None:
        self.url = url
        self.loaded_urls.append(url)


def _make_fake_webview(*, start_side_effect: Any = None) -> types.SimpleNamespace:
    created_windows: list[_FakeWindow] = []
    start_calls: list[dict[str, Any]] = []

    def create_window(title: str, url: str | None = None, js_api: Any = None, **kwargs: Any) -> _FakeWindow:
        win = _FakeWindow(title=title, url=url, js_api=js_api, **kwargs)
        created_windows.append(win)
        return win

    def start(**kwargs: Any) -> None:
        start_calls.append(kwargs)
        if callable(start_side_effect):
            start_side_effect(created_windows, kwargs)
        elif isinstance(start_side_effect, BaseException):
            raise start_side_effect

    return types.SimpleNamespace(
        FOLDER_DIALOG=20,
        FileDialog=types.SimpleNamespace(FOLDER=20),
        settings={"OPEN_EXTERNAL_LINKS_IN_BROWSER": True},
        create_window=create_window,
        start=start,
        created_windows=created_windows,
        start_calls=start_calls,
    )


def _make_fake_pystray() -> types.SimpleNamespace:
    created_icons: list[Any] = []

    class FakeMenuItem:
        def __init__(self, text: str, action: Any, default: bool = False, **kwargs: Any) -> None:
            self.text = text
            self.action = action
            self.default = default
            self.kwargs = kwargs

    class FakeMenu:
        SEPARATOR = "---"

        def __init__(self, *items: Any) -> None:
            self.items = list(items)

    class FakeIcon:
        HAS_NOTIFICATION = True

        def __init__(self, name: str, icon: Any, title: str, menu: Any) -> None:
            self.name = name
            self.icon = icon
            self.title = title
            self.menu = menu
            self.started = False
            self.stopped = False
            self.notifications: list[tuple[str, str | None]] = []
            created_icons.append(self)

        def run_detached(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

        def notify(self, message: str, title: str | None = None) -> None:
            self.notifications.append((message, title))

    return types.SimpleNamespace(
        Menu=FakeMenu,
        MenuItem=FakeMenuItem,
        Icon=FakeIcon,
        created_icons=created_icons,
    )


def _make_paths(tmp_path: Path) -> DesktopPaths:
    app_dir = tmp_path / "staged"
    (app_dir / "app" / "backend").mkdir(parents=True, exist_ok=True)
    (app_dir / "app" / "frontend" / "dist").mkdir(parents=True, exist_ok=True)
    (app_dir / "bin").mkdir(parents=True, exist_ok=True)
    (app_dir / "venv" / "Scripts").mkdir(parents=True, exist_ok=True)
    (app_dir / "venv" / "Scripts" / "python.exe").write_text("", encoding="utf-8")
    return DesktopPaths(app_dir=app_dir, user_data_dir=tmp_path / "LocalAppData" / "SecondBrain")


def _populate_models(models_dir: Path) -> None:
    for repo in ("BAAI/bge-m3", "convaiinnovations/laya-multilingual"):
        snap = models_dir / "huggingface" / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / "rev1"
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "config.json").write_text('{"ok": true}', encoding="utf-8")

    bm25 = models_dir / "fastembed" / "models--Qdrant--bm25" / "snapshots" / "rev1"
    bm25.mkdir(parents=True, exist_ok=True)
    (bm25 / "config.json").write_text('{"bm25": true}', encoding="utf-8")

    fr = models_dir / "flashrank" / "ms-marco-MultiBERT-L-12"
    fr.mkdir(parents=True, exist_ok=True)
    (fr / "flashrank-MultiBERT-L-12.onnx").write_bytes(b"onnx")

    ocr = models_dir / "rapidocr"
    ocr.mkdir(parents=True, exist_ok=True)
    (ocr / "det.onnx").write_bytes(b"det")

    dl_layout = models_dir / "docling" / "models" / "docling-project--docling-layout-heron"
    dl_tf = models_dir / "docling" / "models" / "docling-project--docling-models"
    dl_layout.mkdir(parents=True, exist_ok=True)
    dl_tf.mkdir(parents=True, exist_ok=True)
    (dl_layout / "layout.safetensors").write_bytes(b"layout")
    (dl_tf / "tableformer.safetensors").write_bytes(b"tf")

    manifest = {
        "schema_version": 1,
        "models_ready": True,
        "models": {spec.key: {"cached": True} for spec in MODEL_SPECS},
    }
    (models_dir / "models_ready.json").write_text(json.dumps(manifest), encoding="utf-8")


def _make_mock_supervisor(api_url: str = "http://127.0.0.1:18432", qdrant_url: str = "http://127.0.0.1:16432") -> MagicMock:
    sup = MagicMock(spec=SidecarSupervisor)
    sup.api_url = api_url
    sup.qdrant_url = qdrant_url
    sup.api_port = 18432
    sup.qdrant_port = 16432
    sup.is_qdrant_running = False
    sup.is_backend_running = False
    sup.config = DesktopConfig()

    def _start() -> None:
        sup.is_qdrant_running = True
        sup.is_backend_running = True

    def _stop_backend(timeout_sec: float = 10.0) -> None:
        sup.is_backend_running = False

    def _restart_backend(config: DesktopConfig | None = None, timeout_sec: float = 10.0) -> MagicMock:
        if config is not None:
            sup.config = config
        sup.is_backend_running = True
        return MagicMock()

    def _shutdown(timeout_sec: float = 10.0) -> None:
        sup.is_backend_running = False
        sup.is_qdrant_running = False

    sup.start.side_effect = _start
    sup.stop_backend.side_effect = _stop_backend
    sup.restart_backend.side_effect = _restart_backend
    sup.shutdown.side_effect = _shutdown
    return sup


def _cpu_hw_report() -> HardwareReport:
    return HardwareReport(
        cpu_name="Intel Core i7-13700",
        logical_cores=16,
        physical_cores=8,
        total_ram_gb=32.0,
        available_ram_gb=20.0,
        ram_status="optimal",
        gpus=[],
        has_nvidia_cuda_capable_gpu=False,
        gpu_status="cpu_only",
        target_drive="C:",
        free_disk_gb=80.0,
        disk_status="optimal",
        cuda_disk_sufficient=True,
        recommended_profile="cpu",
        recommended_upload_mode="auto",
        recommendation_reason="Standard CPU recommended.",
    )


def _cuda_hw_report() -> HardwareReport:
    return HardwareReport(
        cpu_name="AMD Ryzen 7 7800X3D",
        logical_cores=16,
        physical_cores=8,
        total_ram_gb=32.0,
        available_ram_gb=24.0,
        ram_status="optimal",
        gpus=[
            GpuAdapterInfo(
                name="NVIDIA GeForce RTX 4070",
                vram_gb=12.0,
                vendor="NVIDIA",
                driver_version="561.09",
                is_nvidia=True,
                cuda_capable=True,
            )
        ],
        has_nvidia_cuda_capable_gpu=True,
        gpu_status="cuda_compatible",
        target_drive="C:",
        free_disk_gb=120.0,
        disk_status="optimal",
        cuda_disk_sufficient=True,
        recommended_profile="cuda",
        recommended_upload_mode="auto",
        recommendation_reason="NVIDIA CUDA 12.4 recommended.",
    )


# ==============================================================================
# R3.2 — setup_ui.html Offline Verification
# ==============================================================================


def test_setup_ui_html_is_offline_self_contained_and_complete():
    html_text = _SETUP_HTML_PATH.read_text(encoding="utf-8")

    # Zero external CDN scripts/links or http(s) URLs
    assert "http://" not in html_text.lower()
    assert "https://" not in html_text.lower()
    assert not re.search(r"<script[^>]+src=", html_text, flags=re.IGNORECASE)
    assert not re.search(r"<link[^>]+href=", html_text, flags=re.IGNORECASE)

    # All 6 required model groups + runtime checks + sparse reindex checkbox + bridge methods
    for required_token in (
        "C-Extension Self-Test",
        "Compute Runtime (CPU / CUDA 12.4)",
        "BAAI/bge-m3",
        "convaiinnovations/laya-multilingual",
        "Qdrant/bm25",
        "ms-marco-MultiBERT-L-12",
        "RapidOCR v3 ONNX",
        "Docling v2 Layout &amp; TableFormer",
        "Also re-index existing documents for BGE-M3 learned sparse vectors",
        "get_initial_state",
        "get_progress",
        "start_setup",
        "switch_compute_profile",
        "pick_models_folder",
        "open_workspace",
        "pywebviewready",
        "#0f172a",
        "#1e293b",
        "#38bdf8",
    ):
        assert required_token in html_text


# ==============================================================================
# R3.3 — SystemTrayController & Icon Asset
# ==============================================================================


def test_tray_icon_generation_and_controller_actions(tmp_path: Path):
    paths = _make_paths(tmp_path)
    cfg = DesktopConfig()
    img = build_tray_icon_image(64)
    assert img is not None
    assert img.size == (64, 64)
    assert img.mode == "RGBA"

    fake_pystray = _make_fake_pystray()
    events: list[str] = []

    tray = SystemTrayController(
        paths=paths,
        config=cfg,
        on_open=lambda: events.append("open"),
        on_open_hardware_mode=lambda: events.append("hardware"),
        on_reclaim_memory=lambda: events.append("reclaim"),
        on_quit=lambda: events.append("quit"),
        pystray_module=fake_pystray,
    )

    assert tray.start() is True
    assert tray.is_running is True
    assert len(fake_pystray.created_icons) == 1

    icon = fake_pystray.created_icons[0]
    assert icon.started is True

    menu_items = [item for item in icon.menu.items if item != fake_pystray.Menu.SEPARATOR]
    labels = [item.text for item in menu_items]
    assert labels == [
        "Open Second Brain",
        "Hardware & Compute Mode (CPU / CUDA)...",
        "Reclaim Memory / VRAM Now",
        "Open Data Folder",
        "Open Logs Folder",
        "Quit Second Brain",
    ]
    assert menu_items[0].default is True

    # Invoke each callback cleanly
    for item in menu_items[:3]:
        item.action(icon, item)
    with patch.object(tray, "_open_folder") as mock_open_folder:
        menu_items[3].action(icon, menu_items[3])
        menu_items[4].action(icon, menu_items[4])
        assert mock_open_folder.call_count == 2
        assert mock_open_folder.call_args_list[0].args[0] == paths.resolve_data_dir(cfg)
        assert mock_open_folder.call_args_list[1].args[0] == paths.logs_dir

    menu_items[5].action(icon, menu_items[5])
    assert events == ["open", "hardware", "reclaim", "quit"]

    assert tray.notify("Memory freed", title="Second Brain") is True
    assert icon.notifications == [("Memory freed", "Second Brain")]

    tray.stop()
    assert icon.stopped is True
    assert tray.is_running is False


def test_tray_reclaim_memory_posts_and_notifies(tmp_path: Path):
    paths = _make_paths(tmp_path)
    _populate_models(paths.models_dir)
    cfg = DesktopConfig(models_ready=True)
    save_desktop_config(cfg, paths)

    sup = _make_mock_supervisor()
    sup.is_backend_running = True
    fake_pystray = _make_fake_pystray()
    tray = SystemTrayController(paths=paths, config=cfg, pystray_module=fake_pystray)
    tray.start()

    app = DesktopApp(
        paths=paths,
        config=cfg,
        supervisor=sup,
        tray_controller=tray,
        webview_module=_make_fake_webview(),
    )

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(
        {
            "status": "ok",
            "worker_pool_reclaimed": True,
            "deferred": False,
            "cuda_cache_cleared": True,
            "active_worker_jobs": 0,
            "memory": {
                "process_rss_mb": 412.5,
                "cuda_available": True,
                "cuda_allocated_mb": 128.0,
            },
        }
    ).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    with patch("src.desktop.app.urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        app.reclaim_memory_async()
        # Wait briefly for daemon background worker thread
        deadline = time.monotonic() + 3.0
        while not fake_pystray.created_icons[0].notifications and time.monotonic() < deadline:
            time.sleep(0.02)

    mock_urlopen.assert_called_once()
    req_arg = mock_urlopen.call_args.args[0]
    assert isinstance(req_arg, urllib.request.Request)
    assert req_arg.full_url == "http://127.0.0.1:18432/api/v1/system/memory/reclaim"
    assert req_arg.get_method() == "POST"

    msg, title = fake_pystray.created_icons[0].notifications[0]
    assert "412 MB RSS" in msg or "413 MB RSS" in msg
    assert "worker pool reclaimed" in msg
    assert "VRAM cache cleared" in msg
    assert "Memory Reclaim" in (title or "")
    tray.stop()


# ==============================================================================
# R3.4 — Window Close (Minimize-to-Tray) vs Quit & Single Instance
# ==============================================================================


def test_window_close_hides_and_shows_one_time_tray_hint_while_quit_shuts_down(tmp_path: Path):
    paths = _make_paths(tmp_path)
    _populate_models(paths.models_dir)
    cfg = DesktopConfig(models_ready=True, show_tray_hint=True)
    save_desktop_config(cfg, paths)

    sup = _make_mock_supervisor()
    fake_pystray = _make_fake_pystray()
    tray = SystemTrayController(paths=paths, config=cfg, pystray_module=fake_pystray)

    observed_events: dict[str, Any] = {}

    def on_start(windows: list[_FakeWindow], start_kwargs: dict[str, Any]) -> None:
        win = windows[0]
        # 1. User clicks X (closing while is_quitting=False)
        res1 = win.events.closing.fire()
        observed_events["close_res_1"] = res1
        observed_events["shutdown_after_close_1"] = sup.shutdown.call_count

        # 2. Second close X should NOT re-notify because show_tray_hint persisted False
        res2 = win.events.closing.fire()
        observed_events["close_res_2"] = res2

        # Simulate page load -> JS external link hook injected
        win.events.loaded.fire()
        observed_events["eval_js_called"] = win.evaluate_js.call_count

        # 3. User triggers Tray Quit
        tray._handle_quit()
        observed_events["is_quitting_after_quit"] = app.is_quitting

    fake_webview = _make_fake_webview(start_side_effect=on_start)
    mutex = MagicMock()
    mutex.acquire.return_value = True

    app = DesktopApp(
        paths=paths,
        config=cfg,
        supervisor=sup,
        webview_module=fake_webview,
        tray_controller=tray,
        mutex=mutex,
    )

    rc = app.run()
    assert rc == 0

    win = fake_webview.created_windows[0]
    assert win.url == "http://127.0.0.1:18432"
    assert observed_events["close_res_1"] == [False]
    assert observed_events["close_res_2"] == [False]
    assert observed_events["shutdown_after_close_1"] == 0
    assert observed_events["eval_js_called"] == 1
    assert observed_events["is_quitting_after_quit"] is True

    assert win.hide.call_count == 2
    assert load_desktop_config(paths).show_tray_hint is False
    assert len(fake_pystray.created_icons[0].notifications) == 1
    assert "still running in the system tray" in fake_pystray.created_icons[0].notifications[0][0]

    # Quit performed ordered shutdown
    sup.start.assert_called_once()
    sup.shutdown.assert_called_once()
    assert fake_pystray.created_icons[0].stopped is True
    win.destroy.assert_called_once()
    mutex.release.assert_called_once()

    # Check start() parameters isolated WebView2 cache cleanly
    start_kw = fake_webview.start_calls[0]
    assert start_kw["storage_path"] == str(paths.webview2_cache_dir)
    assert start_kw["private_mode"] is False
    if sys.platform == "win32":
        assert start_kw["gui"] == "edgechromium"


def test_second_instance_sends_focus_and_exits_without_starting_sidecars(tmp_path: Path):
    paths = _make_paths(tmp_path)
    sup = _make_mock_supervisor()
    fake_webview = _make_fake_webview()

    held_mutex = MagicMock()
    held_mutex.acquire.return_value = False

    app = DesktopApp(
        paths=paths,
        supervisor=sup,
        webview_module=fake_webview,
        mutex=held_mutex,
    )

    with patch("src.desktop.app.send_control_command", return_value=True) as mock_send:
        rc = app.run()

    assert rc == 0
    mock_send.assert_called_once_with(paths, "FOCUS")
    sup.start.assert_not_called()
    assert fake_webview.created_windows == []


def test_main_quit_flag_signals_running_instance_and_polls_mutex(tmp_path: Path):
    paths = _make_paths(tmp_path)
    poll_states = iter([True, True, False])

    with (
        patch("src.desktop.app.DesktopPaths", return_value=paths),
        patch("src.desktop.app.send_control_command", return_value=True) as mock_send,
        patch("src.desktop.app.is_mutex_held", side_effect=lambda _name: next(poll_states, False)) as mock_held,
        patch("src.desktop.app.time.sleep") as mock_sleep,
    ):
        rc = main(["--quit"])

    assert rc == 0
    mock_send.assert_called_once_with(paths, "QUIT", timeout=2.0)
    assert mock_held.call_count == 3
    assert mock_sleep.call_count == 2


def test_main_second_instance_sends_focus_and_returns_zero(tmp_path: Path):
    paths = _make_paths(tmp_path)
    mock_mutex = MagicMock()
    mock_mutex.acquire.return_value = False

    with (
        patch("src.desktop.app.DesktopPaths", return_value=paths),
        patch("src.desktop.app.SingleInstanceMutex", return_value=mock_mutex),
        patch("src.desktop.app.send_control_command", return_value=True) as mock_send,
        patch("src.desktop.app.DesktopApp") as mock_app_cls,
    ):
        rc = main([])

    assert rc == 0
    mock_send.assert_called_once_with(paths, "FOCUS", timeout=2.0)
    mock_app_cls.assert_not_called()


# ==============================================================================
# First-Run Gated Bootstrap & Folder Import
# ==============================================================================


def test_first_run_gated_bootstrap_and_bridge_transition(tmp_path: Path):
    paths = _make_paths(tmp_path)
    cfg = DesktopConfig(models_ready=False, compute_profile="cpu")
    save_desktop_config(cfg, paths)

    sup = _make_mock_supervisor()
    bootstrap = MagicMock(spec=RuntimeBootstrapService)
    bootstrap.config = cfg
    bootstrap.inspect_models_status.return_value = {
        "models_ready": False,
        "models": [
            {"key": spec.key, "display_name": spec.display_name, "cached": False, "path": str(paths.models_dir), "size_mb": 0.0, "detail": "Not downloaded"}
            for spec in MODEL_SPECS
        ],
        "total_size_mb": 0.0,
        "models_dir": str(paths.models_dir),
    }

    call_sequence: list[str] = []

    def fake_verify() -> RuntimeCheckResult:
        call_sequence.append("verify_python_runtime")
        return RuntimeCheckResult(ok=True, python_exe=str(paths.base_venv_python))

    def fake_ensure_compute(profile: str, **kwargs: Any) -> ComputeRuntimeResult:
        call_sequence.append(f"ensure_compute_runtime:{profile}")
        assert kwargs["is_backend_running"]() is False
        return ComputeRuntimeResult(ok=True, compute_profile="cpu", python_exe=str(paths.base_venv_python))

    def fake_ensure_models(**kwargs: Any) -> dict[str, Any]:
        call_sequence.append("ensure_models_ready")
        cb = kwargs.get("progress_cb")
        if cb:
            cb(BootstrapProgress(step="complete", percent=100.0, detail="Cached", is_running=False, completed=True))
        return {"models_ready": True, "models": [], "total_size_mb": 3800.0, "models_dir": str(paths.models_dir)}

    def fake_sup_start() -> None:
        call_sequence.append("supervisor.start")
        sup.is_qdrant_running = True
        sup.is_backend_running = True

    bootstrap.verify_python_runtime.side_effect = fake_verify
    bootstrap.ensure_compute_runtime.side_effect = fake_ensure_compute
    bootstrap.ensure_models_ready.side_effect = fake_ensure_models
    sup.start.side_effect = fake_sup_start

    observed: dict[str, Any] = {}

    def on_start(windows: list[_FakeWindow], _kwargs: dict[str, Any]) -> None:
        win = windows[0]
        # Gated startup MUST open setup_ui.html WITHOUT calling supervisor.start()
        observed["initial_url"] = win.url
        observed["sup_calls_before_setup"] = sup.start.call_count

        with patch("src.desktop.app.probe_system_hardware", return_value=_cpu_hw_report()):
            initial = win.js_api.get_initial_state()
            observed["initial_state"] = initial

            res = win.js_api.start_setup("cpu", False)
            observed["start_setup_res"] = res
            assert app.wait_for_worker(timeout=5.0) is True
            observed["progress_after"] = win.js_api.get_progress()

    fake_webview = _make_fake_webview(start_side_effect=on_start)
    fake_pystray = _make_fake_pystray()
    tray = SystemTrayController(paths=paths, config=cfg, pystray_module=fake_pystray)
    mutex = MagicMock()
    mutex.acquire.return_value = True

    app = DesktopApp(
        paths=paths,
        config=cfg,
        supervisor=sup,
        bootstrap=bootstrap,
        webview_module=fake_webview,
        tray_controller=tray,
        mutex=mutex,
    )

    assert app.run() == 0
    assert observed["initial_url"].endswith("setup_ui.html")
    assert observed["sup_calls_before_setup"] == 0
    assert observed["start_setup_res"] == {"ok": True, "started": True}
    assert call_sequence == [
        "verify_python_runtime",
        "ensure_compute_runtime:cpu",
        "ensure_models_ready",
        "supervisor.start",
    ]
    assert observed["progress_after"]["completed"] is True
    assert observed["progress_after"]["percent"] == 100.0
    assert fake_webview.created_windows[0].loaded_urls[-1] == "http://127.0.0.1:18432"


def test_first_run_runtime_failure_aborts_before_starting_sidecars(tmp_path: Path):
    paths = _make_paths(tmp_path)
    cfg = DesktopConfig(models_ready=False, compute_profile="cpu")
    sup = _make_mock_supervisor()
    bootstrap = MagicMock(spec=RuntimeBootstrapService)
    bootstrap.config = cfg
    bootstrap.verify_python_runtime.return_value = RuntimeCheckResult(
        ok=False,
        python_exe=str(paths.base_venv_python),
        error_category="app_control_blocked",
        error_detail="Blocked by Windows Application Control",
        remediation="Allowlist Second Brain folder.",
    )

    app = DesktopApp(
        paths=paths,
        config=cfg,
        supervisor=sup,
        bootstrap=bootstrap,
        webview_module=_make_fake_webview(),
    )
    with patch("src.desktop.app.probe_system_hardware", return_value=_cpu_hw_report()):
        res = app.bridge_api.start_setup("cpu", False)
    assert res == {"ok": True, "started": True}
    assert app.wait_for_worker(timeout=5.0) is True

    prog = app.bridge_api.get_progress()
    assert prog["is_running"] is False
    assert prog["completed"] is False
    assert prog["error_category"] == "app_control_blocked"
    assert "Application Control" in prog["error"]
    assert "Allowlist" in prog["remediation"]
    sup.start.assert_not_called()


def test_pick_models_folder_imports_and_updates_progress(tmp_path: Path):
    paths = _make_paths(tmp_path)
    cfg = DesktopConfig(models_ready=False)
    sup = _make_mock_supervisor()
    bootstrap = MagicMock(spec=RuntimeBootstrapService)
    bootstrap.config = cfg
    usb_dir = tmp_path / "usb_models"
    usb_dir.mkdir()
    bootstrap.import_local_models_dir.return_value = {
        "models_ready": True,
        "total_size_mb": 3835.0,
        "models": [
            {"key": spec.key, "display_name": spec.display_name, "cached": True, "path": str(paths.models_dir), "size_mb": 100.0, "detail": "Cached"}
            for spec in MODEL_SPECS
        ],
    }

    fake_webview = _make_fake_webview()
    win = fake_webview.create_window("Second Brain", url="file:///setup_ui.html")
    win.create_file_dialog.return_value = (str(usb_dir),)

    app = DesktopApp(
        paths=paths,
        config=cfg,
        supervisor=sup,
        bootstrap=bootstrap,
        webview_module=fake_webview,
    )
    app._window = win

    res = app.bridge_api.pick_models_folder()
    assert res["ok"] is True
    assert res["cancelled"] is False
    assert res["models_ready"] is True
    assert res["selected_path"] == str(usb_dir)
    bootstrap.import_local_models_dir.assert_called_once_with(str(usb_dir))
    assert all(v == "cached" for v in app.get_progress()["models"].values())


# ==============================================================================
# Tray CPU <-> CUDA Switch & WinError 32 Order Guard
# ==============================================================================


def test_tray_compute_profile_switch_stops_backend_before_cuda_install(tmp_path: Path):
    paths = _make_paths(tmp_path)
    _populate_models(paths.models_dir)
    cfg = DesktopConfig(models_ready=True, compute_profile="cpu", cuda_runtime_ready=False)
    save_desktop_config(cfg, paths)

    sup = _make_mock_supervisor()
    sup.is_qdrant_running = True
    sup.is_backend_running = True

    bootstrap = MagicMock(spec=RuntimeBootstrapService)
    bootstrap.config = cfg

    order: list[str] = []

    def fake_stop_backend(timeout_sec: float = 10.0) -> None:
        order.append("supervisor.stop_backend")
        sup.is_backend_running = False

    def fake_ensure_compute(target_profile: str, **kwargs: Any) -> ComputeRuntimeResult:
        order.append(f"bootstrap.ensure_compute_runtime:{target_profile}")
        # Crucial WinError 32 invariant: uvicorn MUST already be stopped
        assert kwargs["is_backend_running"]() is False
        assert kwargs["existing_sparse_mode"] == "bm25"
        assert kwargs["reindex_sparse"] is False
        updated_cfg = DesktopConfig(
            models_ready=True,
            compute_profile="cuda",
            cuda_runtime_ready=True,
            sparse_mode_override="bm25",
        )
        bootstrap.config = updated_cfg
        return ComputeRuntimeResult(
            ok=True,
            compute_profile="cuda",
            python_exe=str(paths.cuda_venv_python),
            cuda_runtime_ready=True,
            sparse_mode_override="bm25",
            reindex_recommended=True,
        )

    def fake_verify() -> RuntimeCheckResult:
        order.append("bootstrap.verify_python_runtime")
        return RuntimeCheckResult(ok=True, python_exe=str(paths.cuda_venv_python), cuda_available=True)

    def fake_restart_backend(config: DesktopConfig | None = None, timeout_sec: float = 10.0) -> MagicMock:
        order.append("supervisor.restart_backend")
        sup.config = config
        sup.is_backend_running = True
        return MagicMock()

    sup.stop_backend.side_effect = fake_stop_backend
    sup.restart_backend.side_effect = fake_restart_backend
    bootstrap.ensure_compute_runtime.side_effect = fake_ensure_compute
    bootstrap.verify_python_runtime.side_effect = fake_verify

    fake_webview = _make_fake_webview()
    win = fake_webview.create_window("Second Brain", url=sup.api_url)
    fake_pystray = _make_fake_pystray()
    tray = SystemTrayController(paths=paths, config=cfg, pystray_module=fake_pystray)
    tray.start()

    app = DesktopApp(
        paths=paths,
        config=cfg,
        supervisor=sup,
        bootstrap=bootstrap,
        webview_module=fake_webview,
        tray_controller=tray,
    )
    app._window = win

    # Trigger tray "Hardware & Compute Mode"
    app.open_hardware_mode()
    assert win.url.endswith("setup_ui.html?mode=switch")
    assert win.show.called

    with (
        patch("src.desktop.app.probe_system_hardware", return_value=_cuda_hw_report()),
        patch.object(app, "_inspect_collection_sparse_mode_sync", return_value="bm25"),
    ):
        res = app.bridge_api.switch_compute_profile("cuda", reindex_sparse=False)
        assert res == {"ok": True, "started": True}
        assert app.wait_for_worker(timeout=5.0) is True

    assert order == [
        "supervisor.stop_backend",
        "bootstrap.ensure_compute_runtime:cuda",
        "bootstrap.verify_python_runtime",
        "supervisor.restart_backend",
    ]
    assert win.url == "http://127.0.0.1:18432"
    assert app.get_progress()["completed"] is True
    assert app.get_progress()["reindex_recommended"] is True
    assert any("BM25" in n[0] for n in fake_pystray.created_icons[0].notifications)
    tray.stop()


# ==============================================================================
# Edge --app Fallback, External Links & Bridge Introspection Safety
# ==============================================================================


def test_edge_app_fallback_when_webview2_raises_and_keeps_tray_alive(tmp_path: Path):
    paths = _make_paths(tmp_path)
    _populate_models(paths.models_dir)
    cfg = DesktopConfig(models_ready=True)
    save_desktop_config(cfg, paths)

    sup = _make_mock_supervisor()
    shutdown_completed = threading.Event()

    def _delayed_shutdown(timeout_sec: float = 10.0) -> None:
        time.sleep(0.08)
        sup.is_backend_running = False
        sup.is_qdrant_running = False
        shutdown_completed.set()

    sup.shutdown.side_effect = _delayed_shutdown
    fake_pystray = _make_fake_pystray()
    tray = SystemTrayController(paths=paths, config=cfg, pystray_module=fake_pystray)
    fake_webview = _make_fake_webview(start_side_effect=RuntimeError("WebView2 runtime missing"))

    edge_calls: list[tuple[str, Path]] = []
    app_ref: list[DesktopApp] = []

    def fake_edge_launcher(url: str, cache_dir: Path) -> None:
        edge_calls.append((url, cache_dir))
        # Verify sidecar and tray are STILL active when fallback runs, then simulate daemon quit
        assert sup.is_backend_running is True
        assert tray.is_running is True
        app_ref[0].request_quit()

    mutex = MagicMock()
    mutex.acquire.return_value = True

    app = DesktopApp(
        paths=paths,
        config=cfg,
        supervisor=sup,
        webview_module=fake_webview,
        tray_controller=tray,
        edge_launcher=fake_edge_launcher,
        mutex=mutex,
    )
    app_ref.append(app)

    rc = app.run()
    assert rc == 0
    assert shutdown_completed.is_set()
    assert edge_calls == [("http://127.0.0.1:18432", paths.webview2_cache_dir)]
    assert (paths.logs_dir / "launcher.log").is_file()
    log_text = (paths.logs_dir / "launcher.log").read_text(encoding="utf-8")
    assert "WebView2 runtime missing" in log_text
    sup.shutdown.assert_called_once()
    mutex.release.assert_called_once()


def test_external_link_validation_and_bridge_api_introspection_safety(tmp_path: Path):
    paths = _make_paths(tmp_path)
    app = DesktopApp(paths=paths, config=DesktopConfig())
    api = DesktopBridgeApi(app)

    # Only expected public bridge methods, and all internal objects stay on `_app`
    public_attrs = [name for name in dir(api) if not name.startswith("_")]
    assert set(public_attrs) == {
        "get_initial_state",
        "get_progress",
        "start_setup",
        "switch_compute_profile",
        "pick_models_folder",
        "open_workspace",
        "open_external_url",
    }

    with patch("src.desktop.app.webbrowser.open", return_value=True) as mock_browser:
        assert api.open_external_url("https://docs.python.org/3/") is True
        assert api.open_external_url("http://example.com/guide") is True
        assert api.open_external_url("file:///C:/Windows/System32/calc.exe") is False
        assert api.open_external_url("javascript:alert(1)") is False
        assert api.open_external_url("https://example.com/\r\nbad") is False
        assert api.open_external_url("https://example.com/\x00bad") is False
        assert mock_browser.call_count == 2


def test_stale_cuda_runtime_ready_without_cuda_venv_triggers_bootstrap(tmp_path: Path):
    paths = _make_paths(tmp_path)
    _populate_models(paths.models_dir)
    cfg = DesktopConfig(models_ready=True, compute_profile="cuda", cuda_runtime_ready=True)
    save_desktop_config(cfg, paths)

    sup = _make_mock_supervisor()
    app = DesktopApp(
        paths=paths,
        config=cfg,
        supervisor=sup,
        webview_module=_make_fake_webview(),
    )

    with patch("src.desktop.app.probe_system_hardware", return_value=_cuda_hw_report()):
        assert app.needs_bootstrap() is True
        assert app.get_initial_state()["cuda_runtime_ready"] is False

        paths.cuda_venv_python.parent.mkdir(parents=True, exist_ok=True)
        paths.cuda_venv_python.write_text("", encoding="utf-8")

        assert app.needs_bootstrap() is False
        assert app.get_initial_state()["cuda_runtime_ready"] is True


def test_switch_compute_profile_reindex_dispatch_rollback_and_busy_guard(tmp_path: Path):
    paths = _make_paths(tmp_path)
    _populate_models(paths.models_dir)
    cfg = DesktopConfig(models_ready=True, compute_profile="cpu", cuda_runtime_ready=False)
    save_desktop_config(cfg, paths)

    sup = _make_mock_supervisor()
    sup.is_qdrant_running = True
    sup.is_backend_running = True

    bootstrap = MagicMock(spec=RuntimeBootstrapService)
    bootstrap.config = cfg

    # 1. Verify busy worker guard protects _progress_state from concurrent start_setup/switch calls
    worker_started = threading.Event()
    release_worker = threading.Event()

    def blocking_stop(timeout_sec: float = 10.0) -> None:
        worker_started.set()
        release_worker.wait(timeout=5.0)
        sup.is_backend_running = False

    def ok_ensure(target_profile: str, **kwargs: Any) -> ComputeRuntimeResult:
        updated = DesktopConfig(
            models_ready=True,
            compute_profile=target_profile,  # type: ignore[arg-type]
            cuda_runtime_ready=True,
            sparse_mode_override="auto",
        )
        bootstrap.config = updated
        save_desktop_config(updated, paths)
        return ComputeRuntimeResult(
            ok=True,
            compute_profile=target_profile,  # type: ignore[arg-type]
            python_exe=str(paths.cuda_venv_python),
            cuda_runtime_ready=True,
            sparse_mode_override="auto",
            reindex_recommended=False,
        )

    sup.stop_backend.side_effect = blocking_stop
    bootstrap.ensure_compute_runtime.side_effect = ok_ensure
    bootstrap.verify_python_runtime.return_value = RuntimeCheckResult(ok=True, python_exe=str(paths.cuda_venv_python))

    app = DesktopApp(
        paths=paths,
        config=cfg,
        supervisor=sup,
        bootstrap=bootstrap,
        webview_module=_make_fake_webview(),
    )

    with (
        patch("src.desktop.app.probe_system_hardware", return_value=_cuda_hw_report()),
        patch.object(app, "_inspect_collection_sparse_mode_sync", return_value="bm25"),
        patch.object(app, "_trigger_corpus_reindex", return_value=(3, None)) as mock_reindex,
    ):
        first_res = app.bridge_api.switch_compute_profile("cuda", reindex_sparse=True)
        assert first_res == {"ok": True, "started": True}
        assert worker_started.wait(timeout=3.0)

        # Concurrent invalid call while worker runs must NOT overwrite _progress_state
        concurrent_res = app.bridge_api.start_setup("invalid_profile")
        assert concurrent_res["ok"] is False
        assert "already running" in concurrent_res["error"].lower()
        assert app.get_progress()["is_running"] is True
        assert app.get_progress()["error"] is None

        release_worker.set()
        assert app.wait_for_worker(timeout=5.0) is True
        mock_reindex.assert_called_once_with(sup.api_url)
        assert "Re-indexing 3 document(s)" in app.get_progress()["detail"]

    # 2. Verify post-switch verify_python_runtime failure rolls back DesktopConfig cleanly
    sup.stop_backend.side_effect = lambda timeout_sec=10.0: None
    cfg_cpu = DesktopConfig(models_ready=True, compute_profile="cpu", cuda_runtime_ready=False, sparse_mode_override="bm25")
    save_desktop_config(cfg_cpu, paths)
    app_rb = DesktopApp(
        paths=paths,
        config=cfg_cpu,
        supervisor=sup,
        bootstrap=bootstrap,
        webview_module=_make_fake_webview(),
    )
    bootstrap.config = cfg_cpu
    bootstrap.verify_python_runtime.return_value = RuntimeCheckResult(
        ok=False,
        python_exe=str(paths.cuda_venv_python),
        error_category="app_control_blocked",
        error_detail="Blocked CUDA DLL",
        remediation="Allowlist folder.",
    )

    with (
        patch("src.desktop.app.probe_system_hardware", return_value=_cuda_hw_report()),
        patch.object(app_rb, "_inspect_collection_sparse_mode_sync", return_value="bm25"),
    ):
        rb_res = app_rb.bridge_api.switch_compute_profile("cuda")
        assert rb_res == {"ok": True, "started": True}
        assert app_rb.wait_for_worker(timeout=5.0) is True

    persisted_after_fail = load_desktop_config(paths)
    assert persisted_after_fail.compute_profile == "cpu"
    assert persisted_after_fail.cuda_runtime_ready is False
    assert persisted_after_fail.sparse_mode_override == "bm25"
