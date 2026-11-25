"""WebView2 host lifecycle hooks, JS external-link guard, folder dialog, and Edge fallback."""
from __future__ import annotations

import json
import logging
import subprocess
import urllib.parse
from typing import TYPE_CHECKING, Any

from src.desktop.config import save_desktop_config
from src.desktop.win32_window import _find_msedge_exe, _focus_win32_window

if TYPE_CHECKING:
    from src.desktop.app import DesktopApp


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


def resolve_webview(app: DesktopApp) -> Any | None:
    """Return injected pywebview module or import `webview` lazily."""
    if app._webview_module is not None:
        return app._webview_module
    try:
        import webview

        return webview
    except Exception as exc:
        app._log(logging.ERROR, "Failed to import pywebview: %s", exc)
        return None


def navigate_window(app: DesktopApp, url: str) -> None:
    """Navigate active WebView2 window to `url` if available."""
    window = app._window
    if window is None:
        return
    try:
        window.load_url(url)
    except Exception as exc:
        app._log(logging.WARNING, "Failed to navigate WebView2 to %s: %s", url, exc)


def pick_models_folder(app: DesktopApp) -> dict[str, Any]:
    """Open native folder dialog and import pre-cached models via `import_local_models_dir`."""
    window = app._window
    if window is None:
        return {"ok": False, "cancelled": True, "error": "Desktop window is not active."}

    webview_mod = resolve_webview(app)
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
        app._log(logging.WARNING, "Folder dialog failed: %s", exc)
        return {"ok": False, "cancelled": False, "error": str(exc)}

    folder_path: str | None = None
    if isinstance(selected, (list, tuple)):
        folder_path = str(selected[0]) if selected else None
    elif isinstance(selected, str) and selected.strip():
        folder_path = selected.strip()

    if not folder_path:
        return {"ok": False, "cancelled": True}

    try:
        status = app.bootstrap.import_local_models_dir(folder_path)
        app.config = app.bootstrap.config
        model_map: dict[str, str] = {}
        models_serialized: list[dict[str, Any]] = []
        for item in status.get("models", []):
            item_dict = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            models_serialized.append(item_dict)
            key = str(item_dict.get("key", ""))
            if key:
                model_map[key] = "cached" if item_dict.get("cached") else "pending"

        app._update_progress(
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
        app._log(logging.WARNING, "Local model import failed for '%s': %s", folder_path, exc)
        return {"ok": False, "cancelled": False, "selected_path": folder_path, "error": str(exc)}


def open_workspace(app: DesktopApp) -> dict[str, Any]:
    """Navigate the WebView2 window back to the Second Brain React workspace."""
    try:
        if not app.supervisor.is_qdrant_running or not app.supervisor.is_backend_running:
            if app.needs_bootstrap():
                return {
                    "ok": False,
                    "error": "Models and runtime must be verified before opening the workspace.",
                }
            app.supervisor.start()
        app._bridge_mode = "setup"
        navigate_window(app, app.supervisor.api_url)
        return {"ok": True, "url": app.supervisor.api_url}
    except Exception as exc:
        app._log_exception("Failed to open workspace")
        return {"ok": False, "error": str(exc)}


def open_external_url(app: DesktopApp, url: str) -> bool:
    """Validate `http://` / `https://` scheme and open external link in system browser."""
    from src.desktop import app as app_mod

    try:
        raw_url = (url or "").strip()
        if not raw_url or any(ch in raw_url for ch in ("\r", "\n", "\x00")):
            return False
        parsed = urllib.parse.urlparse(raw_url)
        if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
            return False
        return bool(app_mod.webbrowser.open(parsed.geturl()))
    except Exception as exc:
        app._log(logging.WARNING, "Failed to open external URL '%s': %s", url, exc)
        return False


def focus_window(app: DesktopApp) -> None:
    """Restore and focus the desktop window, or reopen Edge fallback app window."""
    if app._window is not None:
        try:
            app._window.show()
            if hasattr(app._window, "restore"):
                app._window.restore()
        except Exception as exc:
            app._log(logging.WARNING, "Failed to restore WebView2 window: %s", exc)
        _focus_win32_window("Second Brain")
        return

    if app._using_edge_fallback and app._fallback_url:
        launch_edge_app(app, app._fallback_url)


def open_hardware_mode(app: DesktopApp) -> None:
    """Switch window to `setup_ui.html?mode=switch` and bring it to foreground."""
    app._bridge_mode = "switch"
    app._progress_store.reset_for_switch_if_idle()
    switch_url = app.setup_ui_url(mode="switch")
    if app._window is not None:
        navigate_window(app, switch_url)
        focus_window(app)
        return

    if app._using_edge_fallback:
        app.tray.notify(
            "Hardware & Compute Mode requires the embedded WebView2 window.",
            title="Second Brain",
        )


def on_window_closing(app: DesktopApp) -> bool:
    """Hide window to system tray on close (`X`) unless `is_quitting` is True."""
    if app.is_quitting:
        return True

    if app._window is not None:
        try:
            app._window.hide()
        except Exception as exc:
            app._log(logging.WARNING, "Failed to hide window on close: %s", exc)

    if app.config.show_tray_hint:
        app.tray.notify(
            "Second Brain is still running in the system tray",
            title="Second Brain",
        )
        app.config.show_tray_hint = False
        try:
            save_desktop_config(app.config, app.paths)
        except Exception as exc:
            app._log(logging.WARNING, "Failed to persist show_tray_hint=False: %s", exc)

    return False


def on_window_loaded(app: DesktopApp) -> None:
    """Attach external-link interception script and Win32 icon whenever a document finishes loading."""
    from src.desktop import app as app_mod

    window = app._window
    if window is None:
        return
    if app_mod.sys.platform == "win32":
        ico_path = app.resolve_app_icon_path()
        if ico_path is not None:
            raw_title = getattr(window, "title", None)
            title = raw_title if isinstance(raw_title, str) and raw_title else "Second Brain"
            app_mod._apply_win32_window_icon(title, ico_path)
    try:
        js = _build_external_link_interceptor_js(app.supervisor.api_port)
        window.evaluate_js(js)
    except Exception as exc:
        app._log(logging.DEBUG, "External link hook injection skipped: %s", exc)


def launch_edge_app(app: DesktopApp, target_url: str) -> None:
    """Fallback launcher invoking Microsoft Edge `--app` with isolated user-data dir."""
    from src.desktop import app as app_mod

    app.paths.webview2_cache_dir.mkdir(parents=True, exist_ok=True)
    if app._edge_launcher is not None:
        app._edge_launcher(target_url, app.paths.webview2_cache_dir)
        return

    edge_exe = _find_msedge_exe()
    if edge_exe is not None:
        subprocess.Popen(
            [
                str(edge_exe),
                f"--app={target_url}",
                f"--user-data-dir={app.paths.webview2_cache_dir}",
            ]
        )
        return

    app._log(logging.WARNING, "msedge.exe not found; falling back to default web browser for %s", target_url)
    app_mod.webbrowser.open(target_url)


def run_webview_or_fallback(app: DesktopApp, initial_url: str) -> None:
    """Create and run the native WebView2 window, falling back to Edge `--app` on failure."""
    from src.desktop import app as app_mod

    try:
        webview_mod = resolve_webview(app)
        if webview_mod is None:
            raise RuntimeError("pywebview runtime could not be imported.")

        if hasattr(webview_mod, "settings") and isinstance(webview_mod.settings, dict):
            webview_mod.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True

        window = webview_mod.create_window(
            "Second Brain",
            url=initial_url,
            js_api=app.bridge_api,
            width=1360,
            height=880,
            min_size=(960, 640),
        )
        app._window = window

        events = getattr(window, "events", None)
        if events is not None:
            if hasattr(events, "closing"):
                events.closing += app._on_window_closing
            if hasattr(events, "loaded"):
                events.loaded += app._on_window_loaded

        start_kwargs: dict[str, Any] = {
            "storage_path": str(app.paths.webview2_cache_dir),
            "private_mode": False,
        }
        if app_mod.sys.platform == "win32":
            start_kwargs["gui"] = "edgechromium"
        if (ico_path := app.resolve_app_icon_path()) is not None:
            start_kwargs["icon"] = str(ico_path)

        webview_mod.start(**start_kwargs)
    except Exception:
        app._log_exception("WebView2 startup failed; falling back to Microsoft Edge --app mode")
        app._window = None
        app._using_edge_fallback = True
        app._fallback_url = initial_url
        app._launch_edge_app(initial_url)
        app._quit_event.wait()

