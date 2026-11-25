"""Thread-safe progress state store and pywebview JS bridge API."""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from src.desktop.models import MODEL_SPECS

if TYPE_CHECKING:
    from src.desktop.app import DesktopApp


class ProgressStore:
    """Thread-safe state store backing `setup_ui.html` progress polling."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = self.default_state()

    @staticmethod
    def default_state() -> dict[str, Any]:
        """Return a fresh idle progress dictionary covering all 6 model groups."""
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

    def update(
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
        """Atomically update selected progress fields and append deduplicated log lines."""
        with self._lock:
            st = self._state
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

    def sync_model_states_if_idle(self, model_map: dict[str, str]) -> None:
        """Update cached/pending model states when no setup/switch worker is running."""
        with self._lock:
            if not self._state["is_running"]:
                self._state["models"].update(model_map)

    def reset_for_switch_if_idle(self) -> None:
        """Reset progress to switch prompt state when no worker is currently active."""
        with self._lock:
            if not self._state["is_running"]:
                fresh = self.default_state()
                fresh["detail"] = "Inspect hardware or switch between CPU and CUDA 12.4 mode."
                self._state = fresh

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of current bootstrap/switch progress."""
        with self._lock:
            snap = dict(self._state)
            snap["models"] = dict(self._state["models"])
            snap["steps"] = dict(self._state["steps"])
            snap["logs"] = list(self._state["logs"])
            return snap


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
