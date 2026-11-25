"""Desktop bootstrap, hardware state inspection, and CPU/CUDA compute switch workflows."""
from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from src.desktop.bootstrap import detect_collection_sparse_mode
from src.desktop.models import MODEL_SPECS
from src.desktop.workflow_runners import (
    _fail_with_rollback,
    _run_compute_stage,
    rollback_compute_config,
    start_setup,
    switch_compute_profile,
)

if TYPE_CHECKING:
    from src.desktop.app import DesktopApp


def get_qdrant_collection_name() -> str:
    """Resolve configured Qdrant collection name without failing in minimal test contexts."""
    try:
        from src.config import get_settings

        return get_settings().qdrant_collection
    except Exception:
        return "second_brain_knowledge"


def needs_bootstrap(app: DesktopApp) -> bool:
    """Determine whether first-run runtime/model setup must gate sidecar startup."""
    if not app.config.models_ready:
        return True
    if app.config.compute_profile == "cuda" and (
        not app.config.cuda_runtime_ready or not app.paths.cuda_venv_python.is_file()
    ):
        return True
    status = app.bootstrap.inspect_models_status()
    return not bool(status.get("models_ready"))


def inspect_collection_sparse_mode_sync(app: DesktopApp) -> Literal["bm25", "bge_m3"] | None:
    """Query Qdrant's active collection `sparse_mode` safely if Qdrant is running."""
    if not app.supervisor.is_qdrant_running:
        return None

    collection = get_qdrant_collection_name()
    qdrant_url = app.supervisor.qdrant_url

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
        app._log(logging.DEBUG, "Collection sparse_mode probe skipped: %s", exc)
        return None


def get_initial_state(app: DesktopApp) -> dict[str, Any]:
    """Build complete initial state payload consumed by `setup_ui.html`."""
    from src.desktop import app as app_mod

    data_dir = app.paths.resolve_data_dir(app.config)
    hw_report = app_mod.probe_system_hardware(data_dir=data_dir)
    models_status = app.bootstrap.inspect_models_status()
    collection_sparse_mode = app._inspect_collection_sparse_mode_sync()

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

    app._progress_store.sync_model_states_if_idle(progress_model_map)

    return {
        "mode": app._bridge_mode,
        "active_profile": app.config.compute_profile,
        "cuda_runtime_ready": bool(
            app.config.cuda_runtime_ready and app.paths.cuda_venv_python.is_file()
        ),
        "models_ready": bool(models_status.get("models_ready")),
        "sparse_mode_override": app.config.sparse_mode_override,
        "collection_sparse_mode": collection_sparse_mode,
        "backend_running": bool(app.supervisor.is_backend_running),
        "qdrant_running": bool(app.supervisor.is_qdrant_running),
        "api_url": app.supervisor.api_url,
        "data_dir": str(data_dir),
        "cuda_venv_dir": str(app.paths.cuda_venv_dir),
        "models_dir": str(models_status.get("models_dir", app.paths.models_dir)),
        "total_size_mb": float(models_status.get("total_size_mb", 0.0)),
        "hardware": hw_report.model_dump(),
        "models": serialized_models,
        "progress": app.get_progress(),
    }


def validate_target_profile(
    app: DesktopApp,
    compute_profile: str,
    *,
    force: bool,
) -> tuple[Literal["cpu", "cuda"] | None, str | None]:
    """Validate requested compute profile against hardware VRAM and disk requirements."""
    from src.desktop import app as app_mod

    normalized = (compute_profile or "cpu").strip().lower()
    if normalized not in ("cpu", "cuda"):
        return None, f"Unsupported compute profile '{compute_profile}'. Expected 'cpu' or 'cuda'."

    if normalized == "cuda" and not force:
        hw = app_mod.probe_system_hardware(data_dir=app.paths.resolve_data_dir(app.config))
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


def launch_guarded_worker(
    app: DesktopApp,
    *,
    compute_profile: str,
    force: bool,
    init_progress: dict[str, Any],
    worker_factory: Callable[[Literal["cpu", "cuda"]], Callable[[], None]],
    thread_name: str,
) -> dict[str, Any]:
    """Check active worker lock BEFORE validating or mutating progress state."""
    with app._worker_lock:
        if app._worker_thread is not None and app._worker_thread.is_alive():
            return {"ok": False, "error": "Another setup or compute switch task is already running."}

        target_profile, validation_err = validate_target_profile(app, compute_profile, force=force)
        if target_profile is None:
            app._update_progress(
                step="error",
                detail=validation_err,
                error=validation_err,
                is_running=False,
                completed=False,
            )
            return {"ok": False, "error": validation_err}

        app._update_progress(**init_progress)
        thread = threading.Thread(
            target=worker_factory(target_profile),
            name=thread_name,
            daemon=True,
        )
        app._worker_thread = thread
        thread.start()
    return {"ok": True, "started": True}


__all__ = [
    "_fail_with_rollback",
    "_run_compute_stage",
    "get_initial_state",
    "get_qdrant_collection_name",
    "inspect_collection_sparse_mode_sync",
    "launch_guarded_worker",
    "needs_bootstrap",
    "rollback_compute_config",
    "start_setup",
    "switch_compute_profile",
    "validate_target_profile",
]
