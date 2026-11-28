"""Background worker runners for desktop first-run setup and CPU/CUDA compute switching."""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from src.desktop.config import save_desktop_config
from src.desktop.models import MODEL_SPECS, BootstrapProgress

if TYPE_CHECKING:
    from src.desktop.app import DesktopApp


def rollback_compute_config(
    app: DesktopApp,
    prev_profile: Literal["cpu", "cuda"],
    prev_cuda_ready: bool,
    prev_sparse_override: Literal["auto", "bge_m3", "bm25"] | None,
) -> None:
    """Restore prior compute profile state on failed wheel installation or runtime check."""
    app.config.compute_profile = prev_profile
    app.config.cuda_runtime_ready = prev_cuda_ready
    app.config.sparse_mode_override = prev_sparse_override
    app.bootstrap.config = app.config
    app.supervisor.config = app.config
    try:
        save_desktop_config(app.config, app.paths)
    except Exception as exc:
        app._log(logging.WARNING, "Failed to persist compute config rollback: %s", exc)


def _fail_with_rollback(
    app: DesktopApp,
    *,
    prev_state: tuple[Literal["cpu", "cuda"], bool, Literal["auto", "bge_m3", "bm25"] | None],
    step: str,
    percent: float,
    message: str,
    log_label: str,
    remediation: str | None = None,
    error_category: str | None = None,
    steps_update: dict[str, str] | None = None,
) -> None:
    rollback_compute_config(app, *prev_state)
    app._log(logging.ERROR, "%s: %s", log_label, message)
    kwargs: dict[str, Any] = {
        "step": step,
        "percent": percent,
        "detail": message,
        "error": message,
        "is_running": False,
        "completed": False,
        "steps_update": steps_update,
    }
    if remediation is not None:
        kwargs["remediation"] = remediation
    if error_category is not None:
        kwargs["error_category"] = error_category
    app._update_progress(**kwargs)


def _run_compute_stage(
    app: DesktopApp,
    *,
    target_profile: Literal["cpu", "cuda"],
    prev_state: tuple[Literal["cpu", "cuda"], bool, Literal["auto", "bge_m3", "bm25"] | None],
    existing_sparse_mode: str | None,
    reindex_sparse: bool,
    progress_base: float,
    progress_span: float,
    fail_percent: float,
    fail_log_label: str,
):
    def _on_compute_progress(prog: BootstrapProgress) -> None:
        scaled = progress_base + (prog.percent / 100.0) * progress_span
        app._update_progress(
            step=prog.step,
            percent=scaled,
            detail=prog.detail,
            is_running=True,
            completed=False,
        )

    compute_res = app.bootstrap.ensure_compute_runtime(
        target_profile,
        is_backend_running=lambda: app.supervisor.is_backend_running,
        existing_sparse_mode=existing_sparse_mode,
        reindex_sparse=reindex_sparse,
        progress_cb=_on_compute_progress,
    )
    if not compute_res.ok:
        _fail_with_rollback(
            app,
            prev_state=prev_state,
            step="compute_runtime",
            percent=fail_percent,
            message=compute_res.detail,
            log_label=fail_log_label,
            steps_update={"compute_runtime": "failed"},
        )
        return None
    return compute_res


def start_setup(
    app: DesktopApp,
    compute_profile: str = "cpu",
    reindex_sparse: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Start first-run verification, compute runtime setup, model downloads, and sidecars."""
    from src.desktop.workflows import launch_guarded_worker

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
            prev_state = (
                app.config.compute_profile,
                app.config.cuda_runtime_ready,
                app.config.sparse_mode_override,
            )
            try:
                app._log(logging.INFO, "Starting desktop bootstrap (profile=%s)...", target_profile)
                rt_result = app.bootstrap.verify_python_runtime()
                if not rt_result.ok:
                    msg = rt_result.error_detail or "Python runtime self-test failed."
                    app._log(logging.ERROR, "Runtime check failed (%s): %s", rt_result.error_category, msg)
                    app._update_progress(
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

                app._update_progress(
                    step="compute_runtime",
                    percent=10.0,
                    detail=f"Preparing compute runtime ({target_profile.upper()})...",
                    steps_update={"runtime_check": "ok", "compute_runtime": "running"},
                )

                compute_res = _run_compute_stage(
                    app,
                    target_profile=target_profile,
                    prev_state=prev_state,
                    existing_sparse_mode=None,
                    reindex_sparse=reindex_sparse,
                    progress_base=10.0,
                    progress_span=25.0,
                    fail_percent=20.0,
                    fail_log_label="Compute runtime setup failed",
                )
                if compute_res is None:
                    return

                if target_profile == "cuda":
                    cuda_rt = app.bootstrap.verify_python_runtime()
                    if not cuda_rt.ok:
                        _fail_with_rollback(
                            app,
                            prev_state=prev_state,
                            step="compute_runtime",
                            percent=34.0,
                            message=cuda_rt.error_detail or "CUDA Python runtime verification failed.",
                            log_label="CUDA runtime verification failed",
                            remediation=cuda_rt.remediation,
                            error_category=cuda_rt.error_category,
                            steps_update={"compute_runtime": "failed"},
                        )
                        return

                app._update_progress(
                    step="models",
                    percent=35.0,
                    detail="Checking offline AI model groups...",
                    steps_update={"compute_runtime": "ok"},
                )

                def _on_models_progress(prog: BootstrapProgress) -> None:
                    scaled = 35.0 + (prog.percent / 100.0) * 57.0
                    app._update_progress(
                        step=prog.step,
                        percent=scaled,
                        detail=prog.detail,
                        models=prog.models if prog.models else None,
                        is_running=True,
                        completed=False,
                    )

                app.bootstrap.ensure_models_ready(progress_cb=_on_models_progress)

                app.config = app.bootstrap.config
                app.supervisor.config = app.config

                app._update_progress(
                    step="start_sidecars",
                    percent=94.0,
                    detail="Starting Qdrant & Second Brain API sidecars...",
                    models={spec.key: "cached" for spec in MODEL_SPECS},
                    is_running=True,
                    completed=False,
                )

                app.supervisor.start()
                api_url = app.supervisor.api_url

                detail_msg = "Setup complete! Opening Second Brain workspace..."
                if reindex_sparse:
                    queued, reindex_err = app._trigger_corpus_reindex(api_url)
                    if reindex_err is not None:
                        detail_msg = (
                            "Setup complete (warning: background sparse re-index trigger failed; "
                            f"retry from workspace): {reindex_err}"
                        )
                    elif queued > 0:
                        detail_msg = f"Setup complete! Re-indexing {queued} existing document(s) in background..."

                app._update_progress(
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
                app._log(logging.INFO, "Bootstrap complete; navigating to %s", api_url)
                app._navigate_window(api_url)
            except Exception as exc:
                rollback_compute_config(app, *prev_state)
                app._log_exception("Desktop bootstrap failed")
                app._update_progress(
                    step="error",
                    detail=str(exc),
                    error=str(exc),
                    is_running=False,
                    completed=False,
                )

        return _worker

    return launch_guarded_worker(
        app,
        compute_profile=compute_profile,
        force=force,
        init_progress=init_progress,
        worker_factory=_build_worker,
        thread_name="desktop-bootstrap-setup",
    )


def switch_compute_profile(
    app: DesktopApp,
    compute_profile: str,
    reindex_sparse: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Safely swap CPU/CUDA compute runtime with `stop_backend()` before wheel modification."""
    from src.desktop.workflows import launch_guarded_worker

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
            prev_state = (
                app.config.compute_profile,
                app.config.cuda_runtime_ready,
                app.config.sparse_mode_override,
            )
            try:
                app._log(
                    logging.INFO,
                    "Switching compute profile to %s (reindex_sparse=%s)...",
                    target_profile,
                    reindex_sparse,
                )
                existing_sparse = app._inspect_collection_sparse_mode_sync()

                # Stop uvicorn BEFORE modifying runtime wheels to prevent WinError 32 file locks.
                app.supervisor.stop_backend(timeout_sec=10.0)

                compute_res = _run_compute_stage(
                    app,
                    target_profile=target_profile,
                    prev_state=prev_state,
                    existing_sparse_mode=existing_sparse,
                    reindex_sparse=reindex_sparse,
                    progress_base=15.0,
                    progress_span=60.0,
                    fail_percent=40.0,
                    fail_log_label="Compute profile switch failed",
                )
                if compute_res is None:
                    return

                app._update_progress(
                    step="runtime_check",
                    percent=80.0,
                    detail="Verifying target compute Python runtime...",
                    steps_update={"compute_runtime": "ok", "runtime_check": "running"},
                )

                rt_result = app.bootstrap.verify_python_runtime()
                if not rt_result.ok:
                    _fail_with_rollback(
                        app,
                        prev_state=prev_state,
                        step="runtime_check",
                        percent=82.0,
                        message=rt_result.error_detail or "Runtime verification failed after compute switch.",
                        log_label="Post-switch runtime check failed",
                        remediation=rt_result.remediation,
                        error_category=rt_result.error_category,
                        steps_update={"runtime_check": "failed"},
                    )
                    return

                app.config = app.bootstrap.config
                app.supervisor.config = app.config

                app._update_progress(
                    step="restart_backend",
                    percent=90.0,
                    detail="Restarting Second Brain API backend on new compute profile...",
                    steps_update={"runtime_check": "ok"},
                )

                if not app.supervisor.is_qdrant_running:
                    app.supervisor.start_qdrant()
                app.supervisor.restart_backend(app.config)
                api_url = app.supervisor.api_url

                detail_msg = f"Switched to {target_profile.upper()} mode. Returning to workspace..."
                if reindex_sparse:
                    queued, reindex_err = app._trigger_corpus_reindex(api_url)
                    if reindex_err is not None:
                        detail_msg = (
                            f"Switched to {target_profile.upper()} mode, "
                            f"but background re-index dispatch failed: {reindex_err}"
                        )
                        app.tray.notify(detail_msg, title="Second Brain")
                    elif queued > 0:
                        detail_msg = (
                            f"Switched to {target_profile.upper()} mode. "
                            f"Re-indexing {queued} document(s) in background..."
                        )
                        app.tray.notify(
                            f"Switched to {target_profile.upper()} mode. Re-indexing {queued} document(s) for learned sparse vectors.",
                            title="Second Brain",
                        )
                    else:
                        detail_msg = (
                            f"Switched to {target_profile.upper()} mode (no ready documents needed re-indexing)."
                        )
                elif compute_res.reindex_recommended:
                    app.tray.notify(
                        "Switched to CUDA mode. Existing collection remains pinned to BM25 until re-indexed.",
                        title="Second Brain",
                    )

                app._bridge_mode = "setup"
                app._update_progress(
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

                app._navigate_window(api_url)
            except Exception as exc:
                rollback_compute_config(app, *prev_state)
                app._log_exception("Compute profile switch failed")
                app._update_progress(
                    step="error",
                    detail=str(exc),
                    error=str(exc),
                    is_running=False,
                    completed=False,
                )

        return _worker

    return launch_guarded_worker(
        app,
        compute_profile=compute_profile,
        force=force,
        init_progress=init_progress,
        worker_factory=_build_worker,
        thread_name="desktop-compute-switch",
    )
