"""First-run C-extension verification, CPU/CUDA 12.4 wheel switcher, and resumable model bootstrap."""
from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from src.desktop.compute_runtime import (
    _BASE_SYNCED_MARKER,
    _CUDA_INDEX_URL,
    apply_sparse_mode_policy,
    build_uv_sync_command,
    detect_collection_sparse_mode,
    ensure_compute_runtime,
)
from src.desktop.config import DesktopConfig, DesktopPaths, load_desktop_config
from src.desktop.env import CREATE_NO_WINDOW as _CREATE_NO_WINDOW
from src.desktop.env import build_model_cache_env
from src.desktop.model_cache import (
    _MANIFEST_FILENAME,
    _RECOGNIZED_MODEL_SUBDIRS,
    _compute_marker_fingerprint,
    _copy_missing_tree,
    _hf_repo_cache_dir,
    _inspect_docling_models,
    _inspect_hf_repo,
    _inspect_subdir_group,
    _scan_dir_metrics,
    import_local_models_dir,
    inspect_models_status,
    inspect_single_model,
    read_manifest_ready,
    write_models_ready_manifest,
)
from src.desktop.model_downloader import download_model_group, ensure_models_ready
from src.desktop.models import (
    MODEL_SPECS,
    _MODEL_SPEC_BY_KEY,
    BootstrapProgress,
    ComputeRuntimeResult,
    ErrorCategory,
    ModelCacheStatus,
    ModelSpec,
    RuntimeCheckResult,
)
from src.desktop.runtime_verify import (
    _RUNTIME_PROBE_SCRIPT,
    _VERIFY_TIMEOUT_SEC,
    _classify_runtime_failure,
    _extract_last_json_object,
    run_subprocess_capture,
    run_subprocess_streaming,
    verify_python_runtime,
)


class RuntimeBootstrapService:
    """Verify native runtimes, manage CPU/CUDA 12.4 venv wheels, and download offline model weights."""

    def __init__(
        self,
        paths: DesktopPaths | None = None,
        config: DesktopConfig | None = None,
        *,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        stream_runner: Callable[[list[str], Path | None, dict[str, str], Callable[[str], None]], int]
        | None = None,
        download_hooks: dict[str, Callable[[Path], Any]] | None = None,
    ) -> None:
        self.paths: DesktopPaths = paths or DesktopPaths()
        self.config: DesktopConfig = config or load_desktop_config(self.paths)
        self._command_runner = command_runner
        self._stream_runner = stream_runner
        self._download_hooks: dict[str, Callable[[Path], Any]] = dict(download_hooks or {})
        self.progress = BootstrapProgress()

    @property
    def models_dir(self) -> Path:
        """Resolved active `models/` directory honoring `DesktopConfig.data_dir_override`."""
        return self.paths.resolve_data_dir(self.config) / "models"

    @property
    def manifest_path(self) -> Path:
        """Path to `<models_dir>/models_ready.json` verification manifest."""
        return self.models_dir / _MANIFEST_FILENAME

    def _emit_progress(
        self,
        progress_cb: Callable[[BootstrapProgress], None] | None,
        *,
        step: str,
        percent: float,
        detail: str,
        error: str | None = None,
        is_running: bool = True,
        completed: bool = False,
        models: dict[str, str] | None = None,
    ) -> BootstrapProgress:
        snapshot = BootstrapProgress(
            step=step,
            percent=round(max(0.0, min(100.0, percent)), 1),
            detail=detail,
            error=error,
            is_running=is_running,
            completed=completed,
            models=dict(models if models is not None else self.progress.models),
        )
        self.progress = snapshot
        if progress_cb is not None:
            progress_cb(snapshot)
        return snapshot

    def _run_subprocess_capture(
        self,
        cmd: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float = _VERIFY_TIMEOUT_SEC,
    ) -> subprocess.CompletedProcess[str]:
        return run_subprocess_capture(
            cmd,
            cwd=cwd,
            env=env,
            timeout=timeout,
            command_runner=self._command_runner,
        )

    def _run_subprocess_streaming(
        self,
        cmd: list[str],
        *,
        cwd: Path | None,
        env: dict[str, str],
        on_line: Callable[[str], None],
    ) -> int:
        return run_subprocess_streaming(
            cmd,
            cwd=cwd,
            env=env,
            on_line=on_line,
            stream_runner=self._stream_runner,
        )

    def verify_python_runtime(self, python_exe: Path | None = None) -> RuntimeCheckResult:
        """Verify that `torch`, `onnxruntime`, `pymupdf`, `pandas`, `cv2`, and `laya` load cleanly."""
        return verify_python_runtime(
            self.paths,
            self.config,
            python_exe,
            capture_fn=self._run_subprocess_capture,
        )

    def _build_uv_sync_command(self, uv_exe: str) -> list[str]:
        """Build `uv sync` command preserving `--extra desktop` when defined or unpinned in tests."""
        return build_uv_sync_command(self.paths.backend_dir, uv_exe)

    @staticmethod
    def _apply_sparse_mode_policy(
        config: DesktopConfig,
        *,
        target_profile: Literal["cpu", "cuda"],
        existing_sparse_mode: str | None,
        reindex_sparse: bool,
    ) -> bool:
        """Apply corpus-preserving sparse_mode pinning and return `reindex_recommended`."""
        return apply_sparse_mode_policy(
            config,
            target_profile=target_profile,
            existing_sparse_mode=existing_sparse_mode,
            reindex_sparse=reindex_sparse,
        )

    def ensure_compute_runtime(
        self,
        target_profile: Literal["cpu", "cuda"],
        *,
        is_backend_running: Callable[[], bool] | None = None,
        existing_sparse_mode: str | None = None,
        reindex_sparse: bool = False,
        purge_cuda_venv: bool = False,
        verify_runtime: bool = False,
        progress_cb: Callable[[BootstrapProgress], None] | None = None,
    ) -> ComputeRuntimeResult:
        """Switch compute runtime between CPU and CUDA 12.4 in isolated user runtime virtualenv."""
        return ensure_compute_runtime(
            self.paths,
            self.config,
            target_profile,
            is_backend_running=is_backend_running,
            existing_sparse_mode=existing_sparse_mode,
            reindex_sparse=reindex_sparse,
            purge_cuda_venv=purge_cuda_venv,
            verify_runtime=verify_runtime,
            progress_cb=progress_cb,
            emit_progress_fn=self._emit_progress,
            stream_fn=self._run_subprocess_streaming,
            verify_runtime_fn=self.verify_python_runtime,
            uv_sync_cmd_fn=self._build_uv_sync_command,
        )

    def _inspect_single_model(self, spec: ModelSpec) -> tuple[ModelCacheStatus, list[tuple[str, int]]]:
        return inspect_single_model(self.models_dir, spec)

    def _read_manifest_ready(self) -> tuple[bool, dict[str, Any] | None]:
        return read_manifest_ready(self.manifest_path)

    def inspect_models_status(self) -> dict[str, Any]:
        """Inspect `models_ready.json` and on-disk markers for all 6 model groups."""
        return inspect_models_status(
            self.models_dir,
            self.manifest_path,
            inspect_single_fn=self._inspect_single_model,
            read_manifest_fn=self._read_manifest_ready,
        )

    def _inject_model_cache_env(self) -> None:
        """Configure AI model cache environment variables before importing or invoking downloaders."""
        os.environ.update(build_model_cache_env(self.models_dir, for_download=True))

    def _download_model_group(self, spec: ModelSpec) -> None:
        """Download and verify a single model group into its target cache subdirectory."""
        download_model_group(
            self.models_dir,
            spec,
            download_hooks=self._download_hooks,
            inspect_single_fn=self._inspect_single_model,
        )

    def _write_models_ready_manifest(self) -> dict[str, Any]:
        return write_models_ready_manifest(
            self.models_dir,
            self.manifest_path,
            self.config.last_installed_version,
            inspect_single_fn=self._inspect_single_model,
        )

    def ensure_models_ready(
        self,
        force: bool = False,
        progress_cb: Callable[[BootstrapProgress], None] | None = None,
    ) -> dict[str, Any]:
        """Download and verify all 6 required model groups with warm fast-path and clean resume."""
        return ensure_models_ready(
            self.paths,
            self.config,
            self.models_dir,
            self.manifest_path,
            force=force,
            progress_cb=progress_cb,
            emit_progress_fn=self._emit_progress,
            download_hooks=self._download_hooks,
            inject_env_fn=self._inject_model_cache_env,
            inspect_status_fn=self.inspect_models_status,
            inspect_single_fn=self._inspect_single_model,
            download_group_fn=self._download_model_group,
            write_manifest_fn=self._write_models_ready_manifest,
        )

    def import_local_models_dir(self, source_dir: Path | str) -> dict[str, Any]:
        """Import pre-cached model directories from a local offline bundle into `models_dir`."""
        return import_local_models_dir(
            self.paths,
            self.config,
            self.models_dir,
            self.manifest_path,
            source_dir,
            inspect_single_fn=self._inspect_single_model,
            write_manifest_fn=self._write_models_ready_manifest,
            inspect_status_fn=self.inspect_models_status,
        )


__all__ = [
    "BootstrapProgress",
    "ComputeRuntimeResult",
    "ErrorCategory",
    "MODEL_SPECS",
    "ModelCacheStatus",
    "ModelSpec",
    "RuntimeBootstrapService",
    "RuntimeCheckResult",
    "_BASE_SYNCED_MARKER",
    "_CREATE_NO_WINDOW",
    "_CUDA_INDEX_URL",
    "_MANIFEST_FILENAME",
    "_MODEL_SPEC_BY_KEY",
    "_RECOGNIZED_MODEL_SUBDIRS",
    "_RUNTIME_PROBE_SCRIPT",
    "_VERIFY_TIMEOUT_SEC",
    "_classify_runtime_failure",
    "_compute_marker_fingerprint",
    "_copy_missing_tree",
    "_extract_last_json_object",
    "_hf_repo_cache_dir",
    "_inspect_docling_models",
    "_inspect_hf_repo",
    "_inspect_subdir_group",
    "_scan_dir_metrics",
    "apply_sparse_mode_policy",
    "detect_collection_sparse_mode",
]
