"""CPU / NVIDIA CUDA 12.4 compute runtime switching and sparse-mode corpus policy."""
from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from src.desktop.config import DesktopConfig, DesktopPaths, save_desktop_config
from src.desktop.models import BootstrapProgress, ComputeRuntimeResult, RuntimeCheckResult

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient

logger = logging.getLogger(__name__)

_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu124"
_BASE_SYNCED_MARKER = ".base_synced"


async def detect_collection_sparse_mode(
    qdrant: AsyncQdrantClient | None,
    collection: str,
) -> Literal["bm25", "bge_m3"] | None:
    """Inspect an existing Qdrant collection's `sparse_mode` payload safely without throwing."""
    if qdrant is None or not collection:
        return None
    try:
        if not await qdrant.collection_exists(collection):
            return None
        points, _ = await qdrant.scroll(
            collection_name=collection,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None
        payload = getattr(points[0], "payload", None) or {}
        mode = str(payload.get("sparse_mode") or "").strip().lower()
        if mode in ("bm25", "bge_m3"):
            return mode  # type: ignore[return-value]
    except Exception as exc:
        logger.debug("Failed to detect collection sparse_mode for '%s': %s", collection, exc)
    return None


def apply_sparse_mode_policy(
    config: DesktopConfig,
    *,
    target_profile: Literal["cpu", "cuda"],
    existing_sparse_mode: str | None,
    reindex_sparse: bool,
) -> bool:
    """Apply corpus-preserving sparse_mode pinning and return `reindex_recommended`."""
    normalized_sparse = (existing_sparse_mode or "").strip().lower()
    if normalized_sparse in ("bm25", "bge_m3"):
        if reindex_sparse:
            config.sparse_mode_override = "auto"
            return False
        config.sparse_mode_override = normalized_sparse  # type: ignore[assignment]
        return target_profile == "cuda" and normalized_sparse == "bm25"

    if reindex_sparse:
        config.sparse_mode_override = "auto"
    return False


def build_uv_sync_command(backend_dir: Path, uv_exe: str) -> list[str]:
    """Build `uv sync` command preserving `--extra desktop` when defined or unpinned in tests."""
    cmd = [uv_exe, "sync", "--native-tls", "--frozen", "--no-dev"]
    pyproject = backend_dir / "pyproject.toml"
    include_desktop_extra = True
    if pyproject.is_file():
        try:
            text = pyproject.read_text(encoding="utf-8", errors="ignore")
            if "[project.optional-dependencies]" not in text or "desktop" not in text:
                include_desktop_extra = False
        except OSError:
            pass
    if include_desktop_extra:
        cmd.extend(["--extra", "desktop"])
    cmd.append("--no-install-project")
    return cmd


def ensure_compute_runtime(
    paths: DesktopPaths,
    config: DesktopConfig,
    target_profile: Literal["cpu", "cuda"],
    *,
    is_backend_running: Callable[[], bool] | None = None,
    existing_sparse_mode: str | None = None,
    reindex_sparse: bool = False,
    purge_cuda_venv: bool = False,
    verify_runtime: bool = False,
    progress_cb: Callable[[BootstrapProgress], None] | None = None,
    emit_progress_fn: Callable[..., BootstrapProgress],
    stream_fn: Callable[..., int],
    verify_runtime_fn: Callable[[Path], RuntimeCheckResult],
    uv_sync_cmd_fn: Callable[[str], list[str]] | None = None,
) -> ComputeRuntimeResult:
    """Switch compute runtime between CPU and CUDA 12.4 in isolated user runtime virtualenv."""
    if is_backend_running is not None and is_backend_running():
        raise RuntimeError(
            "Backend sidecar must be stopped before modifying Python compute runtime wheels "
            "(prevents WinError 32 file locks)."
        )

    paths.ensure_directories(config)
    commands_run: list[list[str]] = []

    if target_profile == "cpu":
        emit_progress_fn(
            progress_cb,
            step="switch_cpu",
            percent=35.0,
            detail="Switching active compute profile to Standard CPU mode...",
            is_running=True,
        )
        config.compute_profile = "cpu"
        if purge_cuda_venv and paths.cuda_venv_dir.exists():
            emit_progress_fn(
                progress_cb,
                step="purge_cuda_venv",
                percent=70.0,
                detail=f"Removing cached CUDA virtualenv at {paths.cuda_venv_dir}...",
                is_running=True,
            )
            shutil.rmtree(paths.cuda_venv_dir)
            config.cuda_runtime_ready = False

        reindex_recommended = apply_sparse_mode_policy(
            config,
            target_profile="cpu",
            existing_sparse_mode=existing_sparse_mode,
            reindex_sparse=reindex_sparse,
        )
        save_desktop_config(config, paths)

        active_py = str(paths.resolve_backend_python("cpu", cuda_runtime_ready=config.cuda_runtime_ready))
        detail = "Switched compute profile to Standard CPU mode."
        emit_progress_fn(
            progress_cb,
            step="complete",
            percent=100.0,
            detail=detail,
            is_running=False,
            completed=True,
        )
        return ComputeRuntimeResult(
            ok=True,
            compute_profile="cpu",
            python_exe=active_py,
            cuda_runtime_ready=config.cuda_runtime_ready,
            sparse_mode_override=config.sparse_mode_override,
            reindex_recommended=reindex_recommended,
            detail=detail,
            commands_run=commands_run,
        )

    uv_exe = str(paths.uv_exe)
    cuda_venv_dir = paths.cuda_venv_dir
    cuda_venv_py = paths.cuda_venv_python
    base_python = paths.base_venv_python
    sync_marker = cuda_venv_dir / _BASE_SYNCED_MARKER
    previous_profile = config.compute_profile

    def _fail(step_name: str, percent: float, err_msg: str) -> ComputeRuntimeResult:
        emit_progress_fn(
            progress_cb,
            step=step_name,
            percent=percent,
            detail=err_msg,
            error=err_msg,
            is_running=False,
            completed=False,
        )
        return ComputeRuntimeResult(
            ok=False,
            compute_profile=previous_profile,
            python_exe=str(cuda_venv_py),
            cuda_runtime_ready=False,
            sparse_mode_override=config.sparse_mode_override,
            reindex_recommended=False,
            detail=err_msg,
            commands_run=commands_run,
        )

    base_env = os.environ.copy()
    base_env["UV_NATIVE_TLS"] = "true"
    sync_env = dict(base_env)
    sync_env["UV_PROJECT_ENVIRONMENT"] = str(cuda_venv_dir)
    sync_cwd = paths.backend_dir if paths.backend_dir.is_dir() else None
    sync_cmd = (
        uv_sync_cmd_fn(uv_exe)
        if uv_sync_cmd_fn is not None
        else build_uv_sync_command(paths.backend_dir, uv_exe)
    )

    steps: list[tuple[str, float, float, list[str], Path | None, dict[str, str], str]] = []
    if not cuda_venv_py.is_file():
        steps.append(
            (
                "create_cuda_venv",
                5.0,
                20.0,
                [uv_exe, "venv", "--relocatable", str(cuda_venv_dir), "--python", str(base_python)],
                sync_cwd,
                base_env,
                f"Creating isolated CUDA runtime virtualenv at {cuda_venv_dir}...",
            )
        )
    if not sync_marker.is_file():
        steps.append(
            (
                "sync_cuda_venv",
                20.0,
                45.0,
                sync_cmd,
                sync_cwd,
                sync_env,
                "Installing base backend dependencies into CUDA runtime virtualenv...",
            )
        )

    steps.extend(
        [
            (
                "install_torch_cu124",
                45.0,
                80.0,
                [
                    uv_exe,
                    "pip",
                    "install",
                    "--native-tls",
                    "--python",
                    str(cuda_venv_py),
                    "--reinstall-package",
                    "torch",
                    "--reinstall-package",
                    "torchvision",
                    "--index-url",
                    _CUDA_INDEX_URL,
                    "torch>=2.4.0",
                    "torchvision>=0.19.0",
                ],
                sync_cwd,
                base_env,
                "Downloading PyTorch + torchvision CUDA 12.4 wheels...",
            ),
            (
                "uninstall_onnxruntime",
                80.0,
                88.0,
                [
                    uv_exe,
                    "pip",
                    "uninstall",
                    "--python",
                    str(cuda_venv_py),
                    "onnxruntime",
                ],
                sync_cwd,
                base_env,
                "Removing CPU-only onnxruntime package prior to GPU wheel install...",
            ),
            (
                "install_onnxruntime_gpu",
                88.0,
                98.0,
                [
                    uv_exe,
                    "pip",
                    "install",
                    "--native-tls",
                    "--python",
                    str(cuda_venv_py),
                    "onnxruntime-gpu==1.20.0",
                ],
                sync_cwd,
                base_env,
                "Installing onnxruntime-gpu==1.20.0 for accelerated Docling & OCR...",
            ),
        ]
    )

    for step_name, start_pct, end_pct, cmd, step_cwd, step_env, banner in steps:
        emit_progress_fn(
            progress_cb,
            step=step_name,
            percent=start_pct,
            detail=banner,
            is_running=True,
        )
        commands_run.append(cmd)
        captured_lines: list[str] = []

        def _handle_line(line: str, _step: str = step_name, _pct: float = start_pct, _end: float = end_pct) -> None:
            captured_lines.append(line)
            if len(captured_lines) > 80:
                del captured_lines[:-80]
            emit_progress_fn(
                progress_cb,
                step=_step,
                percent=min(_end - 1.0, _pct + min(8.0, len(captured_lines) * 0.5)),
                detail=line,
                is_running=True,
            )

        try:
            rc = stream_fn(
                cmd,
                cwd=step_cwd,
                env=step_env,
                on_line=_handle_line,
            )
        except OSError as exc:
            return _fail(step_name, start_pct, f"Failed to execute `{cmd[0]}` during {step_name}: {exc}")

        if rc != 0:
            tail = "\n".join(captured_lines[-10:])
            if step_name == "uninstall_onnxruntime" and (
                "not installed" in tail.lower() or "skipping" in tail.lower()
            ):
                continue
            return _fail(
                step_name,
                start_pct,
                f"Command failed (exit code {rc}) at step '{step_name}' "
                f"(index: {_CUDA_INDEX_URL if 'cu124' in ' '.join(cmd) else 'PyPI'}): "
                f"{tail or 'see launcher.log for details'}",
            )

        if step_name == "sync_cuda_venv":
            cuda_venv_dir.mkdir(parents=True, exist_ok=True)
            sync_marker.write_text("ok\n", encoding="utf-8")

    if verify_runtime:
        rt_check = verify_runtime_fn(cuda_venv_py)
        if not rt_check.ok:
            config.cuda_runtime_ready = False
            config.compute_profile = previous_profile
            save_desktop_config(config, paths)
            return _fail(
                "verify_cuda_runtime",
                98.0,
                rt_check.error_detail or "CUDA Python runtime verification failed.",
            )

    config.cuda_runtime_ready = True
    config.compute_profile = "cuda"
    reindex_recommended = apply_sparse_mode_policy(
        config,
        target_profile="cuda",
        existing_sparse_mode=existing_sparse_mode,
        reindex_sparse=reindex_sparse,
    )
    save_desktop_config(config, paths)

    detail = "NVIDIA CUDA 12.4 runtime environment is ready."
    emit_progress_fn(
        progress_cb,
        step="complete",
        percent=100.0,
        detail=detail,
        is_running=False,
        completed=True,
    )
    return ComputeRuntimeResult(
        ok=True,
        compute_profile="cuda",
        python_exe=str(cuda_venv_py),
        cuda_runtime_ready=True,
        sparse_mode_override=config.sparse_mode_override,
        reindex_recommended=reindex_recommended,
        detail=detail,
        commands_run=commands_run,
    )
