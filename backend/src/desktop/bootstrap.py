"""First-run C-extension verification, CPU/CUDA 12.4 wheel switcher, and resumable model bootstrap."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

from pydantic import BaseModel, Field

from src.desktop.config import (
    DesktopConfig,
    DesktopPaths,
    load_desktop_config,
    save_desktop_config,
)

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient

logger = logging.getLogger(__name__)

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_VERIFY_TIMEOUT_SEC = 30.0
_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu124"
_MANIFEST_FILENAME = "models_ready.json"
_BASE_SYNCED_MARKER = ".base_synced"
_RECOGNIZED_MODEL_SUBDIRS: tuple[str, ...] = (
    "huggingface",
    "fastembed",
    "flashrank",
    "rapidocr",
    "docling",
    "torch",
)

ErrorCategory = Literal[
    "ok",
    "app_control_blocked",
    "missing_vc_redist",
    "missing_package",
    "unknown",
]


class ModelSpec(BaseModel):
    """Canonical specification for a required offline AI model group."""

    key: str
    display_name: str
    repo_or_id: str
    cache_subdir: str
    approx_size_mb: int
    description: str


class ModelCacheStatus(BaseModel):
    """Cached status and footprint of a single model group inside `models_dir`."""

    key: str
    display_name: str
    cached: bool
    path: str
    size_mb: float = 0.0
    detail: str = ""


class RuntimeCheckResult(BaseModel):
    """Structured outcome of C-extension and GPU/ONNX runtime verification."""

    ok: bool
    python_exe: str
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    cuda_available: bool = False
    onnx_providers: list[str] = Field(default_factory=list)
    error_category: ErrorCategory = "ok"
    error_detail: str | None = None
    remediation: str | None = None


class BootstrapProgress(BaseModel):
    """Real-time progress payload emitted during wheel installation and model download."""

    step: str = "idle"
    percent: float = 0.0
    detail: str = ""
    error: str | None = None
    is_running: bool = False
    completed: bool = False
    models: dict[str, str] = Field(default_factory=dict)


class ComputeRuntimeResult(BaseModel):
    """Outcome of switching between CPU and NVIDIA CUDA 12.4 sidecar runtimes."""

    ok: bool
    compute_profile: Literal["cpu", "cuda"]
    python_exe: str
    cuda_runtime_ready: bool = False
    sparse_mode_override: Literal["auto", "bge_m3", "bm25"] | None = None
    reindex_recommended: bool = False
    detail: str = ""
    commands_run: list[list[str]] = Field(default_factory=list)


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="bge_m3",
        display_name="BAAI/bge-m3 Embeddings & Sparse",
        repo_or_id="BAAI/bge-m3",
        cache_subdir="huggingface/hub",
        approx_size_mb=2270,
        description="Dense + learned sparse embeddings and tokenizer for offline chunking and hybrid RAG.",
    ),
    ModelSpec(
        key="laya_multilingual",
        display_name="Laya Multilingual Classifier",
        repo_or_id="convaiinnovations/laya-multilingual",
        cache_subdir="huggingface/hub",
        approx_size_mb=550,
        description="Multilingual domain, subdomain, epistemic modality, and authority classifier.",
    ),
    ModelSpec(
        key="fastembed_bm25",
        display_name="FastEmbed Qdrant/bm25",
        repo_or_id="Qdrant/bm25",
        cache_subdir="fastembed",
        approx_size_mb=25,
        description="CPU-friendly BM25 sparse lexical tokenizer and IDF tables.",
    ),
    ModelSpec(
        key="flashrank",
        display_name="FlashRank MultiBERT Reranker",
        repo_or_id="ms-marco-MultiBERT-L-12",
        cache_subdir="flashrank",
        approx_size_mb=520,
        description="Multilingual ONNX cross-encoder reranker for evidence scoring.",
    ),
    ModelSpec(
        key="rapidocr",
        display_name="RapidOCR v3 ONNX Engine",
        repo_or_id="RapidOCR v3 ONNX",
        cache_subdir="rapidocr",
        approx_size_mb=20,
        description="CPU ONNX OCR models for exploded-SHX CAD engineering drawings.",
    ),
    ModelSpec(
        key="docling_layout_tableformer",
        display_name="Docling v2 Layout & TableFormer",
        repo_or_id="docling-project/docling-models",
        cache_subdir="docling/models",
        approx_size_mb=450,
        description="Docling v2 layout object detection, TableFormer table structure, and OCR artifacts.",
    ),
)

_MODEL_SPEC_BY_KEY: dict[str, ModelSpec] = {spec.key: spec for spec in MODEL_SPECS}

_RUNTIME_PROBE_SCRIPT = (
    "import importlib, json\n"
    "import torch\n"
    "import onnxruntime as ort\n"
    "for mod in ('pymupdf', 'fitz', 'pandas', 'cv2', 'laya'):\n"
    "    importlib.import_module(mod)\n"
    "print(json.dumps({\n"
    "    'torch_version': getattr(torch, '__version__', None),\n"
    "    'torch_cuda_version': getattr(getattr(torch, 'version', None), 'cuda', None),\n"
    "    'cuda_available': bool(torch.cuda.is_available()),\n"
    "    'onnx_providers': list(ort.get_available_providers()),\n"
    "}))\n"
)


def _classify_runtime_failure(
    returncode: int | None,
    combined_output: str,
) -> tuple[ErrorCategory, str]:
    """Classify Python C-extension startup/import failures into actionable categories."""
    lower = combined_output.lower()

    # Windows Application Control / Smart App Control exit codes (0xC0000428, 0xC000007B)
    app_control_codes = {-1073740760, 3221226536, 0xC0000428}
    policy_c000007b_codes = {-1073741701, 3221225595, 0xC000007B}
    app_control_markers = (
        "application control policy",
        "application control",
        "smart app control",
        "winerror 4551",
        "blocked by",
        "digitally signed",
        "0xc0000428",
    )
    is_c000007b_policy = (
        returncode in policy_c000007b_codes or "0xc000007b" in lower
    ) and any(tok in lower for tok in ("policy", "control", "blocked", "signed", "signature"))

    if (
        returncode in app_control_codes
        or any(marker in lower for marker in app_control_markers)
        or is_c000007b_policy
    ):
        return (
            "app_control_blocked",
            "Windows Application Control / Smart App Control blocked native Python C-extensions (.dll/.pyd). "
            "Allowlist the Second Brain installation and runtime folders via Windows Security / WDAC policy, "
            "or install to a trusted administrator-approved path.",
        )

    vc_markers = (
        "vcruntime140",
        "msvcp140",
        "concrt140",
        "vcomp140",
        "c runtime",
        "redistributable",
        "winerror 126",
    )
    if any(marker in lower for marker in vc_markers) or (
        "dll load failed" in lower and "specified module could not be found" in lower
    ):
        return (
            "missing_vc_redist",
            "Missing Microsoft Visual C++ 2015–2022 Redistributable (x64). "
            "Install `vc_redist.x64.exe` from Microsoft or run the Second Brain installer again to restore C runtime DLLs.",
        )

    if "modulenotfounderror" in lower or "no module named" in lower:
        return (
            "missing_package",
            "Required Python dependency is missing from the runtime virtualenv. "
            "Re-run setup or repair the Second Brain installation to restore dependencies.",
        )

    return (
        "unknown",
        "Python runtime self-test failed unexpectedly. Check `logs/launcher.log` for full diagnostic details.",
    )


def _extract_last_json_object(stdout: str) -> dict[str, Any] | None:
    """Parse the final JSON object emitted on stdout even if libraries printed warnings earlier."""
    lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    return parsed
            except ValueError:
                continue
    return None


def _scan_dir_metrics(root: Path, *, max_depth: int = 5) -> tuple[int, int, bool, list[tuple[str, int]]]:
    """Return `(regular_bytes, marker_count, has_incomplete, marker_entries)` using bounded `os.scandir`."""
    if not root.is_dir():
        return 0, 0, False, []

    total_bytes = 0
    marker_count = 0
    has_incomplete = False
    markers: list[tuple[str, int]] = []
    stack: list[tuple[Path, int]] = [(root, 0)]

    while stack:
        current, depth = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    name_lower = entry.name.lower()
                    if entry.is_dir(follow_symlinks=False):
                        if depth < max_depth:
                            stack.append((Path(entry.path), depth + 1))
                        continue

                    if name_lower.endswith((".incomplete", ".tmp", ".part")):
                        has_incomplete = True
                        continue

                    try:
                        # Follow symlinks for marker validation, but avoid double-counting HF blobs.
                        is_sym = entry.is_symlink()
                        stat_res = entry.stat(follow_symlinks=True)
                        size = int(stat_res.st_size)
                    except OSError:
                        continue

                    if size > 0:
                        marker_count += 1
                        rel = os.path.relpath(entry.path, root)
                        markers.append((rel, size))
                        if not is_sym:
                            total_bytes += size
        except OSError:
            continue

    return total_bytes, marker_count, has_incomplete, markers


def _hf_repo_cache_dir(models_dir: Path, repo_id: str) -> Path:
    return models_dir / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"


def _inspect_hf_repo(repo_dir: Path, display_name: str, key: str) -> tuple[ModelCacheStatus, list[tuple[str, int]]]:
    """Inspect a HuggingFace Hub `models--org--name` directory without double-counting symlinks."""
    if not repo_dir.is_dir():
        return (
            ModelCacheStatus(
                key=key,
                display_name=display_name,
                cached=False,
                path=str(repo_dir),
                size_mb=0.0,
                detail="Not downloaded yet.",
            ),
            [],
        )

    total_bytes, _, has_incomplete, all_markers = _scan_dir_metrics(repo_dir, max_depth=4)
    snapshots_dir = repo_dir / "snapshots"
    _, snap_markers_count, _, snap_markers = _scan_dir_metrics(snapshots_dir, max_depth=3)

    if has_incomplete:
        return (
            ModelCacheStatus(
                key=key,
                display_name=display_name,
                cached=False,
                path=str(repo_dir),
                size_mb=round(total_bytes / (1024.0 * 1024.0), 2),
                detail="Interrupted download detected (.incomplete blobs present; ready to resume).",
            ),
            all_markers,
        )

    if snap_markers_count > 0:
        return (
            ModelCacheStatus(
                key=key,
                display_name=display_name,
                cached=True,
                path=str(repo_dir),
                size_mb=round(total_bytes / (1024.0 * 1024.0), 2),
                detail=f"Cached ({snap_markers_count} snapshot file(s)).",
            ),
            snap_markers,
        )

    return (
        ModelCacheStatus(
            key=key,
            display_name=display_name,
            cached=False,
            path=str(repo_dir),
            size_mb=round(total_bytes / (1024.0 * 1024.0), 2),
            detail="HuggingFace snapshot files not present yet.",
        ),
        all_markers,
    )


def _inspect_subdir_group(
    target_dir: Path,
    *,
    key: str,
    display_name: str,
    preferred_hf_repo: str | None = None,
) -> tuple[ModelCacheStatus, list[tuple[str, int]]]:
    """Inspect non-HF or hybrid cache directories (`fastembed`, `flashrank`, `rapidocr`, `docling`)."""
    if preferred_hf_repo is not None:
        hf_subdir = target_dir / f"models--{preferred_hf_repo.replace('/', '--')}"
        if hf_subdir.is_dir():
            return _inspect_hf_repo(hf_subdir, display_name=display_name, key=key)

    if not target_dir.is_dir():
        return (
            ModelCacheStatus(
                key=key,
                display_name=display_name,
                cached=False,
                path=str(target_dir),
                size_mb=0.0,
                detail="Not downloaded yet.",
            ),
            [],
        )

    total_bytes, marker_count, has_incomplete, markers = _scan_dir_metrics(target_dir, max_depth=4)
    if has_incomplete:
        return (
            ModelCacheStatus(
                key=key,
                display_name=display_name,
                cached=False,
                path=str(target_dir),
                size_mb=round(total_bytes / (1024.0 * 1024.0), 2),
                detail="Partial download files detected (.incomplete/.tmp); ready to resume.",
            ),
            markers,
        )

    if marker_count > 0:
        return (
            ModelCacheStatus(
                key=key,
                display_name=display_name,
                cached=True,
                path=str(target_dir),
                size_mb=round(total_bytes / (1024.0 * 1024.0), 2),
                detail=f"Cached ({marker_count} artifact file(s)).",
            ),
            markers,
        )

    return (
        ModelCacheStatus(
            key=key,
            display_name=display_name,
            cached=False,
            path=str(target_dir),
            size_mb=0.0,
            detail="Directory exists but contains no model artifacts.",
        ),
        [],
    )


def _inspect_docling_models(
    docling_models_dir: Path,
    *,
    key: str,
    display_name: str,
) -> tuple[ModelCacheStatus, list[tuple[str, int]]]:
    """Inspect `<models_dir>/docling/models`, requiring both layout and TableFormer artifacts."""
    status, markers = _inspect_subdir_group(
        docling_models_dir,
        key=key,
        display_name=display_name,
    )
    if not status.cached:
        return status, markers

    layout_dirs = [
        d
        for d in docling_models_dir.iterdir()
        if d.is_dir() and d.name.startswith("docling-project--docling-layout")
    ]
    layout_ready = any(_scan_dir_metrics(d, max_depth=3)[1] > 0 for d in layout_dirs)
    if not layout_ready:
        return (
            ModelCacheStatus(
                key=key,
                display_name=display_name,
                cached=False,
                path=str(docling_models_dir),
                size_mb=status.size_mb,
                detail="Docling layout artifacts are missing; ready to resume.",
            ),
            markers,
        )

    tableformer_dir = docling_models_dir / "docling-project--docling-models"
    if not tableformer_dir.is_dir() or _scan_dir_metrics(tableformer_dir, max_depth=3)[1] == 0:
        return (
            ModelCacheStatus(
                key=key,
                display_name=display_name,
                cached=False,
                path=str(docling_models_dir),
                size_mb=status.size_mb,
                detail="Docling layout cached, but TableFormer artifacts are missing; ready to resume.",
            ),
            markers,
        )

    return status, markers


def _compute_marker_fingerprint(markers: list[tuple[str, int]]) -> str:
    """Compute a fast deterministic SHA-256 fingerprint from relative marker paths and byte sizes."""
    digest = hashlib.sha256()
    for rel_path, size in sorted(markers):
        digest.update(f"{rel_path}:{size}\n".encode("utf-8", errors="replace"))
    return digest.hexdigest()


def _copy_missing_tree(src: Path, dst: Path) -> int:
    """Recursively copy files from `src` into `dst` without overwriting valid existing files."""
    copied = 0
    dst.mkdir(parents=True, exist_ok=True)
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        dst_root = dst if rel == "." else dst / rel
        dst_root.mkdir(parents=True, exist_ok=True)
        for filename in files:
            if filename.lower().endswith((".incomplete", ".tmp", ".part")):
                continue
            src_file = Path(root) / filename
            dst_file = dst_root / filename
            if dst_file.exists():
                try:
                    if dst_file.stat().st_size > 0:
                        continue
                except OSError:
                    pass
            shutil.copy2(src_file, dst_file)
            copied += 1
    return copied


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
        if self._command_runner is not None:
            return self._command_runner(
                cmd,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                timeout=timeout,
            )
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=_CREATE_NO_WINDOW,
            check=False,
        )

    def _run_subprocess_streaming(
        self,
        cmd: list[str],
        *,
        cwd: Path | None,
        env: dict[str, str],
        on_line: Callable[[str], None],
    ) -> int:
        if self._stream_runner is not None:
            return int(self._stream_runner(cmd, cwd, env, on_line))

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=_CREATE_NO_WINDOW,
        )
        try:
            if proc.stdout is not None:
                for raw_line in proc.stdout:
                    cleaned = str(raw_line).rstrip("\r\n")
                    if cleaned:
                        on_line(cleaned)
        finally:
            if proc.stdout is not None:
                try:
                    proc.stdout.close()
                except OSError:
                    pass
        return int(proc.wait())

    def verify_python_runtime(self, python_exe: Path | None = None) -> RuntimeCheckResult:
        """Verify that `torch`, `onnxruntime`, `pymupdf`, `pandas`, `cv2`, and `laya` load cleanly."""
        target_py = (
            Path(python_exe)
            if python_exe is not None
            else self.paths.resolve_backend_python(
                self.config.compute_profile,
                cuda_runtime_ready=self.config.cuda_runtime_ready,
            )
        )
        cmd = [str(target_py), "-c", _RUNTIME_PROBE_SCRIPT]
        try:
            proc = self._run_subprocess_capture(cmd, timeout=_VERIFY_TIMEOUT_SEC)
        except subprocess.TimeoutExpired as exc:
            return RuntimeCheckResult(
                ok=False,
                python_exe=str(target_py),
                error_category="unknown",
                error_detail=f"Python runtime verification timed out after {_VERIFY_TIMEOUT_SEC:.0f}s: {exc}",
                remediation="Verify antivirus/SmartScreen is not stalling Python DLL initialization and retry.",
            )
        except OSError as exc:
            combined = f"{type(exc).__name__}: {exc}"
            category, remediation = _classify_runtime_failure(getattr(exc, "winerror", None), combined)
            return RuntimeCheckResult(
                ok=False,
                python_exe=str(target_py),
                error_category=category,
                error_detail=combined,
                remediation=remediation,
            )

        combined_output = "\n".join(
            part for part in ((proc.stderr or "").strip(), (proc.stdout or "").strip()) if part
        )

        if proc.returncode != 0:
            category, remediation = _classify_runtime_failure(proc.returncode, combined_output)
            return RuntimeCheckResult(
                ok=False,
                python_exe=str(target_py),
                error_category=category,
                error_detail=combined_output or f"Process exited with code {proc.returncode}",
                remediation=remediation,
            )

        payload = _extract_last_json_object(proc.stdout or "")
        if payload is None:
            return RuntimeCheckResult(
                ok=False,
                python_exe=str(target_py),
                error_category="unknown",
                error_detail=combined_output or "Runtime probe exited 0 without JSON payload.",
                remediation="Check `logs/launcher.log` and re-run runtime verification.",
            )

        return RuntimeCheckResult(
            ok=True,
            python_exe=str(target_py),
            torch_version=payload.get("torch_version"),
            torch_cuda_version=payload.get("torch_cuda_version"),
            cuda_available=bool(payload.get("cuda_available", False)),
            onnx_providers=list(payload.get("onnx_providers") or []),
            error_category="ok",
            error_detail=None,
            remediation=None,
        )

    def _build_uv_sync_command(self, uv_exe: str) -> list[str]:
        """Build `uv sync` command preserving `--extra desktop` when defined or unpinned in tests."""
        cmd = [uv_exe, "sync", "--native-tls", "--frozen", "--no-dev"]
        pyproject = self.paths.backend_dir / "pyproject.toml"
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

    @staticmethod
    def _apply_sparse_mode_policy(
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
        if is_backend_running is not None and is_backend_running():
            raise RuntimeError(
                "Backend sidecar must be stopped before modifying Python compute runtime wheels "
                "(prevents WinError 32 file locks)."
            )

        self.paths.ensure_directories(self.config)
        commands_run: list[list[str]] = []

        if target_profile == "cpu":
            self._emit_progress(
                progress_cb,
                step="switch_cpu",
                percent=35.0,
                detail="Switching active compute profile to Standard CPU mode...",
                is_running=True,
            )
            self.config.compute_profile = "cpu"
            if purge_cuda_venv and self.paths.cuda_venv_dir.exists():
                self._emit_progress(
                    progress_cb,
                    step="purge_cuda_venv",
                    percent=70.0,
                    detail=f"Removing cached CUDA virtualenv at {self.paths.cuda_venv_dir}...",
                    is_running=True,
                )
                shutil.rmtree(self.paths.cuda_venv_dir)
                self.config.cuda_runtime_ready = False

            reindex_recommended = self._apply_sparse_mode_policy(
                self.config,
                target_profile="cpu",
                existing_sparse_mode=existing_sparse_mode,
                reindex_sparse=reindex_sparse,
            )
            save_desktop_config(self.config, self.paths)

            active_py = str(
                self.paths.resolve_backend_python("cpu", cuda_runtime_ready=self.config.cuda_runtime_ready)
            )
            detail = "Switched compute profile to Standard CPU mode."
            self._emit_progress(
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
                cuda_runtime_ready=self.config.cuda_runtime_ready,
                sparse_mode_override=self.config.sparse_mode_override,
                reindex_recommended=reindex_recommended,
                detail=detail,
                commands_run=commands_run,
            )

        # Target profile: CUDA 12.4 in %LOCALAPPDATA%\SecondBrain\runtime\venv
        uv_exe = str(self.paths.uv_exe)
        cuda_venv_dir = self.paths.cuda_venv_dir
        cuda_venv_py = self.paths.cuda_venv_python
        base_python = self.paths.base_venv_python
        sync_marker = cuda_venv_dir / _BASE_SYNCED_MARKER
        previous_profile = self.config.compute_profile
        previous_cuda_ready = self.config.cuda_runtime_ready

        base_env = os.environ.copy()
        base_env["UV_NATIVE_TLS"] = "true"
        sync_env = dict(base_env)
        sync_env["UV_PROJECT_ENVIRONMENT"] = str(cuda_venv_dir)
        sync_cwd = self.paths.backend_dir if self.paths.backend_dir.is_dir() else None

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
                    self._build_uv_sync_command(uv_exe),
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
            self._emit_progress(
                progress_cb,
                step=step_name,
                percent=start_pct,
                detail=banner,
                is_running=True,
            )
            commands_run.append(cmd)
            captured_lines: list[str] = []

            def _handle_line(line: str, _step: str = step_name, _pct: float = start_pct) -> None:
                captured_lines.append(line)
                if len(captured_lines) > 80:
                    del captured_lines[:-80]
                self._emit_progress(
                    progress_cb,
                    step=_step,
                    percent=min(end_pct - 1.0, _pct + min(8.0, len(captured_lines) * 0.5)),
                    detail=line,
                    is_running=True,
                )

            try:
                rc = self._run_subprocess_streaming(
                    cmd,
                    cwd=step_cwd,
                    env=step_env,
                    on_line=_handle_line,
                )
            except OSError as exc:
                err_msg = f"Failed to execute `{cmd[0]}` during {step_name}: {exc}"
                self._emit_progress(
                    progress_cb,
                    step=step_name,
                    percent=start_pct,
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
                    sparse_mode_override=self.config.sparse_mode_override,
                    reindex_recommended=False,
                    detail=err_msg,
                    commands_run=commands_run,
                )

            if rc != 0:
                tail = "\n".join(captured_lines[-10:])
                if step_name == "uninstall_onnxruntime" and (
                    "not installed" in tail.lower() or "skipping" in tail.lower()
                ):
                    continue

                err_msg = (
                    f"Command failed (exit code {rc}) at step '{step_name}' "
                    f"(index: {_CUDA_INDEX_URL if 'cu124' in ' '.join(cmd) else 'PyPI'}): "
                    f"{tail or 'see launcher.log for details'}"
                )
                self._emit_progress(
                    progress_cb,
                    step=step_name,
                    percent=start_pct,
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
                    sparse_mode_override=self.config.sparse_mode_override,
                    reindex_recommended=False,
                    detail=err_msg,
                    commands_run=commands_run,
                )

            if step_name == "sync_cuda_venv":
                cuda_venv_dir.mkdir(parents=True, exist_ok=True)
                sync_marker.write_text("ok\n", encoding="utf-8")

        if verify_runtime:
            rt_check = self.verify_python_runtime(cuda_venv_py)
            if not rt_check.ok:
                self.config.cuda_runtime_ready = False
                self.config.compute_profile = previous_profile
                save_desktop_config(self.config, self.paths)
                err_msg = rt_check.error_detail or "CUDA Python runtime verification failed."
                self._emit_progress(
                    progress_cb,
                    step="verify_cuda_runtime",
                    percent=98.0,
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
                    sparse_mode_override=self.config.sparse_mode_override,
                    reindex_recommended=False,
                    detail=err_msg,
                    commands_run=commands_run,
                )

        self.config.cuda_runtime_ready = True
        self.config.compute_profile = "cuda"
        reindex_recommended = self._apply_sparse_mode_policy(
            self.config,
            target_profile="cuda",
            existing_sparse_mode=existing_sparse_mode,
            reindex_sparse=reindex_sparse,
        )
        save_desktop_config(self.config, self.paths)

        detail = "NVIDIA CUDA 12.4 runtime environment is ready."
        self._emit_progress(
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
            sparse_mode_override=self.config.sparse_mode_override,
            reindex_recommended=reindex_recommended,
            detail=detail,
            commands_run=commands_run,
        )

    def _inspect_single_model(self, spec: ModelSpec) -> tuple[ModelCacheStatus, list[tuple[str, int]]]:
        models_root = self.models_dir
        if spec.key == "bge_m3":
            return _inspect_hf_repo(
                _hf_repo_cache_dir(models_root, spec.repo_or_id),
                display_name=spec.display_name,
                key=spec.key,
            )
        if spec.key == "laya_multilingual":
            return _inspect_hf_repo(
                _hf_repo_cache_dir(models_root, spec.repo_or_id),
                display_name=spec.display_name,
                key=spec.key,
            )
        if spec.key == "fastembed_bm25":
            return _inspect_subdir_group(
                models_root / "fastembed",
                key=spec.key,
                display_name=spec.display_name,
                preferred_hf_repo=spec.repo_or_id,
            )
        if spec.key == "flashrank":
            return _inspect_subdir_group(
                models_root / "flashrank" / spec.repo_or_id,
                key=spec.key,
                display_name=spec.display_name,
            )
        if spec.key == "rapidocr":
            return _inspect_subdir_group(
                models_root / "rapidocr",
                key=spec.key,
                display_name=spec.display_name,
            )
        if spec.key == "docling_layout_tableformer":
            return _inspect_docling_models(
                models_root / "docling" / "models",
                key=spec.key,
                display_name=spec.display_name,
            )
        return (
            ModelCacheStatus(
                key=spec.key,
                display_name=spec.display_name,
                cached=False,
                path=str(models_root / spec.cache_subdir),
                size_mb=0.0,
                detail="Unknown model group.",
            ),
            [],
        )

    def _read_manifest_ready(self) -> tuple[bool, dict[str, Any] | None]:
        manifest_file = self.manifest_path
        if not manifest_file.is_file():
            return False, None
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            return False, None
        if not isinstance(data, dict) or data.get("models_ready") is not True:
            return False, None
        models_map = data.get("models")
        if not isinstance(models_map, dict):
            return False, None
        for spec in MODEL_SPECS:
            entry = models_map.get(spec.key)
            if not isinstance(entry, dict) or entry.get("cached") is not True:
                return False, data
        return True, data

    def inspect_models_status(self) -> dict[str, Any]:
        """Inspect `models_ready.json` and on-disk markers for all 6 model groups in `< 200 ms`."""
        statuses: list[ModelCacheStatus] = []
        total_size_mb = 0.0
        for spec in MODEL_SPECS:
            status, _ = self._inspect_single_model(spec)
            statuses.append(status)
            total_size_mb += status.size_mb

        manifest_ready, _ = self._read_manifest_ready()
        all_markers_present = all(item.cached for item in statuses)
        models_ready = bool(manifest_ready and all_markers_present)

        return {
            "models_ready": models_ready,
            "models": statuses,
            "total_size_mb": round(total_size_mb, 2),
            "models_dir": str(self.models_dir),
        }

    def _inject_model_cache_env(self) -> None:
        """Configure AI model cache environment variables before importing or invoking downloaders."""
        models_root = self.models_dir
        hf_home = models_root / "huggingface"
        hf_hub = hf_home / "hub"
        torch_home = models_root / "torch"
        fastembed_dir = models_root / "fastembed"
        rapidocr_dir = models_root / "rapidocr"
        docling_dir = models_root / "docling"
        docling_models_dir = docling_dir / "models"

        for directory in (
            models_root,
            hf_home,
            hf_hub,
            torch_home,
            fastembed_dir,
            rapidocr_dir,
            docling_dir,
            docling_models_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HF_HUB_CACHE"] = str(hf_hub)
        os.environ["TORCH_HOME"] = str(torch_home)
        os.environ["FASTEMBED_CACHE_PATH"] = str(fastembed_dir)
        os.environ["RAPIDOCR_MODEL_DIR"] = str(rapidocr_dir)
        os.environ["DOCLING_CACHE_DIR"] = str(docling_dir)
        os.environ["DOCLING_ARTIFACTS_PATH"] = str(docling_models_dir)
        os.environ["FLASHRANK_ALLOW_DOWNLOAD"] = "true"

    def _download_model_group(self, spec: ModelSpec) -> None:
        """Download and verify a single model group into its target cache subdirectory."""
        models_root = self.models_dir
        if spec.key in self._download_hooks:
            target = models_root / spec.cache_subdir
            target.mkdir(parents=True, exist_ok=True)
            self._download_hooks[spec.key](target)
            return

        if spec.key in ("bge_m3", "laya_multilingual"):
            from huggingface_hub import snapshot_download

            hf_hub_cache = models_root / "huggingface" / "hub"
            hf_hub_cache.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=spec.repo_or_id,
                cache_dir=str(hf_hub_cache),
            )
            return

        if spec.key == "fastembed_bm25":
            from fastembed import SparseTextEmbedding

            fastembed_dir = models_root / "fastembed"
            fastembed_dir.mkdir(parents=True, exist_ok=True)
            SparseTextEmbedding(
                model_name=spec.repo_or_id,
                cache_dir=str(fastembed_dir),
            )
            return

        if spec.key == "flashrank":
            from src.rag.reranker import FlashRankReranker

            flashrank_dir = models_root / "flashrank"
            flashrank_dir.mkdir(parents=True, exist_ok=True)
            reranker = FlashRankReranker(
                model_name=spec.repo_or_id,
                cache_dir=flashrank_dir,
                allow_download=True,
            )
            ranker = reranker._get_or_load_ranker()
            if ranker is None:
                raise RuntimeError(
                    f"FlashRankReranker._get_or_load_ranker() returned None for {spec.repo_or_id}"
                )
            return

        if spec.key == "rapidocr":
            from src.ingestion import engineering_extractor

            rapidocr_dir = models_root / "rapidocr"
            rapidocr_dir.mkdir(parents=True, exist_ok=True)
            os.environ["RAPIDOCR_MODEL_DIR"] = str(rapidocr_dir)
            engineering_extractor._RAPIDOCR_ENGINE = None
            engineering_extractor._RAPIDOCR_INIT_ATTEMPTED = False
            engine = engineering_extractor.get_rapidocr_engine()
            if engine is None:
                raise RuntimeError("RapidOCR engine initialization returned None.")

            # RapidOCR v3 initializes ONNX sessions lazily; trigger det/cls/rec loaders explicitly.
            if hasattr(engine, "_load_det_model"):
                engine._load_det_model()
            if getattr(engine, "use_cls", True) and hasattr(engine, "_load_cls_model"):
                engine._load_cls_model()
            if getattr(engine, "use_rec", True) and hasattr(engine, "_load_rec_model"):
                engine._load_rec_model()

            status_after, _ = self._inspect_single_model(spec)
            if not status_after.cached and callable(engine):
                import numpy as np

                dummy = np.full((64, 192, 3), 255, dtype=np.uint8)
                dummy[20:44, 30:160] = 0
                try:
                    engine(dummy)
                except Exception:
                    pass
            return

        if spec.key == "docling_layout_tableformer":
            from docling.utils.model_downloader import download_models

            docling_models_dir = models_root / "docling" / "models"
            docling_models_dir.mkdir(parents=True, exist_ok=True)
            download_models(
                output_dir=docling_models_dir,
                with_layout=True,
                with_tableformer=True,
                with_code_formula=False,
                with_picture_classifier=False,
                with_rapidocr=True,
                progress=False,
            )
            return

        raise ValueError(f"Unsupported model spec key: {spec.key}")

    def _write_models_ready_manifest(self) -> dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat()
        model_entries: dict[str, Any] = {}
        for spec in MODEL_SPECS:
            status, markers = self._inspect_single_model(spec)
            if not status.cached:
                raise RuntimeError(
                    f"Model group '{spec.key}' did not produce valid cache markers at {status.path}"
                )
            model_entries[spec.key] = {
                "display_name": spec.display_name,
                "repo_or_id": spec.repo_or_id,
                "cache_subdir": spec.cache_subdir,
                "cached": True,
                "path": status.path,
                "size_mb": status.size_mb,
                "marker_count": len(markers),
                "checksum": _compute_marker_fingerprint(markers),
                "verified_at": now_iso,
            }

        manifest = {
            "schema_version": 1,
            "models_ready": True,
            "verified_at": now_iso,
            "app_version": self.config.last_installed_version,
            "models_dir": str(self.models_dir),
            "models": model_entries,
        }
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    def ensure_models_ready(
        self,
        force: bool = False,
        progress_cb: Callable[[BootstrapProgress], None] | None = None,
    ) -> dict[str, Any]:
        """Download and verify all 6 required model groups with warm fast-path and clean resume."""
        self.paths.ensure_directories(self.config)
        self._inject_model_cache_env()

        if not force:
            current_status = self.inspect_models_status()
            if current_status["models_ready"]:
                if not self.config.models_ready:
                    self.config.models_ready = True
                    save_desktop_config(self.config, self.paths)
                model_states = {spec.key: "cached" for spec in MODEL_SPECS}
                self._emit_progress(
                    progress_cb,
                    step="complete",
                    percent=100.0,
                    detail="All AI models are already cached and verified.",
                    is_running=False,
                    completed=True,
                    models=model_states,
                )
                return current_status

        if force and self.manifest_path.is_file():
            try:
                self.manifest_path.unlink()
            except OSError:
                pass

        model_states: dict[str, str] = {}
        for spec in MODEL_SPECS:
            st, _ = self._inspect_single_model(spec)
            model_states[spec.key] = "cached" if (not force and st.cached) else "pending"

        total_groups = len(MODEL_SPECS)
        self._emit_progress(
            progress_cb,
            step="prepare_models",
            percent=2.0,
            detail="Preparing AI model downloads...",
            is_running=True,
            completed=False,
            models=model_states,
        )

        for idx, spec in enumerate(MODEL_SPECS):
            start_pct = 5.0 + (idx / total_groups) * 90.0
            end_pct = 5.0 + ((idx + 1) / total_groups) * 90.0

            if not force and model_states.get(spec.key) == "cached":
                self._emit_progress(
                    progress_cb,
                    step=f"verify_{spec.key}",
                    percent=end_pct,
                    detail=f"Verified cached {spec.display_name}.",
                    is_running=True,
                    completed=False,
                    models=model_states,
                )
                continue

            model_states[spec.key] = "downloading"
            self._emit_progress(
                progress_cb,
                step=f"download_{spec.key}",
                percent=start_pct,
                detail=f"Downloading {spec.display_name} ({spec.repo_or_id})...",
                is_running=True,
                completed=False,
                models=model_states,
            )

            try:
                self._download_model_group(spec)
                post_status, _ = self._inspect_single_model(spec)
                if not post_status.cached:
                    raise RuntimeError(
                        f"Download finished for {spec.display_name}, but cache validation failed: {post_status.detail}"
                    )
            except Exception as exc:
                # Never delete existing HuggingFace blobs so subsequent runs resume cleanly.
                model_states[spec.key] = "failed"
                endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
                err_detail = f"Failed downloading {spec.display_name} ({spec.repo_or_id}, endpoint={endpoint}): {exc}"
                self.config.models_ready = False
                save_desktop_config(self.config, self.paths)
                self._emit_progress(
                    progress_cb,
                    step=f"download_{spec.key}",
                    percent=start_pct,
                    detail=err_detail,
                    error=str(exc),
                    is_running=False,
                    completed=False,
                    models=model_states,
                )
                raise

            model_states[spec.key] = "cached"
            self._emit_progress(
                progress_cb,
                step=f"download_{spec.key}",
                percent=end_pct,
                detail=f"Completed {spec.display_name}.",
                is_running=True,
                completed=False,
                models=model_states,
            )

        self._write_models_ready_manifest()
        self.config.models_ready = True
        save_desktop_config(self.config, self.paths)

        final_status = self.inspect_models_status()
        self._emit_progress(
            progress_cb,
            step="complete",
            percent=100.0,
            detail="All 6 offline AI model groups downloaded and verified.",
            is_running=False,
            completed=True,
            models=model_states,
        )
        return final_status

    def import_local_models_dir(self, source_dir: Path | str) -> dict[str, Any]:
        """Import pre-cached model directories from a local offline bundle into `models_dir`."""
        source = Path(source_dir).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Local models directory not found: {source}")

        self.paths.ensure_directories(self.config)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        for subdir in _RECOGNIZED_MODEL_SUBDIRS:
            src_sub = source / subdir
            if src_sub.is_dir():
                _copy_missing_tree(src_sub, self.models_dir / subdir)

        all_cached = all(self._inspect_single_model(spec)[0].cached for spec in MODEL_SPECS)
        if all_cached:
            self._write_models_ready_manifest()
            self.config.models_ready = True
            save_desktop_config(self.config, self.paths)
        else:
            src_manifest = source / _MANIFEST_FILENAME
            if src_manifest.is_file() and not self.manifest_path.exists():
                shutil.copy2(src_manifest, self.manifest_path)

        return self.inspect_models_status()
