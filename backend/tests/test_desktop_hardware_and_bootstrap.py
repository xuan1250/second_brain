"""Unit and API integration tests for Phase 2 runtime verification, CUDA switcher, and model bootstrap."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from qdrant_client import AsyncQdrantClient, models

from src.api.main import create_app
from src.config import get_settings
from src.desktop import (
    MODEL_SPECS,
    BootstrapProgress,
    ComputeRuntimeResult,
    DesktopConfig,
    DesktopPaths,
    GpuAdapterInfo,
    HardwareReport,
    ModelCacheStatus,
    RuntimeBootstrapService,
    RuntimeCheckResult,
    detect_collection_sparse_mode,
    load_desktop_config,
    save_desktop_config,
)
from src.ingestion.indexer import (
    DeterministicBM25Encoder,
    DeterministicStubEmbedder,
    set_embedder,
    set_sparse_encoder,
)
from src.rag.graph import close_checkpointer
from src.rag.reranker import set_reranker
from src.rag.retriever import set_active_qdrant_client


def _make_staged_paths(tmp_path: Path) -> DesktopPaths:
    app_dir = tmp_path / "staged_app"
    (app_dir / "app" / "backend").mkdir(parents=True, exist_ok=True)
    (app_dir / "app" / "frontend" / "dist").mkdir(parents=True, exist_ok=True)
    (app_dir / "bin").mkdir(parents=True, exist_ok=True)
    uv_name = "uv.exe" if sys.platform == "win32" else "uv"
    (app_dir / "bin" / uv_name).write_text("", encoding="utf-8")

    py_rel = Path("Scripts") / "python.exe" if sys.platform == "win32" else Path("bin") / "python"
    base_py = app_dir / "venv" / py_rel
    base_py.parent.mkdir(parents=True, exist_ok=True)
    base_py.write_text("", encoding="utf-8")

    user_home = tmp_path / "LocalAppData" / "SecondBrain"
    return DesktopPaths(app_dir=app_dir, user_data_dir=user_home)


def _populate_all_six_model_caches(models_dir: Path, *, write_manifest: bool = True) -> None:
    """Create valid non-empty directory markers for all 6 canonical model groups."""
    for repo in ("BAAI/bge-m3", "convaiinnovations/laya-multilingual"):
        snap_dir = models_dir / "huggingface" / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / "rev1"
        snap_dir.mkdir(parents=True, exist_ok=True)
        (snap_dir / "config.json").write_text('{"ok": true}', encoding="utf-8")
        (snap_dir / "tokenizer_config.json").write_text('{"ok": true}', encoding="utf-8")
        (snap_dir / "sparse_linear.pt").write_bytes(b"weights")

    bm25_snap = models_dir / "fastembed" / "models--Qdrant--bm25" / "snapshots" / "rev1"
    bm25_snap.mkdir(parents=True, exist_ok=True)
    (bm25_snap / "config.json").write_text('{"bm25": true}', encoding="utf-8")

    fr_dir = models_dir / "flashrank" / "ms-marco-MultiBERT-L-12"
    fr_dir.mkdir(parents=True, exist_ok=True)
    (fr_dir / "flashrank-MultiBERT-L-12.onnx").write_bytes(b"onnx")

    ocr_dir = models_dir / "rapidocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    (ocr_dir / "ch_PP-OCRv4_det_infer.onnx").write_bytes(b"det-onnx")
    (ocr_dir / "ch_PP-OCRv4_rec_infer.onnx").write_bytes(b"rec-onnx")

    dl_layout = models_dir / "docling" / "models" / "docling-project--docling-layout-heron"
    dl_tf = models_dir / "docling" / "models" / "docling-project--docling-models"
    dl_layout.mkdir(parents=True, exist_ok=True)
    dl_tf.mkdir(parents=True, exist_ok=True)
    (dl_layout / "model.safetensors").write_bytes(b"layout")
    (dl_tf / "tableformer.safetensors").write_bytes(b"tableformer")

    if write_manifest:
        manifest = {
            "schema_version": 1,
            "models_ready": True,
            "verified_at": "2026-11-23T00:00:00+00:00",
            "models": {spec.key: {"cached": True} for spec in MODEL_SPECS},
        }
        (models_dir / "models_ready.json").write_text(json.dumps(manifest), encoding="utf-8")


# ==============================================================================
# R2.1 — C-Extension Environment Self-Test
# ==============================================================================


def test_verify_python_runtime_success_and_noisy_stdout(tmp_path: Path):
    paths = _make_staged_paths(tmp_path)
    payload = {
        "torch_version": "2.5.1+cu124",
        "torch_cuda_version": "12.4",
        "cuda_available": True,
        "onnx_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }

    def fake_runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert cmd[0] == str(paths.base_venv_python)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=f"Some library warning line\n{json.dumps(payload)}\n",
            stderr="",
        )

    service = RuntimeBootstrapService(paths=paths, command_runner=fake_runner)
    result = service.verify_python_runtime()

    assert isinstance(result, RuntimeCheckResult)
    assert result.ok is True
    assert result.error_category == "ok"
    assert result.torch_version == "2.5.1+cu124"
    assert result.torch_cuda_version == "12.4"
    assert result.cuda_available is True
    assert result.onnx_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert result.error_detail is None
    assert result.remediation is None


@pytest.mark.parametrize(
    ("returncode", "stderr_text", "expected_category", "remediation_hint"),
    [
        (
            1,
            "OSError: [WinError 4551] An Application Control policy has blocked this file: c10.dll",
            "app_control_blocked",
            "Application Control",
        ),
        (
            -1073740760,
            "DLL load failed (0xc0000428): file is not digitally signed; Smart App Control",
            "app_control_blocked",
            "Allowlist",
        ),
        (
            -1073741701,
            "0xc000007b blocked by Windows Application Control policy",
            "app_control_blocked",
            "Application Control",
        ),
        (
            1,
            "ImportError: DLL load failed while importing _C: The specified module could not be found (vcruntime140_1.dll, WinError 126)",
            "missing_vc_redist",
            "Visual C++",
        ),
        (
            1,
            "ImportError: DLL load failed while importing torch_cpu: msvcp140.dll missing",
            "missing_vc_redist",
            "vc_redist.x64.exe",
        ),
        (
            1,
            "ModuleNotFoundError: No module named 'laya'",
            "missing_package",
            "dependency",
        ),
    ],
)
def test_verify_python_runtime_failure_classification(
    tmp_path: Path,
    returncode: int,
    stderr_text: str,
    expected_category: str,
    remediation_hint: str,
):
    paths = _make_staged_paths(tmp_path)

    def fake_runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout="", stderr=stderr_text)

    service = RuntimeBootstrapService(paths=paths, command_runner=fake_runner)
    result = service.verify_python_runtime()

    assert result.ok is False
    assert result.error_category == expected_category
    assert result.error_detail
    assert result.remediation is not None
    assert remediation_hint.lower() in result.remediation.lower()


def test_verify_python_runtime_timeout(tmp_path: Path):
    paths = _make_staged_paths(tmp_path)

    def timeout_runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=30.0)

    service = RuntimeBootstrapService(paths=paths, command_runner=timeout_runner)
    result = service.verify_python_runtime()

    assert result.ok is False
    assert result.error_category == "unknown"
    assert "timed out" in (result.error_detail or "").lower()


# ==============================================================================
# R2.2 — CPU <-> NVIDIA CUDA 12.4 Runtime Switcher & WinError 32 Guard
# ==============================================================================


def test_ensure_compute_runtime_refuses_when_backend_running(tmp_path: Path):
    paths = _make_staged_paths(tmp_path)
    called = False

    def should_not_run(*args: Any, **kwargs: Any) -> int:
        nonlocal called
        called = True
        return 0

    service = RuntimeBootstrapService(paths=paths, stream_runner=should_not_run)
    with pytest.raises(RuntimeError, match="WinError 32"):
        service.ensure_compute_runtime("cuda", is_backend_running=lambda: True)

    assert called is False


def test_ensure_compute_runtime_cpu_to_cuda_commands_and_sparse_pinning(tmp_path: Path):
    paths = _make_staged_paths(tmp_path)
    observed_calls: list[tuple[list[str], Path | None, dict[str, str]]] = []
    progress_events: list[BootstrapProgress] = []

    def fake_stream(
        cmd: list[str],
        cwd: Path | None,
        env: dict[str, str],
        on_line: Any,
    ) -> int:
        observed_calls.append((cmd, cwd, env))
        on_line(f"Executing: {' '.join(cmd[:4])}")
        # Simulate uv venv creating the interpreter on disk
        if len(cmd) >= 2 and cmd[1] == "venv":
            paths.cuda_venv_python.parent.mkdir(parents=True, exist_ok=True)
            paths.cuda_venv_python.write_text("", encoding="utf-8")
        return 0

    service = RuntimeBootstrapService(
        paths=paths,
        config=DesktopConfig(compute_profile="cpu", cuda_runtime_ready=False),
        stream_runner=fake_stream,
    )
    res = service.ensure_compute_runtime(
        "cuda",
        is_backend_running=lambda: False,
        existing_sparse_mode="bm25",
        reindex_sparse=False,
        progress_cb=progress_events.append,
    )

    assert isinstance(res, ComputeRuntimeResult)
    assert res.ok is True
    assert res.compute_profile == "cuda"
    assert res.cuda_runtime_ready is True
    assert res.python_exe == str(paths.cuda_venv_python)
    assert res.sparse_mode_override == "bm25"
    assert res.reindex_recommended is True
    assert len(observed_calls) == 5

    venv_cmd, _, _ = observed_calls[0]
    assert venv_cmd == [
        str(paths.uv_exe),
        "venv",
        "--relocatable",
        str(paths.cuda_venv_dir),
        "--python",
        str(paths.base_venv_python),
    ]

    sync_cmd, sync_cwd, sync_env = observed_calls[1]
    assert sync_cmd[:5] == [str(paths.uv_exe), "sync", "--native-tls", "--frozen", "--no-dev"]
    assert "--extra" in sync_cmd and "desktop" in sync_cmd
    assert "--no-install-project" in sync_cmd
    assert sync_cwd == paths.backend_dir
    assert sync_env["UV_PROJECT_ENVIRONMENT"] == str(paths.cuda_venv_dir)
    assert sync_env["UV_NATIVE_TLS"] == "true"

    torch_cmd = observed_calls[2][0]
    assert torch_cmd == [
        str(paths.uv_exe),
        "pip",
        "install",
        "--native-tls",
        "--python",
        str(paths.cuda_venv_python),
        "--reinstall-package",
        "torch",
        "--reinstall-package",
        "torchvision",
        "--index-url",
        "https://download.pytorch.org/whl/cu124",
        "torch>=2.4.0",
        "torchvision>=0.19.0",
    ]

    uninstall_cmd = observed_calls[3][0]
    assert uninstall_cmd == [
        str(paths.uv_exe),
        "pip",
        "uninstall",
        "--python",
        str(paths.cuda_venv_python),
        "onnxruntime",
    ]

    ort_gpu_cmd = observed_calls[4][0]
    assert ort_gpu_cmd == [
        str(paths.uv_exe),
        "pip",
        "install",
        "--native-tls",
        "--python",
        str(paths.cuda_venv_python),
        "onnxruntime-gpu==1.20.0",
    ]

    persisted = load_desktop_config(paths)
    assert persisted.compute_profile == "cuda"
    assert persisted.cuda_runtime_ready is True
    assert persisted.sparse_mode_override == "bm25"
    assert progress_events[-1].completed is True
    assert progress_events[-1].percent == 100.0

    # When cuda_venv_python already exists and reindex_sparse=True, venv/sync are skipped and sparse becomes auto
    observed_calls.clear()
    res_reindex = service.ensure_compute_runtime(
        "cuda",
        is_backend_running=lambda: False,
        existing_sparse_mode="bm25",
        reindex_sparse=True,
    )
    assert res_reindex.ok is True
    assert len(observed_calls) == 3
    assert res_reindex.sparse_mode_override == "auto"
    assert res_reindex.reindex_recommended is False
    assert load_desktop_config(paths).sparse_mode_override == "auto"


def test_ensure_compute_runtime_cuda_to_cpu_preserves_or_purges_cuda_venv(tmp_path: Path):
    paths = _make_staged_paths(tmp_path)
    paths.cuda_venv_python.parent.mkdir(parents=True, exist_ok=True)
    paths.cuda_venv_python.write_text("py", encoding="utf-8")

    cfg = DesktopConfig(compute_profile="cuda", cuda_runtime_ready=True, sparse_mode_override="bm25")
    save_desktop_config(cfg, paths)
    service = RuntimeBootstrapService(paths=paths, config=cfg)

    res_keep = service.ensure_compute_runtime("cpu", is_backend_running=lambda: False)
    assert res_keep.ok is True
    assert res_keep.compute_profile == "cpu"
    assert res_keep.python_exe == str(paths.base_venv_python)
    assert res_keep.cuda_runtime_ready is True
    assert paths.cuda_venv_dir.is_dir()
    assert load_desktop_config(paths).compute_profile == "cpu"

    res_purge = service.ensure_compute_runtime(
        "cpu",
        is_backend_running=lambda: False,
        purge_cuda_venv=True,
    )
    assert res_purge.ok is True
    assert res_purge.cuda_runtime_ready is False
    assert not paths.cuda_venv_dir.exists()


def test_ensure_compute_runtime_wheel_failure_reports_error(tmp_path: Path):
    paths = _make_staged_paths(tmp_path)
    events: list[BootstrapProgress] = []

    def failing_stream(cmd: list[str], cwd: Path | None, env: dict[str, str], on_line: Any) -> int:
        if any("cu124" in arg for arg in cmd):
            on_line("ERROR: TLS handshake failed to download.pytorch.org")
            return 2
        if len(cmd) >= 2 and cmd[1] == "venv":
            paths.cuda_venv_python.parent.mkdir(parents=True, exist_ok=True)
            paths.cuda_venv_python.write_text("", encoding="utf-8")
        return 0

    service = RuntimeBootstrapService(paths=paths, stream_runner=failing_stream)
    res = service.ensure_compute_runtime("cuda", progress_cb=events.append)

    assert res.ok is False
    assert res.cuda_runtime_ready is False
    assert "cu124" in res.detail
    assert events[-1].error is not None
    assert events[-1].is_running is False


def test_sync_cuda_venv_retries_after_partial_failure_and_verify_runtime_blocks_persist(tmp_path: Path):
    paths = _make_staged_paths(tmp_path)
    calls: list[list[str]] = []
    fail_sync = True

    def fake_stream(cmd: list[str], cwd: Path | None, env: dict[str, str], on_line: Any) -> int:
        calls.append(cmd)
        if len(cmd) >= 2 and cmd[1] == "venv":
            paths.cuda_venv_python.parent.mkdir(parents=True, exist_ok=True)
            paths.cuda_venv_python.write_text("", encoding="utf-8")
            return 0
        if len(cmd) >= 2 and cmd[1] == "sync" and fail_sync:
            on_line("Transient PyPI failure")
            return 1
        return 0

    def failing_verify(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="OSError: [WinError 4551] An Application Control policy has blocked this file",
        )

    cfg = DesktopConfig(compute_profile="cpu", cuda_runtime_ready=False)
    save_desktop_config(cfg, paths)
    service = RuntimeBootstrapService(
        paths=paths,
        config=cfg,
        command_runner=failing_verify,
        stream_runner=fake_stream,
    )

    # 1. Fail during sync_cuda_venv after venv python was already created
    first = service.ensure_compute_runtime("cuda", verify_runtime=True)
    assert first.ok is False
    assert paths.cuda_venv_python.is_file()
    assert not (paths.cuda_venv_dir / ".base_synced").exists()

    # 2. Retry succeeds in sync_cuda_venv (NOT skipped!), then verify_runtime fails -> must NOT persist cuda_runtime_ready=True
    fail_sync = False
    calls.clear()
    second = service.ensure_compute_runtime("cuda", verify_runtime=True)
    assert second.ok is False
    assert any(len(c) >= 2 and c[1] == "sync" for c in calls)
    assert (paths.cuda_venv_dir / ".base_synced").is_file()
    persisted = load_desktop_config(paths)
    assert persisted.cuda_runtime_ready is False
    assert persisted.compute_profile == "cpu"


def test_inspect_docling_models_requires_layout_and_tableformer(tmp_path: Path):
    paths = _make_staged_paths(tmp_path)
    _populate_all_six_model_caches(paths.models_dir, write_manifest=True)
    service = RuntimeBootstrapService(paths=paths)
    assert service.inspect_models_status()["models_ready"] is True

    # Delete layout folder while keeping docling-models (TableFormer)
    layout_dir = paths.models_dir / "docling" / "models" / "docling-project--docling-layout-heron"
    for child in list(layout_dir.iterdir()):
        child.unlink()
    layout_dir.rmdir()

    status = service.inspect_models_status()
    assert status["models_ready"] is False
    docling_entry = next(m for m in status["models"] if m.key == "docling_layout_tableformer")
    assert docling_entry.cached is False
    assert "layout" in docling_entry.detail.lower()


# ==============================================================================
# R2.3 — Option 1 Resumable Model Downloader & Warm Fast-Path
# ==============================================================================


def test_ensure_models_ready_downloads_all_six_groups_and_injects_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = _make_staged_paths(tmp_path)
    models_dir = paths.models_dir
    service = RuntimeBootstrapService(paths=paths)
    progress_events: list[BootstrapProgress] = []

    hf_calls: list[tuple[str, str | None]] = []
    fastembed_calls: list[tuple[str, str | None]] = []
    flashrank_calls: list[tuple[str, Path]] = []
    rapidocr_calls: list[str] = []
    docling_calls: list[dict[str, Any]] = []

    def fake_snapshot_download(repo_id: str, cache_dir: str | None = None, **kwargs: Any) -> str:
        hf_calls.append((repo_id, cache_dir))
        assert cache_dir == str(models_dir / "huggingface" / "hub")
        assert os.environ["HF_HOME"] == str(models_dir / "huggingface")
        assert os.environ["HF_HUB_CACHE"] == str(models_dir / "huggingface" / "hub")
        snap = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "rev1"
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "config.json").write_text('{"ok": true}', encoding="utf-8")
        return str(snap)

    class FakeSparseTextEmbedding:
        def __init__(self, model_name: str, cache_dir: str | None = None, **kwargs: Any) -> None:
            fastembed_calls.append((model_name, cache_dir))
            assert cache_dir == str(models_dir / "fastembed")
            assert os.environ["FASTEMBED_CACHE_PATH"] == str(models_dir / "fastembed")
            snap = Path(cache_dir) / "models--Qdrant--bm25" / "snapshots" / "rev1"
            snap.mkdir(parents=True, exist_ok=True)
            (snap / "config.json").write_text('{"bm25": true}', encoding="utf-8")

    class FakeFlashRankReranker:
        def __init__(self, model_name: str, cache_dir: Path, allow_download: bool = False) -> None:
            self.model_name = model_name
            self.cache_dir = Path(cache_dir)
            self.allow_download = allow_download

        def _get_or_load_ranker(self) -> Any:
            assert self.allow_download is True
            assert os.environ["FLASHRANK_ALLOW_DOWNLOAD"] == "true"
            flashrank_calls.append((self.model_name, self.cache_dir))
            out = self.cache_dir / self.model_name
            out.mkdir(parents=True, exist_ok=True)
            (out / "flashrank-MultiBERT-L-12.onnx").write_bytes(b"ranker-weights")
            return object()

    class FakeRapidOcrEngine:
        use_cls = True
        use_rec = True

        def _load_det_model(self) -> None:
            out = Path(os.environ["RAPIDOCR_MODEL_DIR"])
            out.mkdir(parents=True, exist_ok=True)
            (out / "ch_PP-OCRv4_det_infer.onnx").write_bytes(b"det")

        def _load_cls_model(self) -> None:
            out = Path(os.environ["RAPIDOCR_MODEL_DIR"])
            (out / "ch_ppocr_mobile_v2.0_cls_infer.onnx").write_bytes(b"cls")

        def _load_rec_model(self) -> None:
            out = Path(os.environ["RAPIDOCR_MODEL_DIR"])
            (out / "ch_PP-OCRv4_rec_infer.onnx").write_bytes(b"rec")

    def fake_get_rapidocr_engine() -> FakeRapidOcrEngine:
        rapidocr_calls.append(os.environ["RAPIDOCR_MODEL_DIR"])
        return FakeRapidOcrEngine()

    def fake_docling_download_models(**kwargs: Any) -> Path:
        docling_calls.append(kwargs)
        out_dir = Path(kwargs["output_dir"])
        assert kwargs["with_layout"] is True
        assert kwargs["with_tableformer"] is True
        assert kwargs["with_code_formula"] is False
        assert kwargs["with_picture_classifier"] is False
        assert kwargs["with_rapidocr"] is True
        (out_dir / "docling-project--docling-layout-heron").mkdir(parents=True, exist_ok=True)
        (out_dir / "docling-project--docling-layout-heron" / "model.safetensors").write_bytes(b"layout")
        (out_dir / "docling-project--docling-models").mkdir(parents=True, exist_ok=True)
        (out_dir / "docling-project--docling-models" / "tableformer.safetensors").write_bytes(b"tf")
        return out_dir

    fake_hf_mod = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    fake_fe_mod = types.SimpleNamespace(SparseTextEmbedding=FakeSparseTextEmbedding)
    fake_dl_mod = types.SimpleNamespace(download_models=fake_docling_download_models)

    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_mod)
    monkeypatch.setitem(sys.modules, "fastembed", fake_fe_mod)
    monkeypatch.setitem(sys.modules, "docling.utils.model_downloader", fake_dl_mod)

    with (
        patch("src.rag.reranker.FlashRankReranker", FakeFlashRankReranker),
        patch("src.ingestion.engineering_extractor.get_rapidocr_engine", fake_get_rapidocr_engine),
    ):
        status = service.ensure_models_ready(progress_cb=progress_events.append)

    assert status["models_ready"] is True
    assert len(status["models"]) == 6
    assert all(isinstance(m, ModelCacheStatus) and m.cached for m in status["models"])
    assert hf_calls == [
        ("BAAI/bge-m3", str(models_dir / "huggingface" / "hub")),
        ("convaiinnovations/laya-multilingual", str(models_dir / "huggingface" / "hub")),
    ]
    assert fastembed_calls == [("Qdrant/bm25", str(models_dir / "fastembed"))]
    assert flashrank_calls == [("ms-marco-MultiBERT-L-12", models_dir / "flashrank")]
    assert rapidocr_calls == [str(models_dir / "rapidocr")]
    assert len(docling_calls) == 1
    assert (models_dir / "rapidocr" / "ch_PP-OCRv4_rec_infer.onnx").is_file()
    assert (models_dir / "models_ready.json").is_file()

    manifest = json.loads((models_dir / "models_ready.json").read_text(encoding="utf-8"))
    assert manifest["models_ready"] is True
    assert set(manifest["models"].keys()) == {spec.key for spec in MODEL_SPECS}
    assert load_desktop_config(paths).models_ready is True
    assert progress_events[-1].completed is True
    assert progress_events[-1].percent == 100.0


def test_warm_fast_path_under_200ms_and_zero_downloads(tmp_path: Path):
    paths = _make_staged_paths(tmp_path)
    _populate_all_six_model_caches(paths.models_dir, write_manifest=True)

    called_hooks: list[str] = []
    hooks = {
        spec.key: (lambda _dir, k=spec.key: called_hooks.append(k))
        for spec in MODEL_SPECS
    }
    service = RuntimeBootstrapService(paths=paths, download_hooks=hooks)

    t0 = time.perf_counter()
    status = service.ensure_models_ready()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert status["models_ready"] is True
    assert len(status["models"]) == 6
    assert called_hooks == []
    assert elapsed_ms < 200.0
    assert load_desktop_config(paths).models_ready is True


def test_partial_download_failure_preserves_hf_blobs_and_resumes_cleanly(tmp_path: Path):
    paths = _make_staged_paths(tmp_path)
    models_dir = paths.models_dir
    attempts: dict[str, int] = {}
    events: list[BootstrapProgress] = []

    laya_blobs_dir = (
        models_dir / "huggingface" / "hub" / "models--convaiinnovations--laya-multilingual" / "blobs"
    )
    incomplete_blob = laya_blobs_dir / "abc123.incomplete"

    def hook_bge(target: Path) -> None:
        attempts["bge_m3"] = attempts.get("bge_m3", 0) + 1
        snap = target / "models--BAAI--bge-m3" / "snapshots" / "rev1"
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "config.json").write_text('{"ok": true}', encoding="utf-8")

    def hook_laya(target: Path) -> None:
        attempts["laya_multilingual"] = attempts.get("laya_multilingual", 0) + 1
        if attempts["laya_multilingual"] == 1:
            laya_blobs_dir.mkdir(parents=True, exist_ok=True)
            incomplete_blob.write_bytes(b"partial-hf-blob")
            raise ConnectionError("Simulated transient network drop")
        assert incomplete_blob.is_file(), "Partial HF blob must not be deleted on failure!"
        incomplete_blob.unlink()
        snap = target / "models--convaiinnovations--laya-multilingual" / "snapshots" / "rev1"
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "model.safetensors").write_bytes(b"laya-weights")

    def hook_fastembed(target: Path) -> None:
        attempts["fastembed_bm25"] = attempts.get("fastembed_bm25", 0) + 1
        snap = target / "models--Qdrant--bm25" / "snapshots" / "rev1"
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "config.json").write_text('{"ok": true}', encoding="utf-8")

    def hook_flashrank(target: Path) -> None:
        attempts["flashrank"] = attempts.get("flashrank", 0) + 1
        out = target / "ms-marco-MultiBERT-L-12"
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.onnx").write_bytes(b"onnx")

    def hook_rapidocr(target: Path) -> None:
        attempts["rapidocr"] = attempts.get("rapidocr", 0) + 1
        target.mkdir(parents=True, exist_ok=True)
        (target / "det.onnx").write_bytes(b"det")

    def hook_docling(target: Path) -> None:
        attempts["docling_layout_tableformer"] = attempts.get("docling_layout_tableformer", 0) + 1
        (target / "docling-project--docling-layout-heron").mkdir(parents=True, exist_ok=True)
        (target / "docling-project--docling-layout-heron" / "layout.onnx").write_bytes(b"layout")
        (target / "docling-project--docling-models").mkdir(parents=True, exist_ok=True)
        (target / "docling-project--docling-models" / "tf.bin").write_bytes(b"tf")

    service = RuntimeBootstrapService(
        paths=paths,
        download_hooks={
            "bge_m3": hook_bge,
            "laya_multilingual": hook_laya,
            "fastembed_bm25": hook_fastembed,
            "flashrank": hook_flashrank,
            "rapidocr": hook_rapidocr,
            "docling_layout_tableformer": hook_docling,
        },
    )

    with pytest.raises(ConnectionError, match="transient network drop"):
        service.ensure_models_ready(progress_cb=events.append)

    assert incomplete_blob.is_file()
    assert not (models_dir / "models_ready.json").exists()
    assert service.inspect_models_status()["models_ready"] is False
    assert events[-1].error is not None
    assert events[-1].models["bge_m3"] == "cached"
    assert events[-1].models["laya_multilingual"] == "failed"

    # Retry skips already-cached bge_m3 and resumes from laya_multilingual through all 6
    resumed = service.ensure_models_ready(progress_cb=events.append)
    assert resumed["models_ready"] is True
    assert attempts["bge_m3"] == 1
    assert attempts["laya_multilingual"] == 2
    assert attempts["docling_layout_tableformer"] == 1
    assert (models_dir / "models_ready.json").is_file()


def test_import_local_models_dir_populates_cache_and_marks_ready(tmp_path: Path):
    paths = _make_staged_paths(tmp_path)
    offline_bundle = tmp_path / "usb_models"
    _populate_all_six_model_caches(offline_bundle, write_manifest=False)

    service = RuntimeBootstrapService(paths=paths)
    assert service.inspect_models_status()["models_ready"] is False

    imported_status = service.import_local_models_dir(offline_bundle)
    assert imported_status["models_ready"] is True
    assert len(imported_status["models"]) == 6
    assert all(m.cached for m in imported_status["models"])
    assert (paths.models_dir / "models_ready.json").is_file()
    assert load_desktop_config(paths).models_ready is True

    with pytest.raises(FileNotFoundError):
        service.import_local_models_dir(tmp_path / "does_not_exist")


# ==============================================================================
# R2.4 — Hardware, Bootstrap Status & Compute Profile API Routes
# ==============================================================================


async def _build_isolated_api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qdrant: AsyncQdrantClient):
    desktop_home = tmp_path / "desktop_home"
    data_dir = desktop_home / "data"
    db_dir = data_dir / "db"
    dist_dir = tmp_path / "frontend_dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "index.html").write_text("<html><body>Desktop UI</body></html>", encoding="utf-8")

    monkeypatch.setenv("SECOND_BRAIN_DESKTOP_HOME", str(desktop_home))
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("DB_DIR", str(db_dir))
    monkeypatch.setenv("FRONTEND_DIST_DIR", str(dist_dir))
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("INGESTION_USE_PROCESS_POOL", "false")
    monkeypatch.setenv("INGESTION_ENABLE_LLM_CLASSIFICATION_FALLBACK", "false")

    get_settings.cache_clear()
    set_embedder(DeterministicStubEmbedder(dimensions=32))
    set_sparse_encoder(DeterministicBM25Encoder())
    set_reranker(None)
    set_active_qdrant_client(qdrant)

    app = create_app()
    app.state.qdrant = qdrant
    return app, desktop_home


def _cpu_only_report() -> HardwareReport:
    return HardwareReport(
        cpu_name="Intel Core i7-13700",
        logical_cores=16,
        physical_cores=8,
        total_ram_gb=32.0,
        available_ram_gb=20.0,
        ram_status="optimal",
        gpus=[GpuAdapterInfo(name="Intel(R) UHD Graphics 770", vram_gb=1.0, vendor="Intel")],
        has_nvidia_cuda_capable_gpu=False,
        gpu_status="cpu_only",
        target_drive="C:",
        free_disk_gb=80.0,
        disk_status="optimal",
        cuda_disk_sufficient=True,
        is_onedrive_path=False,
        onedrive_warning=None,
        recommended_profile="cpu",
        recommended_upload_mode="auto",
        recommendation_reason="No discrete NVIDIA GPU detected; Standard CPU mode is recommended.",
    )


def _rtx_8gb_report() -> HardwareReport:
    return HardwareReport(
        cpu_name="AMD Ryzen 7 7800X3D",
        logical_cores=16,
        physical_cores=8,
        total_ram_gb=32.0,
        available_ram_gb=24.0,
        ram_status="optimal",
        gpus=[
            GpuAdapterInfo(
                name="NVIDIA GeForce RTX 4060",
                vram_gb=8.0,
                vendor="NVIDIA",
                driver_version="560.94",
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
        is_onedrive_path=False,
        onedrive_warning=None,
        recommended_profile="cuda",
        recommended_upload_mode="auto",
        recommendation_reason="Detected NVIDIA GPU 'NVIDIA GeForce RTX 4060' with 8.0 GB VRAM (>= 4.0 GB).",
    )


async def test_detect_collection_sparse_mode_and_system_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    qdrant = AsyncQdrantClient(location=":memory:")
    app, desktop_home = await _build_isolated_api_client(tmp_path, monkeypatch, qdrant)
    paths = DesktopPaths(user_data_dir=desktop_home)

    try:
        async with LifespanManager(app) as manager:
            transport = ASGITransport(app=manager.app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                # Collection exists but is empty -> sparse mode None; unreachable/None also return None
                assert await detect_collection_sparse_mode(None, "second_brain_knowledge") is None
                broken_qdrant = MagicMock()
                broken_qdrant.collection_exists.side_effect = ConnectionError("Qdrant down")
                assert await detect_collection_sparse_mode(broken_qdrant, "second_brain_knowledge") is None
                assert await detect_collection_sparse_mode(qdrant, "second_brain_knowledge") is None

                # Seed one point indexed with sparse_mode="bm25"
                await qdrant.upsert(
                    collection_name="second_brain_knowledge",
                    points=[
                        models.PointStruct(
                            id="11111111-1111-1111-1111-111111111111",
                            vector={"dense": [0.1] * 32, "sparse": models.SparseVector(indices=[1], values=[1.0])},
                            payload={"doc_id": "d1", "sparse_mode": "bm25"},
                        )
                    ],
                )
                assert await detect_collection_sparse_mode(qdrant, "second_brain_knowledge") == "bm25"

                # 1. GET /api/v1/system/hardware on CPU-only host
                with patch("src.api.routes.system.probe_system_hardware", return_value=_cpu_only_report()):
                    hw_cpu = await client.get("/api/v1/system/hardware")
                assert hw_cpu.status_code == 200
                hw_cpu_data = hw_cpu.json()
                assert hw_cpu_data["recommended_profile"] == "cpu"
                assert hw_cpu_data["has_nvidia_cuda_capable_gpu"] is False
                assert hw_cpu_data["active_compute_profile"] == "cpu"
                assert hw_cpu_data["collection_sparse_mode"] == "bm25"
                assert hw_cpu_data["sparse_mode_pinned"] is False

                # 2. POST /api/v1/system/compute-profile rejects CUDA on CPU-only host OR low disk unless force=True
                with patch("src.api.routes.system.probe_system_hardware", return_value=_cpu_only_report()):
                    bad_cuda = await client.post(
                        "/api/v1/system/compute-profile",
                        json={"profile": "cuda", "force": False},
                    )
                assert bad_cuda.status_code == 400
                assert load_desktop_config(paths).compute_profile == "cpu"

                low_disk_cuda = _rtx_8gb_report().model_copy(
                    update={"free_disk_gb": 9.5, "disk_status": "limited", "cuda_disk_sufficient": False}
                )
                with patch("src.api.routes.system.probe_system_hardware", return_value=low_disk_cuda):
                    bad_disk_cuda = await client.post(
                        "/api/v1/system/compute-profile",
                        json={"profile": "cuda", "force": False},
                    )
                assert bad_disk_cuda.status_code == 400
                assert "12.0 GB" in bad_disk_cuda.json()["detail"]
                assert load_desktop_config(paths).compute_profile == "cpu"

                # Force override on CPU-only host succeeds
                with patch("src.api.routes.system.probe_system_hardware", return_value=_cpu_only_report()):
                    forced_cuda = await client.post(
                        "/api/v1/system/compute-profile",
                        json={"profile": "cuda", "force": True},
                    )
                assert forced_cuda.status_code == 200
                assert forced_cuda.json()["compute_profile"] == "cuda"
                assert forced_cuda.json()["sparse_mode_override"] == "bm25"
                assert forced_cuda.json()["wheel_install_required"] is True
                assert forced_cuda.json()["restart_required"] is True

                # Reset back to CPU before testing RTX 8 GB transition
                back_cpu = await client.post("/api/v1/system/compute-profile", json={"profile": "cpu"})
                assert back_cpu.status_code == 200
                assert back_cpu.json()["compute_profile"] == "cpu"

                # 3. RTX 8 GB host -> GET hardware + POST compute-profile pins bm25 then opts into reindex_sparse
                with patch("src.api.routes.system.probe_system_hardware", return_value=_rtx_8gb_report()):
                    hw_rtx = await client.get("/api/v1/system/hardware")
                    assert hw_rtx.status_code == 200
                    assert hw_rtx.json()["recommended_profile"] == "cuda"
                    assert hw_rtx.json()["has_nvidia_cuda_capable_gpu"] is True

                    sw_cuda = await client.post(
                        "/api/v1/system/compute-profile",
                        json={"profile": "cuda"},
                    )
                assert sw_cuda.status_code == 200
                sw_data = sw_cuda.json()
                assert sw_data["compute_profile"] == "cuda"
                assert sw_data["previous_profile"] == "cpu"
                assert sw_data["collection_sparse_mode"] == "bm25"
                assert sw_data["sparse_mode_override"] == "bm25"
                assert sw_data["reindex_recommended"] is True
                assert sw_data["wheel_install_required"] is True
                assert sw_data["restart_required"] is True

                # Opt into reindex_sparse=True -> unpins to "auto"
                with patch("src.api.routes.system.probe_system_hardware", return_value=_rtx_8gb_report()):
                    sw_reindex = await client.post(
                        "/api/v1/system/compute-profile",
                        json={"profile": "cuda", "reindex_sparse": True},
                    )
                assert sw_reindex.status_code == 200
                assert sw_reindex.json()["sparse_mode_override"] == "auto"
                assert sw_reindex.json()["reindex_recommended"] is False

                # 4. GET /api/v1/system/bootstrap-status before and after models ready
                bs_before = await client.get("/api/v1/system/bootstrap-status")
                assert bs_before.status_code == 200
                bs_before_data = bs_before.json()
                assert bs_before_data["models_ready"] is False
                assert len(bs_before_data["models"]) == 6
                assert {m["key"] for m in bs_before_data["models"]} == {s.key for s in MODEL_SPECS}

                _populate_all_six_model_caches(paths.models_dir, write_manifest=True)
                app.state.bootstrap_progress = BootstrapProgress(
                    step="complete",
                    percent=100.0,
                    detail="Verified",
                    is_running=False,
                    completed=True,
                )

                bs_after = await client.get("/api/v1/system/bootstrap-status")
                assert bs_after.status_code == 200
                bs_after_data = bs_after.json()
                assert bs_after_data["models_ready"] is True
                assert bs_after_data["compute_profile"] == "cuda"
                assert all(m["cached"] for m in bs_after_data["models"])
                assert bs_after_data["progress"]["completed"] is True

                # 5. POST /api/v1/system/reindex queues ready documents with parsed checkpoints
                from src.db.models import Document

                async with app.state.session_factory() as db_sess:
                    ready_doc = Document(
                        id="doc-reindex-1",
                        filename="indexed.pdf",
                        file_type="pdf",
                        sha256="a" * 64,
                        status="ready",
                    )
                    db_sess.add(ready_doc)
                    await db_sess.commit()

                parsed_dir = paths.resolve_data_dir() / "parsed"
                md_dir = paths.resolve_data_dir() / "markdown"
                parsed_dir.mkdir(parents=True, exist_ok=True)
                md_dir.mkdir(parents=True, exist_ok=True)
                (parsed_dir / f"{'a' * 64}.json").write_text(
                    json.dumps({"docling_document": None, "extra_meta": {"engine": "pymupdf4llm"}}),
                    encoding="utf-8",
                )
                (md_dir / f"{'a' * 64}.md").write_text("# Indexed\n\nBody text.", encoding="utf-8")

                orch = app.state.orchestrator
                with patch.object(orch, "complete_approved_ingestion", return_value=ready_doc) as mock_complete:
                    reindex_resp = await client.post("/api/v1/system/reindex")
                    assert reindex_resp.status_code == 200
                    reindex_data = reindex_resp.json()
                    assert reindex_data["queued_count"] == 1
                    assert reindex_data["skipped_count"] == 0
                    await asyncio.sleep(0)
                    mock_complete.assert_called_once()
    finally:
        await qdrant.close()
        set_embedder(None)
        set_sparse_encoder(None)
        set_reranker(None)
        set_active_qdrant_client(None)
        await close_checkpointer()
        get_settings.cache_clear()
