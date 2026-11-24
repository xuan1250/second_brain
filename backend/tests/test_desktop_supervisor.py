"""Unit and lifecycle tests for Phase 1 desktop hardware, config, Win32 safety, and sidecar supervisor."""
from __future__ import annotations

import io
import json
import socket
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.api.routes.system import _get_process_rss_mb
from src.desktop import (
    ControlChannelServer,
    DesktopConfig,
    DesktopPaths,
    GpuAdapterInfo,
    HardwareReport,
    SidecarSupervisor,
    SingleInstanceMutex,
    WindowsJobObject,
    find_free_loopback_port,
    is_mutex_held,
    load_desktop_config,
    probe_system_hardware,
    recommend_compute_profile,
    save_desktop_config,
    send_control_command,
)
from src.desktop.hardware import _check_onedrive_path, _probe_gpus


# ==============================================================================
# Hardware Recommendations & Probes
# ==============================================================================


def test_hardware_recommendation_cpu_only_optimal():
    intel = GpuAdapterInfo(name="Intel(R) UHD Graphics 770", vram_gb=1.0, vendor="Intel")
    rec = recommend_compute_profile(total_ram_gb=16.0, gpus=[intel], free_disk_gb=40.0)

    assert rec.ram_status == "optimal"
    assert rec.recommended_upload_mode == "auto"
    assert rec.has_nvidia_cuda_capable_gpu is False
    assert rec.gpu_status == "cpu_only"
    assert rec.disk_status == "optimal"
    assert rec.cuda_disk_sufficient is True
    assert rec.recommended_profile == "cpu"
    assert "Intel(R) UHD Graphics 770" in rec.recommendation_reason
    assert rec.blockers == []
    assert rec.warnings == []


def test_hardware_recommendation_ram_thresholds():
    # < 8 GB RAM -> blocker + fast upload
    low = recommend_compute_profile(total_ram_gb=7.5, gpus=[], free_disk_gb=30.0)
    assert low.ram_status == "insufficient"
    assert low.recommended_upload_mode == "fast"
    assert any("8.0 GB" in b for b in low.blockers)

    # 8 - 15.9 GB RAM -> warning + fast upload
    mid = recommend_compute_profile(total_ram_gb=12.0, gpus=[], free_disk_gb=30.0)
    assert mid.ram_status == "limited"
    assert mid.recommended_upload_mode == "fast"
    assert mid.blockers == []
    assert any("Fast upload" in w for w in mid.warnings)

    # >= 16 GB RAM -> optimal + auto upload
    high = recommend_compute_profile(total_ram_gb=32.0, gpus=[], free_disk_gb=30.0)
    assert high.ram_status == "optimal"
    assert high.recommended_upload_mode == "auto"


def test_hardware_recommendation_nvidia_cuda_and_disk_thresholds():
    rtx = GpuAdapterInfo(
        name="NVIDIA GeForce RTX 4060",
        vram_gb=8.0,
        vendor="NVIDIA",
        driver_version="560.94",
        is_nvidia=True,
        cuda_capable=True,
    )

    # Sufficient VRAM + sufficient CUDA disk (>= 12 GB) -> cuda
    rec_cuda = recommend_compute_profile(total_ram_gb=16.0, gpus=[rtx], free_disk_gb=20.0)
    assert rec_cuda.has_nvidia_cuda_capable_gpu is True
    assert rec_cuda.gpu_status == "cuda_compatible"
    assert rec_cuda.recommended_profile == "cuda"
    assert rec_cuda.cuda_disk_sufficient is True
    assert rec_cuda.disk_status == "optimal"

    # Sufficient VRAM + 10 GB disk (< 12 GB) -> falls back to cpu with clear explanation
    rec_disk_constrained = recommend_compute_profile(
        total_ram_gb=16.0,
        gpus=[rtx],
        free_disk_gb=10.0,
    )
    assert rec_disk_constrained.has_nvidia_cuda_capable_gpu is True
    assert rec_disk_constrained.gpu_status == "cuda_compatible"
    assert rec_disk_constrained.cuda_disk_sufficient is False
    assert rec_disk_constrained.disk_status == "limited"
    assert rec_disk_constrained.recommended_profile == "cpu"
    assert "12.0 GB" in rec_disk_constrained.recommendation_reason

    # 13 GB disk -> limited disk warning, but cuda_disk_sufficient=True
    rec_13 = recommend_compute_profile(total_ram_gb=16.0, gpus=[rtx], free_disk_gb=13.0)
    assert rec_13.disk_status == "limited"
    assert rec_13.cuda_disk_sufficient is True
    assert rec_13.recommended_profile == "cuda"

    # < 8 GB disk -> insufficient blocker
    rec_tiny_disk = recommend_compute_profile(total_ram_gb=16.0, gpus=[], free_disk_gb=5.0)
    assert rec_tiny_disk.disk_status == "insufficient"
    assert rec_tiny_disk.cuda_disk_sufficient is False
    assert any("8.0 GB" in b for b in rec_tiny_disk.blockers)

    # NVIDIA < 4 GB VRAM -> cpu
    low_vram = GpuAdapterInfo(
        name="NVIDIA GeForce MX450",
        vram_gb=2.0,
        vendor="NVIDIA",
        is_nvidia=True,
        cuda_capable=False,
    )
    rec_low_vram = recommend_compute_profile(total_ram_gb=16.0, gpus=[low_vram], free_disk_gb=30.0)
    assert rec_low_vram.has_nvidia_cuda_capable_gpu is False
    assert rec_low_vram.gpu_status == "nvidia_low_vram"
    assert rec_low_vram.recommended_profile == "cpu"
    assert "4.0 GB" in rec_low_vram.recommendation_reason


def test_onedrive_path_detection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    od_root = tmp_path / "CloudSync"
    od_root.mkdir()
    monkeypatch.setenv("OneDrive", str(od_root))
    monkeypatch.delenv("OneDriveCommercial", raising=False)
    monkeypatch.delenv("OneDriveConsumer", raising=False)

    flagged, msg = _check_onedrive_path(od_root / "SecondBrain" / "data")
    assert flagged is True
    assert msg is not None and "OneDrive" in msg

    # Sibling folder sharing prefix name must not false-positive
    sibling = tmp_path / "CloudSyncExtra" / "SecondBrain"
    flagged_sibling, _ = _check_onedrive_path(sibling)
    assert flagged_sibling is False

    # Direct segment match even without env var
    monkeypatch.delenv("OneDrive", raising=False)
    flagged_seg, msg_seg = _check_onedrive_path(tmp_path / "OneDrive - Acme" / "SecondBrain")
    assert flagged_seg is True
    assert msg_seg is not None

    flagged_plain, _ = _check_onedrive_path(tmp_path / "OneDrive" / "SecondBrain")
    assert flagged_plain is True

    # Clean non-OneDrive local path (even with "onedrive" substring in an unrelated word)
    flagged_clean, msg_clean = _check_onedrive_path(tmp_path / "non_onedrive" / "SecondBrain")
    assert flagged_clean is False
    assert msg_clean is None


def test_probe_gpus_merges_without_doubling_nvidia():
    nv = GpuAdapterInfo(
        name="NVIDIA GeForce RTX 4070",
        vram_gb=12.0,
        vendor="NVIDIA",
        driver_version="561.0",
        is_nvidia=True,
        cuda_capable=True,
    )
    wmi_nv = GpuAdapterInfo(
        name="NVIDIA GeForce RTX 4070",
        vram_gb=4.0,
        vendor="NVIDIA",
        is_nvidia=True,
        cuda_capable=True,
    )
    wmi_intel = GpuAdapterInfo(
        name="Intel(R) Iris(R) Xe Graphics",
        vram_gb=1.0,
        vendor="Intel",
        is_nvidia=False,
        cuda_capable=False,
    )
    with (
        patch("src.desktop.hardware._probe_nvidia_smi", return_value=[nv]),
        patch("src.desktop.hardware._probe_wmi_gpus", return_value=[wmi_nv, wmi_intel]),
    ):
        gpus = _probe_gpus()

    assert len(gpus) == 2
    assert gpus[0].name == "NVIDIA GeForce RTX 4070"
    assert gpus[0].vram_gb == 12.0
    assert gpus[1].name == "Intel(R) Iris(R) Xe Graphics"


def test_probe_system_hardware_smoke(tmp_path: Path):
    report = probe_system_hardware(data_dir=tmp_path)
    assert isinstance(report, HardwareReport)
    assert report.logical_cores >= 1
    assert report.total_ram_gb > 0.0
    assert report.free_disk_gb >= 0.0
    assert report.recommended_profile in ("cpu", "cuda")


# ==============================================================================
# DesktopPaths, DesktopConfig, Port Allocation
# ==============================================================================


def test_desktop_paths_staged_vs_dev_and_env_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    custom_home = tmp_path / "custom_home"
    monkeypatch.setenv("SECOND_BRAIN_DESKTOP_HOME", str(custom_home))

    dev_paths = DesktopPaths()
    assert dev_paths.user_data_dir == custom_home.resolve()
    assert dev_paths.is_staged_install is False
    assert dev_paths.backend_dir == dev_paths.app_dir / "backend"
    assert dev_paths.frontend_dist_dir == dev_paths.app_dir / "frontend" / "dist"

    staged_app = tmp_path / "staged_install"
    (staged_app / "app" / "backend").mkdir(parents=True)
    (staged_app / "app" / "frontend" / "dist").mkdir(parents=True)
    (staged_app / "bin").mkdir(parents=True)
    (staged_app / "venv" / "Scripts").mkdir(parents=True)
    (staged_app / "venv" / "Scripts" / "python.exe").write_text("", encoding="utf-8")

    staged_paths = DesktopPaths(app_dir=staged_app, user_data_dir=custom_home)
    assert staged_paths.is_staged_install is True
    assert staged_paths.backend_dir == staged_app.resolve() / "app" / "backend"
    assert staged_paths.frontend_dist_dir == staged_app.resolve() / "app" / "frontend" / "dist"
    assert staged_paths.qdrant_exe.name.startswith("qdrant")
    assert staged_paths.uv_exe.name.startswith("uv")

    # CPU resolution -> staged base venv python
    assert staged_paths.resolve_backend_python("cpu") == staged_paths.base_venv_python

    # CUDA resolution falls back to base until cuda venv exe exists and cuda_runtime_ready=True
    assert staged_paths.resolve_backend_python("cuda", cuda_runtime_ready=True) == staged_paths.base_venv_python

    cuda_py = staged_paths.cuda_venv_python
    cuda_py.parent.mkdir(parents=True, exist_ok=True)
    cuda_py.write_text("", encoding="utf-8")
    assert staged_paths.resolve_backend_python("cuda", cuda_runtime_ready=False) == staged_paths.base_venv_python
    assert staged_paths.resolve_backend_python("cuda", cuda_runtime_ready=True) == cuda_py


def test_desktop_paths_directories_and_uninstall_targets(tmp_path: Path):
    user_home = tmp_path / "SecondBrainHome"
    custom_data = tmp_path / "ExternalData"
    paths = DesktopPaths(app_dir=tmp_path / "app", user_data_dir=user_home)
    cfg = DesktopConfig(data_dir_override=str(custom_data))
    save_desktop_config(cfg, paths)
    paths.ensure_directories(cfg)

    for sub in ("raw", "parsed", "markdown", "models", "db", "qdrant_storage"):
        assert (custom_data / sub).is_dir()
    for user_sub in ("runtime", "webview2_cache", "config", "logs"):
        assert (user_home / user_sub).is_dir()

    # Partial uninstall (models only, keep user data)
    model_only_targets = paths.iter_uninstall_targets(
        include_models=True,
        include_user_data=False,
        config=cfg,
    )
    assert paths.runtime_dir in model_only_targets
    assert paths.webview2_cache_dir in model_only_targets
    assert paths.logs_dir in model_only_targets
    assert custom_data / "models" in model_only_targets
    assert user_home not in model_only_targets

    # Full purge discovers persisted data_dir_override from desktop_config.json even if config=None
    purge_targets = paths.iter_uninstall_targets(include_models=True, include_user_data=True)
    assert custom_data.resolve() in purge_targets
    assert paths.config_dir in purge_targets
    assert user_home.resolve() in purge_targets
    assert len(purge_targets) == len({str(p).lower() for p in purge_targets})


def test_desktop_config_round_trip_and_corrupt_recovery(tmp_path: Path):
    paths = DesktopPaths(app_dir=tmp_path, user_data_dir=tmp_path / "user")
    assert load_desktop_config(paths) == DesktopConfig()

    cfg = DesktopConfig(
        compute_profile="cuda",
        sparse_mode_override="bm25",
        models_ready=True,
        cuda_runtime_ready=True,
        preferred_port=8123,
        preferred_qdrant_port=6444,
        show_tray_hint=False,
    )
    saved_path = save_desktop_config(cfg, paths)
    assert saved_path == paths.config_path
    assert load_desktop_config(paths) == cfg

    # Corrupt JSON recovers gracefully to defaults
    paths.config_path.write_text("{not valid json", encoding="utf-8")
    assert load_desktop_config(paths) == DesktopConfig()


def test_find_free_loopback_port_collision_fallback():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        busy_port = int(occupied.getsockname()[1])

        fallback_port = find_free_loopback_port(busy_port)
        assert fallback_port != busy_port
        assert 1 <= fallback_port <= 65535

    # When preferred port is free, it should return the preferred port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = int(probe.getsockname()[1])
    assert find_free_loopback_port(free_port) == free_port


# ==============================================================================
# Win32 Job Object, SingleInstanceMutex & Control Channel
# ==============================================================================


def test_windows_job_object_and_process_termination():
    if sys.platform == "win32":
        with WindowsJobObject() as job:
            assert job.is_active is True
            proc = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                creationflags=0x08000000,
            )
            assigned = job.assign_process(proc)
            assert assigned is True
            assert proc.poll() is None

        # Exiting context closes the Job Object, which kills the assigned child
        ret = proc.wait(timeout=5.0)
        assert ret is not None
    else:
        with WindowsJobObject() as job:
            assert job.is_active is False
            assert job.assign_process(MagicMock()) is False


def test_single_instance_mutex_exclusivity():
    if sys.platform != "win32":
        mutex = SingleInstanceMutex(name=f"Local\\SecondBrainTest_{uuid.uuid4().hex}")
        assert mutex.acquire() is True
        assert is_mutex_held(mutex.name) is False
        mutex.release()
        return

    mutex_name = f"Local\\SecondBrainTest_{uuid.uuid4().hex}"
    assert is_mutex_held(mutex_name) is False

    primary = SingleInstanceMutex(name=mutex_name)
    assert primary.acquire() is True
    assert primary.is_primary is True
    assert is_mutex_held(mutex_name) is True

    secondary = SingleInstanceMutex(name=mutex_name)
    assert secondary.acquire() is False
    assert secondary.already_exists is True
    assert secondary.is_primary is False

    primary.release()
    assert is_mutex_held(mutex_name) is False


def test_control_channel_server_auth_and_callbacks(tmp_path: Path):
    paths = DesktopPaths(app_dir=tmp_path, user_data_dir=tmp_path / "user")
    events: list[str] = []

    server = ControlChannelServer(
        paths,
        on_focus=lambda: events.append("FOCUS"),
        on_quit=lambda: events.append("QUIT"),
    )
    port = server.start()
    try:
        assert port > 0
        assert paths.control_file_path.is_file()
        meta = json.loads(paths.control_file_path.read_text(encoding="utf-8"))
        assert meta["port"] == port
        assert len(meta["token"]) == 32

        # Invalid token must be rejected
        assert send_control_command(paths, "FOCUS", token_override="wrong-token") is False
        assert events == []

        # Unknown command rejected
        assert send_control_command(paths, "NOOP") is False
        assert events == []

        # Valid authenticated commands
        assert send_control_command(paths, "FOCUS") is True
        assert send_control_command(paths, "QUIT") is True
        assert events == ["FOCUS", "QUIT"]
    finally:
        server.stop()

    assert not paths.control_file_path.exists()


# ==============================================================================
# SidecarSupervisor Env Injection & Lifecycle
# ==============================================================================


def test_sidecar_supervisor_env_injection_cpu_cuda_and_docling(tmp_path: Path):
    app_dir = tmp_path / "staged"
    (app_dir / "app" / "backend").mkdir(parents=True)
    (app_dir / "app" / "frontend" / "dist").mkdir(parents=True)
    paths = DesktopPaths(app_dir=app_dir, user_data_dir=tmp_path / "user")

    # CPU config without docling/models directory
    cpu_cfg = DesktopConfig(compute_profile="cpu")
    sup_cpu = SidecarSupervisor(paths, cpu_cfg, api_port=8111, qdrant_port=6399)

    q_env = sup_cpu.build_qdrant_env()
    assert q_env["QDRANT__SERVICE__HOST"] == "127.0.0.1"
    assert q_env["QDRANT__SERVICE__HTTP_PORT"] == "6399"
    assert q_env["QDRANT__SERVICE__GRPC_PORT"] == "0"
    assert q_env["QDRANT__STORAGE__STORAGE_PATH"] == str(paths.qdrant_storage_dir)
    assert q_env["QDRANT__TELEMETRY_DISABLED"] == "true"

    b_env_cpu = sup_cpu.build_backend_env()
    assert b_env_cpu["SECOND_BRAIN_DESKTOP_HOME"] == str(paths.user_data_dir)
    assert b_env_cpu["SECOND_BRAIN_APP_DIR"] == str(paths.app_dir)
    assert b_env_cpu["NO_PROXY"] == "127.0.0.1,localhost"
    assert b_env_cpu["DATA_DIR"] == str(paths.data_dir)
    assert b_env_cpu["DB_DIR"] == str(paths.db_dir)
    assert b_env_cpu["FRONTEND_DIST_DIR"] == str(paths.frontend_dist_dir)
    assert b_env_cpu["QDRANT_URL"] == "http://127.0.0.1:6399"
    assert b_env_cpu["CORS_ORIGINS"] == "http://127.0.0.1:8111,http://localhost:8111"
    assert b_env_cpu["EMBEDDING_DEVICE"] == "cpu"
    assert b_env_cpu["INGESTION_DEVICE"] == "cpu"
    assert b_env_cpu["EMBEDDING_FP16"] == "false"
    assert b_env_cpu["EMBEDDING_SPARSE_MODE"] == "bm25"
    assert b_env_cpu["FLASHRANK_ALLOW_DOWNLOAD"] == "true"
    assert b_env_cpu["PYTHONPATH"] == str(paths.backend_dir)
    assert "DOCLING_ARTIFACTS_PATH" not in b_env_cpu

    # Empty docling/models dir must still NOT inject DOCLING_ARTIFACTS_PATH
    docling_models = paths.models_dir / "docling" / "models"
    docling_models.mkdir(parents=True)
    assert "DOCLING_ARTIFACTS_PATH" not in sup_cpu.build_backend_env()

    # Non-empty docling/models dir injects DOCLING_ARTIFACTS_PATH
    (docling_models / "model.safetensors").write_bytes(b"ok")
    assert sup_cpu.build_backend_env()["DOCLING_ARTIFACTS_PATH"] == str(docling_models)

    # CUDA config without sparse_mode_override -> sparse_mode="auto"
    cuda_cfg = DesktopConfig(compute_profile="cuda", cuda_runtime_ready=True)
    sup_cuda = SidecarSupervisor(paths, cuda_cfg, api_port=8112, qdrant_port=6398)
    b_env_cuda = sup_cuda.build_backend_env()
    assert b_env_cuda["EMBEDDING_DEVICE"] == "cuda"
    assert b_env_cuda["INGESTION_DEVICE"] == "cuda"
    assert b_env_cuda["EMBEDDING_FP16"] == "true"
    assert b_env_cuda["EMBEDDING_SPARSE_MODE"] == "auto"

    # CUDA config without cuda_runtime_ready must keep CPU device/fp16 matching base interpreter
    unready_cuda_cfg = DesktopConfig(compute_profile="cuda", cuda_runtime_ready=False)
    sup_unready = SidecarSupervisor(paths, unready_cuda_cfg, api_port=8114, qdrant_port=6396)
    b_env_unready = sup_unready.build_backend_env()
    assert b_env_unready["EMBEDDING_DEVICE"] == "cpu"
    assert b_env_unready["INGESTION_DEVICE"] == "cpu"
    assert b_env_unready["EMBEDDING_FP16"] == "false"

    # CUDA config with sparse_mode_override="bm25" -> honors pinned bm25
    pinned_cfg = DesktopConfig(
        compute_profile="cuda",
        cuda_runtime_ready=True,
        sparse_mode_override="bm25",
    )
    sup_pinned = SidecarSupervisor(paths, pinned_cfg, api_port=8113, qdrant_port=6397)
    assert sup_pinned.build_backend_env()["EMBEDDING_SPARSE_MODE"] == "bm25"


def _make_mock_http_ok() -> MagicMock:
    resp = MagicMock()
    resp.status = 200
    resp.getcode.return_value = 200
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _make_mock_popen_proc(log_line: str = "ready\n") -> MagicMock:
    proc = MagicMock()
    proc.pid = 4242
    proc.poll.return_value = None
    proc.wait.return_value = 0
    proc.stdout = io.StringIO(log_line)
    proc._handle = None
    return proc


def test_sidecar_supervisor_lifecycle_and_ordered_shutdown(tmp_path: Path):
    app_dir = tmp_path / "staged"
    (app_dir / "app" / "backend").mkdir(parents=True)
    (app_dir / "app" / "frontend" / "dist").mkdir(parents=True)
    (app_dir / "venv" / "Scripts").mkdir(parents=True)
    (app_dir / "venv" / "Scripts" / "python.exe").write_text("", encoding="utf-8")

    paths = DesktopPaths(app_dir=app_dir, user_data_dir=tmp_path / "user")
    cfg = DesktopConfig(compute_profile="cpu")
    supervisor = SidecarSupervisor(
        paths,
        cfg,
        api_port=18000,
        qdrant_port=16333,
        startup_timeout_sec=5.0,
        poll_interval_sec=0.01,
    )

    qdrant_proc = _make_mock_popen_proc("qdrant ready\n")
    backend_proc_1 = _make_mock_popen_proc("uvicorn ready\n")
    backend_proc_2 = _make_mock_popen_proc("uvicorn restarted\n")

    lifecycle_events: list[str] = []
    qdrant_proc.terminate.side_effect = lambda: lifecycle_events.append("terminate_qdrant")
    backend_proc_1.terminate.side_effect = lambda: lifecycle_events.append("terminate_backend_1")
    backend_proc_2.terminate.side_effect = lambda: lifecycle_events.append("terminate_backend_2")
    # Simulate wait timeout on second backend stop to verify kill() escalation
    backend_proc_2.wait.side_effect = subprocess.TimeoutExpired("uvicorn", 1.0)
    backend_proc_2.kill.side_effect = lambda: lifecycle_events.append("kill_backend_2")

    spawned_procs = [qdrant_proc, backend_proc_1, backend_proc_2]
    popen_calls: list[dict[str, Any]] = []

    def fake_popen(cmd: list[str], **kwargs: Any) -> MagicMock:
        popen_calls.append({"cmd": cmd, "kwargs": kwargs})
        return spawned_procs.pop(0)

    mock_job = MagicMock()
    with (
        patch.object(supervisor, "_job_object", mock_job),
        patch.object(supervisor, "_ensure_job_object", return_value=mock_job),
        patch("src.desktop.supervisor.subprocess.Popen", side_effect=fake_popen),
        patch("src.desktop.supervisor.urllib.request.urlopen", return_value=_make_mock_http_ok()),
    ):
        supervisor.start()
        assert supervisor.is_qdrant_running is True
        assert supervisor.is_backend_running is True
        assert mock_job.assign_process.call_count == 2

        if sys.platform == "win32":
            assert popen_calls[0]["kwargs"]["creationflags"] == 0x08000000
            assert popen_calls[1]["kwargs"]["creationflags"] == 0x08000000

        # Restart backend only (keeps Qdrant running)
        supervisor.restart_backend(DesktopConfig(compute_profile="cpu", sparse_mode_override="bm25"))
        assert "terminate_backend_1" in lifecycle_events
        assert "terminate_qdrant" not in lifecycle_events
        assert len(popen_calls) == 3

        # Ordered shutdown stops uvicorn BEFORE qdrant and closes job object
        supervisor.shutdown(timeout_sec=0.5)
        assert lifecycle_events.index("terminate_backend_2") < lifecycle_events.index("terminate_qdrant")
        assert "kill_backend_2" in lifecycle_events
        mock_job.close.assert_called_once()

    assert (paths.logs_dir / "launcher.log").is_file()
    assert (paths.logs_dir / "qdrant.log").is_file()
    assert (paths.logs_dir / "backend.log").is_file()


def test_terminate_process_skips_pipe_close_when_reader_still_alive(tmp_path: Path):
    paths = DesktopPaths(app_dir=tmp_path, user_data_dir=tmp_path / "user")
    supervisor = SidecarSupervisor(paths, DesktopConfig(), api_port=18011, qdrant_port=16344)

    proc_alive = _make_mock_popen_proc()
    proc_alive.stdout = MagicMock()
    live_reader = MagicMock(spec=threading.Thread)
    live_reader.is_alive.return_value = True

    supervisor._terminate_process(proc_alive, "Backend", timeout_sec=0.1, reader_thread=live_reader)
    live_reader.join.assert_called_once_with(timeout=1.5)
    proc_alive.stdout.close.assert_not_called()

    proc_done = _make_mock_popen_proc()
    proc_done.stdout = MagicMock()
    finished_reader = MagicMock(spec=threading.Thread)
    finished_reader.is_alive.side_effect = [True, False]

    supervisor._terminate_process(proc_done, "Backend", timeout_sec=0.1, reader_thread=finished_reader)
    finished_reader.join.assert_called_once_with(timeout=1.5)
    proc_done.stdout.close.assert_called_once()


def test_terminate_process_kills_win32_process_tree_for_popen_only(tmp_path: Path):
    paths = DesktopPaths(app_dir=tmp_path, user_data_dir=tmp_path / "user")
    supervisor = SidecarSupervisor(paths, DesktopConfig(), api_port=18012, qdrant_port=16345)

    plain_mock_proc = _make_mock_popen_proc()
    popen_spec_proc = MagicMock(spec=subprocess.Popen)
    popen_spec_proc.pid = 9876
    popen_spec_proc.poll.return_value = None
    popen_spec_proc.wait.return_value = 0
    popen_spec_proc.stdout = None

    with (
        patch("src.desktop.supervisor.sys.platform", "win32"),
        patch("src.desktop.supervisor._kill_process_tree_win32") as mock_kill_tree,
    ):
        supervisor._terminate_process(plain_mock_proc, "MockOnly", timeout_sec=0.1)
        mock_kill_tree.assert_not_called()

        supervisor._terminate_process(popen_spec_proc, "RealPopen", timeout_sec=0.1)
        mock_kill_tree.assert_called_once_with(9876)


def test_sidecar_supervisor_shutdown_closes_job_object_when_stop_raises(tmp_path: Path):
    paths = DesktopPaths(app_dir=tmp_path, user_data_dir=tmp_path / "user")
    supervisor = SidecarSupervisor(paths, DesktopConfig(), api_port=18013, qdrant_port=16346)
    supervisor._ensure_loggers()

    mock_job = MagicMock()
    with (
        patch.object(supervisor, "_job_object", mock_job),
        patch.object(supervisor, "stop_backend", side_effect=RuntimeError("backend stop failure")),
        patch.object(supervisor, "stop_qdrant") as mock_stop_qdrant,
    ):
        with pytest.raises(RuntimeError, match="backend stop failure"):
            supervisor.shutdown(timeout_sec=0.2)

    mock_stop_qdrant.assert_called_once_with(timeout_sec=0.2)
    mock_job.close.assert_called_once()
    assert supervisor._log_handlers == {}


# ==============================================================================
# Windows Process RSS Telemetry
# ==============================================================================


def test_get_process_rss_mb_windows_native():
    if sys.platform != "win32":
        pytest.skip("Win32 GetProcessMemoryInfo check applies to Windows hosts")

    with patch("os.path.exists", return_value=False):
        rss_mb = _get_process_rss_mb()

    assert isinstance(rss_mb, float)
    assert rss_mb > 0.0
