"""Python C-extension runtime probe and subprocess execution helpers."""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.desktop.config import DesktopConfig, DesktopPaths
from src.desktop.env import CREATE_NO_WINDOW, run_hidden_subprocess
from src.desktop.models import ErrorCategory, RuntimeCheckResult

_VERIFY_TIMEOUT_SEC = 30.0

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


def run_subprocess_capture(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = _VERIFY_TIMEOUT_SEC,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command with captured output, delegating to `command_runner` when injected."""
    if command_runner is not None:
        return command_runner(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            timeout=timeout,
        )
    return run_hidden_subprocess(cmd, cwd=cwd, env=env, timeout=timeout)


def run_subprocess_streaming(
    cmd: list[str],
    *,
    cwd: Path | None,
    env: dict[str, str],
    on_line: Callable[[str], None],
    stream_runner: Callable[[list[str], Path | None, dict[str, str], Callable[[str], None]], int]
    | None = None,
) -> int:
    """Run a command and stream stdout/stderr lines to `on_line`, honoring `stream_runner`."""
    if stream_runner is not None:
        return int(stream_runner(cmd, cwd, env, on_line))

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
        creationflags=CREATE_NO_WINDOW,
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


def verify_python_runtime(
    paths: DesktopPaths,
    config: DesktopConfig,
    python_exe: Path | None = None,
    *,
    capture_fn: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> RuntimeCheckResult:
    """Verify that `torch`, `onnxruntime`, `pymupdf`, `pandas`, `cv2`, and `laya` load cleanly."""
    target_py = (
        Path(python_exe)
        if python_exe is not None
        else paths.resolve_backend_python(
            config.compute_profile,
            cuda_runtime_ready=config.cuda_runtime_ready,
        )
    )
    cmd = [str(target_py), "-c", _RUNTIME_PROBE_SCRIPT]
    try:
        if capture_fn is not None:
            proc = capture_fn(cmd, timeout=_VERIFY_TIMEOUT_SEC)
        else:
            proc = run_subprocess_capture(
                cmd,
                timeout=_VERIFY_TIMEOUT_SEC,
                command_runner=command_runner,
            )
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
