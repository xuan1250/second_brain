"""Shared desktop environment variable construction and hidden-console subprocess execution."""
from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def build_model_cache_env(models_dir: Path | str, *, for_download: bool = False) -> dict[str, str]:
    """Build canonical cache environment variables rooted at `models_dir`.

    When `for_download=True`, cache directories are created eagerly and
    `DOCLING_ARTIFACTS_PATH` is injected unconditionally so Docling downloads into
    `<models_dir>/docling/models`. When `for_download=False`, directories are not
    created and `DOCLING_ARTIFACTS_PATH` is injected only if non-empty artifacts
    already exist on disk.
    """
    models_root = Path(models_dir)
    hf_home = models_root / "huggingface"
    hf_hub = hf_home / "hub"
    torch_home = models_root / "torch"
    fastembed_dir = models_root / "fastembed"
    rapidocr_dir = models_root / "rapidocr"
    docling_dir = models_root / "docling"
    docling_models_dir = docling_dir / "models"

    if for_download:
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

    env_vars: dict[str, str] = {
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_hub),
        "TORCH_HOME": str(torch_home),
        "FASTEMBED_CACHE_PATH": str(fastembed_dir),
        "RAPIDOCR_MODEL_DIR": str(rapidocr_dir),
        "DOCLING_CACHE_DIR": str(docling_dir),
        "FLASHRANK_ALLOW_DOWNLOAD": "true",
    }

    if for_download or (docling_models_dir.is_dir() and any(docling_models_dir.iterdir())):
        env_vars["DOCLING_ARTIFACTS_PATH"] = str(docling_models_dir)

    return env_vars


def run_hidden_subprocess(
    cmd: Sequence[str],
    *,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute a subprocess with captured UTF-8 output and no visible Windows console."""
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=CREATE_NO_WINDOW,
        check=False,
    )
