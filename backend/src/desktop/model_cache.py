"""On-disk model cache inspection, atomic manifest persistence, and local bundle import."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.desktop.config import DesktopConfig, DesktopPaths, save_desktop_config
from src.desktop.io_utils import atomic_write_json
from src.desktop.model_cache_inspect import (
    _hf_repo_cache_dir,
    _inspect_docling_models,
    _inspect_hf_repo,
    _inspect_subdir_group,
    _scan_dir_metrics,
    inspect_single_model,
)
from src.desktop.models import MODEL_SPECS, ModelCacheStatus, ModelSpec

_MANIFEST_FILENAME = "models_ready.json"
_RECOGNIZED_MODEL_SUBDIRS: tuple[str, ...] = (
    "huggingface",
    "fastembed",
    "flashrank",
    "rapidocr",
    "docling",
    "torch",
)


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


def read_manifest_ready(manifest_file: Path) -> tuple[bool, dict[str, Any] | None]:
    """Read and validate `models_ready.json` across all 6 canonical model groups."""
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


def inspect_models_status(
    models_dir: Path,
    manifest_path: Path | None = None,
    *,
    inspect_single_fn: Callable[[ModelSpec], tuple[ModelCacheStatus, list[tuple[str, int]]]] | None = None,
    read_manifest_fn: Callable[[], tuple[bool, dict[str, Any] | None]] | None = None,
) -> dict[str, Any]:
    """Inspect `models_ready.json` and on-disk markers for all 6 model groups."""
    resolved_manifest = manifest_path if manifest_path is not None else (models_dir / _MANIFEST_FILENAME)
    statuses: list[ModelCacheStatus] = []
    total_size_mb = 0.0
    for spec in MODEL_SPECS:
        status, _ = (
            inspect_single_fn(spec)
            if inspect_single_fn is not None
            else inspect_single_model(models_dir, spec)
        )
        statuses.append(status)
        total_size_mb += status.size_mb

    manifest_ready, _ = (
        read_manifest_fn()
        if read_manifest_fn is not None
        else read_manifest_ready(resolved_manifest)
    )
    all_markers_present = all(item.cached for item in statuses)
    models_ready = bool(manifest_ready and all_markers_present)

    return {
        "models_ready": models_ready,
        "models": statuses,
        "total_size_mb": round(total_size_mb, 2),
        "models_dir": str(models_dir),
    }


def write_models_ready_manifest(
    models_dir: Path,
    manifest_path: Path,
    app_version: str,
    *,
    inspect_single_fn: Callable[[ModelSpec], tuple[ModelCacheStatus, list[tuple[str, int]]]] | None = None,
) -> dict[str, Any]:
    """Verify all 6 model groups and atomically write `models_ready.json`."""
    now_iso = datetime.now(timezone.utc).isoformat()
    model_entries: dict[str, Any] = {}
    for spec in MODEL_SPECS:
        status, markers = (
            inspect_single_fn(spec)
            if inspect_single_fn is not None
            else inspect_single_model(models_dir, spec)
        )
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
        "app_version": app_version,
        "models_dir": str(models_dir),
        "models": model_entries,
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, manifest)
    return manifest


def import_local_models_dir(
    paths: DesktopPaths,
    config: DesktopConfig,
    models_dir: Path,
    manifest_path: Path,
    source_dir: Path | str,
    *,
    inspect_single_fn: Callable[[ModelSpec], tuple[ModelCacheStatus, list[tuple[str, int]]]] | None = None,
    write_manifest_fn: Callable[[], dict[str, Any]] | None = None,
    inspect_status_fn: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Import pre-cached model directories from a local offline bundle into `models_dir`."""
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Local models directory not found: {source}")

    paths.ensure_directories(config)
    models_dir.mkdir(parents=True, exist_ok=True)

    for subdir in _RECOGNIZED_MODEL_SUBDIRS:
        src_sub = source / subdir
        if src_sub.is_dir():
            _copy_missing_tree(src_sub, models_dir / subdir)

    single_fn = inspect_single_fn or (lambda spec: inspect_single_model(models_dir, spec))
    all_cached = all(single_fn(spec)[0].cached for spec in MODEL_SPECS)
    if all_cached:
        if write_manifest_fn is not None:
            write_manifest_fn()
        else:
            write_models_ready_manifest(
                models_dir,
                manifest_path,
                config.last_installed_version,
                inspect_single_fn=single_fn,
            )
        config.models_ready = True
        save_desktop_config(config, paths)
    else:
        src_manifest = source / _MANIFEST_FILENAME
        if src_manifest.is_file() and not manifest_path.exists():
            shutil.copy2(src_manifest, manifest_path)

    return inspect_status_fn() if inspect_status_fn is not None else inspect_models_status(models_dir, manifest_path)


__all__ = [
    "_MANIFEST_FILENAME",
    "_RECOGNIZED_MODEL_SUBDIRS",
    "_compute_marker_fingerprint",
    "_copy_missing_tree",
    "_hf_repo_cache_dir",
    "_inspect_docling_models",
    "_inspect_hf_repo",
    "_inspect_subdir_group",
    "_scan_dir_metrics",
    "import_local_models_dir",
    "inspect_models_status",
    "inspect_single_model",
    "read_manifest_ready",
    "write_models_ready_manifest",
]
