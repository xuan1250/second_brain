"""On-disk HuggingFace, FastEmbed, FlashRank, RapidOCR, and Docling cache inspection helpers."""
from __future__ import annotations

import os
from pathlib import Path

from src.desktop.models import ModelCacheStatus, ModelSpec


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


def inspect_single_model(models_root: Path, spec: ModelSpec) -> tuple[ModelCacheStatus, list[tuple[str, int]]]:
    """Inspect cache markers for a single canonical model group under `models_root`."""
    if spec.key in ("bge_m3", "laya_multilingual"):
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
