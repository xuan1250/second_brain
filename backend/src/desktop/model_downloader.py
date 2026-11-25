"""Resumable offline model downloaders for all 6 canonical AI model groups."""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.desktop.config import DesktopConfig, DesktopPaths, save_desktop_config
from src.desktop.env import build_model_cache_env
from src.desktop.model_cache import (
    inspect_models_status,
    inspect_single_model,
    write_models_ready_manifest,
)
from src.desktop.models import MODEL_SPECS, BootstrapProgress, ModelCacheStatus, ModelSpec


def download_model_group(
    models_root: Path,
    spec: ModelSpec,
    *,
    download_hooks: dict[str, Callable[[Path], Any]] | None = None,
    inspect_single_fn: Callable[[ModelSpec], tuple[ModelCacheStatus, list[tuple[str, int]]]] | None = None,
) -> None:
    """Download and verify a single model group into its target cache subdirectory."""
    if download_hooks and spec.key in download_hooks:
        target = models_root / spec.cache_subdir
        target.mkdir(parents=True, exist_ok=True)
        download_hooks[spec.key](target)
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
        engineering_extractor.reset_rapidocr_engine()
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

        status_after, _ = (
            inspect_single_fn(spec)
            if inspect_single_fn is not None
            else inspect_single_model(models_root, spec)
        )
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


def ensure_models_ready(
    paths: DesktopPaths,
    config: DesktopConfig,
    models_dir: Path,
    manifest_path: Path,
    *,
    force: bool = False,
    progress_cb: Callable[[BootstrapProgress], None] | None = None,
    emit_progress_fn: Callable[..., BootstrapProgress],
    download_hooks: dict[str, Callable[[Path], Any]] | None = None,
    inject_env_fn: Callable[[], None] | None = None,
    inspect_status_fn: Callable[[], dict[str, Any]] | None = None,
    inspect_single_fn: Callable[[ModelSpec], tuple[ModelCacheStatus, list[tuple[str, int]]]] | None = None,
    download_group_fn: Callable[[ModelSpec], None] | None = None,
    write_manifest_fn: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Download and verify all 6 required model groups with warm fast-path and clean resume."""
    paths.ensure_directories(config)
    if inject_env_fn is not None:
        inject_env_fn()
    else:
        os.environ.update(build_model_cache_env(models_dir, for_download=True))

    status_fn = inspect_status_fn or (lambda: inspect_models_status(models_dir, manifest_path))
    single_fn = inspect_single_fn or (lambda spec: inspect_single_model(models_dir, spec))
    group_fn = download_group_fn or (
        lambda spec: download_model_group(
            models_dir,
            spec,
            download_hooks=download_hooks,
            inspect_single_fn=single_fn,
        )
    )
    manifest_fn = write_manifest_fn or (
        lambda: write_models_ready_manifest(
            models_dir,
            manifest_path,
            config.last_installed_version,
            inspect_single_fn=single_fn,
        )
    )

    if not force:
        current_status = status_fn()
        if current_status["models_ready"]:
            if not config.models_ready:
                config.models_ready = True
                save_desktop_config(config, paths)
            model_states = {spec.key: "cached" for spec in MODEL_SPECS}
            emit_progress_fn(
                progress_cb,
                step="complete",
                percent=100.0,
                detail="All AI models are already cached and verified.",
                is_running=False,
                completed=True,
                models=model_states,
            )
            return current_status

    if force and manifest_path.is_file():
        try:
            manifest_path.unlink()
        except OSError:
            pass

    model_states: dict[str, str] = {}
    for spec in MODEL_SPECS:
        st, _ = single_fn(spec)
        model_states[spec.key] = "cached" if (not force and st.cached) else "pending"

    total_groups = len(MODEL_SPECS)
    emit_progress_fn(
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
            emit_progress_fn(
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
        emit_progress_fn(
            progress_cb,
            step=f"download_{spec.key}",
            percent=start_pct,
            detail=f"Downloading {spec.display_name} ({spec.repo_or_id})...",
            is_running=True,
            completed=False,
            models=model_states,
        )

        try:
            group_fn(spec)
            post_status, _ = single_fn(spec)
            if not post_status.cached:
                raise RuntimeError(
                    f"Download finished for {spec.display_name}, but cache validation failed: {post_status.detail}"
                )
        except Exception as exc:
            # Never delete existing HuggingFace blobs so subsequent runs resume cleanly.
            model_states[spec.key] = "failed"
            endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
            err_detail = f"Failed downloading {spec.display_name} ({spec.repo_or_id}, endpoint={endpoint}): {exc}"
            config.models_ready = False
            save_desktop_config(config, paths)
            emit_progress_fn(
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
        emit_progress_fn(
            progress_cb,
            step=f"download_{spec.key}",
            percent=end_pct,
            detail=f"Completed {spec.display_name}.",
            is_running=True,
            completed=False,
            models=model_states,
        )

    manifest_fn()
    config.models_ready = True
    save_desktop_config(config, paths)

    final_status = status_fn()
    emit_progress_fn(
        progress_cb,
        step="complete",
        percent=100.0,
        detail="All 6 offline AI model groups downloaded and verified.",
        is_running=False,
        completed=True,
        models=model_states,
    )
    return final_status
