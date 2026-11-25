"""Compute device resolution, CUDA cache reclamation, and HuggingFace hub cache path helpers."""
from __future__ import annotations

import gc
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_compute_device(requested: str = "auto") -> str:
    """Resolve compute device ('auto' | 'cuda' | 'cpu') with safe fallback if CUDA is unavailable."""
    req = (requested or "auto").strip().lower()
    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    if req == "cuda":
        if cuda_available:
            return "cuda"
        logger.warning("CUDA requested but torch.cuda.is_available() is False; falling back to CPU")
        return "cpu"
    if req == "auto":
        return "cuda" if cuda_available else "cpu"
    return "cpu"


def empty_torch_cuda_cache() -> bool:
    """Safely invoke garbage collection and empty PyTorch CUDA cache if available."""
    try:
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return True
        return False
    except Exception as exc:
        logger.debug("CUDA cache release skipped or failed: %s", exc)
        return False


def release_cuda_cache() -> bool:
    """Unload resident classifier weights, run GC, and empty PyTorch CUDA cache if available."""
    try:
        from src.ingestion.classifier import unload_classifier

        unload_classifier()
    except Exception:
        pass
    return empty_torch_cuda_cache()


def resolve_sparse_mode(sparse_mode: str = "auto", resolved_device: str = "cpu", provider: str = "local") -> str:
    """Resolve sparse representation mode ('bge_m3' vs 'bm25')."""
    if (provider or "").strip().lower() != "local":
        return "bm25"
    mode = (sparse_mode or "auto").strip().lower()
    if mode == "auto":
        return "bge_m3" if resolved_device == "cuda" else "bm25"
    if mode == "bge_m3":
        return "bge_m3"
    return "bm25"


def _resolve_hf_hub_cache(cache_dir: Path | str | None) -> str | None:
    """Resolve and normalize a Hugging Face cache directory to point to its 'hub' subdirectory.

    Prevents cache fragmentation where passing a root huggingface directory
    causes models to be downloaded into both root and 'hub/' subdirectories.
    """
    if cache_dir is not None:
        p = Path(cache_dir)
        if p.name == "hub":
            return str(p)
        if p.name == "huggingface" or (p / "hub").exists():
            return str(p / "hub")
        return str(p)

    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        return hf_hub_cache

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        p = Path(hf_home)
        if p.name == "hub":
            return str(p)
        return str(p / "hub")

    return None
