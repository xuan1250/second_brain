"""Canonical desktop bootstrap models, progress schemas, and offline AI model specifications."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


def _default_model_id(config_cls_name: str, field_name: str, fallback: str) -> str:
    """Resolve canonical model identifier defaults from `src.config` when available."""
    try:
        from src import config as app_config

        cfg_cls = getattr(app_config, config_cls_name)
        default_val = cfg_cls.model_fields[field_name].default
        return str(default_val) if default_val else fallback
    except Exception:
        return fallback


_DEFAULT_EMBEDDING_MODEL = _default_model_id("EmbeddingConfig", "model_name", "BAAI/bge-m3")
_DEFAULT_LAYA_MODEL = _default_model_id(
    "IngestionConfig",
    "laya_model_name",
    "convaiinnovations/laya-multilingual",
)
_DEFAULT_RERANKER_MODEL = _default_model_id("RerankerConfig", "model_name", "ms-marco-MultiBERT-L-12")

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
        repo_or_id=_DEFAULT_EMBEDDING_MODEL,
        cache_subdir="huggingface/hub",
        approx_size_mb=2270,
        description="Dense + learned sparse embeddings and tokenizer for offline chunking and hybrid RAG.",
    ),
    ModelSpec(
        key="laya_multilingual",
        display_name="Laya Multilingual Classifier",
        repo_or_id=_DEFAULT_LAYA_MODEL,
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
        repo_or_id=_DEFAULT_RERANKER_MODEL,
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
