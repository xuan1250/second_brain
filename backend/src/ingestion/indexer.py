"""Hybrid dense + BM25 sparse vector indexer and process-level shared embedder singleton.

Backwards-compatible facade re-exporting symbols from `src.ingestion.indexing`.
"""
from __future__ import annotations

from src.ingestion.indexing.device import (
    _resolve_hf_hub_cache,
    empty_torch_cuda_cache,
    release_cuda_cache,
    resolve_compute_device,
    resolve_sparse_mode,
)
from src.ingestion.indexing.embedders import (
    BaseEmbedder,
    DeterministicStubEmbedder,
    LocalBgeM3Embedder,
    OpenAIEmbedder,
    create_embedder,
    get_embedder,
    set_embedder,
)
from src.ingestion.indexing.hybrid_indexer import (
    HybridIndexer,
    check_collection_sparse_mode_compatibility,
)
from src.ingestion.indexing.sparse import (
    BgeM3SparseEncoder,
    DeterministicBM25Encoder,
    FastEmbedBM25Encoder,
    SparseEncoder,
    get_sparse_encoder,
    set_sparse_encoder,
)

__all__ = [
    "BaseEmbedder",
    "BgeM3SparseEncoder",
    "DeterministicBM25Encoder",
    "DeterministicStubEmbedder",
    "FastEmbedBM25Encoder",
    "HybridIndexer",
    "LocalBgeM3Embedder",
    "OpenAIEmbedder",
    "SparseEncoder",
    "_resolve_hf_hub_cache",
    "check_collection_sparse_mode_compatibility",
    "create_embedder",
    "empty_torch_cuda_cache",
    "get_embedder",
    "get_sparse_encoder",
    "release_cuda_cache",
    "resolve_compute_device",
    "resolve_sparse_mode",
    "set_embedder",
    "set_sparse_encoder",
]
