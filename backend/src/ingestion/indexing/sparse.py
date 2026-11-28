"""Sparse BM25 and learned BGE-M3 sparse encoders with process-level singleton."""
from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import zlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qdrant_client import models

from src.config import get_settings
from src.ingestion.indexing.device import resolve_compute_device, resolve_sparse_mode
from src.ingestion.indexing.embedders import LocalBgeM3Embedder, get_embedder

logger = logging.getLogger(__name__)


class SparseEncoder(ABC):
    """Abstract sparse BM25 vector extractor."""

    @abstractmethod
    def encode_sync(self, texts: Sequence[str]) -> list[models.SparseVector]:
        """Compute Qdrant SparseVector (indices, values) for each text."""

    async def encode(self, texts: Sequence[str]) -> list[models.SparseVector]:
        if not texts:
            return []
        return await asyncio.to_thread(self.encode_sync, texts)


class FastEmbedBM25Encoder(SparseEncoder):
    """FastEmbed Qdrant/bm25 sparse vector encoder with offline fallback."""

    def __init__(self, model_name: str = "Qdrant/bm25", cache_dir: Path | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from fastembed import SparseTextEmbedding

            cache_path = str(self.cache_dir) if self.cache_dir else os.environ.get("FASTEMBED_CACHE_PATH")
            self._model = SparseTextEmbedding(model_name=self.model_name, cache_dir=cache_path)
        return self._model

    def encode_sync(self, texts: Sequence[str]) -> list[models.SparseVector]:
        if not texts:
            return []
        try:
            model = self._get_model()
            embeddings = list(model.embed(list(texts)))
            return [
                models.SparseVector(
                    indices=[int(i) for i in emb.indices.tolist()],
                    values=[float(v) for v in emb.values.tolist()],
                )
                for emb in embeddings
            ]
        except Exception:
            return DeterministicBM25Encoder().encode_sync(texts)


class DeterministicBM25Encoder(SparseEncoder):
    """Deterministic term-frequency sparse encoder (offline fallback / stub)."""

    def encode_sync(self, texts: Sequence[str]) -> list[models.SparseVector]:
        out: list[models.SparseVector] = []
        for text in texts:
            tokens = re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)
            tf: dict[int, float] = {}
            for tok in tokens:
                idx = zlib.crc32(tok.encode("utf-8")) & 0x7FFFFFFF
                tf[idx] = tf.get(idx, 0.0) + 1.0
            sorted_items = sorted(tf.items())
            indices = [k for k, _ in sorted_items]
            values = [1.0 + math.log(v) for _, v in sorted_items]
            out.append(models.SparseVector(indices=indices, values=values))
        return out


class BgeM3SparseEncoder(SparseEncoder):
    """Sparse encoder leveraging LocalBgeM3Embedder's learned sparse head with BM25 fallback."""

    def __init__(self, embedder: LocalBgeM3Embedder | None = None) -> None:
        self.embedder = embedder

    def _resolve_embedder(self) -> LocalBgeM3Embedder:
        if self.embedder is not None:
            return self.embedder
        emb = get_embedder()
        if isinstance(emb, LocalBgeM3Embedder):
            return emb
        raise ValueError("BgeM3SparseEncoder requires a LocalBgeM3Embedder instance")

    def encode_sync(self, texts: Sequence[str]) -> list[models.SparseVector]:
        if not texts:
            return []
        try:
            emb = self._resolve_embedder()
            if emb.supports_native_sparse:
                _, sparse_vecs = emb.embed_documents_with_sparse_sync(texts)
                return sparse_vecs
        except Exception as exc:
            logger.warning("BgeM3SparseEncoder failed, falling back to FastEmbed BM25: %s", exc)
        return FastEmbedBM25Encoder().encode_sync(texts)


_sparse_singleton: SparseEncoder | None = None


def get_sparse_encoder(use_fastembed: bool = True) -> SparseEncoder:
    """Return the process-level shared sparse encoder singleton."""
    global _sparse_singleton
    if _sparse_singleton is None:
        settings = get_settings()
        resolved_device = resolve_compute_device(settings.embedding.device)
        mode = resolve_sparse_mode(settings.embedding.sparse_mode, resolved_device, settings.embedding.provider)
        if mode == "bge_m3":
            _sparse_singleton = BgeM3SparseEncoder()
        elif use_fastembed:
            fe_cache = settings.data_dir / "models" / "fastembed"
            fe_cache.mkdir(parents=True, exist_ok=True)
            _sparse_singleton = FastEmbedBM25Encoder(cache_dir=fe_cache)
        else:
            _sparse_singleton = DeterministicBM25Encoder()
    return _sparse_singleton


def set_sparse_encoder(encoder: SparseEncoder | None) -> None:
    """Override or reset the process-level sparse encoder singleton."""
    global _sparse_singleton
    _sparse_singleton = encoder
