"""Dense embedder implementations and process-level embedder singleton."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qdrant_client import models

from src.config import EmbeddingConfig, get_settings
from src.ingestion.indexing.device import (
    _resolve_hf_hub_cache,
    resolve_compute_device,
    resolve_sparse_mode,
)

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract dense embedder interface shared by ingestion and retrieval."""

    dimensions: int

    @property
    def supports_native_sparse(self) -> bool:
        """Whether embedder produces native learned sparse vectors alongside dense vectors."""
        return False

    @abstractmethod
    def embed_documents_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronously embed a batch of document texts."""

    def embed_query_sync(self, text: str) -> list[float]:
        """Synchronously embed a single search query."""
        return self.embed_documents_sync([text])[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously embed documents via asyncio.to_thread (releases GIL during inference)."""
        if not texts:
            return []
        return await asyncio.to_thread(self.embed_documents_sync, texts)

    async def embed_query(self, text: str) -> list[float]:
        """Asynchronously embed a single query via asyncio.to_thread."""
        return await asyncio.to_thread(self.embed_query_sync, text)

    async def embed_documents_with_sparse(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], list[models.SparseVector]]:
        """Asynchronously compute both dense and sparse vectors in a single pass."""
        raise NotImplementedError("Native sparse embeddings not supported on this embedder")

    async def embed_query_with_sparse(self, text: str) -> tuple[list[float], models.SparseVector]:
        """Asynchronously compute dense and sparse vectors for a single query."""
        dense_vecs, sparse_vecs = await self.embed_documents_with_sparse([text])
        return dense_vecs[0], sparse_vecs[0]


class LocalBgeM3Embedder(BaseEmbedder):
    """Local sentence-transformers embedder (BAAI/bge-m3, 1024 dims) with fp16 CUDA and learned sparse vector support."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        dimensions: int = 1024,
        cache_dir: Path | None = None,
        device: str = "auto",
        fp16: bool = True,
        max_length: int = 8192,
        sparse_mode: str = "auto",
        use_native_sparse: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self.cache_dir = cache_dir
        self.device = device
        self.fp16 = fp16
        self.max_length = max_length
        self.sparse_mode = sparse_mode
        self.resolved_device = resolve_compute_device(device)
        self.resolved_sparse_mode = resolve_sparse_mode(sparse_mode, self.resolved_device, provider="local")
        if use_native_sparse is not None:
            self.resolved_sparse_mode = "bge_m3" if use_native_sparse else "bm25"
        self._model: Any = None
        self._sparse_linear: Any = None
        self._sparse_load_attempted: bool = False

    @property
    def supports_native_sparse(self) -> bool:
        if self.resolved_sparse_mode != "bge_m3":
            return False
        return self._get_sparse_linear() is not None

    def _get_model(self) -> Any:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            cache_folder = _resolve_hf_hub_cache(self.cache_dir)
            use_fp16 = self.resolved_device == "cuda" and self.fp16
            model_kwargs: dict[str, Any] = {}
            if use_fp16:
                model_kwargs["torch_dtype"] = torch.float16

            self._model = SentenceTransformer(
                self.model_name,
                device=self.resolved_device,
                cache_folder=cache_folder,
                model_kwargs=model_kwargs,
            )
            if hasattr(self._model, "max_seq_length"):
                self._model.max_seq_length = self.max_length
        return self._model

    def _get_sparse_linear(self) -> Any:
        if self._sparse_linear is None and not self._sparse_load_attempted:
            self._sparse_load_attempted = True
            if self.resolved_sparse_mode != "bge_m3":
                return None
            try:
                import torch
                import torch.nn as nn
                from huggingface_hub import hf_hub_download

                cache_folder = _resolve_hf_hub_cache(self.cache_dir)
                weight_path = hf_hub_download(
                    repo_id=self.model_name,
                    filename="sparse_linear.pt",
                    cache_dir=cache_folder,
                )
                linear = nn.Linear(self.dimensions, 1)
                state = torch.load(weight_path, map_location=self.resolved_device, weights_only=True)
                if isinstance(state, nn.Module):
                    linear = state.to(self.resolved_device)
                elif isinstance(state, dict):
                    if "weight" in state:
                        linear.load_state_dict(state)
                    elif "linear.weight" in state:
                        linear.load_state_dict({k.replace("linear.", ""): v for k, v in state.items()})
                    else:
                        linear.load_state_dict(state, strict=False)
                    linear = linear.to(self.resolved_device)
                else:
                    linear = linear.to(self.resolved_device)

                if self.resolved_device == "cuda" and self.fp16:
                    linear = linear.half()
                linear.eval()
                self._sparse_linear = linear
            except Exception as exc:
                logger.warning(
                    "Could not load sparse_linear.pt for BGE-M3 learned sparse weights (%s); falling back to BM25",
                    exc,
                )
                self.resolved_sparse_mode = "bm25"
                self._sparse_linear = None
        return self._sparse_linear

    def embed_documents_sync(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        if hasattr(vecs, "tolist"):
            return [v.tolist() for v in vecs]
        return [list(v) for v in vecs]

    def embed_documents_with_sparse_sync(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], list[models.SparseVector]]:
        if not texts:
            return [], []
        model = self._get_model()
        sparse_linear = self._get_sparse_linear()

        if sparse_linear is None:
            from src.ingestion.indexing.sparse import FastEmbedBM25Encoder

            dense = self.embed_documents_sync(list(texts))
            sparse = FastEmbedBM25Encoder(cache_dir=self.cache_dir).encode_sync(texts)
            return dense, sparse

        import torch

        with torch.inference_mode():
            features = model.encode(
                list(texts),
                output_value=None,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if isinstance(features, dict):
                features = [features]

            special_ids: set[int] = set()
            tokenizer = getattr(model, "tokenizer", None)
            if tokenizer is not None and hasattr(tokenizer, "all_special_ids"):
                special_ids = set(tokenizer.all_special_ids)

            dense_results: list[list[float]] = []
            sparse_results: list[models.SparseVector] = []

            for feat in features:
                dense_tensor = feat["sentence_embedding"].detach().cpu()
                dense_results.append(dense_tensor.float().tolist())

                token_embs = feat["token_embeddings"]
                input_ids = feat["input_ids"]

                sparse_logits = sparse_linear(token_embs)
                sparse_relu = torch.relu(sparse_logits).squeeze(-1)

                token_weights: dict[int, float] = {}
                for tok_id_t, w_t in zip(input_ids, sparse_relu):
                    tok_id = int(tok_id_t.item())
                    if tok_id in special_ids:
                        continue
                    w = float(w_t.item())
                    if w > 0:
                        if tok_id not in token_weights or w > token_weights[tok_id]:
                            token_weights[tok_id] = w

                if token_weights:
                    sorted_items = sorted(token_weights.items())
                    sparse_vec = models.SparseVector(
                        indices=[k for k, _ in sorted_items],
                        values=[v for _, v in sorted_items],
                    )
                else:
                    sparse_vec = models.SparseVector(indices=[], values=[])
                sparse_results.append(sparse_vec)

            return dense_results, sparse_results

    async def embed_documents_with_sparse(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], list[models.SparseVector]]:
        if not texts:
            return [], []
        return await asyncio.to_thread(self.embed_documents_with_sparse_sync, texts)


class OpenAIEmbedder(BaseEmbedder):
    """Cloud OpenAI / OpenAI-compatible embeddings (text-embedding-3-small / text-embedding-3-large)."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimensions: int = 1536,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self.base_url = base_url
        self.api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from langchain_openai import OpenAIEmbeddings

            kwargs: dict[str, Any] = {"model": self.model_name, "dimensions": self.dimensions}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = OpenAIEmbeddings(**kwargs)
        return self._client

    def embed_documents_sync(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._get_client().embed_documents(texts)

    def embed_query_sync(self, text: str) -> list[float]:
        return self._get_client().embed_query(text)


class DeterministicStubEmbedder(BaseEmbedder):
    """Deterministic unit-norm hash embedder for fast offline testing."""

    def __init__(self, dimensions: int = 32) -> None:
        self.dimensions = dimensions

    def embed_documents_sync(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            raw: list[float] = []
            counter = 0
            while len(raw) < self.dimensions:
                digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
                for i in range(0, len(digest), 4):
                    if len(raw) >= self.dimensions:
                        break
                    val = int.from_bytes(digest[i : i + 4], "little", signed=True) / float(2**31)
                    raw.append(val)
                counter += 1
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            results.append([x / norm for x in raw])
        return results


_embedder_singleton: BaseEmbedder | None = None


def create_embedder(config: EmbeddingConfig | None = None, data_dir: Path | None = None) -> BaseEmbedder:
    """Instantiate an embedder from configuration."""
    settings = get_settings()
    cfg = config or settings.embedding
    resolved_data = Path(data_dir) if data_dir else settings.data_dir
    provider = (cfg.provider or "local").lower()

    if provider == "stub":
        return DeterministicStubEmbedder(dimensions=cfg.dimensions)
    if provider == "openai":
        return OpenAIEmbedder(
            model_name=cfg.model_name or "text-embedding-3-small",
            dimensions=cfg.dimensions,
            base_url=settings.llm.base_url,
        )
    hf_cache = resolved_data / "models" / "huggingface" / "hub"
    hf_cache.mkdir(parents=True, exist_ok=True)
    return LocalBgeM3Embedder(
        model_name=cfg.model_name or "BAAI/bge-m3",
        dimensions=cfg.dimensions,
        cache_dir=hf_cache,
        device=cfg.device,
        fp16=cfg.fp16,
        max_length=cfg.max_length,
        sparse_mode=cfg.sparse_mode,
    )


def get_embedder() -> BaseEmbedder:
    """Return the process-level shared dense embedder singleton."""
    global _embedder_singleton
    if _embedder_singleton is None:
        _embedder_singleton = create_embedder()
    return _embedder_singleton


def set_embedder(embedder: BaseEmbedder | None) -> None:
    """Override or reset the process-level dense embedder singleton (useful for tests)."""
    global _embedder_singleton
    _embedder_singleton = embedder
