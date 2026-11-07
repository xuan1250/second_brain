"""Hybrid dense + BM25 sparse vector indexer and process-level shared embedder singleton."""
from __future__ import annotations

import asyncio
import gc
import hashlib
import logging
import math
import os
import re
import uuid
import zlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import EmbeddingConfig, get_settings
from src.db.models import Document, DocumentChunk, DomainRegistry
from src.ingestion.entity_extractor import sync_chunk_entities_to_db
from src.db.qdrant import DENSE, SPARSE, init_qdrant_collection
from src.ingestion.chunker import ChunkPayload

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


def release_cuda_cache() -> bool:
    try:
        from src.ingestion.classifier import unload_classifier
        unload_classifier()
    except Exception:
        pass
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


class BaseEmbedder(ABC):
    """Abstract dense embedder interface shared by ingestion and Phase 3 retrieval."""

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

            cache_folder = str(self.cache_dir) if self.cache_dir else os.environ.get("HF_HOME")
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

                cache_folder = str(self.cache_dir) if self.cache_dir else os.environ.get("HF_HOME")
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
            # Offline fallback if FastEmbed BM25 ONNX weights are not cached locally
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


# Process-level singletons in the main process (shared across ingestion and Phase 3 retrieval)
_embedder_singleton: BaseEmbedder | None = None
_sparse_singleton: SparseEncoder | None = None


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
    # Default: local BAAI/bge-m3
    hf_cache = resolved_data / "models" / "huggingface"
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


async def check_collection_sparse_mode_compatibility(
    client: AsyncQdrantClient,
    collection: str,
    active_sparse_mode: str,
) -> bool:
    """Check if existing Qdrant points have matching sparse_mode and log a warning if mismatched."""
    try:
        if not await client.collection_exists(collection):
            return True
        res, _ = await client.scroll(
            collection_name=collection,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not res:
            return True
        stored_mode = (res[0].payload or {}).get("sparse_mode")
        if stored_mode and stored_mode != active_sparse_mode:
            logger.warning(
                "Collection '%s' contains points indexed with sparse_mode='%s', but active sparse_mode is '%s'. "
                "Hybrid search recall may be degraded across incompatible sparse vector vocabularies. "
                "Re-indexing with 'uv run python -m src.ingestion.cli <path> --force' is recommended.",
                collection,
                stored_mode,
                active_sparse_mode,
            )
            return False
    except Exception as exc:
        logger.debug("Failed checking sparse_mode compatibility: %s", exc)
    return True


class HybridIndexer:
    """Indexes parent-child chunks into Qdrant (dense + sparse) and SQLite (DocumentChunk)."""

    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        embedder: BaseEmbedder | None = None,
        sparse_encoder: SparseEncoder | None = None,
        collection: str | None = None,
        batch_size: int = 32,
        clear_cuda_cache: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.qdrant = qdrant
        self.embedder = embedder or get_embedder()
        self.sparse_encoder = sparse_encoder or get_sparse_encoder()
        self.collection = collection or settings.qdrant_collection
        self.batch_size = max(1, batch_size)
        self.clear_cuda_cache = (
            clear_cuda_cache
            if clear_cuda_cache is not None
            else settings.ingestion.clear_cuda_cache_after_indexing
        )

    async def index_chunks(
        self,
        session: AsyncSession,
        doc_id: str,
        chunks: list[ChunkPayload],
    ) -> int:
        """Embed and upsert chunks in batches into Qdrant and record DocumentChunk rows in SQLite."""
        if not chunks:
            return 0

        await init_qdrant_collection(
            self.qdrant,
            collection=self.collection,
            dimensions=self.embedder.dimensions,
        )

        active_sparse_mode = "bge_m3" if isinstance(self.sparse_encoder, BgeM3SparseEncoder) else "bm25"
        await check_collection_sparse_mode_compatibility(
            self.qdrant,
            self.collection,
            active_sparse_mode=active_sparse_mode,
        )

        # Purge existing Qdrant vectors for doc_id to ensure retry idempotency.
        # Fail fast on error so ingestion aborts before modifying SQLite chunk records.
        await self.qdrant.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                )
            ),
            wait=True,
        )

        # Purge existing SQLite DocumentChunk rows for doc_id
        await session.execute(delete(DocumentChunk).where(DocumentChunk.doc_id == doc_id))
        await session.flush()

        db_rows: list[DocumentChunk] = []
        upserted_point_ids: list[str] = []

        try:
            for start in range(0, len(chunks), self.batch_size):
                batch = chunks[start : start + self.batch_size]
                contextualized_texts = [c.contextualized_text for c in batch]

                if isinstance(self.sparse_encoder, BgeM3SparseEncoder) and getattr(
                    self.embedder, "supports_native_sparse", False
                ):
                    dense_vectors, sparse_vectors = await self.embedder.embed_documents_with_sparse(contextualized_texts)
                else:
                    dense_vectors = await self.embedder.embed_documents(contextualized_texts)
                    sparse_vectors = await self.sparse_encoder.encode(contextualized_texts)

                points: list[models.PointStruct] = []
                batch_point_ids: list[str] = []
                for chunk, dense_vec, sparse_vec in zip(batch, dense_vectors, sparse_vectors):
                    point_id = str(uuid.UUID(chunk.chunk_id))
                    batch_point_ids.append(point_id)
                    payload: dict[str, Any] = {
                        "doc_id": doc_id,
                        "chunk_id": chunk.chunk_id,
                        "source_filename": chunk.source_filename,
                        "file_type": chunk.file_type,
                        "heading_path": chunk.heading_path,
                        "page_numbers": list(chunk.page_numbers),
                        "token_count": int(chunk.token_count),
                        "child_text": chunk.child_text,
                        "parent_content": chunk.parent_content,
                        "sparse_mode": active_sparse_mode,
                        "domain": chunk.domain_key or "general",
                        "subdomain": chunk.subdomain_key or "general",
                        "epistemic_type": chunk.epistemic_modality or "general",
                        "authority_tier": int(chunk.authority_tier),
                        "jurisdiction": chunk.jurisdiction,
                        "effective_year": chunk.effective_year,
                        "is_current": True,
                        "associated_entities": list(chunk.associated_entities),
                        "contextual_prefix": chunk.contextual_prefix,
                    }
                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector={
                                DENSE: dense_vec,
                                SPARSE: sparse_vec,
                            },
                            payload=payload,
                        )
                    )
                    page_str = ",".join(str(p) for p in chunk.page_numbers) if chunk.page_numbers else None
                    db_rows.append(
                        DocumentChunk(
                            id=chunk.chunk_id,
                            doc_id=doc_id,
                            point_id=point_id,
                            parent_id=chunk.parent_id,
                            chunk_index=chunk.chunk_index,
                            heading_path=chunk.heading_path,
                            page_numbers=page_str,
                            token_count=chunk.token_count,
                            text=chunk.child_text,
                            parent_content=chunk.parent_content,
                            domain_key=chunk.domain_key or "general",
                            subdomain_key=chunk.subdomain_key or "general",
                            epistemic_modality=chunk.epistemic_modality or "general",
                            contextual_prefix=chunk.contextual_prefix,
                        )
                    )

                # Pre-upsert SQLite barrier: check document existence and valid status
                doc = await session.get(Document, doc_id, populate_existing=True)
                if doc is None or doc.status in ("failed", "cancelled"):
                    logger.warning(
                        "Document %s was deleted or aborted before Qdrant upsert; cancelling vector indexing",
                        doc_id,
                    )
                    if upserted_point_ids:
                        logger.warning(
                            "Rolling back %d previously upserted points for aborted doc %s",
                            len(upserted_point_ids),
                            doc_id,
                        )
                        try:
                            await self.qdrant.delete(
                                collection_name=self.collection,
                                points_selector=models.PointIdsList(points=upserted_point_ids),
                                wait=True,
                            )
                        except Exception as rollback_err:
                            logger.error(
                                "Failed to rollback Qdrant points for doc %s (%d points orphaned: %s): %s",
                                doc_id,
                                len(upserted_point_ids),
                                upserted_point_ids[:10],
                                rollback_err,
                            )
                    return 0

                await self.qdrant.upsert(
                    collection_name=self.collection,
                    points=points,
                    wait=True,
                )
                upserted_point_ids.extend(batch_point_ids)

            session.add_all(db_rows)
            try:
                from sqlalchemy import select
                dom_res = await session.execute(select(DomainRegistry.id, DomainRegistry.slug))
                slug_to_id = {row[1]: row[0] for row in dom_res.all()}
                for c in chunks:
                    if c.associated_entities:
                        d_id = slug_to_id.get(c.domain_key or "general")
                        if d_id:
                            await sync_chunk_entities_to_db(
                                session,
                                chunk_id=c.chunk_id,
                                domain_id=d_id,
                                entities=c.associated_entities,
                            )
            except Exception as ent_err:
                logger.debug("Entity database sync skipped: %s", ent_err)
            await session.commit()
            return len(db_rows)
        except Exception as exc:
            if upserted_point_ids:
                logger.warning(
                    "Indexing failed for doc %s; rolling back %d orphan Qdrant points",
                    doc_id,
                    len(upserted_point_ids),
                )
                try:
                    await self.qdrant.delete(
                        collection_name=self.collection,
                        points_selector=models.PointIdsList(points=upserted_point_ids),
                        wait=True,
                    )
                except Exception as rollback_err:
                    logger.error(
                        "Failed to rollback Qdrant points for doc %s (%d points orphaned: %s): %s",
                        doc_id,
                        len(upserted_point_ids),
                        upserted_point_ids[:10],
                        rollback_err,
                    )
            raise
        finally:
            if self.clear_cuda_cache:
                release_cuda_cache()

