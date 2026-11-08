"""Unit tests and GPU live benchmark for Docker GPU-accelerated ingestion (Option B)."""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import yaml
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import Settings, load_settings
from src.db.models import Document, SQLModel
from src.db.qdrant import DENSE, SPARSE, init_qdrant_collection
from src.db.session import create_engine, create_session_factory, init_db
from src.ingestion.chunker import ChunkPayload
from src.ingestion.indexer import (
    BaseEmbedder,
    BgeM3SparseEncoder,
    DeterministicBM25Encoder,
    DeterministicStubEmbedder,
    HybridIndexer,
    LocalBgeM3Embedder,
    _resolve_hf_hub_cache,
    check_collection_sparse_mode_compatibility,
    create_embedder,
    release_cuda_cache,
    resolve_compute_device,
    resolve_sparse_mode,
)
from src.ingestion.parser import DocumentParserService, ParseResult
from src.ingestion.worker import IngestionOrchestrator
from src.rag.retriever import HybridRetriever


# ==============================================================================
# 1. Device and Sparse Mode Resolution Tests (AC3, AC5)
# ==============================================================================


def test_resolve_compute_device_auto_when_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_compute_device("auto") == "cuda"


def test_resolve_compute_device_auto_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_compute_device("auto") == "cpu"


def test_resolve_compute_device_cuda_fallback_logs_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with caplog.at_level(logging.WARNING):
        dev = resolve_compute_device("cuda")
    assert dev == "cpu"
    assert "CUDA requested but torch.cuda.is_available() is False; falling back to CPU" in caplog.text


def test_resolve_compute_device_explicit_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_compute_device("cpu") == "cpu"


def test_resolve_sparse_mode() -> None:
    # Non-local provider always falls back to bm25
    assert resolve_sparse_mode("auto", resolved_device="cuda", provider="openai") == "bm25"
    assert resolve_sparse_mode("bge_m3", resolved_device="cuda", provider="openai") == "bm25"

    # Local provider resolution
    assert resolve_sparse_mode("auto", resolved_device="cuda", provider="local") == "bge_m3"
    assert resolve_sparse_mode("auto", resolved_device="cpu", provider="local") == "bm25"
    assert resolve_sparse_mode("bge_m3", resolved_device="cuda", provider="local") == "bge_m3"
    assert resolve_sparse_mode("bge_m3", resolved_device="cpu", provider="local") == "bge_m3"
    assert resolve_sparse_mode("bm25", resolved_device="cuda", provider="local") == "bm25"


def test_resolve_hf_hub_cache_explicit_paths(tmp_path: Path) -> None:
    # Path named huggingface should automatically append hub
    hf_path = tmp_path / "models" / "huggingface"
    assert _resolve_hf_hub_cache(hf_path) == str(hf_path / "hub")

    # Path already named hub should be preserved as-is
    hub_path = tmp_path / "models" / "huggingface" / "hub"
    assert _resolve_hf_hub_cache(hub_path) == str(hub_path)

    # Path with existing hub subdirectory should resolve to hub
    custom_dir = tmp_path / "custom_cache"
    (custom_dir / "hub").mkdir(parents=True)
    assert _resolve_hf_hub_cache(custom_dir) == str(custom_dir / "hub")

    # Path without hub directory and not named huggingface preserved as-is
    plain_dir = tmp_path / "plain_dir"
    plain_dir.mkdir(parents=True)
    assert _resolve_hf_hub_cache(plain_dir) == str(plain_dir)


def test_resolve_hf_hub_cache_env_vars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    # Neither set
    assert _resolve_hf_hub_cache(None) is None

    # HF_HUB_CACHE takes precedence
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub_cache"))
    assert _resolve_hf_hub_cache(None) == str(tmp_path / "hub_cache")

    # HF_HOME when HF_HUB_CACHE is not set
    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "huggingface"))
    assert _resolve_hf_hub_cache(None) == str(tmp_path / "huggingface" / "hub")

    # HF_HOME already ending in hub
    monkeypatch.setenv("HF_HOME", str(tmp_path / "huggingface" / "hub"))
    assert _resolve_hf_hub_cache(None) == str(tmp_path / "huggingface" / "hub")


def test_create_embedder_default_cache_dir(tmp_path: Path) -> None:
    embedder = create_embedder(data_dir=tmp_path)
    assert isinstance(embedder, LocalBgeM3Embedder)
    expected_cache = tmp_path / "models" / "huggingface" / "hub"
    assert embedder.cache_dir == expected_cache
    assert expected_cache.exists()


# ==============================================================================
# 2. Config & Env Precedence Tests (AC3)
# ==============================================================================


def test_load_settings_gpu_defaults(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
embedding:
  provider: local
  device: auto
  fp16: true
  sparse_mode: auto
  max_length: 8192
ingestion:
  device: auto
""",
        encoding="utf-8",
    )
    settings = load_settings(cfg_file)
    assert settings.embedding.device == "auto"
    assert settings.embedding.fp16 is True
    assert settings.embedding.sparse_mode == "auto"
    assert settings.embedding.max_length == 8192
    assert settings.ingestion.device == "auto"


def test_load_settings_gpu_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("embedding: {}\ningestion: {}\n", encoding="utf-8")

    monkeypatch.setenv("EMBEDDING_DEVICE", "cuda")
    monkeypatch.setenv("EMBEDDING_FP16", "false")
    monkeypatch.setenv("EMBEDDING_SPARSE_MODE", "bm25")
    monkeypatch.setenv("EMBEDDING_MAX_LENGTH", "4096")
    monkeypatch.setenv("INGESTION_DEVICE", "cuda")

    settings = load_settings(cfg_file)
    assert settings.embedding.device == "cuda"
    assert settings.embedding.fp16 is False
    assert settings.embedding.sparse_mode == "bm25"
    assert settings.embedding.max_length == 4096
    assert settings.ingestion.device == "cuda"


def test_load_settings_memory_reclaim_defaults_and_env_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_yaml = tmp_path / "config_defaults.yaml"
    empty_yaml.write_text("{}", encoding="utf-8")

    # 1. Defaults verification
    settings = load_settings(empty_yaml)
    assert settings.ingestion.worker_idle_timeout_sec == 60
    assert settings.ingestion.worker_max_tasks == 0
    assert settings.ingestion.clear_cuda_cache_after_indexing is True

    # 2. Env overrides verification
    monkeypatch.setenv("INGESTION_WORKER_IDLE_TIMEOUT_SEC", "120")
    monkeypatch.setenv("INGESTION_WORKER_MAX_TASKS", "10")
    monkeypatch.setenv("INGESTION_CLEAR_CUDA_CACHE", "false")

    settings_env = load_settings(empty_yaml)
    assert settings_env.ingestion.worker_idle_timeout_sec == 120
    assert settings_env.ingestion.worker_max_tasks == 10
    assert settings_env.ingestion.clear_cuda_cache_after_indexing is False


# ==============================================================================
# 3. Full BGE-M3 Single-Pass Dense + Learned Sparse Unit Tests (AC4)
# ==============================================================================


class FakeTokenizer:
    all_special_ids = [0, 1, 2]


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.max_seq_length = 512

    def encode(self, texts: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        # Emit synthetic sentence and token embeddings
        results: list[dict[str, Any]] = []
        for _ in texts:
            # 4 tokens: special 0, word 101, repeated word 101, special 2
            input_ids = torch.tensor([0, 101, 101, 2], dtype=torch.long)
            # token embeddings: [4, 1024]
            token_embs = torch.zeros(4, 1024, dtype=torch.float32)
            token_embs[1] = 0.5  # first occurrence of 101
            token_embs[2] = 0.9  # second occurrence of 101
            sentence_emb = torch.ones(1024, dtype=torch.float32) / (1024**0.5)
            results.append(
                {
                    "input_ids": input_ids,
                    "token_embeddings": token_embs,
                    "sentence_embedding": sentence_emb,
                }
            )
        return results


def test_local_bge_m3_single_pass_sparse_and_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = LocalBgeM3Embedder(device="cpu", sparse_mode="bge_m3")
    embedder._model = FakeSentenceTransformer()

    # Create a linear head that projects token_embs to scalar weight = average of first elements
    sparse_linear = torch.nn.Linear(1024, 1, bias=False)
    with torch.no_grad():
        sparse_linear.weight.zero_()
        sparse_linear.weight[0, 0] = 1.0  # Output equals token_embs[i, 0]
    embedder._sparse_linear = sparse_linear

    assert embedder.supports_native_sparse is True

    dense_vecs, sparse_vecs = embedder.embed_documents_with_sparse_sync(["hello world"])
    assert len(dense_vecs) == 1
    assert len(sparse_vecs) == 1
    assert len(dense_vecs[0]) == 1024

    sparse_vec = sparse_vecs[0]
    # Special tokens 0 and 2 should be filtered out
    assert 0 not in sparse_vec.indices
    assert 2 not in sparse_vec.indices
    # Repeated token 101 should keep maximum weight: max(0.5, 0.9) = 0.9
    assert sparse_vec.indices == [101]
    assert pytest.approx(sparse_vec.values[0], abs=1e-4) == 0.9


def test_local_bge_m3_fallback_when_sparse_linear_unavailable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    embedder = LocalBgeM3Embedder(device="cpu", sparse_mode="bge_m3")
    # Simulate failed download of sparse_linear.pt
    with patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("Network offline")):
        with caplog.at_level(logging.WARNING):
            linear = embedder._get_sparse_linear()
    assert linear is None
    assert embedder.resolved_sparse_mode == "bm25"
    assert embedder.supports_native_sparse is False
    assert "falling back to BM25" in caplog.text


def test_bge_m3_sparse_encoder_delegation() -> None:
    embedder = LocalBgeM3Embedder(device="cpu", sparse_mode="bge_m3")
    embedder._model = FakeSentenceTransformer()
    sparse_linear = torch.nn.Linear(1024, 1, bias=False)
    with torch.no_grad():
        sparse_linear.weight.zero_()
        sparse_linear.weight[0, 0] = 1.0
    embedder._sparse_linear = sparse_linear

    encoder = BgeM3SparseEncoder(embedder=embedder)
    vecs = encoder.encode_sync(["test document"])
    assert len(vecs) == 1
    assert vecs[0].indices == [101]


# ==============================================================================
# 4. HybridIndexer & HybridRetriever Integration Tests (AC4)
# ==============================================================================


class SinglePassTrackedEmbedder(BaseEmbedder):
    """Embedder that tracks calls to ensure single-pass execution."""

    dimensions: int = 64

    def __init__(self) -> None:
        self.single_pass_call_count = 0
        self.separate_dense_call_count = 0

    @property
    def supports_native_sparse(self) -> bool:
        return True

    def embed_documents_sync(self, texts: list[str]) -> list[list[float]]:
        self.separate_dense_call_count += 1
        return [[0.1] * self.dimensions for _ in texts]

    def embed_documents_with_sparse_sync(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], list[models.SparseVector]]:
        self.single_pass_call_count += 1
        dense = [[0.1] * self.dimensions for _ in texts]
        sparse = [models.SparseVector(indices=[42, 99], values=[0.8, 0.4]) for _ in texts]
        return dense, sparse

    async def embed_documents_with_sparse(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], list[models.SparseVector]]:
        return self.embed_documents_with_sparse_sync(texts)


@pytest.mark.asyncio
async def test_hybrid_indexer_and_retriever_single_pass(tmp_path: Path) -> None:
    qdrant = AsyncQdrantClient(location=":memory:")
    collection = "test_gpu_collection"
    track_emb = SinglePassTrackedEmbedder()
    encoder = BgeM3SparseEncoder(embedder=MagicMock(supports_native_sparse=True))
    # Wire encoder to use track_emb
    encoder.encode_sync = lambda texts: track_emb.embed_documents_with_sparse_sync(texts)[1]  # type: ignore[assignment]

    indexer = HybridIndexer(
        qdrant=qdrant,
        embedder=track_emb,
        sparse_encoder=encoder,
        collection=collection,
        batch_size=10,
    )

    db_path = tmp_path / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    chunk = ChunkPayload(
        chunk_id="11111111-1111-1111-1111-111111111111",
        chunk_index=0,
        parent_id="parent_1",
        heading_path="Intro",
        page_numbers=[1],
        token_count=10,
        child_text="GPU acceleration test passage",
        contextualized_text="test.pdf > Intro: GPU acceleration test passage",
        parent_content="Parent GPU acceleration test passage",
        source_filename="test.pdf",
        file_type="pdf",
    )

    async with session_factory() as session:
        session.add(Document(id="test_doc_1", filename="test.pdf", file_type="pdf", sha256="1" * 64, status="indexing"))
        await session.commit()
        indexed_count = await indexer.index_chunks(session, "test_doc_1", [chunk])
    assert indexed_count == 1
    # Verify single-pass was executed, not separate calls
    assert track_emb.single_pass_call_count == 1
    assert track_emb.separate_dense_call_count == 0

    # Verify sparse_mode was stamped on payload
    point = await qdrant.retrieve(collection_name=collection, ids=["11111111-1111-1111-1111-111111111111"])
    assert len(point) == 1
    assert point[0].payload is not None
    assert point[0].payload.get("sparse_mode") == "bge_m3"

    # Verify retriever also uses single-pass
    retriever = HybridRetriever(
        qdrant=qdrant,
        embedder=track_emb,
        sparse_encoder=encoder,
        collection=collection,
    )
    results = await retriever.retrieve("GPU test query")
    assert len(results) >= 1
    assert track_emb.single_pass_call_count == 2
    assert track_emb.separate_dense_call_count == 0

    await engine.dispose()
    await qdrant.close()


@pytest.mark.asyncio
async def test_check_collection_sparse_mode_compatibility(caplog: pytest.LogCaptureFixture) -> None:
    qdrant = AsyncQdrantClient(location=":memory:")
    collection = "sparse_compat_test"
    await init_qdrant_collection(qdrant, collection, dimensions=16)

    # Insert a point with sparse_mode = "bm25"
    await qdrant.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(
                id="22222222-2222-2222-2222-222222222222",
                vector={
                    DENSE: [0.1] * 16,
                    SPARSE: models.SparseVector(indices=[1], values=[1.0]),
                },
                payload={"sparse_mode": "bm25"},
            )
        ],
    )

    # Active sparse_mode is "bge_m3" -> should log warning and return False
    with caplog.at_level(logging.WARNING):
        compat = await check_collection_sparse_mode_compatibility(qdrant, collection, "bge_m3")
    assert compat is False
    assert "Collection 'sparse_compat_test' contains points indexed with sparse_mode='bm25'" in caplog.text

    # Matching mode should return True
    compat_ok = await check_collection_sparse_mode_compatibility(qdrant, collection, "bm25")
    assert compat_ok is True

    await qdrant.close()


# ==============================================================================
# 5. Docling Accelerator Wiring Tests
# ==============================================================================


def test_document_parser_service_device_wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import AcceleratorDevice, PdfPipelineOptions

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    service_gpu = DocumentParserService(data_dir=tmp_path, device="cuda")
    converter_gpu = service_gpu._get_converter()
    format_opt = converter_gpu.format_to_options.get(InputFormat.PDF)
    assert format_opt is not None
    assert isinstance(format_opt.pipeline_options, PdfPipelineOptions)
    assert format_opt.pipeline_options.accelerator_options.device == AcceleratorDevice.CUDA

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    service_cpu = DocumentParserService(data_dir=tmp_path, device="auto")
    converter_cpu = service_cpu._get_converter()
    format_opt_cpu = converter_cpu.format_to_options.get(InputFormat.PDF)
    assert format_opt_cpu is not None
    assert format_opt_cpu.pipeline_options.accelerator_options.device == AcceleratorDevice.CPU


# ==============================================================================
# 6. Container Artifacts Contract Tests (AC1, AC2)
# ==============================================================================


def test_docker_artifacts_contracts() -> None:
    root = Path(__file__).resolve().parent.parent.parent

    # Dockerfile.gpu assertions
    dockerfile_gpu = root / "Dockerfile.gpu"
    assert dockerfile_gpu.is_file(), "Dockerfile.gpu must exist at repository root"
    content = dockerfile_gpu.read_text(encoding="utf-8")
    assert "TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124" in content
    assert "uv pip install" in content
    assert "--reinstall-package torch" in content
    assert "NVIDIA_VISIBLE_DEVICES=all" in content
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility" in content
    assert "appuser" in content

    # docker-compose.gpu.yml assertions
    compose_gpu = root / "docker-compose.gpu.yml"
    assert compose_gpu.is_file(), "docker-compose.gpu.yml must exist at repository root"
    raw_compose = yaml.safe_load(compose_gpu.read_text(encoding="utf-8"))
    sb = raw_compose.get("services", {}).get("second-brain", {})
    assert sb.get("build", {}).get("dockerfile") == "Dockerfile.gpu"

    deploy_res = sb.get("deploy", {}).get("resources", {}).get("reservations", {})
    devices = deploy_res.get("devices", [])
    assert len(devices) >= 1
    assert devices[0].get("driver") == "nvidia"
    assert devices[0].get("count") == "all"
    assert devices[0].get("capabilities") == ["gpu"]


# ==============================================================================
# 7. Live 50+ Page Benchmark Test (AC6)
# ==============================================================================


@pytest.mark.live
@pytest.mark.asyncio
async def test_gpu_benchmark_50_page_pdf(tmp_path: Path) -> None:
    """Benchmark ingestion throughput and VRAM envelope (<= 8 GB) on a 52-page document."""
    import time
    import pymupdf as fitz
    from src.ingestion.worker import IngestionOrchestrator

    # Generate synthetic 52-page PDF
    pdf_path = tmp_path / "benchmark_52_pages.pdf"
    doc = fitz.open()
    for page_idx in range(1, 53):
        page = doc.new_page(width=595, height=842)
        text = (
            f"Section {page_idx}: Enterprise Architecture & Data Ingestion Specification\n\n"
            f"This is page {page_idx} of 52 in the comprehensive technical benchmark document. "
            "It evaluates high-throughput hybrid retrieval combining dense semantic representations "
            "with learned sparse vector projections. The document covers system scaling, memory profiling, "
            "and container execution boundaries across heterogeneous accelerated runtime clusters.\n\n"
            "Key technical metrics:\n"
            f"- Latency budget: {page_idx * 12} milliseconds\n"
            f"- Ingestion throughput target: 12.5 pages per second\n"
            "- Target VRAM ceiling: <= 8192 MB (RTX 2000 Ada envelope)\n"
        )
        page.insert_text((50, 72), text, fontsize=11)
    doc.save(str(pdf_path))
    doc.close()

    assert pdf_path.is_file()

    # Track VRAM before and after
    cuda_active = torch.cuda.is_available()
    if cuda_active:
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    qdrant = AsyncQdrantClient(location=":memory:")
    db_file = tmp_path / "bench.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_file.as_posix()}")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    orchestrator = IngestionOrchestrator(
        session_factory=session_factory,
        qdrant=qdrant,
        data_dir=tmp_path,
        use_process_pool=False,
    )

    doc_record = await orchestrator.ingest_file(pdf_path, filename="benchmark_52_pages.pdf")
    total_elapsed = time.perf_counter() - t0

    assert doc_record.status == "ready"
    assert doc_record.page_count == 52
    assert doc_record.chunk_count > 0

    if cuda_active:
        peak_vram_bytes = torch.cuda.max_memory_allocated()
        peak_vram_gb = peak_vram_bytes / (1024**3)
        # AC6 constraint: peak VRAM must be <= 8 GB (typically < 3.5 GB)
        assert peak_vram_gb < 8.0, f"Peak VRAM {peak_vram_gb:.2f} GB exceeded 8 GB envelope"

    # Reclaim worker pool and clear CUDA cache, verifying VRAM drops back to baseline
    assert orchestrator.get_worker_state()["active_worker_jobs"] == 0
    reclaimed = orchestrator.reclaim_worker_pool(wait=True)
    assert orchestrator.get_worker_state()["worker_pool_active"] is False

    cuda_cleared = release_cuda_cache()
    if cuda_active:
        assert cuda_cleared is True
        post_reclaim_vram_gb = torch.cuda.memory_allocated() / (1024**3)
        assert post_reclaim_vram_gb < 2.0, f"Post-reclaim VRAM {post_reclaim_vram_gb:.2f} GB exceeded 2 GB envelope"

    await engine.dispose()
    await qdrant.close()



# ==============================================================================
# 8. Adaptive Memory & VRAM Reclaim Tests (Phase 3)
# ==============================================================================


@pytest.mark.asyncio
async def test_release_cuda_cache_calls_empty_cache_when_cuda_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Direct release_cuda_cache behavior
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with patch.object(torch.cuda, "empty_cache") as mock_empty:
        assert release_cuda_cache() is True
        mock_empty.assert_called_once()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert release_cuda_cache() is False

    def raise_err():
        raise RuntimeError("CUDA device error")

    monkeypatch.setattr(torch.cuda, "is_available", raise_err)
    assert release_cuda_cache() is False

    # 2. HybridIndexer.index_chunks invoking empty_cache when clear_cuda_cache=True and skipping when False
    db_file = tmp_path / "test_indexer_reclaim.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_file.as_posix()}")
    await init_db(engine)
    session_factory = create_session_factory(engine)

    async with session_factory() as session:
        session.add(
            Document(id="doc-reclaim-1", filename="1.txt", file_type="txt", sha256="sha-rec-1", size_bytes=10, status="queued")
        )
        session.add(
            Document(id="doc-reclaim-2", filename="2.txt", file_type="txt", sha256="sha-rec-2", size_bytes=10, status="queued")
        )
        await session.commit()

    qdrant = AsyncQdrantClient(location=":memory:")
    embedder = DeterministicStubEmbedder(dimensions=32)
    sparse_encoder = DeterministicBM25Encoder()

    chunk1 = ChunkPayload(
        chunk_id="00000000-0000-0000-0000-000000000001",
        parent_id=None,
        chunk_index=0,
        heading_path="Root",
        page_numbers=[1],
        token_count=10,
        child_text="chunk text 1",
        parent_content="chunk text 1",
        contextualized_text="chunk text 1",
        source_filename="1.txt",
        file_type="txt",
    )
    chunk2 = ChunkPayload(
        chunk_id="00000000-0000-0000-0000-000000000002",
        parent_id=None,
        chunk_index=0,
        heading_path="Root",
        page_numbers=[1],
        token_count=10,
        child_text="chunk text 2",
        parent_content="chunk text 2",
        contextualized_text="chunk text 2",
        source_filename="2.txt",
        file_type="txt",
    )

    indexer_enabled = HybridIndexer(
        qdrant=qdrant,
        embedder=embedder,
        sparse_encoder=sparse_encoder,
        collection="test_cuda_reclaim_enabled",
        clear_cuda_cache=True,
    )
    indexer_disabled = HybridIndexer(
        qdrant=qdrant,
        embedder=embedder,
        sparse_encoder=sparse_encoder,
        collection="test_cuda_reclaim_disabled",
        clear_cuda_cache=False,
    )

    with patch("src.ingestion.indexer.release_cuda_cache") as mock_rel:
        async with session_factory() as session:
            count = await indexer_enabled.index_chunks(session, doc_id="doc-reclaim-1", chunks=[chunk1])
            assert count == 1
        mock_rel.assert_called_once()

    with patch("src.ingestion.indexer.release_cuda_cache") as mock_rel:
        async with session_factory() as session:
            count = await indexer_disabled.index_chunks(session, doc_id="doc-reclaim-2", chunks=[chunk2])
            assert count == 1
        mock_rel.assert_not_called()

    await qdrant.close()
    await engine.dispose()


@pytest.mark.asyncio
async def test_worker_max_tasks_recycles_only_after_batch_idle_never_mid_document(tmp_path: Path) -> None:
    db_file = tmp_path / "test_max_tasks.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_file.as_posix()}")
    await init_db(engine)
    session_factory = create_session_factory(engine)

    async with session_factory() as session:
        for i in range(1, 4):
            session.add(
                Document(
                    id=f"doc-mt-{i}",
                    filename=f"{i}.txt",
                    file_type="txt",
                    sha256=f"sha-mt-{i}",
                    size_bytes=10,
                    status="queued",
                )
            )
        await session.commit()

    qdrant = AsyncQdrantClient(location=":memory:")
    embedder = DeterministicStubEmbedder(dimensions=32)
    sparse_encoder = DeterministicBM25Encoder()

    orchestrator = IngestionOrchestrator(
        session_factory=session_factory,
        qdrant=qdrant,
        embedder=embedder,
        sparse_encoder=sparse_encoder,
        data_dir=tmp_path,
        max_workers=2,
        worker_max_tasks=3,
        use_process_pool=True,
    )

    initial_pool = orchestrator.get_pool()
    assert initial_pool is not None
    assert orchestrator.get_worker_state()["worker_pool_active"] is True

    mock_parse = ParseResult(
        sha256="dummy",
        filename="dummy.txt",
        file_type="txt",
        markdown="dummy content",
        markdown_path=tmp_path / "dummy.md",
        parsed_path=tmp_path / "dummy.json",
        page_count=1,
    )

    worker_pool_references: list[Any] = []
    loop = asyncio.get_running_loop()

    async def fake_run_in_executor(executor: Any, func: Any, *args: Any) -> Any:
        func_name = getattr(func, "__name__", "")
        if func_name == "_worker_parse_task":
            worker_pool_references.append(("parse", executor))
            return mock_parse, 1
        elif func_name == "_worker_chunk_task":
            worker_pool_references.append(("chunk", executor))
            unique_chunks = [
                ChunkPayload(
                    chunk_id=str(uuid.uuid4()),
                    parent_id=None,
                    chunk_index=0,
                    heading_path="Root",
                    page_numbers=[1],
                    token_count=10,
                    child_text="dummy chunk",
                    parent_content="dummy chunk",
                    contextualized_text="dummy chunk",
                    source_filename="dummy.txt",
                    file_type="txt",
                )
            ]
            return unique_chunks, 1
        fut = loop.create_future()
        fut.set_result(func(*args))
        return await fut

    with (
        patch.object(loop, "run_in_executor", side_effect=fake_run_in_executor),
        patch.object(orchestrator, "_update_state", new=AsyncMock()),
    ):
        # 1. Process document 1
        await orchestrator.process_document("doc-mt-1")
        # After doc 1, completed_docs == 1 (< 2), pool remains alive and same instance
        assert orchestrator._pool is initial_pool
        assert orchestrator._pool_completed_docs == 1
        assert orchestrator.get_worker_state()["worker_reclaim_count"] == 0

        # 2. Run doc 2 and doc 3 concurrently to demonstrate same worker kept across Stage 1 + Stage 2
        # and across concurrent uploads until idle
        await asyncio.gather(
            orchestrator.process_document("doc-mt-2"),
            orchestrator.process_document("doc-mt-3"),
        )

        # Confirm parse and chunk tasks all executed on the exact same initial pool instance
        assert len(worker_pool_references) == 6  # 3 docs * 2 stages
        for stage, pool_ref in worker_pool_references:
            assert pool_ref is initial_pool

        # Once the batch is idle and completed docs >= 2 (here 3), pool is recycled
        assert orchestrator._pool is None
        assert orchestrator._pool_reclaim_count == 1
        assert orchestrator._pool_completed_docs == 0

    orchestrator.shutdown(wait=True)
    await qdrant.close()
    await engine.dispose()


@pytest.mark.asyncio
async def test_multi_file_concurrent_upload_does_not_reclaim_or_flush_cuda_mid_batch(tmp_path: Path) -> None:
    db_file = tmp_path / "test_concurrency.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_file.as_posix()}")
    await init_db(engine)
    session_factory = create_session_factory(engine)

    async with session_factory() as session:
        for i in range(1, 5):
            session.add(
                Document(
                    id=f"doc-conc-{i}",
                    filename=f"{i}.txt",
                    file_type="txt",
                    sha256=f"sha-conc-{i}",
                    size_bytes=10,
                    status="queued",
                )
            )
        await session.commit()

    qdrant = AsyncQdrantClient(location=":memory:")
    embedder = DeterministicStubEmbedder(dimensions=32)
    sparse_encoder = DeterministicBM25Encoder()

    # Bound concurrency to 2 with max_workers=2
    orchestrator = IngestionOrchestrator(
        session_factory=session_factory,
        qdrant=qdrant,
        embedder=embedder,
        sparse_encoder=sparse_encoder,
        data_dir=tmp_path,
        max_workers=2,
        clear_cuda_cache=True,
        worker_idle_timeout_sec=60,
    )

    initial_pool = orchestrator.get_pool()
    assert initial_pool is not None

    mock_parse = ParseResult(
        sha256="dummy",
        filename="dummy.txt",
        file_type="txt",
        markdown="dummy content",
        markdown_path=tmp_path / "dummy.md",
        parsed_path=tmp_path / "dummy.json",
        page_count=1,
    )

    current_concurrent_docs = 0
    max_concurrent_docs = 0
    cuda_release_active_indexing_snapshots: list[int] = []
    pool_alive_during_batch: list[bool] = []

    def mock_release_cuda() -> bool:
        cuda_release_active_indexing_snapshots.append(orchestrator._active_indexing_jobs)
        return True

    real_index_chunks = HybridIndexer.index_chunks

    async def tracking_index_chunks(self_idx: Any, session: Any, doc_id: str, chunks: Any) -> int:
        nonlocal current_concurrent_docs, max_concurrent_docs
        current_concurrent_docs += 1
        if current_concurrent_docs > max_concurrent_docs:
            max_concurrent_docs = current_concurrent_docs

        pool_alive_during_batch.append(orchestrator._pool is not None)
        await asyncio.sleep(0.05)
        res = await real_index_chunks(self_idx, session, doc_id, chunks)
        current_concurrent_docs -= 1
        return res

    loop = asyncio.get_running_loop()

    async def fake_run_in_executor(executor: Any, func: Any, *args: Any) -> Any:
        func_name = getattr(func, "__name__", "")
        if func_name == "_worker_parse_task":
            return mock_parse, 1
        elif func_name == "_worker_chunk_task":
            unique_chunks = [
                ChunkPayload(
                    chunk_id=str(uuid.uuid4()),
                    parent_id=None,
                    chunk_index=0,
                    heading_path="Root",
                    page_numbers=[1],
                    token_count=10,
                    child_text="dummy chunk",
                    parent_content="dummy chunk",
                    contextualized_text="dummy chunk",
                    source_filename="dummy.txt",
                    file_type="txt",
                )
            ]
            return unique_chunks, 1
        fut = loop.create_future()
        fut.set_result(func(*args))
        return await fut

    with (
        patch.object(loop, "run_in_executor", side_effect=fake_run_in_executor),
        patch("src.ingestion.worker.release_cuda_cache", side_effect=mock_release_cuda),
        patch.object(HybridIndexer, "index_chunks", tracking_index_chunks),
        patch.object(orchestrator, "_update_state", new=AsyncMock()),
    ):
        async def trigger_mid_batch_reclaim() -> None:
            await asyncio.sleep(0.02)
            # Reclaim requested while files are active -> must defer and NOT reap pool yet
            reclaimed = orchestrator.reclaim_worker_pool(wait=False, force=False)
            assert reclaimed is False
            assert orchestrator.get_worker_state()["reclaim_deferred"] is True
            assert orchestrator._pool is not None

        await asyncio.gather(
            orchestrator.process_document("doc-conc-1"),
            orchestrator.process_document("doc-conc-2"),
            orchestrator.process_document("doc-conc-3"),
            orchestrator.process_document("doc-conc-4"),
            trigger_mid_batch_reclaim(),
        )

    # 1. Semaphore bounded concurrency to max_workers=2
    assert max_concurrent_docs <= 2

    # 2. Worker pool was alive for all files during the batch
    assert len(pool_alive_during_batch) == 4
    assert all(pool_alive_during_batch)

    # 3. release_cuda_cache was only called when _active_indexing_jobs == 0
    assert len(cuda_release_active_indexing_snapshots) >= 1
    for active_snap in cuda_release_active_indexing_snapshots:
        assert active_snap == 0

    # 4. Immediate deferred reclaim fired the moment the last document finished
    assert orchestrator.get_worker_state()["active_worker_jobs"] == 0
    assert orchestrator._pool is None
    assert orchestrator._pool_reclaim_count == 1
    assert orchestrator.get_worker_state()["reclaim_deferred"] is False

    orchestrator.shutdown(wait=True)
    await qdrant.close()
    await engine.dispose()


@pytest.mark.asyncio
async def test_worker_idle_reaper_keeps_warm_during_batch_and_reclaims_on_idle(tmp_path: Path) -> None:
    db_file = tmp_path / "test_reaper.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_file.as_posix()}")
    await init_db(engine)
    session_factory = create_session_factory(engine)

    async with session_factory() as session:
        for i in range(1, 4):
            session.add(
                Document(
                    id=f"doc-reap-{i}",
                    filename=f"{i}.txt",
                    file_type="txt",
                    sha256=f"sha-reap-{i}",
                    size_bytes=10,
                    status="queued",
                )
            )
        await session.commit()

    qdrant = AsyncQdrantClient(location=":memory:")
    embedder = DeterministicStubEmbedder(dimensions=32)
    sparse_encoder = DeterministicBM25Encoder()

    orchestrator = IngestionOrchestrator(
        session_factory=session_factory,
        qdrant=qdrant,
        embedder=embedder,
        sparse_encoder=sparse_encoder,
        data_dir=tmp_path,
        worker_idle_timeout_sec=0.15,
        use_process_pool=True,
    )

    mock_parse = ParseResult(
        sha256="dummy",
        filename="dummy.txt",
        file_type="txt",
        markdown="dummy content",
        markdown_path=tmp_path / "dummy.md",
        parsed_path=tmp_path / "dummy.json",
        page_count=1,
    )

    loop = asyncio.get_running_loop()

    async def fake_run_in_executor(executor: Any, func: Any, *args: Any) -> Any:
        func_name = getattr(func, "__name__", "")
        if func_name == "_worker_parse_task":
            return mock_parse, 1
        elif func_name == "_worker_chunk_task":
            unique_chunks = [
                ChunkPayload(
                    chunk_id=str(uuid.uuid4()),
                    parent_id=None,
                    chunk_index=0,
                    heading_path="Root",
                    page_numbers=[1],
                    token_count=10,
                    child_text="dummy chunk",
                    parent_content="dummy chunk",
                    contextualized_text="dummy chunk",
                    source_filename="dummy.txt",
                    file_type="txt",
                )
            ]
            return unique_chunks, 1
        fut = loop.create_future()
        fut.set_result(func(*args))
        return await fut

    with (
        patch.object(loop, "run_in_executor", side_effect=fake_run_in_executor),
        patch.object(orchestrator, "_update_state", new=AsyncMock()),
    ):
        # 1. Ingest doc 1
        await orchestrator.process_document("doc-reap-1")
        pool1 = orchestrator._pool
        assert pool1 is not None

        # 2. Immediately ingest doc 2 (within warm window)
        await orchestrator.process_document("doc-reap-2")
        assert orchestrator._pool is pool1
        assert orchestrator.get_worker_state()["worker_reclaim_count"] == 0

        # 3. Wait 0.25s for 0.15s idle timer to reap worker pool
        await asyncio.sleep(0.25)
        assert orchestrator._pool is None
        assert orchestrator.get_worker_state()["worker_reclaim_count"] == 1

        # 4. Ingest doc 3 -> fresh pool lazily spawns
        await orchestrator.process_document("doc-reap-3")
        pool2 = orchestrator._pool
        assert pool2 is not None
        assert pool2 is not pool1

    orchestrator.shutdown(wait=True)
    await qdrant.close()
    await engine.dispose()


@pytest.mark.asyncio
async def test_reclaim_worker_pool_defers_when_job_active_and_reclaims_on_release(tmp_path: Path) -> None:
    orchestrator = IngestionOrchestrator(
        data_dir=tmp_path,
        worker_idle_timeout_sec=60,
    )
    pool = orchestrator.get_pool()
    assert pool is not None
    assert orchestrator.get_worker_state()["worker_pool_active"] is True

    # 1. Acquire slot: active_worker_jobs becomes 1
    orchestrator._acquire_worker_slot()
    assert orchestrator.get_worker_state()["active_worker_jobs"] == 1

    # 2. Reclaim while active -> defers
    reclaimed = orchestrator.reclaim_worker_pool(wait=False, force=False)
    assert reclaimed is False
    assert orchestrator._reclaim_on_idle is True
    assert orchestrator._pool is not None
    assert orchestrator.get_worker_state()["reclaim_deferred"] is True

    # 3. Release slot -> immediately reclaims pool
    orchestrator._release_worker_slot()
    assert orchestrator._pool is None
    assert orchestrator._reclaim_on_idle is False
    assert orchestrator.get_worker_state()["worker_pool_active"] is False
    assert orchestrator.get_worker_state()["worker_reclaim_count"] == 1
    assert orchestrator.get_worker_state()["reclaim_deferred"] is False

    orchestrator.shutdown(wait=True)
