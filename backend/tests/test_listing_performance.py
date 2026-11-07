"""Listing Performance Benchmark (Phase 4).

Verifies:
1. GET /api/v1/documents executes in < 20ms for 100 documents.
2. Zero filesystem reads to data/parsed/ or data/markdown/ occur during listing.
3. All 100 items expose parsing_engine and engine_fallback_reason.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from unittest.mock import patch
import builtins

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app
from src.config import get_settings
from src.db.models import Document, IngestionJob
from src.db.session import create_engine, create_session_factory, init_db
from src.ingestion.indexer import (
    DeterministicBM25Encoder,
    DeterministicStubEmbedder,
    set_embedder,
    set_sparse_encoder,
)
from src.rag.graph import close_checkpointer
from src.rag.llm import set_chat_model
from src.rag.reranker import set_reranker
from src.rag.retriever import set_active_qdrant_client

STUB_DIM = 32


@pytest.fixture
async def benchmark_client(tmp_path: Path, monkeypatch):
    """Run FastAPI app inside LifespanManager with 100 pre-populated documents in SQLite."""
    data_dir = tmp_path / "data"
    db_dir = tmp_path / "data" / "db"
    parsed_dir = data_dir / "parsed"
    markdown_dir = data_dir / "markdown"
    dist_dir = tmp_path / "frontend_dist"

    for d in (data_dir, db_dir, parsed_dir, markdown_dir, dist_dir):
        d.mkdir(parents=True, exist_ok=True)

    (dist_dir / "index.html").write_text(
        "<!doctype html><html><body>Test SPA</body></html>", encoding="utf-8"
    )

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("DB_DIR", str(db_dir))
    monkeypatch.setenv("FRONTEND_DIST_DIR", str(dist_dir))
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("INGESTION_USE_PROCESS_POOL", "false")
    monkeypatch.delenv("PREFER_PYMUPDF4LLM", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    get_settings.cache_clear()
    set_embedder(DeterministicStubEmbedder(dimensions=STUB_DIM))
    set_sparse_encoder(DeterministicBM25Encoder())
    set_chat_model(None)
    set_reranker(None)
    set_active_qdrant_client(None)

    # Pre-seed 100 documents in SQLite
    sqlite_url = f"sqlite+aiosqlite:///{(db_dir / 'second_brain.db').as_posix()}"
    engine = create_engine(sqlite_url)
    await init_db(engine)
    session_factory = create_session_factory(engine)

    now = datetime.now(timezone.utc)
    docs_to_add: list[Document] = []
    jobs_to_add: list[IngestionJob] = []

    for i in range(100):
        doc_id = f"bench_doc_{i:03d}"
        sha = f"{i:03d}" + "a" * 61
        is_docling = (i % 2 == 0)
        engine_name = "docling" if is_docling else "pymupdf4llm"
        fallback_reason = None if is_docling else "Docling GPU failed (CUDA OOM); fell back to PyMuPDF CPU"

        doc = Document(
            id=doc_id,
            filename=f"document_{i:03d}.pdf",
            file_type="pdf",
            sha256=sha,
            size_bytes=1024 * (i + 1),
            page_count=i + 1,
            chunk_count=(i + 1) * 2,
            status="ready",
            parsing_engine=engine_name,
            engine_fallback_reason=fallback_reason,
            created_at=now,
            updated_at=now,
        )
        job = IngestionJob(
            id=f"job_{i:03d}",
            doc_id=doc_id,
            status="ready",
            stage="ready",
            progress=1.0,
            duration_ms=250 + i * 5,
            stage_timings='{"parsing_ms": 100, "chunking_ms": 50, "indexing_ms": 100}',
            created_at=now,
            updated_at=now,
        )
        docs_to_add.append(doc)
        jobs_to_add.append(job)

        # Write dummy checkpoint files to disk so they exist if something illicitly tries to read them
        parsed_payload = {
            "sha256": sha,
            "filename": doc.filename,
            "file_type": "pdf",
            "page_count": doc.page_count,
            "extra_meta": {"engine": engine_name, "engine_fallback_reason": fallback_reason},
        }
        (parsed_dir / f"{sha}.json").write_text(json.dumps(parsed_payload), encoding="utf-8")
        (markdown_dir / f"{sha}.md").write_text(f"# Document {i}\n\nContent", encoding="utf-8")

    async with session_factory() as session:
        for doc, job in zip(docs_to_add, jobs_to_add):
            session.add(doc)
            session.add(job)
        await session.commit()

    await engine.dispose()

    test_app = create_app()
    async with LifespanManager(test_app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            client.app = test_app
            yield client, data_dir

    set_chat_model(None)
    set_embedder(None)
    set_sparse_encoder(None)
    set_reranker(None)
    set_active_qdrant_client(None)
    await close_checkpointer()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_listing_performance_and_zero_disk_io(benchmark_client):
    """Verify GET /api/v1/documents returns 100 docs in < 20ms with 0 disk reads to parsed/markdown."""
    client, data_dir = benchmark_client

    # 1. Warm-up request
    warmup_resp = await client.get("/api/v1/documents?limit=100")
    assert warmup_resp.status_code == 200
    assert len(warmup_resp.json()["items"]) == 100

    # 2. Filesystem read guard
    orig_open = builtins.open
    orig_read_text = Path.read_text

    disk_reads_to_parsed_or_markdown: list[str] = []

    def guarded_open(file, *args, **kwargs):
        path_str = str(file).replace("\\", "/")
        if "/data/parsed/" in path_str or "/data/markdown/" in path_str:
            disk_reads_to_parsed_or_markdown.append(path_str)
            raise AssertionError(f"Illicit filesystem open detected: {path_str}")
        return orig_open(file, *args, **kwargs)

    def guarded_read_text(self, *args, **kwargs):
        path_str = str(self).replace("\\", "/")
        if "/data/parsed/" in path_str or "/data/markdown/" in path_str:
            disk_reads_to_parsed_or_markdown.append(path_str)
            raise AssertionError(f"Illicit filesystem read_text detected: {path_str}")
        return orig_read_text(self, *args, **kwargs)

    # 3. Benchmark 5 iterations with filesystem guard active
    durations: list[float] = []
    with patch("builtins.open", side_effect=guarded_open), patch.object(
        Path, "read_text", side_effect=guarded_read_text
    ):
        for _ in range(5):
            t0 = time.perf_counter()
            resp = await client.get("/api/v1/documents?limit=100")
            elapsed_ms = (time.perf_counter() - t0) * 1000
            durations.append(elapsed_ms)
            assert resp.status_code == 200

    assert len(disk_reads_to_parsed_or_markdown) == 0, (
        f"Detected {len(disk_reads_to_parsed_or_markdown)} filesystem reads to parsed/markdown: "
        f"{disk_reads_to_parsed_or_markdown}"
    )

    # 4. Verify all 100 items contain parsing_engine and engine_fallback_reason
    data = resp.json()
    items = data["items"]
    assert len(items) == 100

    docling_count = 0
    pymupdf_count = 0
    fallback_count = 0

    for item in items:
        engine = item.get("parsing_engine")
        fallback = item.get("engine_fallback_reason")
        assert engine in ("docling", "pymupdf4llm"), f"Unexpected engine: {engine}"
        if engine == "docling":
            docling_count += 1
            assert fallback is None
        else:
            pymupdf_count += 1
            assert fallback == "Docling GPU failed (CUDA OOM); fell back to PyMuPDF CPU"
            fallback_count += 1

    assert docling_count == 50
    assert pymupdf_count == 50
    assert fallback_count == 50

    # 5. Verify listing latency: median response time is < 20ms
    sorted_durations = sorted(durations)
    median_duration = sorted_durations[len(sorted_durations) // 2]
    min_duration = min(durations)
    print(f"\n[Listing Benchmark] 100 items: min={min_duration:.2f}ms, median={median_duration:.2f}ms, all={durations}")

    assert median_duration < 20.0, (
        f"Listing 100 documents took median {median_duration:.2f}ms (threshold: 20ms)"
    )

