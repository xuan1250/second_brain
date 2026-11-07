"""Integration tests for Phase 4 FastAPI Headless Gateway & Streaming API."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import openpyxl
import pymupdf
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import io
from unittest.mock import MagicMock, patch

import torch

from src.api.main import create_app
from src.api.routes.system import _get_process_rss_mb, collect_memory_stats
from src.api.schemas import MemoryReclaimResponse, MemoryStatsResponse, SystemStatsResponse
from src.config import get_settings
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


def parse_sse_stream(raw_body: str) -> list[dict[str, Any]]:
    """Parse raw SSE wire text (`event: ...\ndata: ...\n\n`) into structured event dicts."""
    frames: list[dict[str, Any]] = []
    for block in raw_body.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.partition(":")[2].strip()
            elif line.startswith("data:"):
                data_lines.append(line.partition(":")[2].strip())
        data_raw = "\n".join(data_lines)
        frames.append(
            {
                "event": event_name,
                "data": json.loads(data_raw) if data_raw else {},
            }
        )
    return frames


@pytest.fixture
def sample_xlsx_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "q3_metrics.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "FinancialMetrics"
    ws.append(["Metric", "Q3_2025", "YoY_Change"])
    ws.append(["Operating Margin", "24.8%", "+3.2%"])
    ws.append(["Gross Profit", "$142M", "+18.0%"])
    wb.save(path)
    return path.read_bytes()


@pytest.fixture
def sample_pdf_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "board_summary.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1. Executive Summary", fontsize=16)
    page.insert_text(
        (72, 110),
        "Operating margin expanded by 3.2% year-over-year driven by supply chain automation.",
        fontsize=11,
    )
    doc.save(str(path))
    doc.close()
    return path.read_bytes()


@pytest.fixture
async def api_client(tmp_path: Path, monkeypatch):
    """Run FastAPI inside `LifespanManager` with isolated temp storage, stub embedder, and in-memory Qdrant."""
    data_dir = tmp_path / "data"
    db_dir = tmp_path / "data" / "db"
    dist_dir = tmp_path / "frontend_dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "index.html").write_text("<!doctype html><html><body>Second Brain SPA</body></html>", encoding="utf-8")

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("DB_DIR", str(db_dir))
    monkeypatch.setenv("FRONTEND_DIST_DIR", str(dist_dir))
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("INGESTION_USE_PROCESS_POOL", "false")
    monkeypatch.setenv("PREFER_PYMUPDF4LLM", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    get_settings.cache_clear()
    set_embedder(DeterministicStubEmbedder(dimensions=STUB_DIM))
    set_sparse_encoder(DeterministicBM25Encoder())
    set_chat_model(None)
    set_reranker(None)
    set_active_qdrant_client(None)

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


async def _wait_for_document_ready(client: AsyncClient, doc_id: str, timeout_s: float = 15.0) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last_payload: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        resp = await client.get(f"/api/v1/documents/{doc_id}/status")
        assert resp.status_code == 200
        last_payload = resp.json()
        if last_payload["status"] in ("ready", "failed"):
            return last_payload
        if last_payload["status"] == "awaiting_approval":
            await client.post(f"/api/v1/documents/{doc_id}/approve-classification", json={})
        await asyncio.sleep(0.05)
    raise AssertionError(f"Document {doc_id} did not reach ready status in time: {last_payload}")


async def test_health_stats_openapi_and_static_spa(api_client):
    """Verify `/api/v1/system/health`, `/api/v1/system/stats`, OpenAPI `/docs`, and SPA static serving."""
    client, _ = api_client

    # 1. OpenAPI docs endpoint
    docs_resp = await client.get("/docs")
    assert docs_resp.status_code == 200

    # 2. Health endpoint
    health_resp = await client.get("/api/v1/system/health")
    assert health_resp.status_code == 200
    health = health_resp.json()
    assert health["status"] == "ok"
    assert health["database"] == "ok"
    assert health["qdrant"] == "ok"
    assert health["collection"] == "second_brain_knowledge"

    # 3. Stats endpoint (read-only configuration + counts)
    stats_resp = await client.get("/api/v1/system/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total_documents"] == 0
    assert stats["total_chunks"] == 0
    assert stats["qdrant_vectors_count"] == 0
    assert stats["embedding"]["model_name"] == "BAAI/bge-m3"
    assert stats["reranker"]["model_name"] == "ms-marco-MultiBERT-L-12"
    assert isinstance(stats["configured_providers"], list)
    assert any(p["provider"] == "openai" and p["configured"] is True for p in stats["available_providers"])

    # 4. Verify PATCH /api/v1/system/config is not exposed (read-only config)
    patch_resp = await client.patch("/api/v1/system/config", json={"provider": "anthropic"})
    assert patch_resp.status_code in (404, 405)

    # 5. Conditional static SPA mount serves index.html at /
    spa_resp = await client.get("/")
    assert spa_resp.status_code == 200
    assert "Second Brain SPA" in spa_resp.text


async def test_document_upload_deduplication_preview_and_atomic_delete(
    api_client,
    sample_xlsx_bytes: bytes,
):
    """Verify upload -> status polling -> deduplication -> list filters -> raw/markdown preview -> atomic delete."""
    client, data_dir = api_client

    # Reject unsupported file extension
    bad_resp = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.txt", b"plain text not allowed", "text/plain")},
    )
    assert bad_resp.status_code == 400

    # 1. Upload valid XLSX document -> 202 Accepted
    up_resp = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("q3_metrics.xlsx", sample_xlsx_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert up_resp.status_code == 202
    up_data = up_resp.json()
    doc_id = up_data["id"]
    sha256 = up_data["sha256"]
    assert up_data["filename"] == "q3_metrics.xlsx"
    assert up_data["file_type"] == "xlsx"
    assert up_data["duplicate"] is False

    # 2. Poll `/status` until ingestion completes
    status_data = await _wait_for_document_ready(client, doc_id)
    assert status_data["status"] == "ready"
    assert status_data["chunk_count"] >= 1
    assert status_data["progress"] == 1.0
    assert "indexing_ms" in status_data["stage_timings"]

    # 3. Re-upload identical file -> 200 OK with duplicate=True (idempotency without re-parsing)
    dup_resp = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("q3_metrics_copy.xlsx", sample_xlsx_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert dup_resp.status_code == 200
    dup_data = dup_resp.json()
    assert dup_data["id"] == doc_id
    assert dup_data["duplicate"] is True
    assert dup_data["status"] == "ready"

    # 4. Paginated list & filters
    list_resp = await client.get("/api/v1/documents", params={"status": "ready", "file_type": "xlsx", "search": "q3_metrics"})
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total"] == 1
    assert len(list_data["items"]) == 1
    assert list_data["documents"][0]["id"] == doc_id

    # 5. Raw file endpoint
    raw_resp = await client.get(f"/api/v1/documents/{doc_id}/raw")
    assert raw_resp.status_code == 200
    assert raw_resp.content == sample_xlsx_bytes
    assert "q3_metrics.xlsx" in raw_resp.headers.get("content-disposition", "")

    # 6. Markdown mirror & extracted heading outline endpoint
    md_resp = await client.get(f"/api/v1/documents/{doc_id}/markdown")
    assert md_resp.status_code == 200
    md_data = md_resp.json()
    assert md_data["doc_id"] == doc_id
    assert "Operating Margin" in md_data["markdown"]
    assert len(md_data["headings"]) >= 2
    assert any("FinancialMetrics" in h["heading_path"] for h in md_data["headings"])

    # 7. Check system stats reflect indexed document and vectors
    stats_mid = (await client.get("/api/v1/system/stats")).json()
    assert stats_mid["total_documents"] == 1
    assert stats_mid["ready_documents"] == 1
    assert stats_mid["total_chunks"] == status_data["chunk_count"]
    assert stats_mid["qdrant_vectors_count"] == status_data["chunk_count"]

    # 8. Atomic deletion across Qdrant, SQLite, and disk artifacts
    del_resp = await client.delete(f"/api/v1/documents/{doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # Verify 404 after deletion and 0 vectors / 0 artifacts remaining
    assert (await client.get(f"/api/v1/documents/{doc_id}")).status_code == 404
    assert (await client.delete(f"/api/v1/documents/{doc_id}")).status_code == 404
    stats_after = (await client.get("/api/v1/system/stats")).json()
    assert stats_after["total_documents"] == 0
    assert stats_after["total_chunks"] == 0
    assert stats_after["qdrant_vectors_count"] == 0
    assert not any(data_dir.rglob(f"{sha256}*"))


async def test_document_retry_endpoint(
    api_client,
    sample_pdf_bytes: bytes,
):
    """Verify POST /api/v1/documents/{doc_id}/retry handles 404, 409, ready no-op, and retry execution."""
    client, data_dir = api_client

    # 1. 404 on nonexistent doc_id
    resp = await client.post("/api/v1/documents/nonexistent-id/retry")
    assert resp.status_code == 404

    # 2. Upload valid PDF
    up_resp = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("board_summary.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert up_resp.status_code == 202
    doc_id = up_resp.json()["id"]

    # Poll until ready
    status_data = await _wait_for_document_ready(client, doc_id)
    assert status_data["status"] == "ready"

    # 3. Retry on ready document without force_reparse -> returns existing document (no-op)
    retry_noop = await client.post(f"/api/v1/documents/{doc_id}/retry")
    assert retry_noop.status_code == 200
    assert retry_noop.json()["status"] == "ready"

    # 4. Retry with force_reparse=true -> re-enqueues document
    retry_force = await client.post(f"/api/v1/documents/{doc_id}/retry", params={"force_reparse": "true"})
    assert retry_force.status_code == 200
    assert retry_force.json()["status"] == "queued"

    # Poll until ready again
    status_data2 = await _wait_for_document_ready(client, doc_id)
    assert status_data2["status"] == "ready"

    # 5. Verify 409 Conflict when document is processing
    from src.db.session import get_session_factory
    from src.db.models import Document
    factory = get_session_factory()
    async with factory() as session:
        doc = await session.get(Document, doc_id)
        assert doc is not None
        doc.status = "parsing"
        session.add(doc)
        await session.commit()

    resp_conflict = await client.post(f"/api/v1/documents/{doc_id}/retry")
    assert resp_conflict.status_code == 409
    assert "Document is currently processing" in resp_conflict.json()["detail"]

    # Reset back to failed
    async with factory() as session:
        doc = await session.get(Document, doc_id)
        assert doc is not None
        doc.status = "failed"
        doc.error = "Simulated failure"
        session.add(doc)
        await session.commit()

    # 6. Retry failed document with prefer_fast_path=true
    retry_failed = await client.post(
        f"/api/v1/documents/{doc_id}/retry",
        params={"prefer_fast_path": "true"},
    )
    assert retry_failed.status_code == 200
    assert retry_failed.json()["status"] == "queued"

    status_data3 = await _wait_for_document_ready(client, doc_id)
    assert status_data3["status"] == "ready"


async def test_chat_sse_streaming_threads_and_error_frame(
    api_client,
    sample_xlsx_bytes: bytes,
):
    """Verify SSE chat completions (`step`, `citations`, `token`, `done`, `error`) and thread CRUD."""
    client, _ = api_client

    # Ingest a document first so retrieval has grounded chunks
    up_resp = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("q3_metrics.xlsx", sample_xlsx_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    doc_id = up_resp.json()["id"]
    await _wait_for_document_ready(client, doc_id)

    # Configure fake LLM responses for rewrite -> grade -> synthesize
    fake_llm = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content=json.dumps(
                        {
                            "needs_retrieval": True,
                            "rewritten_queries": ["Q3 2025 operating margin and gross profit"],
                        }
                    )
                ),
                AIMessage(
                    content=json.dumps(
                        {
                            "relevant_indices": [1],
                            "reasoning": "Contains Q3 2025 operating margin metrics",
                        }
                    )
                ),
                AIMessage(
                    content="Operating margin in Q3 2025 reached 24.8% (+3.2% YoY) and gross profit was $142M [1]."
                ),
            ]
        )
    )
    set_chat_model(fake_llm)

    thread_id = "api-test-thread-1"
    chat_resp = await client.post(
        "/api/v1/chat/completions",
        json={
            "message": "What was the operating margin in Q3 2025?",
            "thread_id": thread_id,
            "filter_doc_ids": [doc_id],
            "provider": "openai",
            "model": "gpt-4o-mini",
        },
    )
    assert chat_resp.status_code == 200
    assert chat_resp.headers.get("x-accel-buffering") == "no"
    assert "text/event-stream" in chat_resp.headers.get("content-type", "")

    frames = parse_sse_stream(chat_resp.text)
    event_types = [f["event"] for f in frames]
    assert "step" in event_types
    assert "citations" in event_types
    assert "token" in event_types
    assert event_types[-1] == "done"

    step_nodes = [f["data"]["node"] for f in frames if f["event"] == "step"]
    assert step_nodes == ["rewrite_query", "retrieve", "rerank", "grade_context", "synthesize"]

    citation_frames = [f for f in frames if f["event"] == "citations"]
    assert len(citation_frames) == 1
    citations = citation_frames[0]["data"]["citations"]
    assert len(citations) >= 1
    assert citations[0]["doc_id"] == doc_id
    assert citations[0]["source_filename"] == "q3_metrics.xlsx"
    assert citations[0]["page_numbers"] == []

    streamed_tokens = "".join(f["data"]["content"] for f in frames if f["event"] == "token")
    done_payload = frames[-1]["data"]
    assert done_payload["thread_id"] == thread_id
    assert "24.8%" in done_payload["message"]
    assert "[1]" in done_payload["message"]
    assert streamed_tokens == done_payload["message"]

    # Verify `GET /api/v1/chat/threads` lists the thread
    threads_resp = await client.get("/api/v1/chat/threads")
    assert threads_resp.status_code == 200
    threads_data = threads_resp.json()
    assert threads_data["total"] == 1
    assert threads_data["limit"] == 50
    assert threads_data["offset"] == 0
    assert threads_data["threads"][0]["thread_id"] == thread_id
    assert threads_data["threads"][0]["title"] == "What was the operating margin in Q3 2025?"

    paginated_resp = await client.get("/api/v1/chat/threads", params={"limit": 10, "offset": 1})
    assert paginated_resp.status_code == 200
    pag_data = paginated_resp.json()
    assert pag_data["total"] == 1
    assert pag_data["limit"] == 10
    assert pag_data["offset"] == 1
    assert len(pag_data["threads"]) == 0

    # Verify `GET /api/v1/chat/threads/{thread_id}` returns full message & citation history
    detail_resp = await client.get(f"/api/v1/chat/threads/{thread_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["thread_id"] == thread_id
    assert len(detail["messages"]) == 2
    assert detail["messages"][0]["role"] == "user"
    assert detail["messages"][1]["role"] == "assistant"
    assert "24.8%" in detail["messages"][1]["content"]
    assert len(detail["messages"][1]["citations"]) >= 1
    assert len(detail["citations"]) >= 1

    # Verify `DELETE /api/v1/chat/threads/{thread_id}` removes thread and checkpoint state
    del_thread_resp = await client.delete(f"/api/v1/chat/threads/{thread_id}")
    assert del_thread_resp.status_code == 200
    assert del_thread_resp.json()["deleted"] is True

    assert (await client.get(f"/api/v1/chat/threads/{thread_id}")).status_code == 404
    assert (await client.get("/api/v1/chat/threads")).json()["total"] == 0

    # Verify `event: error` SSE frame when an empty query is submitted
    err_resp = await client.post(
        "/api/v1/chat/completions",
        json={"message": "   ", "thread_id": "api-test-thread-err"},
    )
    assert err_resp.status_code == 200
    err_frames = parse_sse_stream(err_resp.text)
    assert len(err_frames) == 1
    assert err_frames[0]["event"] == "error"
    assert "Query must not be empty" in err_frames[0]["data"]["message"]




# ==============================================================================
# System Memory Telemetry & On-Demand Reclaim Endpoints (Phase 3)
# ==============================================================================


def test_get_process_rss_mb_proc_status_parsing():
    """Verify /proc/self/status VmRSS parsing."""
    mock_status = "Name:\tpython\nVmPeak:\t  204800 kB\nVmRSS:\t  524288 kB\nThreads:\t4\n"
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", return_value=io.StringIO(mock_status)):
            rss = _get_process_rss_mb()
            assert rss == 512.0


def test_get_process_rss_mb_fallback():
    """Verify fallback does not crash when /proc/self/status is absent."""
    with patch("os.path.exists", return_value=False):
        rss = _get_process_rss_mb()
        assert isinstance(rss, float)
        assert rss >= 0.0


def test_collect_memory_stats_cpu_and_none_orchestrator():
    """Verify memory stats with no orchestrator and CPU only."""
    with patch.object(torch.cuda, "is_available", return_value=False):
        stats = collect_memory_stats(orchestrator=None, settings=None)
        assert isinstance(stats, MemoryStatsResponse)
        assert stats.cuda_available is False
        assert stats.cuda_allocated_mb == 0.0
        assert stats.cuda_reserved_mb == 0.0
        assert stats.cuda_driver_free_mb is None
        assert stats.cuda_driver_total_mb is None
        assert stats.worker_pool_active is False
        assert stats.worker_idle_timeout_sec == 60.0
        assert stats.worker_max_tasks == 0


def test_collect_memory_stats_with_cuda():
    """Verify memory stats correctly reads torch.cuda metrics."""
    with patch.object(torch.cuda, "is_available", return_value=True):
        with patch.object(torch.cuda, "memory_allocated", return_value=1024 * 1024 * 100):
            with patch.object(torch.cuda, "memory_reserved", return_value=1024 * 1024 * 250):
                with patch.object(torch.cuda, "mem_get_info", return_value=(4 * 1024**3, 8 * 1024**3)):
                    stats = collect_memory_stats(orchestrator=None, settings=None)
                    assert stats.cuda_available is True
                    assert stats.cuda_allocated_mb == 100.0
                    assert stats.cuda_reserved_mb == 250.0
                    assert stats.cuda_driver_free_mb == 4096.0
                    assert stats.cuda_driver_total_mb == 8192.0


def test_collect_memory_stats_with_orchestrator_mock():
    """Verify memory stats extracts fields from orchestrator state."""
    mock_orch = MagicMock()
    mock_orch.get_worker_state.return_value = {
        "worker_pool_active": True,
        "active_worker_jobs": 2,
        "active_indexing_jobs": 1,
        "worker_reclaim_count": 3,
        "reclaim_deferred": True,
        "worker_idle_timeout_sec": 30.0,
        "worker_max_tasks": 10,
    }

    stats = collect_memory_stats(orchestrator=mock_orch, settings=None)
    assert stats.worker_pool_active is True
    assert stats.active_worker_jobs == 2
    assert stats.active_indexing_jobs == 1
    assert stats.worker_reclaim_count == 3
    assert stats.reclaim_deferred is True
    assert stats.worker_idle_timeout_sec == 30.0
    assert stats.worker_max_tasks == 10


async def test_system_memory_and_reclaim_endpoints(api_client):
    """Verify GET /api/v1/system/memory, POST /api/v1/system/memory/reclaim, and enriched GET /api/v1/system/stats."""
    client, _ = api_client

    # 1. GET /api/v1/system/memory
    resp = await client.get("/api/v1/system/memory")
    assert resp.status_code == 200
    data = resp.json()

    for key in (
        "process_rss_mb",
        "cuda_available",
        "worker_pool_active",
        "active_worker_jobs",
        "active_indexing_jobs",
        "worker_reclaim_count",
        "reclaim_deferred",
    ):
        assert key in data, f"Missing key {key} in /api/v1/system/memory response"

    validated_mem = MemoryStatsResponse(**data)
    assert isinstance(validated_mem.process_rss_mb, float)
    assert validated_mem.active_worker_jobs == 0
    assert validated_mem.active_indexing_jobs == 0

    # 2. POST /api/v1/system/memory/reclaim
    rec_resp = await client.post("/api/v1/system/memory/reclaim")
    assert rec_resp.status_code == 200
    rec_data = rec_resp.json()
    assert rec_data["status"] == "ok"
    assert "worker_pool_reclaimed" in rec_data
    assert "deferred" in rec_data
    assert "cuda_cache_cleared" in rec_data
    assert "memory" in rec_data

    validated_rec = MemoryReclaimResponse(**rec_data)
    assert validated_rec.status == "ok"
    assert validated_rec.deferred is False
    assert isinstance(validated_rec.memory, MemoryStatsResponse)

    # 3. GET /api/v1/system/stats
    stats_resp = await client.get("/api/v1/system/stats")
    assert stats_resp.status_code == 200
    stats_data = stats_resp.json()
    assert "ingestion" in stats_data
    assert "memory" in stats_data

    validated_stats = SystemStatsResponse(**stats_data)
    assert validated_stats.memory is not None
    assert isinstance(validated_stats.memory.process_rss_mb, float)
    assert validated_stats.ingestion["device"] == "auto"
    assert validated_stats.ingestion["resolved_device"] in ("cuda", "cpu")


async def test_api_system_memory_reclaim_deferred_when_worker_active(api_client):
    """Verify POST /api/v1/system/memory/reclaim defers when active_worker_jobs > 0."""
    client, _ = api_client
    orch = client.app.state.orchestrator

    orch._active_worker_jobs = 1
    orch._pool = MagicMock()

    try:
        resp = await client.post("/api/v1/system/memory/reclaim")
        assert resp.status_code == 200
        data = resp.json()

        assert data["worker_pool_reclaimed"] is False
        assert data["deferred"] is True
        assert data["active_worker_jobs"] == 1
        assert data["memory"]["reclaim_deferred"] is True
    finally:
        orch._active_worker_jobs = 0
        orch._reclaim_on_idle = False
        orch._pool = None


async def test_api_system_memory_reclaim_guards_cuda_when_indexing_active(api_client):
    """Verify POST /api/v1/system/memory/reclaim does NOT clear CUDA cache mid-indexing."""
    client, _ = api_client
    orch = client.app.state.orchestrator

    orch._active_indexing_jobs = 1
    with patch("src.api.routes.system.release_cuda_cache") as mock_release:
        try:
            resp = await client.post("/api/v1/system/memory/reclaim")
            assert resp.status_code == 200
            data = resp.json()
            assert data["cuda_cache_cleared"] is False
            mock_release.assert_not_called()
        finally:
            orch._active_indexing_jobs = 0


async def test_api_system_ingestion_pause_and_resume(api_client):
    """Verify POST /api/v1/system/ingestion/pause and /resume endpoints and stats reflection."""
    client, _ = api_client
    orch = client.app.state.orchestrator

    # Ensure starting in unpaused state
    orch.resume()
    assert orch.is_paused is False

    # 1. Pause ingestion
    resp_pause = await client.post("/api/v1/system/ingestion/pause")
    assert resp_pause.status_code == 200
    data_pause = resp_pause.json()
    assert data_pause["status"] == "ok"
    assert data_pause["ingestion_paused"] is True
    assert orch.is_paused is True

    # 2. Verify stats reports ingestion_paused: True
    resp_stats = await client.get("/api/v1/system/stats")
    assert resp_stats.status_code == 200
    assert resp_stats.json()["ingestion_paused"] is True

    # 3. Resume ingestion
    resp_resume = await client.post("/api/v1/system/ingestion/resume")
    assert resp_resume.status_code == 200
    data_resume = resp_resume.json()
    assert data_resume["status"] == "ok"
    assert data_resume["ingestion_paused"] is False
    assert orch.is_paused is False

    # 4. Verify stats reports ingestion_paused: False
    resp_stats2 = await client.get("/api/v1/system/stats")
    assert resp_stats2.status_code == 200
    assert resp_stats2.json()["ingestion_paused"] is False


@pytest.mark.asyncio
async def test_concurrent_retry_requests_conflict_rejection(api_client, sample_xlsx_bytes):
    """Verify that competing concurrent retry requests for the same document reject all but one with HTTP 409 Conflict."""
    client, _ = api_client
    up_resp = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("concurrent_retry.xlsx", sample_xlsx_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert up_resp.status_code == 202
    doc_id = up_resp.json()["id"]
    status_data = await _wait_for_document_ready(client, doc_id)
    assert status_data["status"] == "ready"

    # Concurrently fire 2 retry calls with force_reparse=True
    responses = await asyncio.gather(
        client.post(f"/api/v1/documents/{doc_id}/retry", params={"force_reparse": True}),
        client.post(f"/api/v1/documents/{doc_id}/retry", params={"force_reparse": True}),
        return_exceptions=True,
    )
    status_codes = sorted([r.status_code for r in responses if hasattr(r, "status_code")])
    assert 200 in status_codes
    assert 409 in status_codes

    await _wait_for_document_ready(client, doc_id)


@pytest.mark.asyncio
async def test_chat_threads_pagination(api_client):
    """Verify limit and offset pagination on /api/v1/chat/threads."""
    client, _ = api_client
    import datetime
    from src.db.models import ChatThread
    from src.db.session import get_session_factory

    factory = get_session_factory()
    base_time = datetime.datetime.now(datetime.timezone.utc)
    async with factory() as session:
        for i in range(5):
            thread = ChatThread(
                thread_id=f"thread-{i}",
                title=f"Thread {i}",
                created_at=base_time + datetime.timedelta(seconds=i),
                updated_at=base_time + datetime.timedelta(seconds=i),
            )
            session.add(thread)
        await session.commit()

    # Query with limit=2, offset=1
    res = await client.get("/api/v1/chat/threads", params={"limit": 2, "offset": 1})
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 5
    assert data["limit"] == 2
    assert data["offset"] == 1
    assert len(data["threads"]) == 2
    # Verify ordering (newest first): thread-4, thread-3, thread-2, thread-1, thread-0
    # offset 1, limit 2 should give thread-3 and thread-2
    assert data["threads"][0]["thread_id"] == "thread-3"
    assert data["threads"][1]["thread_id"] == "thread-2"



@pytest.mark.asyncio
async def test_chat_multi_turn_citations_persisted_via_api(api_client, sample_xlsx_bytes):
    """Verify that multi-turn chat via REST API preserves citations on prior turns
    and returns empty citations on fallback turns when retrieved via GET /api/v1/chat/threads/{thread_id}.
    """
    client, _ = api_client

    # 1. Ingest sample document
    up_resp = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("q3_metrics.xlsx", sample_xlsx_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert up_resp.status_code == 202
    doc_id = up_resp.json()["id"]
    await _wait_for_document_ready(client, doc_id)

    # 2. Configure Fake LLM for Turn 1 (synthesis with citations) and Turn 2 (fallback notice)
    fake_llm = GenericFakeChatModel(
        messages=iter(
            [
                # Turn 1: Rewrite -> Grade -> Synthesize
                AIMessage(
                    content=json.dumps(
                        {"needs_retrieval": True, "rewritten_queries": ["Q3 2025 operating margin"]}
                    )
                ),
                AIMessage(
                    content=json.dumps(
                        {"relevant_indices": [0], "reasoning": "Contains financial metrics table"}
                    )
                ),
                AIMessage(
                    content="Operating margin in Q3 2025 reached 24.8% (+3.2% YoY) [1]."
                ),
                # Turn 2: Rewrite -> Grade (no match) -> Fallback notice
                AIMessage(
                    content=json.dumps(
                        {"needs_retrieval": True, "rewritten_queries": ["quantum warp engine"]}
                    )
                ),
                AIMessage(
                    content=json.dumps(
                        {"relevant_indices": [], "reasoning": "No relevant info about warp engines"}
                    )
                ),
            ]
        )
    )
    set_chat_model(fake_llm)

    thread_id = "api-multi-turn-thread-1"

    # Turn 1: Post question that yields citations
    resp1 = await client.post(
        "/api/v1/chat/completions",
        json={
            "message": "What was the operating margin in Q3 2025?",
            "thread_id": thread_id,
            "filter_doc_ids": [doc_id],
        },
    )
    assert resp1.status_code == 200
    frames1 = parse_sse_stream(resp1.text)
    cits_frames1 = [f for f in frames1 if f["event"] == "citations"]
    assert len(cits_frames1) == 1
    assert len(cits_frames1[0]["data"]["citations"]) >= 1

    # Turn 2: Post question on same thread that triggers fallback notice
    resp2 = await client.post(
        "/api/v1/chat/completions",
        json={
            "message": "What is the quantum warp engine output?",
            "thread_id": thread_id,
            "filter_doc_ids": [doc_id],
        },
    )
    assert resp2.status_code == 200
    frames2 = parse_sse_stream(resp2.text)
    done_frames2 = [f for f in frames2 if f["event"] == "done"]
    assert len(done_frames2) == 1
    assert done_frames2[0]["data"]["citations"] == []

    # 3. GET /api/v1/chat/threads/{thread_id} to verify historical message persistence
    detail_resp = await client.get(f"/api/v1/chat/threads/{thread_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["thread_id"] == thread_id
    msgs = detail["messages"]
    assert len(msgs) == 4

    # Message 0: User query 1
    assert msgs[0]["role"] == "user"
    assert "operating margin" in msgs[0]["content"].lower()

    # Message 1: Assistant response 1 (retains citations!)
    assert msgs[1]["role"] == "assistant"
    assert "24.8%" in msgs[1]["content"]
    assert len(msgs[1]["citations"]) >= 1
    assert msgs[1]["citations"][0]["doc_id"] == doc_id

    # Message 2: User query 2
    assert msgs[2]["role"] == "user"
    assert "warp engine" in msgs[2]["content"].lower()

    # Message 3: Assistant response 2 (fallback notice has citations == [])
    assert msgs[3]["role"] == "assistant"
    assert "could not find sufficient relevant evidence" in msgs[3]["content"].lower()
    assert msgs[3]["citations"] == []


# ============================================================================
# Phase 2: Domain Taxonomy, Lexicon & Workspace Settings Tests
# ============================================================================


async def test_domains_and_subdomains_crud(api_client):
    """Verify full CRUD over custom domains and subdomains, including auto-prefixing and guards."""
    client, _ = api_client

    # 1. GET /api/v1/domains (verify 4 starter domains)
    resp = await client.get("/api/v1/domains")
    assert resp.status_code == 200
    domains = resp.json()
    assert len(domains) >= 4
    slug_map = {d["slug"]: d for d in domains}
    for starter in ("engineering", "economics", "law", "general"):
        assert starter in slug_map
        assert slug_map[starter]["is_system"] is True
        assert len(slug_map[starter]["subdomains"]) >= 1

    # 2. POST /api/v1/domains (create custom domain)
    create_payload = {
        "slug": "medicine",
        "name": "Medicine & Healthcare\n",  # test newline stripping
        "description": "Medical devices, clinical trials, and healthcare administration.",
        "icon": "🩺",
        "color_token": "rose",
        "chunking_strategy": "standard_markdown",
        "epistemic_type": "empirical",
        "authority_priority": 2,
    }
    resp = await client.post("/api/v1/domains", json=create_payload)
    assert resp.status_code == 201
    created = resp.json()
    assert created["slug"] == "medicine"
    assert created["name"] == "Medicine & Healthcare"  # stripped
    assert created["is_system"] is False
    assert created["color_token"] == "rose"
    assert created["document_count"] == 0
    # Auto-created default subdomain
    assert any(s["slug"] == "medicine.general" for s in created["subdomains"])

    # Duplicate slug check
    dup_resp = await client.post("/api/v1/domains", json=create_payload)
    assert dup_resp.status_code == 409

    # 3. POST /api/v1/domains/{slug}/subdomains (auto-prefixing and uniqueness)
    # 3a. auto-prefixed
    sub1_resp = await client.post(
        "/api/v1/domains/medicine/subdomains",
        json={"slug": "cardiology", "name": "Cardiology"},
    )
    assert sub1_resp.status_code == 201
    assert sub1_resp.json()["slug"] == "medicine.cardiology"

    # 3b. already prefixed
    sub2_resp = await client.post(
        "/api/v1/domains/medicine/subdomains",
        json={"slug": "medicine.oncology", "name": "Oncology", "standard_bodies": "ASCO, ESMO"},
    )
    assert sub2_resp.status_code == 201
    assert sub2_resp.json()["slug"] == "medicine.oncology"

    # 3c. duplicate subdomain slug
    dup_sub_resp = await client.post(
        "/api/v1/domains/medicine/subdomains",
        json={"slug": "cardiology", "name": "Cardiology Duplicate"},
    )
    assert dup_sub_resp.status_code == 409

    # 4. PATCH /api/v1/domains/{slug}/subdomains/{sub_slug}
    patch_sub_resp = await client.patch(
        "/api/v1/domains/medicine/subdomains/medicine.cardiology",
        json={"name": "Cardiovascular Medicine", "standard_bodies": "ACC, AHA"},
    )
    assert patch_sub_resp.status_code == 200
    assert patch_sub_resp.json()["name"] == "Cardiovascular Medicine"
    assert patch_sub_resp.json()["standard_bodies"] == "ACC, AHA"

    # 5. System starter domain field lock: PATCH /api/v1/domains/engineering
    # 5a. Visual/tuning fields modification succeeds
    patch_eng_resp = await client.patch(
        "/api/v1/domains/engineering",
        json={
            "icon": "🛠️",
            "color_token": "blue",
            "description": "Updated engineering description with at least ten characters.",
        },
    )
    assert patch_eng_resp.status_code == 200
    assert patch_eng_resp.json()["icon"] == "🛠️"
    assert patch_eng_resp.json()["color_token"] == "blue"

    # 5b. Locked name/epistemic modification fails with 400 Bad Request
    fail_eng_name = await client.patch(
        "/api/v1/domains/engineering",
        json={"name": "Modified Engineering Name"},
    )
    assert fail_eng_name.status_code == 400

    fail_eng_epistemic = await client.patch(
        "/api/v1/domains/engineering",
        json={"epistemic_type": "normative"},
    )
    assert fail_eng_epistemic.status_code == 400

    # 6. Subdomain deletion and guards
    # 6a. Delete root general.general fails with 400
    del_root_resp = await client.delete("/api/v1/domains/general/subdomains/general.general")
    assert del_root_resp.status_code == 400

    # 6b. Delete non-last subdomain succeeds
    del_sub_resp = await client.delete("/api/v1/domains/medicine/subdomains/medicine.cardiology")
    assert del_sub_resp.status_code == 200

    # Delete medicine.oncology
    del_sub2_resp = await client.delete("/api/v1/domains/medicine/subdomains/medicine.oncology")
    assert del_sub2_resp.status_code == 200

    # 6c. Now medicine only has medicine.general left; deleting it fails with 400 (cannot delete last subdomain)
    del_last_resp = await client.delete("/api/v1/domains/medicine/subdomains/medicine.general")
    assert del_last_resp.status_code == 400


async def test_domain_deletion_guard_and_reassignment(api_client, sample_pdf_bytes):
    """Verify system domain deletion lock, in-flight 409 guard, target reassignment, and re_index queueing."""
    from sqlalchemy import select
    from src.db.models import Document, IngestionJob, _now
    from src.db.session import get_session_factory
    from src.ingestion.parser import ParseResult

    client, data_dir = api_client

    # 1. System domain deletion lock
    del_eng_resp = await client.delete("/api/v1/domains/engineering")
    assert del_eng_resp.status_code == 400

    # 2. Create custom domain for deletion testing
    create_resp = await client.post(
        "/api/v1/domains",
        json={
            "slug": "biotech",
            "name": "Biotechnology",
            "description": "Genomics, synthetic biology, and molecular genetics.",
        },
    )
    assert create_resp.status_code == 201

    # Target domain == slug check
    self_target = await client.delete("/api/v1/domains/biotech?target_domain=biotech")
    assert self_target.status_code == 400

    # 3. In-flight 409 guard
    session_factory = get_session_factory()
    doc_id = "test-inflight-doc-1"
    async with session_factory() as session:
        doc = Document(
            id=doc_id,
            filename="gene_editing.pdf",
            file_type="pdf",
            sha256="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            size_bytes=1024,
            status="parsing",  # in-flight state!
            primary_domain_id="biotech",
            created_at=_now(),
            updated_at=_now(),
        )
        session.add(doc)
        await session.commit()

    del_inflight_resp = await client.delete("/api/v1/domains/biotech?target_domain=general")
    assert del_inflight_resp.status_code == 409

    # 4. Successful deletion with target reassignment & re_index=True
    # Create cached parse files on disk and transition doc to "ready"
    sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    raw_dir = data_dir / "raw"
    parsed_dir = data_dir / "parsed"
    md_dir = data_dir / "markdown"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / f"{sha}.pdf").write_bytes(sample_pdf_bytes)
    (parsed_dir / f"{sha}.json").write_text(
        json.dumps({"schema_name": "DoclingDocument", "pages": []}), encoding="utf-8"
    )
    (md_dir / f"{sha}.md").write_text(
        "# Biotechnology\nSample markdown content for re-index.", encoding="utf-8"
    )

    async with session_factory() as session:
        d = await session.get(Document, doc_id)
        assert d is not None
        d.status = "ready"
        session.add(d)
        await session.commit()

    del_ok_resp = await client.delete(
        "/api/v1/domains/biotech?target_domain=general&re_index=true"
    )
    assert del_ok_resp.status_code == 200
    del_data = del_ok_resp.json()
    assert del_data["deleted_slug"] == "biotech"
    assert del_data["target_domain"] == "general"
    assert del_data["reassigned_documents"] >= 1
    assert del_data["re_index_queued"] >= 1

    # Verify document in SQLite was reassigned to general and queued for re-index
    async with session_factory() as session:
        reassigned_doc = await session.get(Document, doc_id)
        assert reassigned_doc is not None
        assert reassigned_doc.primary_domain_id == "general"
        assert reassigned_doc.status in ("queued", "ready")
        job_res = await session.execute(
            select(IngestionJob).where(IngestionJob.doc_id == doc_id)
        )
        assert len(job_res.scalars().all()) >= 1


async def test_lexicon_crud_and_limits(api_client):
    """Verify lexicon term management, domain filter, search, and validation."""
    client, _ = api_client

    # 1. GET /api/v1/lexicon
    resp = await client.get("/api/v1/lexicon")
    assert resp.status_code == 200
    initial_terms = resp.json()
    assert isinstance(initial_terms, list)

    # 2. POST /api/v1/lexicon
    create_payload = {
        "term": "yield\n",  # test newline stripping
        "domain_id": "economics",
        "specific_sense": "Annualized percentage return on an investment or bond coupon.",
        "contrastive_warning": "Do not confuse with mechanical yield point or agricultural harvest yield.",
    }
    resp = await client.post("/api/v1/lexicon", json=create_payload)
    assert resp.status_code == 201
    entry = resp.json()
    entry_id = entry["id"]
    assert entry["term"] == "yield"  # stripped
    assert entry["domain_id"] == "economics"
    assert "Annualized percentage" in entry["specific_sense"]

    # 3. GET /api/v1/lexicon with filter and search
    search_resp = await client.get("/api/v1/lexicon?domain_id=economics&search=yield")
    assert search_resp.status_code == 200
    results = search_resp.json()
    assert len(results) >= 1
    assert any(r["id"] == entry_id for r in results)

    # 4. DELETE /api/v1/lexicon/{entry_id}
    del_resp = await client.delete(f"/api/v1/lexicon/{entry_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # 5. Delete again returns 404
    del_again = await client.delete(f"/api/v1/lexicon/{entry_id}")
    assert del_again.status_code == 404


async def test_workspace_settings_and_upload_defaults(api_client, sample_pdf_bytes):
    """Verify WorkspaceSettings endpoints, upload defaults fallback, and document domain filtering."""
    client, _ = api_client

    # 1. GET /api/v1/settings
    resp = await client.get("/api/v1/settings")
    assert resp.status_code == 200
    settings_data = resp.json()
    assert "default_upload_mode" in settings_data
    assert "default_approval_mode" in settings_data
    assert "high_confidence_threshold" in settings_data
    assert isinstance(settings_data["available_providers"], list)

    # 2. PATCH /api/v1/settings
    patch_payload = {
        "default_upload_mode": "fast",
        "default_approval_mode": "auto_all",
        "high_confidence_threshold": 0.92,
        "default_llm_provider": "openai",
        "default_llm_model": "gpt-4o",
    }
    patch_resp = await client.patch("/api/v1/settings", json=patch_payload)
    assert patch_resp.status_code == 200
    updated_settings = patch_resp.json()
    assert updated_settings["default_upload_mode"] == "fast"
    assert updated_settings["default_approval_mode"] == "auto_all"
    assert updated_settings["high_confidence_threshold"] == 0.92
    assert updated_settings["default_llm_provider"] == "openai"
    assert updated_settings["default_llm_model"] == "gpt-4o"

    # 3. Upload document without prefer_fast_path or approval_mode
    # It must inherit default_approval_mode="auto_all" and default_upload_mode="fast"
    files = {"file": ("workspace_test.pdf", sample_pdf_bytes, "application/pdf")}
    upload_resp = await client.post("/api/v1/documents/upload", files=files)
    assert upload_resp.status_code in (200, 202)
    doc_id = upload_resp.json()["id"]

    # Wait for document ready
    ready_doc = await _wait_for_document_ready(client, doc_id)
    assert ready_doc["status"] == "ready"

    # 4. Test GET /api/v1/documents?domain=...
    # Query with matching domain (e.g. general or engineering)
    primary_domain = ready_doc.get("primary_domain_id") or "general"
    list_matching = await client.get(f"/api/v1/documents?domain={primary_domain}")
    assert list_matching.status_code == 200
    matched_ids = [d["id"] for d in list_matching.json()["items"]]
    assert doc_id in matched_ids

    # Query with different domain (e.g. non-matching domain)
    list_non_matching = await client.get("/api/v1/documents?domain=nonexistent_domain")
    assert list_non_matching.status_code == 200
    non_matched_ids = [d["id"] for d in list_non_matching.json()["items"]]
    assert doc_id not in non_matched_ids


async def test_domain_limits_and_validation(api_client):
    """Verify domain input bounds, regex validation, and maximum 10 active domains cap."""
    client, _ = api_client

    # 1. Invalid slug (starts with number)
    resp1 = await client.post(
        "/api/v1/domains",
        json={"slug": "123invalid", "name": "Invalid Domain", "description": "Valid description length."},
    )
    assert resp1.status_code == 422

    # 2. Invalid color_token
    resp2 = await client.post(
        "/api/v1/domains",
        json={
            "slug": "badcolor",
            "name": "Bad Color Domain",
            "description": "Valid description length.",
            "color_token": "neon_pink",
        },
    )
    assert resp2.status_code == 422

    # 3. Description too short (< 10 chars)
    resp3 = await client.post(
        "/api/v1/domains",
        json={"slug": "shortdesc", "name": "Short Desc", "description": "Too short"},
    )
    assert resp3.status_code == 422

    # 4. Max 10 active domains limit
    # First see how many active domains exist
    domains_resp = await client.get("/api/v1/domains")
    current_count = len(domains_resp.json())

    created_slugs = []
    try:
        # Create domains up to 10 active
        for i in range(current_count, 10):
            slug = f"capdomain{i}"
            res = await client.post(
                "/api/v1/domains",
                json={
                    "slug": slug,
                    "name": f"Cap Domain {i}",
                    "description": f"Domain {i} to fill workspace domain capacity.",
                },
            )
            assert res.status_code == 201
            created_slugs.append(slug)

        # 11th domain creation must return 400 Bad Request
        cap_exceeded_resp = await client.post(
            "/api/v1/domains",
            json={
                "slug": "eleventhdomain",
                "name": "Eleventh Domain",
                "description": "This should be rejected due to 10 active domain limit.",
            },
        )
        assert cap_exceeded_resp.status_code == 400
        assert "maximum limit" in cap_exceeded_resp.json()["detail"].lower()
    finally:
        # Clean up created test domains
        for s in created_slugs:
            await client.delete(f"/api/v1/domains/{s}?target_domain=general")

