"""Integration and unit tests for User-Controlled Ingestion Modes (Phase 4).

Verifies:
1. prefer_fast_path=True forces PyMuPDF on complex layout.
2. prefer_fast_path=False forces Docling on simple digital text PDF.
3. prefer_fast_path=None executes 5ms preflight probe heuristic (Auto mode).
4. Duplicate upload with prefer_fast_path=False on existing pymupdf4llm doc triggers re-parse upgrade (duplicate=False, 202 ACCEPTED).
5. Duplicate upload with prefer_fast_path=True on existing docling doc is protected from downgrade (duplicate=True, 200 OK).
6. Explicit downgrade via retry with force_reparse=True and prefer_fast_path=True succeeds.
7. Qdrant vector purge removes stale chunk points upon re-parsing (zero orphan chunk points).
8. Checkpoint invalidation occurs when cached engine mismatches requested mode.
9. Failover attribution records engine_fallback_reason on mock Docling crash.
"""
from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pymupdf
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from qdrant_client import models as qmodels

from src.api.main import create_app
from src.config import get_settings
from src.db.models import Document
from src.db.session import create_engine, create_session_factory, init_db
from src.ingestion.indexer import (
    DeterministicBM25Encoder,
    DeterministicStubEmbedder,
    set_embedder,
    set_sparse_encoder,
)
from src.ingestion.parser import DocumentParserService, compute_sha256, inspect_pdf_fast_path
from src.rag.graph import close_checkpointer
from src.rag.llm import set_chat_model
from src.rag.reranker import set_reranker
from src.rag.retriever import set_active_qdrant_client

STUB_DIM = 32


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Simple digital text PDF (>120 chars, 0 images) that qualifies for fast path under preflight probe."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1. Executive Summary", fontsize=16)
    page.insert_text(
        (72, 110),
        "Operating margin expanded by 3.2% year-over-year driven by supply chain automation. "
        "Customer retention remains above ninety-four percent across all geographic operating segments. "
        "The digital transformation initiatives launched in fiscal year 2025 have delivered consistent operational efficiencies.",
        fontsize=11,
    )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def complex_pdf_bytes() -> bytes:
    """Complex layout PDF with multiple images and minimal text that fails fast-path heuristic."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Fig. 1 Layout Overview", fontsize=10)
    pix = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 10, 10), False)
    img_bytes = pix.tobytes("png")
    for i in range(3):
        page.insert_image(pymupdf.Rect(50, 50 + i * 30, 80, 80 + i * 30), stream=img_bytes)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
async def modes_client(tmp_path: Path, monkeypatch):
    """Run FastAPI inside LifespanManager with isolated temp storage, stub embedder, and in-memory Qdrant."""
    data_dir = tmp_path / "data"
    db_dir = tmp_path / "data" / "db"
    dist_dir = tmp_path / "frontend_dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
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


async def _wait_for_document_ready(
    client: AsyncClient, doc_id: str, timeout_s: float = 15.0
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last_payload: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        resp = await client.get(f"/api/v1/documents/{doc_id}/status")
        assert resp.status_code == 200
        last_payload = resp.json()
        if last_payload["status"] in ("ready", "failed"):
            return last_payload
        await asyncio.sleep(0.05)
    raise AssertionError(f"Document {doc_id} did not reach ready status in time: {last_payload}")


def _mock_docling_success(file_path: Path, file_type: str = "pdf"):
    """Mock Docling parser return value."""
    return (
        "# High-Fidelity Docling Content\n\nDeep document layout parsed via Docling GPU",
        {"pages": [{"page_no": 1}]},
        1,
        {"engine": "docling"},
    )


@pytest.mark.asyncio
async def test_prefer_fast_path_true_forces_pymupdf_on_complex_layout(
    modes_client, complex_pdf_bytes
):
    """1. prefer_fast_path=True forces PyMuPDF on complex/scanned layout."""
    client, _ = modes_client

    # Verify complex layout would normally fail fast-path probe
    tmp_pdf = Path("test_complex_check.pdf")
    try:
        tmp_pdf.write_bytes(complex_pdf_bytes)
        is_fast, reason = inspect_pdf_fast_path(tmp_pdf)
        assert is_fast is False, f"Expected complex PDF to fail fast-path, got: {reason}"
    finally:
        tmp_pdf.unlink(missing_ok=True)

    resp = await client.post(
        "/api/v1/documents/upload?prefer_fast_path=true",
        files={"file": ("forced_fast_complex.pdf", io.BytesIO(complex_pdf_bytes), "application/pdf")},
    )
    assert resp.status_code == 202
    data = resp.json()
    doc_id = data["id"]
    assert data["duplicate"] is False

    final = await _wait_for_document_ready(client, doc_id)
    assert final["status"] == "ready"
    assert final["parsing_engine"] == "pymupdf4llm"
    assert final["engine_fallback_reason"] is None


@pytest.mark.asyncio
async def test_prefer_fast_path_false_forces_docling_on_digital_text(
    modes_client, sample_pdf_bytes
):
    """2. prefer_fast_path=False forces Docling on simple digital text PDF."""
    client, _ = modes_client

    # Verify digital text would normally pass fast-path probe
    tmp_pdf = Path("test_digital_check.pdf")
    try:
        tmp_pdf.write_bytes(sample_pdf_bytes)
        is_fast, reason = inspect_pdf_fast_path(tmp_pdf)
        assert is_fast is True, f"Expected digital PDF to pass fast-path, got: {reason}"
    finally:
        tmp_pdf.unlink(missing_ok=True)

    with patch.object(
        DocumentParserService, "_parse_with_docling", side_effect=_mock_docling_success
    ) as mock_dl:
        resp = await client.post(
            "/api/v1/documents/upload?prefer_fast_path=false",
            files={"file": ("forced_docling_digital.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
        )
        assert resp.status_code == 202
        data = resp.json()
        doc_id = data["id"]
        assert data["duplicate"] is False

        final = await _wait_for_document_ready(client, doc_id)
        assert final["status"] == "ready"
        assert final["parsing_engine"] == "docling"
        assert final["engine_fallback_reason"] is None
        assert mock_dl.called


@pytest.mark.asyncio
async def test_prefer_fast_path_none_executes_probe_heuristic(
    modes_client, sample_pdf_bytes, complex_pdf_bytes
):
    """3. prefer_fast_path=None executes 5ms preflight probe heuristic (Auto mode)."""
    client, _ = modes_client

    # Digital text PDF -> probe classifies as digital_text -> PyMuPDF
    resp_digital = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("auto_digital.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
    )
    assert resp_digital.status_code == 202
    doc_id_digital = resp_digital.json()["id"]
    final_digital = await _wait_for_document_ready(client, doc_id_digital)
    assert final_digital["status"] == "ready"
    assert final_digital["parsing_engine"] == "pymupdf4llm"

    # Complex layout PDF -> probe classifies as heavy_layout -> Docling
    with patch.object(
        DocumentParserService, "_parse_with_docling", side_effect=_mock_docling_success
    ) as mock_dl:
        resp_complex = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("auto_complex.pdf", io.BytesIO(complex_pdf_bytes), "application/pdf")},
        )
        assert resp_complex.status_code == 202
        doc_id_complex = resp_complex.json()["id"]
        final_complex = await _wait_for_document_ready(client, doc_id_complex)
        assert final_complex["status"] == "ready"
        assert final_complex["parsing_engine"] == "docling"
        assert mock_dl.called


@pytest.mark.asyncio
async def test_duplicate_upload_mode_upgrade_and_downgrade_protection(
    modes_client, sample_pdf_bytes
):
    """4. Duplicate upload with prefer_fast_path=False triggers re-parse upgrade.
    5. Duplicate upload with prefer_fast_path=True on existing docling doc is protected against downgrade.
    6. Explicit downgrade via retry with force_reparse=True and prefer_fast_path=True succeeds.
    """
    client, _ = modes_client

    # Step 1: Upload with fast path (PyMuPDF)
    resp1 = await client.post(
        "/api/v1/documents/upload?prefer_fast_path=true",
        files={"file": ("sample.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
    )
    assert resp1.status_code == 202
    doc_id = resp1.json()["id"]
    final1 = await _wait_for_document_ready(client, doc_id)
    assert final1["parsing_engine"] == "pymupdf4llm"

    # Step 2: Re-upload identical file requesting GPU upgrade (prefer_fast_path=False)
    with patch.object(
        DocumentParserService, "_parse_with_docling", side_effect=_mock_docling_success
    ):
        resp_upgrade = await client.post(
            "/api/v1/documents/upload?prefer_fast_path=false",
            files={"file": ("sample.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
        )
        assert resp_upgrade.status_code == 202
        up_data = resp_upgrade.json()
        assert up_data["id"] == doc_id
        assert up_data["duplicate"] is False
        assert up_data["status"] == "queued"

        final_up = await _wait_for_document_ready(client, doc_id)
        assert final_up["status"] == "ready"
        assert final_up["parsing_engine"] == "docling"

    # Step 3: Re-upload identical file with prefer_fast_path=True on existing docling doc -> Protected against downgrade!
    resp_downgrade_attempt = await client.post(
        "/api/v1/documents/upload?prefer_fast_path=true",
        files={"file": ("sample.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
    )
    assert resp_downgrade_attempt.status_code == 200
    down_data = resp_downgrade_attempt.json()
    assert down_data["id"] == doc_id
    assert down_data["duplicate"] is True
    assert down_data["parsing_engine"] == "docling"

    # Step 4: Explicit downgrade via retry with force_reparse=true&prefer_fast_path=true
    resp_retry = await client.post(
        f"/api/v1/documents/{doc_id}/retry?force_reparse=true&prefer_fast_path=true"
    )
    assert resp_retry.status_code == 200
    retry_data = resp_retry.json()
    assert retry_data["status"] == "queued"

    final_retry = await _wait_for_document_ready(client, doc_id)
    assert final_retry["status"] == "ready"
    assert final_retry["parsing_engine"] == "pymupdf4llm"


@pytest.mark.asyncio
async def test_qdrant_vector_purge_removes_stale_chunks_on_reparse(
    modes_client, sample_pdf_bytes
):
    """7. Qdrant vector purge removes stale chunk points upon re-parsing (zero orphan chunk points)."""
    client, _ = modes_client
    qdrant = client.app.state.qdrant
    collection_name = client.app.state.settings.qdrant_collection

    def _mock_docling_3_chunks(file_path: Path, file_type: str = "pdf"):
        md = (
            "# Section A\n\nDetailed multi-line paragraph describing section A in full depth.\n\n"
            "# Section B\n\nDetailed multi-line paragraph describing section B in full depth.\n\n"
            "# Section C\n\nDetailed multi-line paragraph describing section C in full depth."
        )
        return md, {"pages": [{"page_no": 1}]}, 1, {"engine": "docling"}

    # 1. Initial upload with Docling generating 3 chunks
    with patch.object(
        DocumentParserService, "_parse_with_docling", side_effect=_mock_docling_3_chunks
    ):
        resp1 = await client.post(
            "/api/v1/documents/upload?prefer_fast_path=false",
            files={"file": ("purge_target.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
        )
        assert resp1.status_code == 202
        doc_id = resp1.json()["id"]

        final1 = await _wait_for_document_ready(client, doc_id)
        assert final1["status"] == "ready"
        assert final1["parsing_engine"] == "docling"
        assert final1["chunk_count"] == 3

    # Record initial chunk points in Qdrant (3 points)
    pts1, _ = await qdrant.scroll(
        collection_name=collection_name,
        scroll_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
        ),
        limit=100,
        with_payload=True,
    )
    assert len(pts1) == 3
    initial_point_ids = {p.id for p in pts1}

    # 2. Re-parse downgrade via retry to Fast CPU PyMuPDF (which only produces 1 chunk)
    resp_retry = await client.post(
        f"/api/v1/documents/{doc_id}/retry?force_reparse=true&prefer_fast_path=true"
    )
    assert resp_retry.status_code == 200

    final2 = await _wait_for_document_ready(client, doc_id)
    assert final2["status"] == "ready"
    assert final2["parsing_engine"] == "pymupdf4llm"
    assert final2["chunk_count"] == 1

    # Record updated points in Qdrant (should now only be 1 point)
    pts2, _ = await qdrant.scroll(
        collection_name=collection_name,
        scroll_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
        ),
        limit=100,
        with_payload=True,
    )
    assert len(pts2) == 1
    new_point_ids = {p.id for p in pts2}

    # Stale chunk points (chunks 1 and 2 from Docling) must be purged completely
    stale_point_ids = initial_point_ids - new_point_ids
    assert len(stale_point_ids) == 2, "Expected 2 stale points to be removed by Qdrant purge"

    # Exact count in Qdrant must equal chunk_count (zero orphan chunk points)
    count_resp = await qdrant.count(
        collection_name=collection_name,
        count_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
        ),
        exact=True,
    )
    assert count_resp.count == 1 == final2["chunk_count"] == len(pts2)


def test_checkpoint_invalidation_on_engine_mismatch(tmp_path: Path):
    """8. Checkpoint invalidation occurs when cached engine mismatches requested mode."""
    data_dir = tmp_path / "data"
    parsed_dir = data_dir / "parsed"
    markdown_dir = data_dir / "markdown"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy PDF
    pdf_path = tmp_path / "test_mismatch.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Sample Content")
    doc.save(str(pdf_path))
    doc.close()

    sha = compute_sha256(pdf_path)

    # Pre-seed cache with pymupdf4llm engine
    checkpoint_payload = {
        "sha256": sha,
        "filename": pdf_path.name,
        "file_type": "pdf",
        "page_count": 1,
        "docling_document": None,
        "extra_meta": {"engine": "pymupdf4llm"},
    }
    (parsed_dir / f"{sha}.json").write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    (markdown_dir / f"{sha}.md").write_text("# PyMuPDF Text", encoding="utf-8")

    parser = DocumentParserService(data_dir=data_dir)

    # 1. Reading with prefer_fallback=True hits the cache
    res1 = parser.parse(pdf_path, use_pymupdf_fallback=True)
    assert res1.from_cache is True
    assert res1.extra_meta["engine"] == "pymupdf4llm"

    # 2. Reading with prefer_fallback=False detects mismatch (cached is pymupdf4llm, wanted docling)
    with patch.object(
        DocumentParserService, "_parse_with_docling", side_effect=_mock_docling_success
    ) as mock_dl:
        res2 = parser.parse(pdf_path, use_pymupdf_fallback=False)
        assert res2.from_cache is False
        assert res2.extra_meta["engine"] == "docling"
        assert mock_dl.called

    # 3. Now cache holds docling. Reading with prefer_fallback=True detects mismatch again
    res3 = parser.parse(pdf_path, use_pymupdf_fallback=True)
    assert res3.from_cache is False
    assert res3.extra_meta["engine"] == "pymupdf4llm"


@pytest.mark.asyncio
async def test_failover_attribution_records_engine_fallback_reason(modes_client, sample_pdf_bytes):
    """9. Failover attribution records engine_fallback_reason on mock Docling crash."""
    client, _ = modes_client

    def _mock_docling_raise(file_path: Path, file_type: str = "pdf"):
        raise RuntimeError("CUDA out of memory: tried to allocate 2.00 GiB")

    with patch.object(
        DocumentParserService, "_parse_with_docling", side_effect=_mock_docling_raise
    ):
        resp = await client.post(
            "/api/v1/documents/upload?prefer_fast_path=false",
            files={"file": ("failing_doc.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
        )
        assert resp.status_code == 202
        doc_id = resp.json()["id"]

        final = await _wait_for_document_ready(client, doc_id)
        assert final["status"] == "ready"
        assert final["parsing_engine"] == "pymupdf4llm"
        assert final["engine_fallback_reason"] is not None
        assert "Docling GPU failed" in final["engine_fallback_reason"]
        assert "CUDA out of memory" in final["engine_fallback_reason"]
        assert "fell back to PyMuPDF CPU" in final["engine_fallback_reason"]


@pytest.mark.asyncio
async def test_duplicate_upload_upgrade_concurrency_cas_coalesces_in_flight(
    modes_client, sample_pdf_bytes
):
    """10. Verify CAS guard on duplicate-upload upgrades: when document is in-flight ('parsing'),
    a subsequent upgrade request triggers res.rowcount == 0 and coalesces into the in-flight
    record with duplicate=True rather than spawning a concurrent task.
    """
    client, _ = modes_client

    # 1. Upload initial document with prefer_fast_path=True -> pymupdf4llm
    resp1 = await client.post(
        "/api/v1/documents/upload?prefer_fast_path=true",
        files={"file": ("sample_cas.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
    )
    assert resp1.status_code == 202
    doc_id = resp1.json()["id"]
    final1 = await _wait_for_document_ready(client, doc_id)
    assert final1["parsing_engine"] == "pymupdf4llm"
    assert final1["status"] == "ready"

    # 2. Simulate document actively processing ('parsing')
    factory = client.app.state.session_factory
    async with factory() as session:
        doc = await session.get(Document, doc_id)
        assert doc is not None
        doc.status = "parsing"
        await session.commit()

    # 3. Fire duplicate upload requesting GPU upgrade (prefer_fast_path=False)
    # Since status is 'parsing', the CAS update (where status in ['ready', 'failed']) will affect 0 rows.
    # It must coalesce safely into the in-flight record with duplicate=True and HTTP 200.
    from src.db.models import IngestionJob
    from sqlalchemy import select, func

    async with factory() as session:
        job_count_before = (
            await session.execute(
                select(func.count()).select_from(IngestionJob).where(IngestionJob.doc_id == doc_id)
            )
        ).scalar_one()

    with patch("src.api.routes.documents._schedule_ingestion_task") as mock_sched:
        resp_upgrade = await client.post(
            "/api/v1/documents/upload?prefer_fast_path=false",
            files={"file": ("sample_cas.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
        )
        assert resp_upgrade.status_code == 200
        up_data = resp_upgrade.json()
        assert up_data["id"] == doc_id
        assert up_data["duplicate"] is True
        assert up_data["status"] == "parsing"
        mock_sched.assert_not_called()

    async with factory() as session:
        job_count_after = (
            await session.execute(
                select(func.count()).select_from(IngestionJob).where(IngestionJob.doc_id == doc_id)
            )
        ).scalar_one()
    assert job_count_after == job_count_before

    # 4. Also test when status is 'chunking' or 'queued'
    for inflight_status in ("chunking", "queued"):
        async with factory() as session:
            doc = await session.get(Document, doc_id)
            doc.status = inflight_status
            await session.commit()

        resp_dup = await client.post(
            "/api/v1/documents/upload?prefer_fast_path=false",
            files={"file": ("sample_cas.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
        )
        assert resp_dup.status_code == 200
        dup_data = resp_dup.json()
        assert dup_data["id"] == doc_id
        assert dup_data["duplicate"] is True
        assert dup_data["status"] == inflight_status

    # 5. Cleanup: restore ready status
    async with factory() as session:
        doc = await session.get(Document, doc_id)
        doc.status = "ready"
        await session.commit()


@pytest.mark.asyncio
async def test_auto_mode_docling_failover_records_fallback_reason(
    modes_client, complex_pdf_bytes
):
    """11. Auto-Mode Docling Fallback Attribution:
    In Auto mode (no prefer_fast_path), a complex layout PDF fails the fast-path probe,
    attempts Docling GPU, and on mock failure falls back to PyMuPDF CPU, recording
    'Auto Docling failed (...); fell back to PyMuPDF CPU' in extra['engine_fallback_reason']
    and persisting to Document.engine_fallback_reason column and status API.
    """
    client, _ = modes_client

    def _mock_docling_crash(file_path: Path, file_type: str = "pdf"):
        raise RuntimeError("Auto Docling timeout / engine failure")

    with patch.object(
        DocumentParserService, "_parse_with_docling", side_effect=_mock_docling_crash
    ):
        # Auto mode: omit prefer_fast_path parameter entirely
        resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("auto_failover_complex.pdf", io.BytesIO(complex_pdf_bytes), "application/pdf")},
        )
        assert resp.status_code == 202
        doc_id = resp.json()["id"]

        final = await _wait_for_document_ready(client, doc_id)
        assert final["status"] == "ready"
        assert final["parsing_engine"] == "pymupdf4llm"
        assert final["engine_fallback_reason"] is not None
        assert final["engine_fallback_reason"].startswith("Auto Docling failed")
        assert "Auto Docling timeout / engine failure" in final["engine_fallback_reason"]
        assert "fell back to PyMuPDF CPU" in final["engine_fallback_reason"]

        # Directly verify DB column persistence
        factory = client.app.state.session_factory
        async with factory() as session:
            doc = await session.get(Document, doc_id)
            assert doc is not None
            assert doc.engine_fallback_reason is not None
            assert doc.engine_fallback_reason.startswith("Auto Docling failed")
            assert "fell back to PyMuPDF CPU" in doc.engine_fallback_reason
