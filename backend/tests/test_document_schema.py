"""Tests for Document model schema, DocumentResponse serialization, idempotent migrations, and zero-disk-I/O response building."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import text
from sqlmodel import select

from src.api.routes.documents import _build_document_response
from src.api.schemas import DocumentResponse
from src.db.models import Document, IngestionJob
from src.db.session import create_engine, create_session_factory, init_db


def _now() -> datetime:
    return datetime.now(timezone.utc)


def test_document_model_instantiation():
    """Verify Document model instantiates with default and explicit parsing_engine/fallback_reason."""
    doc_default = Document(
        filename="test.pdf",
        file_type="pdf",
        sha256="a" * 64,
    )
    assert doc_default.parsing_engine is None
    assert doc_default.engine_fallback_reason is None

    doc_explicit = Document(
        filename="test2.pdf",
        file_type="pdf",
        sha256="b" * 64,
        parsing_engine="docling",
        engine_fallback_reason=None,
    )
    assert doc_explicit.parsing_engine == "docling"
    assert doc_explicit.engine_fallback_reason is None

    doc_fallback = Document(
        filename="test3.pdf",
        file_type="pdf",
        sha256="c" * 64,
        parsing_engine="pymupdf4llm",
        engine_fallback_reason="Docling GPU failed (CUDA OOM); fell back to PyMuPDF CPU",
    )
    assert doc_fallback.parsing_engine == "pymupdf4llm"
    assert doc_fallback.engine_fallback_reason == "Docling GPU failed (CUDA OOM); fell back to PyMuPDF CPU"


def test_document_response_serialization():
    """Verify DocumentResponse serializes and deserializes parsing_engine and engine_fallback_reason."""
    now = _now()
    resp = DocumentResponse(
        id="doc_123",
        filename="test.pdf",
        file_type="pdf",
        sha256="d" * 64,
        size_bytes=1024,
        status="ready",
        parsing_engine="docling",
        engine_fallback_reason=None,
        created_at=now,
        updated_at=now,
    )
    data = resp.model_dump()
    assert data["parsing_engine"] == "docling"
    assert data["engine_fallback_reason"] is None

    json_str = resp.model_dump_json()
    assert '"parsing_engine":"docling"' in json_str
    assert '"engine_fallback_reason":null' in json_str

    deserialized = DocumentResponse.model_validate_json(json_str)
    assert deserialized.parsing_engine == "docling"
    assert deserialized.engine_fallback_reason is None

    # Test with fallback reason
    resp_fallback = DocumentResponse(
        id="doc_456",
        filename="test.pdf",
        file_type="pdf",
        sha256="e" * 64,
        size_bytes=2048,
        status="ready",
        parsing_engine="pymupdf4llm",
        engine_fallback_reason="Fallback due to timeout",
        created_at=now,
        updated_at=now,
    )
    data_fallback = resp_fallback.model_dump()
    assert data_fallback["parsing_engine"] == "pymupdf4llm"
    assert data_fallback["engine_fallback_reason"] == "Fallback due to timeout"


async def test_init_db_migration_on_legacy_schema(tmp_path):
    """Verify init_db() migrates existing SQLite tables lacking parsing columns idempotently."""
    db_file = tmp_path / "legacy.db"
    db_url = f"sqlite+aiosqlite:///{db_file.as_posix()}"

    # 1. Create a legacy database using raw sqlite3 without parsing_engine or engine_fallback_reason
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """
        CREATE TABLE documents (
            id VARCHAR PRIMARY KEY,
            filename VARCHAR NOT NULL,
            file_type VARCHAR NOT NULL,
            sha256 VARCHAR NOT NULL,
            size_bytes INTEGER NOT NULL,
            page_count INTEGER,
            chunk_count INTEGER NOT NULL,
            status VARCHAR NOT NULL,
            error VARCHAR,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO documents (id, filename, file_type, sha256, size_bytes, chunk_count, status, created_at, updated_at)
        VALUES ('doc_legacy', 'legacy.pdf', 'pdf', 'f' * 64, 500, 0, 'ready', '2026-10-01 00:00:00', '2026-10-01 00:00:00')
        """
    )
    conn.commit()

    # Verify columns in raw sqlite before migration
    columns_before = [row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()]
    assert "parsing_engine" not in columns_before
    assert "engine_fallback_reason" not in columns_before
    conn.close()

    # 2. Run init_db() on the legacy database
    engine = create_engine(db_url)
    await init_db(engine)

    # 3. Verify columns exist now
    factory = create_session_factory(engine)
    async with factory() as session:
        result = await session.execute(text("PRAGMA table_info(documents)"))
        columns_after = [row[1] for row in result.fetchall()]
        assert "parsing_engine" in columns_after
        assert "engine_fallback_reason" in columns_after

        # Verify existing row can be loaded through SQLModel
        doc = await session.get(Document, "doc_legacy")
        assert doc is not None
        assert doc.id == "doc_legacy"
        assert doc.parsing_engine is None
        assert doc.engine_fallback_reason is None

    # 4. Verify idempotency: run init_db() again on already-migrated database
    await init_db(engine)
    async with factory() as session:
        result = await session.execute(text("PRAGMA table_info(documents)"))
        columns_rerun = [row[1] for row in result.fetchall()]
        assert columns_rerun.count("parsing_engine") == 1
        assert columns_rerun.count("engine_fallback_reason") == 1

    await engine.dispose()


def test_build_document_response_zero_disk_io():
    """Verify _build_document_response reads from Document model without touching disk files."""
    doc = Document(
        id="doc_fast",
        filename="paper.pdf",
        file_type="pdf",
        sha256="1234567890abcdef" * 4,
        size_bytes=4096,
        page_count=12,
        chunk_count=24,
        status="ready",
        parsing_engine="docling",
        engine_fallback_reason=None,
    )
    job = IngestionJob(
        doc_id=doc.id,
        status="ready",
        stage="ready",
        progress=1.0,
        duration_ms=2500,
        stage_timings='{"parsing_ms": 1200, "chunking_ms": 300, "indexing_ms": 1000}',
    )

    # Patch open to ensure zero disk reads occur during response construction
    with patch("builtins.open", side_effect=AssertionError("Disk read detected during _build_document_response!")):
        resp = _build_document_response(doc, job)

    assert resp.id == doc.id
    assert resp.parsing_engine == "docling"
    assert resp.engine_fallback_reason is None
    assert resp.status == "ready"
    assert resp.stage_timings == {"parsing_ms": 1200, "chunking_ms": 300, "indexing_ms": 1000}


async def test_update_state_persists_parsing_attribution(tmp_path):
    """Verify orchestrator _update_state persists parsing_engine and preserves it across stages."""
    from src.ingestion.worker import IngestionOrchestrator

    db_file = tmp_path / "worker_test.db"
    db_url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
    engine = create_engine(db_url)
    await init_db(engine)
    factory = create_session_factory(engine)

    # Create test document and job
    doc_id = "doc_orch_test"
    job_id = "job_orch_test"
    async with factory() as session:
        doc = Document(
            id=doc_id,
            filename="report.pdf",
            file_type="pdf",
            sha256="9" * 64,
            status="queued",
        )
        job = IngestionJob(
            id=job_id,
            doc_id=doc_id,
            status="queued",
            stage="queued",
        )
        session.add(doc)
        session.add(job)
        await session.commit()

    orchestrator = IngestionOrchestrator(
        session_factory=factory,
        data_dir=tmp_path / "data",
    )

    # Stage 1 complete -> update state with parsing_engine and fallback reason
    await orchestrator._update_state(
        doc_id=doc_id,
        job_id=job_id,
        status="chunking",
        stage="chunking",
        progress=0.55,
        timings={"parsing_ms": 150},
        page_count=5,
        parsing_engine="pymupdf4llm",
        engine_fallback_reason="Explicit fast-path chosen",
    )

    async with factory() as session:
        doc_check = await session.get(Document, doc_id)
        assert doc_check is not None
        assert doc_check.status == "chunking"
        assert doc_check.parsing_engine == "pymupdf4llm"
        assert doc_check.engine_fallback_reason == "Explicit fast-path chosen"

    # Stage 3 (indexing) without passing parsing_engine -> existing attribution must not be overwritten
    await orchestrator._update_state(
        doc_id=doc_id,
        job_id=job_id,
        status="indexing",
        stage="indexing",
        progress=0.8,
        timings={"parsing_ms": 150, "chunking_ms": 50},
    )

    async with factory() as session:
        doc_check2 = await session.get(Document, doc_id)
        assert doc_check2 is not None
        assert doc_check2.status == "indexing"
        assert doc_check2.parsing_engine == "pymupdf4llm"
        assert doc_check2.engine_fallback_reason == "Explicit fast-path chosen"

    # Stage 4 (ready)
    await orchestrator._update_state(
        doc_id=doc_id,
        job_id=job_id,
        status="ready",
        stage="ready",
        progress=1.0,
        timings={"parsing_ms": 150, "chunking_ms": 50, "indexing_ms": 200},
        duration_ms=400,
        chunk_count=10,
        page_count=5,
        parsing_engine="pymupdf4llm",
        engine_fallback_reason="Explicit fast-path chosen",
    )

    async with factory() as session:
        doc_check3 = await session.get(Document, doc_id)
        assert doc_check3 is not None
        assert doc_check3.status == "ready"
        assert doc_check3.parsing_engine == "pymupdf4llm"
        assert doc_check3.engine_fallback_reason == "Explicit fast-path chosen"

    await engine.dispose()

