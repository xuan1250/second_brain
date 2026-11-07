"""Unit tests for Phase 1 Batch Ingestion Resilience and Cross-Tier Consistency."""
from __future__ import annotations

import asyncio
from concurrent.futures.process import BrokenProcessPool
import json
from pathlib import Path
import uuid

import docx
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from sqlalchemy import func, select

from src.db.models import Document, DocumentChunk, IngestionJob
from src.db.qdrant import init_qdrant_collection
from src.db.repository import delete_document_atomic
from src.db.session import create_engine, create_session_factory, init_db
from src.ingestion.chunker import ChunkPayload
from src.ingestion.indexer import (
    DeterministicBM25Encoder,
    DeterministicStubEmbedder,
    HybridIndexer,
)
from src.ingestion.worker import IngestionOrchestrator

TEST_COLLECTION = "test_resilience_collection"
STUB_DIM = 32


@pytest.fixture
async def db_session_factory(tmp_path: Path):
    db_file = tmp_path / "db" / "resilience_test.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_file.as_posix()}")
    await init_db(engine)
    factory = create_session_factory(engine)
    yield factory
    await engine.dispose()


@pytest.fixture
async def memory_qdrant():
    client = AsyncQdrantClient(location=":memory:")
    yield client
    await client.close()


@pytest.fixture
def sample_docx(tmp_path: Path) -> Path:
    path = tmp_path / "resilience_sample.docx"
    d = docx.Document()
    d.add_heading("Resilience Architecture", level=1)
    d.add_paragraph("Robust worker process management and consistency barriers.")
    d.save(path)
    return path


@pytest.mark.asyncio
async def test_broken_process_pool_auto_recovery(
    tmp_path: Path,
    db_session_factory,
    memory_qdrant: AsyncQdrantClient,
    sample_docx: Path,
):
    """Verify BrokenProcessPool triggers pool reclamation, marks document failed, and heals for subsequent files."""
    data_dir = tmp_path / "data"
    stub_embedder = DeterministicStubEmbedder(dimensions=STUB_DIM)
    stub_sparse = DeterministicBM25Encoder()

    async with IngestionOrchestrator(
        session_factory=db_session_factory,
        qdrant=memory_qdrant,
        embedder=stub_embedder,
        sparse_encoder=stub_sparse,
        data_dir=data_dir,
        collection=TEST_COLLECTION,
        max_workers=1,
        use_process_pool=True,
    ) as orchestrator:
        # Document 1: Simulate poison PDF causing BrokenProcessPool during parsing
        sha1 = "1" * 64
        async with db_session_factory() as session:
            doc1 = Document(
                filename="poison.pdf",
                file_type="pdf",
                sha256=sha1,
                size_bytes=1024,
                status="queued",
            )
            session.add(doc1)
            await session.commit()
            await session.refresh(doc1)
            doc1_id = doc1.id

        original_run_in_executor = asyncio.get_running_loop().run_in_executor
        broken_triggered = False

        def broken_executor(pool, func, *args):
            nonlocal broken_triggered
            if not broken_triggered:
                broken_triggered = True
                f = asyncio.Future()
                f.set_exception(BrokenProcessPool("Simulated poison PDF worker crash (segfault)"))
                return f
            return original_run_in_executor(pool, func, *args)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(asyncio.get_running_loop(), "run_in_executor", broken_executor)
            res1 = await orchestrator.process_document(doc1_id)

        assert res1.status == "failed"
        assert "fatal worker process termination" in (res1.error or "")

        # Verify Document row in SQLite is marked failed with poison doc message
        async with db_session_factory() as session:
            d1 = await session.get(Document, doc1_id)
            assert d1 is not None
            assert d1.status == "failed"
            assert "fatal worker process termination" in (d1.error or "")

        # Verify pool was reclaimed and active jobs returned to 0
        state = orchestrator.get_worker_state()
        assert state["worker_reclaim_count"] >= 1
        assert state["executing_tasks"] == 0

        # Document 2: Verify fresh pool recovers and processes healthy file end-to-end
        doc2 = await orchestrator.ingest_file(sample_docx)
        assert doc2.status == "ready"
        assert doc2.chunk_count > 0


@pytest.mark.asyncio
async def test_pre_upsert_barrier_aborts_on_deleted_doc(
    tmp_path: Path,
    db_session_factory,
    memory_qdrant: AsyncQdrantClient,
):
    """Verify that a document deleted or aborted before Qdrant upsert is discarded without creating ghost vectors."""
    stub_embedder = DeterministicStubEmbedder(dimensions=STUB_DIM)
    stub_sparse = DeterministicBM25Encoder()
    indexer = HybridIndexer(
        qdrant=memory_qdrant,
        embedder=stub_embedder,
        sparse_encoder=stub_sparse,
        collection=TEST_COLLECTION,
    )

    chunks = [
        ChunkPayload(
            chunk_id=str(uuid.uuid4()),
            chunk_index=0,
            parent_id=str(uuid.uuid4()),
            source_filename="deleted.docx",
            file_type="docx",
            heading_path="Heading 1",
            page_numbers=[],
            token_count=10,
            child_text="Some text",
            contextualized_text="deleted.docx > Heading 1: Some text",
            parent_content="Some text",
        )
    ]

    # Case 1: Document does not exist in SQLite
    async with db_session_factory() as session:
        count = await indexer.index_chunks(session, doc_id="missing_doc_id", chunks=chunks)
        assert count == 0

    # Case 2: Document exists but status is failed/cancelled
    async with db_session_factory() as session:
        doc = Document(
            filename="aborted.docx",
            file_type="docx",
            sha256="2" * 64,
            size_bytes=100,
            status="failed",
        )
        session.add(doc)
        await session.commit()
        await session.refresh(doc)
        doc_id = doc.id

    async with db_session_factory() as session:
        count = await indexer.index_chunks(session, doc_id=doc_id, chunks=chunks)
        assert count == 0

    # Ensure zero orphan points exist in Qdrant
    points, _ = await memory_qdrant.scroll(collection_name=TEST_COLLECTION, limit=10)
    assert len(points) == 0


@pytest.mark.asyncio
async def test_compensating_rollback_purges_partial_points(
    tmp_path: Path,
    db_session_factory,
    memory_qdrant: AsyncQdrantClient,
):
    """Verify that if an exception occurs mid-indexing, previously upserted points are rolled back from Qdrant."""
    stub_embedder = DeterministicStubEmbedder(dimensions=STUB_DIM)
    stub_sparse = DeterministicBM25Encoder()

    # batch_size=1 so each chunk executes a separate upsert batch
    indexer = HybridIndexer(
        qdrant=memory_qdrant,
        embedder=stub_embedder,
        sparse_encoder=stub_sparse,
        collection=TEST_COLLECTION,
        batch_size=1,
    )

    async with db_session_factory() as session:
        doc = Document(
            filename="partial.docx",
            file_type="docx",
            sha256="3" * 64,
            size_bytes=200,
            status="indexing",
        )
        session.add(doc)
        await session.commit()
        await session.refresh(doc)
        doc_id = doc.id

    chunks = [
        ChunkPayload(
            chunk_id=str(uuid.uuid4()),
            chunk_index=i,
            parent_id=str(uuid.uuid4()),
            source_filename="partial.docx",
            file_type="docx",
            heading_path=f"Heading {i}",
            page_numbers=[],
            token_count=10,
            child_text=f"Text {i}",
            contextualized_text=f"partial.docx > Heading {i}: Text {i}",
            parent_content=f"Text {i}",
        )
        for i in range(3)
    ]

    original_upsert = memory_qdrant.upsert
    upsert_call_count = 0

    async def failing_upsert(*args, **kwargs):
        nonlocal upsert_call_count
        upsert_call_count += 1
        if upsert_call_count == 2:
            raise RuntimeError("Simulated Qdrant network failure on batch 2")
        return await original_upsert(*args, **kwargs)

    memory_qdrant.upsert = failing_upsert

    async with db_session_factory() as session:
        with pytest.raises(RuntimeError, match="Simulated Qdrant network failure on batch 2"):
            await indexer.index_chunks(session, doc_id=doc_id, chunks=chunks)

    memory_qdrant.upsert = original_upsert

    # Confirm that batch 1 points were purged by compensating rollback
    points, _ = await memory_qdrant.scroll(collection_name=TEST_COLLECTION, limit=10)
    assert len(points) == 0


@pytest.mark.asyncio
async def test_in_flight_cancellation(
    tmp_path: Path,
    db_session_factory,
    memory_qdrant: AsyncQdrantClient,
    sample_docx: Path,
):
    """Verify that in-flight cancellation terminates stage progression cleanly and cleans up worker state."""
    data_dir = tmp_path / "data"
    stub_embedder = DeterministicStubEmbedder(dimensions=STUB_DIM)
    stub_sparse = DeterministicBM25Encoder()

    async with IngestionOrchestrator(
        session_factory=db_session_factory,
        qdrant=memory_qdrant,
        embedder=stub_embedder,
        sparse_encoder=stub_sparse,
        data_dir=data_dir,
        collection=TEST_COLLECTION,
        max_workers=1,
        use_process_pool=False,
    ) as orchestrator:
        sha = "4" * 64
        async with db_session_factory() as session:
            doc = Document(
                filename="cancel.docx",
                file_type="docx",
                sha256=sha,
                size_bytes=100,
                status="queued",
            )
            session.add(doc)
            await session.commit()
            await session.refresh(doc)
            doc_id = doc.id

        original_update = orchestrator._update_state

        async def hooked_update(d_id, j_id, **kwargs):
            if kwargs.get("stage") == "parsing":
                orchestrator.cancel_document(d_id)
            return await original_update(d_id, j_id, **kwargs)

        orchestrator._update_state = hooked_update

        res = await orchestrator.process_document(doc_id=doc_id, raw_path=sample_docx)
        assert res.status == "failed"
        assert res.error == "Ingestion cancelled"

        # Check DB state
        async with db_session_factory() as session:
            d = await session.get(Document, doc_id)
            assert d is not None
            assert d.status == "failed"
            assert d.error == "Ingestion cancelled"

        # Check worker slots clean
        assert orchestrator.get_worker_state()["executing_tasks"] == 0


@pytest.mark.asyncio
async def test_atomic_delete_signals_cancellation(
    tmp_path: Path,
    db_session_factory,
    memory_qdrant: AsyncQdrantClient,
):
    """Verify delete_document_atomic cancels orchestrator work when orchestrator is provided."""
    data_dir = tmp_path / "data"
    stub_embedder = DeterministicStubEmbedder(dimensions=STUB_DIM)
    stub_sparse = DeterministicBM25Encoder()

    async with IngestionOrchestrator(
        session_factory=db_session_factory,
        qdrant=memory_qdrant,
        embedder=stub_embedder,
        sparse_encoder=stub_sparse,
        data_dir=data_dir,
        collection=TEST_COLLECTION,
        max_workers=1,
        use_process_pool=False,
    ) as orchestrator:
        sha = "5" * 64
        async with db_session_factory() as session:
            doc = Document(
                filename="delete_cancel.docx",
                file_type="docx",
                sha256=sha,
                size_bytes=100,
                status="queued",
            )
            session.add(doc)
            await session.commit()
            await session.refresh(doc)
            doc_id = doc.id

        # Register a fake cancellation event to verify orchestrator receives signal
        cancel_event = asyncio.Event()
        orchestrator._cancellations[doc_id] = cancel_event
        await init_qdrant_collection(memory_qdrant, TEST_COLLECTION, STUB_DIM)

        async with db_session_factory() as session:
            deleted = await delete_document_atomic(
                session=session,
                qdrant=memory_qdrant,
                doc_id=doc_id,
                data_dir=data_dir,
                collection=TEST_COLLECTION,
                orchestrator=orchestrator,
            )
            assert deleted is True
            assert cancel_event.is_set()


@pytest.mark.asyncio
async def test_mid_batch_worker_recycling(
    tmp_path: Path,
    db_session_factory,
    memory_qdrant: AsyncQdrantClient,
    sample_docx: Path,
):
    """Verify worker_max_tasks > 0 triggers mid-batch recycling after Stage 2."""
    data_dir = tmp_path / "data"
    stub_embedder = DeterministicStubEmbedder(dimensions=STUB_DIM)
    stub_sparse = DeterministicBM25Encoder()

    async with IngestionOrchestrator(
        session_factory=db_session_factory,
        qdrant=memory_qdrant,
        embedder=stub_embedder,
        sparse_encoder=stub_sparse,
        data_dir=data_dir,
        collection=TEST_COLLECTION,
        max_workers=1,
        use_process_pool=True,
        worker_max_tasks=1,
    ) as orchestrator:
        doc = await orchestrator.ingest_file(sample_docx)
        assert doc.status == "ready"
        state = orchestrator.get_worker_state()
        assert state["worker_reclaim_count"] >= 1


def test_inspect_pdf_fast_path_heuristic(tmp_path: Path):
    """Verify PyMuPDF fast-path heuristic identifies digital text vs scanned/heavy layout and handles errors."""
    import pymupdf
    from src.ingestion.parser import inspect_pdf_fast_path

    # 1. Clean digital text PDF (> 120 chars/page, 0 images)
    digital_pdf = tmp_path / "digital.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    rect = pymupdf.Rect(50, 50, 500, 700)
    page.insert_textbox(rect, ("Second Brain batch ingestion test digital content. " * 5 + "\n") * 5)
    doc.save(str(digital_pdf))
    doc.close()

    is_fast, reason = inspect_pdf_fast_path(digital_pdf)
    assert is_fast is True
    assert "digital_text" in reason

    # 2. Scanned / sparse text PDF (< 120 chars/page)
    sparse_pdf = tmp_path / "sparse.pdf"
    doc2 = pymupdf.open()
    page2 = doc2.new_page()
    page2.insert_text((72, 72), "Minimal")
    doc2.save(str(sparse_pdf))
    doc2.close()

    is_fast2, reason2 = inspect_pdf_fast_path(sparse_pdf)
    assert is_fast2 is False
    assert "scanned_or_heavy_layout" in reason2

    # 3. Corrupted / non-PDF file
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"garbage not pdf")
    is_fast3, reason3 = inspect_pdf_fast_path(corrupt_pdf)
    assert is_fast3 is False
    assert "preflight_inspection_error" in reason3


def test_document_parser_fast_path_routing(tmp_path: Path):
    """Verify DocumentParserService routes digital PDFs to pymupdf4llm when prefer_pymupdf4llm is None (auto)."""
    import pymupdf
    from src.ingestion.parser import DocumentParserService

    digital_pdf = tmp_path / "digital_router.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    rect = pymupdf.Rect(50, 50, 500, 700)
    page.insert_textbox(rect, ("Fast path qualification text for DocumentParserService routing test. " * 5 + "\n") * 5)
    doc.save(str(digital_pdf))
    doc.close()

    parser = DocumentParserService(data_dir=tmp_path / "data", prefer_pymupdf4llm=None)
    res = parser.parse(digital_pdf)
    assert res.extra_meta.get("engine") == "pymupdf4llm"
    assert res.page_count == 1
    assert "Fast path qualification text" in res.markdown


@pytest.mark.asyncio
async def test_stage_checkpointing_resumes_without_reparsing(
    tmp_path: Path,
    db_session_factory,
    memory_qdrant: AsyncQdrantClient,
    sample_docx: Path,
):
    """Verify that retrying an ingestion when parsed JSON/MD exist skips Stage 1 (Parsing) with parsing_ms=1."""
    data_dir = tmp_path / "data"
    stub_embedder = DeterministicStubEmbedder(dimensions=STUB_DIM)
    stub_sparse = DeterministicBM25Encoder()

    async with IngestionOrchestrator(
        session_factory=db_session_factory,
        qdrant=memory_qdrant,
        embedder=stub_embedder,
        sparse_encoder=stub_sparse,
        data_dir=data_dir,
        collection=TEST_COLLECTION,
        max_workers=1,
        use_process_pool=False,
    ) as orchestrator:
        # Initial ingestion to generate checkpoint artifacts
        doc = await orchestrator.ingest_file(sample_docx)
        assert doc.status == "ready"

        sha = doc.sha256
        parsed_path = data_dir / "parsed" / f"{sha}.json"
        markdown_path = data_dir / "markdown" / f"{sha}.md"
        assert parsed_path.is_file()
        assert markdown_path.is_file()

        # Simulate job failure / retry by resetting Document status to queued and creating new job
        async with db_session_factory() as session:
            d = await session.get(Document, doc.id)
            assert d is not None
            d.status = "queued"
            d.error = None
            session.add(d)
            job = IngestionJob(
                doc_id=d.id,
                status="queued",
                stage="queued",
                progress=0.0,
                stage_timings="{}",
            )
            session.add(job)
            await session.commit()
            job_id = job.id

        # Spy on worker parse task to ensure it is NOT called during checkpointed retry
        parse_task_called = False

        from src.ingestion import worker

        def fail_if_parse_called(*args, **kwargs):
            nonlocal parse_task_called
            parse_task_called = True
            raise RuntimeError("_worker_parse_task should NOT be called when checkpoint exists!")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(worker, "_worker_parse_task", fail_if_parse_called)

            res = await orchestrator.process_document(
                doc_id=doc.id,
                job_id=job_id,
                force_reparse=False,
            )

        assert parse_task_called is False
        assert res.status == "ready"

        # Check job timings: parsing_ms must be 1 (checkpointed)
        async with db_session_factory() as session:
            j = await session.get(IngestionJob, job_id)
            assert j is not None
            timings = json.loads(j.stage_timings)
            assert timings["parsing_ms"] == 1
            assert "chunking_ms" in timings
            assert "indexing_ms" in timings


@pytest.mark.asyncio
async def test_force_reparse_bypasses_checkpoint(
    tmp_path: Path,
    db_session_factory,
    memory_qdrant: AsyncQdrantClient,
    sample_docx: Path,
):
    """Verify that force_reparse=True executes Stage 1 parsing even when checkpoint files exist."""
    data_dir = tmp_path / "data"
    stub_embedder = DeterministicStubEmbedder(dimensions=STUB_DIM)
    stub_sparse = DeterministicBM25Encoder()

    async with IngestionOrchestrator(
        session_factory=db_session_factory,
        qdrant=memory_qdrant,
        embedder=stub_embedder,
        sparse_encoder=stub_sparse,
        data_dir=data_dir,
        collection=TEST_COLLECTION,
        max_workers=1,
        use_process_pool=False,
    ) as orchestrator:
        doc = await orchestrator.ingest_file(sample_docx)
        assert doc.status == "ready"

        # Reset status
        async with db_session_factory() as session:
            d = await session.get(Document, doc.id)
            assert d is not None
            d.status = "queued"
            session.add(d)
            job = IngestionJob(doc_id=d.id, status="queued", stage="queued", progress=0.0, stage_timings="{}")
            session.add(job)
            await session.commit()
            job_id = job.id

        parse_task_called = False
        from src.ingestion import worker

        real_parse = worker._worker_parse_task

        def spy_parse(*args, **kwargs):
            nonlocal parse_task_called
            parse_task_called = True
            return real_parse(*args, **kwargs)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(worker, "_worker_parse_task", spy_parse)
            res = await orchestrator.process_document(
                doc_id=doc.id,
                job_id=job_id,
                force_reparse=True,
            )

        assert parse_task_called is True
        assert res.status == "ready"


def test_fast_path_heuristic_digital_vs_scanned(tmp_path: Path):
    """Verify PyMuPDF fast-path heuristic correctly classifies digital text vs rasterized scanned PDFs."""
    import pymupdf
    from src.ingestion.parser import inspect_pdf_fast_path

    # 1. Clean digital text PDF (> 120 chars/page, 0 images) -> fast path qualifies
    digital_pdf = tmp_path / "digital_text_fastpath.pdf"
    doc_digital = pymupdf.open()
    page_digital = doc_digital.new_page()
    rect = pymupdf.Rect(50, 50, 500, 700)
    page_digital.insert_textbox(
        rect,
        ("Second Brain fast-path qualification digital text sample with dense characters. " * 4 + "\n") * 5,
    )
    doc_digital.save(str(digital_pdf))
    doc_digital.close()

    is_fast_digital, reason_digital = inspect_pdf_fast_path(digital_pdf)
    assert is_fast_digital is True
    assert "digital_text" in reason_digital

    # 2. Rasterized image PDF (embedded pixmap, 0 characters) -> fast path rejects
    raster_pdf = tmp_path / "rasterized_scan_fastpath.pdf"
    doc_raster = pymupdf.open()
    page_raster = doc_raster.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), 0)
    page_raster.insert_image(pymupdf.Rect(50, 50, 200, 200), pixmap=pix)
    doc_raster.save(str(raster_pdf))
    doc_raster.close()

    is_fast_raster, reason_raster = inspect_pdf_fast_path(raster_pdf)
    assert is_fast_raster is False
    assert "scanned_or_heavy_layout" in reason_raster


@pytest.mark.asyncio
async def test_concurrent_duplicate_upload_integrity_handling(
    tmp_path: Path,
    db_session_factory,
    memory_qdrant: AsyncQdrantClient,
    sample_docx: Path,
):
    """Simulate concurrent execution of upload_documents with identical hash/content;
    asserts neither raises unhandled IntegrityError and one returns duplicate=True."""
    import io
    from fastapi import FastAPI, UploadFile
    from starlette.requests import Request
    from starlette.responses import Response

    from src.api.routes import documents as doc_route_mod
    from src.api.routes.documents import upload_documents
    from src.config import Settings, StorageConfig

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    stub_embedder = DeterministicStubEmbedder(dimensions=STUB_DIM)
    stub_sparse = DeterministicBM25Encoder()

    async with IngestionOrchestrator(
        session_factory=db_session_factory,
        qdrant=memory_qdrant,
        embedder=stub_embedder,
        sparse_encoder=stub_sparse,
        data_dir=data_dir,
        collection=TEST_COLLECTION,
        max_workers=1,
        use_process_pool=False,
    ) as orchestrator:
        settings = Settings(
            storage=StorageConfig(
                data_dir=str(data_dir),
                db_dir=str(tmp_path / "db"),
                frontend_dist_dir=str(tmp_path / "frontend_dist"),
            )
        )
        file_bytes = sample_docx.read_bytes()

        app = FastAPI()
        app.state.background_tasks = set()
        scope1 = {"type": "http", "method": "POST", "headers": [], "app": app}
        scope2 = {"type": "http", "method": "POST", "headers": [], "app": app}
        req1 = Request(scope1)
        req2 = Request(scope2)
        resp1 = Response()
        resp2 = Response()

        f1 = UploadFile(file=io.BytesIO(file_bytes), filename="race_sample.docx")
        f2 = UploadFile(file=io.BytesIO(file_bytes), filename="race_sample.docx")

        # Synchronize at get_document_by_sha to deterministically trigger SQLite UNIQUE constraint collision
        barrier = asyncio.Barrier(2)
        orig_get_sha = doc_route_mod.get_document_by_sha
        sha_call_count = 0

        async def synced_get_sha(session, sha):
            nonlocal sha_call_count
            res = await orig_get_sha(session, sha)
            sha_call_count += 1
            if sha_call_count <= 2:
                await barrier.wait()
            return res

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(doc_route_mod, "get_document_by_sha", synced_get_sha)

            async with db_session_factory() as session1, db_session_factory() as session2:
                res1, res2 = await asyncio.gather(
                    upload_documents(
                        request=req1,
                        response=resp1,
                        file=f1,
                        files=None,
                        session=session1,
                        orchestrator=orchestrator,
                        settings=settings,
                    ),
                    upload_documents(
                        request=req2,
                        response=resp2,
                        file=f2,
                        files=None,
                        session=session2,
                        orchestrator=orchestrator,
                        settings=settings,
                    ),
                )

        assert res1 is not None
        assert res2 is not None

        duplicates = [res1.duplicate, res2.duplicate]
        assert any(duplicates), f"Expected at least one duplicate=True, got {duplicates}"
        assert not all(duplicates), f"Expected exactly one original and one duplicate, got {duplicates}"
        assert res1.id == res2.id
        assert res1.sha256 == res2.sha256

        # Drain scheduled background tasks
        if app.state.background_tasks:
            await asyncio.gather(*list(app.state.background_tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_retry_document_purges_previous_chunks_and_points(
    tmp_path: Path,
    db_session_factory,
    memory_qdrant: AsyncQdrantClient,
    sample_docx: Path,
):
    """Verify that retrying an already-indexed document purges previous vectors in Qdrant and chunks in SQLite, preserving chunk count invariance."""
    data_dir = tmp_path / "data"
    stub_embedder = DeterministicStubEmbedder(dimensions=STUB_DIM)
    stub_sparse = DeterministicBM25Encoder()

    async with IngestionOrchestrator(
        session_factory=db_session_factory,
        qdrant=memory_qdrant,
        embedder=stub_embedder,
        sparse_encoder=stub_sparse,
        data_dir=data_dir,
        collection=TEST_COLLECTION,
        max_workers=1,
        use_process_pool=False,
    ) as orchestrator:
        # 1. Initial ingestion
        doc = await orchestrator.ingest_file(sample_docx)
        assert doc.status == "ready"
        initial_chunks = doc.chunk_count
        assert initial_chunks > 0

        # Check count in SQLite
        async with db_session_factory() as session:
            initial_db_chunks = (
                await session.execute(select(func.count()).where(DocumentChunk.doc_id == doc.id))
            ).scalar()
        assert initial_db_chunks == initial_chunks

        # Check points in Qdrant
        qdrant_points = await memory_qdrant.count(
            collection_name=TEST_COLLECTION,
            count_filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc.id))]),
            exact=True,
        )
        assert qdrant_points.count == initial_chunks

        # 2. Re-process / retry document with force_reparse=True
        retried_doc = await orchestrator.process_document(doc.id, force_reparse=True)
        assert retried_doc.status == "ready"

        # 3. Invariance checks: SQLite chunk count must NOT double
        async with db_session_factory() as session:
            after_db_chunks = (
                await session.execute(select(func.count()).where(DocumentChunk.doc_id == doc.id))
            ).scalar()
        assert after_db_chunks == initial_chunks, f"Expected {initial_chunks} chunks, got {after_db_chunks} (duplicates detected!)"

        # 4. Invariance checks: Qdrant points count must NOT double
        after_qdrant_points = await memory_qdrant.count(
            collection_name=TEST_COLLECTION,
            count_filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc.id))]),
            exact=True,
        )
        assert after_qdrant_points.count == initial_chunks, f"Expected {initial_chunks} points in Qdrant, got {after_qdrant_points.count} (duplicate vectors detected!)"


