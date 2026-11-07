"""Unit and integration tests for Phase 2 Document Ingestion Pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import docx
import openpyxl
import pptx
import pymupdf
import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select

from src.db.models import Document, DocumentChunk, IngestionJob
from src.db.qdrant import DENSE, SPARSE
from src.db.session import create_engine, create_session_factory, init_db
from src.ingestion.chunker import HierarchicalChunker, format_contextualized_text
from src.ingestion.cli import collect_target_files, main as cli_main
from src.ingestion.indexer import (
    DeterministicBM25Encoder,
    DeterministicStubEmbedder,
    HybridIndexer,
    LocalBgeM3Embedder,
)
from src.ingestion.parser import DocumentParserService, compute_sha256
from src.ingestion.worker import IngestionOrchestrator

TEST_COLLECTION = "test_second_brain_ingestion"
STUB_DIM = 32


@pytest.fixture
async def db_session_factory(tmp_path: Path):
    db_file = tmp_path / "db" / "second_brain.db"
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
    path = tmp_path / "architecture_spec.docx"
    d = docx.Document()
    d.add_heading("System Architecture", level=1)
    d.add_paragraph("The Second Brain platform uses a decoupled FastAPI core with Qdrant hybrid search.")
    d.add_heading("Storage Layer", level=2)
    d.add_paragraph("Relational state lives in SQLite WAL mode while vectors live in Qdrant.")
    table = d.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Component"
    table.cell(0, 1).text = "Technology"
    table.cell(1, 0).text = "Vector DB"
    table.cell(1, 1).text = "Qdrant Hybrid"
    d.save(path)
    return path


@pytest.fixture
def sample_pptx(tmp_path: Path) -> Path:
    path = tmp_path / "quarterly_review.pptx"
    prs = pptx.Presentation()
    slide_layout = prs.slide_layouts[1]

    s1 = prs.slides.add_slide(slide_layout)
    s1.shapes.title.text = "Q3 Executive Summary"
    tf1 = s1.placeholders[1].text_frame
    tf1.text = "Operating margin expanded by 3.2% year-over-year."
    p = tf1.add_paragraph()
    p.text = "Supply chain optimization reduced unit cost."

    s2 = prs.slides.add_slide(slide_layout)
    s2.shapes.title.text = "Q4 Strategic Roadmap"
    tf2 = s2.placeholders[1].text_frame
    tf2.text = "Launch multilingual Corrective RAG assistant."

    prs.save(path)
    return path


@pytest.fixture
def sample_xlsx(tmp_path: Path) -> Path:
    path = tmp_path / "budget_2026.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "DepartmentBudget"
    ws.append(["RowID", "Department", "Region", "AllocatedUSD"])
    # 120 rows -> spans 3 windows of 50 rows each (1-50, 51-100, 101-120)
    for i in range(1, 121):
        ws.append([i, f"Dept_{i % 5}", "APAC" if i % 2 == 0 else "EMEA", i * 1000])
    wb.save(path)
    return path


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "financial_report.pdf"
    doc = pymupdf.open()

    p1 = doc.new_page()
    p1.insert_text((72, 72), "1. Financial Overview", fontsize=16)
    p1.insert_text(
        (72, 110),
        "Revenue grew 18% driven by enterprise subscriptions and strong retention.",
        fontsize=11,
    )

    p2 = doc.new_page()
    p2.insert_text((72, 72), "2. Risk Factors", fontsize=16)
    p2.insert_text(
        (72, 110),
        "Foreign exchange fluctuations may impact international operating margins.",
        fontsize=11,
    )

    doc.save(str(path))
    doc.close()
    return path


def test_parse_docx_and_cache(tmp_path: Path, sample_docx: Path):
    data_dir = tmp_path / "data"
    parser = DocumentParserService(data_dir=data_dir)
    res1 = parser.parse(sample_docx)

    assert res1.file_type == "docx"
    assert not res1.from_cache
    assert res1.page_count is None
    assert "System Architecture" in res1.markdown
    assert "Storage Layer" in res1.markdown
    assert res1.parsed_path.is_file()
    assert res1.markdown_path.is_file()
    assert res1.docling_dict is not None

    # Second call loads from cache without re-running Docling
    res2 = parser.parse(sample_docx)
    assert res2.from_cache
    assert res2.markdown == res1.markdown


def test_parse_pptx_preserves_slides_and_bullets(tmp_path: Path, sample_pptx: Path):
    data_dir = tmp_path / "data"
    parser = DocumentParserService(data_dir=data_dir)
    res = parser.parse(sample_pptx)

    assert res.file_type == "pptx"
    assert "Q3 Executive Summary" in res.markdown
    assert "Operating margin expanded by 3.2%" in res.markdown
    assert "Q4 Strategic Roadmap" in res.markdown

    chunker = HierarchicalChunker(child_max_tokens=128, parent_max_tokens=512)
    chunks = chunker.chunk(res)
    assert len(chunks) >= 2
    # PPTX chunks carry slide numbers in page_numbers
    all_pages = {p for c in chunks for p in c.page_numbers}
    assert 1 in all_pages
    assert 2 in all_pages


def test_parse_xlsx_windowed_headers_and_schema(tmp_path: Path, sample_xlsx: Path):
    data_dir = tmp_path / "data"
    parser = DocumentParserService(data_dir=data_dir, excel_window_rows=50)
    res = parser.parse(sample_xlsx)

    assert res.file_type == "xlsx"
    assert "### Sheet Schema Summary" in res.markdown
    assert "**Columns (4)**: RowID, Department, Region, AllocatedUSD" in res.markdown
    assert "**Total Data Rows**: 120" in res.markdown
    assert "### DepartmentBudget (Rows 1-50)" in res.markdown
    assert "### DepartmentBudget (Rows 51-100)" in res.markdown
    assert "### DepartmentBudget (Rows 101-120)" in res.markdown
    # Table header row is repeated in each window
    assert res.markdown.count("| RowID | Department | Region | AllocatedUSD |") >= 3

    chunker = HierarchicalChunker(child_max_tokens=200, parent_max_tokens=800)
    chunks = chunker.chunk(res)
    assert len(chunks) >= 3
    # XLSX chunks must have empty list for page_numbers
    for c in chunks:
        assert c.page_numbers == []
        assert c.contextualized_text.startswith("budget_2026.xlsx > ")
    table_chunks = [c for c in chunks if "Rows " in c.heading_path]
    assert len(table_chunks) >= 3
    for tc in table_chunks:
        assert "| RowID | Department | Region | AllocatedUSD |" in tc.child_text


def test_parse_pdf_pymupdf4llm_fallback_and_page_numbers(tmp_path: Path, sample_pdf: Path):
    data_dir = tmp_path / "data"
    parser = DocumentParserService(data_dir=data_dir, prefer_pymupdf4llm=True)
    res = parser.parse(sample_pdf)

    assert res.file_type == "pdf"
    assert res.page_count == 2
    assert "Financial Overview" in res.markdown
    assert "Risk Factors" in res.markdown

    chunker = HierarchicalChunker(child_max_tokens=128, parent_max_tokens=512)
    chunks = chunker.chunk(res)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c.page_numbers) >= 1
        assert all(isinstance(p, int) and p >= 1 for p in c.page_numbers)


def test_parent_child_hierarchy_and_breadcrumb_injection(tmp_path: Path, sample_docx: Path):
    data_dir = tmp_path / "data"
    parser = DocumentParserService(data_dir=data_dir)
    parse_res = parser.parse(sample_docx)

    # Force small child_max_tokens so multiple children group under one parent window
    chunker = HierarchicalChunker(child_max_tokens=12, parent_max_tokens=256)
    chunks = chunker.chunk(parse_res)

    assert len(chunks) >= 2
    for c in chunks:
        expected_prefix = format_contextualized_text(sample_docx.name, c.heading_path, c.child_text)
        assert c.contextualized_text == expected_prefix
        assert c.page_numbers == []  # DOCX has no page numbers
        assert c.child_text in c.parent_content
        assert c.token_count > 0


async def test_orchestrator_end_to_end_and_idempotent_hash_skip(
    tmp_path: Path,
    db_session_factory,
    memory_qdrant: AsyncQdrantClient,
    sample_docx: Path,
    sample_xlsx: Path,
):
    data_dir = tmp_path / "data"
    stub_embedder = DeterministicStubEmbedder(dimensions=STUB_DIM)
    stub_sparse = DeterministicBM25Encoder()

    # Verify process pool (spawn) works end-to-end on Windows
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
        doc1 = await orchestrator.ingest_file(sample_docx)
        assert doc1.status == "ready"
        assert doc1.chunk_count > 0
        assert doc1.sha256 == compute_sha256(sample_docx)

        # Verify SQLite DocumentChunk and IngestionJob records
        async with db_session_factory() as session:
            chunks_db = (
                await session.execute(select(DocumentChunk).where(DocumentChunk.doc_id == doc1.id))
            ).scalars().all()
            assert len(chunks_db) == doc1.chunk_count

            jobs_db = (
                await session.execute(select(IngestionJob).where(IngestionJob.doc_id == doc1.id))
            ).scalars().all()
            assert len(jobs_db) == 1
            assert jobs_db[0].status == "ready"
            assert jobs_db[0].duration_ms is not None and jobs_db[0].duration_ms > 0
            timings = json.loads(jobs_db[0].stage_timings or "{}")
            assert {"parsing_ms", "chunking_ms", "indexing_ms", "total_ms"} <= set(timings.keys())

        # Verify Qdrant points and payload schema
        scroll_pts, _ = await memory_qdrant.scroll(
            collection_name=TEST_COLLECTION,
            limit=50,
            with_payload=True,
            with_vectors=True,
        )
        assert len(scroll_pts) == doc1.chunk_count
        pt = scroll_pts[0]
        assert pt.payload is not None
        required_keys = {
            "doc_id",
            "chunk_id",
            "source_filename",
            "file_type",
            "heading_path",
            "page_numbers",
            "token_count",
            "child_text",
            "parent_content",
        }
        assert required_keys <= set(pt.payload.keys())
        assert pt.payload["doc_id"] == doc1.id
        assert pt.payload["source_filename"] == sample_docx.name
        assert pt.payload["file_type"] == "docx"
        assert pt.payload["page_numbers"] == []
        assert len(pt.vector[DENSE]) == STUB_DIM
        assert len(pt.vector[SPARSE].indices) > 0

        # Re-submitting identical file hash skips re-parsing/re-indexing
        doc1_repeat = await orchestrator.ingest_file(sample_docx)
        assert doc1_repeat.id == doc1.id
        assert doc1_repeat.status == "ready"

        async with db_session_factory() as session:
            jobs_after = (
                await session.execute(select(IngestionJob).where(IngestionJob.doc_id == doc1.id))
            ).scalars().all()
            # No second job created on idempotent skip
            assert len(jobs_after) == 1

        # Also ingest XLSX
        doc_xlsx = await orchestrator.ingest_file(sample_xlsx)
        assert doc_xlsx.status == "ready"
        assert doc_xlsx.chunk_count >= 3


def test_cli_collect_and_batch_run(tmp_path: Path, sample_docx: Path, sample_pptx: Path, monkeypatch):
    folder = tmp_path / "batch_docs"
    folder.mkdir()
    p1 = folder / sample_docx.name
    p2 = folder / sample_pptx.name
    p1.write_bytes(sample_docx.read_bytes())
    p2.write_bytes(sample_pptx.read_bytes())

    found = collect_target_files(folder)
    assert len(found) == 2

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "cli_data"))
    monkeypatch.setenv("DB_DIR", str(tmp_path / "cli_db"))
    from src.config import get_settings

    get_settings.cache_clear()
    exit_code = cli_main([str(folder), "--stub-embedder", "--in-memory-qdrant", "--no-pool"])
    assert exit_code == 0
    get_settings.cache_clear()


@pytest.mark.live
async def test_live_bge_m3_and_fastembed_indexing(
    tmp_path: Path,
    db_session_factory,
    memory_qdrant: AsyncQdrantClient,
    sample_docx: Path,
):
    """Live test verifying real BAAI/bge-m3 (1024-dim) + FastEmbed Qdrant/bm25 indexing."""
    data_dir = tmp_path / "data"
    parser = DocumentParserService(data_dir=data_dir)
    parsed = parser.parse(sample_docx)
    chunker = HierarchicalChunker()
    chunks = chunker.chunk(parsed)

    embedder = LocalBgeM3Embedder()
    indexer = HybridIndexer(
        qdrant=memory_qdrant,
        embedder=embedder,
        collection="live_bge_collection",
    )
    async with db_session_factory() as session:
        doc = Document(filename=sample_docx.name, file_type="docx", sha256=parsed.sha256, status="indexing")
        session.add(doc)
        await session.commit()
        count = await indexer.index_chunks(session, doc_id=doc.id, chunks=chunks)
        assert count == len(chunks)

    pts, _ = await memory_qdrant.scroll("live_bge_collection", limit=10, with_vectors=True)
    assert len(pts) == len(chunks)
    assert len(pts[0].vector[DENSE]) == 1024
