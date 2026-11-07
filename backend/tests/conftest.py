"""Shared pytest fixtures and deterministic multi-format document generators (PDF, DOCX, XLSX, PPTX)."""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import docx
import openpyxl
import pptx
from pptx.util import Inches
import pymupdf
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from qdrant_client import AsyncQdrantClient

from src.api.main import create_app
from src.config import get_settings
from src.db.session import create_engine, create_session_factory, init_db
from src.ingestion.indexer import (
    DeterministicBM25Encoder,
    DeterministicStubEmbedder,
    set_embedder,
    set_sparse_encoder,
)
from src.rag.graph import close_checkpointer
from src.rag.llm import set_chat_model
from src.rag.reranker import FlashRankReranker, set_reranker
from src.rag.retriever import set_active_qdrant_client

E2E_STUB_DIMENSIONS = 1024


class DeterministicStubRanker:
    """Fast, offline deterministic cross-encoder stub for FlashRankReranker."""

    def rerank(self, request: Any) -> list[dict[str, Any]]:
        query_terms = {w.lower() for w in (request.query or "").split() if len(w) > 2}
        results: list[dict[str, Any]] = []
        for idx, passage in enumerate(request.passages):
            text_lower = (passage.get("text") or "").lower()
            overlap = sum(1 for term in query_terms if term in text_lower)
            score = min(0.99, 0.50 + 0.08 * overlap - idx * 0.001)
            results.append(
                {
                    "id": passage["id"],
                    "text": passage["text"],
                    "meta": passage["meta"],
                    "score": round(score, 4),
                }
            )
        return results


def generate_sample_pdf(target_path: Path) -> Path:
    """Create a deterministic 2-page PDF with headings, bullet lists, and a structured table."""
    doc = pymupdf.open()

    # Page 1: Executive Summary + Bullet List + Table
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text((54, 60), "# 1. Q3 2025 Financial Overview", fontsize=16)
    p1.insert_text(
        (54, 90),
        "In Q3 2025, the company delivered strong operational efficiency across all regions.",
        fontsize=11,
    )
    p1.insert_text((54, 118), "## Key Highlights", fontsize=13)
    p1.insert_text(
        (64, 142),
        "- Operating margin expanded by 3.2% year-over-year due to supply chain automation.",
        fontsize=10,
    )
    p1.insert_text(
        (64, 160),
        "- Gross profit reached $142M in Q3 2025 with 96% enterprise net retention.",
        fontsize=10,
    )
    p1.insert_text(
        (64, 178),
        "- R&D cloud infrastructure spend decreased by 14% after CPU vector indexing.",
        fontsize=10,
    )

    # Draw grid table + pipe rows so both Docling TableFormer and pymupdf4llm emit Markdown table pipes
    table_top = 210
    row_height = 24
    col_widths = [180, 120, 120]
    rows_data = [
        ["Metric", "Q3 2025", "YoY Change"],
        ["Operating Margin", "24.8%", "+3.2%"],
        ["Gross Profit", "$142M", "+18.0%"],
    ]
    for r_idx, row in enumerate(rows_data):
        y0 = table_top + r_idx * row_height
        y1 = y0 + row_height
        x_cursor = 54
        for c_idx, cell_value in enumerate(row):
            w = col_widths[c_idx]
            rect = pymupdf.Rect(x_cursor, y0, x_cursor + w, y1)
            p1.draw_rect(rect, color=(0, 0, 0), width=0.75)
            p1.insert_text((x_cursor + 6, y0 + 16), cell_value, fontsize=10)
            x_cursor += w

    p1.insert_text(
        (54, 310),
        "| Metric | Q3 2025 | YoY Change |\n| --- | --- | --- |\n| Operating Margin | 24.8% | +3.2% |\n| Gross Profit | $142M | +18.0% |",
        fontsize=9,
    )

    # Page 2: Risk Factors & Outlook
    p2 = doc.new_page(width=612, height=792)
    p2.insert_text((54, 60), "# 2. Strategic Outlook & Risk Factors", fontsize=16)
    p2.insert_text(
        (54, 92),
        "- Foreign exchange fluctuations in APAC and EMEA are hedged through Q2 2026.",
        fontsize=10,
    )
    p2.insert_text(
        (54, 112),
        "- Full-year 2025 revenue guidance is raised to $580M.",
        fontsize=10,
    )

    doc.save(str(target_path))
    doc.close()
    return target_path


def generate_sample_docx(target_path: Path) -> Path:
    """Create a deterministic DOCX with nested headings, bullet lists, and a multi-row table."""
    document = docx.Document()
    document.add_heading("Second Brain Architecture Specification", level=1)
    document.add_paragraph(
        "The Second Brain platform combines Docling v2 document parsing, Qdrant hybrid vector search, "
        "and a 7-node LangGraph Corrective RAG conversational engine."
    )

    document.add_heading("Core Capabilities", level=2)
    document.add_paragraph(
        "Idempotent SHA-256 document ingestion with AST and Markdown disk caching.",
        style="List Bullet",
    )
    document.add_paragraph(
        "Hybrid retrieval combining 1024-dim BAAI/bge-m3 dense vectors and FastEmbed BM25 sparse vectors.",
        style="List Bullet",
    )
    document.add_paragraph(
        "Multilingual cross-encoder reranking via FlashRank ms-marco-MultiBERT-L-12.",
        style="List Bullet",
    )

    document.add_heading("Storage Tier Matrix", level=2)
    table = document.add_table(rows=4, cols=3)
    headers = ["Tier", "Technology", "Responsibility"]
    for idx, text in enumerate(headers):
        table.cell(0, idx).text = text

    matrix_rows = [
        ("Vector Store", "Qdrant Hybrid", "1024-dim dense + BM25 sparse RRF search"),
        ("Relational DB", "SQLite WAL", "Document lifecycle, chunks, and chat threads"),
        ("Checkpointer", "AsyncSqliteSaver", "Multi-turn LangGraph conversation memory"),
    ]
    for r_idx, row_vals in enumerate(matrix_rows, start=1):
        for c_idx, val in enumerate(row_vals):
            table.cell(r_idx, c_idx).text = val

    document.save(str(target_path))
    return target_path


def generate_sample_xlsx(target_path: Path) -> Path:
    """Create a deterministic XLSX workbook with 65 rows (spanning two 50-row windows)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "RegionalBudget"
    ws.append(["RowID", "Department", "Region", "Quarter", "AllocatedUSD"])
    for i in range(1, 66):
        dept = "Engineering" if i % 2 == 0 else "Operations"
        region = "APAC" if i % 3 == 0 else "EMEA"
        ws.append([i, dept, region, "Q3_2025", i * 2500])
    wb.save(str(target_path))
    return target_path


def generate_sample_pptx(target_path: Path) -> Path:
    """Create a deterministic 2-slide PPTX presentation with titles, bullet lists, and a slide table."""
    prs = pptx.Presentation()
    bullet_layout = prs.slide_layouts[1]

    # Slide 1: Title + Bullets + Table
    s1 = prs.slides.add_slide(bullet_layout)
    s1.shapes.title.text = "Q3 Executive Briefing"
    tf1 = s1.placeholders[1].text_frame
    tf1.text = "- Operating margin expanded by 3.2% year-over-year."
    p2 = tf1.add_paragraph()
    p2.text = "- Gross profit reached $142M driven by enterprise renewals."
    p2.level = 1

    table_shape = s1.shapes.add_table(3, 3, Inches(1.0), Inches(3.8), Inches(7.5), Inches(1.8))
    tbl = table_shape.table
    slide_rows = [
        ("KPI", "Q3 Actual", "Status"),
        ("Operating Margin", "24.8%", "Exceeded Target"),
        ("Net Retention", "96.0%", "On Track"),
    ]
    for r_idx, row_vals in enumerate(slide_rows):
        for c_idx, val in enumerate(row_vals):
            tbl.cell(r_idx, c_idx).text = val

    # Slide 2: Roadmap
    s2 = prs.slides.add_slide(bullet_layout)
    s2.shapes.title.text = "Q4 Product Roadmap"
    tf2 = s2.placeholders[1].text_frame
    tf2.text = "- Ship multi-stage Docker Compose deployment with named volumes."
    p2_sub = tf2.add_paragraph()
    p2_sub.text = "- Enable per-request cloud LLM provider switching."
    p2_sub.level = 1

    prs.save(str(target_path))
    return target_path


@pytest.fixture
def e2e_sample_pdf(tmp_path: Path) -> Path:
    return generate_sample_pdf(tmp_path / "q3_financial_report.pdf")


@pytest.fixture
def e2e_sample_docx(tmp_path: Path) -> Path:
    return generate_sample_docx(tmp_path / "architecture_spec.docx")


@pytest.fixture
def e2e_sample_xlsx(tmp_path: Path) -> Path:
    return generate_sample_xlsx(tmp_path / "regional_budget_2025.xlsx")


@pytest.fixture
def e2e_sample_pptx(tmp_path: Path) -> Path:
    return generate_sample_pptx(tmp_path / "q3_executive_briefing.pptx")


@pytest.fixture
def e2e_multi_format_files(
    e2e_sample_pdf: Path,
    e2e_sample_docx: Path,
    e2e_sample_xlsx: Path,
    e2e_sample_pptx: Path,
) -> dict[str, Path]:
    """Return a mapping of `{'pdf': Path, 'docx': Path, 'xlsx': Path, 'pptx': Path}`."""
    return {
        "pdf": e2e_sample_pdf,
        "docx": e2e_sample_docx,
        "xlsx": e2e_sample_xlsx,
        "pptx": e2e_sample_pptx,
    }


@pytest.fixture
def stub_embedder() -> DeterministicStubEmbedder:
    """Return a 1024-dim deterministic stub embedder matching BAAI/bge-m3 vector dimensions."""
    return DeterministicStubEmbedder(dimensions=E2E_STUB_DIMENSIONS)


@pytest.fixture
def stub_sparse_encoder() -> DeterministicBM25Encoder:
    """Return a deterministic offline BM25 sparse vector encoder."""
    return DeterministicBM25Encoder()


@pytest.fixture
def stub_reranker() -> FlashRankReranker:
    """Return a FlashRankReranker backed by a deterministic offline ranker."""
    return FlashRankReranker(top_n=6, ranker=DeterministicStubRanker())


@pytest.fixture
async def in_memory_qdrant() -> AsyncIterator[AsyncQdrantClient]:
    """Yield an isolated in-memory AsyncQdrantClient."""
    client = AsyncQdrantClient(location=":memory:")
    yield client
    await client.close()


@pytest.fixture
def fake_chat_model_factory() -> Callable[[list[str]], GenericFakeChatModel]:
    """Factory that builds a GenericFakeChatModel from a list of scripted response strings."""

    def _build(responses: list[str]) -> GenericFakeChatModel:
        return GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in responses]))

    return _build


@pytest.fixture
async def isolated_sqlite_factory(tmp_path: Path):
    """Yield an isolated async_sessionmaker backed by a temporary SQLite WAL database."""
    db_path = tmp_path / "db" / "second_brain.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    await init_db(engine)
    factory = create_session_factory(engine)
    yield factory
    await engine.dispose()


@pytest.fixture
async def e2e_api_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_embedder: DeterministicStubEmbedder,
    stub_sparse_encoder: DeterministicBM25Encoder,
    stub_reranker: FlashRankReranker,
) -> AsyncIterator[tuple[AsyncClient, Path, Any]]:
    """Spin up the full FastAPI app inside LifespanManager with 1024-dim stub vectors and in-memory Qdrant."""
    data_dir = tmp_path / "data"
    db_dir = tmp_path / "data" / "db"
    dist_dir = tmp_path / "frontend_dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "index.html").write_text(
        "<!doctype html><html><head><title>Second Brain</title></head>"
        "<body><div id='root'>Second Brain Workspace</div></body></html>",
        encoding="utf-8",
    )

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("DB_DIR", str(db_dir))
    monkeypatch.setenv("FRONTEND_DIST_DIR", str(dist_dir))
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("INGESTION_USE_PROCESS_POOL", "false")
    monkeypatch.setenv("PREFER_PYMUPDF4LLM", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-e2e-offline-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-e2e-offline-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-e2e-offline-key")

    get_settings.cache_clear()
    set_embedder(stub_embedder)
    set_sparse_encoder(stub_sparse_encoder)
    set_reranker(stub_reranker)
    set_chat_model(None)
    set_active_qdrant_client(None)

    app = create_app()
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client, data_dir, app

    set_chat_model(None)
    set_embedder(None)
    set_sparse_encoder(None)
    set_reranker(None)
    set_active_qdrant_client(None)
    await close_checkpointer()
    get_settings.cache_clear()
