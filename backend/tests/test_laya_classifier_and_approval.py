"""Comprehensive test suite for Phase 2: Laya classifier, approval gate, contextual chunking, and entity extraction."""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from src.api.main import create_app
from src.config import Settings, get_settings
from src.db.models import (
    CrossDomainEntity,
    Document,
    DocumentChunk,
    DomainLexicon,
    DomainRegistry,
    EntityDomainMapping,
    IngestionJob,
    SubdomainRegistry,
)
from src.db.qdrant import init_qdrant_collection
from src.db.session import create_session_factory, init_db
from src.ingestion.chunker import (
    ChunkPayload,
    HierarchicalChunker,
    format_contextual_prefix,
    format_contextualized_text,
)
from src.ingestion.classifier import (
    DEFAULT_SUBDOMAIN_FOR_DOMAIN,
    DomainClassificationResult,
    DomainSpec,
    LayaDomainClassifier,
    RecommendedDomainProfile,
    SectionOverride,
    StubLayaClassifier,
    SubdomainSpec,
    get_classifier,
    get_taxonomy_catalog,
    invalidate_taxonomy_catalog,
    is_classifier_resident,
    refresh_taxonomy_catalog,
    set_classifier_force_stub,
    unload_classifier,
)
from src.ingestion.entity_extractor import EntityExtractor, sync_chunk_entities_to_db
from src.ingestion.indexer import DeterministicBM25Encoder, DeterministicStubEmbedder, HybridIndexer, release_cuda_cache
from src.ingestion.parser import ParseResult
from src.ingestion.worker import IngestionOrchestrator


@pytest.fixture(autouse=True)
def setup_stub_classifier():
    """Ensure deterministic stub classifier is used during tests."""
    set_classifier_force_stub(True)
    yield
    set_classifier_force_stub(False)
    unload_classifier()


@pytest.fixture
def in_memory_engine():
    return create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)


@pytest.fixture
async def seeded_session_factory(in_memory_engine):
    """Initialize DB (canonical domains & subdomains are automatically seeded by init_db)."""
    await init_db(in_memory_engine)
    session_factory = create_session_factory(in_memory_engine)
    return session_factory


# ==============================================================================
# 1. Classifier Unit Tests (Bilingual EN/VI, Subdomains, Authority & Overrides)
# ==============================================================================

@pytest.mark.asyncio
async def test_stub_classifier_bilingual_engineering():
    """Verify bilingual English and Vietnamese engineering text classification."""
    classifier = StubLayaClassifier()

    # English engineering standard
    en_text = (
        "High voltage transformer insulation breakdown voltage under IEEE C57.104 standard. "
        "Dielectric oil analysis shows dissolved gas concentration in kilovolts."
    )
    en_res = await classifier.classify_document(en_text, filename="transformer_manual.pdf")
    assert en_res.recommended.primary_domain == "engineering"
    assert en_res.recommended.primary_subdomain == "eng.electrical"
    assert en_res.recommended.epistemic_modality == "empirical"
    assert en_res.recommended.authority_tier == 1
    assert en_res.recommended.jurisdiction == "IEEE"

    # Vietnamese engineering standard
    vi_text = (
        "Tiêu chuẩn kỹ thuật TCVN 6306:2015 về máy biến áp lực. Đánh giá suy giảm điện áp, dòng điện định mức "
        "và độ bền cách điện của dầu biến áp theo quy chuẩn vận hành."
    )
    vi_res = await classifier.classify_document(vi_text, filename="tcvn_may_bien_ap.pdf")
    assert vi_res.recommended.primary_domain == "engineering"
    assert vi_res.recommended.primary_subdomain == "eng.electrical"
    assert vi_res.recommended.epistemic_modality == "empirical"
    assert vi_res.recommended.authority_tier == 1
    assert vi_res.recommended.jurisdiction == "VN-STANDARD"
    assert vi_res.recommended.effective_year == 2015


@pytest.mark.asyncio
async def test_stub_classifier_bilingual_economics():
    """Verify bilingual economics and accounting text classification."""
    classifier = StubLayaClassifier()

    # English financial filing
    en_text = (
        "Consolidated balance sheet under US GAAP ASC 360. Straight-line depreciation of property, plant, and equipment "
        "with revenue recognition and EBITDA valuation schedules."
    )
    en_res = await classifier.classify_document(en_text, filename="annual_report_10k.pdf")
    assert en_res.recommended.primary_domain == "economics"
    assert en_res.recommended.primary_subdomain == "econ.accounting"
    assert en_res.recommended.epistemic_modality == "evaluative"
    assert en_res.recommended.authority_tier == 1
    assert en_res.recommended.jurisdiction == "US-GAAP"

    # Vietnamese accounting standard
    vi_text = (
        "Báo cáo tài chính theo chuẩn mực kế toán Việt Nam VAS 03 và Thông tư 200/2014/TT-BTC. "
        "Bảng cân đối kế toán, khấu hao tài sản cố định và chi phí hoạt động sản xuất kinh doanh."
    )
    vi_res = await classifier.classify_document(vi_text, filename="bao_cao_tai_chinh.xlsx")
    assert vi_res.recommended.primary_domain == "economics"
    assert vi_res.recommended.primary_subdomain == "econ.accounting"
    assert vi_res.recommended.epistemic_modality == "evaluative"
    assert vi_res.recommended.authority_tier == 2
    assert vi_res.recommended.jurisdiction == "VN-MINISTRY"
    assert vi_res.recommended.effective_year == 2014


@pytest.mark.asyncio
async def test_stub_classifier_bilingual_law():
    """Verify statutory codes and administrative decrees (US and Vietnam)."""
    classifier = StubLayaClassifier()

    # US regulation
    us_text = (
        "Environmental Protection Agency mandates under 40 CFR 761 regarding disposal of PCB transformers. "
        "Statutory penalties and legal liability for civil non-compliance."
    )
    us_res = await classifier.classify_document(us_text, filename="epa_statute.pdf")
    assert us_res.recommended.primary_domain == "law"
    assert us_res.recommended.primary_subdomain == "law.regulation"
    assert us_res.recommended.epistemic_modality == "normative"
    assert us_res.recommended.authority_tier == 2
    assert us_res.recommended.jurisdiction == "US-FED"

    # Vietnamese law and decree
    vi_text = (
        "Căn cứ Luật số 55/2014/QH13 về Bảo vệ Môi trường và Nghị định 08/2022/NĐ-CP. "
        "Quy định trách nhiệm pháp lý, xử phạt vi phạm hành chính và bồi thường thiệt hại."
    )
    vi_res = await classifier.classify_document(vi_text, filename="nghi_dinh_08.pdf")
    assert vi_res.recommended.primary_domain == "law"
    assert vi_res.recommended.epistemic_modality == "normative"
    assert vi_res.recommended.authority_tier in (1, 2)
    assert vi_res.recommended.jurisdiction in ("VN-STATUTE", "VN-GOV")


@pytest.mark.asyncio
async def test_section_overrides_detection():
    """Verify H1/H2 section scanning detects overrides and sets is_hybrid = True."""
    classifier = StubLayaClassifier()
    doc_text = "Engineering specifications for power transmission and transformer windings."
    sections = [
        {"heading_path": "Chapter 1 > Technical Specs", "text": "Dielectric voltage ratings and transformer tolerances."},
        {"heading_path": "Chapter 5 > Appendix: Legal Liabilities and Compliance", "text": "Contractual liabilities, warranties, and indemnities pursuant to governing law."},
    ]

    res = await classifier.classify_document(doc_text, sections=sections, filename="manual.pdf")
    assert res.recommended.primary_domain == "engineering"
    assert res.recommended.is_hybrid is True
    assert len(res.recommended.section_overrides) >= 1

    override = res.recommended.section_overrides[0]
    assert "Legal Liabilities" in override.heading_path
    assert override.domain == "law"
    assert override.epistemic_modality == "normative"
    assert override.confidence >= 0.75


@pytest.mark.asyncio
async def test_classifier_unload_and_residency():
    """Verify classifier memory management and residency state."""
    laya_classifier = LayaDomainClassifier(model_name="convaiinnovations/laya-multilingual")
    assert laya_classifier.is_resident is False

    # Force unload
    unloaded = laya_classifier.unload()
    assert unloaded is False
    assert is_classifier_resident() is False


# ==============================================================================
# 2. Entity Extractor Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_entity_extractor_comprehensive(seeded_session_factory):
    """Verify extraction of equipment tags, standard codes, and statutory citations."""
    sample_text = (
        "Power transformer MBA-T1 and secondary unit T-101 are operated pursuant to IEEE C57.104 and IEC 60076. "
        "Compliance with national standard TCVN 6306:2015 and technical regulation QCVN 09:2015/BKHCN is mandatory. "
        "Capitalized under US GAAP ASC 360, IFRS 16, and Thông tư 200/2014/TT-BTC. "
        "Subject to 40 CFR 761, Luật số 55/2014/QH13, and Nghị định 08/2022/NĐ-CP Điều 12 Khoản 3."
    )
    extractor = EntityExtractor()
    entities = extractor.extract_entities(sample_text)

    # Check key expected entities
    assert any("MBA-T1" in e for e in entities)
    assert any("T-101" in e for e in entities)
    assert any("IEEE C57.104" in e for e in entities)
    assert any("IEC 60076" in e for e in entities)
    assert any("TCVN 6306:2015" in e for e in entities)
    assert any("ASC 360" in e for e in entities)
    assert any("IFRS 16" in e for e in entities)
    assert any("Thông tư 200/2014/TT-BTC" in e for e in entities)
    assert any("40 CFR 761" in e for e in entities)
    assert any("Luật số 55/2014/QH13" in e for e in entities)
    assert any("Nghị định 08/2022/NĐ-CP" in e for e in entities)
    assert any("Điều 12 Khoản 3" in e for e in entities)

    # Sync to SQLite
    async with seeded_session_factory() as session:
        # Get engineering domain ID
        res_d = await session.execute(select(DomainRegistry).where(DomainRegistry.slug == "engineering"))
        dom = res_d.scalars().first()
        assert dom is not None

        dummy_chunk_id = str(uuid.uuid4())
        await sync_chunk_entities_to_db(
            session,
            chunk_id=dummy_chunk_id,
            domain_id=dom.id,
            entities=entities,
        )
        await session.commit()

        # Verify entities persisted in cross_domain_entities & entity_domain_mappings
        res_entities = await session.execute(select(CrossDomainEntity))
        db_entities = res_entities.scalars().all()
        assert len(db_entities) >= 5

        res_maps = await session.execute(
            select(EntityDomainMapping).where(EntityDomainMapping.chunk_id == dummy_chunk_id)
        )
        db_maps = res_maps.scalars().all()
        assert len(db_maps) == len(db_entities)


# ==============================================================================
# 3. HierarchicalChunker Domain Context & Boundary Guards Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_chunker_domain_contextual_anchoring():
    """Verify clean child_text taking full budget and contextual prefix embedding."""
    chunker = HierarchicalChunker(child_max_tokens=256)

    md = (
        "# Engineering Manual\n\n"
        "## Transformer Dielectric Analysis\n\n"
        "Dielectric breakdown voltage under IEEE C57.104 must exceed 30 kV. "
        "The transformer MBA-T1 insulation degradation is monitored quarterly."
    )
    parse_result = ParseResult(
        sha256="testsha123",
        filename="transformer_manual.pdf",
        file_type="pdf",
        markdown=md,
        markdown_path=Path("/tmp/m.md"),
        parsed_path=Path("/tmp/p.json"),
    )

    profile = RecommendedDomainProfile(
        primary_domain="engineering",
        primary_subdomain="eng.electrical",
        epistemic_modality="empirical",
        authority_tier=1,
        jurisdiction="IEEE",
    )

    chunks = chunker.chunk(parse_result, domain_profile=profile)
    assert len(chunks) >= 1
    c = chunks[0]

    # Verify clean child text
    assert not c.child_text.startswith("[Domain:")
    assert "Dielectric breakdown voltage" in c.child_text

    # Verify contextual prefix
    assert c.contextual_prefix is not None
    assert "[Domain: Engineering" in c.contextual_prefix
    assert "Subdomain: eng.electrical" in c.contextual_prefix
    assert "Epistemic: Empirical" in c.contextual_prefix
    assert "Authority: Tier 1" in c.contextual_prefix

    # Verify contextualized text incorporates prefix + child_text
    assert c.contextualized_text == f"{c.contextual_prefix}\n{c.child_text}"
    assert c.domain_key == "engineering"
    assert c.subdomain_key == "eng.electrical"


@pytest.mark.asyncio
async def test_chunker_sticky_section_overrides_and_no_cross_merge():
    """Verify sticky section overrides and ensure assemble_parent_child never merges across domain_key."""
    chunker = HierarchicalChunker(child_max_tokens=256, parent_max_tokens=1024)

    md = (
        "# Main Document\n\n"
        "## Section 1: Mechanical Stress\n\n"
        "Mechanical fatigue and thermal degradation curve.\n\n"
        "## Section 2: Statutory Compliance\n\n"
        "All operators must adhere to statutory requirements pursuant to 40 CFR 761.\n"
    )
    parse_result = ParseResult(
        sha256="testsha456",
        filename="system_spec.pdf",
        file_type="pdf",
        markdown=md,
        markdown_path=Path("/tmp/m.md"),
        parsed_path=Path("/tmp/p.json"),
    )

    profile = RecommendedDomainProfile(
        primary_domain="engineering",
        primary_subdomain="eng.mechanical",
        epistemic_modality="empirical",
        authority_tier=2,
        is_hybrid=True,
        section_overrides=[
            SectionOverride(
                heading_path="Main Document > Section 2: Statutory Compliance",
                domain="law",
                subdomain="law.regulation",
                epistemic_modality="normative",
                confidence=0.88,
                authority_tier=1,
            )
        ],
    )

    chunks = chunker.chunk(parse_result, domain_profile=profile)
    assert len(chunks) >= 2

    c_eng = next(c for c in chunks if "Mechanical Stress" in c.heading_path)
    c_law = next(c for c in chunks if "Statutory Compliance" in c.heading_path)

    assert c_eng.domain_key == "engineering"
    assert c_eng.epistemic_modality == "empirical"

    assert c_law.domain_key == "law"
    assert c_law.epistemic_modality == "normative"

    # Invariant: parent_id must NEVER be shared between different domains!
    assert c_eng.parent_id != c_law.parent_id


@pytest.mark.asyncio
async def test_chunker_bilingual_law_boundary_guard():
    """Verify law statutory clauses and provisos are preserved as indivisible units."""
    chunker = HierarchicalChunker(child_max_tokens=150)

    # Vietnamese statutory clause with exception proviso
    md = (
        "# Quy chế quản lý\n\n"
        "## Điều 15. Tiêu chuẩn an toàn điện\n\n"
        "Khoản 1. Người vận hành phải có chứng chỉ hành nghề theo quy định.\n\n"
        "Khoản 2. Không được tự ý mở tủ điện khi đang mang tải cao thế, "
        "trừ trường hợp khẩn cấp có sự đồng ý bằng văn bản của kỹ sư trưởng.\n\n"
        "Khoản 3. Vi phạm quy định sẽ bị đình chỉ công tác 30 ngày."
    )
    parse_result = ParseResult(
        sha256="testlaw789",
        filename="quy_che.pdf",
        file_type="pdf",
        markdown=md,
        markdown_path=Path("/tmp/m.md"),
        parsed_path=Path("/tmp/p.json"),
    )
    profile = RecommendedDomainProfile(primary_domain="law", primary_subdomain="law.statute")

    chunks = chunker.chunk(parse_result, domain_profile=profile)
    assert len(chunks) >= 1

    # Ensure Khoản 2 and its proviso "trừ trường hợp" stay together in a single chunk
    k2_chunk = next(c for c in chunks if "Khoản 2" in c.child_text)
    assert "trừ trường hợp" in k2_chunk.child_text.lower()


@pytest.mark.asyncio
async def test_chunker_engineering_atomic_latex_equations():
    """Verify display LaTeX equations are kept intact without splitting."""
    chunker = HierarchicalChunker(child_max_tokens=120)

    md = (
        "# Electrical Theory\n\n"
        "## Dielectric Breakdown Formula\n\n"
        "The breakdown voltage is modeled through the electromagnetic equation:\n\n"
        "$$\n"
        "V_{break} = \\int_{0}^{d} E(x) \\cdot e^{-\\alpha x} dx + \\sum_{k=1}^{N} \\frac{Q_k}{4\\pi \\epsilon_0 r_k}\n"
        "$$\n\n"
        "where $E(x)$ is the electric field gradient across the gap."
    )
    parse_result = ParseResult(
        sha256="testmath101",
        filename="theory.pdf",
        file_type="pdf",
        markdown=md,
        markdown_path=Path("/tmp/m.md"),
        parsed_path=Path("/tmp/p.json"),
    )
    profile = RecommendedDomainProfile(primary_domain="engineering", primary_subdomain="eng.electrical")

    chunks = chunker.chunk(parse_result, domain_profile=profile)
    eq_chunk = next((c for c in chunks if "V_{break}" in c.child_text), None)
    assert eq_chunk is not None
    assert "$$" in eq_chunk.child_text
    assert "\\sum_{k=1}^{N}" in eq_chunk.child_text


@pytest.mark.asyncio
async def test_chunker_economics_period_banner_preservation():
    """Verify currency and period banners are retained across sliced financial tables."""
    chunker = HierarchicalChunker(child_max_tokens=60)

    # Table with financial banner
    md = (
        "# Financial Statements\n\n"
        "## Balance Sheet\n\n"
        "(Đơn vị tính: triệu đồng - Kỳ kế toán: Năm 2025)\n\n"
        "| Khoản mục | Năm 2024 | Năm 2025 |\n"
        "|---|---|---|\n"
        "| Tài sản cố định hữu hình | 120.000 | 145.000 |\n"
        "| Hao mòn lũy kế | (30.000) | (42.000) |\n"
        "| Tài sản cố định vô hình | 15.000 | 18.000 |\n"
        "| Phải thu khách hàng | 45.000 | 52.000 |\n"
    )
    parse_result = ParseResult(
        sha256="testecon202",
        filename="balance_sheet.docx",
        file_type="docx",
        markdown=md,
        markdown_path=Path("/tmp/m.md"),
        parsed_path=Path("/tmp/p.json"),
    )
    profile = RecommendedDomainProfile(primary_domain="economics", primary_subdomain="econ.accounting")

    chunks = chunker.chunk(parse_result, domain_profile=profile)
    assert len(chunks) >= 1
    # Check that banner notice is maintained
    for c in chunks:
        assert "Đơn vị tính: triệu đồng" in c.child_text or "Kỳ kế toán" in c.child_text


# ==============================================================================
# 4. Orchestrator Human-in-the-Loop Gate & 3-Mode Routing Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_orchestrator_review_all_mode_pauses_at_awaiting_approval(tmp_path, seeded_session_factory):
    """Verify review_all pauses at awaiting_approval (0.45) and releases worker pool slot."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy parsed file
    sha = "testsha_approval_gate"
    (data_dir / "parsed").mkdir(parents=True, exist_ok=True)
    (data_dir / "markdown").mkdir(parents=True, exist_ok=True)
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)

    test_file = data_dir / "raw" / f"{sha}.pdf"
    test_file.write_text("Dummy content for testing approval gate.")

    parsed_json = {
        "sha256": sha,
        "filename": "gate_test.pdf",
        "file_type": "pdf",
        "markdown": "# Technical Specification\n\nTransformer voltage under IEEE C57.104.",
        "page_count": 2,
        "extra_meta": {"engine": "pymupdf4llm"},
    }
    (data_dir / "parsed" / f"{sha}.json").write_text(json.dumps(parsed_json))
    (data_dir / "markdown" / f"{sha}.md").write_text(parsed_json["markdown"])

    qdrant = AsyncQdrantClient(location=":memory:")
    embedder = DeterministicStubEmbedder(dimensions=1024)
    sparse = DeterministicBM25Encoder()

    async with IngestionOrchestrator(
        session_factory=seeded_session_factory,
        qdrant=qdrant,
        embedder=embedder,
        sparse_encoder=sparse,
        data_dir=data_dir,
        use_process_pool=False,
    ) as orchestrator:
        # Create Document and Job in DB
        async with seeded_session_factory() as session:
            doc = Document(
                filename="gate_test.pdf",
                file_type="pdf",
                sha256=sha,
                status="queued",
            )
            session.add(doc)
            await session.commit()
            await session.refresh(doc)
            doc_id = doc.id

        # Run process_to_classification with review_all
        classified_doc, is_auto = await orchestrator.process_to_classification(
            doc_id=doc_id,
            approval_mode="review_all",
        )

        assert is_auto is False
        assert classified_doc.status == "awaiting_approval"
        assert classified_doc.primary_domain_id is not None
        assert classified_doc.classification_result is not None

        # Verify worker pool slot was released (no starvation!)
        worker_state = orchestrator.get_worker_state()
        assert worker_state["active_worker_jobs"] == 0

        # Now resume via complete_approved_ingestion
        completed_doc = await orchestrator.complete_approved_ingestion(doc_id=doc_id)
        assert completed_doc.status == "ready"
        assert completed_doc.chunk_count >= 1

    await qdrant.close()


@pytest.mark.asyncio
async def test_orchestrator_auto_all_mode(tmp_path, seeded_session_factory):
    """Verify auto_all mode proceeds straight from queued to ready."""
    data_dir = tmp_path / "data"
    sha = "testsha_auto_all"
    (data_dir / "parsed").mkdir(parents=True, exist_ok=True)
    (data_dir / "markdown").mkdir(parents=True, exist_ok=True)
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)

    parsed_json = {
        "sha256": sha,
        "filename": "auto_all.pdf",
        "file_type": "pdf",
        "markdown": "# Balance Sheet\n\nDepreciation under US GAAP ASC 360.",
        "page_count": 1,
        "extra_meta": {"engine": "pymupdf4llm"},
    }
    (data_dir / "parsed" / f"{sha}.json").write_text(json.dumps(parsed_json))
    (data_dir / "markdown" / f"{sha}.md").write_text(parsed_json["markdown"])

    qdrant = AsyncQdrantClient(location=":memory:")
    embedder = DeterministicStubEmbedder(dimensions=1024)
    sparse = DeterministicBM25Encoder()

    async with IngestionOrchestrator(
        session_factory=seeded_session_factory,
        qdrant=qdrant,
        embedder=embedder,
        sparse_encoder=sparse,
        data_dir=data_dir,
        use_process_pool=False,
    ) as orchestrator:
        async with seeded_session_factory() as session:
            doc = Document(filename="auto_all.pdf", file_type="pdf", sha256=sha, status="queued")
            session.add(doc)
            await session.commit()
            await session.refresh(doc)
            doc_id = doc.id

        # Backward-compatible process_document wrapper with default auto_all
        ready_doc = await orchestrator.process_document(doc_id=doc_id, approval_mode="auto_all")
        assert ready_doc.status == "ready"
        assert ready_doc.chunk_count >= 1
        assert ready_doc.primary_domain_id is not None

    await qdrant.close()


# ==============================================================================
# 5. API Routes Tests (approve-classification, approve-batch, metadata, retry)
# ==============================================================================

@pytest.mark.asyncio
async def test_api_routes_approve_and_metadata(tmp_path, seeded_session_factory):
    """Test POST /approve-classification, POST /approve-batch, and PATCH /metadata endpoints."""
    data_dir = tmp_path / "data"
    (data_dir / "parsed").mkdir(parents=True, exist_ok=True)
    (data_dir / "markdown").mkdir(parents=True, exist_ok=True)
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)

    sha = "a" * 64
    raw_dest = data_dir / "raw" / f"{sha}.pdf"
    raw_dest.write_text("Dummy raw content for testing api approval.")
    parsed_json = {
        "sha256": sha,
        "filename": "api_approval.pdf",
        "file_type": "pdf",
        "markdown": "# Environmental Act\n\nStatutory mandates pursuant to 40 CFR 761.",
        "page_count": 1,
        "extra_meta": {"engine": "pymupdf4llm"},
    }
    (data_dir / "parsed" / f"{sha}.json").write_text(json.dumps(parsed_json))
    (data_dir / "markdown" / f"{sha}.md").write_text(parsed_json["markdown"])

    # Create Document in awaiting_approval state
    async with seeded_session_factory() as session:
        doc = Document(
            filename="api_approval.pdf",
            file_type="pdf",
            sha256=sha,
            status="awaiting_approval",
            primary_domain_id="law",
            authority_tier=2,
            classification_result='{"primary_domain": "law", "primary_subdomain": "law.regulation", "epistemic_modality": "normative", "confidence": 0.88}',
        )
        session.add(doc)
        await session.commit()
        await session.refresh(doc)
        doc_id = doc.id

        job = IngestionJob(doc_id=doc_id, status="awaiting_approval", stage="awaiting_approval", progress=0.45)
        session.add(job)
        await session.commit()

    qdrant = AsyncQdrantClient(location=":memory:")
    embedder = DeterministicStubEmbedder(dimensions=1024)
    sparse = DeterministicBM25Encoder()

    orchestrator = IngestionOrchestrator(
        session_factory=seeded_session_factory,
        qdrant=qdrant,
        embedder=embedder,
        sparse_encoder=sparse,
        data_dir=data_dir,
        use_process_pool=False,
    )

    app = create_app()
    test_settings = get_settings().model_copy(
        update={"storage": get_settings().storage.model_copy(update={"data_dir": str(data_dir)})}
    )
    app.state.settings = test_settings
    app.state.orchestrator = orchestrator
    app.state.session_factory = seeded_session_factory
    app.state.qdrant = qdrant

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Test POST /approve-classification with overrides
        approve_payload = {
            "authority_tier": 1,
            "jurisdiction": "US-FED",
        }
        res_approve = await client.post(
            f"/api/v1/documents/{doc_id}/approve-classification",
            json=approve_payload,
        )
        assert res_approve.status_code == 200
        data = res_approve.json()
        assert data["status"] == "ready"
        assert data["authority_tier"] == 1
        assert data["jurisdiction"] == "US-FED"
        assert data["chunk_count"] >= 1

        # 2. Test PATCH /metadata
        meta_payload = {
            "effective_year": 2026,
            "is_current": True,
        }
        res_meta = await client.patch(
            f"/api/v1/documents/{doc_id}/metadata",
            json=meta_payload,
        )
        assert res_meta.status_code == 200
        meta_data = res_meta.json()
        assert meta_data["effective_year"] == 2026
        assert meta_data["is_current"] is True

        # 3. Test retry guard exclusion when classifying
        async with seeded_session_factory() as session:
            test_doc = await session.get(Document, doc_id)
            test_doc.status = "classifying"
            session.add(test_doc)
            await session.commit()

        res_retry = await client.post(f"/api/v1/documents/{doc_id}/retry")
        assert res_retry.status_code == 409  # Conflict! Cannot retry during active classification!

    orchestrator.shutdown()
    await qdrant.close()


@pytest.mark.asyncio
async def test_system_telemetry_includes_classifier(tmp_path, seeded_session_factory):
    """Verify /api/v1/system/stats and /api/v1/system/memory expose classifier residency and telemetry."""
    data_dir = tmp_path
    qdrant = AsyncQdrantClient(location=":memory:")
    embedder = DeterministicStubEmbedder(dimensions=1024)
    sparse = DeterministicBM25Encoder()

    orchestrator = IngestionOrchestrator(
        session_factory=seeded_session_factory,
        qdrant=qdrant,
        embedder=embedder,
        sparse_encoder=sparse,
        data_dir=tmp_path,
        use_process_pool=False,
    )

    app = create_app()
    test_settings = get_settings().model_copy(
        update={"storage": get_settings().storage.model_copy(update={"data_dir": str(data_dir)})}
    )
    app.state.settings = test_settings
    app.state.orchestrator = orchestrator
    app.state.session_factory = seeded_session_factory
    app.state.qdrant = qdrant

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Check /api/v1/system/stats
        stats_res = await client.get("/api/v1/system/stats")
        assert stats_res.status_code == 200
        stats_data = stats_res.json()
        assert "classifier" in stats_data
        assert stats_data["classifier"]["model_name"] == "convaiinnovations/laya-multilingual"
        assert "resident" in stats_data["classifier"]
        assert stats_data["classifier"]["enable_llm_classification_fallback"] is True
        assert stats_data["classifier"]["high_confidence_threshold"] == 0.85

        # Check /api/v1/system/memory
        mem_res = await client.get("/api/v1/system/memory")
        assert mem_res.status_code == 200
        mem_data = mem_res.json()
        assert "classifier_resident" in mem_data
        assert "classifier_model_name" in mem_data

    orchestrator.shutdown()
    await qdrant.close()


# ==============================================================================
# 9. Phase 3 Dynamic Taxonomy, Chunker Presets & ReDoS Protection Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_custom_domain_classifier_and_chunking_strategy():
    """Verify custom domain classification and chunking strategy assignment in StubLayaClassifier."""
    invalidate_taxonomy_catalog()
    cat = get_taxonomy_catalog()
    cat.domains["biomedical"] = DomainSpec(
        slug="biomedical",
        name="Biomedical Engineering",
        epistemic_type="empirical",
        chunking_strategy="statutory_clauses",
        classifier_keywords="cardiac,prosthetic,catheter,pacemaker",
        subdomains={
            "bio.devices": SubdomainSpec(
                slug="bio.devices",
                name="Cardiovascular Devices",
                domain_slug="biomedical",
                standard_bodies="FDA, ISO 13485",
            )
        },
    )

    classifier = StubLayaClassifier()
    text = "Clinical trial report on cardiac catheter and implantable pacemaker biocompatibility standards."
    res = await classifier.classify_document(text, filename="pacemaker_trial.pdf")

    assert res.recommended.primary_domain == "biomedical"
    assert res.recommended.primary_subdomain == "bio.devices"
    assert res.recommended.epistemic_modality == "empirical"
    assert res.recommended.chunking_strategy == "statutory_clauses"


@pytest.mark.asyncio
async def test_custom_domain_blank_keywords_fallback():
    """Verify fallback tokenization from domain name and subdomains when classifier_keywords is blank."""
    invalidate_taxonomy_catalog()
    cat = get_taxonomy_catalog()
    cat.domains["astrophysics"] = DomainSpec(
        slug="astrophysics",
        name="Astrophysics & Cosmology",
        epistemic_type="empirical",
        chunking_strategy="academic_equations",
        classifier_keywords="",  # Blank keywords
        subdomains={
            "astro.galaxies": SubdomainSpec(
                slug="astro.galaxies",
                name="Galactic Evolution",
                domain_slug="astrophysics",
                standard_bodies="Hubble, JWST",
            )
        },
    )

    classifier = StubLayaClassifier()
    text = "Observational telemetry studying galactic evolution and redshift distributions from deep space surveys."
    res = await classifier.classify_document(text, filename="galaxy_survey.pdf")

    assert res.recommended.primary_domain == "astrophysics"
    assert res.recommended.primary_subdomain == "astro.galaxies"
    assert res.recommended.chunking_strategy == "academic_equations"


@pytest.mark.asyncio
async def test_redos_prevention_on_special_characters():
    """Verify re.escape() prevents ReDoS and regex crashes on user-supplied special characters."""
    invalidate_taxonomy_catalog()
    cat = get_taxonomy_catalog()
    cat.domains["math_special"] = DomainSpec(
        slug="math_special",
        name="Special Math & Computing",
        epistemic_type="empirical",
        chunking_strategy="standard_markdown",
        classifier_keywords="c++, (regex.*)+, [bracket], a+b*c, $100",
        subdomains={
            "math.cpp": SubdomainSpec(
                slug="math.cpp",
                name="C++ Tooling",
                domain_slug="math_special",
            )
        },
    )

    classifier = StubLayaClassifier()
    text = "Enterprise systems developed in c++ with complex algorithms and [bracket] syntax."
    res = await classifier.classify_document(text, filename="cpp_guide.txt")

    assert res.recommended.primary_domain == "math_special"
    assert res.recommended.primary_subdomain == "math.cpp"


@pytest.mark.asyncio
async def test_hierarchical_chunker_custom_chunking_strategy_dispatch():
    """Verify HierarchicalChunker dispatches custom chunking_strategy carried on profile & overrides."""
    chunker = HierarchicalChunker(child_max_tokens=60, parent_max_tokens=200)

    # 1. Custom domain profile with statutory_clauses strategy
    profile = RecommendedDomainProfile(
        primary_domain="custom_compliance",
        primary_subdomain="custom_compliance.general",
        chunking_strategy="statutory_clauses",
    )

    legal_text = (
        "## Title I: Compliance Provisions\n\n"
        "Điều 1. Phạm vi điều chỉnh\n"
        "Quy định này áp dụng đối với tất cả các tổ chức, cá nhân tham gia vào hoạt động vận hành kỹ thuật hệ thống Second Brain trên lãnh thổ Việt Nam và quốc tế.\n\n"
        "Điều 2. Đối tượng áp dụng\n"
        "Các cơ quan quản lý nhà nước, doanh nghiệp cung cấp dịch vụ phân tích dữ liệu đa ngành và chuyên gia tư vấn tuân thủ quy chuẩn kỹ thuật an toàn thông tin."
    )

    chunks = chunker.chunk(
        ParseResult(
            sha256="compliance_sha",
            filename="compliance.md",
            file_type="md",
            markdown=legal_text,
            markdown_path=Path("/tmp/comp.md"),
            parsed_path=Path("/tmp/comp.json"),
        ),
        domain_profile=profile,
    )
    assert len(chunks) >= 2
    assert any("Điều 1" in c.child_text for c in chunks)
    assert any("Điều 2" in c.child_text for c in chunks)

    # 2. Section override carrying financial_tables strategy
    override_profile = RecommendedDomainProfile(
        primary_domain="custom_compliance",
        primary_subdomain="custom_compliance.general",
        chunking_strategy="statutory_clauses",
        section_overrides=[
            SectionOverride(
                heading_path="Appendix > Financial Balances",
                domain="custom_finance",
                subdomain="custom_finance.general",
                epistemic_modality="evaluative",
                chunking_strategy="financial_tables",
            )
        ],
    )

    table_text = (
        "## Appendix > Financial Balances\n\n"
        "| Period | Revenue ($M) | Operating Expense ($M) | Net Margin ($M) | Depreciation Schedule | Notes |\n"
        "|---|---|---|---|---|---|\n"
        "| Q1 2024 | 120.5 | 85.2 | 35.3 | Straight line over 5 years | Confirmed by external audit |\n"
        "| Q2 2024 | 145.0 | 95.1 | 49.9 | Straight line over 5 years | Confirmed by external audit |\n"
        "| Q3 2024 | 160.2 | 102.4 | 57.8 | Straight line over 5 years | Confirmed by external audit |\n"
        "| Q4 2024 | 190.8 | 115.6 | 75.2 | Straight line over 5 years | Confirmed by external audit |\n"
    )

    tbl_chunks = chunker.chunk(
        ParseResult(
            sha256="balances_sha",
            filename="balances.md",
            file_type="md",
            markdown=table_text,
            markdown_path=Path("/tmp/bal.md"),
            parsed_path=Path("/tmp/bal.json"),
        ),
        domain_profile=override_profile,
    )
    assert len(tbl_chunks) >= 1
    assert all("|" in c.child_text for c in tbl_chunks)


@pytest.mark.asyncio
async def test_epistemic_partitioned_prompt_and_xml_context():
    """Verify _classify_with_llm separates SystemMessage and HumanMessage with XML <document_sample> tags."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from unittest.mock import patch

    captured_messages = []

    class MockChatModel:
        async def ainvoke(self, messages):
            captured_messages.extend(messages)
            return AIMessage(
                content=json.dumps({
                    "primary_domain": "engineering",
                    "primary_subdomain": "eng.technology",
                    "epistemic_modality": "empirical",
                    "confidence": 0.95,
                    "authority_tier": 1,
                    "jurisdiction": "IEEE",
                    "effective_year": 2024,
                    "reasoning": "Technical software architecture document",
                })
            )

    classifier = LayaDomainClassifier()

    with patch("src.ingestion.classifier.get_settings") as mock_settings, patch("src.config.get_settings", mock_settings):
        mock_settings.return_value.llm.provider = "openai"
        mock_settings.return_value.llm.model_name = "mock-gpt-4o"
        with patch("langchain_openai.ChatOpenAI", return_value=MockChatModel()):
            profile = await classifier._classify_with_llm(
                "Software architecture and distributed database replication.",
                filename="architecture.pdf",
            )

    assert profile is not None
    assert profile.primary_domain == "engineering"
    assert profile.chunking_strategy == "academic_equations"
    assert len(captured_messages) == 2

    sys_msg, human_msg = captured_messages
    assert isinstance(sys_msg, SystemMessage)
    assert isinstance(human_msg, HumanMessage)

    # Verify XML <document_sample> tag boundary isolation
    assert '<document_sample filename="architecture.pdf">' in human_msg.content
    assert "</document_sample>" in human_msg.content

    # Verify dynamic taxonomy catalog domains in system prompt
    assert "Allowed domains:" in sys_msg.content
    assert "'engineering'" in sys_msg.content
    assert "'economics'" in sys_msg.content
    assert "'law'" in sys_msg.content
