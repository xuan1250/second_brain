"""Unit and integration tests for Phase 2 engineering chunker, entity stemmer, and Qdrant record_type indexing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymupdf as fitz
import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select

from scripts.generate_engineering_fixtures import generate_all_fixtures
from src.db.models import Document, DocumentChunk
from src.db.session import create_engine, create_session_factory, init_db
from src.ingestion import chunker as chunker_mod
from src.ingestion.classifier import RecommendedDomainProfile
from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.engineering_extractor import extract_engineering_pdf
from src.ingestion.entity_extractor import EntityExtractor, expand_engineering_tag_stems
from src.ingestion.indexer import DeterministicBM25Encoder, DeterministicStubEmbedder, HybridIndexer
from src.ingestion.parser import ParseResult, compute_sha256

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "engineering"


@pytest.fixture(scope="module")
def engineering_fixtures() -> dict[str, Path]:
    """Ensure the 4 compact engineering fixtures exist and return their paths."""
    expected = {
        "cabinet_wiring": FIXTURE_DIR / "cabinet_wiring_p2.pdf",
        "ce_chart": FIXTURE_DIR / "process_ce_chart_p3_p4.pdf",
        "esds_logic": FIXTURE_DIR / "esds_logic_p8_p51.pdf",
        "lv_swgr": FIXTURE_DIR / "lv_swgr_logic_p3.pdf",
    }
    if not all(p.is_file() for p in expected.values()):
        generate_all_fixtures(out_dir=FIXTURE_DIR)
    return expected


@pytest.fixture
async def db_session_factory(tmp_path: Path):
    db_file = tmp_path / "db" / "second_brain_eng.db"
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


def _make_engineering_parse_result(
    pdf_path: Path,
    *,
    json_roundtrip: bool = False,
    docling_dict: dict[str, Any] | None = None,
) -> ParseResult:
    extracted = extract_engineering_pdf(pdf_path)
    assert extracted is not None, f"Expected engineering profile for {pdf_path.name}"
    extra_meta: dict[str, Any] = {
        "engine": "engineering_cad",
        "has_engineering_records": True,
        "engineering_profile": extracted["engineering_profile"],
    }
    if json_roundtrip:
        extra_meta = json.loads(json.dumps(extra_meta))

    return ParseResult(
        sha256=compute_sha256(pdf_path),
        filename=pdf_path.name,
        file_type="pdf",
        markdown=extracted["markdown"],
        markdown_path=pdf_path.with_suffix(".md"),
        parsed_path=pdf_path.with_suffix(".json"),
        docling_dict=docling_dict,
        page_count=extracted["page_count"],
        extra_meta=extra_meta,
    )


def _eng_profile() -> RecommendedDomainProfile:
    return RecommendedDomainProfile(
        primary_domain="engineering",
        primary_subdomain="eng.instrumentation",
        epistemic_modality="empirical",
        authority_tier=1,
        jurisdiction="ISA",
    )


# ==============================================================================
# 1. Entity Extractor & Stemming Tests
# ==============================================================================


def test_entity_extractor_engineering_stemming_rules():
    """Verify ISA loop cross-linking, _R12 suffix stripping, voting splits, and query extraction."""
    extractor = EntityExtractor()

    # TT-10205A and TI10205A BOTH emit LOOP-10205A
    tt_entities = extractor.extract_entities("Field transmitter TT-10205A")
    ti_entities = extractor.extract_entities("DCS indicator TI10205A")
    assert "LOOP-10205A" in tt_entities
    assert "LOOP-10205A" in ti_entities

    # HZ-10001 and HZ10001_R12 BOTH emit HZ10001 and HZ-10001
    hz_ce_entities = extractor.extract_entities("Cause HZ-10001 ESD-1")
    hz_logic_entities = extractor.extract_entities("From: HZ10001_R12")
    assert {"HZ10001", "HZ-10001"}.issubset(set(hz_ce_entities))
    assert {"HZ10001", "HZ-10001"}.issubset(set(hz_logic_entities))

    # LAHH-80101A/B/C expands to individual voting channels + compact stem
    lahh_entities = extractor.extract_entities("Cause LAHH-80101A/B/C 2 out of 3")
    for expected_stem in ("LAHH-80101A", "LAHH-80101B", "LAHH-80101C", "LAHH80101"):
        assert expected_stem in lahh_entities

    # Query tag extraction handles lowercase tags without matching English prose words
    query_stems = extractor.extract_query_tags_and_stems(
        "Which effects trigger on lahh-80101a and tt-10205a?"
    )
    for expected_stem in ("LAHH-80101A", "TT-10205A", "LOOP-10205A"):
        assert expected_stem in query_stems
    for prose_word in ("Which", "WHICH", "effects", "EFFECTS", "trigger", "TRIGGER", "and", "AND"):
        assert prose_word not in query_stems


def test_entity_extractor_non_empty_across_all_four_fixtures(engineering_fixtures: dict[str, Path]):
    """Verify EntityExtractor extracts non-empty entities on every fixture's extracted content."""
    extractor = EntityExtractor()

    for key, pdf_path in engineering_fixtures.items():
        extracted = extract_engineering_pdf(pdf_path)
        assert extracted is not None
        records = extracted["engineering_profile"]["records"]
        assert len(records) >= 1, f"Expected records in {key}"

        combined_text = "\n".join(r["summary_text"] for r in records)
        entities = extractor.extract_entities(combined_text, domain_key="engineering")
        assert entities != [], f"Expected non-empty entities for {key}"

        record_tags = [tag for r in records for tag in r.get("tags", [])]
        expanded = sorted(set(entities) | set(expand_engineering_tag_stems(record_tags)))
        assert len(expanded) > 0


# ==============================================================================
# 2. HierarchicalChunker Engineering Bypass & Per-Mode Tests
# ==============================================================================


def test_chunker_bypasses_docling_ast_when_engineering_records_present(
    monkeypatch: pytest.MonkeyPatch,
    engineering_fixtures: dict[str, Path],
):
    """Verify _chunk_docling_ast and HybridChunker are NEVER invoked when has_engineering_records=True."""
    def _forbid(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("HybridChunker / _chunk_docling_ast must NOT run on engineering CAD documents")

    monkeypatch.setattr(chunker_mod, "HybridChunker", _forbid)
    monkeypatch.setattr(HierarchicalChunker, "_chunk_docling_ast", _forbid)

    fake_docling_dict = {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "name": "dummy",
        "texts": [{"self_ref": "#/texts/0", "text": "1 = dummy 6 = flattened"}],
    }
    parse_res = _make_engineering_parse_result(
        engineering_fixtures["ce_chart"],
        json_roundtrip=True,
        docling_dict=fake_docling_dict,
    )

    chunker = HierarchicalChunker(child_max_tokens=256, parent_max_tokens=1024)
    chunks = chunker.chunk(parse_res, domain_profile=_eng_profile())
    assert len(chunks) > 0
    assert all("1 = dummy" not in c.child_text for c in chunks)


def test_chunker_cabinet_wiring_schedule_grouping_and_headers(engineering_fixtures: dict[str, Path]):
    """Verify Cabinet Wiring groups 3-5 rows per child, repeats sheet column headers, and links stems."""
    parse_res = _make_engineering_parse_result(engineering_fixtures["cabinet_wiring"], json_roundtrip=True)
    chunker = HierarchicalChunker(child_max_tokens=256, parent_max_tokens=1024)
    chunks = chunker.chunk(parse_res, domain_profile=_eng_profile())

    assert len(chunks) >= 4
    assert all(c.record_type == "schedule_row" for c in chunks)
    assert all(c.page_numbers == [1] for c in chunks)

    # Every schedule_row child chunk repeats sheet banner and column headers
    for c in chunks:
        assert "Schedule: CABINET WIRING DRAWING" in c.child_text
        assert "Columns:" in c.child_text
        assert "FIELD TAG" in c.child_text
        assert "DCS TAG" in c.child_text
        assert "TERMINAL BLOCK" in c.child_text
        assert "FCS NODE" in c.child_text
        assert "IO MODULE" in c.child_text
        row_lines = [ln for ln in c.child_text.splitlines() if ln.startswith("ITEM:")]
        assert 1 <= len(row_lines) <= 5
        assert c.token_count <= 256

    # Row 01 tags stay together in first chunk with LOOP-10205A stem
    first_chunk = chunks[0]
    row_lines_first = [ln for ln in first_chunk.child_text.splitlines() if ln.startswith("ITEM:")]
    assert 3 <= len(row_lines_first) <= 5
    for token in ("TT-10205A", "TI10205A", "1FTB1", "FCS0201", "AAI143-H"):
        assert token in first_chunk.child_text
        assert token in first_chunk.associated_entities
    assert "LOOP-10205A" in first_chunk.associated_entities

    # Parent window wraps multiple sibling child chunks
    parent_groups: dict[str, list] = {}
    for c in chunks:
        parent_groups.setdefault(c.parent_id, []).append(c)
    assert any(len(group) >= 2 for group in parent_groups.values())


def test_chunker_process_ce_legend_atomic_matrix_and_parent_preamble(engineering_fixtures: dict[str, Path]):
    """Verify Page 1 Legend chunk, atomic Page 2 C&E chunks, and Legend summary in parent_content."""
    parse_res = _make_engineering_parse_result(engineering_fixtures["ce_chart"], json_roundtrip=True)
    chunker = HierarchicalChunker(child_max_tokens=256, parent_max_tokens=1024)
    chunks = chunker.chunk(parse_res, domain_profile=_eng_profile())

    legend_chunks = [c for c in chunks if c.record_type == "document_legend_notes"]
    cause_chunks = [c for c in chunks if c.record_type == "matrix_cause_row"]
    effect_chunks = [c for c in chunks if c.record_type == "matrix_effect_col"]

    assert len(legend_chunks) >= 1
    assert all(c.page_numbers == [1] for c in legend_chunks)
    assert "X = CLOSE / TRIP / ACTIVE SIGNAL" in legend_chunks[0].child_text

    assert len(cause_chunks) >= 5
    assert len(effect_chunks) >= 6
    assert all(c.page_numbers == [2] for c in cause_chunks + effect_chunks)

    # Each matrix record gets its own focused parent window with Legend prepended to parent_content
    cause_parent_ids = {c.parent_id for c in cause_chunks}
    effect_parent_ids = {c.parent_id for c in effect_chunks}
    assert len(cause_parent_ids) == len(cause_chunks)
    assert len(effect_parent_ids) == len(effect_chunks)
    assert cause_parent_ids.isdisjoint(effect_parent_ids)

    for mc in cause_chunks + effect_chunks:
        assert "Legend Symbols: X = CLOSE / TRIP / ACTIVE SIGNAL" in mc.parent_content
        assert mc.child_text in mc.parent_content

    lahh_chunk = next(c for c in cause_chunks if "LAHH-80101A/B/C" in c.child_text)
    for stem in ("LAHH-80101A", "LAHH-80101B", "LAHH-80101C", "LAHH80101"):
        assert stem in lahh_chunk.associated_entities


def test_chunker_esds_logic_diagrams_and_hz_stems(engineering_fixtures: dict[str, Path]):
    """Verify ESDS Logic pages remain atomic cad_logic_diagram chunks with HZ10001 stems."""
    parse_res = _make_engineering_parse_result(engineering_fixtures["esds_logic"], json_roundtrip=True)
    chunker = HierarchicalChunker(child_max_tokens=256, parent_max_tokens=1024)
    chunks = chunker.chunk(parse_res, domain_profile=_eng_profile())

    assert len(chunks) == 2
    assert all(c.record_type == "cad_logic_diagram" for c in chunks)

    c_p1 = next(c for c in chunks if c.page_numbers == [1])
    c_p2 = next(c for c in chunks if c.page_numbers == [2])
    assert "1.1. LTRCV_12" in c_p1.heading_path
    assert "1.18. MAIN_07" in c_p2.heading_path
    assert c_p1.parent_id != c_p2.parent_id

    for c in (c_p1, c_p2):
        assert "HZ10001_R12" in c.associated_entities
        assert "HZ10001" in c.associated_entities
        assert "HZ-10001" in c.associated_entities


def test_chunker_lv_swgr_exploded_cad_ocr_entities(engineering_fixtures: dict[str, Path]):
    """Verify LV SWGR Logic produces exploded_cad_ocr chunk(s) with non-empty associated_entities."""
    parse_res = _make_engineering_parse_result(engineering_fixtures["lv_swgr"], json_roundtrip=True)
    chunker = HierarchicalChunker(child_max_tokens=256, parent_max_tokens=1024)
    chunks = chunker.chunk(parse_res, domain_profile=_eng_profile())

    assert len(chunks) >= 1
    assert all(c.record_type == "exploded_cad_ocr" for c in chunks)
    assert all(c.page_numbers == [1] for c in chunks)
    all_entities = {ent for c in chunks for ent in c.associated_entities}
    assert len(all_entities) > 0
    assert any("MCC-401A" in ent or "CB-401" in ent for ent in all_entities)


def test_chunker_engineering_edge_cases_synthesized_pages_splits_and_mixed_prose():
    """Verify synthesized pages fallback, oversized record splitting, schedule windowing, and mixed prose."""
    # 1. Synthesized pages + oversized matrix record + mixed standard_prose
    long_effects = "; ".join(f"XV-1010{i} (Isolation Valve {i} [Close]) -> Mark 'X'" for i in range(1, 16))
    synth_profile = {
        "schema_version": "1.0",
        "page_modes": {"1": "matrix_rotated_headers", "2": "standard_prose"},
        "legend_context": {"symbol_map": {"X": "CLOSE / TRIP"}},
        "pages": [],
        "records": [
            {
                "record_id": "p1_cause_big",
                "record_type": "matrix_cause_row",
                "page_number": 1,
                "title": "Cause LAHH-80101A (ESD-1)",
                "summary_text": (
                    "CAUSE TAG: LAHH-80101A | GROUP: ESD-1 | VOTING: 2 out of 3\n"
                    f"Triggered Effects (15): {long_effects}"
                ),
                "tags": ["LAHH-80101A"],
            },
            {
                "record_id": "p2_prose_1",
                "record_type": "standard_prose",
                "page_number": 2,
                "title": "General Design Basis",
                "summary_text": "All emergency shutdown systems shall fail safe on loss of instrument air or power.",
                "tags": [],
            },
        ],
    }
    parse_res = ParseResult(
        sha256="synth_sha_001",
        filename="mixed_eng.pdf",
        file_type="pdf",
        markdown="",
        markdown_path=Path("/tmp/mixed.md"),
        parsed_path=Path("/tmp/mixed.json"),
        extra_meta={"has_engineering_records": True, "engineering_profile": synth_profile},
    )
    chunker = HierarchicalChunker(child_max_tokens=75, parent_max_tokens=1024)
    chunks = chunker.chunk(parse_res, domain_profile=_eng_profile())

    matrix_slices = [c for c in chunks if c.record_type == "matrix_cause_row"]
    prose_chunks = [c for c in chunks if c.record_type == "standard_prose"]
    assert len(matrix_slices) >= 2
    assert len({c.parent_id for c in matrix_slices}) == 1  # stay in one focused parent window
    for ms in matrix_slices:
        assert ms.child_text.startswith("Cause LAHH-80101A (ESD-1) [matrix_cause_row]")
        assert "Legend Symbols: X = CLOSE / TRIP" in ms.parent_content
    assert len(prose_chunks) == 1
    assert prose_chunks[0].page_numbers == [2]

    # 2. Schedule windowing across >20 rows and oversized single row fallback
    many_rows = [
        {
            "record_id": f"p1_row_{i + 1}",
            "record_type": "schedule_row",
            "page_number": 1,
            "title": "IO SCHEDULE",
            "summary_text": f"ITEM: {i + 1:02d} | TAG: TT-{10100 + i} | NODE: FCS0201",
            "tags": [f"TT-{10100 + i}", "FCS0201"],
            "metadata": {"sheet_title": "IO SCHEDULE", "headers": ["ITEM", "TAG", "NODE"]},
        }
        for i in range(25)
    ]
    sched_parse = ParseResult(
        sha256="sched_win_sha",
        filename="io_schedule.pdf",
        file_type="pdf",
        markdown="",
        markdown_path=Path("/tmp/io.md"),
        parsed_path=Path("/tmp/io.json"),
        extra_meta={
            "has_engineering_records": True,
            "engineering_profile": {
                "schema_version": "1.0",
                "page_modes": {"1": "coord_grid_schedule"},
                "pages": [{"page_number": 1, "mode": "coord_grid_schedule", "records": many_rows}],
                "records": many_rows,
            },
        },
    )
    large_parent_chunker = HierarchicalChunker(child_max_tokens=200, parent_max_tokens=4096)
    sched_chunks = large_parent_chunker.chunk(sched_parse, domain_profile=_eng_profile())
    assert len(sched_chunks) == 5
    # 25 rows grouped by 5s -> rows 0..19 in w0 and rows 20..24 in w1 -> 2 distinct parent_ids
    assert len({c.parent_id for c in sched_chunks}) == 2

    # 3. Single oversized schedule row still emits min 1 row with sheet header
    huge_row = {
        "record_id": "p1_row_huge",
        "record_type": "schedule_row",
        "page_number": 1,
        "title": "BIG SCHEDULE",
        "summary_text": "ITEM: 01 | TAG: TT-99901 | DESC: " + " ".join(f"SIGNAL_{i}" for i in range(60)),
        "tags": ["TT-99901"],
        "metadata": {"sheet_title": "BIG SCHEDULE", "headers": ["ITEM", "TAG", "DESC"]},
    }
    huge_parse = ParseResult(
        sha256="huge_sched_sha",
        filename="big_sched.pdf",
        file_type="pdf",
        markdown="",
        markdown_path=Path("/tmp/big.md"),
        parsed_path=Path("/tmp/big.json"),
        extra_meta={
            "has_engineering_records": True,
            "engineering_profile": {
                "schema_version": "1.0",
                "page_modes": {"1": "coord_grid_schedule"},
                "pages": [{"page_number": 1, "mode": "coord_grid_schedule", "records": [huge_row]}],
                "records": [huge_row],
            },
        },
    )
    tight_chunker = HierarchicalChunker(child_max_tokens=50, parent_max_tokens=512)
    huge_chunks = tight_chunker.chunk(huge_parse, domain_profile=_eng_profile())
    assert len(huge_chunks) == 1
    assert "Schedule: BIG SCHEDULE | Columns: ITEM, TAG, DESC" in huge_chunks[0].child_text
    assert "TT-99901" in huge_chunks[0].associated_entities

    # 4. Mixed PDF with standard_prose page (records=[] and text in markdown <!-- page: 2 -->)
    mixed_pages_parse = ParseResult(
        sha256="mixed_pages_sha",
        filename="mixed_pages.pdf",
        file_type="pdf",
        markdown=(
            "<!-- page: 1 -->\n# 1.1. LTRCV_12 (Page 1)\nPOU / Sheet: 1.1. LTRCV_12\n\n"
            "<!-- page: 2 -->\nGeneral instrumentation philosophy requires fail-safe valve closure on ESD-1 trip."
        ),
        markdown_path=Path("/tmp/mp.md"),
        parsed_path=Path("/tmp/mp.json"),
        extra_meta={
            "has_engineering_records": True,
            "engineering_profile": {
                "schema_version": "1.0",
                "page_modes": {"1": "cad_logic_diagram", "2": "standard_prose"},
                "pages": [
                    {
                        "page_number": 1,
                        "mode": "cad_logic_diagram",
                        "records": [
                            {
                                "record_id": "p1_logic_1",
                                "record_type": "cad_logic_diagram",
                                "page_number": 1,
                                "title": "1.1. LTRCV_12",
                                "summary_text": "POU / Sheet: 1.1. LTRCV_12\nOutput Signals (To): XV-10101A",
                                "tags": ["LTRCV_12", "XV-10101A"],
                            }
                        ],
                    },
                    {"page_number": 2, "mode": "standard_prose", "records": []},
                ],
                "records": [
                    {
                        "record_id": "p1_logic_1",
                        "record_type": "cad_logic_diagram",
                        "page_number": 1,
                        "title": "1.1. LTRCV_12",
                        "summary_text": "POU / Sheet: 1.1. LTRCV_12\nOutput Signals (To): XV-10101A",
                        "tags": ["LTRCV_12", "XV-10101A"],
                    }
                ],
            },
        },
    )
    mixed_chunks = HierarchicalChunker(child_max_tokens=256, parent_max_tokens=1024).chunk(
        mixed_pages_parse,
        domain_profile=_eng_profile(),
    )
    assert [c.record_type for c in mixed_chunks] == ["cad_logic_diagram", "standard_prose"]
    assert mixed_chunks[1].page_numbers == [2]
    assert "fail-safe valve closure" in mixed_chunks[1].child_text


# ==============================================================================
# 3. HybridIndexer Qdrant record_type & SQLite Empty Text Persistence
# ==============================================================================


@pytest.mark.asyncio
async def test_indexer_persists_record_type_in_qdrant_and_keeps_sqlite_text_empty(
    db_session_factory,
    memory_qdrant: AsyncQdrantClient,
    engineering_fixtures: dict[str, Path],
):
    """Verify HybridIndexer stores record_type in Qdrant payloads while DocumentChunk text stays ''."""
    parse_res = _make_engineering_parse_result(engineering_fixtures["ce_chart"], json_roundtrip=True)
    chunker = HierarchicalChunker(child_max_tokens=256, parent_max_tokens=1024)
    chunks = chunker.chunk(parse_res, domain_profile=_eng_profile())
    assert len(chunks) > 0

    embedder = DeterministicStubEmbedder(dimensions=32)
    sparse = DeterministicBM25Encoder()
    collection = "test_eng_record_type_collection"
    indexer = HybridIndexer(
        qdrant=memory_qdrant,
        embedder=embedder,
        sparse_encoder=sparse,
        collection=collection,
    )

    async with db_session_factory() as session:
        doc = Document(
            filename=parse_res.filename,
            file_type="pdf",
            sha256=parse_res.sha256,
            status="indexing",
        )
        session.add(doc)
        await session.commit()
        await session.refresh(doc)
        doc_id = doc.id

        indexed_count = await indexer.index_chunks(session, doc_id=doc_id, chunks=chunks)
        assert indexed_count == len(chunks)

        db_chunks = (
            await session.execute(select(DocumentChunk).where(DocumentChunk.doc_id == doc_id))
        ).scalars().all()
        assert len(db_chunks) == len(chunks)
        for row in db_chunks:
            assert row.text == ""
            assert row.parent_content == ""

    points, _ = await memory_qdrant.scroll(
        collection_name=collection,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    assert len(points) == len(chunks)
    qdrant_record_types = {pt.payload.get("record_type") for pt in points if pt.payload}
    assert {"document_legend_notes", "matrix_cause_row", "matrix_effect_col"} <= qdrant_record_types
    for pt in points:
        assert pt.payload is not None
        assert pt.payload["doc_id"] == doc_id
        assert pt.payload["child_text"]
        assert pt.payload["parent_content"]


def test_review_regressions_prose_main_exclusion_and_extra_isa_loops():
    """Verify MAINTENANCE prose does not match MAIN* signals and ZI/VI/XI loops cross-link."""
    from src.ingestion.engineering_extractor import _extract_record_tags

    assert _extract_record_tags("MAINTENANCE MAINLINE SCHEDULE MAIN_07") == ["MAIN_07"]

    ex = EntityExtractor()
    assert ex.extract_query_tags_and_stems("Check MAINTENANCE schedule and MAINLINE valve") == []

    zt_stems = set(ex.extract_query_tags_and_stems("Inspect ZT-10101A vs ZI10101A"))
    assert "LOOP-10101A" in zt_stems
    assert {"ZT-10101A", "ZI10101A"} <= zt_stems

    vt_stems = set(ex.extract_query_tags_and_stems("Inspect VT-30321A and VI30321A"))
    assert "LOOP-30321A" in vt_stems

    chunker = HierarchicalChunker(child_max_tokens=40, parent_max_tokens=160)
    oversized = "; ".join(["word"] * 50 + ["tail"])
    slices = chunker._split_lines_with_repeated_header(
        oversized,
        header_line="Header: Test",
        max_tokens=40,
    )
    assert len(slices) >= 2
    for sl in slices:
        assert chunker.count_tokens(sl) <= 45


def test_entity_extractor_fire_gas_zones_and_composite_tags():
    """Verify bidirectional FZ/GZ zone normalization and F&G composite tag extraction without prose leakage."""
    ex = EntityExtractor()

    for sample, expected in [
        ("FZ-001", {"FZ-001", "FZ001"}),
        ("FZ001", {"FZ-001", "FZ001"}),
        ("GZ-002", {"GZ-002", "GZ002"}),
        ("GZ002", {"GZ-002", "GZ002"}),
        ("GA-FZ001", {"GA-FZ001", "GAFZ001", "FZ001", "FZ-001"}),
        ("GA-FZ001J", {"GA-FZ001J", "FZ001", "FZ-001"}),
        ("HR-FZ006-02A", {"HR-FZ006-02A", "FZ006", "FZ-006"}),
        ("BC-FZ001B", {"BC-FZ001B", "BCFZ001B", "FZ001", "FZ-001"}),
        ("BC-FZ006-02AB", {"BC-FZ006-02AB", "FZ006", "FZ-006"}),
        ("FGS-ESD-3-1", {"FGS-ESD-3-1", "FGSESD31"}),
        ("HS-00401BJ", {"HS-00401BJ", "HS00401BJ"}),
        ("HS-00401MA", {"HS-00401MA", "HS00401MA"}),
    ]:
        extracted = set(ex.extract_entities(sample))
        assert expected <= extracted, f"Missing {expected - extracted} in {extracted} for {sample}"

    q_stems = set(ex.extract_query_tags_and_stems("what are causes will active FZ-001 signal?"))
    assert {"FZ-001", "FZ001"} <= q_stems
    assert ex.extract_query_tags_and_stems("Check FIRE ZONE signal and GAS alarm") == []

