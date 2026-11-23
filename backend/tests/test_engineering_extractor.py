"""Unit and fixture tests for Phase 1 per-page engineering PDF profiler and extractors."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pymupdf as fitz
import pytest

from scripts.generate_engineering_fixtures import generate_all_fixtures
from src.ingestion.engineering_extractor import (
    LegendContext,
    _merge_adjacent_voting_spans,
    compute_bbox_iou,
    compute_safe_ocr_dpi,
    deduplicate_ocr_detections,
    extract_cad_logic_sheet,
    extract_coord_grid_schedule,
    extract_engineering_pdf,
    extract_exploded_cad_ocr,
    extract_legend_context,
    extract_rotated_ce_matrix,
    profile_pdf_page,
)

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


def _build_portrait_prose_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "1. Executive Summary", fontsize=16)
    page.insert_text(
        (72, 110),
        "Operating margin expanded by 3.2% year-over-year driven by supply chain automation. "
        "Customer retention remains above ninety-four percent across all geographic operating segments.",
        fontsize=11,
    )
    doc.save(str(path))
    doc.close()
    return path


def _build_image_heavy_portrait_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Fig. 1 Layout Overview", fontsize=10)
    pix = fitz.Pixmap(fitz.csRGB, (0, 0, 12, 12), False)
    img_bytes = pix.tobytes("png")
    for idx in range(3):
        page.insert_image(fitz.Rect(60, 100 + idx * 40, 100, 140 + idx * 40), stream=img_bytes)
    doc.save(str(path))
    doc.close()
    return path


def test_fixtures_total_size_under_300kb(engineering_fixtures: dict[str, Path]):
    total_bytes = sum(p.stat().st_size for p in engineering_fixtures.values())
    assert total_bytes < 300 * 1024, f"Fixtures total {total_bytes} bytes exceeds 300 KB"
    for key, path in engineering_fixtures.items():
        assert path.is_file(), f"Missing fixture {key}: {path}"


def test_fixture_generator_supports_source_dir_slicing(tmp_path: Path, engineering_fixtures: dict[str, Path]):
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    # Create multi-page source PDFs matching the expected page indices (P2, P3-P4, P8/P51, P3)
    for name, total_pages, copy_from in [
        ("Cabinet_Wiring_Rev1.pdf", 3, engineering_fixtures["cabinet_wiring"]),
        ("Process_Cause_and_Effect_Chart.pdf", 5, engineering_fixtures["ce_chart"]),
        ("ESDS_Logic_Diagrams.pdf", 52, engineering_fixtures["esds_logic"]),
        ("LV_SWGR_Logic_Main.pdf", 4, engineering_fixtures["lv_swgr"]),
    ]:
        out = fitz.open()
        for _ in range(total_pages):
            out.new_page(width=1190, height=842)
        src_doc = fitz.open(str(copy_from))
        if name.startswith("Cabinet"):
            out.delete_page(1)
            out.insert_pdf(src_doc, from_page=0, to_page=0, start_at=1)
        elif name.startswith("Process"):
            out.delete_page(2)
            out.delete_page(2)
            out.insert_pdf(src_doc, from_page=0, to_page=1, start_at=2)
        elif name.startswith("ESDS"):
            out.delete_page(7)
            out.insert_pdf(src_doc, from_page=0, to_page=0, start_at=7)
            out.delete_page(50)
            out.insert_pdf(src_doc, from_page=1, to_page=1, start_at=50)
        else:
            out.delete_page(2)
            out.insert_pdf(src_doc, from_page=0, to_page=0, start_at=2)
        src_doc.close()
        out.save(str(src_dir / name))
        out.close()

    sliced_out = tmp_path / "sliced"
    generated = generate_all_fixtures(out_dir=sliced_out, source_dir=src_dir)
    assert len(generated) == 4
    with fitz.open(str(generated["esds_logic_p8_p51.pdf"])) as sliced_esds:
        assert len(sliced_esds) == 2
        assert "LTRCV_12" in sliced_esds[0].get_text("text")
        assert "MAIN_07" in sliced_esds[1].get_text("text")


def test_profile_pdf_page_modes_and_speed(tmp_path: Path, engineering_fixtures: dict[str, Path]):
    prose_pdf = _build_portrait_prose_pdf(tmp_path / "prose.pdf")
    img_pdf = _build_image_heavy_portrait_pdf(tmp_path / "complex.pdf")

    with fitz.open(str(prose_pdf)) as doc:
        t0 = time.perf_counter()
        mode, _ = profile_pdf_page(doc[0])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert mode == "standard_prose"
        assert elapsed_ms < 50.0

    with fitz.open(str(img_pdf)) as doc:
        mode, metrics = profile_pdf_page(doc[0])
        assert mode == "standard_prose"
        assert metrics["drawing_count"] <= 300

    with fitz.open(str(engineering_fixtures["cabinet_wiring"])) as doc:
        t0 = time.perf_counter()
        mode, metrics = profile_pdf_page(doc[0])
        assert (time.perf_counter() - t0) * 1000.0 < 50.0
        assert mode == "coord_grid_schedule"
        assert metrics["drawing_count"] > 160
        assert metrics["column_band_count"] >= 10

    with fitz.open(str(engineering_fixtures["ce_chart"])) as doc:
        m1, _ = profile_pdf_page(doc[0])
        m2, metrics2 = profile_pdf_page(doc[1])
        assert m1 == "document_legend_notes"
        assert m2 == "matrix_rotated_headers"
        assert metrics2["rotated_span_count"] >= 15
        assert metrics2["drawing_count"] > 160

    with fitz.open(str(engineering_fixtures["esds_logic"])) as doc:
        m1, met1 = profile_pdf_page(doc[0])
        m2, met2 = profile_pdf_page(doc[1])
        assert m1 == "cad_logic_diagram"
        assert m2 == "cad_logic_diagram"
        assert met1["drawing_count"] > 160
        assert met2["drawing_count"] > 160

    with fitz.open(str(engineering_fixtures["lv_swgr"])) as doc:
        mode, metrics = profile_pdf_page(doc[0])
        assert mode == "exploded_cad_ocr"
        assert metrics["word_count"] == 0
        assert metrics["drawing_count"] > 350


def test_cabinet_wiring_schedule_extraction(engineering_fixtures: dict[str, Path]):
    with fitz.open(str(engineering_fixtures["cabinet_wiring"])) as doc:
        res = extract_coord_grid_schedule(doc[0], page_number=1)

    assert res.mode == "coord_grid_schedule"
    assert len(res.records) >= 16
    assert all(r.record_type == "schedule_row" for r in res.records)

    row1 = next(r for r in res.records if "TT-10205A" in r.summary_text)
    cols = row1.metadata["columns"]
    assert cols["FIELD TAG"] == "TT-10205A"
    assert cols["DCS TAG"] == "TI10205A"
    assert cols["TERMINAL BLOCK"] == "1FTB1"
    assert cols["FCS NODE"] == "FCS0201"
    assert cols["IO MODULE"] == "AAI143-H"
    assert "TT-10205A" in row1.tags
    assert "TI10205A" in row1.tags
    assert "1FTB1" in row1.tags
    assert "FCS0201" in row1.tags
    assert "AAI143-H" in row1.tags
    assert "CABINET WIRING DRAWING" in res.markdown
    assert "| ITEM | FIELD TAG | DCS TAG |" in res.markdown


def test_coord_grid_schedule_x_centroid_fallback(tmp_path: Path):
    """When grid lines are absent/incomplete, dynamic x-centroid clustering still binds columns."""
    pdf_path = tmp_path / "borderless_schedule.pdf"
    doc = fitz.open()
    page = doc.new_page(width=1190, height=842)
    xs = [50, 120, 240, 360, 480, 600, 740]
    headers = ["ITEM", "FIELD TAG", "DCS TAG", "TERMINAL", "FCS NODE", "MODULE", "SERVICE"]
    for x, h in zip(xs, headers):
        page.insert_text((x, 100), h, fontsize=9)
    rows = [
        ["01", "TT-10205A", "TI10205A", "1FTB1", "FCS0201", "AAI143-H", "TANK TOP TEMP"],
        ["02", "FT-10507", "FIC10507", "1FTB2", "FCS0201", "AAI143-H", "SEND OUT FLOW"],
        ["03", "PT-10803", "PIA30101", "1RTB1", "SCS0211", "AAV144-S", "SUCTION PRESS"],
        ["04", "LT-80101A", "LIA80101A", "1RTB2", "SCS0211", "AAI143-H", "TANK LEVEL"],
    ]
    for r_idx, r_vals in enumerate(rows):
        for x, val in zip(xs, r_vals):
            page.insert_text((x, 130 + r_idx * 25), val, fontsize=9)
    doc.save(str(pdf_path))
    doc.close()

    with fitz.open(str(pdf_path)) as doc:
        mode, _ = profile_pdf_page(doc[0])
        assert mode == "coord_grid_schedule"
        res = extract_coord_grid_schedule(doc[0], page_number=1)
        assert len(res.records) == 4
        assert res.records[0].metadata["columns"]["FIELD TAG"] == "TT-10205A"
        assert res.records[0].metadata["columns"]["MODULE"] == "AAI143-H"


def test_process_ce_legend_and_bidirectional_matrix(engineering_fixtures: dict[str, Path]):
    with fitz.open(str(engineering_fixtures["ce_chart"])) as doc:
        legend_ctx = extract_legend_context(doc)
        assert legend_ctx.symbol_map["X"] == "CLOSE / TRIP / ACTIVE SIGNAL"
        assert legend_ctx.symbol_map["O"] == "OPEN"
        assert legend_ctx.symbol_map["P"] == "PERMISSIVE"
        assert "NOTE 1" in legend_ctx.notes_map
        assert "NOTE 36" in legend_ctx.notes_map
        assert "Legend Symbols:" in legend_ctx.format_summary()
        assert len(legend_ctx.lookup_notes(["NOTE 1", "36"])) == 2

        res = extract_rotated_ce_matrix(doc[1], page_number=2, legend_context=legend_ctx)

    cause_records = [r for r in res.records if r.record_type == "matrix_cause_row"]
    effect_records = [r for r in res.records if r.record_type == "matrix_effect_col"]

    assert len(cause_records) >= 5
    assert len(effect_records) >= 6

    # Forward cause check: LAHH-80101A/B/C -> 2 out of 3 + triggered X/O effects + Legend + NOTE 1
    lahh_rec = next(r for r in cause_records if "LAHH-80101A/B/C" in r.summary_text)
    assert lahh_rec.metadata["voting"] == "2 out of 3"
    assert "XV-10101A" in lahh_rec.summary_text
    assert "XV-10101B" in lahh_rec.summary_text
    assert "HA-10003B" in lahh_rec.summary_text
    assert "FCV-20231" in lahh_rec.summary_text
    assert "CLOSE / TRIP / ACTIVE SIGNAL" in lahh_rec.summary_text
    assert "NOTE 1:" in lahh_rec.summary_text

    # Forward cause check: HZ-10001 -> NOTE 36
    hz_rec = next(r for r in cause_records if "HZ-10001" in r.summary_text)
    assert "NOTE 36:" in hz_rec.summary_text
    assert "UZA30321A" in hz_rec.summary_text

    # Reverse effect check: XV-10101A & HA-10003B
    xv_rec = next(r for r in effect_records if r.metadata["effect_tag"] == "XV-10101A")
    assert "LAHH-80101A/B/C" in xv_rec.summary_text
    assert "HZ-10001" in xv_rec.summary_text
    assert "UA-FGS-ESD-1" in xv_rec.summary_text
    assert "CLOSE / TRIP / ACTIVE SIGNAL" in xv_rec.summary_text

    ha_rec = next(r for r in effect_records if r.metadata["effect_tag"] == "HA-10003B")
    assert "PALL-14261A/B/C" in ha_rec.summary_text
    assert "PDALL-30504" in ha_rec.summary_text
    assert "PERMISSIVE" in ha_rec.summary_text


def test_esds_logic_diagram_extraction(engineering_fixtures: dict[str, Path]):
    with fitz.open(str(engineering_fixtures["esds_logic"])) as doc:
        r1 = extract_cad_logic_sheet(doc[0], page_number=1)
        r2 = extract_cad_logic_sheet(doc[1], page_number=2)

    assert r1.mode == "cad_logic_diagram"
    assert len(r1.records) == 1
    rec1 = r1.records[0]
    assert rec1.record_type == "cad_logic_diagram"
    assert "LTRCV_12" in rec1.title
    assert "C&E PAGE4" in rec1.metadata["cross_references"]
    assert "LAHH-80101A" in rec1.metadata["inputs"]
    assert "HZ10001_R12" in rec1.metadata["inputs"]
    assert "UAFGSESD1_R12" in rec1.metadata["inputs"]
    assert set(["AND", "OR", "RS", "TON"]).issubset(set(rec1.metadata["logic_blocks"]))
    assert "XV-10101A" in rec1.metadata["outputs"]
    assert "RSESD001_R12" in rec1.metadata["outputs"]

    assert r2.mode == "cad_logic_diagram"
    rec2 = r2.records[0]
    assert "MAIN_07" in rec2.title
    assert "C&E PAGE4" in rec2.metadata["cross_references"]
    assert "PALL-14261A" in rec2.metadata["inputs"]
    assert "HA-10003B" in rec2.metadata["outputs"]
    assert "XV-10101A" in rec2.metadata["outputs"]


def test_dpi_clamping_and_iou_deduplication():
    # Standard letter stays at 300 DPI
    assert compute_safe_ocr_dpi(792, 612) == 300
    # A3 landscape (1190 pt) scales down slightly to guarantee <= 4800 px longest edge
    dpi_a3 = compute_safe_ocr_dpi(1190, 842)
    assert 150 <= dpi_a3 <= 300
    assert int(1190 * dpi_a3 / 72.0) <= 4800
    # A2 (1684 pt) scales smoothly inside [150, 300]
    dpi_a2 = compute_safe_ocr_dpi(1684, 1190)
    assert 150 <= dpi_a2 <= 300
    assert int(1684 * dpi_a2 / 72.0) <= 4800
    # Oversized A0 (3370 pt) clamps hard to stay <= 4800 px
    dpi_a0 = compute_safe_ocr_dpi(3370, 2384)
    assert int(3370 * dpi_a0 / 72.0) <= 4800

    # IoU deduplication collapses overlapping tile duplicates
    assert compute_bbox_iou([10, 10, 110, 50], [12, 12, 112, 52]) > 0.8
    deduped = deduplicate_ocr_detections(
        [
            {"text": "MCC-401A", "bbox": [100.0, 175.0, 280.0, 210.0], "score": 0.98, "tile": "top_left"},
            {"text": "MCC-401A", "bbox": [103.0, 176.0, 282.0, 211.0], "score": 0.91, "tile": "top_right"},
            {"text": "CB-401", "bbox": [700.0, 175.0, 840.0, 210.0], "score": 0.97, "tile": "top_right"},
        ]
    )
    assert len(deduped) == 2
    assert [d["text"] for d in deduped] == ["MCC-401A", "CB-401"]


def test_exploded_cad_ocr_mock_and_real_rapidocr(engineering_fixtures: dict[str, Path]):
    with fitz.open(str(engineering_fixtures["lv_swgr"])) as doc:
        page = doc[0]
        assert len(page.get_text("words")) == 0
        assert len(page.get_drawings()) > 350

        # 1. Mock engine test verifying 5 tiles (2x2 + title_block), batch=1 calls, and IoU dedup
        calls: list[tuple[int, int]] = []

        def fake_ocr(tile_arr: np.ndarray):
            calls.append(tile_arr.shape[:2])
            if len(calls) == 1:
                return [([[20, 20], [180, 20], [180, 70], [20, 70]], "MCC-401A", 0.99)], 0.01
            if len(calls) == 2:
                return [([[20, 20], [180, 20], [180, 70], [20, 70]], "CB-401", 0.97)], 0.01
            if len(calls) == 5:
                return [([[20, 20], [220, 20], [220, 70], [20, 70]], "LV SWGR LOGIC", 0.96)], 0.01
            return [], 0.01

        mock_res = extract_exploded_cad_ocr(page, page_number=1, ocr_engine=fake_ocr)
        assert len(calls) == 5
        assert mock_res.metrics["rendered_width_px"] <= 4800
        assert mock_res.metrics["rendered_height_px"] <= 4800
        assert "MCC-401A" in mock_res.markdown
        assert "CB-401" in mock_res.markdown
        assert "LV SWGR LOGIC" in mock_res.markdown

        # 2. Real RapidOCR test on CPU recovering vector-stroke schematic labels
        real_res = extract_exploded_cad_ocr(page, page_number=1)
        assert real_res.mode == "exploded_cad_ocr"
        assert len(real_res.records) == 1
        assert real_res.records[0].record_type == "exploded_cad_ocr"
        combined = real_res.markdown.upper()
        for expected_token in ("400V BUS A", "MCC-401A", "CB-401", "TRIP COIL", "LV SWGR LOGIC"):
            assert expected_token in combined, f"Expected '{expected_token}' in OCR markdown:\n{combined}"


def test_extract_engineering_pdf_contract(tmp_path: Path, engineering_fixtures: dict[str, Path]):
    # Pure prose PDF -> returns None
    prose_pdf = _build_portrait_prose_pdf(tmp_path / "prose_only.pdf")
    assert extract_engineering_pdf(prose_pdf) is None

    # Image-heavy portrait PDF -> returns None
    complex_pdf = _build_image_heavy_portrait_pdf(tmp_path / "image_portrait.pdf")
    assert extract_engineering_pdf(complex_pdf) is None

    # Process C&E Chart -> returns canonical engineering_profile with LegendContext + bidirectional records
    ce_result = extract_engineering_pdf(engineering_fixtures["ce_chart"])
    assert ce_result is not None
    assert ce_result["has_engineering_records"] is True
    assert ce_result["page_count"] == 2
    assert "<!-- page: 1 -->" in ce_result["markdown"]
    assert "<!-- page: 2 -->" in ce_result["markdown"]

    profile = ce_result["engineering_profile"]
    assert profile["schema_version"] == "1.0"
    assert profile["page_modes"][1] == "document_legend_notes"
    assert profile["page_modes"][2] == "matrix_rotated_headers"
    assert profile["legend_context"]["symbol_map"]["X"] == "CLOSE / TRIP / ACTIVE SIGNAL"
    assert len(profile["pages"]) == 2
    assert any(r["record_type"] == "document_legend_notes" for r in profile["records"])
    assert any(r["record_type"] == "matrix_cause_row" for r in profile["records"])
    assert any(r["record_type"] == "matrix_effect_col" for r in profile["records"])


# ==============================================================================
# Phase 1 Parser, Worker & Documents Integration Tests
# ==============================================================================


def test_inspect_pdf_fast_path_cad_and_prose_guards(
    tmp_path: Path, engineering_fixtures: dict[str, Path]
):
    """Hardened preflight probe rejects all CAD layouts and retains fast-path for standard portrait prose."""
    from src.ingestion.parser import inspect_pdf_fast_path

    for name, fixture_path in engineering_fixtures.items():
        is_fast, reason = inspect_pdf_fast_path(fixture_path)
        assert is_fast is False, f"Expected CAD fixture {name} to fail fast path, got: {reason}"
        assert "engineering_cad_layout" in reason, f"Expected engineering_cad_layout reason for {name}, got: {reason}"

    prose_pdf = _build_portrait_prose_pdf(tmp_path / "standard_prose.pdf")
    is_fast_prose, prose_reason = inspect_pdf_fast_path(prose_pdf)
    assert is_fast_prose is True, f"Expected portrait prose PDF to pass fast path, got: {prose_reason}"
    assert "digital_text" in prose_reason

    img_pdf = _build_image_heavy_portrait_pdf(tmp_path / "complex_portrait.pdf")
    is_fast_img, img_reason = inspect_pdf_fast_path(img_pdf)
    assert is_fast_img is False
    assert "scanned_or_heavy_layout" in img_reason

    # Landscape PDF with >=25 vector drawings but standard_prose text falls back to landscape_cad_grid guard
    landscape_grid_pdf = tmp_path / "landscape_grid_prose.pdf"
    doc = fitz.open()
    page = doc.new_page(width=1190, height=842)
    page.insert_text(
        (72, 72),
        "General equipment overview notes covering installation tolerances, inspection steps, "
        "and commissioning verification guidelines for field engineers across all plant zones.",
        fontsize=11,
    )
    for idx in range(28):
        page.draw_line((60, 140 + idx * 12), (460, 140 + idx * 12))
    doc.save(str(landscape_grid_pdf))
    doc.close()

    is_fast_land, land_reason = inspect_pdf_fast_path(landscape_grid_pdf)
    assert is_fast_land is False
    assert "engineering_cad_layout" in land_reason
    assert "landscape_cad_grid" in land_reason


def test_parser_service_engineering_cad_all_fixtures_and_cache_guards(
    tmp_path: Path, engineering_fixtures: dict[str, Path]
):
    """DocumentParserService extracts all 4 CAD fixtures under engineering_cad in Auto/High-Fidelity modes and honors cache guards."""
    import json
    from unittest.mock import patch

    from src.ingestion.parser import DocumentParserService

    # 1. Both prefer_fallback=None (Auto) and prefer_fallback=False (High-Fidelity) route CAD fixtures to engineering_cad
    for mode_label, fallback_arg in (("auto", None), ("high_fidelity", False)):
        mode_data_dir = tmp_path / f"data_{mode_label}"
        parser = DocumentParserService(data_dir=mode_data_dir)
        for name, fixture_path in engineering_fixtures.items():
            res = parser.parse(fixture_path, use_pymupdf_fallback=fallback_arg)
            assert res.from_cache is False, f"{name} ({mode_label}) should not hit empty cache"
            assert res.extra_meta["engine"] == "engineering_cad"
            assert res.extra_meta["has_engineering_records"] is True

            profile = res.extra_meta["engineering_profile"]
            assert "records" in profile and len(profile["records"]) >= 1
            assert "legend_context" in profile
            assert "pages" in profile and len(profile["pages"]) >= 1

            assert res.parsed_path.is_file()
            assert res.markdown_path.is_file()
            cached_disk = json.loads(res.parsed_path.read_text(encoding="utf-8"))
            assert cached_disk["extra_meta"]["engine"] == "engineering_cad"
            assert cached_disk["extra_meta"]["has_engineering_records"] is True
            disk_profile = cached_disk["extra_meta"]["engineering_profile"]
            assert "records" in disk_profile
            assert "legend_context" in disk_profile
            assert "pages" in disk_profile

    # 2. Cache hit under prefer_fallback=False (zero re-parse!) and invalidation under prefer_fallback=True
    cache_data_dir = tmp_path / "data_cache_guard"
    parser = DocumentParserService(data_dir=cache_data_dir)
    cad_fixture = engineering_fixtures["cabinet_wiring"]

    initial = parser.parse(cad_fixture, use_pymupdf_fallback=None)
    assert initial.from_cache is False
    assert initial.extra_meta["engine"] == "engineering_cad"

    with (
        patch("src.ingestion.parser.extract_engineering_pdf") as mock_eng,
        patch.object(DocumentParserService, "_parse_with_docling") as mock_dl,
    ):
        cached_hi = parser.parse(cad_fixture, use_pymupdf_fallback=False)
        assert cached_hi.from_cache is True
        assert cached_hi.extra_meta["engine"] == "engineering_cad"
        mock_eng.assert_not_called()
        mock_dl.assert_not_called()

    fast_override = parser.parse(cad_fixture, use_pymupdf_fallback=True)
    assert fast_override.from_cache is False
    assert fast_override.extra_meta["engine"] == "pymupdf4llm"

    # 3. Unexpected exception in extract_engineering_pdf falls through cleanly to standard PDF routing
    prose_pdf = _build_portrait_prose_pdf(tmp_path / "fallback_prose.pdf")
    with patch("src.ingestion.parser.extract_engineering_pdf", side_effect=RuntimeError("boom")):
        prose_res = parser.parse(prose_pdf, use_pymupdf_fallback=None)
    assert prose_res.extra_meta["engine"] == "pymupdf4llm"


@pytest.mark.asyncio
async def test_orchestrator_cad_upgrade_protection_and_checkpoint_reuse(
    tmp_path: Path, engineering_fixtures: dict[str, Path]
):
    """IngestionOrchestrator protects engineering_cad documents from redundant mode upgrades and reuses checkpoints."""
    from unittest.mock import patch

    from qdrant_client import AsyncQdrantClient
    from sqlalchemy import func, select

    from src.db.models import IngestionJob
    from src.db.session import create_engine, create_session_factory, init_db
    from src.ingestion.classifier import set_classifier_force_stub, unload_classifier
    from src.ingestion.indexer import DeterministicBM25Encoder, DeterministicStubEmbedder
    from src.ingestion.worker import IngestionOrchestrator

    db_file = tmp_path / "db" / "worker_cad.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_file.as_posix()}")
    await init_db(engine)
    session_factory = create_session_factory(engine)
    qdrant = AsyncQdrantClient(location=":memory:")
    set_classifier_force_stub(True)

    try:
        async with IngestionOrchestrator(
            session_factory=session_factory,
            qdrant=qdrant,
            embedder=DeterministicStubEmbedder(dimensions=32),
            sparse_encoder=DeterministicBM25Encoder(),
            data_dir=tmp_path / "data",
            collection="test_eng_worker",
            max_workers=1,
            use_process_pool=False,
        ) as orchestrator:
            cad_pdf = engineering_fixtures["cabinet_wiring"]
            doc1 = await orchestrator.ingest_file(cad_pdf, approval_mode="auto_all")
            assert doc1.status == "ready"
            assert doc1.parsing_engine == "engineering_cad"

            # Re-ingesting with prefer_pymupdf4llm=False must NOT trigger re-parse upgrade
            with patch("src.ingestion.worker._worker_parse_task") as mock_parse_task:
                doc2 = await orchestrator.ingest_file(
                    cad_pdf, prefer_pymupdf4llm=False, approval_mode="auto_all"
                )
                assert doc2.id == doc1.id
                assert doc2.parsing_engine == "engineering_cad"
                mock_parse_task.assert_not_called()

            # Stage 1 classification call with prefer_pymupdf4llm=False reuses engineering_cad disk checkpoint
            with patch("src.ingestion.worker._worker_parse_task") as mock_parse_task:
                doc3, _ = await orchestrator.process_to_classification(
                    doc1.id, prefer_pymupdf4llm=False, approval_mode="auto_all"
                )
                assert doc3.parsing_engine == "engineering_cad"
                mock_parse_task.assert_not_called()

            async with session_factory() as session:
                job_count = (
                    await session.execute(
                        select(func.count()).select_from(IngestionJob).where(IngestionJob.doc_id == doc1.id)
                    )
                ).scalar_one()
            assert job_count == 1
    finally:
        set_classifier_force_stub(False)
        unload_classifier()
        await qdrant.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_fastapi_upload_cad_dedup_downgrade_protection_and_explicit_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engineering_fixtures: dict[str, Path]
):
    """End-to-end FastAPI upload + duplicate protection + explicit retry override for CAD PDFs."""
    import asyncio
    import io
    from unittest.mock import patch

    from asgi_lifespan import LifespanManager
    from httpx import ASGITransport, AsyncClient
    from sqlalchemy import func, select

    from src.api.main import create_app
    from src.config import get_settings
    from src.db.models import IngestionJob
    from src.ingestion.classifier import set_classifier_force_stub, unload_classifier
    from src.ingestion.indexer import (
        DeterministicBM25Encoder,
        DeterministicStubEmbedder,
        set_embedder,
        set_sparse_encoder,
    )
    from src.ingestion.parser import DocumentParserService
    from src.rag.graph import close_checkpointer
    from src.rag.llm import set_chat_model
    from src.rag.reranker import set_reranker
    from src.rag.retriever import set_active_qdrant_client

    data_dir = tmp_path / "data"
    db_dir = tmp_path / "data" / "db"
    dist_dir = tmp_path / "frontend_dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "index.html").write_text("<!doctype html><html><body>Test SPA</body></html>", encoding="utf-8")

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("DB_DIR", str(db_dir))
    monkeypatch.setenv("FRONTEND_DIST_DIR", str(dist_dir))
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("INGESTION_USE_PROCESS_POOL", "false")
    monkeypatch.setenv("INGESTION_ENABLE_LLM_CLASSIFICATION_FALLBACK", "false")
    monkeypatch.delenv("PREFER_PYMUPDF4LLM", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    get_settings.cache_clear()
    set_classifier_force_stub(True)
    set_embedder(DeterministicStubEmbedder(dimensions=32))
    set_sparse_encoder(DeterministicBM25Encoder())
    set_chat_model(None)
    set_reranker(None)
    set_active_qdrant_client(None)

    async def _wait_ready(client: AsyncClient, doc_id: str, timeout_s: float = 20.0) -> dict:
        deadline = asyncio.get_running_loop().time() + timeout_s
        last_payload: dict = {}
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

    cad_bytes = engineering_fixtures["cabinet_wiring"].read_bytes()

    try:
        test_app = create_app()
        async with LifespanManager(test_app) as manager:
            transport = ASGITransport(app=manager.app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                # 1. Initial Auto upload parses under parsing_engine == "engineering_cad"
                resp_init = await client.post(
                    "/api/v1/documents/upload?approval_mode=auto_all",
                    files={"file": ("cabinet_wiring_p2.pdf", io.BytesIO(cad_bytes), "application/pdf")},
                )
                assert resp_init.status_code == 202
                init_data = resp_init.json()
                assert init_data["duplicate"] is False
                doc_id = init_data["id"]

                final_init = await _wait_ready(client, doc_id)
                assert final_init["status"] == "ready"
                assert final_init["parsing_engine"] == "engineering_cad"

                # 2. Duplicate upload with prefer_fast_path=false returns duplicate=True (HTTP 200) without re-parse
                with patch.object(DocumentParserService, "parse") as mock_parse:
                    resp_dup_hi = await client.post(
                        "/api/v1/documents/upload?prefer_fast_path=false",
                        files={"file": ("cabinet_wiring_p2.pdf", io.BytesIO(cad_bytes), "application/pdf")},
                    )
                    assert resp_dup_hi.status_code == 200
                    dup_hi = resp_dup_hi.json()
                    assert dup_hi["id"] == doc_id
                    assert dup_hi["duplicate"] is True
                    assert dup_hi["parsing_engine"] == "engineering_cad"
                    mock_parse.assert_not_called()

                # 3. Duplicate upload with prefer_fast_path=true is protected from downgrade (HTTP 200, keeps engineering_cad)
                with patch.object(DocumentParserService, "parse") as mock_parse:
                    resp_dup_fast = await client.post(
                        "/api/v1/documents/upload?prefer_fast_path=true",
                        files={"file": ("cabinet_wiring_p2.pdf", io.BytesIO(cad_bytes), "application/pdf")},
                    )
                    assert resp_dup_fast.status_code == 200
                    dup_fast = resp_dup_fast.json()
                    assert dup_fast["id"] == doc_id
                    assert dup_fast["duplicate"] is True
                    assert dup_fast["parsing_engine"] == "engineering_cad"
                    mock_parse.assert_not_called()

                async with test_app.state.session_factory() as session:
                    job_count_before_retry = (
                        await session.execute(
                            select(func.count()).select_from(IngestionJob).where(IngestionJob.doc_id == doc_id)
                        )
                    ).scalar_one()
                assert job_count_before_retry == 1

                # 4. Explicit retry with force_reparse=true&prefer_fast_path=true overrides to pymupdf4llm
                resp_retry = await client.post(
                    f"/api/v1/documents/{doc_id}/retry?force_reparse=true&prefer_fast_path=true"
                )
                assert resp_retry.status_code == 200
                retry_data = resp_retry.json()
                assert retry_data["status"] == "queued"

                final_retry = await _wait_ready(client, doc_id)
                assert final_retry["status"] == "ready"
                assert final_retry["parsing_engine"] == "pymupdf4llm"
    finally:
        set_classifier_force_stub(False)
        unload_classifier()
        set_chat_model(None)
        set_embedder(None)
        set_sparse_encoder(None)
        set_reranker(None)
        set_active_qdrant_client(None)
        await close_checkpointer()
        get_settings.cache_clear()


def _build_rotated_fgs_ce_pdf(path: Path) -> Path:
    """Create a 3-page /Rotate 90 Fire & Gas C&E chart exercising dense grids, merged cells, and continuation pruning."""
    doc = fitz.open()

    # Page 1: Rotated cover/summary sheet (must NOT become matrix_rotated_headers)
    p1 = doc.new_page(width=842.0, height=1191.0)
    # Place unrotated text at top-to-bottom coordinates; set_rotation(90) rotates visual display
    p1.insert_text((80, 180), "Fire and Gas Cause & Effect Chart - Cover Sheet", fontsize=14, fontname="helv")
    p1.insert_text((80, 230), "Project Doc No: 195363-07-LS-0001 | Rev.2", fontsize=10, fontname="helv")
    p1.insert_text((80, 270), "General notes and revision history for LNG receiving terminal.", fontsize=10, fontname="helv")
    p1.set_rotation(90)

    def _add_rotated_matrix_page(
        *,
        effects: list[tuple[str, str, str, str]],  # (pid, action, desc, tag)
        causes: list[dict],
        merged_pairs: bool = False,
        with_notes: bool = False,
    ) -> None:
        # Build visual landscape content directly then rotate page? Note in PyMuPDF 2.0
        # drawing in unrotated (842x1191) rotated-90 space with rotate=90 reproduces real CAD /Rotate 90 PDFs.
        # Specifically: unrotated text written with rotate=90 has unrotated dir=(0,-1) -> visual dir=(1,0),
        # and unrotated text with dir=(1,0) has visual dir=(0,1) -> vertical Effect headers!
        # Let's construct in visual coords (1191x842) and inverse-transform by page.derotation_matrix,
        # which guarantees authentic /Rotate 90 unrotated PDF stream + /Rotate 90 dictionary!
        page = doc.new_page(width=842.0, height=1191.0)
        page.set_rotation(90)
        dm = page.derotation_matrix

        def put_horiz(vx: float, vy: float, text: str, fs: float = 3.5) -> None:
            pt = fitz.Point(vx, vy) * dm
            page.insert_text((pt.x, pt.y), text, fontsize=fs, fontname="helv", rotate=90)

        def put_vert(vx: float, vy: float, text: str, fs: float = 3.5) -> None:
            pt = fitz.Point(vx, vy) * dm
            page.insert_text((pt.x, pt.y), text, fontsize=fs, fontname="helv", rotate=180)

        def vline(vx: float, vy0: float, vy1: float) -> None:
            p_a = fitz.Point(vx, vy0) * dm
            p_b = fitz.Point(vx, vy1) * dm
            if abs(p_a.x - p_b.x) < 1.0:
                page.draw_rect(fitz.Rect(p_a.x - 0.2, min(p_a.y, p_b.y), p_a.x + 0.2, max(p_a.y, p_b.y)), width=0.4)
            else:
                page.draw_rect(fitz.Rect(min(p_a.x, p_b.x), p_a.y - 0.2, max(p_a.x, p_b.x), p_a.y + 0.2), width=0.4)

        def hline(vy: float, vx0: float, vx1: float) -> None:
            p_a = fitz.Point(vx0, vy) * dm
            p_b = fitz.Point(vx1, vy) * dm
            if abs(p_a.y - p_b.y) < 1.0:
                page.draw_rect(fitz.Rect(min(p_a.x, p_b.x), p_a.y - 0.2, max(p_a.x, p_b.x), p_a.y + 0.2), width=0.4)
            else:
                page.draw_rect(fitz.Rect(p_a.x - 0.2, min(p_a.y, p_b.y), p_a.x + 0.2, max(p_a.y, p_b.y)), width=0.4)

        # Layout coordinates in visual 1191x842 space matching real F&G CAD sheet
        hdr_top, hdr_bot = 55.0, 275.0
        row_h = 8.0
        col_w = 14.0
        grid_x0 = 400.0
        grid_x1 = grid_x0 + col_w * len(effects)
        body_bot = hdr_bot + row_h * len(causes)

        # Left cause table borders & headers
        left_x0, rev_x1, zone_x1, tagname_x1, tagno_x1, logic_x1 = 148.0, 162.0, 198.0, 322.0, 344.0, 378.0
        for bx in (left_x0, rev_x1, zone_x1, tagname_x1, tagno_x1, logic_x1, 386.0, grid_x0):
            vline(bx, hdr_bot if bx < 386 else hdr_top, body_bot)
        for idx in range(len(effects) + 1):
            vx = grid_x0 + idx * col_w
            # Omit inner border between MOV-00401 OPEN/CLOSE in middle description/tag band to simulate merged col
            if merged_pairs and idx == 2:
                vline(vx, hdr_top, 140.0)
                vline(vx, 250.0, body_bot)
            else:
                vline(vx, hdr_top, body_bot)

        hline(hdr_top, 386.0, grid_x1)
        hline(105.0, 386.0, grid_x1)
        hline(140.0, 386.0, grid_x1)
        hline(210.0, 386.0, grid_x1)
        hline(250.0, left_x0, grid_x1)
        hline(hdr_bot, left_x0, grid_x1)

        # Vertical header-label column (must be excluded from Effect columns)
        put_vert(393.0, 62.0, "REV")
        put_vert(393.0, 98.0, "P&ID Number")
        put_vert(393.0, 132.0, "FUNCTION PERFORMED")
        put_vert(393.0, 185.0, "TAG NAME")
        put_vert(393.0, 238.0, "DEVICE ID")

        # Horizontal Cause headers
        put_horiz(151.0, 265.0, "REV")
        put_horiz(165.0, 265.0, "FIRE ZONE NO")
        put_horiz(245.0, 270.0, "TAG NAME")
        put_horiz(326.0, 270.0, "TAG NO")
        put_horiz(350.0, 270.0, "LOGIC")

        # Effect column headers (14pt wide)
        for idx, (pid, action, desc, etag) in enumerate(effects):
            cx = grid_x0 + (idx + 0.5) * col_w
            if pid:
                put_vert(cx, 100.0, pid, fs=3.0)
            if action:
                put_vert(cx, 134.0, action, fs=3.2)
            if desc:
                dx = grid_x0 + (idx + 1.0) * col_w if (merged_pairs and idx == 1) else cx
                if not (merged_pairs and idx == 2):
                    put_vert(dx, 200.0, desc, fs=3.2)
            if etag and not (merged_pairs and idx == 2):
                tx = grid_x0 + (idx + 1.0) * col_w if (merged_pairs and idx == 1) else cx
                put_vert(tx, 238.0, etag, fs=3.2)

        # Cause rows + matrix marks
        for r_i, c in enumerate(causes):
            ry0 = hdr_bot + r_i * row_h
            ry1 = ry0 + row_h
            ry_mid = (ry0 + ry1) / 2.0 + 1.0
            # Fine horizontal divider in TAG NAME / TAG NO (plus REV)
            hline(ry1, left_x0, rev_x1)
            hline(ry1, zone_x1, tagno_x1)
            hline(ry1, logic_x1, grid_x0)
            # Only draw horizontal divider across LOGIC & Matrix when not vertically merged with next row
            if not c.get("merge_next_with_prev"):
                hline(ry0, tagno_x1, grid_x1)
                if not c.get("merge_zone_prev"):
                    hline(ry0, rev_x1, zone_x1)
            put_horiz(153.0, ry_mid, "2", fs=3.0)
            if c.get("zone"):
                put_horiz(170.0, ry_mid + 3.0, c["zone"], fs=3.2)
            put_horiz(202.0, ry_mid, c["desc"], fs=3.0)
            put_horiz(324.0, ry_mid, c["tag"], fs=3.0)
            if c.get("voting"):
                if c["voting"] == ">= 2":
                    put_horiz(349.0, ry_mid + 2.0, ">=", fs=3.0)
                    put_horiz(355.0, ry_mid + 2.0, "2", fs=3.0)
                else:
                    put_horiz(352.0, ry_mid, c["voting"], fs=3.0)
            put_horiz(388.0, ry_mid, str(r_i + 1), fs=3.0)
            for col_i, mark_sym in c.get("marks", {}).items():
                mx = grid_x0 + (col_i + 0.5) * col_w - 1.0
                my = ry_mid + (2.0 if c.get("merged_mark_y_offset") else 0.0)
                put_horiz(mx, my, mark_sym, fs=3.4)

        hline(body_bot, left_x0, grid_x1)

        if with_notes:
            ny = body_bot + 10.0
            put_horiz(left_x0, ny, "Note.", fs=3.5)
            put_horiz(left_x0, ny + 8.0, "2 : Confirmed Fire in FZ-006 (Flare KOD) will close PCV-80101.", fs=3.5)

    # Page 2: Dense F&G matrix with merged FZ-001 zone, H vs HH, >= 2 + X merged rows, MOV-00401 OPEN/CLOSE, NOTE 2
    _add_rotated_matrix_page(
        effects=[
            ("", "ALARM", "FZ001 JETTY AREA GAS COMMON ALARM", "GA-FZ001"),
            ("195363-07-PID-0004", "OPEN", "JETTY AREA MOTOR OPERATED VALVE", "MOV-00401"),
            ("195363-07-PID-0004", "CLOSE", "", ""),
            ("PID-0106", "Close (NOTE 2)", "Flare PCV", "PCV-80101"),
        ],
        causes=[
            {"zone": "FZ-001", "desc": "GAS DETECTOR JETTY TRESTLE AREA", "tag": "GSD-10201", "voting": "H", "marks": {}},
            {"merge_zone_prev": True, "desc": "GAS DETECTOR JETTY TRESTLE AREA", "tag": "GSD-10201", "voting": "HH", "marks": {0: "x"}},
            {"merge_zone_prev": True, "desc": "FLAME DETECTOR UNLOADING ARM", "tag": "FSD-10101", "voting": ">= 2", "marks": {3: "X"}},
            {"merge_zone_prev": True, "merge_next_with_prev": True, "desc": "FLAME DETECTOR UNLOADING ARM", "tag": "FSD-10102", "marks": {}},
            {"merge_zone_prev": True, "desc": "SOFT SWITCH MOV OPEN", "tag": "HS-00401MA", "voting": "ON", "marks": {1: "x"}},
            {"merge_zone_prev": True, "desc": "SOFT SWITCH MOV CLOSE", "tag": "HS-00401MB", "voting": "ON", "marks": {2: "x"}},
        ],
        merged_pairs=True,
        with_notes=True,
    )

    # Page 3: Continuation sheet where GA-FZ001 has 0 marks and GA-FZ005 has 1 mark
    _add_rotated_matrix_page(
        effects=[
            ("", "ALARM", "FZ001 JETTY AREA GAS COMMON ALARM", "GA-FZ001"),
            ("", "ALARM", "FZ005 LNG TANK AREA GAS COMMON ALARM", "GA-FZ005"),
        ],
        causes=[
            {"zone": "FZ-005", "desc": "GAS DETECTOR LNG TANK AREA", "tag": "GSD-20101", "voting": "HH", "marks": {1: "x"}},
            {"merge_zone_prev": True, "desc": "GAS DETECTOR LNG TANK AREA", "tag": "GSD-20102", "voting": "HH", "marks": {1: "x"}},
            {"merge_zone_prev": True, "desc": "GAS DETECTOR LNG TANK AREA", "tag": "GSD-20103", "voting": "HH", "marks": {}},
            {"merge_zone_prev": True, "desc": "GAS DETECTOR LNG TANK AREA", "tag": "GSD-20104", "voting": "HH", "marks": {}},
        ],
    )

    doc.save(str(path))
    doc.close()
    return path


def test_rotated_fire_and_gas_ce_matrix_grid_and_continuation(tmp_path: Path):
    """Verify /Rotate 90 F&G C&E extraction: cover classification, merged zones/voting/actions, lowercase 'x', notes, and continuation pruning."""
    merged_unicode = _merge_adjacent_voting_spans(
        [
            {"text": "≥", "bbox": [350.0, 297.0, 351.8, 301.0]},
            {"text": "2", "bbox": [352.5, 297.0, 355.0, 301.0]},
        ]
    )
    assert len(merged_unicode) == 1 and merged_unicode[0]["text"] == "≥ 2"

    pdf_path = _build_rotated_fgs_ce_pdf(tmp_path / "rotated_fgs_ce.pdf")
    res = extract_engineering_pdf(pdf_path)
    assert res is not None
    profile = res["engineering_profile"]

    assert profile["page_modes"][1] != "matrix_rotated_headers"
    assert profile["page_modes"][2] == "matrix_rotated_headers"
    assert profile["page_modes"][3] == "matrix_rotated_headers"
    assert "NOTE 2" in profile["legend_context"]["notes_map"]

    p2_recs = profile["pages"][1]["records"]
    p2_effects = [r for r in p2_recs if r["record_type"] == "matrix_effect_col"]
    p2_causes = [r for r in p2_recs if r["record_type"] == "matrix_cause_row"]

    assert len(p2_causes) == 6
    # Header-label column must not be emitted as an Effect column
    assert all(r["metadata"]["effect_tag"] not in ("REV", "DEVICE ID", "TAG NAME") for r in p2_effects)

    # Lowercase 'x' marks trigger GA-FZ001 ONLY from [HH] (not [H]) without row description bleeding
    ga_eff = next(r for r in p2_effects if r["metadata"]["effect_tag"] == "GA-FZ001")
    ga_causes = [c["cause_tag"] for c in ga_eff["metadata"]["causes"]]
    assert ga_causes == ["GSD-10201"]
    assert ga_eff["metadata"]["causes"][0]["signal_level"] == "HH"
    assert ga_eff["metadata"]["causes"][0]["group"] == "FZ-001"
    assert ga_eff["metadata"]["causes"][0]["description"] == "GAS DETECTOR JETTY TRESTLE AREA"

    # Merged '>= 2' + 'X' row propagates to BOTH FSD-10101 and FSD-10102, and inlines NOTE 2 on PCV-80101
    pcv_eff = next(r for r in p2_effects if r["metadata"]["effect_tag"] == "PCV-80101")
    assert sorted(c["cause_tag"] for c in pcv_eff["metadata"]["causes"]) == ["FSD-10101", "FSD-10102"]
    assert all(c["voting"] == ">= 2" for c in pcv_eff["metadata"]["causes"])
    assert all(c["description"] == "FLAME DETECTOR UNLOADING ARM" for c in pcv_eff["metadata"]["causes"])
    assert any("NOTE 2:" in n for n in pcv_eff["metadata"]["inlined_notes"])

    # Horizontally merged MOV-00401 header splits OPEN (HS-00401MA) vs CLOSE (HS-00401MB)
    mov_open = next(r for r in p2_effects if r["metadata"]["effect_tag"] == "MOV-00401" and r["metadata"]["action"] == "OPEN")
    mov_close = next(r for r in p2_effects if r["metadata"]["effect_tag"] == "MOV-00401" and r["metadata"]["action"] == "CLOSE")
    assert [c["cause_tag"] for c in mov_open["metadata"]["causes"]] == ["HS-00401MA"]
    assert [c["cause_tag"] for c in mov_close["metadata"]["causes"]] == ["HS-00401MB"]

    # Page 3 continuation prunes inactive GA-FZ001 while retaining active GA-FZ005
    p3_effects = [r for r in profile["pages"][2]["records"] if r["record_type"] == "matrix_effect_col"]
    p3_effect_tags = [r["metadata"]["effect_tag"] for r in p3_effects]
    assert "GA-FZ001" not in p3_effect_tags
    assert "GA-FZ005" in p3_effect_tags

