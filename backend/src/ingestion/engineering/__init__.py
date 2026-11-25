"""Engineering CAD PDF profiling and deterministic extraction subpackage."""
from src.ingestion.engineering.ce_matrix import (
    _merge_adjacent_voting_spans,
    extract_rotated_ce_matrix,
)
from src.ingestion.engineering.geometry import _get_visual_text_lines
from src.ingestion.engineering.logic_diagram import extract_cad_logic_sheet
from src.ingestion.engineering.models import (
    ACTION_MARKS,
    EngineeringPageRecord,
    LegendContext,
    PageExtractionResult,
)
from src.ingestion.engineering.ocr import (
    compute_bbox_iou,
    compute_safe_ocr_dpi,
    deduplicate_ocr_detections,
    extract_exploded_cad_ocr,
    get_rapidocr_engine,
    reset_rapidocr_engine,
)
from src.ingestion.engineering.patterns import _extract_record_tags
from src.ingestion.engineering.pipeline import extract_engineering_pdf
from src.ingestion.engineering.profiler import (
    _build_legend_page_result,
    extract_legend_context,
    profile_pdf_page,
)
from src.ingestion.engineering.schedule import extract_coord_grid_schedule

__all__ = [
    "ACTION_MARKS",
    "EngineeringPageRecord",
    "LegendContext",
    "PageExtractionResult",
    "_build_legend_page_result",
    "_extract_record_tags",
    "_get_visual_text_lines",
    "_merge_adjacent_voting_spans",
    "compute_bbox_iou",
    "compute_safe_ocr_dpi",
    "deduplicate_ocr_detections",
    "extract_cad_logic_sheet",
    "extract_coord_grid_schedule",
    "extract_engineering_pdf",
    "extract_exploded_cad_ocr",
    "extract_legend_context",
    "extract_rotated_ce_matrix",
    "get_rapidocr_engine",
    "profile_pdf_page",
    "reset_rapidocr_engine",
]
