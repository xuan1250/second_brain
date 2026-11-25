"""Document-level engineering PDF profiling and per-page extraction pipeline."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pymupdf as fitz

from src.ingestion.engineering.ce_matrix import extract_rotated_ce_matrix
from src.ingestion.engineering.geometry import _get_visual_text_lines
from src.ingestion.engineering.logic_diagram import extract_cad_logic_sheet
from src.ingestion.engineering.models import PageExtractionResult
from src.ingestion.engineering.ocr import extract_exploded_cad_ocr
from src.ingestion.engineering.profiler import (
    _build_legend_page_result,
    extract_legend_context,
    profile_pdf_page,
)
from src.ingestion.engineering.schedule import extract_coord_grid_schedule


def extract_engineering_pdf(
    file_path: Path | str,
    *,
    ocr_engine: Any | None = None,
) -> dict[str, Any] | None:
    """Profile all PDF pages; return None for pure prose PDFs, else canonical engineering profile."""
    path = Path(file_path)
    doc = fitz.open(str(path))
    try:
        page_count = len(doc)
        if page_count == 0:
            return None

        page_modes: dict[int, str] = {}
        page_metrics: dict[int, dict[str, Any]] = {}
        for idx in range(page_count):
            mode, metrics = profile_pdf_page(doc[idx])
            page_modes[idx + 1] = mode
            page_metrics[idx + 1] = metrics

        if all(mode == "standard_prose" for mode in page_modes.values()):
            return None

        legend_ctx = extract_legend_context(doc, page_modes=page_modes)
        matrix_page_count = sum(1 for m in page_modes.values() if m == "matrix_rotated_headers")

        page_results: list[PageExtractionResult] = []
        for idx in range(page_count):
            pno = idx + 1
            page = doc[idx]
            mode = page_modes[pno]
            metrics = page_metrics[pno]
            if mode == "coord_grid_schedule":
                res = extract_coord_grid_schedule(page, pno, metrics=metrics)
            elif mode == "matrix_rotated_headers":
                res = extract_rotated_ce_matrix(
                    page,
                    pno,
                    legend_context=legend_ctx,
                    prune_empty_effects=(matrix_page_count >= 2),
                    metrics=metrics,
                )
            elif mode == "cad_logic_diagram":
                res = extract_cad_logic_sheet(page, pno, metrics=metrics)
            elif mode == "exploded_cad_ocr":
                res = extract_exploded_cad_ocr(page, pno, ocr_engine=ocr_engine, metrics=metrics)
            elif mode == "document_legend_notes":
                res = _build_legend_page_result(page, pno, metrics, legend_ctx)
            else:
                prose_text = _get_visual_text_lines(page).strip()
                md = f"<!-- page: {pno} -->\n{prose_text}\n" if prose_text else f"<!-- page: {pno} -->\n"
                res = PageExtractionResult(
                    page_number=pno,
                    mode="standard_prose",
                    metrics=metrics,
                    records=[],
                    markdown=md,
                )
            page_results.append(res)

        all_records = [rec for pr in page_results for rec in pr.records]
        full_markdown = "\n\n".join(pr.markdown.strip() for pr in page_results if pr.markdown.strip())

        return {
            "markdown": full_markdown,
            "page_count": page_count,
            "has_engineering_records": True,
            "engineering_profile": {
                "schema_version": "1.0",
                "page_modes": page_modes,
                "legend_context": asdict(legend_ctx) if legend_ctx else None,
                "pages": [
                    {
                        "page_number": pr.page_number,
                        "mode": pr.mode,
                        "metrics": pr.metrics,
                        "records": [asdict(r) for r in pr.records],
                    }
                    for pr in page_results
                ],
                "records": [asdict(r) for r in all_records],
            },
        }
    finally:
        doc.close()
