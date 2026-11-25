"""Per-page CAD geometry profiler and Legend/Notes context extraction."""
from __future__ import annotations

import re
from typing import Any, Sequence

import pymupdf as fitz

from src.ingestion.engineering.geometry import (
    _count_rotated_spans,
    _estimate_column_bands,
    _get_visual_drawings,
    _get_visual_text_dict,
    _get_visual_text_lines,
    _get_visual_words,
)
from src.ingestion.engineering.models import (
    ACTION_MARKS,
    EngineeringPageRecord,
    LegendContext,
    PageExtractionResult,
)
from src.ingestion.engineering.patterns import (
    _LOGIC_GATE_RE,
    _NOTE_DEF_START_RE,
    _NOTE_NUM_COLON_RE,
    _NOTE_SECTION_RE,
    _PAGE_NUM_FOOTER_RE,
    _SYMBOL_DEF_INLINE_RE,
    _SYMBOL_DEF_LINE_RE,
    _extract_record_tags,
)


def profile_pdf_page(page: fitz.Page) -> tuple[str, dict[str, Any]]:
    """Classify a single PyMuPDF page in <5ms into one of 6 deterministic geometry modes."""
    rect = page.rect
    width = float(rect.width)
    height = float(rect.height)

    words = _get_visual_words(page)
    drawings = _get_visual_drawings(page)
    text_dict = _get_visual_text_dict(page)
    raw_text = _get_visual_text_lines(page)

    word_count = len(words)
    drawing_count = len(drawings)
    rotated_span_count = _count_rotated_spans(text_dict)
    single_char_mark_count = sum(1 for w in words if str(w[4]).strip().upper() in ACTION_MARKS)
    column_band_count, body_row_count = _estimate_column_bands(words, drawings)

    metrics: dict[str, Any] = {
        "word_count": word_count,
        "drawing_count": drawing_count,
        "rotated_span_count": rotated_span_count,
        "single_char_mark_count": single_char_mark_count,
        "column_band_count": column_band_count,
        "body_row_count": body_row_count,
        "width": round(width, 2),
        "height": round(height, 2),
    }

    # 1. Exploded-SHX / zero-text vector CAD sheet
    if word_count < 30 and drawing_count > 300:
        return "exploded_cad_ocr", metrics

    # 2. Rotated-header matrix (Cause & Effect / Interlock matrix)
    has_matrix_cues = bool(
        re.search(r"\b(?:CAUSE|EFFECT|ESD|PID|INTERLOCK)\b", raw_text, re.IGNORECASE)
    )
    if rotated_span_count >= 10 and (single_char_mark_count >= 2 or has_matrix_cues):
        return "matrix_rotated_headers", metrics

    # 3. Document Legend & Notes page (must not be a full matrix or wide schedule)
    symbol_def_count = len(_SYMBOL_DEF_INLINE_RE.findall(raw_text))
    note_def_count = sum(
        1 for line in raw_text.splitlines() if _NOTE_DEF_START_RE.match(line)
    )
    has_legend_heading = bool(re.search(r"\bLEGEND\b", raw_text, re.IGNORECASE))
    if (
        rotated_span_count < 10
        and column_band_count < 6
        and (
            symbol_def_count >= 2
            or (has_legend_heading and (symbol_def_count >= 1 or note_def_count >= 1))
        )
    ):
        return "document_legend_notes", metrics

    # 4. CAD Logic / FBD Diagram sheet
    has_strong_logic_cue = bool(
        re.search(
            r"\b(?:LTRCV|LTFCS|MAIN_\d+|ESD\s+LOGIC|LOGIC\s+DIAGRAM|POU)\b|\(\*\s*[^*]+\*\)",
            raw_text,
            re.IGNORECASE,
        )
    )
    logic_gate_hits = len(set(_LOGIC_GATE_RE.findall(raw_text)))
    has_from_to = bool(re.search(r"\b(?:From|To)\s*:", raw_text))
    is_landscape_or_boxed = (width > height * 1.15) or (drawing_count > 40)
    if is_landscape_or_boxed and (
        has_strong_logic_cue
        or (logic_gate_hits >= 2 and (has_from_to or column_band_count < 6))
    ):
        return "cad_logic_diagram", metrics

    # 5. Coordinate-grid wide schedule (Cabinet Wiring / I/O List / Instrument Index)
    if column_band_count >= 6 and body_row_count >= 4:
        return "coord_grid_schedule", metrics

    # 6. Default standard prose fallback
    return "standard_prose", metrics


def extract_legend_context(
    doc_or_pages: fitz.Document | Sequence[fitz.Page],
    page_modes: dict[int | str, str] | Sequence[str] | None = None,
) -> LegendContext:
    """Scan legend/notes pages and matrix note blocks to build a LegendContext."""
    pages: list[fitz.Page] = (
        [doc_or_pages[i] for i in range(len(doc_or_pages))]
        if isinstance(doc_or_pages, fitz.Document)
        else list(doc_or_pages)
    )

    symbol_map: dict[str, str] = {}
    notes_map: dict[str, str] = {}
    source_pages: list[int] = []
    raw_chunks: list[str] = []

    for idx, page in enumerate(pages, start=1):
        mode: str | None = None
        if isinstance(page_modes, dict):
            mode = page_modes.get(idx) or page_modes.get(str(idx))
        elif isinstance(page_modes, Sequence) and idx - 1 < len(page_modes):
            mode = page_modes[idx - 1]
        if mode is None:
            mode, _ = profile_pdf_page(page)

        if mode not in ("document_legend_notes", "matrix_rotated_headers"):
            continue

        text = _get_visual_text_lines(page)
        if not text.strip():
            continue

        page_found = False
        current_note_key: str | None = None
        in_notes_section = False

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                current_note_key = None
                continue

            if _NOTE_SECTION_RE.match(line):
                in_notes_section = True
                current_note_key = None
                continue

            if _PAGE_NUM_FOOTER_RE.match(line):
                in_notes_section = False
                current_note_key = None
                continue

            sym_match = _SYMBOL_DEF_LINE_RE.match(line)
            if sym_match and (
                mode == "document_legend_notes"
                or (mode == "matrix_rotated_headers" and len(sym_match.group(2).strip()) >= 3)
            ):
                sym_key = sym_match.group(1).upper()
                sym_val = sym_match.group(2).strip()
                if sym_key not in symbol_map:
                    symbol_map[sym_key] = sym_val
                page_found = True
                current_note_key = None
                in_notes_section = False
                continue

            note_match = _NOTE_DEF_START_RE.match(line)
            if note_match:
                note_num = note_match.group(2)
                note_key = f"NOTE {note_num}"
                note_body = note_match.group(3).strip()
                if note_key not in notes_map:
                    notes_map[note_key] = note_body
                current_note_key = note_key
                page_found = True
                in_notes_section = False
                continue

            if in_notes_section:
                colon_note_match = _NOTE_NUM_COLON_RE.match(line)
                if colon_note_match:
                    note_num = colon_note_match.group(1)
                    note_key = f"NOTE {note_num}"
                    note_body = colon_note_match.group(2).strip()
                    if note_key not in notes_map:
                        notes_map[note_key] = note_body
                    current_note_key = note_key
                    page_found = True
                    continue

            # Append wrapped continuation lines to the active note (including ALL-CAPS CAD notes)
            if (
                current_note_key
                and not line.startswith(("DOC:", "REV:", "SHEET:", "LEGEND", "NOTES:"))
            ):
                notes_map[current_note_key] = f"{notes_map[current_note_key]} {line}".strip()

        if page_found or mode == "document_legend_notes":
            source_pages.append(idx)
            raw_chunks.append(text.strip())

    return LegendContext(
        symbol_map=symbol_map,
        notes_map=notes_map,
        source_pages=source_pages,
        raw_text="\n\n".join(raw_chunks),
    )


def _build_legend_page_result(
    page: fitz.Page,
    page_number: int,
    metrics: dict[str, Any],
    legend_ctx: LegendContext,
) -> PageExtractionResult:
    """Create a PageExtractionResult and EngineeringPageRecord for a Legend & Notes page."""
    raw_text = _get_visual_text_lines(page).strip()
    first_line = next((ln.strip() for ln in raw_text.splitlines() if ln.strip()), f"Legend & Notes (Page {page_number})")
    summary = legend_ctx.format_summary() or raw_text
    tags = _extract_record_tags(f"{summary}\n{raw_text}")

    record = EngineeringPageRecord(
        record_id=f"p{page_number}_legend_1",
        record_type="document_legend_notes",
        page_number=page_number,
        title=first_line,
        summary_text=summary,
        tags=tags,
        bbox=[0.0, 0.0, round(float(page.rect.width), 2), round(float(page.rect.height), 2)],
        metadata={
            "symbol_map": legend_ctx.symbol_map,
            "notes_map": legend_ctx.notes_map,
        },
    )

    md_parts = [f"<!-- page: {page_number} -->", f"# {first_line} (Page {page_number})"]
    if legend_ctx.symbol_map:
        md_parts.append("## Legend Symbols\n" + "\n".join(f"- **{k}**: {v}" for k, v in legend_ctx.symbol_map.items()))
    if legend_ctx.notes_map:
        md_parts.append("## Notes\n" + "\n".join(f"- **{k}**: {v}" for k, v in legend_ctx.notes_map.items()))
    if not legend_ctx.symbol_map and not legend_ctx.notes_map and raw_text:
        md_parts.append(raw_text)

    return PageExtractionResult(
        page_number=page_number,
        mode="document_legend_notes",
        metrics=metrics,
        records=[record],
        markdown="\n\n".join(md_parts).strip() + "\n",
        legend_context=legend_ctx,
    )
