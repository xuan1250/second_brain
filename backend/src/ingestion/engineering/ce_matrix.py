"""Rotated Cause & Effect matrix extraction and bidirectional record builder."""
from __future__ import annotations

from typing import Any

import pymupdf as fitz

from src.ingestion.engineering.ce_matrix_grid import (
    _build_ce_effect_columns,
    _collect_note_refs_from_texts,
    _detect_cause_table_geometry,
    _is_header_label_only_column,
    _merge_adjacent_voting_spans,
    _parse_effect_column_spans,
    _partition_matrix_spans,
)
from src.ingestion.engineering.ce_matrix_records import (
    _build_ce_cause_rows,
    _build_ce_record_and_section,
    _intersect_matrix_marks,
    _match_cause_rows,
    _match_effect_col,
    _parse_cause_row_spans,
)
from src.ingestion.engineering.geometry import (
    _bbox_mid_x,
    _bbox_mid_y,
    _extract_grid_segments,
    _get_visual_drawings,
    _get_visual_text_dict,
    _get_visual_words,
)
from src.ingestion.engineering.models import (
    ACTION_MARKS,
    EngineeringPageRecord,
    LegendContext,
    PageExtractionResult,
)
from src.ingestion.engineering.profiler import extract_legend_context, profile_pdf_page

__all__ = [
    "_build_ce_cause_rows",
    "_build_ce_effect_columns",
    "_build_ce_record_and_section",
    "_collect_note_refs_from_texts",
    "_detect_cause_table_geometry",
    "_intersect_matrix_marks",
    "_is_header_label_only_column",
    "_match_cause_rows",
    "_match_effect_col",
    "_merge_adjacent_voting_spans",
    "_parse_cause_row_spans",
    "_parse_effect_column_spans",
    "_partition_matrix_spans",
    "extract_rotated_ce_matrix",
]


def extract_rotated_ce_matrix(
    page: fitz.Page,
    page_number: int,
    legend_context: LegendContext | None = None,
    *,
    prune_empty_effects: bool = False,
    metrics: dict[str, Any] | None = None,
) -> PageExtractionResult:
    """Reconstruct bidirectional Cause->Effect and Effect<-Cause records from rotated C&E sheets."""
    if metrics is None:
        _, metrics = profile_pdf_page(page)
    text_dict = _get_visual_text_dict(page)
    words = _get_visual_words(page)
    drawings = _get_visual_drawings(page)
    v_segs, h_segs = _extract_grid_segments(drawings)

    if legend_context is None:
        legend_context = extract_legend_context([page], page_modes={1: "matrix_rotated_headers"})

    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    rotated_spans, horiz_spans = _partition_matrix_spans(text_dict, page_width, page_height)

    max_rot_y = max((float(s["bbox"][3]) for s in rotated_spans), default=280.0)
    effect_cols, label_col_x_max = _build_ce_effect_columns(rotated_spans, v_segs, max_rot_y)

    header_y_max = max((c["bbox"][3] for c in effect_cols), default=280.0)
    if effect_cols:
        has_grid_cols = label_col_x_max > 0.0 or any((c["x1"] - c["x0"]) < 35.0 for c in effect_cols)
        matrix_x_min = (
            min(c["x0"] for c in effect_cols)
            if has_grid_cols
            else (min(c["x_center"] for c in effect_cols) - 40.0)
        )
    else:
        matrix_x_min = page_width * 0.4

    geom = _detect_cause_table_geometry(
        horiz_spans,
        v_segs,
        matrix_x_min=matrix_x_min,
        header_y_max=header_y_max,
        page_height=page_height,
    )
    body_y_min = float(geom["body_y_min"])
    body_y_max = float(geom["body_y_max"])

    candidate_marks = [
        w
        for w in words
        if str(w[4]).strip().upper() in ACTION_MARKS
        and _bbox_mid_x(w) >= (matrix_x_min - 2.0)
        and (body_y_min - 2.0) <= _bbox_mid_y(w) <= (body_y_max + 2.0)
    ]

    left_spans = [
        s
        for s in horiz_spans
        if _bbox_mid_x(s["bbox"]) <= matrix_x_min + 4.0
        and float(s["bbox"][2]) <= matrix_x_min + 12.0
        and (body_y_min - 2.0) <= _bbox_mid_y(s["bbox"]) <= body_y_max
    ]
    cause_rows = _build_ce_cause_rows(
        left_spans,
        h_segs,
        matrix_x_min=matrix_x_min,
        geom=geom,
    )

    row_hits, col_hits = _intersect_matrix_marks(
        candidate_marks,
        effect_cols,
        cause_rows,
        h_segs,
        body_y_min=body_y_min,
        body_y_max=body_y_max,
        legend_context=legend_context,
    )

    sheet_title = f"Process Cause & Effect Matrix (Page {page_number})"
    for s in horiz_spans:
        if float(s["bbox"][1]) < header_y_max and "CAUSE" in s["text"].upper() and "EFFECT" in s["text"].upper():
            sheet_title = s["text"].strip()
            break

    records: list[EngineeringPageRecord] = []
    md_sections: list[str] = [
        f"<!-- page: {page_number} -->",
        f"# {sheet_title} (Page {page_number})",
    ]
    if legend_context and (legend_context.symbol_map or legend_context.notes_map):
        md_sections.append(legend_context.format_summary())

    md_sections.append("## Forward Cause-to-Effect Mapping")
    for crow in cause_rows:
        effects = row_hits.get(crow["row_index"], [])
        eff_strs: list[str] = []
        for eff in effects:
            meaning_part = f" ({eff['mark_meaning']})" if eff["mark_meaning"] else ""
            act_part = f" [{eff['action']}]" if eff["action"] else ""
            eff_strs.append(
                f"{eff['effect_tag']} ({eff['effect_description']}{act_part}) -> Mark '{eff['mark']}'{meaning_part}"
            )

        sig_lvl = crow.get("signal_level") or ""
        lvl_part = f" | LEVEL: {sig_lvl}" if sig_lvl else ""
        rec, md_sec = _build_ce_record_and_section(
            record_id=f"p{page_number}_cause_{crow['row_index']}",
            record_type="matrix_cause_row",
            page_number=page_number,
            title=f"Cause {crow['cause_tag']} ({crow['group']})",
            md_heading=f"Cause: {crow['cause_tag']}",
            summary_lines=[
                f"CAUSE TAG: {crow['cause_tag']} | GROUP: {crow['group']} | P&ID: {crow['pid']} | VOTING: {crow['voting'] or 'N/A'}{lvl_part}",
                f"Cause Description: {crow['description']}",
                f"Triggered Effects ({len(effects)}): " + ("; ".join(eff_strs) if eff_strs else "None"),
            ],
            base_note_refs=crow["note_refs"],
            related_items=effects,
            legend_context=legend_context,
            bbox=crow["bbox"],
            metadata={
                "cause_tag": crow["cause_tag"],
                "group": crow["group"],
                "pid": crow["pid"],
                "voting": crow["voting"],
                "signal_level": sig_lvl,
                "description": crow["description"],
                "effects": effects,
            },
        )
        records.append(rec)
        md_sections.append(md_sec)

    has_any_col_hits = any(bool(v) for v in col_hits.values())
    md_sections.append("## Reverse Effect-to-Cause Mapping")
    for ecol in effect_cols:
        causes = col_hits.get(ecol["col_index"], [])
        if prune_empty_effects and has_any_col_hits and not causes:
            continue

        cause_strs: list[str] = []
        for c_item in causes:
            meaning_part = f" ({c_item['mark_meaning']})" if c_item["mark_meaning"] else ""
            qualifiers = [q for q in (c_item.get("signal_level"), c_item.get("voting")) if q]
            vote_part = f" [{' '.join(qualifiers)}]" if qualifiers else ""
            cause_strs.append(
                f"{c_item['cause_tag']} ({c_item['group']}: {c_item['description']}{vote_part}) -> Mark '{c_item['mark']}'{meaning_part}"
            )

        rec, md_sec = _build_ce_record_and_section(
            record_id=f"p{page_number}_effect_{ecol['col_index']}",
            record_type="matrix_effect_col",
            page_number=page_number,
            title=f"Effect {ecol['effect_tag']} ({ecol['action'] or 'Action'})",
            md_heading=f"Effect: {ecol['effect_tag']}",
            summary_lines=[
                f"EFFECT TAG: {ecol['effect_tag']} | P&ID: {ecol['pid']} | ACTION: {ecol['action'] or 'N/A'}",
                f"Effect Description: {ecol['description']}",
                f"Triggering Causes ({len(causes)}): " + ("; ".join(cause_strs) if cause_strs else "None"),
            ],
            base_note_refs=ecol["note_refs"],
            related_items=causes,
            legend_context=legend_context,
            bbox=ecol["bbox"],
            metadata={
                "effect_tag": ecol["effect_tag"],
                "pid": ecol["pid"],
                "description": ecol["description"],
                "action": ecol["action"],
                "causes": causes,
            },
        )
        records.append(rec)
        md_sections.append(md_sec)

    return PageExtractionResult(
        page_number=page_number,
        mode="matrix_rotated_headers",
        metrics=metrics,
        records=records,
        markdown="\n\n".join(md_sections).strip() + "\n",
        legend_context=legend_context,
    )
