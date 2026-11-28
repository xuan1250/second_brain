"""Coordinate-grid schedule extraction (header detection, cell assignment, markdown table formatting)."""
from __future__ import annotations

from typing import Any, Sequence

import pymupdf as fitz

from src.ingestion.constants import format_markdown_table
from src.ingestion.engineering.geometry import (
    _bbox_mid_x,
    _bbox_mid_y,
    _bbox_union,
    _cluster_1d,
    _collect_table_grid_geometry,
    _get_visual_drawings,
    _get_visual_words,
)
from src.ingestion.engineering.models import EngineeringPageRecord, PageExtractionResult
from src.ingestion.engineering.patterns import _SCHEDULE_HEADER_KEYWORDS, _extract_record_tags
from src.ingestion.engineering.profiler import profile_pdf_page


def _extract_page_title_and_context(
    words: Sequence[tuple[Any, ...]],
    table_y_range: tuple[float, float] | None,
    page_height: float,
    default_title: str,
) -> tuple[str, list[str]]:
    """Extract drawing title and sheet metadata tokens from top banner / title block outside table."""
    outside_words: list[tuple[Any, ...]] = []
    for w in words:
        ym = _bbox_mid_y(w)
        if table_y_range is not None:
            if ym < table_y_range[0] - 2.0 or ym > table_y_range[1] + 2.0:
                outside_words.append(w)
        elif ym < page_height * 0.14 or ym > page_height * 0.86:
            outside_words.append(w)

    if not outside_words:
        return default_title, []

    y_mids = [_bbox_mid_y(w) for w in outside_words]
    lines: list[str] = []
    for grp in sorted(
        _cluster_1d(y_mids, tol=5.0),
        key=lambda g: sum(y_mids[i] for i in g) / len(g),
    ):
        row_w = sorted((outside_words[i] for i in grp), key=lambda w: float(w[0]))
        lines.append(" ".join(str(w[4]) for w in row_w).strip())

    title = default_title
    for ln in lines:
        if any(kw in ln.upper() for kw in ("DRAWING", "SCHEDULE", "LIST", "CHART", "CABINET")):
            title = ln
            break
    if title == default_title and lines:
        title = lines[0]

    context_tags = _extract_record_tags("\n".join(lines))
    return title, context_tags


def _fallback_column_intervals_from_words(
    words: Sequence[tuple[Any, ...]],
    header_y_mid: float,
) -> list[tuple[float, float]]:
    """Build [x0, x1] column intervals by merging header words and splitting midpoints."""
    header_words = sorted(
        (w for w in words if abs(_bbox_mid_y(w) - header_y_mid) <= 10.0),
        key=lambda w: float(w[0]),
    )
    if not header_words:
        return []

    groups: list[list[tuple[Any, ...]]] = [[header_words[0]]]
    for w in header_words[1:]:
        prev_x1 = max(float(item[2]) for item in groups[-1])
        if float(w[0]) - prev_x1 <= 12.0:
            groups[-1].append(w)
        else:
            groups.append([w])

    centers = [
        (min(float(item[0]) for item in grp) + max(float(item[2]) for item in grp)) / 2.0
        for grp in groups
    ]
    intervals: list[tuple[float, float]] = []
    for idx, c in enumerate(centers):
        left = 0.0 if idx == 0 else (centers[idx - 1] + c) / 2.0
        right = 1e6 if idx == len(centers) - 1 else (c + centers[idx + 1]) / 2.0
        intervals.append((left, right))
    return intervals


def _detect_schedule_headers(
    table_words: Sequence[tuple[Any, ...]],
    row_groups: Sequence[list[int]],
    y_mids: Sequence[float],
    sig_vx: Sequence[float],
) -> tuple[int, list[tuple[float, float]], list[str]]:
    """Detect header row index, column [x0, x1] intervals, and column header labels."""
    header_idx = 0
    best_score = -1
    for r_idx, grp in enumerate(row_groups[:4]):
        tokens_upper = {str(table_words[i][4]).upper() for i in grp}
        score = len(tokens_upper & _SCHEDULE_HEADER_KEYWORDS)
        if score > best_score:
            best_score = score
            header_idx = r_idx

    header_grp = row_groups[header_idx]
    header_y = sum(y_mids[i] for i in header_grp) / len(header_grp)

    if len(sig_vx) >= 7:
        col_intervals = [(sig_vx[i], sig_vx[i + 1]) for i in range(len(sig_vx) - 1)]
    else:
        col_intervals = _fallback_column_intervals_from_words(table_words, header_y)

    header_names: list[str] = []
    for c_idx, (cx0, cx1) in enumerate(col_intervals):
        col_words = sorted(
            (
                table_words[i]
                for i in header_grp
                if (cx0 - 2.0) <= _bbox_mid_x(table_words[i]) <= (cx1 + 2.0)
            ),
            key=lambda w: (float(w[1]), float(w[0])),
        )
        col_label = " ".join(str(w[4]) for w in col_words).strip()
        header_names.append(col_label or f"COL_{c_idx + 1}")

    return header_idx, col_intervals, header_names


def _assign_row_words_to_columns(
    row_words: Sequence[tuple[Any, ...]],
    col_intervals: Sequence[tuple[float, float]],
    header_names: Sequence[str],
) -> tuple[dict[str, str], list[str]]:
    """Assign row word tokens to column intervals and return (row_dict, ordered_row_cells)."""
    cells: list[list[tuple[Any, ...]]] = [[] for _ in col_intervals]
    for w in row_words:
        wx_mid = _bbox_mid_x(w)
        assigned = False
        for c_idx, (cx0, cx1) in enumerate(col_intervals):
            if (cx0 - 2.0) <= wx_mid <= (cx1 + 2.0):
                cells[c_idx].append(w)
                assigned = True
                break
        if not assigned and col_intervals:
            nearest_idx = min(
                range(len(col_intervals)),
                key=lambda idx: abs(wx_mid - (col_intervals[idx][0] + col_intervals[idx][1]) / 2.0),
            )
            cells[nearest_idx].append(w)

    row_dict: dict[str, str] = {}
    row_list: list[str] = []
    for col_name, cell_words in zip(header_names, cells):
        sorted_cell = sorted(cell_words, key=lambda w: (float(w[1]), float(w[0])))
        val = " ".join(str(w[4]) for w in sorted_cell).strip()
        row_list.append(val)
        if val:
            row_dict[col_name] = val
    return row_dict, row_list


def extract_coord_grid_schedule(
    page: fitz.Page,
    page_number: int,
    *,
    metrics: dict[str, Any] | None = None,
) -> PageExtractionResult:
    """Reconstruct wide coordinate-grid tables/schedules with exact column-header bindings."""
    if metrics is None:
        _, metrics = profile_pdf_page(page)
    words = _get_visual_words(page)
    drawings = _get_visual_drawings(page)
    page_height = float(page.rect.height)

    sig_vx, table_y_range = _collect_table_grid_geometry(drawings)
    title, sheet_context_tags = _extract_page_title_and_context(
        words,
        table_y_range,
        page_height,
        default_title=f"Coordinate Schedule (Page {page_number})",
    )

    if table_y_range is not None:
        table_words = [
            w
            for w in words
            if (table_y_range[0] - 2.0) <= _bbox_mid_y(w) <= (table_y_range[1] + 2.0)
        ]
    else:
        table_words = list(words)

    if not table_words:
        return PageExtractionResult(
            page_number=page_number,
            mode="coord_grid_schedule",
            metrics=metrics,
            records=[],
            markdown=f"<!-- page: {page_number} -->\n# {title} (Page {page_number})\n",
        )

    y_mids = [_bbox_mid_y(w) for w in table_words]
    row_groups = sorted(
        _cluster_1d(y_mids, tol=3.5),
        key=lambda grp: sum(y_mids[i] for i in grp) / len(grp),
    )

    header_idx, col_intervals, header_names = _detect_schedule_headers(
        table_words, row_groups, y_mids, sig_vx
    )

    records: list[EngineeringPageRecord] = []
    table_rows_vals: list[list[str]] = []

    for body_num, grp in enumerate(row_groups[header_idx + 1 :], start=1):
        row_words = [table_words[i] for i in grp]
        if len(row_words) < 2:
            continue

        row_dict, row_list = _assign_row_words_to_columns(row_words, col_intervals, header_names)
        if not row_dict:
            continue

        table_rows_vals.append(row_list)
        summary_text = " | ".join(f"{k}: {v}" for k, v in row_dict.items())
        row_tags = _extract_record_tags(summary_text)
        for st in sheet_context_tags:
            if st not in row_tags:
                row_tags.append(st)

        records.append(
            EngineeringPageRecord(
                record_id=f"p{page_number}_row_{body_num}",
                record_type="schedule_row",
                page_number=page_number,
                title=title,
                summary_text=summary_text,
                tags=row_tags,
                bbox=_bbox_union(row_words),
                metadata={
                    "row_index": body_num,
                    "drawing_title": title,
                    "sheet_context_tags": sheet_context_tags,
                    "columns": row_dict,
                },
            )
        )

    md_lines = [
        f"<!-- page: {page_number} -->",
        f"# {title} (Page {page_number})",
    ]
    if sheet_context_tags:
        md_lines.append(f"Sheet Context: {' | '.join(sheet_context_tags)}")
    md_lines.append("")
    if header_names and table_rows_vals:
        md_lines.append(format_markdown_table(header_names, table_rows_vals, escape_pipes=True))

    return PageExtractionResult(
        page_number=page_number,
        mode="coord_grid_schedule",
        metrics=metrics,
        records=records,
        markdown="\n".join(md_lines).strip() + "\n",
    )
