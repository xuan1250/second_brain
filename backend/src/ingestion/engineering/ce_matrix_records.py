"""Cause row parsing, mark intersection, and bidirectional record assembly for C&E sheets."""
from __future__ import annotations

import re
from typing import Any, Sequence

from src.ingestion.engineering.ce_matrix_grid import (
    _collect_note_refs_from_texts,
    _merge_adjacent_voting_spans,
)
from src.ingestion.engineering.geometry import (
    _bbox_mid_x,
    _bbox_mid_y,
    _bbox_union,
    _cluster_1d,
    _find_cell_span_1d,
    _merge_close_coords,
)
from src.ingestion.engineering.models import (
    EngineeringPageRecord,
    LegendContext,
)
from src.ingestion.engineering.patterns import (
    _FIRE_ZONE_GROUP_RE,
    _GROUP_CODE_RE,
    _NOTE_REF_RE,
    _PID_CELL_RE,
    _PID_RE,
    _SIGNAL_LEVEL_RE,
    _TAG_TOKEN_RE,
    _VOTING_RE,
    _extract_record_tags,
)


def _parse_cause_row_spans(
    spans: Sequence[dict[str, Any]],
    row_idx: int,
    y_center: float,
    *,
    rev_x_range: tuple[float, float] | None = None,
    zone_x_range: tuple[float, float] | None = None,
    row_no_x_min: float | None = None,
) -> dict[str, Any]:
    """Classify horizontal spans belonging to one left-hand C&E Cause row."""
    sorted_spans = sorted(spans, key=lambda s: (float(s["bbox"][0]), float(s["bbox"][1])))
    texts = [str(s["text"]).strip() for s in sorted_spans if str(s["text"]).strip()]

    group_spans: list[dict[str, Any]] = []
    pid = ""
    cause_tag = ""
    voting = ""
    signal_level = ""
    note_refs: list[str] = []
    desc_spans: list[dict[str, Any]] = []

    for sp in sorted_spans:
        txt = str(sp["text"]).strip()
        if not txt:
            continue
        xm = _bbox_mid_x(sp["bbox"])

        if rev_x_range is not None and rev_x_range[0] <= xm <= rev_x_range[1] and re.fullmatch(r"[A-Z0-9]{1,3}", txt, re.I):
            continue
        if row_no_x_min is not None and xm >= row_no_x_min and re.fullmatch(r"\d{1,3}", txt):
            continue
        if zone_x_range is not None and zone_x_range[0] <= xm <= zone_x_range[1]:
            group_spans.append(sp)
            continue
        if _GROUP_CODE_RE.fullmatch(txt) or _FIRE_ZONE_GROUP_RE.fullmatch(txt):
            group_spans.append(sp)
            continue
        if _PID_CELL_RE.fullmatch(txt):
            pid = txt.upper()
            continue
        if _VOTING_RE.fullmatch(txt):
            voting = txt
            continue
        if _SIGNAL_LEVEL_RE.fullmatch(txt) and xm > (zone_x_range[1] if zone_x_range else 0.0):
            signal_level = txt.upper()
            continue
        note_match = _NOTE_REF_RE.fullmatch(txt)
        if note_match or (txt.upper().startswith("NOTE") and _NOTE_REF_RE.search(txt)):
            nm = _NOTE_REF_RE.search(txt)
            if nm:
                note_refs.append(f"NOTE {nm.group(1)}")
            continue
        if not cause_tag and _TAG_TOKEN_RE.fullmatch(txt) and not _PID_CELL_RE.fullmatch(txt):
            cause_tag = txt
            continue
        desc_spans.append(sp)

    group_ordered = sorted(group_spans, key=lambda s: (float(s["bbox"][1]), float(s["bbox"][0])))
    group = " ".join(str(s["text"]).strip() for s in group_ordered).strip()

    desc_spans_ordered = sorted(desc_spans, key=lambda s: (float(s["bbox"][1]), float(s["bbox"][0])))
    description = " ".join(str(s["text"]).strip() for s in desc_spans_ordered).strip()

    if not cause_tag:
        for txt in texts:
            m = _TAG_TOKEN_RE.search(txt)
            if (
                m
                and not _PID_CELL_RE.fullmatch(m.group(0))
                and not _PID_RE.fullmatch(m.group(0))
                and not _GROUP_CODE_RE.fullmatch(m.group(0))
                and not _FIRE_ZONE_GROUP_RE.fullmatch(m.group(0))
            ):
                cause_tag = m.group(0)
                break
    if not cause_tag:
        cause_tag = description or pid or f"CAUSE_ROW_{row_idx + 1}"
    note_refs = _collect_note_refs_from_texts(texts, note_refs)

    return {
        "row_index": row_idx + 1,
        "y_center": round(y_center, 2),
        "group": group,
        "pid": pid,
        "cause_tag": cause_tag,
        "description": description,
        "voting": voting,
        "signal_level": signal_level,
        "note_refs": note_refs,
        "raw_labels": texts,
        "bbox": _bbox_union([s["bbox"] for s in sorted_spans]),
    }


def _build_ce_cause_rows(
    left_spans: Sequence[dict[str, Any]],
    h_segs: Sequence[tuple[float, float, float]],
    *,
    matrix_x_min: float,
    geom: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build Cause rows from fine horizontal grid lines (propagating merged cells) or y-clustering."""
    if not left_spans:
        return []

    left_spans = _merge_adjacent_voting_spans(left_spans)
    body_y_min = float(geom["body_y_min"])
    body_y_max = float(geom["body_y_max"])
    zone_x_range = geom["zone_x_range"]
    rev_x_range = geom["rev_x_range"]
    row_no_x_min = geom["row_no_x_min"]

    tag_probe_x = (zone_x_range[1] + 45.0) if zone_x_range else (matrix_x_min * 0.55)
    probe_h_ys = [
        y
        for y, x0, x1 in h_segs
        if (body_y_min - 8.0) <= y <= (body_y_max + 2.0)
        and (x1 - x0) >= 40.0
        and x0 <= tag_probe_x <= x1
    ]
    merged_ry = _merge_close_coords(probe_h_ys, tol=1.5)
    fine_intervals = [
        (merged_ry[i], merged_ry[i + 1])
        for i in range(len(merged_ry) - 1)
        if 4.5 <= (merged_ry[i + 1] - merged_ry[i]) <= 32.0
    ]

    if len(fine_intervals) >= 4:
        row_buckets: list[list[dict[str, Any]]] = [[] for _ in fine_intervals]
        for sp in left_spans:
            sx = _bbox_mid_x(sp["bbox"])
            sy = _bbox_mid_y(sp["bbox"])
            cell_yr = _find_cell_span_1d(sy, sx, h_segs, max_span=400.0)
            if cell_yr is not None:
                cy0, cy1 = cell_yr
                covered = [
                    idx
                    for idx, (ry0, ry1) in enumerate(fine_intervals)
                    if (cy0 - 0.8) <= ((ry0 + ry1) / 2.0) <= (cy1 + 0.8)
                ]
                if covered:
                    for idx in covered:
                        row_buckets[idx].append(sp)
                    continue
            for idx, (ry0, ry1) in enumerate(fine_intervals):
                if (ry0 - 1.0) <= sy <= (ry1 + 1.0):
                    row_buckets[idx].append(sp)
                    break

        cause_rows: list[dict[str, Any]] = []
        for (ry0, ry1), r_spans in zip(fine_intervals, row_buckets):
            if not r_spans:
                continue
            joined_upper = " ".join(str(s["text"]).upper() for s in r_spans)
            if "CAUSE DESCRIPTION" in joined_upper or "CAUSE TAG" in joined_upper:
                continue
            row = _parse_cause_row_spans(
                r_spans,
                len(cause_rows),
                (ry0 + ry1) / 2.0,
                rev_x_range=rev_x_range,
                zone_x_range=zone_x_range,
                row_no_x_min=row_no_x_min,
            )
            if not row["description"] and row["cause_tag"].startswith("CAUSE_ROW_"):
                continue
            row["y0"] = round(ry0, 2)
            row["y1"] = round(ry1, 2)
            cause_rows.append(row)
        if cause_rows:
            return cause_rows

    left_y_mids = [_bbox_mid_y(s["bbox"]) for s in left_spans]
    row_clusters = sorted(
        _cluster_1d(left_y_mids, tol=14.0),
        key=lambda grp: sum(left_y_mids[i] for i in grp) / len(grp),
    )
    cause_rows = []
    for grp in row_clusters:
        r_spans = [left_spans[i] for i in grp]
        y_center = sum(left_y_mids[i] for i in grp) / len(grp)
        joined_upper = " ".join(str(s["text"]).upper() for s in r_spans)
        if "CAUSE DESCRIPTION" in joined_upper or "CAUSE TAG" in joined_upper:
            continue
        cause_rows.append(
            _parse_cause_row_spans(
                r_spans,
                len(cause_rows),
                y_center,
                rev_x_range=rev_x_range,
                zone_x_range=zone_x_range,
                row_no_x_min=row_no_x_min,
            )
        )
    return cause_rows


def _match_effect_col(
    mx: float,
    effect_cols: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """Match a mark x-coordinate to its Effect column via interval containment or tight nearest center."""
    for col in effect_cols:
        if (float(col["x0"]) - 1.5) <= mx <= (float(col["x1"]) + 1.5):
            return col
    nearest = min(effect_cols, key=lambda c: abs(mx - c["x_center"]))
    max_half = max(16.0, min(38.0, (float(nearest["x1"]) - float(nearest["x0"])) * 0.75))
    return nearest if abs(mx - nearest["x_center"]) <= max_half else None


def _match_cause_rows(
    mx: float,
    my: float,
    cause_rows: Sequence[dict[str, Any]],
    h_segs: Sequence[tuple[float, float, float]],
    *,
    body_y_min: float,
    body_y_max: float,
    row_tol: float,
) -> list[dict[str, Any]]:
    """Match a mark coordinate to one or multiple merged Cause rows using cell horizontal borders."""
    cell_yr = _find_cell_span_1d(my, mx, h_segs, max_span=250.0)
    if cell_yr is not None:
        cy0, cy1 = cell_yr
        if cy0 >= (body_y_min - 8.0) and cy1 <= (body_y_max + 8.0):
            covered = [
                r
                for r in cause_rows
                if (cy0 - 0.8) <= float(r["y_center"]) <= (cy1 + 0.8)
            ]
            if covered:
                return covered

    nearest = min(cause_rows, key=lambda r: abs(my - float(r["y_center"])))
    if abs(my - float(nearest["y_center"])) <= row_tol:
        return [nearest]
    return []


def _intersect_matrix_marks(
    candidate_marks: Sequence[tuple[Any, ...]],
    effect_cols: Sequence[dict[str, Any]],
    cause_rows: Sequence[dict[str, Any]],
    h_segs: Sequence[tuple[float, float, float]],
    *,
    body_y_min: float,
    body_y_max: float,
    legend_context: LegendContext | None,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    """Intersect action mark coordinates with Effect columns and Cause rows."""
    row_hits: dict[int, list[dict[str, Any]]] = {r["row_index"]: [] for r in cause_rows}
    col_hits: dict[int, list[dict[str, Any]]] = {c["col_index"]: [] for c in effect_cols}

    if not effect_cols or not cause_rows:
        return row_hits, col_hits

    if len(cause_rows) >= 2:
        pitches = sorted(
            abs(cause_rows[i + 1]["y_center"] - cause_rows[i]["y_center"])
            for i in range(len(cause_rows) - 1)
        )
        row_tol = max(12.0, min(60.0, pitches[len(pitches) // 2] * 0.65))
    else:
        row_tol = 28.0

    seen_pairs: set[tuple[int, int]] = set()
    for mark_word in candidate_marks:
        mx = _bbox_mid_x(mark_word)
        my = _bbox_mid_y(mark_word)
        raw_sym = str(mark_word[4]).strip()
        norm_sym = raw_sym.upper()

        matched_col = _match_effect_col(mx, effect_cols)
        if matched_col is None:
            continue

        matched_rows = _match_cause_rows(
            mx,
            my,
            cause_rows,
            h_segs,
            body_y_min=body_y_min,
            body_y_max=body_y_max,
            row_tol=row_tol,
        )
        if not matched_rows:
            continue

        meaning = (
            legend_context.symbol_map.get(raw_sym)
            or legend_context.symbol_map.get(norm_sym)
            or ""
            if legend_context
            else ""
        )

        for m_row in matched_rows:
            pair = (m_row["row_index"], matched_col["col_index"])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            row_hits[m_row["row_index"]].append(
                {
                    "effect_tag": matched_col["effect_tag"],
                    "effect_description": matched_col["description"],
                    "pid": matched_col["pid"],
                    "action": matched_col["action"],
                    "mark": norm_sym,
                    "mark_meaning": meaning,
                    "notes": matched_col["note_refs"],
                }
            )
            col_hits[matched_col["col_index"]].append(
                {
                    "cause_tag": m_row["cause_tag"],
                    "group": m_row["group"],
                    "description": m_row["description"],
                    "pid": m_row["pid"],
                    "voting": m_row["voting"],
                    "signal_level": m_row.get("signal_level", ""),
                    "mark": norm_sym,
                    "mark_meaning": meaning,
                    "notes": m_row["note_refs"],
                }
            )

    return row_hits, col_hits


def _build_ce_record_and_section(
    *,
    record_id: str,
    record_type: str,
    page_number: int,
    title: str,
    md_heading: str,
    summary_lines: list[str],
    base_note_refs: Sequence[str],
    related_items: Sequence[dict[str, Any]],
    legend_context: LegendContext | None,
    bbox: list[float],
    metadata: dict[str, Any],
) -> tuple[EngineeringPageRecord, str]:
    """Assemble a Cause or Effect EngineeringPageRecord and its Markdown section with inlined notes."""
    all_note_refs = list(base_note_refs)
    for item in related_items:
        for nr in item["notes"]:
            if nr not in all_note_refs:
                all_note_refs.append(nr)
    inlined_notes = legend_context.lookup_notes(all_note_refs) if legend_context else []

    lines = list(summary_lines)
    if inlined_notes:
        lines.append("Applicable Notes: " + " | ".join(inlined_notes))
    summary = "\n".join(lines)

    rec = EngineeringPageRecord(
        record_id=record_id,
        record_type=record_type,
        page_number=page_number,
        title=title,
        summary_text=summary,
        tags=_extract_record_tags(summary),
        bbox=bbox,
        metadata={
            **metadata,
            "note_refs": all_note_refs,
            "inlined_notes": inlined_notes,
        },
    )
    return rec, f"### {md_heading}\n{summary}"
