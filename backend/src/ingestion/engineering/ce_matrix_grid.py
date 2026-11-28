"""Grid geometry, span partitioning, and Effect column extraction for rotated Cause & Effect sheets."""
from __future__ import annotations

import re
from typing import Any, Sequence

from src.ingestion.engineering.geometry import (
    _bbox_mid_x,
    _bbox_mid_y,
    _bbox_union,
    _cluster_1d,
    _find_cell_span_1d,
    _merge_close_coords,
)
from src.ingestion.engineering.patterns import (
    _CAUSE_TABLE_HEADER_CUES,
    _CE_HEADER_LABEL_TOKENS,
    _EFFECT_ACTION_KEYWORDS,
    _NOTE_REF_RE,
    _NOTE_SECTION_RE,
    _PAGE_NUM_FOOTER_RE,
    _PID_CELL_RE,
    _PID_RE,
    _TAG_TOKEN_RE,
)


def _is_header_label_only_column(spans: Sequence[dict[str, Any]]) -> bool:
    """Return True if rotated spans in a column contain only header-label keywords or dashes."""
    norms: list[str] = []
    for s in spans:
        raw = str(s.get("text") or "").strip()
        if not raw or raw in ("-", "—", "|"):
            continue
        norms.append(re.sub(r"\s+", " ", raw.rstrip(".:")).upper())
    if not norms:
        return True
    return all(tok in _CE_HEADER_LABEL_TOKENS for tok in norms)


def _collect_note_refs_from_texts(texts: Sequence[str], existing: Sequence[str] = ()) -> list[str]:
    """Collect deduplicated 'NOTE N' references from whole or parenthesized span labels."""
    seen: set[str] = set()
    refs: list[str] = []
    for ref in existing:
        if ref and ref not in seen:
            seen.add(ref)
            refs.append(ref)
    for txt in texts:
        for note_num in _NOTE_REF_RE.findall(txt or ""):
            key = f"NOTE {note_num}"
            if key not in seen:
                seen.add(key)
                refs.append(key)
    return refs


def _parse_effect_column_spans(
    spans: Sequence[dict[str, Any]],
    col_idx: int,
    x_center: float,
    *,
    x0: float | None = None,
    x1: float | None = None,
) -> dict[str, Any]:
    """Classify rotated spans belonging to one vertical C&E Effect column."""
    sorted_spans = sorted(spans, key=lambda s: (float(s["bbox"][1]), float(s["bbox"][0])))
    texts = [
        str(s["text"]).strip()
        for s in sorted_spans
        if str(s["text"]).strip() and str(s["text"]).strip() not in ("-", "—")
    ]

    pid = ""
    effect_tag = ""
    action = ""
    note_refs: list[str] = []
    desc_parts: list[str] = []

    for txt in texts:
        if _NOTE_REF_RE.fullmatch(txt) or (txt.upper().startswith("NOTE") and _NOTE_REF_RE.search(txt)):
            note_match = _NOTE_REF_RE.search(txt)
            if note_match:
                note_refs.append(f"NOTE {note_match.group(1)}")
            continue
        if _PID_CELL_RE.fullmatch(txt):
            pid = txt.upper()
            continue
        if not effect_tag and _TAG_TOKEN_RE.fullmatch(txt) and not _PID_CELL_RE.fullmatch(txt):
            effect_tag = txt
            continue
        lower_txt = txt.lower()
        if not action and any(kw in lower_txt for kw in _EFFECT_ACTION_KEYWORDS) and len(txt.split()) <= 3:
            action = txt
            continue
        desc_parts.append(txt)

    if not effect_tag:
        for txt in texts:
            m = _TAG_TOKEN_RE.search(txt)
            if m and not _PID_CELL_RE.fullmatch(m.group(0)) and not _PID_RE.fullmatch(m.group(0)):
                effect_tag = m.group(0)
                break

    description = " ".join(desc_parts).strip()
    if not effect_tag:
        effect_tag = description or pid or f"EFFECT_COL_{col_idx + 1}"
    note_refs = _collect_note_refs_from_texts(texts, note_refs)

    return {
        "col_index": col_idx + 1,
        "x_center": round(x_center, 2),
        "x0": round(x0, 2) if x0 is not None else round(x_center - 14.0, 2),
        "x1": round(x1, 2) if x1 is not None else round(x_center + 14.0, 2),
        "pid": pid,
        "effect_tag": effect_tag,
        "description": description,
        "action": action,
        "note_refs": note_refs,
        "raw_labels": texts,
        "bbox": _bbox_union([s["bbox"] for s in sorted_spans]),
    }


def _build_ce_effect_columns(
    rotated_spans: Sequence[dict[str, Any]],
    v_segs: Sequence[tuple[float, float, float]],
    header_scan_y_max: float,
) -> tuple[list[dict[str, Any]], float]:
    """Group rotated Effect spans into columns, using vector column grid borders when present."""
    if not rotated_spans:
        return [], 0.0

    rot_x_mids = [_bbox_mid_x(s["bbox"]) for s in rotated_spans]
    probe_clusters = sorted(
        _cluster_1d(rot_x_mids, tol=8.0),
        key=lambda grp: sum(rot_x_mids[i] for i in grp) / len(grp),
    )
    label_col_x_max = 0.0
    data_spans: list[dict[str, Any]] = []
    for grp in probe_clusters:
        col_spans = [rotated_spans[i] for i in grp]
        if _is_header_label_only_column(col_spans) and float(col_spans[0]["bbox"][2]) < 430.0:
            label_col_x_max = max(label_col_x_max, max(float(s["bbox"][2]) for s in col_spans))
        else:
            data_spans.extend(col_spans)

    if not data_spans:
        return [], label_col_x_max

    hdr_v_xs = [
        x
        for x, y0, y1 in v_segs
        if x >= (label_col_x_max - 1.5) and y0 < header_scan_y_max and (y1 - y0) >= 35.0
    ]
    merged_vx = _merge_close_coords(hdr_v_xs, tol=2.0)
    col_intervals = [
        (merged_vx[i], merged_vx[i + 1])
        for i in range(len(merged_vx) - 1)
        if 6.0 <= (merged_vx[i + 1] - merged_vx[i]) <= 120.0
    ]

    if len(col_intervals) >= 3:
        grid_min_x, grid_max_x = col_intervals[0][0] - 1.5, col_intervals[-1][1] + 1.5
        covered = sum(
            1
            for s in data_spans
            if grid_min_x <= _bbox_mid_x(s["bbox"]) <= grid_max_x
        )
        if covered >= max(3, int(len(data_spans) * 0.6)):
            buckets: list[list[dict[str, Any]]] = [[] for _ in col_intervals]
            for sp in data_spans:
                sx = _bbox_mid_x(sp["bbox"])
                sy = _bbox_mid_y(sp["bbox"])
                cell_xr = _find_cell_span_1d(sx, sy, v_segs, max_span=120.0)
                if cell_xr is not None:
                    hx0, hx1 = cell_xr
                    assigned = False
                    for idx, (cx0, cx1) in enumerate(col_intervals):
                        c_mid = (cx0 + cx1) / 2.0
                        if (hx0 - 1.0) <= c_mid <= (hx1 + 1.0):
                            buckets[idx].append(sp)
                            assigned = True
                    if assigned:
                        continue
                for idx, (cx0, cx1) in enumerate(col_intervals):
                    if (cx0 - 1.5) <= sx <= (cx1 + 1.5):
                        buckets[idx].append(sp)
                        break

            effect_cols: list[dict[str, Any]] = []
            for (cx0, cx1), col_spans in zip(col_intervals, buckets):
                if not col_spans or _is_header_label_only_column(col_spans):
                    continue
                effect_cols.append(
                    _parse_effect_column_spans(
                        col_spans,
                        len(effect_cols),
                        (cx0 + cx1) / 2.0,
                        x0=cx0,
                        x1=cx1,
                    )
                )
            if effect_cols:
                return effect_cols, label_col_x_max

    dx_mids = [_bbox_mid_x(s["bbox"]) for s in data_spans]
    sorted_x = sorted(dx_mids)
    diffs = sorted(b - a for a, b in zip(sorted_x, sorted_x[1:]) if (b - a) >= 3.0)
    x_tol = min(14.0, max(4.0, diffs[0] * 0.42)) if diffs else 14.0
    col_clusters = sorted(
        _cluster_1d(dx_mids, tol=x_tol),
        key=lambda grp: sum(dx_mids[i] for i in grp) / len(grp),
    )
    effect_cols = []
    centers = [sum(dx_mids[i] for i in grp) / len(grp) for grp in col_clusters]
    half_pitch = (
        sorted(centers[i + 1] - centers[i] for i in range(len(centers) - 1))[0] / 2.0
        if len(centers) >= 2
        else 20.0
    )
    half_pitch = max(6.0, min(40.0, half_pitch))
    for grp, xc in zip(col_clusters, centers):
        col_spans = [data_spans[i] for i in grp]
        if _is_header_label_only_column(col_spans):
            continue
        effect_cols.append(
            _parse_effect_column_spans(
                col_spans,
                len(effect_cols),
                xc,
                x0=xc - half_pitch,
                x1=xc + half_pitch,
            )
        )
    return effect_cols, label_col_x_max


def _merge_adjacent_voting_spans(spans: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine adjacent '≥'/'<=' symbol and numeral spans on the same horizontal row."""
    if len(spans) <= 1:
        return list(spans)
    ordered = sorted(
        spans,
        key=lambda s: (round((float(s["bbox"][1]) + float(s["bbox"][3])) / 4.0), float(s["bbox"][0])),
    )
    merged: list[dict[str, Any]] = []
    idx = 0
    while idx < len(ordered):
        cur = ordered[idx]
        cur_txt = str(cur["text"]).strip()
        if cur_txt in ("≥", ">=", "≤", "<=") and idx + 1 < len(ordered):
            nxt = ordered[idx + 1]
            nxt_txt = str(nxt["text"]).strip()
            cy1 = _bbox_mid_y(cur["bbox"])
            cy2 = _bbox_mid_y(nxt["bbox"])
            dx = float(nxt["bbox"][0]) - float(cur["bbox"][2])
            if abs(cy1 - cy2) <= 3.0 and -1.5 <= dx <= 8.0 and re.fullmatch(r"\d+(?:oo\d+)?", nxt_txt, re.I):
                merged.append(
                    {
                        "text": f"{cur_txt} {nxt_txt}",
                        "bbox": [
                            min(float(cur["bbox"][0]), float(nxt["bbox"][0])),
                            min(float(cur["bbox"][1]), float(nxt["bbox"][1])),
                            max(float(cur["bbox"][2]), float(nxt["bbox"][2])),
                            max(float(cur["bbox"][3]), float(nxt["bbox"][3])),
                        ],
                    }
                )
                idx += 2
                continue
        merged.append(cur)
        idx += 1
    return merged


def _detect_cause_table_geometry(
    horiz_spans: Sequence[dict[str, Any]],
    v_segs: Sequence[tuple[float, float, float]],
    matrix_x_min: float,
    header_y_max: float,
    page_height: float,
) -> dict[str, Any]:
    """Detect Cause header bounds, body y-bounds, and REV/FIRE ZONE/ROW NO column x-ranges."""
    cue_spans: list[dict[str, Any]] = []
    rev_span: dict[str, Any] | None = None
    zone_span: dict[str, Any] | None = None
    tag_name_span: dict[str, Any] | None = None
    logic_span: dict[str, Any] | None = None

    for s in horiz_spans:
        xm = _bbox_mid_x(s["bbox"])
        ym = _bbox_mid_y(s["bbox"])
        if xm > matrix_x_min + 10.0:
            continue
        if not (max(0.0, header_y_max - 45.0) <= ym <= header_y_max + 45.0):
            continue
        norm = re.sub(r"\s+", " ", str(s["text"]).strip().rstrip(".:")).upper()
        if norm == "REV":
            rev_span = s
        if norm in _CAUSE_TABLE_HEADER_CUES:
            cue_spans.append(s)
            if "ZONE" in norm:
                zone_span = s
            elif norm == "TAG NAME":
                tag_name_span = s
            elif norm in ("LOGIC", "VOTING"):
                logic_span = s

    cause_hdr_y_max = max((float(s["bbox"][3]) for s in cue_spans), default=header_y_max)
    body_y_min = max(header_y_max - 2.0, cause_hdr_y_max + 2.0)

    note_y_top: float | None = None
    for s in horiz_spans:
        ym = _bbox_mid_y(s["bbox"])
        if ym <= body_y_min + 30.0:
            continue
        txt = str(s["text"]).strip()
        if (
            _NOTE_SECTION_RE.match(txt)
            or _PAGE_NUM_FOOTER_RE.match(txt)
            or txt.upper() in ("COMPANY", "COMPANY:", "PROJECT TITLE:", "CONTRACTOR", "DOCUMENT TITLE:")
        ):
            y0 = float(s["bbox"][1])
            note_y_top = y0 if note_y_top is None else min(note_y_top, y0)

    long_grid_bottoms = [
        y1
        for x, y0, y1 in v_segs
        if x >= 100.0
        and y0 < body_y_min + 35.0
        and y1 > body_y_min + 60.0
        and y1 < (page_height - 35.0)
        and (y1 - y0) >= 80.0
    ]
    grid_y_max = max(long_grid_bottoms) if long_grid_bottoms else None

    if note_y_top is not None:
        body_y_max = min(note_y_top - 1.5, (grid_y_max + 2.5) if grid_y_max else (page_height - 55.0))
    elif grid_y_max is not None:
        body_y_max = grid_y_max + 2.5
    else:
        body_y_max = page_height - 95.0

    rev_x_range: tuple[float, float] | None = None
    if rev_span is not None:
        rx0, rx1 = float(rev_span["bbox"][0]), float(rev_span["bbox"][2])
        rev_x_range = (rx0 - 8.0, rx1 + 8.0)

    zone_x_range: tuple[float, float] | None = None
    if zone_span is not None:
        zx0, zx1 = float(zone_span["bbox"][0]), float(zone_span["bbox"][2])
        zx_mid = (zx0 + zx1) / 2.0
        body_vx = [
            x
            for x, y0, y1 in v_segs
            if x < matrix_x_min - 50.0 and y1 > body_y_min + 20.0 and (y1 - y0) >= 30.0
        ]
        left_b = max((x for x in body_vx if x < zx_mid), default=(rev_x_range[1] if rev_x_range else zx0 - 15.0))
        right_b = min((x for x in body_vx if x > zx_mid), default=(float(tag_name_span["bbox"][0]) - 2.0 if tag_name_span else zx1 + 12.0))
        if right_b > left_b:
            zone_x_range = (left_b - 1.0, right_b + 1.0)
        else:
            zone_x_range = (zx0 - 12.0, zx1 + 12.0)

    row_no_probe_min = (
        max(float(logic_span["bbox"][2]) + 10.0, matrix_x_min - 28.0)
        if logic_span is not None
        else (matrix_x_min - 26.0)
    )
    near_ints = [
        s
        for s in horiz_spans
        if row_no_probe_min <= _bbox_mid_x(s["bbox"]) < matrix_x_min
        and body_y_min <= _bbox_mid_y(s["bbox"]) <= body_y_max
        and re.fullmatch(r"\d{1,3}", str(s["text"]).strip())
    ]
    row_no_x_min: float | None = row_no_probe_min if len(near_ints) >= 4 else None

    return {
        "body_y_min": body_y_min,
        "body_y_max": body_y_max,
        "rev_x_range": rev_x_range,
        "zone_x_range": zone_x_range,
        "row_no_x_min": row_no_x_min,
    }


def _partition_matrix_spans(
    text_dict: dict[str, Any],
    page_width: float,
    page_height: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition text spans into vertical rotated Effect header spans and horizontal spans."""
    rotated_spans: list[dict[str, Any]] = []
    horiz_spans: list[dict[str, Any]] = []
    for block in text_dict.get("blocks", ()):
        for line in block.get("lines", ()):
            direction = line.get("dir", (1.0, 0.0))
            is_vert = abs(float(direction[0])) < 0.2 and abs(float(direction[1])) > 0.8
            is_horiz = abs(float(direction[0])) > 0.8 and abs(float(direction[1])) < 0.2
            for span in line.get("spans", ()):
                txt = (span.get("text") or "").strip()
                if not txt:
                    continue
                bbox = span["bbox"]
                x_mid = _bbox_mid_x(bbox)
                y_mid = _bbox_mid_y(bbox)
                item = {"text": txt, "bbox": bbox}
                if is_vert:
                    if x_mid < 28.0 or (y_mid > page_height - 95.0 and x_mid > page_width * 0.60):
                        continue
                    rotated_spans.append(item)
                elif is_horiz:
                    horiz_spans.append(item)
    return rotated_spans, horiz_spans
