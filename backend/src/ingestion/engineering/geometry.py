"""Rotation-compensated PyMuPDF extraction, 1D clustering, and vector grid geometry helpers."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import pymupdf as fitz


def _bbox_mid_x(bbox: Sequence[float]) -> float:
    """Return horizontal midpoint of a [x0, y0, x1, y1] bounding box."""
    return (float(bbox[0]) + float(bbox[2])) / 2.0


def _bbox_mid_y(bbox: Sequence[float]) -> float:
    """Return vertical midpoint of a [x0, y0, x1, y1] bounding box."""
    return (float(bbox[1]) + float(bbox[3])) / 2.0


def _bbox_union(bboxes: Sequence[Sequence[float]]) -> list[float]:
    """Return rounded enclosing [x0, y0, x1, y1] union across bounding boxes."""
    return [
        round(min(float(b[0]) for b in bboxes), 2),
        round(min(float(b[1]) for b in bboxes), 2),
        round(max(float(b[2]) for b in bboxes), 2),
        round(max(float(b[3]) for b in bboxes), 2),
    ]


@lru_cache(maxsize=1)
def _pymupdf_coords_need_rotation() -> bool:
    """Detect whether PyMuPDF returns unrotated mediabox coords on rotated pages."""
    probe = fitz.open()
    try:
        page = probe.new_page(width=100, height=200)
        page.insert_text((20, 40), "A")
        page.set_rotation(90)
        words = page.get_text("words")
        if not words:
            return False
        # Unrotated x remains ~20; visual rotated x moves near 200 - 40 = 160
        return float(words[0][0]) < 80.0
    finally:
        probe.close()


def _page_needs_visual_rotation(page: fitz.Page) -> bool:
    return bool(getattr(page, "rotation", 0) % 360) and _pymupdf_coords_need_rotation()


def _transform_dir(direction: Sequence[float], rm: fitz.Matrix) -> tuple[float, float]:
    dx = float(direction[0]) if len(direction) >= 1 else 1.0
    dy = float(direction[1]) if len(direction) >= 2 else 0.0
    return (float(rm.a) * dx + float(rm.c) * dy, float(rm.b) * dx + float(rm.d) * dy)


def _get_visual_text_dict(page: fitz.Page) -> dict[str, Any]:
    """Return page.get_text('dict') normalized into visual page.rect coordinate space."""
    text_dict = page.get_text("dict")
    if not _page_needs_visual_rotation(page):
        return text_dict

    rm = page.rotation_matrix
    out_blocks: list[dict[str, Any]] = []
    for block in text_dict.get("blocks", ()):
        out_block = dict(block)
        if "bbox" in block:
            r = fitz.Rect(block["bbox"]) * rm
            out_block["bbox"] = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
        if "lines" in block:
            out_lines: list[dict[str, Any]] = []
            for line in block.get("lines", ()):
                out_line = dict(line)
                out_line["dir"] = _transform_dir(line.get("dir", (1.0, 0.0)), rm)
                if "bbox" in line:
                    lr = fitz.Rect(line["bbox"]) * rm
                    out_line["bbox"] = (float(lr.x0), float(lr.y0), float(lr.x1), float(lr.y1))
                out_spans: list[dict[str, Any]] = []
                for span in line.get("spans", ()):
                    out_span = dict(span)
                    if "bbox" in span:
                        sr = fitz.Rect(span["bbox"]) * rm
                        out_span["bbox"] = (float(sr.x0), float(sr.y0), float(sr.x1), float(sr.y1))
                    out_spans.append(out_span)
                out_line["spans"] = out_spans
                out_lines.append(out_line)
            out_block["lines"] = out_lines
        out_blocks.append(out_block)
    return {**text_dict, "blocks": out_blocks}


def _get_visual_words(page: fitz.Page) -> list[tuple[Any, ...]]:
    """Return page.get_text('words') normalized into visual page.rect coordinate space."""
    words = list(page.get_text("words"))
    if not words or not _page_needs_visual_rotation(page):
        return words

    rm = page.rotation_matrix
    transformed: list[tuple[Any, ...]] = []
    for w in words:
        r = fitz.Rect(float(w[0]), float(w[1]), float(w[2]), float(w[3])) * rm
        transformed.append((float(r.x0), float(r.y0), float(r.x1), float(r.y1), *w[4:]))
    return transformed


def _get_visual_drawings(page: fitz.Page) -> list[dict[str, Any]]:
    """Return page.get_drawings() normalized into visual page.rect coordinate space."""
    drawings = list(page.get_drawings())
    if not drawings or not _page_needs_visual_rotation(page):
        return drawings

    rm = page.rotation_matrix
    transformed: list[dict[str, Any]] = []
    for drw in drawings:
        nd = dict(drw)
        if drw.get("rect") is not None:
            nd["rect"] = fitz.Rect(drw["rect"]) * rm
        new_items: list[Any] = []
        for item in drw.get("items", ()):
            if not item:
                new_items.append(item)
                continue
            op = item[0]
            if op == "l" and len(item) >= 3:
                new_items.append(("l", item[1] * rm, item[2] * rm, *item[3:]))
            elif op == "re" and len(item) >= 2:
                new_items.append(("re", fitz.Rect(item[1]) * rm, *item[2:]))
            else:
                new_items.append(item)
        nd["items"] = new_items
        transformed.append(nd)
    return transformed


def _cluster_1d(values: Sequence[float], tol: float) -> list[list[int]]:
    """Cluster indices of sorted 1D floats when adjacent values differ by <= tol."""
    if not values:
        return []
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    clusters: list[list[int]] = [[indexed[0][0]]]
    anchor = indexed[0][1]
    for idx, val in indexed[1:]:
        if abs(val - anchor) <= tol:
            clusters[-1].append(idx)
            anchor = sum(values[i] for i in clusters[-1]) / len(clusters[-1])
        else:
            clusters.append([idx])
            anchor = val
    return clusters


def _merge_close_coords(values: Sequence[float], tol: float = 2.0) -> list[float]:
    if not values:
        return []
    clusters = _cluster_1d(values, tol=tol)
    merged = [sum(values[i] for i in grp) / len(grp) for grp in clusters]
    return sorted(merged)


def _get_visual_text_lines(page: fitz.Page) -> str:
    """Extract visually ordered readable text lines, compensating for rotated PDF pages."""
    if not _page_needs_visual_rotation(page):
        return page.get_text("text") or ""

    text_dict = _get_visual_text_dict(page)
    spans: list[dict[str, Any]] = []
    for block in text_dict.get("blocks", ()):
        for line in block.get("lines", ()):
            direction = line.get("dir", (1.0, 0.0))
            if abs(float(direction[1])) > 0.8 and abs(float(direction[0])) < 0.2:
                continue
            for span in line.get("spans", ()):
                txt = (span.get("text") or "").strip()
                if txt:
                    spans.append({"text": txt, "bbox": span["bbox"]})

    if not spans:
        return page.get_text("text") or ""

    y_mids = [_bbox_mid_y(s["bbox"]) for s in spans]
    rows = sorted(
        _cluster_1d(y_mids, tol=3.0),
        key=lambda grp: sum(y_mids[i] for i in grp) / len(grp),
    )
    out_lines: list[str] = []
    for grp in rows:
        row_spans = sorted((spans[i] for i in grp), key=lambda s: float(s["bbox"][0]))
        out_lines.append(" ".join(s["text"] for s in row_spans).strip())
    return "\n".join(ln for ln in out_lines if ln)


def _collect_table_grid_geometry(
    drawings: Sequence[dict[str, Any]],
) -> tuple[list[float], tuple[float, float] | None]:
    """Infer repeating vertical column boundaries and table y-range from PDF vector drawings."""
    v_candidates: list[float] = []
    h_candidates: list[float] = []
    cell_rects: list[fitz.Rect] = []

    def _maybe_add_cell_rect(r: fitz.Rect) -> bool:
        rw, rh = float(r.width), float(r.height)
        if rw <= 2.5 and rh >= 45.0:
            v_candidates.append((float(r.x0) + float(r.x1)) / 2.0)
            h_candidates.extend([float(r.y0), float(r.y1)])
            return True
        if rh <= 2.5 and rw >= 120.0:
            h_candidates.append((float(r.y0) + float(r.y1)) / 2.0)
            return True
        if 22.0 <= rw <= 420.0 and 12.0 <= rh <= 75.0:
            cell_rects.append(r)
            v_candidates.extend([float(r.x0), float(r.x1)])
            h_candidates.extend([float(r.y0), float(r.y1)])
            return True
        return False

    for drw in drawings:
        rect = drw.get("rect")
        if rect is not None and _maybe_add_cell_rect(rect):
            continue

        for item in drw.get("items", ()):
            if not item:
                continue
            op = item[0]
            if op == "l" and len(item) >= 3:
                p1, p2 = item[1], item[2]
                dx = abs(float(p1.x) - float(p2.x))
                dy = abs(float(p1.y) - float(p2.y))
                if dx <= 1.5 and dy >= 45.0:
                    v_candidates.append((float(p1.x) + float(p2.x)) / 2.0)
                    h_candidates.extend([float(p1.y), float(p2.y)])
                elif dy <= 1.5 and dx >= 120.0:
                    h_candidates.append((float(p1.y) + float(p2.y)) / 2.0)
            elif op == "re" and len(item) >= 2:
                _maybe_add_cell_rect(item[1])

    # Only keep vertical boundaries that repeat across multiple rows (>= 3 occurrences) or from long lines
    vx_clusters = _cluster_1d(v_candidates, tol=3.5)
    significant_vx = [
        sum(v_candidates[i] for i in grp) / len(grp)
        for grp in vx_clusters
        if len(grp) >= 3 or len(cell_rects) == 0
    ]
    significant_vx.sort()

    y_range: tuple[float, float] | None = None
    if len(cell_rects) >= 12:
        y_range = (min(float(r.y0) for r in cell_rects), max(float(r.y1) for r in cell_rects))
    elif h_candidates:
        hy_clusters = _cluster_1d(h_candidates, tol=3.5)
        rep_hy = [
            sum(h_candidates[i] for i in grp) / len(grp)
            for grp in hy_clusters
            if len(grp) >= 2
        ]
        if len(rep_hy) >= 4:
            y_range = (min(rep_hy), max(rep_hy))

    return significant_vx, y_range


def _estimate_column_bands(
    words: Sequence[tuple[Any, ...]],
    drawings: Sequence[dict[str, Any]],
) -> tuple[int, int]:
    """Estimate (column_band_count, tabular_body_row_count) quickly for geometry profiling."""
    if not words:
        return 0, 0

    sig_vx, _ = _collect_table_grid_geometry(drawings)
    grid_col_count = max(0, len(sig_vx) - 1) if len(sig_vx) >= 4 else 0

    y_mids = [_bbox_mid_y(w) for w in words]
    y_clusters = _cluster_1d(y_mids, tol=3.5)

    tabular_row_count = 0
    all_phrase_x0: list[float] = []
    for grp in y_clusters:
        row_words = sorted((words[i] for i in grp), key=lambda w: float(w[0]))
        if len(row_words) < 4:
            continue
        phrase_xs: list[float] = [float(row_words[0][0])]
        prev_x1 = float(row_words[0][2])
        for w in row_words[1:]:
            if float(w[0]) - prev_x1 > 10.0:
                phrase_xs.append(float(w[0]))
            prev_x1 = max(prev_x1, float(w[2]))
        if len(phrase_xs) >= 4:
            tabular_row_count += 1
            all_phrase_x0.extend(phrase_xs)

    if tabular_row_count == 0:
        return grid_col_count, 0

    x_clusters = _cluster_1d(all_phrase_x0, tol=14.0)
    min_row_support = max(2, min(4, tabular_row_count // 2))
    aligned_col_count = sum(1 for grp in x_clusters if len(grp) >= min_row_support)

    return max(grid_col_count, aligned_col_count), tabular_row_count


def _count_rotated_spans(text_dict: dict[str, Any]) -> int:
    """Count non-empty spans belonging to vertically rotated text lines."""
    count = 0
    for block in text_dict.get("blocks", ()):
        for line in block.get("lines", ()):
            direction = line.get("dir", (1.0, 0.0))
            if abs(float(direction[0])) < 0.2 and abs(float(direction[1])) > 0.8:
                for span in line.get("spans", ()):
                    if (span.get("text") or "").strip():
                        count += 1
    return count


def _extract_grid_segments(
    drawings: Sequence[dict[str, Any]],
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Extract visual vertical `(x, y0, y1)` and horizontal `(y, x0, x1)` grid border segments."""
    v_segs: list[tuple[float, float, float]] = []
    h_segs: list[tuple[float, float, float]] = []

    def _add_rect_edges(r: fitz.Rect) -> None:
        rw, rh = float(r.width), float(r.height)
        x0, y0, x1, y1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
        if rw <= 2.5 and rh >= 4.0:
            v_segs.append(((x0 + x1) / 2.0, min(y0, y1), max(y0, y1)))
        elif rh <= 2.5 and rw >= 4.0:
            h_segs.append(((y0 + y1) / 2.0, min(x0, x1), max(x0, x1)))
        elif rw > 2.5 and rh > 2.5:
            v_segs.append((x0, min(y0, y1), max(y0, y1)))
            v_segs.append((x1, min(y0, y1), max(y0, y1)))
            h_segs.append((y0, min(x0, x1), max(x0, x1)))
            h_segs.append((y1, min(x0, x1), max(x0, x1)))

    for drw in drawings:
        items = drw.get("items") or ()
        if not items and drw.get("rect") is not None:
            _add_rect_edges(fitz.Rect(drw["rect"]))
            continue
        for item in items:
            if not item:
                continue
            op = item[0]
            if op == "l" and len(item) >= 3:
                p1, p2 = item[1], item[2]
                x1, y1 = float(p1.x), float(p1.y)
                x2, y2 = float(p2.x), float(p2.y)
                if abs(x1 - x2) <= 2.0 and abs(y1 - y2) >= 4.0:
                    v_segs.append(((x1 + x2) / 2.0, min(y1, y2), max(y1, y2)))
                elif abs(y1 - y2) <= 2.0 and abs(x1 - x2) >= 4.0:
                    h_segs.append(((y1 + y2) / 2.0, min(x1, x2), max(x1, x2)))
            elif op == "re" and len(item) >= 2:
                _add_rect_edges(fitz.Rect(item[1]))

    return v_segs, h_segs


def _find_cell_span_1d(
    pos: float,
    cross_pos: float,
    segs: Sequence[tuple[float, float, float]],
    *,
    max_span: float,
) -> tuple[float, float] | None:
    """Find tightest enclosing parallel grid borders crossing `cross_pos` around `pos`."""
    low: list[float] = []
    high: list[float] = []
    for coord, a, b in segs:
        if a - 2.0 <= cross_pos <= b + 2.0:
            if coord < pos - 0.8:
                low.append(coord)
            elif coord > pos + 0.8:
                high.append(coord)
    if not low or not high:
        return None
    start = max(low)
    end = min(high)
    if 0.5 < (end - start) <= max_span:
        return (start, end)
    return None
