"""RapidOCR v3 CPU engine lifecycle, safe DPI calculation, and exploded-SHX tiled OCR extraction."""
from __future__ import annotations

import logging
import os
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pymupdf as fitz

from src.ingestion.engineering.models import EngineeringPageRecord, PageExtractionResult
from src.ingestion.engineering.patterns import _extract_record_tags
from src.ingestion.engineering.profiler import profile_pdf_page

logger = logging.getLogger(__name__)

_RAPIDOCR_ENGINE: Any | None = None
_RAPIDOCR_INIT_ATTEMPTED = False


def get_rapidocr_engine() -> Any | None:
    """Return a lazily cached CPU RapidOCR instance, or None if unavailable."""
    global _RAPIDOCR_ENGINE, _RAPIDOCR_INIT_ATTEMPTED
    if _RAPIDOCR_ENGINE is not None:
        return _RAPIDOCR_ENGINE
    if _RAPIDOCR_INIT_ATTEMPTED:
        return None
    _RAPIDOCR_INIT_ATTEMPTED = True
    try:
        from rapidocr import RapidOCR

        params: dict[str, Any] = {"EngineConfig.onnxruntime.use_cuda": False}
        model_root = (os.environ.get("RAPIDOCR_MODEL_DIR") or "").strip()
        if model_root:
            Path(model_root).mkdir(parents=True, exist_ok=True)
            params["Global.model_root_dir"] = model_root
        _RAPIDOCR_ENGINE = RapidOCR(params=params)
    except Exception as exc:
        logger.warning("RapidOCR initialization failed; exploded-CAD OCR unavailable: %s", exc)
        _RAPIDOCR_ENGINE = None
    return _RAPIDOCR_ENGINE


def reset_rapidocr_engine() -> None:
    """Reset the cached RapidOCR singleton so subsequent calls re-initialize with updated env/config."""
    global _RAPIDOCR_ENGINE, _RAPIDOCR_INIT_ATTEMPTED
    _RAPIDOCR_ENGINE = None
    _RAPIDOCR_INIT_ATTEMPTED = False


def compute_safe_ocr_dpi(
    width_pt: float,
    height_pt: float,
    *,
    target_dpi: int = 300,
    min_dpi: int = 150,
    max_edge_px: int = 4800,
) -> int:
    """Compute OCR render DPI keeping the longest pixel edge <= max_edge_px.

    Scales dynamically inside [min_dpi, target_dpi] for standard/large CAD sheets and clamps
    strictly below max_edge_px even on oversized A0 sheets to prevent OOM bitmaps.
    """
    longest_pt = max(float(width_pt), float(height_pt), 1.0)
    max_allowed_dpi = int((max_edge_px * 72.0) / longest_pt)
    if max_allowed_dpi >= target_dpi:
        return target_dpi
    if max_allowed_dpi >= min_dpi:
        return max_allowed_dpi
    return max(72, max_allowed_dpi)


def _bbox_intersection_and_areas(
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> tuple[float, float, float]:
    ix0 = max(float(box_a[0]), float(box_b[0]))
    iy0 = max(float(box_a[1]), float(box_b[1]))
    ix1 = min(float(box_a[2]), float(box_b[2]))
    iy1 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    return inter, area_a, area_b


def compute_bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute intersection-over-union of two [x0, y0, x1, y1] bounding boxes."""
    inter, area_a, area_b = _bbox_intersection_and_areas(box_a, box_b)
    if inter <= 0.0:
        return 0.0
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _bbox_io_min(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    inter, area_a, area_b = _bbox_intersection_and_areas(box_a, box_b)
    if inter <= 0.0:
        return 0.0
    return inter / min(max(1e-6, area_a), max(1e-6, area_b))


def deduplicate_ocr_detections(
    detections: Sequence[dict[str, Any]],
    *,
    iou_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    """Deduplicate overlapping OCR detections across tiles using IoU and text similarity."""
    if not detections:
        return []

    ranked = sorted(
        detections,
        key=lambda d: (float(d.get("score", 0.0)) * (1.0 + 0.05 * len(str(d.get("text", ""))))),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for cand in ranked:
        c_box = cand["bbox"]
        c_txt = str(cand.get("text", "")).strip()
        if not c_txt:
            continue
        is_dup = False
        for existing in kept:
            e_box = existing["bbox"]
            e_txt = str(existing.get("text", "")).strip()
            iou = compute_bbox_iou(c_box, e_box)
            if iou >= iou_threshold:
                is_dup = True
                break
            io_min = _bbox_io_min(c_box, e_box)
            if io_min >= 0.65:
                sim = SequenceMatcher(None, c_txt.upper(), e_txt.upper()).ratio()
                if sim >= 0.75 or c_txt.upper() in e_txt.upper() or e_txt.upper() in c_txt.upper():
                    is_dup = True
                    break
        if not is_dup:
            kept.append(cand)

    kept.sort(key=lambda d: (round(float(d["bbox"][1]) / 12.0), float(d["bbox"][0])))
    return kept


def _normalize_ocr_output(raw_output: Any) -> list[tuple[list[float], str, float]]:
    """Normalize RapidOCR v3 RapidOCROutput or legacy list/tuple OCR output into [(bbox, text, score)]."""
    if raw_output is None:
        return []

    if hasattr(raw_output, "txts") and hasattr(raw_output, "boxes"):
        txts = getattr(raw_output, "txts") or ()
        boxes = getattr(raw_output, "boxes")
        scores = getattr(raw_output, "scores") or ()
        if boxes is None or len(txts) == 0:
            return []
        normalized: list[tuple[list[float], str, float]] = []
        for idx, txt in enumerate(txts):
            box = boxes[idx] if idx < len(boxes) else None
            score = float(scores[idx]) if idx < len(scores) else 0.9
            if box is None:
                continue
            xs = [float(pt[0]) for pt in box]
            ys = [float(pt[1]) for pt in box]
            normalized.append(([min(xs), min(ys), max(xs), max(ys)], str(txt).strip(), score))
        return normalized

    items = raw_output[0] if isinstance(raw_output, tuple) and len(raw_output) >= 1 else raw_output
    if not isinstance(items, (list, tuple)):
        return []

    normalized = []
    for item in items:
        if isinstance(item, dict) and "bbox" in item and "text" in item:
            normalized.append(([float(x) for x in item["bbox"]], str(item["text"]).strip(), float(item.get("score", 0.9))))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            box, text_part = item[0], item[1]
            score = float(item[2]) if len(item) >= 3 else 0.9
            if isinstance(text_part, (list, tuple)):
                text_str = str(text_part[0])
                score = float(text_part[1]) if len(text_part) > 1 else score
            else:
                text_str = str(text_part)
            if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
                bbox = [float(v) for v in box]
            else:
                xs = [float(pt[0]) for pt in box]
                ys = [float(pt[1]) for pt in box]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            normalized.append((bbox, text_str.strip(), score))
    return normalized


def extract_exploded_cad_ocr(
    page: fitz.Page,
    page_number: int,
    ocr_engine: Any | None = None,
    *,
    metrics: dict[str, Any] | None = None,
) -> PageExtractionResult:
    """Run CPU 2x2 overlapping tiled OCR (+ title-block crop) on exploded-SHX CAD pages."""
    if metrics is None:
        _, metrics = profile_pdf_page(page)
    width_pt = float(page.rect.width)
    height_pt = float(page.rect.height)

    render_dpi = compute_safe_ocr_dpi(width_pt, height_pt, target_dpi=300, min_dpi=150, max_edge_px=4800)
    pix = page.get_pixmap(dpi=render_dpi, colorspace=fitz.csRGB, alpha=False)
    img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    img_h, img_w = img_arr.shape[:2]

    ox = int(img_w * 0.075)
    oy = int(img_h * 0.075)
    hw = img_w // 2
    hh = img_h // 2

    tile_specs = [
        ("top_left", 0, 0, min(img_w, hw + ox), min(img_h, hh + oy)),
        ("top_right", max(0, hw - ox), 0, img_w, min(img_h, hh + oy)),
        ("bottom_left", 0, max(0, hh - oy), min(img_w, hw + ox), img_h),
        ("bottom_right", max(0, hw - ox), max(0, hh - oy), img_w, img_h),
        ("title_block", int(img_w * 0.55), int(img_h * 0.72), img_w, img_h),
    ]

    if ocr_engine is not None:
        engine = ocr_engine
    else:
        from src.ingestion import engineering_extractor

        engine = engineering_extractor.get_rapidocr_engine()

    scale_x = width_pt / max(1, img_w)
    scale_y = height_pt / max(1, img_h)

    raw_detections: list[dict[str, Any]] = []
    if engine is not None:
        for tile_name, x0, y0, x1, y1 in tile_specs:
            tile_img = img_arr[y0:y1, x0:x1]
            if tile_img.size == 0:
                continue
            try:
                ocr_res = engine(tile_img)
            except Exception as exc:
                logger.warning("OCR tile '%s' failed on page %d: %s", tile_name, page_number, exc)
                continue
            for box_px, text_str, score in _normalize_ocr_output(ocr_res):
                if not text_str:
                    continue
                pdf_bbox = [
                    round((box_px[0] + x0) * scale_x, 2),
                    round((box_px[1] + y0) * scale_y, 2),
                    round((box_px[2] + x0) * scale_x, 2),
                    round((box_px[3] + y0) * scale_y, 2),
                ]
                raw_detections.append(
                    {
                        "text": text_str,
                        "bbox": pdf_bbox,
                        "score": round(float(score), 4),
                        "tile": tile_name,
                    }
                )

    deduped = deduplicate_ocr_detections(raw_detections, iou_threshold=0.45)

    lines_text: list[str] = []
    current_line: list[str] = []
    last_y: float | None = None
    for det in deduped:
        y0 = float(det["bbox"][1])
        if last_y is None or abs(y0 - last_y) <= 14.0:
            current_line.append(det["text"])
        else:
            if current_line:
                lines_text.append(" ".join(current_line))
            current_line = [det["text"]]
        last_y = y0
    if current_line:
        lines_text.append(" ".join(current_line))

    assembled_text = "\n".join(lines_text).strip()
    page_title = f"Exploded CAD Schematic OCR (Page {page_number})"
    if lines_text:
        for ln in lines_text:
            if "LOGIC" in ln.upper() or "SWGR" in ln.upper() or "DRAWING" in ln.upper():
                page_title = ln
                break

    tags = _extract_record_tags(assembled_text)
    summary_text = assembled_text or "(No readable OCR tokens recovered)"

    records: list[EngineeringPageRecord] = []
    if deduped:
        records.append(
            EngineeringPageRecord(
                record_id=f"p{page_number}_ocr_1",
                record_type="exploded_cad_ocr",
                page_number=page_number,
                title=page_title,
                summary_text=summary_text,
                tags=tags,
                bbox=[0.0, 0.0, round(width_pt, 2), round(height_pt, 2)],
                metadata={
                    "render_dpi": render_dpi,
                    "rendered_pixels": [img_w, img_h],
                    "tile_count": len(tile_specs),
                    "raw_detection_count": len(raw_detections),
                    "deduped_detection_count": len(deduped),
                    "detections": deduped,
                },
            )
        )

    out_metrics = {
        **metrics,
        "render_dpi": render_dpi,
        "rendered_width_px": img_w,
        "rendered_height_px": img_h,
        "ocr_detection_count": len(deduped),
    }

    markdown = (
        f"<!-- page: {page_number} -->\n"
        f"# {page_title} (Page {page_number})\n\n"
        f"{summary_text}\n"
    )
    return PageExtractionResult(
        page_number=page_number,
        mode="exploded_cad_ocr",
        metrics=out_metrics,
        records=records,
        markdown=markdown,
    )
