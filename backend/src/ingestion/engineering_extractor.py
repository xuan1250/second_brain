"""Per-page engineering PDF geometry profiler and deterministic coordinate/OCR extractors.

Routes 2D CAD engineering sheets (schedules, rotated Cause & Effect matrices, FBD logic diagrams,
and exploded-SHX vector schematics) into structured records and self-contained Markdown without
destructive 1D column flattening or GPU VRAM overhead.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pymupdf as fitz

logger = logging.getLogger(__name__)

ENGINEERING_MODES = (
    "exploded_cad_ocr",
    "matrix_rotated_headers",
    "document_legend_notes",
    "cad_logic_diagram",
    "coord_grid_schedule",
    "standard_prose",
)

ACTION_MARKS = frozenset({"X", "O", "R", "P", "B"})

_TAG_TOKEN_RE = re.compile(
    r"\b(?:"
    r"[A-Z]{2,6}-\d{3,6}[A-Z]?(?:/[A-Z0-9]+)*"
    r"|(?:TI|PI|FI|LI|AI|AO|DI|DO|FIC|PIC|LIC|TIC|PIA|LIA|HIC|TIA|FIA|XV|FCV|PCV|HCV|PSHH|PALL|PAHH|LAHH|PDALL|UZA|UZ|ZI|VI|XI)\d{3,6}[A-Z]?"
    r"|(?:HZ|UA|UZ|UZA|RSESD|LTRCV|LTFCS|MAIN)(?=[-_A-Z0-9]*\d)[-_A-Z0-9]{2,20}"
    r"|(?:LNG-[A-Z0-9-]+|FCS\d{4}|SCS\d{4}|AAI\d{3}-[A-Z]|AAV\d{3}-[A-Z]|ADV\d{3}-[A-Z]|\d[FR]TB\d+|FS\d{2}|RS\d{2}|CB-\d+|MCC-[A-Z0-9]+)"
    r")\b"
)

_PID_RE = re.compile(r"\bPID[-_]?\d{2,6}\b", re.IGNORECASE)
_NOTE_REF_RE = re.compile(r"\bNOTE\s*[-#:]?\s*(\d+)\b", re.IGNORECASE)
_SYMBOL_DEF_LINE_RE = re.compile(r"^\s*([XOPRB])\s*[:=]\s*([A-Za-z0-9].+?)\s*$")
_SYMBOL_DEF_INLINE_RE = re.compile(r"(?:^|\n)\s*([XOPRB])\s*[:=]\s*[A-Za-z][^\n]*")
_NOTE_DEF_START_RE = re.compile(r"^\s*(NOTE\s*[-#:]?\s*(\d+))\s*[:.\-]\s*(.+?)\s*$", re.IGNORECASE)
_VOTING_RE = re.compile(r"\b(?:\d+\s*out\s*of\s*\d+|\d+\s*oo\s*[\dN]+)\b", re.IGNORECASE)
_GROUP_CODE_RE = re.compile(r"^(?:ESD[-_]?\d+|FGS|PSD|SIS|BMS|HVAC)$", re.IGNORECASE)
_LOGIC_GATE_RE = re.compile(r"\b(?:AND|OR|RS|SR|TON|TOF|SET|RESET|NOT|XOR)\b")
_POU_TITLE_RE = re.compile(
    r"\b\d+\.\d+\.\s*[A-Z][A-Z0-9_]+(?:\s*\(\*\s*[^*]+\s*\*\))?",
    re.IGNORECASE,
)
_CROSS_REF_RE = re.compile(r"\(\*\s*([^*\n]+?)\s*\*\)")

_EFFECT_ACTION_KEYWORDS = (
    "close",
    "open",
    "trip",
    "stop",
    "start",
    "alarm",
    "activate",
    "activation",
    "shutdown",
    "isolate",
    "inhibit",
    "energize",
    "de-energize",
    "reset",
)

_SCHEDULE_HEADER_KEYWORDS = frozenset(
    {
        "ITEM",
        "TAG",
        "FIELD",
        "DCS",
        "TERMINAL",
        "BLOCK",
        "TB",
        "FCS",
        "NODE",
        "IO",
        "MODULE",
        "CHANNEL",
        "SERVICE",
        "DESCRIPTION",
        "REMARKS",
        "SIGNAL",
        "LOOP",
        "RANGE",
        "UNIT",
        "CABINET",
    }
)

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


@dataclass
class LegendContext:
    """Document-level action symbol legend and numbered engineering notes."""

    symbol_map: dict[str, str] = field(default_factory=dict)
    notes_map: dict[str, str] = field(default_factory=dict)
    source_pages: list[int] = field(default_factory=list)
    raw_text: str = ""

    def format_summary(self) -> str:
        """Return a concise human-readable summary of symbols and notes."""
        parts: list[str] = []
        if self.symbol_map:
            syms = "; ".join(f"{k} = {v}" for k, v in self.symbol_map.items())
            parts.append(f"Legend Symbols: {syms}")
        if self.notes_map:
            notes = " | ".join(f"{k}: {v}" for k, v in self.notes_map.items())
            parts.append(f"Notes: {notes}")
        return "\n".join(parts)

    def lookup_notes(self, refs: Sequence[str]) -> list[str]:
        """Resolve note references (e.g. 'NOTE 1', '36') into full note strings."""
        resolved: list[str] = []
        seen: set[str] = set()
        for ref in refs:
            if not ref:
                continue
            raw = str(ref).strip()
            match = re.search(r"(?:NOTE\s*[-#:]?\s*)?(\d+)", raw, re.IGNORECASE)
            candidates = [raw.upper()]
            if match:
                candidates.append(f"NOTE {match.group(1)}")
            for cand in candidates:
                if cand in self.notes_map and cand not in seen:
                    seen.add(cand)
                    resolved.append(f"{cand}: {self.notes_map[cand]}")
        return resolved


@dataclass
class EngineeringPageRecord:
    """Atomic structured record extracted from a CAD engineering PDF page."""

    record_id: str
    record_type: str
    page_number: int
    title: str
    summary_text: str
    tags: list[str] = field(default_factory=list)
    bbox: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageExtractionResult:
    """Complete per-page engineering extraction payload."""

    page_number: int
    mode: str
    metrics: dict[str, Any]
    records: list[EngineeringPageRecord]
    markdown: str
    legend_context: LegendContext | None = None


def _extract_record_tags(text: str) -> list[str]:
    """Extract deduplicated industrial tags and identifiers from text preserving order."""
    seen: set[str] = set()
    tags: list[str] = []
    for match in _TAG_TOKEN_RE.finditer(text or ""):
        token = match.group(0).strip()
        if token and token not in seen:
            seen.add(token)
            tags.append(token)
    for match in _NOTE_REF_RE.finditer(text or ""):
        note_key = f"NOTE {match.group(1)}"
        if note_key not in seen:
            seen.add(note_key)
            tags.append(note_key)
    return tags


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


def _collect_table_grid_geometry(
    drawings: Sequence[dict[str, Any]],
) -> tuple[list[float], tuple[float, float] | None]:
    """Infer repeating vertical column boundaries and table y-range from PDF vector drawings."""
    v_candidates: list[float] = []
    h_candidates: list[float] = []
    cell_rects: list[fitz.Rect] = []

    def _maybe_add_cell_rect(r: fitz.Rect) -> bool:
        if 22.0 <= float(r.width) <= 420.0 and 12.0 <= float(r.height) <= 75.0:
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

    # Group words into horizontal y-bands (<= 3.5 pt)
    y_mids = [(float(w[1]) + float(w[3])) / 2.0 for w in words]
    y_clusters = _cluster_1d(y_mids, tol=3.5)

    tabular_row_count = 0
    all_phrase_x0: list[float] = []
    for grp in y_clusters:
        row_words = sorted((words[i] for i in grp), key=lambda w: float(w[0]))
        if len(row_words) < 4:
            continue
        # Count distinct column-like phrases separated by horizontal gaps > 10 pt
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

    # Cluster phrase x0 positions that recur across rows
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


def profile_pdf_page(page: fitz.Page) -> tuple[str, dict[str, Any]]:
    """Classify a single PyMuPDF page in <5ms into one of 6 deterministic geometry modes."""
    rect = page.rect
    width = float(rect.width)
    height = float(rect.height)

    words = page.get_text("words")
    drawings = page.get_drawings()
    text_dict = page.get_text("dict")
    raw_text = page.get_text("text")

    word_count = len(words)
    drawing_count = len(drawings)
    rotated_span_count = _count_rotated_spans(text_dict)
    single_char_mark_count = sum(1 for w in words if str(w[4]).strip() in ACTION_MARKS)
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

        text = page.get_text("text") or ""
        if not text.strip():
            continue

        page_found = False
        current_note_key: str | None = None

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
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


def _extract_page_title_and_context(
    words: Sequence[tuple[Any, ...]],
    table_y_range: tuple[float, float] | None,
    page_height: float,
    default_title: str,
) -> tuple[str, list[str]]:
    """Extract drawing title and sheet metadata tokens from top banner / title block outside table."""
    outside_words: list[tuple[Any, ...]] = []
    for w in words:
        ym = (float(w[1]) + float(w[3])) / 2.0
        if table_y_range is not None:
            if ym < table_y_range[0] - 2.0 or ym > table_y_range[1] + 2.0:
                outside_words.append(w)
        elif ym < page_height * 0.14 or ym > page_height * 0.86:
            outside_words.append(w)

    if not outside_words:
        return default_title, []

    y_mids = [(float(w[1]) + float(w[3])) / 2.0 for w in outside_words]
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
        (w for w in words if abs(((float(w[1]) + float(w[3])) / 2.0) - header_y_mid) <= 10.0),
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


def extract_coord_grid_schedule(page: fitz.Page, page_number: int) -> PageExtractionResult:
    """Reconstruct wide coordinate-grid tables/schedules with exact column-header bindings."""
    _, metrics = profile_pdf_page(page)
    words = page.get_text("words")
    drawings = page.get_drawings()
    page_height = float(page.rect.height)

    sig_vx, table_y_range = _collect_table_grid_geometry(drawings)
    title, sheet_context_tags = _extract_page_title_and_context(
        words,
        table_y_range,
        page_height,
        default_title=f"Coordinate Schedule (Page {page_number})",
    )

    # Filter words belonging to the table grid if y_range is known
    if table_y_range is not None:
        table_words = [
            w
            for w in words
            if (table_y_range[0] - 2.0) <= ((float(w[1]) + float(w[3])) / 2.0) <= (table_y_range[1] + 2.0)
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

    # Group table words into horizontal rows (<= 3.5 pt)
    y_mids = [(float(w[1]) + float(w[3])) / 2.0 for w in table_words]
    row_groups = sorted(
        _cluster_1d(y_mids, tol=3.5),
        key=lambda grp: sum(y_mids[i] for i in grp) / len(grp),
    )

    # Detect header row index (prefer row containing schedule header keywords near top)
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

    # Determine column intervals [x0, x1]
    if len(sig_vx) >= 7:
        col_intervals = [(sig_vx[i], sig_vx[i + 1]) for i in range(len(sig_vx) - 1)]
    else:
        col_intervals = _fallback_column_intervals_from_words(table_words, header_y)

    # Map words in the header band to column names
    header_names: list[str] = []
    for c_idx, (cx0, cx1) in enumerate(col_intervals):
        col_words = sorted(
            (
                table_words[i]
                for i in header_grp
                if (cx0 - 2.0) <= ((float(table_words[i][0]) + float(table_words[i][2])) / 2.0) <= (cx1 + 2.0)
            ),
            key=lambda w: (float(w[1]), float(w[0])),
        )
        col_label = " ".join(str(w[4]) for w in col_words).strip()
        header_names.append(col_label or f"COL_{c_idx + 1}")

    # Drop empty trailing columns only if there are neither header labels nor body values
    records: list[EngineeringPageRecord] = []
    table_rows_vals: list[list[str]] = []

    for body_num, grp in enumerate(row_groups[header_idx + 1 :], start=1):
        row_words = [table_words[i] for i in grp]
        if len(row_words) < 2:
            continue

        cells: list[list[tuple[Any, ...]]] = [[] for _ in col_intervals]
        for w in row_words:
            wx_mid = (float(w[0]) + float(w[2])) / 2.0
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

        if not row_dict:
            continue

        table_rows_vals.append(row_list)
        summary_text = " | ".join(f"{k}: {v}" for k, v in row_dict.items())
        row_tags = _extract_record_tags(summary_text)
        for st in sheet_context_tags:
            if st not in row_tags:
                row_tags.append(st)

        rx0 = min(float(w[0]) for w in row_words)
        ry0 = min(float(w[1]) for w in row_words)
        rx1 = max(float(w[2]) for w in row_words)
        ry1 = max(float(w[3]) for w in row_words)

        records.append(
            EngineeringPageRecord(
                record_id=f"p{page_number}_row_{body_num}",
                record_type="schedule_row",
                page_number=page_number,
                title=title,
                summary_text=summary_text,
                tags=row_tags,
                bbox=[round(rx0, 2), round(ry0, 2), round(rx1, 2), round(ry1, 2)],
                metadata={
                    "row_index": body_num,
                    "drawing_title": title,
                    "sheet_context_tags": sheet_context_tags,
                    "columns": row_dict,
                },
            )
        )

    # Assemble clean Markdown pipe table + header
    md_lines = [
        f"<!-- page: {page_number} -->",
        f"# {title} (Page {page_number})",
    ]
    if sheet_context_tags:
        md_lines.append(f"Sheet Context: {' | '.join(sheet_context_tags)}")
    md_lines.append("")
    if header_names and table_rows_vals:
        md_lines.append("| " + " | ".join(header_names) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(header_names)) + " |")
        for rvals in table_rows_vals:
            escaped = [v.replace("|", "\\|") for v in rvals]
            md_lines.append("| " + " | ".join(escaped) + " |")

    return PageExtractionResult(
        page_number=page_number,
        mode="coord_grid_schedule",
        metrics=metrics,
        records=records,
        markdown="\n".join(md_lines).strip() + "\n",
    )


def _parse_effect_column_spans(
    spans: Sequence[dict[str, Any]],
    col_idx: int,
    x_center: float,
) -> dict[str, Any]:
    """Classify rotated spans belonging to one vertical C&E Effect column."""
    sorted_spans = sorted(spans, key=lambda s: (float(s["bbox"][1]), float(s["bbox"][0])))
    texts = [str(s["text"]).strip() for s in sorted_spans if str(s["text"]).strip()]

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
        if _PID_RE.fullmatch(txt):
            pid = txt.upper()
            continue
        if not effect_tag and _TAG_TOKEN_RE.fullmatch(txt) and not _PID_RE.fullmatch(txt):
            effect_tag = txt
            continue
        lower_txt = txt.lower()
        if not action and any(kw in lower_txt for kw in _EFFECT_ACTION_KEYWORDS) and len(txt.split()) <= 3:
            action = txt
            continue
        desc_parts.append(txt)

    if not effect_tag:
        # Fallback search inside combined text
        for txt in texts:
            m = _TAG_TOKEN_RE.search(txt)
            if m and not _PID_RE.fullmatch(m.group(0)):
                effect_tag = m.group(0)
                break

    description = " ".join(desc_parts).strip()
    if not effect_tag:
        effect_tag = description or pid or f"EFFECT_COL_{col_idx + 1}"

    return {
        "col_index": col_idx + 1,
        "x_center": round(x_center, 2),
        "pid": pid,
        "effect_tag": effect_tag,
        "description": description,
        "action": action,
        "note_refs": note_refs,
        "raw_labels": texts,
        "bbox": [
            round(min(float(s["bbox"][0]) for s in sorted_spans), 2),
            round(min(float(s["bbox"][1]) for s in sorted_spans), 2),
            round(max(float(s["bbox"][2]) for s in sorted_spans), 2),
            round(max(float(s["bbox"][3]) for s in sorted_spans), 2),
        ],
    }


def _parse_cause_row_spans(
    spans: Sequence[dict[str, Any]],
    row_idx: int,
    y_center: float,
) -> dict[str, Any]:
    """Classify horizontal spans belonging to one left-hand C&E Cause row."""
    # Sort top-to-bottom within tight horizontal zones or left-to-right across columns
    sorted_spans = sorted(spans, key=lambda s: (float(s["bbox"][0]), float(s["bbox"][1])))
    texts = [str(s["text"]).strip() for s in sorted_spans if str(s["text"]).strip()]

    group = ""
    pid = ""
    cause_tag = ""
    voting = ""
    note_refs: list[str] = []
    desc_spans: list[dict[str, Any]] = []

    for sp in sorted_spans:
        txt = str(sp["text"]).strip()
        if not txt:
            continue
        if _GROUP_CODE_RE.fullmatch(txt):
            group = txt
            continue
        if _PID_RE.fullmatch(txt):
            pid = txt.upper()
            continue
        if _VOTING_RE.fullmatch(txt):
            voting = txt
            continue
        note_match = _NOTE_REF_RE.fullmatch(txt)
        if note_match or (txt.upper().startswith("NOTE") and _NOTE_REF_RE.search(txt)):
            nm = _NOTE_REF_RE.search(txt)
            if nm:
                note_refs.append(f"NOTE {nm.group(1)}")
            continue
        if not cause_tag and _TAG_TOKEN_RE.fullmatch(txt) and not _PID_RE.fullmatch(txt):
            cause_tag = txt
            continue
        desc_spans.append(sp)

    # Order wrapped description lines top-to-bottom then left-to-right
    desc_spans_ordered = sorted(desc_spans, key=lambda s: (float(s["bbox"][1]), float(s["bbox"][0])))
    description = " ".join(str(s["text"]).strip() for s in desc_spans_ordered).strip()

    if not cause_tag:
        for txt in texts:
            m = _TAG_TOKEN_RE.search(txt)
            if m and not _PID_RE.fullmatch(m.group(0)) and not _GROUP_CODE_RE.fullmatch(m.group(0)):
                cause_tag = m.group(0)
                break
    if not cause_tag:
        cause_tag = description or pid or f"CAUSE_ROW_{row_idx + 1}"

    return {
        "row_index": row_idx + 1,
        "y_center": round(y_center, 2),
        "group": group,
        "pid": pid,
        "cause_tag": cause_tag,
        "description": description,
        "voting": voting,
        "note_refs": note_refs,
        "raw_labels": texts,
        "bbox": [
            round(min(float(s["bbox"][0]) for s in sorted_spans), 2),
            round(min(float(s["bbox"][1]) for s in sorted_spans), 2),
            round(max(float(s["bbox"][2]) for s in sorted_spans), 2),
            round(max(float(s["bbox"][3]) for s in sorted_spans), 2),
        ],
    }


def extract_rotated_ce_matrix(
    page: fitz.Page,
    page_number: int,
    legend_context: LegendContext | None = None,
) -> PageExtractionResult:
    """Reconstruct bidirectional Cause->Effect and Effect<-Cause records from rotated C&E sheets."""
    _, metrics = profile_pdf_page(page)
    text_dict = page.get_text("dict")
    words = page.get_text("words")

    if legend_context is None:
        legend_context = extract_legend_context([page], page_modes={1: "matrix_rotated_headers"})

    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
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
                x_mid = (float(bbox[0]) + float(bbox[2])) / 2.0
                y_mid = (float(bbox[1]) + float(bbox[3])) / 2.0
                item = {"text": txt, "bbox": bbox}
                if is_vert:
                    # Exclude outer margin plot timestamps and bottom-right title block rotated stamps
                    if x_mid < 28.0 or (y_mid > page_height - 95.0 and x_mid > page_width * 0.60):
                        continue
                    rotated_spans.append(item)
                elif is_horiz:
                    horiz_spans.append(item)

    # 1. Cluster rotated Effect column spans by x-center (dx <= 14 pt)
    rot_x_mids = [(float(s["bbox"][0]) + float(s["bbox"][2])) / 2.0 for s in rotated_spans]
    col_clusters = sorted(
        _cluster_1d(rot_x_mids, tol=14.0),
        key=lambda grp: sum(rot_x_mids[i] for i in grp) / len(grp),
    )
    effect_cols: list[dict[str, Any]] = []
    for c_idx, grp in enumerate(col_clusters):
        col_spans = [rotated_spans[i] for i in grp]
        x_center = sum(rot_x_mids[i] for i in grp) / len(grp)
        effect_cols.append(_parse_effect_column_spans(col_spans, c_idx, x_center))

    header_y_max = max((c["bbox"][3] for c in effect_cols), default=280.0)
    matrix_x_min = (min(c["x_center"] for c in effect_cols) - 40.0) if effect_cols else float(page.rect.width) * 0.4

    # 2. Identify horizontal matrix marks (X, O, R, P, B) inside grid body
    candidate_marks = [
        w
        for w in words
        if str(w[4]).strip() in ACTION_MARKS
        and ((float(w[0]) + float(w[2])) / 2.0) >= (matrix_x_min - 5.0)
        and ((float(w[1]) + float(w[3])) / 2.0) > (header_y_max - 5.0)
    ]

    # 3. Cluster left-hand horizontal Cause spans by y-center (dy <= 14 pt)
    left_spans = [
        s
        for s in horiz_spans
        if float(s["bbox"][2]) <= matrix_x_min + 12.0
        and ((float(s["bbox"][1]) + float(s["bbox"][3])) / 2.0) > (header_y_max - 5.0)
        and ((float(s["bbox"][1]) + float(s["bbox"][3])) / 2.0) < float(page.rect.height) - 95.0
    ]
    left_y_mids = [(float(s["bbox"][1]) + float(s["bbox"][3])) / 2.0 for s in left_spans]
    row_clusters = sorted(
        _cluster_1d(left_y_mids, tol=14.0),
        key=lambda grp: sum(left_y_mids[i] for i in grp) / len(grp),
    )

    cause_rows: list[dict[str, Any]] = []
    for grp in row_clusters:
        r_spans = [left_spans[i] for i in grp]
        y_center = sum(left_y_mids[i] for i in grp) / len(grp)
        joined_upper = " ".join(str(s["text"]).upper() for s in r_spans)
        # Skip the column header row for the left-hand table (GROUP | P&ID | CAUSE DESCRIPTION | ...)
        if "CAUSE DESCRIPTION" in joined_upper or "CAUSE TAG" in joined_upper:
            continue
        cause_rows.append(_parse_cause_row_spans(r_spans, len(cause_rows), y_center))

    # 4. Intersect marks with nearest (CauseRow, EffectColumn)
    col_hits: dict[int, list[dict[str, Any]]] = {c["col_index"]: [] for c in effect_cols}
    row_hits: dict[int, list[dict[str, Any]]] = {r["row_index"]: [] for r in cause_rows}

    if effect_cols and cause_rows:
        if len(cause_rows) >= 2:
            pitches = sorted(
                abs(cause_rows[i + 1]["y_center"] - cause_rows[i]["y_center"])
                for i in range(len(cause_rows) - 1)
            )
            median_pitch = pitches[len(pitches) // 2]
            row_tol = max(22.0, min(60.0, median_pitch * 0.60))
        else:
            row_tol = 28.0

        for mark_word in candidate_marks:
            mx = (float(mark_word[0]) + float(mark_word[2])) / 2.0
            my = (float(mark_word[1]) + float(mark_word[3])) / 2.0
            mark_sym = str(mark_word[4]).strip()

            nearest_col = min(effect_cols, key=lambda c: abs(mx - c["x_center"]))
            nearest_row = min(cause_rows, key=lambda r: abs(my - r["y_center"]))

            if abs(mx - nearest_col["x_center"]) > 38.0 or abs(my - nearest_row["y_center"]) > row_tol:
                continue

            meaning = legend_context.symbol_map.get(mark_sym, "") if legend_context else ""
            row_hits[nearest_row["row_index"]].append(
                {
                    "effect_tag": nearest_col["effect_tag"],
                    "effect_description": nearest_col["description"],
                    "pid": nearest_col["pid"],
                    "action": nearest_col["action"],
                    "mark": mark_sym,
                    "mark_meaning": meaning,
                    "notes": nearest_col["note_refs"],
                }
            )
            col_hits[nearest_col["col_index"]].append(
                {
                    "cause_tag": nearest_row["cause_tag"],
                    "group": nearest_row["group"],
                    "description": nearest_row["description"],
                    "pid": nearest_row["pid"],
                    "voting": nearest_row["voting"],
                    "mark": mark_sym,
                    "mark_meaning": meaning,
                    "notes": nearest_row["note_refs"],
                }
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
        all_note_refs = list(crow["note_refs"])
        for eff in effects:
            for nr in eff["notes"]:
                if nr not in all_note_refs:
                    all_note_refs.append(nr)
        inlined_notes = legend_context.lookup_notes(all_note_refs) if legend_context else []

        eff_strs: list[str] = []
        for eff in effects:
            meaning_part = f" ({eff['mark_meaning']})" if eff["mark_meaning"] else ""
            act_part = f" [{eff['action']}]" if eff["action"] else ""
            eff_strs.append(
                f"{eff['effect_tag']} ({eff['effect_description']}{act_part}) -> Mark '{eff['mark']}'{meaning_part}"
            )

        lines = [
            f"CAUSE TAG: {crow['cause_tag']} | GROUP: {crow['group']} | P&ID: {crow['pid']} | VOTING: {crow['voting'] or 'N/A'}",
            f"Cause Description: {crow['description']}",
            f"Triggered Effects ({len(effects)}): " + ("; ".join(eff_strs) if eff_strs else "None"),
        ]
        if inlined_notes:
            lines.append("Applicable Notes: " + " | ".join(inlined_notes))
        summary = "\n".join(lines)

        tags = _extract_record_tags(summary)
        records.append(
            EngineeringPageRecord(
                record_id=f"p{page_number}_cause_{crow['row_index']}",
                record_type="matrix_cause_row",
                page_number=page_number,
                title=f"Cause {crow['cause_tag']} ({crow['group']})",
                summary_text=summary,
                tags=tags,
                bbox=crow["bbox"],
                metadata={
                    "cause_tag": crow["cause_tag"],
                    "group": crow["group"],
                    "pid": crow["pid"],
                    "voting": crow["voting"],
                    "description": crow["description"],
                    "note_refs": all_note_refs,
                    "inlined_notes": inlined_notes,
                    "effects": effects,
                },
            )
        )
        md_sections.append(f"### Cause: {crow['cause_tag']}\n{summary}")

    md_sections.append("## Reverse Effect-to-Cause Mapping")
    for ecol in effect_cols:
        causes = col_hits.get(ecol["col_index"], [])
        all_note_refs = list(ecol["note_refs"])
        for c_item in causes:
            for nr in c_item["notes"]:
                if nr not in all_note_refs:
                    all_note_refs.append(nr)
        inlined_notes = legend_context.lookup_notes(all_note_refs) if legend_context else []

        cause_strs: list[str] = []
        for c_item in causes:
            meaning_part = f" ({c_item['mark_meaning']})" if c_item["mark_meaning"] else ""
            vote_part = f" [{c_item['voting']}]" if c_item["voting"] else ""
            cause_strs.append(
                f"{c_item['cause_tag']} ({c_item['group']}: {c_item['description']}{vote_part}) -> Mark '{c_item['mark']}'{meaning_part}"
            )

        lines = [
            f"EFFECT TAG: {ecol['effect_tag']} | P&ID: {ecol['pid']} | ACTION: {ecol['action'] or 'N/A'}",
            f"Effect Description: {ecol['description']}",
            f"Triggering Causes ({len(causes)}): " + ("; ".join(cause_strs) if cause_strs else "None"),
        ]
        if inlined_notes:
            lines.append("Applicable Notes: " + " | ".join(inlined_notes))
        summary = "\n".join(lines)

        tags = _extract_record_tags(summary)
        records.append(
            EngineeringPageRecord(
                record_id=f"p{page_number}_effect_{ecol['col_index']}",
                record_type="matrix_effect_col",
                page_number=page_number,
                title=f"Effect {ecol['effect_tag']} ({ecol['action'] or 'Action'})",
                summary_text=summary,
                tags=tags,
                bbox=ecol["bbox"],
                metadata={
                    "effect_tag": ecol["effect_tag"],
                    "pid": ecol["pid"],
                    "description": ecol["description"],
                    "action": ecol["action"],
                    "note_refs": all_note_refs,
                    "inlined_notes": inlined_notes,
                    "causes": causes,
                },
            )
        )
        md_sections.append(f"### Effect: {ecol['effect_tag']}\n{summary}")

    return PageExtractionResult(
        page_number=page_number,
        mode="matrix_rotated_headers",
        metrics=metrics,
        records=records,
        markdown="\n\n".join(md_sections).strip() + "\n",
        legend_context=legend_context,
    )


def extract_cad_logic_sheet(page: fitz.Page, page_number: int) -> PageExtractionResult:
    """Group POU title, cross-references, inputs (From), logic blocks, and outputs (To) on CAD logic sheets."""
    _, metrics = profile_pdf_page(page)
    text_dict = page.get_text("dict")
    raw_text = page.get_text("text") or ""
    width = float(page.rect.width)
    height = float(page.rect.height)

    spans: list[dict[str, Any]] = []
    for block in text_dict.get("blocks", ()):
        for line in block.get("lines", ()):
            for span in line.get("spans", ()):
                txt = (span.get("text") or "").strip()
                if txt:
                    spans.append({"text": txt, "bbox": span["bbox"]})

    # Extract POU / sheet title and cross-references
    pou_match = _POU_TITLE_RE.search(raw_text)
    pou_title = pou_match.group(0).strip() if pou_match else f"CAD Logic Diagram (Page {page_number})"
    cross_refs = [m.strip() for m in _CROSS_REF_RE.findall(raw_text) if m.strip()]

    inputs: list[str] = []
    logic_blocks: list[str] = []
    outputs: list[str] = []

    for sp in sorted(spans, key=lambda s: (float(s["bbox"][1]), float(s["bbox"][0]))):
        txt = sp["text"]
        x_mid = (float(sp["bbox"][0]) + float(sp["bbox"][2])) / 2.0
        y_mid = (float(sp["bbox"][1]) + float(sp["bbox"][3])) / 2.0

        # Skip bottom-right title block and top title header
        if y_mid < 85.0 or (y_mid > height - 115.0 and x_mid > width - 400.0):
            continue

        if re.match(r"^From\s*:", txt, re.IGNORECASE):
            val = re.sub(r"^From\s*:\s*", "", txt, flags=re.IGNORECASE).strip()
            if val and val not in inputs:
                inputs.append(val)
            continue

        if re.match(r"^To\s*:", txt, re.IGNORECASE):
            val = re.sub(r"^To\s*:\s*", "", txt, flags=re.IGNORECASE).strip()
            if val and val not in outputs:
                outputs.append(val)
            continue

        if _LOGIC_GATE_RE.fullmatch(txt):
            logic_blocks.append(txt.upper())
            continue

        # Positional fallback when explicit From:/To: prefixes are omitted
        if x_mid < width * 0.30 and _TAG_TOKEN_RE.search(txt):
            for t in _extract_record_tags(txt):
                if t not in inputs:
                    inputs.append(t)
        elif x_mid > width * 0.72 and _TAG_TOKEN_RE.search(txt):
            for t in _extract_record_tags(txt):
                if t not in outputs:
                    outputs.append(t)

    summary_lines = [
        f"POU / Sheet: {pou_title}",
        f"Cross-References: {', '.join(cross_refs) if cross_refs else 'None'}",
        f"Input Signals (From): {', '.join(inputs) if inputs else 'None'}",
        f"Logic Function Blocks: {', '.join(logic_blocks) if logic_blocks else 'None'}",
        f"Output Signals (To): {', '.join(outputs) if outputs else 'None'}",
    ]
    summary_text = "\n".join(summary_lines)
    tags = _extract_record_tags(f"{pou_title}\n{summary_text}")

    record = EngineeringPageRecord(
        record_id=f"p{page_number}_logic_1",
        record_type="cad_logic_diagram",
        page_number=page_number,
        title=pou_title,
        summary_text=summary_text,
        tags=tags,
        bbox=[0.0, 0.0, round(width, 2), round(height, 2)],
        metadata={
            "pou_title": pou_title,
            "cross_references": cross_refs,
            "inputs": inputs,
            "logic_blocks": logic_blocks,
            "outputs": outputs,
        },
    )

    markdown = (
        f"<!-- page: {page_number} -->\n"
        f"# {pou_title} (Page {page_number})\n\n"
        f"{summary_text}\n"
    )
    return PageExtractionResult(
        page_number=page_number,
        mode="cad_logic_diagram",
        metrics=metrics,
        records=[record],
        markdown=markdown,
    )


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
    # Oversized sheet where min_dpi itself would breach max_edge_px: honor hard max_edge_px cap
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

    # Prefer higher confidence and longer text when resolving overlap duplicates
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

    # Sort top-to-bottom in line bands (12 pt), then left-to-right
    kept.sort(key=lambda d: (round(float(d["bbox"][1]) / 12.0), float(d["bbox"][0])))
    return kept


def _normalize_ocr_output(raw_output: Any) -> list[tuple[list[float], str, float]]:
    """Normalize RapidOCR v3 RapidOCROutput or legacy list/tuple OCR output into [(bbox, text, score)]."""
    if raw_output is None:
        return []

    # RapidOCR v3 RapidOCROutput object with .boxes, .txts, .scores
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

    # Legacy RapidOCR / custom mock tuple: (results_list, elapse) or raw list
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
) -> PageExtractionResult:
    """Run CPU 2x2 overlapping tiled OCR (+ title-block crop) on exploded-SHX CAD pages."""
    _, metrics = profile_pdf_page(page)
    width_pt = float(page.rect.width)
    height_pt = float(page.rect.height)

    render_dpi = compute_safe_ocr_dpi(width_pt, height_pt, target_dpi=300, min_dpi=150, max_edge_px=4800)
    pix = page.get_pixmap(dpi=render_dpi, colorspace=fitz.csRGB, alpha=False)
    img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    img_h, img_w = img_arr.shape[:2]

    # 2x2 grid with 15% total seam overlap + bottom-right title-block crop
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

    engine = ocr_engine if ocr_engine is not None else get_rapidocr_engine()
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

    # Assemble spatial text blocks grouped by horizontal line bands
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

    metrics = {
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
        metrics=metrics,
        records=records,
        markdown=markdown,
    )


def _build_legend_page_result(
    page: fitz.Page,
    page_number: int,
    metrics: dict[str, Any],
    legend_ctx: LegendContext,
) -> PageExtractionResult:
    """Create a PageExtractionResult and EngineeringPageRecord for a Legend & Notes page."""
    raw_text = (page.get_text("text") or "").strip()
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

        page_results: list[PageExtractionResult] = []
        for idx in range(page_count):
            pno = idx + 1
            page = doc[idx]
            mode = page_modes[pno]
            if mode == "coord_grid_schedule":
                res = extract_coord_grid_schedule(page, pno)
            elif mode == "matrix_rotated_headers":
                res = extract_rotated_ce_matrix(page, pno, legend_context=legend_ctx)
            elif mode == "cad_logic_diagram":
                res = extract_cad_logic_sheet(page, pno)
            elif mode == "exploded_cad_ocr":
                res = extract_exploded_cad_ocr(page, pno, ocr_engine=ocr_engine)
            elif mode == "document_legend_notes":
                res = _build_legend_page_result(page, pno, page_metrics[pno], legend_ctx)
            else:
                prose_text = (page.get_text("text") or "").strip()
                md = f"<!-- page: {pno} -->\n{prose_text}\n" if prose_text else f"<!-- page: {pno} -->\n"
                res = PageExtractionResult(
                    page_number=pno,
                    mode="standard_prose",
                    metrics=page_metrics[pno],
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
