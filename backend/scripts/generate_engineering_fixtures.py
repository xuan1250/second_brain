"""Generate or slice compact engineering CAD PDF fixtures for deterministic offline testing.

Usage:
    python backend/scripts/generate_engineering_fixtures.py
    python backend/scripts/generate_engineering_fixtures.py --source-dir /path/to/pdfs --out-dir backend/tests/fixtures/engineering
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pymupdf as fitz

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "engineering"

PAGE_WIDTH = 1190.0
PAGE_HEIGHT = 842.0

# Clean vector stroke glyphs on a normalized (0..1, 0..1) box (origin top-left, y downward)
# Designed with high stroke clarity and spacing so CPU RapidOCR reads exploded-SHX text accurately.
_STROKE_GLYPHS: dict[str, list[list[tuple[float, float]]]] = {
    "A": [[(0.05, 1.0), (0.5, 0.0), (0.95, 1.0)], [(0.22, 0.65), (0.78, 0.65)]],
    "B": [
        [(0.1, 0.0), (0.1, 1.0)],
        [(0.1, 0.0), (0.72, 0.0), (0.9, 0.14), (0.9, 0.34), (0.72, 0.48), (0.1, 0.48)],
        [(0.72, 0.48), (0.92, 0.62), (0.92, 0.86), (0.72, 1.0), (0.1, 1.0)],
    ],
    "C": [[(0.85, 0.22), (0.68, 0.02), (0.3, 0.02), (0.1, 0.24), (0.1, 0.76), (0.3, 0.98), (0.68, 0.98), (0.85, 0.78)]],
    "D": [[(0.1, 0.0), (0.1, 1.0)], [(0.1, 0.0), (0.65, 0.0), (0.92, 0.22), (0.92, 0.78), (0.65, 1.0), (0.1, 1.0)]],
    "E": [[(0.9, 0.0), (0.1, 0.0), (0.1, 1.0), (0.9, 1.0)], [(0.1, 0.48), (0.75, 0.48)]],
    "F": [[(0.9, 0.0), (0.1, 0.0), (0.1, 1.0)], [(0.1, 0.48), (0.75, 0.48)]],
    "G": [[(0.85, 0.22), (0.68, 0.02), (0.32, 0.02), (0.1, 0.24), (0.1, 0.76), (0.32, 0.98), (0.75, 0.98), (0.88, 0.76), (0.88, 0.54), (0.52, 0.54)]],
    "H": [[(0.1, 0.0), (0.1, 1.0)], [(0.9, 0.0), (0.9, 1.0)], [(0.1, 0.48), (0.9, 0.48)]],
    "I": [[(0.25, 0.0), (0.75, 0.0)], [(0.5, 0.0), (0.5, 1.0)], [(0.25, 1.0), (0.75, 1.0)]],
    "J": [[(0.82, 0.0), (0.82, 0.78), (0.62, 1.0), (0.28, 1.0), (0.1, 0.8)]],
    "K": [[(0.12, 0.0), (0.12, 1.0)], [(0.88, 0.0), (0.12, 0.52), (0.9, 1.0)]],
    "L": [[(0.16, 0.0), (0.16, 1.0), (0.88, 1.0)]],
    "M": [[(0.08, 1.0), (0.08, 0.0), (0.5, 0.58), (0.92, 0.0), (0.92, 1.0)]],
    "N": [[(0.1, 1.0), (0.1, 0.0), (0.9, 1.0), (0.9, 0.0)]],
    "O": [[(0.32, 0.0), (0.68, 0.0), (0.9, 0.22), (0.9, 0.78), (0.68, 1.0), (0.32, 1.0), (0.1, 0.78), (0.1, 0.22), (0.32, 0.0)]],
    "P": [[(0.12, 1.0), (0.12, 0.0), (0.7, 0.0), (0.88, 0.14), (0.88, 0.36), (0.7, 0.5), (0.12, 0.5)]],
    "R": [
        [(0.12, 1.0), (0.12, 0.0), (0.7, 0.0), (0.88, 0.14), (0.88, 0.35), (0.7, 0.48), (0.12, 0.48)],
        [(0.5, 0.48), (0.9, 1.0)],
    ],
    "S": [[(0.84, 0.16), (0.68, 0.02), (0.32, 0.02), (0.14, 0.16), (0.14, 0.34), (0.34, 0.46), (0.68, 0.52), (0.88, 0.66), (0.88, 0.84), (0.68, 0.98), (0.32, 0.98), (0.12, 0.84)]],
    "T": [[(0.06, 0.0), (0.94, 0.0)], [(0.5, 0.0), (0.5, 1.0)]],
    "U": [[(0.12, 0.0), (0.12, 0.76), (0.32, 0.98), (0.68, 0.98), (0.88, 0.76), (0.88, 0.0)]],
    "V": [[(0.08, 0.0), (0.5, 1.0), (0.92, 0.0)]],
    "W": [[(0.04, 0.0), (0.27, 1.0), (0.5, 0.35), (0.73, 1.0), (0.96, 0.0)]],
    "X": [[(0.1, 0.0), (0.9, 1.0)], [(0.9, 0.0), (0.1, 1.0)]],
    "Y": [[(0.08, 0.0), (0.5, 0.48), (0.92, 0.0)], [(0.5, 0.48), (0.5, 1.0)]],
    "Z": [[(0.08, 0.0), (0.92, 0.0), (0.08, 1.0), (0.92, 1.0)]],
    "0": [
        [(0.32, 0.0), (0.68, 0.0), (0.88, 0.22), (0.88, 0.78), (0.68, 1.0), (0.32, 1.0), (0.12, 0.78), (0.12, 0.22), (0.32, 0.0)],
    ],
    "1": [[(0.26, 0.2), (0.54, 0.0), (0.54, 1.0)], [(0.2, 1.0), (0.86, 1.0)]],
    "2": [[(0.12, 0.22), (0.3, 0.0), (0.72, 0.0), (0.9, 0.18), (0.9, 0.38), (0.1, 1.0), (0.92, 1.0)]],
    "3": [[(0.12, 0.15), (0.3, 0.0), (0.72, 0.0), (0.9, 0.16), (0.9, 0.35), (0.68, 0.48), (0.42, 0.48)], [(0.68, 0.48), (0.9, 0.62), (0.9, 0.84), (0.72, 1.0), (0.3, 1.0), (0.12, 0.84)]],
    "4": [[(0.72, 1.0), (0.72, 0.0), (0.08, 0.68), (0.94, 0.68)]],
    "5": [[(0.88, 0.0), (0.16, 0.0), (0.14, 0.44), (0.7, 0.44), (0.9, 0.6), (0.9, 0.84), (0.7, 1.0), (0.24, 1.0), (0.1, 0.85)]],
    "6": [[(0.82, 0.1), (0.64, 0.0), (0.32, 0.0), (0.1, 0.24), (0.1, 0.8), (0.32, 1.0), (0.72, 1.0), (0.9, 0.8), (0.9, 0.56), (0.72, 0.42), (0.12, 0.42)]],
    "7": [[(0.1, 0.0), (0.92, 0.0), (0.36, 1.0)]],
    "8": [
        [(0.3, 0.48), (0.12, 0.35), (0.12, 0.14), (0.3, 0.0), (0.7, 0.0), (0.88, 0.14), (0.88, 0.35), (0.7, 0.48), (0.3, 0.48)],
        [(0.3, 0.48), (0.1, 0.64), (0.1, 0.86), (0.3, 1.0), (0.7, 1.0), (0.9, 0.86), (0.9, 0.64), (0.7, 0.48)],
    ],
    "9": [[(0.18, 0.9), (0.36, 1.0), (0.68, 1.0), (0.9, 0.76), (0.9, 0.2), (0.68, 0.0), (0.28, 0.0), (0.1, 0.2), (0.1, 0.44), (0.28, 0.58), (0.88, 0.58)]],
    "-": [[(0.15, 0.52), (0.85, 0.52)]],
    "_": [[(0.05, 0.95), (0.95, 0.95)]],
    ".": [[(0.46, 0.92), (0.54, 0.92), (0.54, 1.0), (0.46, 1.0), (0.46, 0.92)]],
    "/": [[(0.18, 1.0), (0.82, 0.0)]],
}


def _draw_cad_frame(page: fitz.Page, title_lines: Sequence[str], doc_no: str) -> None:
    """Draw outer CAD border and bottom-right title block using individual vector primitives."""
    w, h = page.rect.width, page.rect.height
    page.draw_rect(fitz.Rect(20, 20, w - 20, h - 20), color=(0, 0, 0), width=1.4)
    page.draw_rect(fitz.Rect(28, 28, w - 28, h - 28), color=(0, 0, 0), width=0.8)

    # Bottom-right CAD title block
    tb_x0, tb_y0, tb_x1, tb_y1 = w - 390, h - 108, w - 28, h - 28
    page.draw_rect(fitz.Rect(tb_x0, tb_y0, tb_x1, tb_y1), color=(0, 0, 0), width=1.0)
    page.draw_line((tb_x0, tb_y0 + 28), (tb_x1, tb_y0 + 28), color=(0, 0, 0), width=0.7)
    page.draw_line((tb_x0, tb_y0 + 54), (tb_x1, tb_y0 + 54), color=(0, 0, 0), width=0.7)
    page.draw_line((tb_x0 + 185, tb_y0 + 54), (tb_x0 + 185, tb_y1), color=(0, 0, 0), width=0.7)

    if title_lines:
        page.insert_text((tb_x0 + 10, tb_y0 + 19), title_lines[0], fontsize=10, fontname="helv")
    if len(title_lines) > 1:
        page.insert_text((tb_x0 + 10, tb_y0 + 44), title_lines[1], fontsize=8.5, fontname="helv")
    page.insert_text((tb_x0 + 10, tb_y0 + 70), f"DOC: {doc_no}", fontsize=8, fontname="helv")
    if len(title_lines) > 2:
        page.insert_text((tb_x0 + 195, tb_y0 + 70), title_lines[2], fontsize=8, fontname="helv")


def build_cabinet_wiring_pdf(out_path: Path) -> Path:
    """Build 1-page landscape 1190x842 CABINET WIRING DRAWING (>160 vector lines, >=10 cols, 18 rows)."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)

    # Top title banner
    page.draw_rect(fitz.Rect(40, 36, 1150, 86), color=(0, 0, 0), width=1.0)
    page.insert_text((52, 62), "CABINET WIRING DRAWING - LNG-CCR-DCS-MC01 (FS12 / RS31)", fontsize=13, fontname="helv")
    page.insert_text((880, 62), "SHEET 2 OF 12 | REV 04", fontsize=10, fontname="helv")

    _draw_cad_frame(
        page,
        ["CABINET WIRING DRAWING", "CABINET: LNG-CCR-DCS-MC01", "NODE: FS12 / RS31"],
        "LNG-INST-CW-002",
    )

    columns = [
        ("ITEM", 40, 90),
        ("FIELD TAG", 90, 195),
        ("DCS TAG", 195, 300),
        ("TERMINAL BLOCK", 300, 420),
        ("TB NO", 420, 485),
        ("FCS NODE", 485, 580),
        ("IO MODULE", 580, 680),
        ("CHANNEL", 680, 760),
        ("SERVICE DESCRIPTION", 760, 1005),
        ("REMARKS", 1005, 1150),
    ]

    rows = [
        ("01", "TT-10205A", "TI10205A", "1FTB1", "1-2", "FCS0201", "AAI143-H", "CH-01", "LNG TANK TOP TEMP", "RTD 3-WIRE"),
        ("02", "FT-10507", "FIC10507", "1FTB2", "3-4", "FCS0201", "AAI143-H", "CH-02", "LNG SEND OUT FLOW", "4-20mA HART"),
        ("03", "PT-10803", "PIA30101", "1RTB1", "5-6", "SCS0211", "AAV144-S", "CH-03", "SUCTION HEADER PRESS", "4-20mA"),
        ("04", "TT-10205B", "TI10205B", "1FTB1", "7-8", "FCS0201", "AAI143-H", "CH-04", "LNG TANK MID TEMP", "RTD 3-WIRE"),
        ("05", "LT-80101A", "LIA80101A", "1RTB2", "1-2", "SCS0211", "AAI143-H", "CH-05", "LNG TANK LEVEL CH A", "ESD 2oo3"),
        ("06", "LT-80101B", "LIA80101B", "1RTB2", "3-4", "SCS0211", "AAI143-H", "CH-06", "LNG TANK LEVEL CH B", "ESD 2oo3"),
        ("07", "LT-80101C", "LIA80101C", "1RTB2", "5-6", "SCS0211", "AAI143-H", "CH-07", "LNG TANK LEVEL CH C", "ESD 2oo3"),
        ("08", "PT-14261A", "PIA14261A", "1RTB3", "1-2", "SCS0211", "AAV144-S", "CH-08", "HP PUMP SUCTION PRESS A", "ESD 2oo3"),
        ("09", "PT-14261B", "PIA14261B", "1RTB3", "3-4", "SCS0211", "AAV144-S", "CH-09", "HP PUMP SUCTION PRESS B", "ESD 2oo3"),
        ("10", "PT-14261C", "PIA14261C", "1RTB3", "5-6", "SCS0211", "AAV144-S", "CH-10", "HP PUMP SUCTION PRESS C", "ESD 2oo3"),
        ("11", "FT-20231", "FIC20231", "1FTB3", "1-2", "FCS0201", "AAI143-H", "CH-11", "PUMP RECIRC FLOW", "4-20mA HART"),
        ("12", "PT-10804", "PIC10804", "1FTB3", "3-4", "FCS0201", "AAI143-H", "CH-12", "VENT HEADER PRESS", "4-20mA HART"),
        ("13", "TT-10512", "TI10512", "1FTB4", "1-2", "FCS0202", "AAI143-H", "CH-13", "BOG SUCTION TEMP", "RTD 3-WIRE"),
        ("14", "PDT-30504", "PDIA30504", "1RTB4", "1-2", "SCS0211", "AAI143-H", "CH-14", "BOG COMP DIFF PRESS", "PSD INTERLOCK"),
        ("15", "ZT-10101A", "ZI10101A", "1FTB5", "1-2", "FCS0202", "ADV151-P", "CH-15", "INLET ESDV LIMIT SW", "24VDC DI"),
        ("16", "ZT-10101B", "ZI10101B", "1FTB5", "3-4", "FCS0202", "ADV151-P", "CH-16", "OUTLET ESDV LIMIT SW", "24VDC DI"),
        ("17", "XT-10411", "XI10411", "1FTB6", "1-2", "FCS0202", "AAV144-S", "CH-17", "VAPORIZER OUTLET FLOW", "4-20mA"),
        ("18", "VT-30321A", "VI30321A", "1RTB5", "1-2", "SCS0211", "AAI143-H", "CH-18", "COMPRESSOR VIBRATION", "PROXIMITY"),
    ]

    header_y0 = 102.0
    header_h = 28.0
    row_h = 28.0

    # Draw header cells individually (>160 drawings across table)
    for col_name, cx0, cx1 in columns:
        page.draw_rect(fitz.Rect(cx0, header_y0, cx1, header_y0 + header_h), color=(0, 0, 0), width=0.9)
        page.insert_text((cx0 + 5, header_y0 + 18), col_name, fontsize=8.5, fontname="helv")

    for r_idx, row_vals in enumerate(rows):
        ry0 = header_y0 + header_h + r_idx * row_h
        ry1 = ry0 + row_h
        for (_, cx0, cx1), val in zip(columns, row_vals):
            page.draw_rect(fitz.Rect(cx0, ry0, cx1, ry1), color=(0, 0, 0), width=0.55)
            page.insert_text((cx0 + 5, ry0 + 18), val, fontsize=8.5, fontname="helv")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path), garbage=4, deflate=True)
    doc.close()
    return out_path


def build_process_ce_chart_pdf(out_path: Path) -> Path:
    """Build 2-page landscape 1190x842 Process Cause & Effect Chart (Page 1 Legend, Page 2 Matrix)."""
    doc = fitz.open()

    # Page 1: document_legend_notes
    p1 = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    _draw_cad_frame(
        p1,
        ["PROCESS CAUSE AND EFFECT CHART - LEGEND AND NOTES", "LNG TERMINAL SAFETY SYSTEMS", "PAGE 1 OF 2"],
        "LNG-PROC-CE-001",
    )
    p1.insert_text((50, 72), "PROCESS CAUSE AND EFFECT CHART - LEGEND AND NOTES", fontsize=15, fontname="helv")
    p1.draw_line((50, 82), (760, 82), color=(0, 0, 0), width=1.2)

    p1.insert_text((50, 120), "LEGEND - ACTION SYMBOLS", fontsize=11.5, fontname="helv")
    legend_rows = [
        ("X", "CLOSE / TRIP / ACTIVE SIGNAL"),
        ("O", "OPEN"),
        ("P", "PERMISSIVE"),
        ("R", "RESET"),
        ("B", "BYPASS / OVERRIDE"),
    ]
    for idx, (sym, desc) in enumerate(legend_rows):
        y0 = 135 + idx * 32
        p1.draw_rect(fitz.Rect(50, y0, 110, y0 + 28), color=(0, 0, 0), width=0.8)
        p1.draw_rect(fitz.Rect(110, y0, 560, y0 + 28), color=(0, 0, 0), width=0.8)
        p1.insert_text((74, y0 + 19), sym, fontsize=10, fontname="helv")
        p1.insert_text((124, y0 + 19), f"{sym} = {desc}", fontsize=10, fontname="helv")

    p1.insert_text((50, 340), "GENERAL NOTES", fontsize=11.5, fontname="helv")
    notes = [
        "NOTE 1: ESD-1 shutdown initiates immediate closure of isolation valves and stops cryogenic pumps.",
        "NOTE 2: Trip signals shall latch in the safety PLC until all process conditions return to normal.",
        "NOTE 36: Requires manual reset from CCR after fault clearance and permissive verification.",
    ]
    for idx, note_text in enumerate(notes):
        ny = 370 + idx * 34
        p1.draw_rect(fitz.Rect(50, ny - 18, 1080, ny + 10), color=(0, 0, 0), width=0.6)
        p1.insert_text((60, ny), note_text, fontsize=10, fontname="helv")

    # Page 2: matrix_rotated_headers
    p2 = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    _draw_cad_frame(
        p2,
        ["PROCESS CAUSE AND EFFECT CHART - MATRIX", "ESD / FGS / PSD INTERLOCKS", "PAGE 2 OF 2"],
        "LNG-PROC-CE-001",
    )
    p2.insert_text((45, 58), "PROCESS CAUSE AND EFFECT CHART - INTERLOCK MATRIX (SHEET 2)", fontsize=13, fontname="helv")

    # Left cause columns
    cause_cols = [
        ("GROUP", 40, 90),
        ("P&ID", 90, 150),
        ("CAUSE DESCRIPTION", 150, 295),
        ("CAUSE TAG", 295, 385),
        ("VOTING", 385, 438),
        ("NOTE", 438, 480),
    ]
    # 6 rotated Effect columns (>= 60 pt spacing; centers at 520, 600, 680, 760, 840, 920)
    effect_cols = [
        {
            "x_center": 520.0,
            "pid": "PID-0101",
            "tag": "XV-10101A",
            "desc": "LNG Inlet ESDV",
            "action": "Close",
            "note": "NOTE 1",
        },
        {
            "x_center": 600.0,
            "pid": "PID-0101",
            "tag": "XV-10101B",
            "desc": "LNG Outlet ESDV",
            "action": "Close",
            "note": "NOTE 1",
        },
        {
            "x_center": 680.0,
            "pid": "PID-0102",
            "tag": "HA-10003B",
            "desc": "CCR Trip Alarm",
            "action": "Alarm activation",
            "note": "",
        },
        {
            "x_center": 760.0,
            "pid": "PID-0103",
            "tag": "FCV-20231",
            "desc": "Pump Recirc Valve",
            "action": "Open",
            "note": "",
        },
        {
            "x_center": 840.0,
            "pid": "PID-0104",
            "tag": "PCV-10803",
            "desc": "Vent Control Valve",
            "action": "Close",
            "note": "NOTE 36",
        },
        {
            "x_center": 920.0,
            "pid": "PID-0105",
            "tag": "UZA30321A",
            "desc": "Compressor Trip Output",
            "action": "Trip",
            "note": "",
        },
    ]

    header_top = 80.0
    header_bottom = 350.0
    col_half_w = 40.0  # 80 pt wide columns

    # Draw rotated Effect headers (dir=(0, -1)); vertical sub-segments avoid overlap while staying on one x per column
    for ecol in effect_cols:
        cx = ecol["x_center"]
        p2.draw_rect(fitz.Rect(cx - col_half_w, header_top, cx + col_half_w, header_bottom), color=(0, 0, 0), width=0.8)
        # Horizontal sub-band ticks in header column
        for tick_y in (128.0, 215.0, 278.0, 322.0):
            p2.draw_line((cx - col_half_w, tick_y), (cx + col_half_w, tick_y), color=(0, 0, 0), width=0.45)

        # Insert 90-deg rotated text (grows upward from insertion point)
        p2.insert_text((cx + 3, 124.0), ecol["pid"], fontsize=8.0, fontname="helv", rotate=90)
        p2.insert_text((cx + 3, 210.0), ecol["desc"], fontsize=8.0, fontname="helv", rotate=90)
        p2.insert_text((cx + 3, 274.0), ecol["tag"], fontsize=8.5, fontname="helv", rotate=90)
        p2.insert_text((cx + 3, 318.0), ecol["action"], fontsize=8.0, fontname="helv", rotate=90)
        if ecol["note"]:
            p2.insert_text((cx + 3, 347.0), ecol["note"], fontsize=7.5, fontname="helv", rotate=90)

    # Left cause header row right above body rows (y=322..350)
    for cname, cx0, cx1 in cause_cols:
        p2.draw_rect(fitz.Rect(cx0, 322.0, cx1, header_bottom), color=(0, 0, 0), width=0.8)
        p2.insert_text((cx0 + 4, 340.0), cname, fontsize=8.0, fontname="helv")

    cause_rows = [
        {
            "y_center": 380.0,
            "group": "ESD-1",
            "pid": "PID-0101",
            "desc_lines": ["LNG Tank High-High", "Level"],
            "tag": "LAHH-80101A/B/C",
            "voting": "2 out of 3",
            "note": "NOTE 1",
            "marks": {0: "X", 1: "X", 2: "X", 3: "O"},
        },
        {
            "y_center": 440.0,
            "group": "ESD-1",
            "pid": "PID-0102",
            "desc_lines": ["Manual Emergency", "Shutdown Pushbutton"],
            "tag": "HZ-10001",
            "voting": "1 out of 1",
            "note": "NOTE 36",
            "marks": {0: "X", 2: "X", 4: "X", 5: "X"},
        },
        {
            "y_center": 500.0,
            "group": "ESD-2",
            "pid": "PID-0103",
            "desc_lines": ["HP Pump Suction", "Pressure Low-Low"],
            "tag": "PALL-14261A/B/C",
            "voting": "2 out of 3",
            "note": "",
            "marks": {2: "X", 3: "O"},
        },
        {
            "y_center": 560.0,
            "group": "FGS",
            "pid": "PID-0104",
            "desc_lines": ["Fire & Gas Confirmed", "Area ESD"],
            "tag": "UA-FGS-ESD-1",
            "voting": "2ooN",
            "note": "NOTE 1",
            "marks": {0: "X", 4: "X", 5: "X"},
        },
        {
            "y_center": 620.0,
            "group": "PSD",
            "pid": "PID-0105",
            "desc_lines": ["BOG Compressor Diff", "Pressure Low-Low"],
            "tag": "PDALL-30504",
            "voting": "1 out of 1",
            "note": "",
            "marks": {2: "P", 5: "X"},
        },
    ]

    row_half_h = 30.0
    for crow in cause_rows:
        yc = crow["y_center"]
        y0, y1 = yc - row_half_h, yc + row_half_h
        left_vals = [
            crow["group"],
            crow["pid"],
            crow["desc_lines"],
            crow["tag"],
            crow["voting"],
            crow["note"],
        ]
        for (_, cx0, cx1), val in zip(cause_cols, left_vals):
            p2.draw_rect(fitz.Rect(cx0, y0, cx1, y1), color=(0, 0, 0), width=0.6)
            if isinstance(val, list):
                p2.insert_text((cx0 + 4, yc - 2), val[0], fontsize=8.0, fontname="helv")
                p2.insert_text((cx0 + 4, yc + 9), val[1], fontsize=8.0, fontname="helv")
            elif val:
                p2.insert_text((cx0 + 4, yc + 3), val, fontsize=8.0, fontname="helv")

        for col_idx, ecol in enumerate(effect_cols):
            cx = ecol["x_center"]
            p2.draw_rect(fitz.Rect(cx - col_half_w, y0, cx + col_half_w, y1), color=(0, 0, 0), width=0.6)
            mark = crow["marks"].get(col_idx)
            if mark:
                p2.insert_text((cx - 3, yc + 4), mark, fontsize=10.5, fontname="helv")

    # Add crisp grid cross-hatch ticks and spare matrix columns/rows to guarantee > 160 drawings
    for extra_col in range(6, 10):
        ex0 = 960.0 + (extra_col - 6) * 45.0
        ex1 = ex0 + 45.0
        p2.draw_rect(fitz.Rect(ex0, header_top, ex1, header_bottom), color=(0, 0, 0), width=0.5)
        for crow in cause_rows:
            yc = crow["y_center"]
            p2.draw_rect(fitz.Rect(ex0, yc - row_half_h, ex1, yc + row_half_h), color=(0, 0, 0), width=0.45)

    for extra_row in range(4):
        ey0 = 650.0 + extra_row * 22.0
        ey1 = ey0 + 22.0
        for _, cx0, cx1 in cause_cols:
            p2.draw_rect(fitz.Rect(cx0, ey0, cx1, ey1), color=(0, 0, 0), width=0.45)
        for ecol in effect_cols:
            cx = ecol["x_center"]
            p2.draw_rect(fitz.Rect(cx - col_half_w, ey0, cx + col_half_w, ey1), color=(0, 0, 0), width=0.45)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path), garbage=4, deflate=True)
    doc.close()
    return out_path


def _draw_logic_page(
    page: fitz.Page,
    *,
    pou_title: str,
    cross_ref: str,
    inputs: Sequence[str],
    blocks: Sequence[tuple[str, float, float]],
    outputs: Sequence[str],
    doc_no: str,
    sheet_label: str,
) -> None:
    """Draw an ESD FBD logic page with >160 vector lines/boxes, inputs, gate blocks, and outputs."""
    _draw_cad_frame(page, ["ESD LOGIC DIAGRAM", pou_title, sheet_label], doc_no)

    # Header bar
    page.draw_rect(fitz.Rect(40, 36, 1150, 82), color=(0, 0, 0), width=1.0)
    page.insert_text((54, 64), f"{pou_title}   {cross_ref}", fontsize=13.5, fontname="helv")
    page.insert_text((920, 64), "ESD LOGIC DIAGRAM", fontsize=11, fontname="helv")

    # Background CAD logic grid ticks (>160 drawings)
    for gx in range(50, 1140, 45):
        page.draw_line((gx, 90), (gx, 96), color=(0.4, 0.4, 0.4), width=0.4)
        page.draw_line((gx, 720), (gx, 726), color=(0.4, 0.4, 0.4), width=0.4)
    for gy in range(110, 710, 40):
        page.draw_line((40, gy), (48, gy), color=(0.4, 0.4, 0.4), width=0.4)
        page.draw_line((1142, gy), (1150, gy), color=(0.4, 0.4, 0.4), width=0.4)

    # Left input terminals & wiring lines
    input_ys: list[float] = []
    for idx, inp in enumerate(inputs):
        y = 160.0 + idx * 96.0
        input_ys.append(y)
        page.draw_rect(fitz.Rect(64, y - 18, 280, y + 18), color=(0, 0, 0), width=0.9)
        page.insert_text((74, y + 4), inp, fontsize=10, fontname="helv")
        # Stepped bus wiring lines toward first gate column
        page.draw_line((280, y), (350, y), color=(0, 0, 0), width=0.9)
        page.draw_line((350, y), (350, 320.0), color=(0, 0, 0), width=0.7)
        page.draw_line((350, y), (420, y), color=(0, 0, 0), width=0.7)
        page.draw_circle((350, y), 2.5, color=(0, 0, 0), fill=(0, 0, 0))

    # Central FBD gate blocks
    for gate_name, bx, by in blocks:
        page.draw_rect(fitz.Rect(bx, by, bx + 115, by + 90), color=(0, 0, 0), width=1.1)
        page.draw_line((bx, by + 26), (bx + 115, by + 26), color=(0, 0, 0), width=0.7)
        page.insert_text((bx + 36, by + 18), gate_name, fontsize=11, fontname="helv")
        for pin in range(3):
            py = by + 40 + pin * 18
            page.draw_line((bx - 22, py), (bx, py), color=(0, 0, 0), width=0.7)
            page.insert_text((bx + 6, py + 3), f"IN{pin + 1}", fontsize=7.5, fontname="helv")
        page.draw_line((bx + 115, by + 56), (bx + 150, by + 56), color=(0, 0, 0), width=0.9)
        page.insert_text((bx + 88, by + 59), "OUT", fontsize=7.5, fontname="helv")

    # Inter-block bus connections
    for idx in range(len(blocks) - 1):
        _, bx0, by0 = blocks[idx]
        _, bx1, by1 = blocks[idx + 1]
        page.draw_line((bx0 + 150, by0 + 56), (bx1 - 22, by1 + 40), color=(0, 0, 0), width=0.8)

    # Right output terminals
    for idx, out in enumerate(outputs):
        y = 240.0 + idx * 140.0
        page.draw_rect(fitz.Rect(910, y - 20, 1130, y + 20), color=(0, 0, 0), width=0.9)
        page.insert_text((922, y + 4), out, fontsize=10, fontname="helv")
        page.draw_line((820, y), (910, y), color=(0, 0, 0), width=0.9)
        page.draw_line((820, 320.0), (820, y), color=(0, 0, 0), width=0.7)
        page.draw_circle((820, y), 2.5, color=(0, 0, 0), fill=(0, 0, 0))

    # Additional CAD logic rung markers & reference boxes so each page exceeds 160 drawings cleanly
    for r in range(1, 22):
        ry = 115.0 + r * 28.0
        page.draw_line((48, ry), (56, ry), color=(0, 0, 0), width=0.5)
        page.draw_line((1134, ry), (1142, ry), color=(0, 0, 0), width=0.5)
        page.draw_line((320 + (r % 5) * 110, 680), (345 + (r % 5) * 110, 680), color=(0, 0, 0), width=0.4)


def build_esds_logic_pdf(out_path: Path) -> Path:
    """Build 2-page landscape 1190x842 ESDS Logic Diagram PDF (>160 vector lines/page)."""
    doc = fitz.open()

    p1 = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    _draw_logic_page(
        p1,
        pou_title="1.1. LTRCV_12",
        cross_ref="(* C&E PAGE4 *)",
        inputs=[
            "From: LAHH-80101A",
            "From: LAHH-80101B",
            "From: LAHH-80101C",
            "From: HZ10001_R12",
            "From: UAFGSESD1_R12",
        ],
        blocks=[
            ("AND", 440.0, 160.0),
            ("OR", 440.0, 320.0),
            ("RS", 640.0, 220.0),
            ("TON", 640.0, 380.0),
        ],
        outputs=[
            "To: XV-10101A",
            "To: RSESD001_R12",
        ],
        doc_no="LNG-ESD-LOG-008",
        sheet_label="PAGE 8",
    )

    p2 = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    _draw_logic_page(
        p2,
        pou_title="1.18. MAIN_07",
        cross_ref="(* C&E PAGE4 *)",
        inputs=[
            "From: HZ10001_R12",
            "From: RSESD001_R12",
            "From: PALL-14261A",
        ],
        blocks=[
            ("SET", 440.0, 180.0),
            ("RESET", 440.0, 340.0),
            ("AND", 640.0, 200.0),
            ("OR", 640.0, 360.0),
        ],
        outputs=[
            "To: HA-10003B",
            "To: XV-10101A",
        ],
        doc_no="LNG-ESD-LOG-051",
        sheet_label="PAGE 51",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path), garbage=4, deflate=True)
    doc.close()
    return out_path


def _draw_stroke_text(
    page: fitz.Page,
    x: float,
    y: float,
    text: str,
    *,
    char_w: float = 19.0,
    char_h: float = 34.0,
    spacing: float = 8.0,
    stroke_w: float = 3.0,
) -> int:
    """Render uppercase ASCII text purely as open vector polylines (zero PDF text objects)."""
    drawn = 0
    cursor_x = x
    for ch in text.upper():
        if ch == " ":
            cursor_x += char_w * 0.75 + spacing
            continue
        strokes = _STROKE_GLYPHS.get(ch)
        if not strokes:
            cursor_x += char_w + spacing
            continue
        for poly in strokes:
            shape = page.new_shape()
            pts = [(cursor_x + px * char_w, y + py * char_h) for px, py in poly]
            shape.draw_polyline(pts)
            shape.finish(color=(0, 0, 0), width=stroke_w, lineCap=1, lineJoin=1, closePath=False)
            shape.commit()
            drawn += 1
        cursor_x += char_w + spacing
    return drawn


def build_lv_swgr_logic_pdf(out_path: Path) -> Path:
    """Build 1-page landscape 1190x842 exploded-SHX CAD drawing (words==0, drawings>350)."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)

    # CAD border frames (zero text objects!)
    page.draw_rect(fitz.Rect(20, 20, PAGE_WIDTH - 20, PAGE_HEIGHT - 20), color=(0, 0, 0), width=1.6)
    page.draw_rect(fitz.Rect(28, 28, PAGE_WIDTH - 28, PAGE_HEIGHT - 28), color=(0, 0, 0), width=0.9)

    # Bottom-right title block outline
    tb_x0, tb_y0, tb_x1, tb_y1 = PAGE_WIDTH - 420, PAGE_HEIGHT - 135, PAGE_WIDTH - 28, PAGE_HEIGHT - 28
    page.draw_rect(fitz.Rect(tb_x0, tb_y0, tb_x1, tb_y1), color=(0, 0, 0), width=1.2)
    page.draw_line((tb_x0, tb_y0 + 58), (tb_x1, tb_y0 + 58), color=(0, 0, 0), width=0.8)

    # Switchgear 400V Bus A line and breaker schematic geometry
    page.draw_line((80, 120), (1100, 120), color=(0, 0, 0), width=2.8)
    page.draw_line((80, 130), (1100, 130), color=(0, 0, 0), width=1.4)

    # Feeder bays & breaker symbols
    for bay_idx in range(6):
        bx = 160.0 + bay_idx * 170.0
        page.draw_line((bx, 130), (bx, 240), color=(0, 0, 0), width=1.4)
        # Breaker box & cross contact
        page.draw_rect(fitz.Rect(bx - 22, 240, bx + 22, 290), color=(0, 0, 0), width=1.2)
        page.draw_line((bx - 12, 252), (bx + 12, 278), color=(0, 0, 0), width=1.1)
        page.draw_line((bx - 12, 278), (bx + 12, 252), color=(0, 0, 0), width=1.1)
        page.draw_line((bx, 290), (bx, 395), color=(0, 0, 0), width=1.2)
        # Motor / feeder circle
        page.draw_circle((bx, 420), 24, color=(0, 0, 0), width=1.2)

    # Auxiliary control logic ladder rungs & grid ticks (> 260 additional primitives => total > 350)
    for rung in range(22):
        ry = 545.0 + rung * 11.0
        page.draw_line((70, ry), (680, ry), color=(0, 0, 0), width=0.6)
        for col in range(12):
            cx = 90.0 + col * 50.0
            page.draw_line((cx, ry - 4), (cx, ry + 4), color=(0, 0, 0), width=0.6)

    # Exploded-SHX stroke labels clearly placed within the 2x2 tiles + title block (clear of tile seam edges)
    # Top-left tile: 400V BUS A, MCC-401A
    _draw_stroke_text(page, 95, 58, "400V BUS A", char_w=19, char_h=34, spacing=8, stroke_w=3.0)
    _draw_stroke_text(page, 95, 175, "MCC-401A", char_w=19, char_h=34, spacing=8, stroke_w=3.0)

    # Top-right tile: CB-401
    _draw_stroke_text(page, 700, 175, "CB-401", char_w=20, char_h=34, spacing=9, stroke_w=3.0)

    # Bottom-left tile: TRIP COIL (placed cleanly at y=468 inside bottom-left tile)
    page.draw_rect(fitz.Rect(80, 456, 390, 524), color=(1, 1, 1), fill=(1, 1, 1))
    page.draw_rect(fitz.Rect(80, 456, 390, 524), color=(0, 0, 0), width=1.0)
    _draw_stroke_text(page, 98, 472, "TRIP COIL", char_w=19, char_h=34, spacing=8, stroke_w=3.0)

    # Bottom-right title block: LV SWGR LOGIC
    _draw_stroke_text(page, tb_x0 + 22, tb_y0 + 13, "LV SWGR LOGIC", char_w=17, char_h=30, spacing=8, stroke_w=2.8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path), garbage=4, deflate=True)
    doc.close()
    return out_path


def slice_pdf_pages(source_pdf: Path, out_path: Path, zero_based_pages: Iterable[int]) -> Path:
    """Deterministically slice specific 0-based pages from a source PDF using fitz.open + insert_pdf."""
    src = fitz.open(str(source_pdf))
    dst = fitz.open()
    try:
        for pno in zero_based_pages:
            if 0 <= pno < len(src):
                dst.insert_pdf(src, from_page=pno, to_page=pno)
        if len(dst) == 0:
            raise ValueError(f"No valid pages {list(zero_based_pages)} in {source_pdf} (len={len(src)})")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dst.save(str(out_path), garbage=4, deflate=True)
    finally:
        dst.close()
        src.close()
    return out_path


def _find_matching_source(source_dir: Path, keywords: Sequence[str]) -> Path | None:
    if not source_dir.is_dir():
        return None
    for pdf_path in sorted(source_dir.glob("*.pdf")):
        lower_name = pdf_path.name.lower()
        if all(kw in lower_name for kw in keywords):
            return pdf_path
    return None


def generate_all_fixtures(out_dir: Path = DEFAULT_OUT_DIR, source_dir: Path | None = None) -> dict[str, Path]:
    """Generate or slice all 4 compact engineering fixtures into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("cabinet_wiring_p2.pdf", ("cabinet", "wiring"), [1], build_cabinet_wiring_pdf),
        ("process_ce_chart_p3_p4.pdf", ("cause", "effect"), [2, 3], build_process_ce_chart_pdf),
        ("esds_logic_p8_p51.pdf", ("esds", "logic"), [7, 50], build_esds_logic_pdf),
        ("lv_swgr_logic_p3.pdf", ("lv", "swgr"), [2], build_lv_swgr_logic_pdf),
    ]

    generated: dict[str, Path] = {}
    for filename, keywords, page_indices, builder in specs:
        target = out_dir / filename
        matched_src = _find_matching_source(source_dir, keywords) if source_dir else None
        if matched_src is not None:
            slice_pdf_pages(matched_src, target, page_indices)
        else:
            builder(target)
        generated[filename] = target
    return generated


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate compact CAD engineering PDF fixtures.")
    parser.add_argument("--source-dir", type=Path, default=None, help="Optional directory containing source PDFs to slice.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for generated fixtures.")
    args = parser.parse_args(argv)

    paths = generate_all_fixtures(out_dir=args.out_dir, source_dir=args.source_dir)
    total_bytes = sum(p.stat().st_size for p in paths.values())
    for name, p in paths.items():
        print(f"{name}: {p.stat().st_size:,} bytes ({p})")
    print(f"Total fixture size: {total_bytes:,} bytes ({total_bytes / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
