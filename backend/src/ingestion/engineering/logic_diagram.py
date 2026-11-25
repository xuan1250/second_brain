"""CAD logic / FBD diagram sheet extractor."""
from __future__ import annotations

import re
from typing import Any

import pymupdf as fitz

from src.ingestion.engineering.geometry import (
    _bbox_mid_x,
    _bbox_mid_y,
    _get_visual_text_dict,
    _get_visual_text_lines,
)
from src.ingestion.engineering.models import EngineeringPageRecord, PageExtractionResult
from src.ingestion.engineering.patterns import (
    _CROSS_REF_RE,
    _LOGIC_GATE_RE,
    _POU_TITLE_RE,
    _TAG_TOKEN_RE,
    _extract_record_tags,
)
from src.ingestion.engineering.profiler import profile_pdf_page


def extract_cad_logic_sheet(
    page: fitz.Page,
    page_number: int,
    *,
    metrics: dict[str, Any] | None = None,
) -> PageExtractionResult:
    """Group POU title, cross-references, inputs (From), logic blocks, and outputs (To) on CAD logic sheets."""
    if metrics is None:
        _, metrics = profile_pdf_page(page)
    text_dict = _get_visual_text_dict(page)
    raw_text = _get_visual_text_lines(page)
    width = float(page.rect.width)
    height = float(page.rect.height)

    spans: list[dict[str, Any]] = []
    for block in text_dict.get("blocks", ()):
        for line in block.get("lines", ()):
            for span in line.get("spans", ()):
                txt = (span.get("text") or "").strip()
                if txt:
                    spans.append({"text": txt, "bbox": span["bbox"]})

    pou_match = _POU_TITLE_RE.search(raw_text)
    pou_title = pou_match.group(0).strip() if pou_match else f"CAD Logic Diagram (Page {page_number})"
    cross_refs = [m.strip() for m in _CROSS_REF_RE.findall(raw_text) if m.strip()]

    inputs: list[str] = []
    logic_blocks: list[str] = []
    outputs: list[str] = []

    for sp in sorted(spans, key=lambda s: (float(s["bbox"][1]), float(s["bbox"][0]))):
        txt = sp["text"]
        x_mid = _bbox_mid_x(sp["bbox"])
        y_mid = _bbox_mid_y(sp["bbox"])

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
