"""Office document fallback parsers (DOCX/PPTX) and windowed Excel workbook parser."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import openpyxl

from src.ingestion.constants import format_markdown_table


def parse_docx_fallback(
    file_path: Path,
    exc: Exception,
) -> tuple[str, dict[str, Any] | None, int | None, dict[str, Any]]:
    """Extract paragraph text from DOCX when Docling conversion is unavailable."""
    import docx

    doc_obj = docx.Document(file_path)
    paras = [p.text for p in doc_obj.paragraphs if p.text.strip()]
    md_text = "\n\n".join(paras)
    return md_text, None, None, {"engine": "python-docx", "engine_fallback_reason": str(exc)}


def parse_pptx_fallback(
    file_path: Path,
    exc: Exception,
) -> tuple[str, dict[str, Any] | None, int | None, dict[str, Any]]:
    """Extract per-slide shape text from PPTX when Docling conversion is unavailable."""
    import pptx

    prs = pptx.Presentation(file_path)
    slides_text = []
    for idx, slide in enumerate(prs.slides):
        st = [shape.text for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()]
        if st:
            slides_text.append(f"<!-- page: {idx + 1} -->\n" + "\n".join(st))
    md_text = "\n\n".join(slides_text)
    return md_text, None, len(prs.slides), {"engine": "python-pptx", "engine_fallback_reason": str(exc)}


def parse_xlsx_workbook(
    file_path: Path,
    orig_filename: str,
    *,
    excel_window_rows: int = 50,
    get_converter_fn: Callable[[], Any] | None = None,
) -> tuple[str, dict[str, Any] | None, int | None, dict[str, Any]]:
    """Extract Excel sheets with schema summary and bounded Markdown row windows with repeated headers."""
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    md_sections: list[str] = [f"# Workbook: {orig_filename}"]
    sheets_meta: list[dict[str, Any]] = []

    try:
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows_iter = ws.iter_rows(values_only=True)
            raw_headers: list[str] = []
            for r in rows_iter:
                if any(cell is not None and str(cell).strip() != "" for cell in r):
                    raw_headers = [
                        str(cell).strip().replace("|", "\\|").replace("\n", " ")
                        if cell is not None
                        else f"Col_{i + 1}"
                        for i, cell in enumerate(r)
                    ]
                    break

            if not raw_headers:
                continue

            num_cols = len(raw_headers)
            data_rows: list[list[str]] = []
            for r in rows_iter:
                if not any(cell is not None and str(cell).strip() != "" for cell in r):
                    continue
                row_vals = [
                    str(r[i]).strip().replace("|", "\\|").replace("\n", " ")
                    if i < len(r) and r[i] is not None
                    else ""
                    for i in range(num_cols)
                ]
                data_rows.append(row_vals)

            total_rows = len(data_rows)
            col_list_str = ", ".join(raw_headers)
            schema_summary = (
                f"## Sheet: {sheet_name}\n\n"
                f"### Sheet Schema Summary\n"
                f"- **Sheet Name**: {sheet_name}\n"
                f"- **Columns ({num_cols})**: {col_list_str}\n"
                f"- **Total Data Rows**: {total_rows}\n"
            )
            md_sections.append(schema_summary)

            window = max(1, excel_window_rows)
            if total_rows == 0:
                empty_tbl = format_markdown_table(raw_headers, [], escape_pipes=False)
                md_sections.append(f"### {sheet_name} (Rows 0-0)\n\n{empty_tbl}")
            else:
                for start_idx in range(0, total_rows, window):
                    end_idx = min(start_idx + window, total_rows)
                    slice_rows = data_rows[start_idx:end_idx]
                    table_block = format_markdown_table(raw_headers, slice_rows, escape_pipes=False)
                    md_sections.append(
                        f"### {sheet_name} (Rows {start_idx + 1}-{end_idx})\n\n{table_block}"
                    )

            sheets_meta.append(
                {
                    "sheet_name": sheet_name,
                    "columns": raw_headers,
                    "row_count": total_rows,
                }
            )
    finally:
        wb.close()

    doc_dict: dict[str, Any] | None = None
    total_workbook_rows = sum(s["row_count"] for s in sheets_meta)
    if total_workbook_rows <= 500 and get_converter_fn is not None:
        try:
            conv = get_converter_fn()
            dl_doc = conv.convert(file_path).document
            doc_dict = dl_doc.export_to_dict()
        except Exception:
            doc_dict = None

    full_md = "\n\n".join(md_sections)
    return full_md, doc_dict, None, {"engine": "openpyxl+docling", "sheets": sheets_meta}
