"""Document parsing service: Docling v2 primary router, pymupdf4llm fallback, and Excel windowing."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openpyxl
try:
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import DoclingDocument
except (ImportError, OSError):
    DocumentConverter = None  # type: ignore
    DoclingDocument = None  # type: ignore

from src.config import get_settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx"}
EXCEL_WINDOW_ROWS = 50


@dataclass
class ParseResult:
    """Normalized parsing output with persisted AST and Markdown cache paths."""

    sha256: str
    filename: str
    file_type: str  # 'pdf' | 'docx' | 'pptx' | 'xlsx'
    markdown: str
    markdown_path: Path
    parsed_path: Path
    docling_dict: dict[str, Any] | None = None
    page_count: int | None = None
    from_cache: bool = False
    extra_meta: dict[str, Any] = field(default_factory=dict)


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_pdf_fast_path(file_path: Path) -> tuple[bool, str]:
    """Lightweight 5ms preflight probe using PyMuPDF to classify digital text vs scanned/heavy layout."""
    import pymupdf as fitz
    try:
        doc = fitz.open(str(file_path))
        try:
            sample_pages = min(3, len(doc))
            total_text_chars = 0
            total_images = 0

            for i in range(sample_pages):
                page = doc[i]
                total_text_chars += len(page.get_text().strip())
                total_images += len(page.get_images())

            avg_chars = total_text_chars / max(1, sample_pages)
            avg_images = total_images / max(1, sample_pages)

            if avg_chars > 120 and avg_images <= 2:
                return True, f"digital_text (avg {int(avg_chars)} chars/page)"
            return False, f"scanned_or_heavy_layout (avg {avg_images:.1f} images/page)"
        finally:
            doc.close()
    except Exception as e:
        return False, f"preflight_inspection_error: {e}"


def load_parsed_checkpoint(
    data_dir: Path,
    sha: str,
    filename: str,
    file_type: str,
) -> ParseResult | None:
    """Load ParseResult directly from cached data/parsed/{sha}.json and data/markdown/{sha}.md."""
    parsed_path = data_dir / "parsed" / f"{sha}.json"
    markdown_path = data_dir / "markdown" / f"{sha}.md"
    if not (parsed_path.is_file() and markdown_path.is_file()):
        return None
    try:
        cached_json = json.loads(parsed_path.read_text(encoding="utf-8"))
        md_text = markdown_path.read_text(encoding="utf-8")
        doc_dict = cached_json.get("docling_document")
        if doc_dict is None and cached_json.get("schema_name") == "DoclingDocument":
            doc_dict = cached_json
        page_count = cached_json.get("page_count")
        if page_count is None and isinstance(doc_dict, dict) and "pages" in doc_dict:
            page_count = len(doc_dict["pages"]) or None
        return ParseResult(
            sha256=sha,
            filename=filename,
            file_type=file_type,
            markdown=md_text,
            markdown_path=markdown_path,
            parsed_path=parsed_path,
            docling_dict=doc_dict,
            page_count=page_count,
            from_cache=True,
            extra_meta=cached_json.get("extra_meta", {}),
        )
    except Exception as exc:
        logger.warning("Failed to load parsed checkpoint for %s: %s", sha, exc)
        return None


class DocumentParserService:
    """Multi-format document parser with AST/Markdown disk caching."""

    def __init__(
        self,
        data_dir: Path | None = None,
        prefer_pymupdf4llm: bool | None = None,
        excel_window_rows: int = EXCEL_WINDOW_ROWS,
        max_upload_size_mb: int | None = None,
        device: str | None = None,
    ) -> None:
        settings = get_settings()
        self.data_dir = Path(data_dir) if data_dir else settings.data_dir
        self.parsed_dir = self.data_dir / "parsed"
        self.markdown_dir = self.data_dir / "markdown"
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        if prefer_pymupdf4llm is None:
            env_val = os.environ.get("PREFER_PYMUPDF4LLM")
            if env_val is not None and env_val.strip():
                self.prefer_pymupdf4llm: bool | None = env_val.lower() in (
                    "1",
                    "true",
                    "yes",
                )
            else:
                self.prefer_pymupdf4llm = None
        else:
            self.prefer_pymupdf4llm = prefer_pymupdf4llm
        self.excel_window_rows = excel_window_rows
        self.max_upload_size_mb = (
            max_upload_size_mb if max_upload_size_mb is not None else settings.ingestion.max_upload_size_mb
        )
        self.device = (device or settings.ingestion.device or "auto").strip().lower()
        self._converter: DocumentConverter | None = None

    def _get_converter(self) -> Any:
        if self._converter is None:
            if DocumentConverter is None:
                raise ImportError("Docling DocumentConverter is unavailable")
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions
                from docling.document_converter import DocumentConverter as DLConverter, PdfFormatOption

                from src.ingestion.indexer import resolve_compute_device

                resolved = resolve_compute_device(self.device)
                acc_dev = AcceleratorDevice.CUDA if resolved == "cuda" else AcceleratorDevice.CPU
                pipeline_opts = PdfPipelineOptions(
                    accelerator_options=AcceleratorOptions(device=acc_dev, num_threads=4)
                )
                self._converter = DLConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
                    }
                )
            except Exception as exc:
                raise ImportError(f"Docling initialization failed: {exc}") from exc
        return self._converter

    def validate_file(self, file_path: Path) -> str:
        """Validate extension, existence, and size limit. Returns normalized file_type without dot."""
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
        max_bytes = self.max_upload_size_mb * 1024 * 1024
        size = file_path.stat().st_size
        if size > max_bytes:
            raise ValueError(f"File size {size} bytes exceeds maximum allowed ({self.max_upload_size_mb} MB)")
        return ext.lstrip(".")

    def parse(
        self,
        file_path: Path | str,
        *,
        sha256: str | None = None,
        filename: str | None = None,
        force_reparse: bool = False,
        use_pymupdf_fallback: bool | None = None,
    ) -> ParseResult:
        """Parse a document into structured Markdown and AST, using disk cache if available."""
        path = Path(file_path)
        file_type = self.validate_file(path)
        sha = sha256 or compute_sha256(path)
        orig_filename = filename or path.name

        parsed_path = self.parsed_dir / f"{sha}.json"
        markdown_path = self.markdown_dir / f"{sha}.md"

        prefer_fallback = (
            self.prefer_pymupdf4llm if use_pymupdf_fallback is None else use_pymupdf_fallback
        )

        if not force_reparse:
            cached = load_parsed_checkpoint(self.data_dir, sha, orig_filename, file_type)
            if cached is not None:
                cached_engine = cached.extra_meta.get("engine")
                mismatch = (
                    (prefer_fallback is True and cached_engine != "pymupdf4llm")
                    or (prefer_fallback is False and cached_engine != "docling")
                )
                if mismatch:
                    logger.info(
                        "Cached parsed checkpoint engine '%s' contradicts requested mode (prefer_fallback=%s); invalidating checkpoint and forcing re-parse",
                        cached_engine,
                        prefer_fallback,
                    )
                else:
                    return cached

        if file_type == "xlsx":
            md_text, doc_dict, page_count, extra = self._parse_xlsx(path, orig_filename)
        elif file_type == "pdf":
            md_text, doc_dict, page_count, extra = self._parse_pdf(path, prefer_fallback=prefer_fallback)
        else:
            # .docx and .pptx via Docling v2
            md_text, doc_dict, page_count, extra = self._parse_with_docling(path, file_type=file_type)

        payload_to_cache: dict[str, Any] = {
            "sha256": sha,
            "filename": orig_filename,
            "file_type": file_type,
            "page_count": page_count,
            "docling_document": doc_dict,
            "extra_meta": extra,
        }
        parsed_path.write_text(json.dumps(payload_to_cache, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown_path.write_text(md_text, encoding="utf-8")

        return ParseResult(
            sha256=sha,
            filename=orig_filename,
            file_type=file_type,
            markdown=md_text,
            markdown_path=markdown_path,
            parsed_path=parsed_path,
            docling_dict=doc_dict,
            page_count=page_count,
            from_cache=False,
            extra_meta=extra,
        )

    def _parse_with_docling(
        self, file_path: Path, file_type: str
    ) -> tuple[str, dict[str, Any] | None, int | None, dict[str, Any]]:
        try:
            conv = self._get_converter()
            result = conv.convert(file_path)
            dl_doc: DoclingDocument = result.document
            md_text = dl_doc.export_to_markdown()
            doc_dict = dl_doc.export_to_dict()
            pages = doc_dict.get("pages") or {}
            page_count: int | None = len(pages) if pages else None
            if file_type == "docx":
                page_count = None
            return md_text, doc_dict, page_count, {"engine": "docling"}
        except Exception as exc:
            if file_type == "docx":
                import docx

                doc_obj = docx.Document(file_path)
                paras = [p.text for p in doc_obj.paragraphs if p.text.strip()]
                md_text = "\n\n".join(paras)
                return md_text, None, None, {"engine": "python-docx", "engine_fallback_reason": str(exc)}
            elif file_type == "pptx":
                import pptx

                prs = pptx.Presentation(file_path)
                slides_text = []
                for idx, slide in enumerate(prs.slides):
                    st = [shape.text for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()]
                    if st:
                        slides_text.append(f"<!-- page: {idx+1} -->\n" + "\n".join(st))
                md_text = "\n\n".join(slides_text)
                return md_text, None, len(prs.slides), {"engine": "python-pptx", "engine_fallback_reason": str(exc)}
            raise

    def _parse_pdf(
        self, file_path: Path, *, prefer_fallback: bool | None = None
    ) -> tuple[str, dict[str, Any] | None, int | None, dict[str, Any]]:
        fallback_reason = None
        if prefer_fallback is True:
            return self._parse_pdf_pymupdf4llm(file_path)

        if prefer_fallback is False:
            try:
                md_text, doc_dict, page_count, extra = self._parse_with_docling(file_path, file_type="pdf")
                if md_text and md_text.strip():
                    return md_text, doc_dict, page_count, extra
            except Exception as exc:
                fallback_reason = f"Docling GPU failed ({exc}); fell back to PyMuPDF CPU"
                logger.warning(
                    "Explicit GPU parse failed for %s: %s; falling back to pymupdf4llm",
                    file_path.name,
                    exc,
                )
            md_text, doc_dict, page_count, extra = self._parse_pdf_pymupdf4llm(file_path)
            if fallback_reason:
                extra["engine_fallback_reason"] = fallback_reason
            return md_text, doc_dict, page_count, extra

        # prefer_fallback is None: auto-detect fast path
        is_fast_path, reason = inspect_pdf_fast_path(file_path)
        if is_fast_path:
            logger.info("PDF %s qualified for Fast-Path: %s; using pymupdf4llm", file_path.name, reason)
            return self._parse_pdf_pymupdf4llm(file_path)

        fallback_reason = None
        try:
            md_text, doc_dict, page_count, extra = self._parse_with_docling(file_path, file_type="pdf")
            if md_text and md_text.strip():
                return md_text, doc_dict, page_count, extra
            fallback_reason = "Auto Docling returned empty text; fell back to PyMuPDF CPU"
            logger.warning(
                "Auto Docling returned empty text for %s; falling back to pymupdf4llm",
                file_path.name,
            )
        except Exception as exc:
            fallback_reason = f"Auto Docling failed ({exc}); fell back to PyMuPDF CPU"
            logger.warning(
                "Auto Docling failed for %s: %s; falling back to pymupdf4llm",
                file_path.name,
                exc,
            )
        md_text, doc_dict, page_count, extra = self._parse_pdf_pymupdf4llm(file_path)
        if fallback_reason:
            extra["engine_fallback_reason"] = fallback_reason
        return md_text, doc_dict, page_count, extra

    def _parse_pdf_pymupdf4llm(
        self, file_path: Path
    ) -> tuple[str, dict[str, Any] | None, int | None, dict[str, Any]]:
        import pymupdf4llm

        page_chunks: list[dict[str, Any]] = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
        pages_meta: list[dict[str, Any]] = []
        md_parts: list[str] = []
        for idx, p in enumerate(page_chunks, start=1):
            page_no = int(p.get("metadata", {}).get("page", idx) or idx)
            text = (p.get("text") or "").strip()
            if text:
                md_parts.append(f"<!-- page: {page_no} -->\n{text}")
                pages_meta.append({"page_number": page_no, "text": text})
        md_text = "\n\n".join(md_parts)
        return md_text, None, len(page_chunks), {"engine": "pymupdf4llm", "pages": pages_meta}

    def _parse_xlsx(
        self, file_path: Path, orig_filename: str
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
                            str(cell).strip().replace("|", "\\|").replace("\n", " ") if cell is not None else f"Col_{i+1}"
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

                header_line = "| " + " | ".join(raw_headers) + " |"
                sep_line = "| " + " | ".join(["---"] * num_cols) + " |"

                window = max(1, self.excel_window_rows)
                if total_rows == 0:
                    md_sections.append(f"### {sheet_name} (Rows 0-0)\n\n{header_line}\n{sep_line}")
                else:
                    for start_idx in range(0, total_rows, window):
                        end_idx = min(start_idx + window, total_rows)
                        slice_rows = data_rows[start_idx:end_idx]
                        body_lines = ["| " + " | ".join(row) + " |" for row in slice_rows]
                        table_block = "\n".join([header_line, sep_line, *body_lines])
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
        # For reasonable-sized workbooks, also attempt Docling AST capture if possible
        total_workbook_rows = sum(s["row_count"] for s in sheets_meta)
        if total_workbook_rows <= 500:
            try:
                conv = self._get_converter()
                dl_doc = conv.convert(file_path).document
                doc_dict = dl_doc.export_to_dict()
            except Exception:
                doc_dict = None

        full_md = "\n\n".join(md_sections)
        return full_md, doc_dict, None, {"engine": "openpyxl+docling", "sheets": sheets_meta}

