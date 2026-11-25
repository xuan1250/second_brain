"""Engineering CAD record chunkers for schedules, C&E matrices, logic diagrams, and OCR pages."""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.chunking.domain_splitters import (
    _PAGE_COMMENT_RE,
    split_lines_with_repeated_header,
)
from src.ingestion.chunking.models import build_legend_summary


def _parse_markdown_pages(markdown: str) -> dict[int, str]:
    """Split <!-- page: N --> annotated markdown into per-page raw text."""
    if not markdown:
        return {}
    matches = list(_PAGE_COMMENT_RE.finditer(markdown))
    if not matches:
        return {}
    pages: dict[int, str] = {}
    for idx, m in enumerate(matches):
        pno = int(m.group(1))
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        chunk = markdown[start:end].strip()
        if chunk:
            pages[pno] = f"{pages[pno]}\n\n{chunk}".strip() if pno in pages else chunk
    return pages


def chunk_schedule_records(
    schedule_recs: Sequence[dict[str, Any]],
    *,
    pno: int,
    child_max_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[dict[str, Any]]:
    """Group 3-5 consecutive schedule rows per child chunk while repeating the sheet header."""
    if not schedule_recs:
        return []

    first_meta = schedule_recs[0].get("metadata") or {}
    sheet_title = str(
        first_meta.get("sheet_title")
        or first_meta.get("drawing_title")
        or schedule_recs[0].get("title")
        or f"Schedule (Page {pno})"
    ).strip()

    headers: list[str] = []
    raw_headers = first_meta.get("headers")
    if isinstance(raw_headers, (list, tuple)) and raw_headers:
        headers = [str(h).strip() for h in raw_headers if str(h).strip()]
    else:
        seen_cols: set[str] = set()
        for rec in schedule_recs:
            cols = (rec.get("metadata") or {}).get("columns") or {}
            if isinstance(cols, dict):
                for col_name in cols.keys():
                    c_str = str(col_name).strip()
                    if c_str and c_str not in seen_cols:
                        seen_cols.add(c_str)
                        headers.append(c_str)

    sheet_banner = (
        f"Schedule: {sheet_title} | Columns: {', '.join(headers)}"
        if headers
        else f"Schedule: {sheet_title}"
    )
    heading_path = f"{sheet_title} (Page {pno})"
    target_child_tokens = min(child_max_tokens, 240) if child_max_tokens >= 180 else child_max_tokens

    items: list[dict[str, Any]] = []
    cur_rows: list[str] = []
    cur_tags: list[str] = []
    cur_start_idx = 0

    def _flush_group() -> None:
        if not cur_rows:
            return
        child_text = f"{sheet_banner}\n" + "\n".join(cur_rows)
        items.append(
            {
                "heading_path": heading_path,
                "page_numbers": [pno],
                "child_text": child_text,
                "token_count": max(1, count_tokens(child_text)),
                "record_type": "schedule_row",
                "parent_group_key": f"schedule_p{pno}_w{cur_start_idx // 16}",
                "record_tags": list(dict.fromkeys(cur_tags)),
            }
        )

    for idx, rec in enumerate(schedule_recs):
        row_text = str(rec.get("summary_text") or "").strip()
        if not row_text:
            cols = (rec.get("metadata") or {}).get("columns") or {}
            if isinstance(cols, dict) and cols:
                row_text = " | ".join(f"{k}: {v}" for k, v in cols.items())
            else:
                row_text = str(rec.get("title") or "").strip()
        if not row_text:
            continue

        candidate_text = f"{sheet_banner}\n" + "\n".join([*cur_rows, row_text])
        cand_tokens = count_tokens(candidate_text)
        if cur_rows and (
            len(cur_rows) >= 5
            or cand_tokens > child_max_tokens
            or (len(cur_rows) >= 3 and cand_tokens > target_child_tokens)
        ):
            _flush_group()
            cur_rows = []
            cur_tags = []
            cur_start_idx = idx

        if not cur_rows:
            cur_start_idx = idx
        cur_rows.append(row_text)
        for tag in rec.get("tags") or []:
            t_str = str(tag).strip()
            if t_str:
                cur_tags.append(t_str)

    _flush_group()
    return items


def build_engineering_intermediate_items(
    markdown: str,
    *,
    profile: dict[str, Any],
    child_max_tokens: int,
    count_tokens: Callable[[str], int],
    split_prose_fn: Callable[[str, str, RecursiveCharacterTextSplitter], list[str]],
    split_lines_fn: Callable[..., list[str]] | None = None,
    chunk_schedule_fn: Callable[..., list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Build intermediate chunk dicts from an engineering_profile without Docling AST flattening."""
    legend_ctx = profile.get("legend_context") or {}
    legend_summary = build_legend_summary(legend_ctx)
    page_modes = profile.get("page_modes") or {}
    page_text_map = _parse_markdown_pages(markdown or "")

    pages = profile.get("pages") or []
    if not pages:
        records_by_page: dict[int, list[dict[str, Any]]] = {}
        for rec in profile.get("records") or []:
            pno = int(rec.get("page_number") or 1)
            records_by_page.setdefault(pno, []).append(rec)
        pages = [
            {
                "page_number": pno,
                "mode": page_modes.get(pno) or page_modes.get(str(pno)) or "",
                "records": recs,
            }
            for pno, recs in sorted(records_by_page.items())
        ]

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(200, child_max_tokens * 4),
        chunk_overlap=max(32, child_max_tokens // 4),
        separators=["\n\n", "\n", " ", ""],
    )

    intermediate: list[dict[str, Any]] = []

    def _split_with_header(text: str, *, header_line: str) -> list[str]:
        if split_lines_fn is not None:
            return split_lines_fn(text, header_line=header_line, max_tokens=child_max_tokens)
        return split_lines_with_repeated_header(
            text,
            header_line=header_line,
            max_tokens=child_max_tokens,
            count_tokens=count_tokens,
        )

    def _make_repeated_header_items(
        text: str,
        *,
        heading_path: str,
        header_line: str,
        pno: int,
        record_type: str,
        parent_group_key: str,
        record_tags: list[str],
        parent_preamble: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        built: list[dict[str, Any]] = []
        for s in _split_with_header(text, header_line=header_line):
            item: dict[str, Any] = {
                "heading_path": heading_path,
                "page_numbers": [pno],
                "child_text": s,
                "token_count": max(1, count_tokens(s)),
                "record_type": record_type,
                "parent_group_key": parent_group_key,
                "record_tags": record_tags,
            }
            if parent_preamble is not None:
                item["parent_preamble"] = parent_preamble
            built.append(item)
        return built

    def _make_prose_items(
        text: str,
        *,
        heading_path: str,
        pno: int,
        record_type: str,
        parent_group_key: str,
        record_tags: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        built: list[dict[str, Any]] = []
        for sub in split_prose_fn(text, heading_path, char_splitter):
            sub_clean = sub.strip()
            if not sub_clean:
                continue
            item: dict[str, Any] = {
                "heading_path": heading_path,
                "page_numbers": [pno],
                "child_text": sub_clean,
                "token_count": max(1, count_tokens(sub_clean)),
                "record_type": record_type,
                "parent_group_key": parent_group_key,
            }
            if record_tags is not None:
                item["record_tags"] = record_tags
            built.append(item)
        return built

    for page in pages:
        pno = int(page.get("page_number") or 1)
        mode = str(page.get("mode") or page_modes.get(pno) or page_modes.get(str(pno)) or "").strip()
        page_recs: list[dict[str, Any]] = list(page.get("records") or [])

        if mode == "standard_prose" and not page_recs:
            raw_prose = str(page.get("markdown") or page.get("text") or page_text_map.get(pno) or "")
            prose_clean = _PAGE_COMMENT_RE.sub("", raw_prose).strip()
            if prose_clean:
                intermediate.extend(
                    _make_prose_items(
                        prose_clean,
                        heading_path=f"Document (Page {pno})",
                        pno=pno,
                        record_type="standard_prose",
                        parent_group_key=f"prose_p{pno}",
                    )
                )
            continue

        if mode == "document_legend_notes" and not page_recs:
            fallback_legend = legend_summary or _PAGE_COMMENT_RE.sub("", page_text_map.get(pno) or "").strip()
            if fallback_legend:
                heading_path = f"Engineering Legend & Notes (Page {pno})"
                intermediate.extend(
                    _make_repeated_header_items(
                        fallback_legend,
                        heading_path=heading_path,
                        header_line=heading_path,
                        pno=pno,
                        record_type="document_legend_notes",
                        parent_group_key=f"legend_p{pno}",
                        record_tags=[],
                    )
                )
            continue

        idx = 0
        while idx < len(page_recs):
            rec = page_recs[idx]
            rtype = str(rec.get("record_type") or mode or "").strip()

            if rtype == "schedule_row":
                run: list[dict[str, Any]] = []
                while idx < len(page_recs) and str(page_recs[idx].get("record_type") or "") == "schedule_row":
                    run.append(page_recs[idx])
                    idx += 1
                if chunk_schedule_fn is not None:
                    intermediate.extend(chunk_schedule_fn(run, pno=pno))
                else:
                    intermediate.extend(
                        chunk_schedule_records(
                            run,
                            pno=pno,
                            child_max_tokens=child_max_tokens,
                            count_tokens=count_tokens,
                        )
                    )
                continue

            idx += 1
            rec_id = str(rec.get("record_id") or f"p{pno}_rec_{idx}").strip()
            rec_title = str(rec.get("title") or rec_id).strip()
            rec_tags = [str(t).strip() for t in (rec.get("tags") or []) if str(t).strip()]
            summary = str(rec.get("summary_text") or "").strip()

            if rtype == "document_legend_notes" or mode == "document_legend_notes":
                heading_path = f"Engineering Legend & Notes (Page {pno})"
                text = summary or legend_summary
                if rec_title and rec_title not in text:
                    text = f"{rec_title}\n{text}".strip() if text else rec_title
                intermediate.extend(
                    _make_repeated_header_items(
                        text,
                        heading_path=heading_path,
                        header_line=heading_path,
                        pno=pno,
                        record_type="document_legend_notes",
                        parent_group_key=f"legend_p{pno}",
                        record_tags=rec_tags,
                    )
                )
            elif rtype in ("matrix_cause_row", "matrix_effect_col"):
                intermediate.extend(
                    _make_repeated_header_items(
                        summary or rec_title,
                        heading_path=f"Cause & Effect Matrix (Page {pno}) > {rec_title}",
                        header_line=f"{rec_title} [{rtype}]",
                        pno=pno,
                        record_type=rtype,
                        parent_group_key=f"ce_{rec_id}",
                        record_tags=rec_tags,
                        parent_preamble=legend_summary or None,
                    )
                )
            elif rtype in ("cad_logic_diagram", "exploded_cad_ocr"):
                intermediate.extend(
                    _make_repeated_header_items(
                        summary or rec_title,
                        heading_path=f"{rec_title} (Page {pno})",
                        header_line=rec_title,
                        pno=pno,
                        record_type=rtype,
                        parent_group_key=f"{rtype}_{rec_id}",
                        record_tags=rec_tags,
                    )
                )
            else:
                prose_text = summary or rec_title
                if prose_text:
                    intermediate.extend(
                        _make_prose_items(
                            prose_text,
                            heading_path=f"{rec_title} (Page {pno})",
                            pno=pno,
                            record_type=rtype or "standard_prose",
                            parent_group_key=f"prose_p{pno}_{rec_id}",
                            record_tags=rec_tags,
                        )
                    )

    return intermediate
