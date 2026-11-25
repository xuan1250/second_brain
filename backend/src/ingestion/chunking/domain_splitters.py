"""Domain-aware section splitters and token-budget packing helpers."""
from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

_PAGE_COMMENT_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")

_LATEX_EQUATION_RE = re.compile(r"(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\])")

_LAW_CLAUSE_SPLIT_RE = re.compile(
    r"(?=(?:^|\n)(?:"
    r"Chương\s+[IVXLCDM\d]+|"
    r"Mục\s+\d+|"
    r"Điều\s+\d+[\.:]|"
    r"Khoản\s+\d+[\.:]|"
    r"Điểm\s+[a-z][\.:]|"
    r"[a-z]\)\s+|"
    r"Section\s+\d+|"
    r"§\s*\d+|"
    r"\(\w+\)\s+"
    r"))",
    re.MULTILINE | re.IGNORECASE,
)

_ECON_BANNER_RE = re.compile(
    r"(\(\s*(?:in|Đơn\s+vị\s+tính:?)\s*(?:millions?|billions?|đồng|triệu\s+đồng|tỷ\s+VNĐ|\$|USD|VND)[^)]*\)|"
    r"(?:Kỳ\s+kế\s+toán|Fiscal\s+year\s+ended|Three\s+months\s+ended|For\s+the\s+year\s+ended)[^\n]+)",
    re.IGNORECASE,
)


def pack_pieces_by_tokens(
    pieces: Sequence[str],
    sep: str,
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[str]:
    """Greedily pack text pieces joined by `sep` within a `max_tokens` budget."""
    packed: list[str] = []
    current: list[str] = []
    for piece in pieces:
        candidate = sep.join([*current, piece])
        if current and count_tokens(candidate) > max_tokens:
            packed.append(sep.join(current))
            current = [piece]
        else:
            current.append(piece)
    if current:
        packed.append(sep.join(current))
    return packed


def split_lines_with_repeated_header(
    text: str,
    *,
    header_line: str,
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[str]:
    """Split multi-line text only when exceeding max_tokens, repeating header_line on each slice."""
    clean = text.strip()
    if not clean:
        return []
    if count_tokens(clean) <= max_tokens:
        return [clean]

    header = header_line.strip()
    header_tokens = count_tokens(header) if header else 0
    body_budget = max(16, max_tokens - header_tokens)

    segments: list[str] = []
    for raw_line in clean.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if count_tokens(line) <= body_budget:
            segments.append(line)
            continue
        semi_parts = [p.strip() for p in line.split("; ") if p.strip()]
        long_parts = (
            pack_pieces_by_tokens(semi_parts, "; ", body_budget, count_tokens)
            if len(semi_parts) > 1
            else [line]
        )
        for seg in long_parts:
            if count_tokens(seg) <= body_budget:
                segments.append(seg)
            else:
                segments.extend(pack_pieces_by_tokens(seg.split(), " ", body_budget, count_tokens))

    if not segments:
        return [clean]

    slices: list[str] = []
    cur_lines: list[str] = [header] if header else []
    cur_tokens = header_tokens

    for seg in segments:
        if header and seg == header and cur_lines == [header]:
            continue
        seg_tokens = count_tokens(seg)
        has_body = len(cur_lines) > (1 if header else 0)
        if has_body and (cur_tokens + seg_tokens > max_tokens):
            slices.append("\n".join(cur_lines))
            cur_lines = [header] if header else []
            cur_tokens = header_tokens
        cur_lines.append(seg)
        cur_tokens += seg_tokens

    if len(cur_lines) > (1 if header else 0) or not slices:
        slices.append("\n".join(cur_lines))

    return slices


def split_section_preserving_tables(
    text: str,
    *,
    child_max_tokens: int,
    count_tokens: Callable[[str], int],
    fallback_splitter: RecursiveCharacterTextSplitter,
) -> list[str]:
    """Split a section into child-sized chunks, repeating Markdown table headers if splitting a table."""
    if count_tokens(text) <= child_max_tokens:
        return [text]

    lines = text.splitlines()
    table_start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("|") and idx + 1 < len(lines) and "---" in lines[idx + 1]:
            table_start = idx
            break

    if table_start is not None:
        preamble = "\n".join(lines[:table_start]).strip()
        header_block = f"{lines[table_start]}\n{lines[table_start + 1]}"
        body_lines = lines[table_start + 2 :]
        header_tokens = count_tokens(header_block)
        budget = max(32, child_max_tokens - header_tokens - 8)

        slices: list[str] = []
        current_rows: list[str] = []
        current_tokens = 0
        first_slice = True

        for row in body_lines:
            row_tokens = count_tokens(row)
            if current_rows and current_tokens + row_tokens > budget:
                prefix = f"{preamble}\n\n{header_block}" if (first_slice and preamble) else header_block
                slices.append(prefix + "\n" + "\n".join(current_rows))
                first_slice = False
                current_rows = [row]
                current_tokens = row_tokens
            else:
                current_rows.append(row)
                current_tokens += row_tokens

        if current_rows:
            prefix = f"{preamble}\n\n{header_block}" if (first_slice and preamble) else header_block
            slices.append(prefix + "\n" + "\n".join(current_rows))
        return slices if slices else [text]

    parts = fallback_splitter.split_text(text)
    final_parts: list[str] = []
    for p in parts:
        if count_tokens(p) <= child_max_tokens:
            final_parts.append(p)
        else:
            words = p.split()
            step = max(16, child_max_tokens // 2)
            for i in range(0, len(words), step):
                chunk_str = " ".join(words[i : i + step])
                if chunk_str:
                    final_parts.append(chunk_str)
    return final_parts


def split_law_section(
    text: str,
    *,
    child_max_tokens: int,
    count_tokens: Callable[[str], int],
    fallback_splitter: RecursiveCharacterTextSplitter,
) -> list[str]:
    """Keep legal clauses (Điều/Khoản/Section) bound together within child_max_tokens."""
    clauses = [c.strip() for c in _LAW_CLAUSE_SPLIT_RE.split(text) if c.strip()]
    if len(clauses) <= 1:
        return split_section_preserving_tables(
            text,
            child_max_tokens=child_max_tokens,
            count_tokens=count_tokens,
            fallback_splitter=fallback_splitter,
        )

    slices: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for clause in clauses:
        clause_tokens = count_tokens(clause)

        if clause_tokens > child_max_tokens:
            if current_chunk:
                slices.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            slices.extend(
                split_section_preserving_tables(
                    clause,
                    child_max_tokens=child_max_tokens,
                    count_tokens=count_tokens,
                    fallback_splitter=fallback_splitter,
                )
            )
        elif current_tokens + clause_tokens > child_max_tokens:
            if current_chunk:
                slices.append("\n\n".join(current_chunk))
            current_chunk = [clause]
            current_tokens = clause_tokens
        else:
            current_chunk.append(clause)
            current_tokens += clause_tokens

    if current_chunk:
        slices.append("\n\n".join(current_chunk))

    return slices if slices else [text]


def split_engineering_section(
    text: str,
    *,
    child_max_tokens: int,
    count_tokens: Callable[[str], int],
    fallback_splitter: RecursiveCharacterTextSplitter,
) -> list[str]:
    """Preserve atomic LaTeX equations and table captions."""
    if not _LATEX_EQUATION_RE.search(text):
        return split_section_preserving_tables(
            text,
            child_max_tokens=child_max_tokens,
            count_tokens=count_tokens,
            fallback_splitter=fallback_splitter,
        )

    parts = _LATEX_EQUATION_RE.split(text)
    slices: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for p in parts:
        if not p.strip():
            continue
        p_tokens = count_tokens(p)
        if current_parts and (current_tokens + p_tokens > child_max_tokens):
            slices.append("\n\n".join(current_parts))
            current_parts = [p]
            current_tokens = p_tokens
        else:
            current_parts.append(p)
            current_tokens += p_tokens

    if current_parts:
        slices.append("\n\n".join(current_parts))

    return slices if slices else [text]


def split_economics_section(
    text: str,
    *,
    child_max_tokens: int,
    count_tokens: Callable[[str], int],
    fallback_splitter: RecursiveCharacterTextSplitter,
    econ_banner: Optional[str] = None,
) -> list[str]:
    """Preserve financial table windows and prefix period/currency headers."""
    raw_slices = split_section_preserving_tables(
        text,
        child_max_tokens=child_max_tokens,
        count_tokens=count_tokens,
        fallback_splitter=fallback_splitter,
    )
    if not econ_banner:
        return raw_slices

    anchored_slices: list[str] = []
    for s in raw_slices:
        if econ_banner not in s:
            anchored_slices.append(f"[{econ_banner}]\n{s}")
        else:
            anchored_slices.append(s)
    return anchored_slices
