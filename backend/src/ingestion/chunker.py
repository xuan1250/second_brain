"""Parent-child hierarchical chunker with domain contextual prefixing and bilingual boundary preservation."""
from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

try:
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
    from docling_core.types.doc import DoclingDocument
except (ImportError, OSError):
    HybridChunker = None  # type: ignore

    class BaseTokenizer:  # type: ignore
        def __init__(self, max_tokens: int = 256, **kwargs: Any) -> None:
            self.max_tokens = max_tokens

        def count_tokens(self, text: str) -> int:
            return 0

    DoclingDocument = None  # type: ignore

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.config import get_settings
from src.ingestion.classifier import (
    RecommendedDomainProfile,
    SectionOverride,
    get_taxonomy_catalog,
)
from src.ingestion.entity_extractor import EntityExtractor, expand_engineering_tag_stems
from src.ingestion.parser import ParseResult

logger = logging.getLogger(__name__)

_PAGE_COMMENT_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")

# Boundary patterns for domain-specific chunking
_LATEX_EQUATION_RE = re.compile(r"(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\])")

# Law statutory boundaries (EN & VI)
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

_LAW_PROVISO_RE = re.compile(
    r"\b(Notwithstanding\b|Provided(?:\s+always)?,\s+that\b|Subject\s+to\b|Trừ\s+trường\s+hợp\b|Với\s+điều\s+kiện\s+là\b)",
    re.IGNORECASE,
)

# Economics period & currency banners
_ECON_BANNER_RE = re.compile(
    r"(\(\s*(?:in|Đơn\s+vị\s+tính:?)\s*(?:millions?|billions?|đồng|triệu\s+đồng|tỷ\s+VNĐ|\$|USD|VND)[^)]*\)|"
    r"(?:Kỳ\s+kế\s+toán|Fiscal\s+year\s+ended|Three\s+months\s+ended|For\s+the\s+year\s+ended)[^\n]+)",
    re.IGNORECASE,
)


class FastRegexTokenizer(BaseTokenizer):
    """Deterministic, zero-network tokenizer aligned with BAAI/bge-m3 token budgets."""

    max_tokens: int = 256
    model_name: str = "BAAI/bge-m3"

    def __init__(self, max_tokens: int = 256, **kwargs: Any) -> None:
        super().__init__(max_tokens=max_tokens, **kwargs)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self) -> Any:
        return self.count_tokens


def _build_tokenizer(max_tokens: int, use_hf_tokenizer: bool = False) -> BaseTokenizer:
    if use_hf_tokenizer:
        try:
            from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
            from transformers import AutoTokenizer

            hf_tok = AutoTokenizer.from_pretrained("BAAI/bge-m3", local_files_only=True)
            return HuggingFaceTokenizer(tokenizer=hf_tok, max_tokens=max_tokens)
        except Exception:
            pass
    return FastRegexTokenizer(max_tokens=max_tokens)


@dataclass
class ChunkPayload:
    """Single child retrieval chunk paired with parent window, domain metadata, and contextual prefix."""

    chunk_id: str
    chunk_index: int
    parent_id: str
    source_filename: str
    file_type: str
    heading_path: str
    page_numbers: list[int] = field(default_factory=list)
    token_count: int = 0
    child_text: str = ""
    contextualized_text: str = ""
    parent_content: str = ""

    # Multi-domain categorization & contextual anchoring
    domain_key: Optional[str] = None
    subdomain_key: Optional[str] = None
    epistemic_modality: Optional[str] = None
    authority_tier: int = 3
    jurisdiction: Optional[str] = None
    effective_year: Optional[int] = None
    contextual_prefix: Optional[str] = None
    associated_entities: list[str] = field(default_factory=list)
    record_type: Optional[str] = None


def format_contextual_prefix(
    domain: Optional[str] = None,
    subdomain: Optional[str] = None,
    epistemic_modality: Optional[str] = None,
    authority_tier: int = 3,
    code: Optional[str] = None,
    source: str = "",
    heading_path: str = "",
) -> str:
    """Construct Anthropic-style contextual prefix header."""
    parts: list[str] = []
    dom = (domain or "general").capitalize()
    parts.append(f"Domain: {dom}")
    if subdomain and subdomain != "general":
        parts.append(f"Subdomain: {subdomain}")
    if epistemic_modality and epistemic_modality != "general":
        parts.append(f"Epistemic: {epistemic_modality.capitalize()}")
    parts.append(f"Authority: Tier {authority_tier}")
    if code:
        parts.append(f"Code: {code}")

    src_str = source or "Document"
    clean_h = (heading_path or "").strip()
    if clean_h and clean_h != "Document":
        src_str = f"{src_str} > {clean_h}"
    parts.append(f"Source: {src_str}")

    return "[" + " | ".join(parts) + "]"


def format_contextualized_text(
    filename: str,
    heading_path: str,
    child_text: str,
    contextual_prefix: Optional[str] = None,
) -> str:
    """Format contextualized string for embedding: f'{contextual_prefix}\n{child_text}'."""
    if contextual_prefix:
        return f"{contextual_prefix}\n{child_text}"
    clean_heading = (heading_path or "Document").strip()
    return f"{filename} > {clean_heading}: {child_text}"


def _build_legend_summary(legend_ctx: dict[str, Any]) -> str:
    """Build a concise legend summary combining symbol_map and short notes/summary."""
    if not legend_ctx or not isinstance(legend_ctx, dict):
        return ""
    parts: list[str] = []
    symbol_map = legend_ctx.get("symbol_map") or {}
    if isinstance(symbol_map, dict) and symbol_map:
        syms = "; ".join(f"{k} = {v}" for k, v in symbol_map.items())
        parts.append(f"Legend Symbols: {syms}")

    short_summary = str(legend_ctx.get("summary") or "").strip()
    if short_summary:
        parts.append(short_summary)
    else:
        notes_map = legend_ctx.get("notes_map") or {}
        if isinstance(notes_map, dict) and notes_map:
            note_items = [f"{k}: {v}" for k, v in notes_map.items()]
            joined_notes = " | ".join(note_items)
            if len(joined_notes) > 360:
                kept: list[str] = []
                total_len = 0
                for item in note_items:
                    if kept and total_len + len(item) + 3 > 340:
                        break
                    kept.append(item)
                    total_len += len(item) + 3
                remaining = len(note_items) - len(kept)
                suffix = f" (+{remaining} more notes)" if remaining > 0 else ""
                joined_notes = " | ".join(kept) + suffix
            parts.append(f"Legend Notes: {joined_notes}")

    if not parts:
        raw_text = str(legend_ctx.get("raw_text") or "").strip()
        if raw_text:
            parts.append(raw_text[:280].strip())

    return "\n".join(parts).strip()


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


class HierarchicalChunker:
    """Builds parent-child chunk hierarchies with domain-specific boundary preservation."""

    def __init__(
        self,
        child_max_tokens: int | None = None,
        parent_max_tokens: int | None = None,
        use_hf_tokenizer: bool = False,
    ) -> None:
        settings = get_settings()
        self.child_max_tokens = child_max_tokens or settings.chunking.child_max_tokens
        self.parent_max_tokens = parent_max_tokens or settings.chunking.parent_max_tokens
        self.tokenizer = _build_tokenizer(self.child_max_tokens, use_hf_tokenizer=use_hf_tokenizer)
        self.entity_extractor = EntityExtractor()

    def count_tokens(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)

    def chunk(
        self,
        parse_result: ParseResult,
        *,
        domain_profile: Optional[RecommendedDomainProfile] = None,
    ) -> list[ChunkPayload]:
        """Chunk a ParseResult into parent-child ChunkPayloads with domain context."""
        file_type = parse_result.file_type.lower().lstrip(".")
        filename = parse_result.filename
        sha256 = parse_result.sha256 or hashlib.sha256((parse_result.markdown or "").encode("utf-8")).hexdigest()

        extra_meta = parse_result.extra_meta or {}
        if extra_meta.get("has_engineering_records") and extra_meta.get("engineering_profile"):
            eng_chunks = self._chunk_engineering_records(
                parse_result,
                extra_meta=extra_meta,
                domain_profile=domain_profile,
            )
            if eng_chunks:
                return eng_chunks

        # If DoclingDocument model validation works and HybridChunker is available, use it
        if file_type != "xlsx" and parse_result.docling_dict and HybridChunker is not None:
            try:
                dl_doc = DoclingDocument.model_validate(parse_result.docling_dict)
                chunks = self._chunk_docling_ast(
                    dl_doc,
                    filename=filename,
                    file_type=file_type,
                    sha256=sha256,
                    domain_profile=domain_profile,
                )
                if chunks:
                    return chunks
            except Exception:
                pass

        return self._chunk_markdown(
            parse_result.markdown,
            filename=filename,
            file_type=file_type,
            sha256=sha256,
            extra_meta=parse_result.extra_meta,
            domain_profile=domain_profile,
        )

    def _match_section_override(
        self,
        heading_path: str,
        overrides: Sequence[SectionOverride],
    ) -> Optional[SectionOverride]:
        """Find matching section override by heading prefix or path."""
        if not overrides or not heading_path:
            return None
        h_clean = heading_path.strip().lower()
        for ov in overrides:
            ov_clean = ov.heading_path.strip().lower()
            if ov_clean == h_clean or ov_clean in h_clean or h_clean in ov_clean:
                return ov
        return None

    def _chunk_docling_ast(
        self,
        dl_doc: Any,
        *,
        filename: str,
        file_type: str,
        sha256: str = "",
        domain_profile: Optional[RecommendedDomainProfile] = None,
    ) -> list[ChunkPayload]:
        hybrid = HybridChunker(
            tokenizer=self.tokenizer,
            repeat_table_header=True,
            merge_peers=True,
        )
        raw_chunks = list(hybrid.chunk(dl_doc))
        if not raw_chunks:
            return []

        intermediate: list[dict[str, Any]] = []
        for rc in raw_chunks:
            text = (rc.text or "").strip()
            if not text:
                continue
            headings = list(rc.meta.headings or []) if rc.meta else []
            heading_path = " > ".join(h.strip() for h in headings if h and h.strip()) or "Document"

            pages: list[int] = []
            if file_type in ("pdf", "pptx") and rc.meta and rc.meta.doc_items:
                page_set: set[int] = set()
                for item in rc.meta.doc_items:
                    for prov in getattr(item, "prov", []) or []:
                        pno = getattr(prov, "page_no", None)
                        if pno is not None:
                            page_set.add(int(pno))
                pages = sorted(page_set)

            intermediate.append(
                {
                    "heading_path": heading_path,
                    "page_numbers": pages,
                    "child_text": text,
                    "token_count": max(1, self.count_tokens(text)),
                }
            )

        return self._assemble_parent_child(
            intermediate,
            filename=filename,
            file_type=file_type,
            sha256=sha256,
            domain_profile=domain_profile,
        )

    def _chunk_markdown(
        self,
        markdown: str,
        *,
        filename: str,
        file_type: str,
        sha256: str = "",
        extra_meta: Optional[dict[str, Any]] = None,
        domain_profile: Optional[RecommendedDomainProfile] = None,
    ) -> list[ChunkPayload]:
        if not markdown or not markdown.strip():
            return []

        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        section_docs = md_splitter.split_text(markdown)

        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(200, self.child_max_tokens * 4),
            chunk_overlap=max(32, self.child_max_tokens // 4),
            separators=["\n\n", "\n", " ", ""],
        )

        intermediate: list[dict[str, Any]] = []
        current_page: Optional[int] = 1 if file_type == "pdf" else None

        active_econ_banner: Optional[str] = None

        for sec in section_docs:
            meta = sec.metadata or {}
            headings = [meta[k].strip() for k in ("h1", "h2", "h3", "h4") if k in meta and meta[k].strip()]
            heading_path = " > ".join(headings) if headings else "Document"

            raw_sec_text = sec.page_content or ""
            found_pages = [int(m) for m in _PAGE_COMMENT_RE.findall(raw_sec_text)]
            if found_pages:
                current_page = found_pages[0]

            cleaned_sec_text = _PAGE_COMMENT_RE.sub("", raw_sec_text).strip()
            if not cleaned_sec_text:
                continue

            # Determine effective domain & chunking strategy for section
            sec_domain, sec_strategy = self._resolve_section_strategy(heading_path, domain_profile)

            # Check for economics currency/period banner
            banner_match = _ECON_BANNER_RE.search(cleaned_sec_text)
            if banner_match:
                active_econ_banner = banner_match.group(1).strip()

            # Domain-specific section splitting
            sub_texts = self._split_section_domain_aware(
                cleaned_sec_text,
                sec_domain=sec_domain,
                sec_strategy=sec_strategy,
                fallback_splitter=char_splitter,
                econ_banner=active_econ_banner if sec_strategy == "financial_tables" else None,
            )

            for sub in sub_texts:
                sub_clean = sub.strip()
                if not sub_clean:
                    continue

                if file_type == "pdf":
                    pages = sorted(set(found_pages)) if found_pages else ([current_page] if current_page else [1])
                elif file_type == "pptx":
                    pages = sorted(set(found_pages)) if found_pages else [1]
                else:
                    pages = []

                intermediate.append(
                    {
                        "heading_path": heading_path,
                        "page_numbers": pages,
                        "child_text": sub_clean,
                        "token_count": max(1, self.count_tokens(sub_clean)),
                    }
                )

        return self._assemble_parent_child(
            intermediate,
            filename=filename,
            file_type=file_type,
            sha256=sha256,
            domain_profile=domain_profile,
        )

    def _split_section_domain_aware(
        self,
        text: str,
        *,
        sec_domain: str,
        sec_strategy: Optional[str] = None,
        fallback_splitter: RecursiveCharacterTextSplitter,
        econ_banner: Optional[str] = None,
    ) -> list[str]:
        """Split text while enforcing Law clause binding, Engineering atomic equations, and Economics banners."""
        if self.count_tokens(text) <= self.child_max_tokens:
            return [text]

        strategy = sec_strategy or get_taxonomy_catalog().get_chunking_strategy(sec_domain)

        # 1. Engineering / Academic Equations: Protect LaTeX equations and Markdown tables
        if strategy == "academic_equations":
            return self._split_engineering_section(text, fallback_splitter)

        # 2. Law / Statutory Clauses: Respect statutory clauses (Điều, Khoản, Provisos)
        if strategy == "statutory_clauses":
            return self._split_law_section(text, fallback_splitter)

        # 3. Economics / Financial Tables: Markdown table slices with repeated period/currency banners
        if strategy == "financial_tables":
            return self._split_economics_section(text, fallback_splitter, econ_banner)

        # Default fallback splitter with table preservation
        return self._split_section_preserving_tables(text, fallback_splitter)

    def _split_law_section(
        self,
        text: str,
        fallback_splitter: RecursiveCharacterTextSplitter,
    ) -> list[str]:
        """Keep legal clauses (Điều/Khoản/Section) and provisos bound together."""
        # Split text into candidate statutory clause blocks
        clauses = [c.strip() for c in _LAW_CLAUSE_SPLIT_RE.split(text) if c.strip()]
        if len(clauses) <= 1:
            return self._split_section_preserving_tables(text, fallback_splitter)

        slices: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for clause in clauses:
            clause_tokens = self.count_tokens(clause)

            if clause_tokens > self.child_max_tokens:
                # If single clause is oversized, flush current chunk and split clause with fallback
                if current_chunk:
                    slices.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                slices.extend(self._split_section_preserving_tables(clause, fallback_splitter))
            elif current_tokens + clause_tokens > self.child_max_tokens:
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

    def _split_engineering_section(
        self,
        text: str,
        fallback_splitter: RecursiveCharacterTextSplitter,
    ) -> list[str]:
        """Preserve atomic LaTeX equations and table captions."""
        # Check for LaTeX display math
        has_latex = bool(_LATEX_EQUATION_RE.search(text))
        if not has_latex:
            return self._split_section_preserving_tables(text, fallback_splitter)

        # Partition text into math and non-math segments
        parts = _LATEX_EQUATION_RE.split(text)
        slices: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for p in parts:
            if not p.strip():
                continue
            p_tokens = self.count_tokens(p)
            if current_parts and (current_tokens + p_tokens > self.child_max_tokens):
                slices.append("\n\n".join(current_parts))
                current_parts = [p]
                current_tokens = p_tokens
            else:
                current_parts.append(p)
                current_tokens += p_tokens

        if current_parts:
            slices.append("\n\n".join(current_parts))

        return slices if slices else [text]

    def _split_economics_section(
        self,
        text: str,
        fallback_splitter: RecursiveCharacterTextSplitter,
        econ_banner: Optional[str] = None,
    ) -> list[str]:
        """Preserve financial table windows and prefix period/currency headers."""
        raw_slices = self._split_section_preserving_tables(text, fallback_splitter)
        if not econ_banner:
            return raw_slices

        anchored_slices: list[str] = []
        for s in raw_slices:
            if econ_banner not in s:
                anchored_slices.append(f"[{econ_banner}]\n{s}")
            else:
                anchored_slices.append(s)
        return anchored_slices

    def _split_section_preserving_tables(
        self,
        text: str,
        fallback_splitter: RecursiveCharacterTextSplitter,
    ) -> list[str]:
        """Split a section into child-sized chunks, repeating Markdown table headers if splitting a table."""
        if self.count_tokens(text) <= self.child_max_tokens:
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
            header_tokens = self.count_tokens(header_block)
            budget = max(32, self.child_max_tokens - header_tokens - 8)

            slices: list[str] = []
            current_rows: list[str] = []
            current_tokens = 0
            first_slice = True

            for row in body_lines:
                row_tokens = self.count_tokens(row)
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
            if self.count_tokens(p) <= self.child_max_tokens:
                final_parts.append(p)
            else:
                words = p.split()
                step = max(16, self.child_max_tokens // 2)
                for i in range(0, len(words), step):
                    chunk_str = " ".join(words[i : i + step])
                    if chunk_str:
                        final_parts.append(chunk_str)
        return final_parts

    def _resolve_section_strategy(
        self,
        heading_path: str,
        domain_profile: Optional[RecommendedDomainProfile],
    ) -> tuple[str, str]:
        """Resolve (domain_key, chunking_strategy) for a heading path."""
        override = self._match_section_override(
            heading_path,
            domain_profile.section_overrides if domain_profile else [],
        )
        sec_domain = (
            override.domain
            if override
            else (domain_profile.primary_domain if domain_profile else "general")
        )
        if override and getattr(override, "chunking_strategy", None) and override.chunking_strategy != "standard_markdown":
            sec_strategy = override.chunking_strategy
        elif override:
            sec_strategy = get_taxonomy_catalog().get_chunking_strategy(override.domain)
        elif domain_profile and getattr(domain_profile, "chunking_strategy", None) and domain_profile.chunking_strategy != "standard_markdown":
            sec_strategy = domain_profile.chunking_strategy
        else:
            sec_strategy = get_taxonomy_catalog().get_chunking_strategy(sec_domain)
        return sec_domain, sec_strategy

    def _pack_pieces_by_tokens(
        self,
        pieces: Sequence[str],
        sep: str,
        max_tokens: int,
    ) -> list[str]:
        """Greedily pack text pieces joined by `sep` within a `max_tokens` budget."""
        packed: list[str] = []
        current: list[str] = []
        for piece in pieces:
            candidate = sep.join([*current, piece])
            if current and self.count_tokens(candidate) > max_tokens:
                packed.append(sep.join(current))
                current = [piece]
            else:
                current.append(piece)
        if current:
            packed.append(sep.join(current))
        return packed

    def _split_lines_with_repeated_header(
        self,
        text: str,
        *,
        header_line: str,
        max_tokens: int,
    ) -> list[str]:
        """Split multi-line text only when exceeding max_tokens, repeating header_line on each slice."""
        clean = text.strip()
        if not clean:
            return []
        if self.count_tokens(clean) <= max_tokens:
            return [clean]

        header = header_line.strip()
        header_tokens = self.count_tokens(header) if header else 0
        body_budget = max(16, max_tokens - header_tokens)

        segments: list[str] = []
        for raw_line in clean.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if self.count_tokens(line) <= body_budget:
                segments.append(line)
                continue
            # Split long single lines cleanly on semicolon clauses before word splitting
            semi_parts = [p.strip() for p in line.split("; ") if p.strip()]
            long_parts = (
                self._pack_pieces_by_tokens(semi_parts, "; ", body_budget)
                if len(semi_parts) > 1
                else [line]
            )
            for seg in long_parts:
                if self.count_tokens(seg) <= body_budget:
                    segments.append(seg)
                else:
                    segments.extend(self._pack_pieces_by_tokens(seg.split(), " ", body_budget))

        if not segments:
            return [clean]

        slices: list[str] = []
        cur_lines: list[str] = [header] if header else []
        cur_tokens = header_tokens

        for seg in segments:
            if header and seg == header and cur_lines == [header]:
                continue
            seg_tokens = self.count_tokens(seg)
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

    def _chunk_schedule_records(
        self,
        schedule_recs: Sequence[dict[str, Any]],
        *,
        pno: int,
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
        target_child_tokens = min(self.child_max_tokens, 240) if self.child_max_tokens >= 180 else self.child_max_tokens

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
                    "token_count": max(1, self.count_tokens(child_text)),
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
            cand_tokens = self.count_tokens(candidate_text)
            if cur_rows and (
                len(cur_rows) >= 5
                or cand_tokens > self.child_max_tokens
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

    def _chunk_engineering_records(
        self,
        parse_result: ParseResult,
        *,
        extra_meta: dict[str, Any],
        domain_profile: Optional[RecommendedDomainProfile] = None,
    ) -> list[ChunkPayload]:
        """Chunk engineering CAD records directly without Docling AST flattening."""
        file_type = parse_result.file_type.lower().lstrip(".")
        filename = parse_result.filename
        sha256 = parse_result.sha256 or hashlib.sha256((parse_result.markdown or "").encode("utf-8")).hexdigest()

        profile = extra_meta.get("engineering_profile") or {}
        if not isinstance(profile, dict):
            return []

        legend_ctx = profile.get("legend_context") or {}
        legend_summary = _build_legend_summary(legend_ctx)
        page_modes = profile.get("page_modes") or {}
        page_text_map = _parse_markdown_pages(parse_result.markdown or "")

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
            chunk_size=max(200, self.child_max_tokens * 4),
            chunk_overlap=max(32, self.child_max_tokens // 4),
            separators=["\n\n", "\n", " ", ""],
        )

        intermediate: list[dict[str, Any]] = []

        def _make_repeated_header_items(
            text: str,
            *,
            heading_path: str,
            header_line: str,
            pno: int,
            record_type: str,
            parent_group_key: str,
            record_tags: list[str],
            parent_preamble: str | None = None,
        ) -> list[dict[str, Any]]:
            built: list[dict[str, Any]] = []
            for s in self._split_lines_with_repeated_header(
                text,
                header_line=header_line,
                max_tokens=self.child_max_tokens,
            ):
                item: dict[str, Any] = {
                    "heading_path": heading_path,
                    "page_numbers": [pno],
                    "child_text": s,
                    "token_count": max(1, self.count_tokens(s)),
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
            record_tags: list[str] | None = None,
        ) -> list[dict[str, Any]]:
            sec_domain, sec_strategy = self._resolve_section_strategy(heading_path, domain_profile)
            built: list[dict[str, Any]] = []
            for sub in self._split_section_domain_aware(
                text,
                sec_domain=sec_domain,
                sec_strategy=sec_strategy,
                fallback_splitter=char_splitter,
            ):
                sub_clean = sub.strip()
                if not sub_clean:
                    continue
                item: dict[str, Any] = {
                    "heading_path": heading_path,
                    "page_numbers": [pno],
                    "child_text": sub_clean,
                    "token_count": max(1, self.count_tokens(sub_clean)),
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
                    intermediate.extend(self._chunk_schedule_records(run, pno=pno))
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

        if not intermediate:
            return []

        return self._assemble_parent_child(
            intermediate,
            filename=filename,
            file_type=file_type,
            sha256=sha256,
            domain_profile=domain_profile,
        )

    def _assemble_parent_child(
        self,
        items: list[dict[str, Any]],
        *,
        filename: str,
        file_type: str,
        sha256: str = "",
        domain_profile: Optional[RecommendedDomainProfile] = None,
    ) -> list[ChunkPayload]:
        """Group adjacent sibling child chunks under the same heading_path, domain_key, parent_group_key, and record_type."""
        if not items:
            return []

        if not sha256:
            sha256 = hashlib.sha256(f"{filename}:{file_type}".encode("utf-8")).hexdigest()

        # Attach domain attributes to each item first
        overrides = domain_profile.section_overrides if domain_profile else []
        for it in items:
            h_path = it["heading_path"]
            override = self._match_section_override(h_path, overrides)
            if override:
                it["domain_key"] = override.domain
                it["subdomain_key"] = override.subdomain
                it["epistemic_modality"] = override.epistemic_modality
                it["authority_tier"] = override.authority_tier
                it["jurisdiction"] = override.jurisdiction
            elif domain_profile:
                it["domain_key"] = domain_profile.primary_domain
                it["subdomain_key"] = domain_profile.primary_subdomain
                it["epistemic_modality"] = domain_profile.epistemic_modality
                it["authority_tier"] = domain_profile.authority_tier
                it["jurisdiction"] = domain_profile.jurisdiction
            else:
                it["domain_key"] = "general"
                it["subdomain_key"] = "general"
                it["epistemic_modality"] = "general"
                it["authority_tier"] = 3
                it["jurisdiction"] = None

            it["effective_year"] = domain_profile.effective_year if domain_profile else None

        payloads: list[ChunkPayload] = []
        idx = 0
        n = len(items)
        chunk_index = 0

        while idx < n:
            group_heading = items[idx]["heading_path"]
            group_domain = items[idx]["domain_key"]
            group_key = items[idx].get("parent_group_key")
            group_record_type = items[idx].get("record_type")
            group: list[dict[str, Any]] = []
            running_tokens = 0

            # Boundary rule: NEVER merge across different heading_path, domain_key, parent_group_key, or record_type
            while (
                idx < n
                and items[idx]["heading_path"] == group_heading
                and items[idx]["domain_key"] == group_domain
                and items[idx].get("parent_group_key") == group_key
                and items[idx].get("record_type") == group_record_type
            ):
                item_tokens = items[idx]["token_count"]
                if group and (running_tokens + item_tokens > self.parent_max_tokens):
                    break
                group.append(items[idx])
                running_tokens += item_tokens
                idx += 1

            parent_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{sha256}:parent:{idx}:{group_domain}"))
            body_joined = "\n\n".join(g["child_text"] for g in group)

            preambles: list[str] = []
            for g in group:
                p_txt = str(g.get("parent_preamble") or "").strip()
                if p_txt and p_txt not in preambles and p_txt not in body_joined:
                    preambles.append(p_txt)
            preamble_block = "\n".join(preambles)

            if preamble_block:
                parent_content = f"[{filename} > {group_heading}]\n{preamble_block}\n\n{body_joined}"
            else:
                parent_content = (
                    f"[{filename} > {group_heading}]\n{body_joined}"
                    if not body_joined.startswith("#")
                    else body_joined
                )

            for g in group:
                child_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{sha256}:child:{chunk_index}"))
                child_text = g["child_text"]

                # Extract entities from clean child_text and union with expanded record_tags
                entities_set = set(
                    self.entity_extractor.extract_entities(child_text, domain_key=g["domain_key"])
                )
                record_tags = g.get("record_tags") or []
                if record_tags:
                    entities_set.update(expand_engineering_tag_stems(record_tags))
                entities = sorted(entities_set)

                # Determine code citation for prefix if any
                code_citation = entities[0] if entities else None

                prefix = format_contextual_prefix(
                    domain=g["domain_key"],
                    subdomain=g["subdomain_key"],
                    epistemic_modality=g["epistemic_modality"],
                    authority_tier=g["authority_tier"],
                    code=code_citation,
                    source=filename,
                    heading_path=group_heading,
                )

                contextualized = format_contextualized_text(
                    filename=filename,
                    heading_path=group_heading,
                    child_text=child_text,
                    contextual_prefix=prefix,
                )

                payloads.append(
                    ChunkPayload(
                        chunk_id=child_id,
                        chunk_index=chunk_index,
                        parent_id=parent_id,
                        source_filename=filename,
                        file_type=file_type,
                        heading_path=group_heading,
                        page_numbers=list(g["page_numbers"]),
                        token_count=int(g["token_count"]),
                        child_text=child_text,
                        contextualized_text=contextualized,
                        parent_content=parent_content,
                        domain_key=g["domain_key"],
                        subdomain_key=g["subdomain_key"],
                        epistemic_modality=g["epistemic_modality"],
                        authority_tier=g["authority_tier"],
                        jurisdiction=g["jurisdiction"],
                        effective_year=g["effective_year"],
                        contextual_prefix=prefix,
                        associated_entities=entities,
                        record_type=g.get("record_type"),
                    )
                )
                chunk_index += 1

        return payloads
