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
from src.ingestion.entity_extractor import EntityExtractor
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

            # Determine effective domain for section
            override = self._match_section_override(
                heading_path,
                domain_profile.section_overrides if domain_profile else [],
            )
            sec_domain = (
                override.domain
                if override
                else (domain_profile.primary_domain if domain_profile else "general")
            )

            # Resolve chunking strategy: override -> profile -> catalog fallback
            if override and getattr(override, "chunking_strategy", None) and override.chunking_strategy != "standard_markdown":
                sec_strategy = override.chunking_strategy
            elif override:
                sec_strategy = get_taxonomy_catalog().get_chunking_strategy(override.domain)
            elif domain_profile and getattr(domain_profile, "chunking_strategy", None) and domain_profile.chunking_strategy != "standard_markdown":
                sec_strategy = domain_profile.chunking_strategy
            else:
                sec_strategy = get_taxonomy_catalog().get_chunking_strategy(sec_domain)

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

    def _assemble_parent_child(
        self,
        items: list[dict[str, Any]],
        *,
        filename: str,
        file_type: str,
        sha256: str = "",
        domain_profile: Optional[RecommendedDomainProfile] = None,
    ) -> list[ChunkPayload]:
        """Group adjacent sibling child chunks under the same heading_path AND domain_key into parent windows."""
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
            group: list[dict[str, Any]] = []
            running_tokens = 0

            # Boundary rule: NEVER merge across different heading_path OR different domain_key
            while (
                idx < n
                and items[idx]["heading_path"] == group_heading
                and items[idx]["domain_key"] == group_domain
            ):
                item_tokens = items[idx]["token_count"]
                if group and (running_tokens + item_tokens > self.parent_max_tokens):
                    break
                group.append(items[idx])
                running_tokens += item_tokens
                idx += 1

            parent_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{sha256}:parent:{idx}:{group_domain}"))
            body_joined = "\n\n".join(g["child_text"] for g in group)
            parent_content = (
                f"[{filename} > {group_heading}]\n{body_joined}"
                if not body_joined.startswith("#")
                else body_joined
            )

            for g in group:
                child_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{sha256}:child:{chunk_index}"))
                child_text = g["child_text"]

                # Extract entities from clean child_text
                entities = self.entity_extractor.extract_entities(child_text, domain_key=g["domain_key"])

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
                    )
                )
                chunk_index += 1

        return payloads
