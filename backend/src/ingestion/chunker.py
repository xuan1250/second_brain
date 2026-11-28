"""Parent-child hierarchical chunker with domain contextual prefixing and bilingual boundary preservation."""
from __future__ import annotations

import hashlib
import logging
import sys
from typing import Any, Optional, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings
from src.ingestion.chunking.docling_ast import (
    assemble_parent_child_payloads,
    build_markdown_items,
    extract_docling_ast_items,
)
from src.ingestion.chunking.domain_splitters import (
    pack_pieces_by_tokens,
    split_economics_section,
    split_engineering_section,
    split_law_section,
    split_lines_with_repeated_header,
    split_section_preserving_tables,
)
from src.ingestion.chunking.engineering_records import (
    _parse_markdown_pages,
    build_engineering_intermediate_items,
    chunk_schedule_records,
)
from src.ingestion.chunking.models import (
    ChunkPayload,
    _build_legend_summary,
    build_legend_summary,
    format_contextual_prefix,
    format_contextualized_text,
)
from src.ingestion.chunking.tokenizer import (
    BaseTokenizer,
    DoclingDocument,
    FastRegexTokenizer,
    HybridChunker,
    _build_tokenizer,
)
from src.ingestion.classifier import (
    RecommendedDomainProfile,
    SectionOverride,
    get_taxonomy_catalog,
)
from src.ingestion.constants import (
    DEFAULT_DOMAIN,
    normalize_chunking_strategy,
)
from src.ingestion.entity_extractor import EntityExtractor
from src.ingestion.parser import ParseResult

logger = logging.getLogger(__name__)


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

        hybrid_cls = getattr(sys.modules[__name__], "HybridChunker", None)
        if file_type != "xlsx" and parse_result.docling_dict and hybrid_cls is not None:
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
        intermediate = extract_docling_ast_items(
            dl_doc,
            tokenizer=self.tokenizer,
            count_tokens=self.count_tokens,
            file_type=file_type,
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

    def _chunk_markdown(
        self,
        markdown: str,
        *,
        filename: str,
        file_type: str,
        sha256: str = "",
        domain_profile: Optional[RecommendedDomainProfile] = None,
    ) -> list[ChunkPayload]:
        intermediate = build_markdown_items(
            markdown,
            file_type=file_type,
            child_max_tokens=self.child_max_tokens,
            count_tokens=self.count_tokens,
            resolve_strategy_fn=self._resolve_section_strategy,
            split_domain_fn=self._split_section_domain_aware,
            domain_profile=domain_profile,
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

        if strategy == "academic_equations":
            return split_engineering_section(
                text,
                child_max_tokens=self.child_max_tokens,
                count_tokens=self.count_tokens,
                fallback_splitter=fallback_splitter,
            )
        if strategy == "statutory_clauses":
            return split_law_section(
                text,
                child_max_tokens=self.child_max_tokens,
                count_tokens=self.count_tokens,
                fallback_splitter=fallback_splitter,
            )
        if strategy == "financial_tables":
            return split_economics_section(
                text,
                child_max_tokens=self.child_max_tokens,
                count_tokens=self.count_tokens,
                fallback_splitter=fallback_splitter,
                econ_banner=econ_banner,
            )
        return split_section_preserving_tables(
            text,
            child_max_tokens=self.child_max_tokens,
            count_tokens=self.count_tokens,
            fallback_splitter=fallback_splitter,
        )

    def _resolve_section_strategy(
        self,
        heading_path: str,
        domain_profile: Optional[RecommendedDomainProfile],
    ) -> tuple[str, str]:
        """Resolve (domain_key, chunking_strategy) for a heading path."""
        catalog = get_taxonomy_catalog()
        override = self._match_section_override(
            heading_path,
            domain_profile.section_overrides if domain_profile else [],
        )
        sec_domain = (
            override.domain
            if override
            else (domain_profile.primary_domain if domain_profile else DEFAULT_DOMAIN)
        )
        if override:
            sec_strategy = normalize_chunking_strategy(
                getattr(override, "chunking_strategy", None),
                catalog.get_chunking_strategy(override.domain),
            )
        elif domain_profile:
            sec_strategy = normalize_chunking_strategy(
                getattr(domain_profile, "chunking_strategy", None),
                catalog.get_chunking_strategy(sec_domain),
            )
        else:
            sec_strategy = catalog.get_chunking_strategy(sec_domain)
        return sec_domain, sec_strategy

    def _pack_pieces_by_tokens(self, pieces: Sequence[str], sep: str, max_tokens: int) -> list[str]:
        return pack_pieces_by_tokens(pieces, sep, max_tokens, self.count_tokens)

    def _split_lines_with_repeated_header(
        self,
        text: str,
        *,
        header_line: str,
        max_tokens: int,
    ) -> list[str]:
        return split_lines_with_repeated_header(
            text,
            header_line=header_line,
            max_tokens=max_tokens,
            count_tokens=self.count_tokens,
        )

    def _chunk_schedule_records(
        self,
        schedule_recs: Sequence[dict[str, Any]],
        *,
        pno: int,
    ) -> list[dict[str, Any]]:
        return chunk_schedule_records(
            schedule_recs,
            pno=pno,
            child_max_tokens=self.child_max_tokens,
            count_tokens=self.count_tokens,
        )

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

        def _split_prose_for_heading(
            text: str,
            heading_path: str,
            fallback_splitter: RecursiveCharacterTextSplitter,
        ) -> list[str]:
            sec_domain, sec_strategy = self._resolve_section_strategy(heading_path, domain_profile)
            return self._split_section_domain_aware(
                text,
                sec_domain=sec_domain,
                sec_strategy=sec_strategy,
                fallback_splitter=fallback_splitter,
            )

        intermediate = build_engineering_intermediate_items(
            parse_result.markdown or "",
            profile=profile,
            child_max_tokens=self.child_max_tokens,
            count_tokens=self.count_tokens,
            split_prose_fn=_split_prose_for_heading,
            split_lines_fn=self._split_lines_with_repeated_header,
            chunk_schedule_fn=self._chunk_schedule_records,
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
        return assemble_parent_child_payloads(
            items,
            filename=filename,
            file_type=file_type,
            sha256=sha256,
            parent_max_tokens=self.parent_max_tokens,
            entity_extractor=self.entity_extractor,
            match_override_fn=self._match_section_override,
            domain_profile=domain_profile,
        )


__all__ = [
    "BaseTokenizer",
    "ChunkPayload",
    "DoclingDocument",
    "FastRegexTokenizer",
    "HierarchicalChunker",
    "HybridChunker",
    "_build_legend_summary",
    "_build_tokenizer",
    "_parse_markdown_pages",
    "build_legend_summary",
    "format_contextual_prefix",
    "format_contextualized_text",
]
