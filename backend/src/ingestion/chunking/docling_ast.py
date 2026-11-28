"""Docling AST extraction, markdown section splitting, and parent-child payload assembly helpers."""
from __future__ import annotations

import hashlib
import sys
import uuid
from collections.abc import Callable, Sequence
from typing import Any, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.ingestion.chunking.domain_splitters import _ECON_BANNER_RE, _PAGE_COMMENT_RE
from src.ingestion.chunking.models import (
    ChunkPayload,
    format_contextual_prefix,
    format_contextualized_text,
)
from src.ingestion.constants import DEFAULT_DOMAIN, DEFAULT_EPISTEMIC
from src.ingestion.entity_extractor import EntityExtractor, expand_engineering_tag_stems


def extract_docling_ast_items(
    dl_doc: Any,
    *,
    tokenizer: Any,
    count_tokens: Callable[[str], int],
    file_type: str,
) -> list[dict[str, Any]]:
    """Run Docling HybridChunker over a DoclingDocument and return intermediate child items."""
    chunker_mod = sys.modules.get("src.ingestion.chunker")
    hybrid_cls = getattr(chunker_mod, "HybridChunker", None) if chunker_mod is not None else None
    if hybrid_cls is None:
        return []

    hybrid = hybrid_cls(
        tokenizer=tokenizer,
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
                "token_count": max(1, count_tokens(text)),
            }
        )
    return intermediate


def build_markdown_items(
    markdown: str,
    *,
    file_type: str,
    child_max_tokens: int,
    count_tokens: Callable[[str], int],
    resolve_strategy_fn: Callable[[str, Optional[Any]], tuple[str, str]],
    split_domain_fn: Callable[..., list[str]],
    domain_profile: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Split markdown by heading hierarchy and domain-aware boundaries into intermediate child items."""
    if not markdown or not markdown.strip():
        return []

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")],
        strip_headers=False,
    )
    section_docs = md_splitter.split_text(markdown)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(200, child_max_tokens * 4),
        chunk_overlap=max(32, child_max_tokens // 4),
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

        sec_domain, sec_strategy = resolve_strategy_fn(heading_path, domain_profile)

        banner_match = _ECON_BANNER_RE.search(cleaned_sec_text)
        if banner_match:
            active_econ_banner = banner_match.group(1).strip()

        sub_texts = split_domain_fn(
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
                    "token_count": max(1, count_tokens(sub_clean)),
                }
            )

    return intermediate


def assemble_parent_child_payloads(
    items: list[dict[str, Any]],
    *,
    filename: str,
    file_type: str,
    sha256: str = "",
    parent_max_tokens: int,
    entity_extractor: EntityExtractor,
    match_override_fn: Callable[[str, Sequence[Any]], Optional[Any]],
    domain_profile: Optional[Any] = None,
) -> list[ChunkPayload]:
    """Group adjacent sibling child chunks under the same heading_path, domain_key, parent_group_key, and record_type."""
    if not items:
        return []

    if not sha256:
        sha256 = hashlib.sha256(f"{filename}:{file_type}".encode("utf-8")).hexdigest()

    overrides = domain_profile.section_overrides if domain_profile else []
    for it in items:
        h_path = it["heading_path"]
        override = match_override_fn(h_path, overrides)
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
            it["domain_key"] = DEFAULT_DOMAIN
            it["subdomain_key"] = DEFAULT_DOMAIN
            it["epistemic_modality"] = DEFAULT_EPISTEMIC
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

        # Never merge across different heading_path, domain_key, parent_group_key, or record_type
        while (
            idx < n
            and items[idx]["heading_path"] == group_heading
            and items[idx]["domain_key"] == group_domain
            and items[idx].get("parent_group_key") == group_key
            and items[idx].get("record_type") == group_record_type
        ):
            item_tokens = items[idx]["token_count"]
            if group and (running_tokens + item_tokens > parent_max_tokens):
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

            entities_set = set(
                entity_extractor.extract_entities(child_text, domain_key=g["domain_key"])
            )
            record_tags = g.get("record_tags") or []
            if record_tags:
                entities_set.update(expand_engineering_tag_stems(record_tags))
            entities = sorted(entities_set)

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
