"""Citation construction, marker sanitization, and XML synthesis context formatting."""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from src.rag.grader import format_documents_xml
from src.rag.qdrant_filters import normalize_page_numbers
from src.rag.state import Citation, format_citation_source

_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")


def validate_and_sanitize_citations(text: str, citations: Sequence[Citation]) -> str:
    """Remove hallucinated out-of-range `[k]` markers not in `1..len(citations)`."""
    max_idx = len(citations)
    if max_idx == 0:
        return _CITATION_MARKER_RE.sub("", text).strip()

    def _replace(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if 1 <= idx <= max_idx:
            return match.group(0)
        return ""

    cleaned = _CITATION_MARKER_RE.sub(_replace, text)
    # Clean up any double spaces before punctuation left by stripped markers
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    return cleaned.strip()


def build_citations_from_chunks(chunks: Sequence[dict[str, Any]]) -> list[Citation]:
    """Create 1-indexed snake_case `Citation` dicts from graded chunks."""
    citations: list[Citation] = []
    for idx, chunk in enumerate(chunks, start=1):
        pages = normalize_page_numbers(chunk.get("page_numbers"))
        excerpt = str(chunk.get("child_text") or chunk.get("parent_content") or "").strip()
        if len(excerpt) > 400:
            excerpt = excerpt[:397] + "..."
        citations.append(
            Citation(
                citation_index=idx,
                doc_id=str(chunk.get("doc_id") or ""),
                chunk_id=str(chunk.get("chunk_id") or ""),
                source_filename=str(chunk.get("source_filename") or ""),
                page_numbers=pages,
                heading_path=str(chunk.get("heading_path") or ""),
                excerpt=excerpt,
                score=round(float(chunk.get("score", 0.0)), 6),
                domain=str(chunk.get("domain") or "general"),
                subdomain=str(chunk.get("subdomain") or "general"),
                authority_tier=int(chunk.get("authority_tier") if chunk.get("authority_tier") is not None else 3),
                epistemic_type=str(chunk.get("epistemic_type") or "general"),
                jurisdiction=str(chunk.get("jurisdiction") or ""),
                is_current=bool(chunk.get("is_current", True)),
            )
        )
    return citations


def format_synthesis_context(
    chunks: Sequence[dict[str, Any]],
    citations: Sequence[Citation],
) -> str:
    """Format graded parent chunks inside sanitized `<documents><document index="N">` tags."""
    enriched: list[dict[str, Any]] = []
    for chunk, cit in zip(chunks, citations):
        item = dict(chunk)
        source_tag = format_citation_source(cit)
        parent_body = str(chunk.get("parent_content") or chunk.get("child_text") or "").strip()
        item["parent_content"] = f"[{cit['citation_index']}] {source_tag}\n{parent_body}"
        enriched.append(item)
    return format_documents_xml(enriched)
