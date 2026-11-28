"""Chunk payload models, Anthropic-style contextual prefix formatters, and legend summary builder."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.ingestion.constants import DEFAULT_DOMAIN


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
    dom = (domain or DEFAULT_DOMAIN).capitalize()
    parts.append(f"Domain: {dom}")
    if subdomain and subdomain != DEFAULT_DOMAIN:
        parts.append(f"Subdomain: {subdomain}")
    if epistemic_modality and epistemic_modality != DEFAULT_DOMAIN:
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
    """Format contextualized string for embedding: f'{contextual_prefix}\\n{child_text}'."""
    if contextual_prefix:
        return f"{contextual_prefix}\n{child_text}"
    clean_heading = (heading_path or "Document").strip()
    return f"{filename} > {clean_heading}: {child_text}"


def build_legend_summary(legend_ctx: dict[str, Any]) -> str:
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


_build_legend_summary = build_legend_summary
