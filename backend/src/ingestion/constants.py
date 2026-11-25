"""Shared constants, engine guards, and formatting helpers for document ingestion."""
from __future__ import annotations

import os
from collections.abc import Sequence

HIGH_FIDELITY_ENGINES: frozenset[str] = frozenset({"docling", "engineering_cad"})

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".pptx", ".xlsx"})

APPROVAL_MODES: frozenset[str] = frozenset({"review_all", "auto_high_confidence", "auto_all"})

CHUNKING_STRATEGIES: frozenset[str] = frozenset(
    {
        "standard_markdown",
        "academic_equations",
        "financial_tables",
        "statutory_clauses",
    }
)

ENGINEERING_RECORD_TYPES: frozenset[str] = frozenset(
    {
        "schedule_row",
        "matrix_cause_row",
        "matrix_effect_col",
        "cad_logic_diagram",
        "exploded_cad_ocr",
        "document_legend_notes",
    }
)

DEFAULT_DOMAIN: str = "general"
DEFAULT_SUBDOMAIN: str = "general.general"
DEFAULT_EPISTEMIC: str = "general"


def resolve_prefer_pymupdf4llm(prefer_pymupdf4llm: bool | None) -> bool | None:
    """Resolve explicit fast-path override or fall back to PREFER_PYMUPDF4LLM env var."""
    if prefer_pymupdf4llm is not None:
        return prefer_pymupdf4llm
    env_val = os.environ.get("PREFER_PYMUPDF4LLM")
    if env_val is not None and env_val.strip():
        return env_val.lower() in ("1", "true", "yes")
    return None


def is_cache_engine_mismatch(cached_engine: str | None, prefer_fallback: bool | None) -> bool:
    """Return True if a cached checkpoint's parsing engine contradicts the requested mode."""
    return (
        (prefer_fallback is True and cached_engine != "pymupdf4llm")
        or (prefer_fallback is False and cached_engine not in HIGH_FIDELITY_ENGINES)
    )


def is_engine_mode_upgrade(existing_engine: str | None, prefer_fast_path: bool | None) -> bool:
    """Return True when high-fidelity parsing is requested for a document not yet parsed by one."""
    return prefer_fast_path is False and existing_engine not in HIGH_FIDELITY_ENGINES


def normalize_chunking_strategy(
    strategy: str | None,
    domain_default: str = "standard_markdown",
) -> str:
    """Return explicit non-default strategy if set, otherwise fall back to domain default."""
    if strategy and strategy != "standard_markdown":
        return strategy
    return domain_default or "standard_markdown"


def format_markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    escape_pipes: bool = True,
) -> str:
    """Format headers and row cells into a GitHub-flavored Markdown pipe table."""
    def _fmt_cell(val: str) -> str:
        return val.replace("|", "\\|") if escape_pipes else val

    header_cells = [_fmt_cell(str(h)) for h in headers]
    lines = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(["---"] * len(header_cells)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt_cell(str(v)) for v in row) + " |")
    return "\n".join(lines)
