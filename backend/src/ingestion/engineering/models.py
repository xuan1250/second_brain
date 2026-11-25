"""Data models for per-page engineering PDF extraction."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

ACTION_MARKS: frozenset[str] = frozenset({"X", "O", "R", "P", "B"})


@dataclass
class LegendContext:
    """Document-level action symbol legend and numbered engineering notes."""

    symbol_map: dict[str, str] = field(default_factory=dict)
    notes_map: dict[str, str] = field(default_factory=dict)
    source_pages: list[int] = field(default_factory=list)
    raw_text: str = ""

    def format_summary(self) -> str:
        """Return a concise human-readable summary of symbols and notes."""
        parts: list[str] = []
        if self.symbol_map:
            syms = "; ".join(f"{k} = {v}" for k, v in self.symbol_map.items())
            parts.append(f"Legend Symbols: {syms}")
        if self.notes_map:
            notes = " | ".join(f"{k}: {v}" for k, v in self.notes_map.items())
            parts.append(f"Notes: {notes}")
        return "\n".join(parts)

    def lookup_notes(self, refs: Sequence[str]) -> list[str]:
        """Resolve note references (e.g. 'NOTE 1', '36') into full note strings."""
        resolved: list[str] = []
        seen: set[str] = set()
        for ref in refs:
            if not ref:
                continue
            raw = str(ref).strip()
            match = re.search(r"(?:NOTE\s*[-#:]?\s*)?(\d+)", raw, re.IGNORECASE)
            candidates = [raw.upper()]
            if match:
                candidates.append(f"NOTE {match.group(1)}")
            for cand in candidates:
                if cand in self.notes_map and cand not in seen:
                    seen.add(cand)
                    resolved.append(f"{cand}: {self.notes_map[cand]}")
        return resolved


@dataclass
class EngineeringPageRecord:
    """Atomic structured record extracted from a CAD engineering PDF page."""

    record_id: str
    record_type: str
    page_number: int
    title: str
    summary_text: str
    tags: list[str] = field(default_factory=list)
    bbox: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageExtractionResult:
    """Complete per-page engineering extraction payload."""

    page_number: int
    mode: str
    metrics: dict[str, Any]
    records: list[EngineeringPageRecord]
    markdown: str
    legend_context: LegendContext | None = None
