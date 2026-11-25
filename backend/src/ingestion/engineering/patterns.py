"""CAD regexes, header cue sets, and tag extraction helpers for engineering PDFs."""
from __future__ import annotations

import re

from src.ingestion.tag_patterns import CAD_TAG_TOKEN_RE

_TAG_TOKEN_RE = CAD_TAG_TOKEN_RE

_PID_RE = re.compile(r"\bPID[-_]?\d{2,6}\b", re.IGNORECASE)
_PID_CELL_RE = re.compile(r"\b(?:[A-Z0-9]+-)*PID[-_]?\d{2,6}(?:[,_]\d+(?:/\d+)?)?\b", re.IGNORECASE)
_NOTE_REF_RE = re.compile(r"\bNOTE\s*[-#:]?\s*(\d+)\b", re.IGNORECASE)
_SYMBOL_DEF_LINE_RE = re.compile(r"^\s*([XOPRB])\s*[:=]\s*([A-Za-z0-9].+?)\s*$")
_SYMBOL_DEF_INLINE_RE = re.compile(r"(?:^|\n)\s*([XOPRB])\s*[:=]\s*[A-Za-z][^\n]*")
_NOTE_DEF_START_RE = re.compile(r"^\s*(NOTE\s*[-#:]?\s*(\d+))\s*[:.\-]\s*(.+?)\s*$", re.IGNORECASE)
_NOTE_SECTION_RE = re.compile(r"^\s*NOTES?\.?:?\s*$", re.IGNORECASE)
_NOTE_NUM_COLON_RE = re.compile(r"^\s*(\d{1,3})\s*:\s*(.+?)\s*$")
_PAGE_NUM_FOOTER_RE = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")
_VOTING_RE = re.compile(
    r"(?:\b(?:\d+\s*out\s*of\s*\d+|\d+\s*oo\s*[\dN]+)\b|(?:≥|>=)\s*\d+)",
    re.IGNORECASE,
)
_SIGNAL_LEVEL_RE = re.compile(r"^(?:HH|LL|H|L|ON|OFF)$", re.IGNORECASE)
_GROUP_CODE_RE = re.compile(r"^(?:ESD[-_]?\d+|FGS|PSD|SIS|BMS|HVAC)$", re.IGNORECASE)
_FIRE_ZONE_GROUP_RE = re.compile(r"^(?:FZ|GZ)-?\d{3}\b", re.IGNORECASE)
_LOGIC_GATE_RE = re.compile(r"\b(?:AND|OR|RS|SR|TON|TOF|SET|RESET|NOT|XOR)\b")
_POU_TITLE_RE = re.compile(
    r"\b\d+\.\d+\.\s*[A-Z][A-Z0-9_]+(?:\s*\(\*\s*[^*]+\s*\*\))?",
    re.IGNORECASE,
)
_CROSS_REF_RE = re.compile(r"\(\*\s*([^*\n]+?)\s*\*\)")

_CE_HEADER_LABEL_TOKENS = frozenset(
    {
        "EFFECT",
        "REV",
        "P&ID NUMBER",
        "P&ID NO",
        "P&ID",
        "NOTE",
        "NOTES",
        "FUNCTION",
        "PERFORMED",
        "FUNCTION PERFORMED",
        "TAG NAME",
        "TAG NO",
        "DEVICE ID",
        "CONTROL DEVICE",
        "SHUTDOWN OR",
        "ID",
        "ACTION",
        "CAUSE",
    }
)

_CAUSE_TABLE_HEADER_CUES = frozenset(
    {
        "CAUSE DESCRIPTION",
        "CAUSE TAG",
        "TAG NAME",
        "TAG NO",
        "FIRE ZONE NO",
        "LOGIC",
        "VOTING",
        "GROUP",
    }
)

_EFFECT_ACTION_KEYWORDS = (
    "close",
    "open",
    "trip",
    "stop",
    "start",
    "alarm",
    "active",
    "activate",
    "activation",
    "shutdown",
    "isolate",
    "inhibit",
    "energize",
    "de-energize",
    "reset",
)

_SCHEDULE_HEADER_KEYWORDS = frozenset(
    {
        "ITEM",
        "TAG",
        "FIELD",
        "DCS",
        "TERMINAL",
        "BLOCK",
        "TB",
        "FCS",
        "NODE",
        "IO",
        "MODULE",
        "CHANNEL",
        "SERVICE",
        "DESCRIPTION",
        "REMARKS",
        "SIGNAL",
        "LOOP",
        "RANGE",
        "UNIT",
        "CABINET",
    }
)


def _extract_record_tags(text: str) -> list[str]:
    """Extract deduplicated industrial tags and identifiers from text preserving order."""
    seen: set[str] = set()
    tags: list[str] = []
    for match in _TAG_TOKEN_RE.finditer(text or ""):
        token = match.group(0).strip()
        if token and token not in seen:
            seen.add(token)
            tags.append(token)
    for match in _NOTE_REF_RE.finditer(text or ""):
        note_key = f"NOTE {match.group(1)}"
        if note_key not in seen:
            seen.add(note_key)
            tags.append(note_key)
    return tags
