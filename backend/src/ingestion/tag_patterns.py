"""Canonical regex patterns for industrial tags, engineering standards, and statutory citations."""
from __future__ import annotations

import re

# Raw industrial tag regex bodies (shared between entity extraction and CAD record extraction)
EQUIPMENT_TAG_BODY = (
    r"(?:MBA-[A-Z0-9]+|T-\d{2,4}|BESS-[A-Z0-9-]+|GEN-\d{2,3}|SWG-\d{2,3}|TR-\d{2,4}|CB-\d{2,3})"
)
DCS_LOOP_TAG_BODY = (
    r"(?:TI|PI|FI|LI|AI|AO|DI|DO|FIC|PIC|LIC|TIC|PIA|LIA|HIC|TIA|FIA|XV|FCV|PCV|HCV"
    r"|PSHH|PALL|PAHH|LAHH|PDALL|UZA|UZ|ZI|VI|XI)\d{3,6}[A-Z]?"
)
ESD_FGS_SIGNAL_BODY = (
    r"(?:(?:HZ|FZ|GZ|UA|UZ|UZA|FGS|RSESD|LTRCV|LTFCS|MAIN)(?=[-_A-Z0-9]*\d)[-_A-Z0-9]{2,20}"
    r"|(?:GA|FA|MA|TA|HR|BC)-(?:FZ|GZ)\d{3}[-_A-Z0-9]*)"
)
CABINET_IO_ID_BODY = (
    r"(?:LNG-[A-Z0-9-]+|FCS\d{4}|SCS\d{4}|AAI\d{3}-[A-Z]|AAV\d{3}-[A-Z]|ADV\d{3}-[A-Z]"
    r"|A[IO]\.[A-Z]\.[A-Z]|\d[FR]TB\d+|FS\d{2}|RS\d{2}|MCC-(?=[A-Z0-9]*\d)[A-Z0-9]+|CB-\d{2,4})"
)
ISA_INSTRUMENT_TAG_BODY = r"[A-Z]{2,6}-\d{3,6}[A-Z]{0,2}(?:/[A-Z0-9]+)*"

# Compiled industrial tag patterns
EQUIPMENT_TAG_RE = re.compile(rf"\b{EQUIPMENT_TAG_BODY}\b")
DCS_LOOP_TAG_RE = re.compile(rf"\b{DCS_LOOP_TAG_BODY}\b")
ESD_FGS_SIGNAL_RE = re.compile(rf"\b{ESD_FGS_SIGNAL_BODY}\b")
CABINET_IO_ID_RE = re.compile(rf"\b{CABINET_IO_ID_BODY}\b")
ISA_INSTRUMENT_TAG_RE = re.compile(rf"\b{ISA_INSTRUMENT_TAG_BODY}\b")

# Unified CAD tag token pattern for engineering PDF record extraction
CAD_TAG_TOKEN_RE = re.compile(
    rf"\b(?:"
    rf"(?:GA|FA|MA|TA|HR|BC)-(?:FZ|GZ)\d{{3}}[-_A-Z0-9]*"
    rf"|{ISA_INSTRUMENT_TAG_BODY}"
    rf"|{DCS_LOOP_TAG_BODY}"
    rf"|{ESD_FGS_SIGNAL_BODY}"
    rf"|{CABINET_IO_ID_BODY}"
    rf")\b"
)

# Standards, accounting regulations, and statutory citations (entity extraction)
INTL_ENGINEERING_STANDARD_RE = re.compile(
    r"\b(?:IEEE\s+[A-Z0-9.]+(?:-\d{4})?|IEC\s+\d+(?:-\d+)?(?:-\d+)?|ASME\s+[A-Z0-9.-]+|ISO\s+\d+(?:-\d+)?|ASTM\s+[A-Z]\d+)\b",
    re.IGNORECASE,
)
VN_ENGINEERING_STANDARD_RE = re.compile(
    r"\b(?:TCVN\s+\d+(?:-\d+)?(?::\d{4})?|QCVN\s+\d+(?:-\d+)?(?::\d{4})?(?:/[A-Z0-9]+)?)\b",
    re.IGNORECASE,
)
INTL_ACCOUNTING_STANDARD_RE = re.compile(
    r"\b(?:ASC\s+\d+|IFRS\s+\d+|IAS\s+\d+|US\s+GAAP|SOX\s+404)\b",
    re.IGNORECASE,
)
VN_ACCOUNTING_STANDARD_RE = re.compile(
    r"\b(?:VAS\s+\d+|Thông\s+tư\s+\d+/\d{4}/TT-BTC|Quyết\s+định\s+\d+/\d{4}/QĐ-BTC)\b",
    re.IGNORECASE,
)
US_STATUTE_CITATION_RE = re.compile(
    r"\b(?:\d+\s+CFR\s+\d+(?:\.\d+)?|\d+\s+U\.S\.C\.\s+§?\s*\d+(?:\([a-z0-9]+\))*)\b",
    re.IGNORECASE,
)
VN_STATUTE_CITATION_RE = re.compile(
    r"\b(?:Luật\s+số\s+[\d/]+QH[\d]+|Nghị\s+định\s+[\d/]+NĐ-CP|Nghị\s+quyết\s+[\d/]+NQ-CP|Quyết\s+định\s+[\d/]+QĐ-TTg)\b",
    re.IGNORECASE,
)
CLAUSE_CITATION_RE = re.compile(
    r"\b(?:Điều\s+\d+\s+Khoản\s+\d+|Section\s+\d+\s+Subsection\s+\([a-z0-9]+\))\b",
    re.IGNORECASE,
)

# Authority/jurisdiction heuristic patterns (classifier heuristics)
VN_LAW_CITATION_RE = re.compile(r"Luật\s+số\s+[\d/]+QH[\d]+", re.IGNORECASE)
VN_LAW_YEAR_RE = re.compile(r"QH(\d{2})")
VN_DECREE_CITATION_RE = re.compile(r"Nghị\s+định\s+[\d/]+NĐ-CP", re.IGNORECASE)
VN_DECREE_YEAR_RE = re.compile(r"(\d{4})/NĐ-CP")
VN_CIRCULAR_CITATION_RE = re.compile(r"Thông\s+tư\s+[\d/]+TT-BTC", re.IGNORECASE)
VN_CIRCULAR_YEAR_RE = re.compile(r"(\d{4})/TT-BTC")
TCVN_PREFIX_RE = re.compile(r"TCVN\s+[\d\-:]+", re.IGNORECASE)
QCVN_PREFIX_RE = re.compile(r"QCVN\s+[\d\-:]+", re.IGNORECASE)
COLON_YEAR_RE = re.compile(r":(\d{4})")
VAS_PREFIX_RE = re.compile(r"\bVAS\s+\d+", re.IGNORECASE)
US_CODE_PREFIX_RE = re.compile(r"\b\d+\s+U\.S\.C\.", re.IGNORECASE)
US_CFR_PREFIX_RE = re.compile(r"\b\d+\s+CFR\s+\d+", re.IGNORECASE)
IEEE_PREFIX_RE = re.compile(r"\bIEEE\s+[A-Z0-9.]+", re.IGNORECASE)
IEEE_YEAR_RE = re.compile(r"IEEE\s+[A-Z0-9.]+(?:-(\d{4}))?", re.IGNORECASE)
IEC_PREFIX_RE = re.compile(r"\bIEC\s+\d+", re.IGNORECASE)
ASME_PREFIX_RE = re.compile(r"\bASME\s+[A-Z0-9.]+", re.IGNORECASE)
IFRS_IAS_PREFIX_RE = re.compile(r"\b(?:IFRS|IAS)\s+\d+", re.IGNORECASE)
US_GAAP_ASC_PREFIX_RE = re.compile(r"\b(?:US\s+GAAP|ASC\s+\d+)", re.IGNORECASE)
FORM_10K_RE = re.compile(r"\bForm\s+10-K\b", re.IGNORECASE)
FORM_10K_YEAR_RE = re.compile(r"Ended\s+[A-Za-z]+\s+\d{1,2},\s+(\d{4})", re.IGNORECASE)
YEAR_4DIGIT_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
