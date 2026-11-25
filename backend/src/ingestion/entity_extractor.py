"""Entity extraction combining catalog gazetteer matching and deterministic regex extraction for standards, tags, and citations."""
from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import CrossDomainEntity, EntityDomainMapping
from src.ingestion.tag_patterns import (
    CABINET_IO_ID_RE,
    CLAUSE_CITATION_RE,
    DCS_LOOP_TAG_RE,
    EQUIPMENT_TAG_RE,
    ESD_FGS_SIGNAL_RE,
    INTL_ACCOUNTING_STANDARD_RE,
    INTL_ENGINEERING_STANDARD_RE,
    ISA_INSTRUMENT_TAG_RE,
    US_STATUTE_CITATION_RE,
    VN_ACCOUNTING_STANDARD_RE,
    VN_ENGINEERING_STANDARD_RE,
    VN_STATUTE_CITATION_RE,
)

logger = logging.getLogger(__name__)

# Deterministic regex patterns for multi-domain entities
ENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "equipment_tag": EQUIPMENT_TAG_RE,
    "intl_engineering_standard": INTL_ENGINEERING_STANDARD_RE,
    "vn_engineering_standard": VN_ENGINEERING_STANDARD_RE,
    "intl_accounting_standard": INTL_ACCOUNTING_STANDARD_RE,
    "vn_accounting_standard": VN_ACCOUNTING_STANDARD_RE,
    "us_statute_citation": US_STATUTE_CITATION_RE,
    "vn_statute_citation": VN_STATUTE_CITATION_RE,
    "clause_citation": CLAUSE_CITATION_RE,
    "dcs_loop_tag": DCS_LOOP_TAG_RE,
    "esd_fgs_signal": ESD_FGS_SIGNAL_RE,
    "cabinet_io_id": CABINET_IO_ID_RE,
    "isa_instrument_tag": ISA_INSTRUMENT_TAG_RE,
}

INDUSTRIAL_TAG_CATEGORIES: frozenset[str] = frozenset(
    {
        "equipment_tag",
        "isa_instrument_tag",
        "dcs_loop_tag",
        "esd_fgs_signal",
        "cabinet_io_id",
    }
)

_LOGIC_SUFFIX_RE = re.compile(r"(?:_R\d+|_\d{2})$")
_COMPACT_TAG_RE = re.compile(r"^([A-Z]{2,6})(\d{3,6}[A-Z]?)$")
_HYPHENATE_PREFIXES: frozenset[str] = frozenset({"HZ", "FZ", "GZ", "UA", "UZ", "XV", "TT", "PT", "FT", "LT"})
_EMBEDDED_ZONE_RE = re.compile(r"^[A-Z]{2,4}-(FZ|GZ)(\d{3})(?:[-_A-Z0-9]*)$")
_VOTING_CHANNEL_RE = re.compile(r"^([A-Z]{2,6}-\d{3,6})(?!\d)([A-Z0-9]+(?:/[A-Z0-9]+)+)$")
_ISA_LOOP_PREFIXES: frozenset[str] = frozenset(
    {
        "TT", "TI", "TIC", "TIA", "TE", "TX",
        "FT", "FI", "FIC", "FIA", "FE",
        "PT", "PI", "PIC", "PIA", "PE",
        "LT", "LI", "LIC", "LIA", "LE",
        "ZT", "ZI",
        "VT", "VI",
        "XT", "XI",
        "AT", "AI", "AIC",
        "PD", "PDT", "PDI",
        "XV", "FCV", "PCV", "HCV",
    }
)
_ISA_LOOP_RE = re.compile(r"^([A-Z]{2,4})-?(\d{4,6})([A-Z]?)$")
_LOOP_STEM_RE = re.compile(r"^LOOP-\d{4,6}[A-Z]?$")

# Role mapping based on entity category
ENTITY_ROLE_MAP: dict[str, str] = {
    "equipment_tag": "physical_spec",
    "isa_instrument_tag": "physical_spec",
    "dcs_loop_tag": "control_loop_ref",
    "esd_fgs_signal": "safety_interlock_ref",
    "cabinet_io_id": "cabinet_io_ref",
    "intl_engineering_standard": "engineering_standard_ref",
    "vn_engineering_standard": "vn_engineering_standard_ref",
    "intl_accounting_standard": "accounting_standard_ref",
    "vn_accounting_standard": "vn_accounting_regulation_ref",
    "us_statute_citation": "statutory_citation",
    "vn_statute_citation": "vn_statutory_citation",
    "clause_citation": "legal_clause_ref",
    "gazetteer_entity": "catalog_entity",
}

ENTITY_TYPE_MAP: dict[str, str] = {
    "equipment_tag": "equipment",
    "isa_instrument_tag": "instrument",
    "dcs_loop_tag": "dcs_tag",
    "esd_fgs_signal": "safety_signal",
    "cabinet_io_id": "hardware_io",
    "intl_engineering_standard": "standard",
    "vn_engineering_standard": "standard",
    "intl_accounting_standard": "accounting_standard",
    "vn_accounting_standard": "accounting_regulation",
    "us_statute_citation": "statute",
    "vn_statute_citation": "statute",
    "clause_citation": "clause",
    "gazetteer_entity": "catalog_entity",
}


def _append_unique(target: list[str], seen: set[str], value: str) -> None:
    if value and value not in seen:
        seen.add(value)
        target.append(value)


def expand_engineering_tag_stems(
    tags: Sequence[str],
    *,
    max_stems_per_tag: int = 8,
) -> list[str]:
    """Expand industrial tags into normalized stems, split voting channels, and ISA loop links."""
    if not tags or max_stems_per_tag <= 0:
        return []

    all_stems: set[str] = set()
    for raw_tag in tags:
        norm = re.sub(r"\s+", " ", raw_tag.strip()).upper()
        if not norm:
            continue

        per_tag: list[str] = []
        local_seen: set[str] = set()

        _append_unique(per_tag, local_seen, norm)
        base = _LOGIC_SUFFIX_RE.sub("", norm) or norm

        split_channels: list[str] = []
        voting_match = _VOTING_CHANNEL_RE.match(base)
        if voting_match:
            stem_prefix, channels_part = voting_match.groups()
            for ch in channels_part.split("/"):
                if ch:
                    ch_tag = f"{stem_prefix}{ch}"
                    split_channels.append(ch_tag)
                    _append_unique(per_tag, local_seen, ch_tag)
            _append_unique(per_tag, local_seen, stem_prefix)
            _append_unique(per_tag, local_seen, re.sub(r"[-_]", "", stem_prefix))

        if base != norm:
            _append_unique(per_tag, local_seen, base)
        if not voting_match:
            compact = re.sub(r"[-_]", "", base)
            if compact != base:
                _append_unique(per_tag, local_seen, compact)

        compact_match = _COMPACT_TAG_RE.match(base)
        if compact_match:
            prefix, rest = compact_match.groups()
            if prefix in _HYPHENATE_PREFIXES:
                _append_unique(per_tag, local_seen, f"{prefix}-{rest}")

        zone_match = _EMBEDDED_ZONE_RE.match(base)
        if zone_match:
            z_prefix, z_digits = zone_match.groups()
            _append_unique(per_tag, local_seen, f"{z_prefix}{z_digits}")
            _append_unique(per_tag, local_seen, f"{z_prefix}-{z_digits}")

        for candidate in (base, *split_channels):
            loop_match = _ISA_LOOP_RE.match(candidate)
            if loop_match:
                prefix, digits, channel = loop_match.groups()
                if prefix in _ISA_LOOP_PREFIXES:
                    _append_unique(per_tag, local_seen, f"LOOP-{digits}{channel}")
                    if channel:
                        _append_unique(per_tag, local_seen, f"LOOP-{digits}")

        all_stems.update(per_tag[:max_stems_per_tag])

    return sorted(all_stems)


def _classify_entity_name(clean_name: str) -> tuple[str, str]:
    """Resolve (entity_type, role_in_domain) for a canonical entity or stemmed tag."""
    if _LOOP_STEM_RE.match(clean_name):
        return "control_loop", "control_loop_ref"
    for cat_name, pat in ENTITY_PATTERNS.items():
        if pat.search(clean_name):
            return (
                ENTITY_TYPE_MAP.get(cat_name, "general"),
                ENTITY_ROLE_MAP.get(cat_name, "entity_mention"),
            )
    return "general", "entity_mention"


class EntityExtractor:
    """Extracts domain entities, standards, tags, and citations using regexes and optional gazetteer."""

    def __init__(self, gazetteer_terms: Optional[list[str]] = None) -> None:
        self.gazetteer_terms = gazetteer_terms or []
        self._gazetteer_re: Optional[re.Pattern[str]] = None
        if self.gazetteer_terms:
            pattern_str = r"\b(" + "|".join(re.escape(t) for t in self.gazetteer_terms if t.strip()) + r")\b"
            self._gazetteer_re = re.compile(pattern_str, re.IGNORECASE)

    def extract_entities(self, text: str, domain_key: Optional[str] = None) -> list[str]:
        """Extract all unique entity citations, tags, and canonical stems from text in deterministic order."""
        del domain_key
        if not text:
            return []

        found: set[str] = set()
        industrial_tags: list[str] = []

        for cat_name, pattern in ENTITY_PATTERNS.items():
            matches = pattern.findall(text)
            for m in matches:
                clean_m = re.sub(r"\s+", " ", m.strip())
                if not clean_m:
                    continue
                if cat_name in INDUSTRIAL_TAG_CATEGORIES:
                    industrial_tags.append(clean_m)
                else:
                    found.add(clean_m)

        if industrial_tags:
            found.update(expand_engineering_tag_stems(industrial_tags))

        if self._gazetteer_re:
            for g_match in self._gazetteer_re.finditer(text):
                found.add(g_match.group(1).strip())

        return sorted(found)

    def extract_query_tags_and_stems(self, query: str) -> list[str]:
        """Extract industrial tags and expanded stems from a user query without matching prose."""
        if not query:
            return []

        raw_tags: list[str] = []
        seen_raw: set[str] = set()
        candidate_texts = [query]

        upper_digit_query = " ".join(
            tok.upper() if any(ch.isdigit() for ch in tok) else tok
            for tok in query.split()
        )
        if upper_digit_query != query:
            candidate_texts.append(upper_digit_query)

        for candidate_text in candidate_texts:
            for cat_name in INDUSTRIAL_TAG_CATEGORIES:
                pattern = ENTITY_PATTERNS[cat_name]
                for m in pattern.findall(candidate_text):
                    clean_m = re.sub(r"\s+", " ", m.strip())
                    if clean_m and clean_m not in seen_raw:
                        seen_raw.add(clean_m)
                        raw_tags.append(clean_m)

        return expand_engineering_tag_stems(raw_tags)


async def sync_chunk_entities_to_db(
    session: AsyncSession,
    *,
    chunk_id: str,
    domain_id: Optional[str],
    entities: Sequence[str],
) -> None:
    """Synchronize extracted entities to cross_domain_entities and entity_domain_mappings in SQLite."""
    if not entities or not domain_id:
        return

    for ent_name in entities:
        if not ent_name or len(ent_name.strip()) < 2:
            continue
        clean_name = ent_name.strip()
        entity_type, role = _classify_entity_name(clean_name)

        stmt = select(CrossDomainEntity).where(CrossDomainEntity.canonical_name == clean_name)
        res = await session.execute(stmt)
        entity_obj = res.scalars().first()

        if entity_obj is None:
            entity_obj = CrossDomainEntity(
                canonical_name=clean_name,
                entity_type=entity_type,
            )
            session.add(entity_obj)
            await session.flush()

        map_stmt = select(EntityDomainMapping).where(
            EntityDomainMapping.entity_id == entity_obj.id,
            EntityDomainMapping.chunk_id == chunk_id,
            EntityDomainMapping.domain_id == domain_id,
        )
        map_res = await session.execute(map_stmt)
        mapping_obj = map_res.scalars().first()

        if mapping_obj is None:
            mapping_obj = EntityDomainMapping(
                entity_id=entity_obj.id,
                chunk_id=chunk_id,
                domain_id=domain_id,
                role_in_domain=role,
            )
            session.add(mapping_obj)
