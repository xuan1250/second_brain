"""Entity extraction combining catalog gazetteer matching and deterministic regex extraction for standards, tags, and citations."""
from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import CrossDomainEntity, DomainLexicon, EntityDomainMapping

logger = logging.getLogger(__name__)

# Deterministic regex patterns for multi-domain entities
ENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
    # Engineering equipment tags
    "equipment_tag": re.compile(r"\b(?:MBA-[A-Z0-9]+|T-\d{2,4}|BESS-[A-Z0-9-]+|GEN-\d{2,3}|SWG-\d{2,3}|TR-\d{2,4}|CB-\d{2,3})\b"),
    # International engineering standards
    "intl_engineering_standard": re.compile(
        r"\b(?:IEEE\s+[A-Z0-9.]+(?:-\d{4})?|IEC\s+\d+(?:-\d+)?(?:-\d+)?|ASME\s+[A-Z0-9.-]+|ISO\s+\d+(?:-\d+)?|ASTM\s+[A-Z]\d+)\b",
        re.IGNORECASE,
    ),
    # Vietnamese engineering standards
    "vn_engineering_standard": re.compile(
        r"\b(?:TCVN\s+\d+(?:-\d+)?(?::\d{4})?|QCVN\s+\d+(?:-\d+)?(?::\d{4})?(?:/[A-Z0-9]+)?)\b",
        re.IGNORECASE,
    ),
    # International accounting & financial standards
    "intl_accounting_standard": re.compile(
        r"\b(?:ASC\s+\d+|IFRS\s+\d+|IAS\s+\d+|US\s+GAAP|SOX\s+404)\b",
        re.IGNORECASE,
    ),
    # Vietnamese accounting & financial regulations
    "vn_accounting_standard": re.compile(
        r"\b(?:VAS\s+\d+|Thông\s+tư\s+\d+/\d{4}/TT-BTC|Quyết\s+định\s+\d+/\d{4}/QĐ-BTC)\b",
        re.IGNORECASE,
    ),
    # US & international statutory citations
    "us_statute_citation": re.compile(
        r"\b(?:\d+\s+CFR\s+\d+(?:\.\d+)?|\d+\s+U\.S\.C\.\s+§?\s*\d+(?:\([a-z0-9]+\))*)\b",
        re.IGNORECASE,
    ),
    # Vietnamese statutory & decree citations
    "vn_statute_citation": re.compile(
        r"\b(?:Luật\s+số\s+[\d/]+QH[\d]+|Nghị\s+định\s+[\d/]+NĐ-CP|Nghị\s+quyết\s+[\d/]+NQ-CP|Quyết\s+định\s+[\d/]+QĐ-TTg)\b",
        re.IGNORECASE,
    ),
    # Statutory clause hierarchy (EN & VI)
    "clause_citation": re.compile(
        r"\b(?:Điều\s+\d+\s+Khoản\s+\d+|Section\s+\d+\s+Subsection\s+\([a-z0-9]+\))\b",
        re.IGNORECASE,
    ),
    # Universal industrial tags (case-sensitive uppercase to avoid matching prose words)
    "dcs_loop_tag": re.compile(
        r"\b(?:TI|PI|FI|LI|AI|AO|DI|DO|FIC|PIC|LIC|TIC|PIA|LIA|HIC|TIA|FIA|XV|FCV|PCV|HCV|PSHH|PALL|PAHH|LAHH|PDALL|UZA|UZ|ZI|VI|XI)\d{3,6}[A-Z]?\b"
    ),
    "esd_fgs_signal": re.compile(
        r"\b(?:(?:HZ|FZ|GZ|UA|UZ|UZA|FGS|RSESD|LTRCV|LTFCS|MAIN)(?=[-_A-Z0-9]*\d)[-_A-Z0-9]{2,20}|(?:GA|FA|MA|TA|HR|BC)-(?:FZ|GZ)\d{3}[-_A-Z0-9]*)\b"
    ),
    "cabinet_io_id": re.compile(
        r"\b(?:LNG-[A-Z0-9-]+|FCS\d{4}|SCS\d{4}|AAI\d{3}-[A-Z]|AAV\d{3}-[A-Z]|ADV\d{3}-[A-Z]|A[IO]\.[A-Z]\.[A-Z]|\d[FR]TB\d+|FS\d{2}|RS\d{2}|MCC-\d{3}[A-Z]?|CB-\d{2,4})\b"
    ),
    "isa_instrument_tag": re.compile(
        r"\b[A-Z]{2,6}-\d{3,6}[A-Z]{0,2}(?:/[A-Z0-9]+)*\b"
    ),
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

        # 1. Exact normalized uppercase tag
        _append_unique(per_tag, local_seen, norm)

        # 2. Strip trailing _R\d+ or _\d{2} logic suffix
        base = _LOGIC_SUFFIX_RE.sub("", norm) or norm

        # 5. Split multi-channel voting tags before capping (priority after original)
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

        # 3. Stripped base & compact alphanumeric stem
        if base != norm:
            _append_unique(per_tag, local_seen, base)
        if not voting_match:
            compact = re.sub(r"[-_]", "", base)
            if compact != base:
                _append_unique(per_tag, local_seen, compact)

        # 4. Hyphenated form for compact prefix+digits tags
        compact_match = _COMPACT_TAG_RE.match(base)
        if compact_match:
            prefix, rest = compact_match.groups()
            if prefix in _HYPHENATE_PREFIXES:
                _append_unique(per_tag, local_seen, f"{prefix}-{rest}")

        # 4b. Cross-link zone-scoped composite tags (e.g. GA-FZ001 -> FZ001, FZ-001)
        zone_match = _EMBEDDED_ZONE_RE.match(base)
        if zone_match:
            z_prefix, z_digits = zone_match.groups()
            _append_unique(per_tag, local_seen, f"{z_prefix}{z_digits}")
            _append_unique(per_tag, local_seen, f"{z_prefix}-{z_digits}")

        # 6. Cross-link ISA loop number stems
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
    """Extracts domain entities, standards, tags, and citations using regexes and SQLite gazetteer."""

    def __init__(self, gazetteer_terms: Optional[list[str]] = None) -> None:
        self.gazetteer_terms = gazetteer_terms or []
        self._gazetteer_re: Optional[re.Pattern[str]] = None
        if self.gazetteer_terms:
            pattern_str = r"\b(" + "|".join(re.escape(t) for t in self.gazetteer_terms if t.strip()) + r")\b"
            self._gazetteer_re = re.compile(pattern_str, re.IGNORECASE)

    @classmethod
    async def load_from_db(cls, session: AsyncSession) -> EntityExtractor:
        """Instantiate EntityExtractor with gazetteer terms from SQLite domain_lexicon and cross_domain_entities."""
        terms: set[str] = set()
        try:
            res_entities = await session.execute(select(CrossDomainEntity.canonical_name))
            for name in res_entities.scalars():
                if name and len(name.strip()) > 2:
                    terms.add(name.strip())

            res_lex = await session.execute(select(DomainLexicon.term))
            for term in res_lex.scalars():
                if term and len(term.strip()) > 3:
                    terms.add(term.strip())
        except Exception as exc:
            logger.debug("Failed loading gazetteer from DB: %s", exc)

        return cls(gazetteer_terms=sorted(terms))

    def extract_entities(self, text: str, domain_key: Optional[str] = None) -> list[str]:
        """Extract all unique entity citations, tags, and canonical stems from text in deterministic order."""
        if not text:
            return []

        found: set[str] = set()
        industrial_tags: list[str] = []

        # Deterministic regex matching across all categories
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

        # Gazetteer matching
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

        # Uppercase only tokens containing a digit so lowercase tag queries still match case-sensitive patterns
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

    def extract_typed_entities(self, text: str) -> list[dict[str, str]]:
        """Extract entities with category and default role metadata."""
        if not text:
            return []

        results: list[dict[str, str]] = []
        seen: set[str] = set()

        for cat_name, pattern in ENTITY_PATTERNS.items():
            for m in pattern.findall(text):
                clean_m = re.sub(r"\s+", " ", m.strip())
                if clean_m and clean_m.lower() not in seen:
                    seen.add(clean_m.lower())
                    results.append(
                        {
                            "name": clean_m,
                            "category": cat_name,
                            "entity_type": ENTITY_TYPE_MAP.get(cat_name, "general"),
                            "role_in_domain": ENTITY_ROLE_MAP.get(cat_name, "spec"),
                        }
                    )

        if self._gazetteer_re:
            for g_match in self._gazetteer_re.finditer(text):
                matched_name = g_match.group(1).strip()
                if matched_name.lower() not in seen:
                    seen.add(matched_name.lower())
                    results.append(
                        {
                            "name": matched_name,
                            "category": "gazetteer_entity",
                            "entity_type": "catalog_entity",
                            "role_in_domain": "catalog_ref",
                        }
                    )

        return results


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

        # Find or create CrossDomainEntity
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

        # Check existing mapping
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
