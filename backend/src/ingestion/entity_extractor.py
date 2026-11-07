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
}

# Role mapping based on entity category
ENTITY_ROLE_MAP: dict[str, str] = {
    "equipment_tag": "physical_spec",
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
    "intl_engineering_standard": "standard",
    "vn_engineering_standard": "standard",
    "intl_accounting_standard": "accounting_standard",
    "vn_accounting_standard": "accounting_regulation",
    "us_statute_citation": "statute",
    "vn_statute_citation": "statute",
    "clause_citation": "clause",
    "gazetteer_entity": "catalog_entity",
}


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
        """Extract all unique entity citations and tags from text in deterministic order."""
        if not text:
            return []

        found: set[str] = set()

        # Deterministic regex matching across all categories
        for cat_name, pattern in ENTITY_PATTERNS.items():
            matches = pattern.findall(text)
            for m in matches:
                clean_m = re.sub(r"\s+", " ", m.strip())
                if clean_m:
                    found.add(clean_m)

        # Gazetteer matching
        if self._gazetteer_re:
            for g_match in self._gazetteer_re.finditer(text):
                found.add(g_match.group(1).strip())

        return sorted(found)

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

        # Find or create CrossDomainEntity
        stmt = select(CrossDomainEntity).where(CrossDomainEntity.canonical_name == clean_name)
        res = await session.execute(stmt)
        entity_obj = res.scalars().first()

        if entity_obj is None:
            # Determine entity type
            entity_type = "general"
            for cat_name, pat in ENTITY_PATTERNS.items():
                if pat.search(clean_name):
                    entity_type = ENTITY_TYPE_MAP.get(cat_name, "general")
                    break

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
            role = "entity_mention"
            for cat_name, pat in ENTITY_PATTERNS.items():
                if pat.search(clean_name):
                    role = ENTITY_ROLE_MAP.get(cat_name, "entity_mention")
                    break

            mapping_obj = EntityDomainMapping(
                entity_id=entity_obj.id,
                chunk_id=chunk_id,
                domain_id=domain_id,
                role_in_domain=role,
            )
            session.add(mapping_obj)
