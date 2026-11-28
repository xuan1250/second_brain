"""Cross-domain entity bridge repository helpers."""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import CrossDomainEntity, _uuid
from src.db.repositories.common import compile_word_boundary_regex


async def upsert_cross_domain_entities(
    session: AsyncSession,
    entities: list[CrossDomainEntity | dict[str, Any]],
) -> list[CrossDomainEntity]:
    """Upsert cross-domain knowledge graph entities by canonical_name."""
    persisted: list[CrossDomainEntity] = []
    for item in entities:
        if isinstance(item, dict):
            name = item["canonical_name"]
            entity_type = item.get("entity_type", "equipment")
            description = item.get("description")
            id_val = item.get("id")
        else:
            name = item.canonical_name
            entity_type = item.entity_type
            description = item.description
            id_val = item.id

        stmt = select(CrossDomainEntity).where(CrossDomainEntity.canonical_name == name)
        res = await session.execute(stmt)
        existing = res.scalars().first()
        if existing is not None:
            existing.entity_type = entity_type
            if description is not None:
                existing.description = description
            session.add(existing)
            persisted.append(existing)
        else:
            new_ent = CrossDomainEntity(
                id=id_val or _uuid(),
                canonical_name=name,
                entity_type=entity_type,
                description=description,
            )
            session.add(new_ent)
            persisted.append(new_ent)

    await session.commit()
    for ent in persisted:
        await session.refresh(ent)
    return persisted


async def get_cross_domain_entity_matches(
    session: AsyncSession,
    text: str,
) -> list[CrossDomainEntity]:
    """Scan input text for canonical names of registered CrossDomainEntities."""
    if not text or not text.strip():
        return []

    stmt = select(CrossDomainEntity)
    res = await session.execute(stmt)
    all_entities = list(res.scalars().all())
    if not all_entities:
        return []

    matches: list[CrossDomainEntity] = []
    for ent in all_entities:
        name = ent.canonical_name.strip()
        if not name:
            continue
        if compile_word_boundary_regex(name).search(text):
            matches.append(ent)
    return matches
