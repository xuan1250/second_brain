"""Polysemic domain lexicon CRUD and text-matching repository helpers."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import DomainLexicon
from src.db.repositories.common import compile_word_boundary_regex


async def add_lexicon_term(
    session: AsyncSession,
    term: str,
    domain_id: str,
    specific_sense: str,
    contrastive_warning: str,
) -> DomainLexicon:
    """Add or update a polysemic term definition in the domain lexicon."""
    norm_term = term.strip()
    stmt = select(DomainLexicon).where(
        DomainLexicon.term == norm_term,
        DomainLexicon.domain_id == domain_id,
    )
    res = await session.execute(stmt)
    existing = res.scalars().first()
    if existing is not None:
        existing.specific_sense = specific_sense
        existing.contrastive_warning = contrastive_warning
        session.add(existing)
        await session.commit()
        await session.refresh(existing)
        return existing

    entry = DomainLexicon(
        term=norm_term,
        domain_id=domain_id,
        specific_sense=specific_sense,
        contrastive_warning=contrastive_warning,
    )
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    return entry


async def list_lexicon_terms(
    session: AsyncSession,
    domain_id: str | None = None,
    search: str | None = None,
) -> list[DomainLexicon]:
    """List lexicon terms with optional domain filter and search term."""
    stmt = select(DomainLexicon)
    if domain_id:
        stmt = stmt.where(DomainLexicon.domain_id == domain_id)
    if search and search.strip():
        search_pattern = f"%{search.strip().lower()}%"
        stmt = stmt.where(
            (DomainLexicon.term.ilike(search_pattern))
            | (DomainLexicon.specific_sense.ilike(search_pattern))
            | (DomainLexicon.contrastive_warning.ilike(search_pattern))
        )
    stmt = stmt.order_by(DomainLexicon.term)
    res = await session.execute(stmt)
    return list(res.scalars().all())


async def delete_lexicon_term(session: AsyncSession, lexicon_id: str) -> bool:
    """Delete a domain lexicon entry by id."""
    item = await session.get(DomainLexicon, lexicon_id)
    if item is None:
        return False
    await session.delete(item)
    await session.commit()
    return True


async def get_lexicon_matches(session: AsyncSession, text: str) -> list[DomainLexicon]:
    """Match polysemic terms in input text across English and Vietnamese multi-word phrases."""
    if not text or not text.strip():
        return []

    stmt = select(DomainLexicon)
    res = await session.execute(stmt)
    all_entries = list(res.scalars().all())
    if not all_entries:
        return []

    term_to_entries: dict[str, list[DomainLexicon]] = {}
    for entry in all_entries:
        key = entry.term.strip().lower()
        term_to_entries.setdefault(key, []).append(entry)

    matches: list[DomainLexicon] = []
    for term_lower, entries in term_to_entries.items():
        if compile_word_boundary_regex(term_lower).search(text):
            matches.extend(entries)

    return matches
