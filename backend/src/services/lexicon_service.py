"""Polysemy lexicon CRUD and cap validation service."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import DomainLexicon
from src.db.repository import (
    add_lexicon_term,
    delete_lexicon_term,
    get_domain_by_slug,
    list_lexicon_terms,
    resolve_domain_by_id_or_slug,
)

if TYPE_CHECKING:
    from src.api.schemas import LexiconCreateRequest, LexiconItemResponse


async def list_lexicon_entries(
    session: AsyncSession,
    *,
    domain_id: str | None = None,
    search: str | None = None,
) -> list[LexiconItemResponse]:
    """List polysemy lexicon entries with optional domain ID/slug filter and search query."""
    from src.api.schemas import LexiconItemResponse

    resolved_domain_id = domain_id
    if domain_id:
        d = await get_domain_by_slug(session, domain_id)
        if d is not None:
            resolved_domain_id = d.id

    entries = await list_lexicon_terms(session, domain_id=resolved_domain_id, search=search)
    return [LexiconItemResponse.model_validate(e) for e in entries]


async def create_lexicon_entry(
    session: AsyncSession,
    payload: LexiconCreateRequest,
) -> LexiconItemResponse:
    """Create or update a lexicon entry while enforcing workspace (500) and domain (100) caps."""
    from src.api.schemas import LexiconItemResponse

    domain = await resolve_domain_by_id_or_slug(session, payload.domain_id)
    if domain is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain '{payload.domain_id}' not found",
        )

    norm_term = payload.term.strip()
    existing_res = await session.execute(
        select(DomainLexicon).where(
            DomainLexicon.term == norm_term,
            DomainLexicon.domain_id == domain.id,
        )
    )
    if existing_res.scalars().first() is None:
        total_count = (
            await session.execute(select(func.count(DomainLexicon.id)))
        ).scalar() or 0
        if total_count >= 500:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum limit of 500 lexicon entries across workspace reached",
            )

        domain_count = (
            await session.execute(
                select(func.count(DomainLexicon.id)).where(DomainLexicon.domain_id == domain.id)
            )
        ).scalar() or 0
        if domain_count >= 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Maximum limit of 100 lexicon entries for domain '{domain.slug}' reached",
            )

    entry = await add_lexicon_term(
        session=session,
        term=norm_term,
        domain_id=domain.id,
        specific_sense=payload.specific_sense,
        contrastive_warning=payload.contrastive_warning,
    )
    return LexiconItemResponse.model_validate(entry)


async def delete_lexicon_entry(session: AsyncSession, entry_id: str) -> dict[str, Any]:
    """Delete a lexicon entry by ID, raising HTTP 404 when absent."""
    success = await delete_lexicon_term(session, lexicon_id=entry_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lexicon entry '{entry_id}' not found",
        )
    return {"deleted": True, "id": entry_id}
