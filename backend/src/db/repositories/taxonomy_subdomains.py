"""Subdomain registry CRUD with cross-store Qdrant payload synchronization."""
from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    Document,
    DocumentChunk,
    SubdomainRegistry,
    _now,
)
from src.db.repositories.common import rewrite_classification_json_for_reassignment


async def list_subdomains(session: AsyncSession, domain_id: str | None = None) -> list[SubdomainRegistry]:
    """List subdomains, optionally filtered by domain_id."""
    stmt = select(SubdomainRegistry)
    if domain_id:
        stmt = stmt.where(SubdomainRegistry.domain_id == domain_id)
    stmt = stmt.order_by(SubdomainRegistry.slug)
    res = await session.execute(stmt)
    return list(res.scalars().all())


async def list_subdomains_by_domain_ids(
    session: AsyncSession,
    domain_ids: Sequence[str],
) -> dict[str, list[SubdomainRegistry]]:
    """Batch-fetch subdomains grouped by domain_id, ordered by slug."""
    ids = list(dict.fromkeys(domain_ids))
    if not ids:
        return {}
    stmt = (
        select(SubdomainRegistry)
        .where(SubdomainRegistry.domain_id.in_(ids))
        .order_by(SubdomainRegistry.slug)
    )
    res = await session.execute(stmt)
    grouped: dict[str, list[SubdomainRegistry]] = {did: [] for did in ids}
    for sub in res.scalars().all():
        grouped.setdefault(sub.domain_id, []).append(sub)
    return grouped


async def create_subdomain(
    session: AsyncSession,
    slug: str,
    name: str,
    domain_id: str,
    standard_bodies: str | None = None,
    id: str | None = None,
) -> SubdomainRegistry:
    """Create or return an existing subdomain entry."""
    stmt = select(SubdomainRegistry).where(SubdomainRegistry.slug == slug)
    res = await session.execute(stmt)
    existing = res.scalars().first()
    if existing is not None:
        return existing

    subdomain = SubdomainRegistry(
        id=id or slug,
        domain_id=domain_id,
        slug=slug,
        name=name,
        standard_bodies=standard_bodies,
    )
    session.add(subdomain)
    await session.commit()
    await session.refresh(subdomain)
    return subdomain


async def update_subdomain(
    session: AsyncSession,
    domain_slug: str,
    subdomain_slug: str,
    name: str | None = None,
    standard_bodies: str | None = None,
) -> SubdomainRegistry:
    """Update mutable attributes (name, standard_bodies) of a subdomain."""
    from src.db.repositories.taxonomy import get_domain_by_slug

    domain = await get_domain_by_slug(session, domain_slug)
    if domain is None:
        raise ValueError(f"Domain '{domain_slug}' not found")
    stmt = select(SubdomainRegistry).where(
        SubdomainRegistry.domain_id == domain.id,
        SubdomainRegistry.slug == subdomain_slug,
    )
    res = await session.execute(stmt)
    sub = res.scalars().first()
    if sub is None:
        raise ValueError(f"Subdomain '{subdomain_slug}' not found in domain '{domain_slug}'")
    if name is not None:
        sub.name = name
    if standard_bodies is not None:
        sub.standard_bodies = standard_bodies
    session.add(sub)
    await session.commit()
    await session.refresh(sub)
    return sub


async def delete_subdomain(
    session: AsyncSession,
    qdrant: AsyncQdrantClient | None,
    collection_name: str | None,
    domain_slug: str,
    subdomain_slug: str,
    target_subdomain_slug: str | None = None,
) -> str:
    """Delete a subdomain, rejecting general.general or the last subdomain, and reassigning records."""
    from src.db.repositories.taxonomy import get_domain_by_slug

    domain = await get_domain_by_slug(session, domain_slug)
    if domain is None:
        raise ValueError(f"Domain '{domain_slug}' not found")

    subs = await list_subdomains(session, domain_id=domain.id)
    if len(subs) <= 1:
        raise ValueError(f"Cannot delete the only remaining subdomain for domain '{domain_slug}'")
    if subdomain_slug == "general.general":
        raise ValueError("Cannot delete root general.general subdomain")

    sub_to_delete = next((s for s in subs if s.slug == subdomain_slug), None)
    if sub_to_delete is None:
        raise ValueError(f"Subdomain '{subdomain_slug}' not found in domain '{domain_slug}'")

    remaining_subs = [s for s in subs if s.slug != subdomain_slug]
    if target_subdomain_slug:
        target_sub = next((s for s in remaining_subs if s.slug == target_subdomain_slug), None)
        if target_sub is None:
            raise ValueError(f"Target subdomain '{target_subdomain_slug}' not found in domain '{domain_slug}'")
    else:
        target_sub = next(
            (s for s in remaining_subs if s.slug == f"{domain.slug}.general"),
            remaining_subs[0],
        )

    all_docs_res = await session.execute(select(Document))
    all_docs = list(all_docs_res.scalars().all())
    for doc in all_docs:
        doc_modified = False
        if doc.primary_subdomain_id == sub_to_delete.id:
            doc.primary_subdomain_id = target_sub.id
            doc_modified = True

        rewritten = rewrite_classification_json_for_reassignment(
            doc.classification_result,
            old_subdomain_slug=subdomain_slug,
            target_subdomain_slug=target_sub.slug,
        )
        if rewritten is not None:
            doc.classification_result = rewritten
            doc_modified = True

        if doc_modified:
            doc.updated_at = _now()
            session.add(doc)

    res_sub_chunks = await session.execute(
        select(DocumentChunk).where(DocumentChunk.subdomain_key == subdomain_slug)
    )
    for c in res_sub_chunks.scalars().all():
        c.subdomain_key = target_sub.slug
        session.add(c)

    await session.flush()

    if qdrant is not None and collection_name:
        await qdrant.set_payload(
            collection_name=collection_name,
            payload={"subdomain": target_sub.slug},
            points=models.Filter(
                must=[
                    models.FieldCondition(key="domain", match=models.MatchValue(value=domain_slug)),
                    models.FieldCondition(key="subdomain", match=models.MatchValue(value=subdomain_slug)),
                ]
            ),
            wait=True,
        )

    await session.delete(sub_to_delete)
    await session.commit()
    return target_sub.slug
