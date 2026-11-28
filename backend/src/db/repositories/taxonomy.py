"""Multi-domain and subdomain registry CRUD with cross-store Qdrant payload synchronization."""
from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    Document,
    DocumentChunk,
    DomainRegistry,
    EntityDomainMapping,
    SubdomainRegistry,
    _now,
)
from src.db.repositories.common import rewrite_classification_json_for_reassignment
from src.db.repositories.taxonomy_subdomains import (
    create_subdomain,
    delete_subdomain,
    list_subdomains,
    list_subdomains_by_domain_ids,
    update_subdomain,
)

__all__ = [
    "create_domain",
    "create_subdomain",
    "delete_domain_and_reassign",
    "delete_subdomain",
    "get_domain_by_slug",
    "list_domains",
    "list_subdomains",
    "list_subdomains_by_domain_ids",
    "resolve_domain_by_id_or_slug",
    "update_domain",
    "update_subdomain",
]


async def list_domains(session: AsyncSession) -> list[DomainRegistry]:
    """Return all registered domains ordered by authority priority and slug."""
    stmt = select(DomainRegistry).order_by(DomainRegistry.authority_priority, DomainRegistry.slug)
    res = await session.execute(stmt)
    return list(res.scalars().all())


async def get_domain_by_slug(session: AsyncSession, slug: str) -> DomainRegistry | None:
    """Retrieve a domain by its slug."""
    stmt = select(DomainRegistry).where(DomainRegistry.slug == slug)
    res = await session.execute(stmt)
    return res.scalars().first()


async def resolve_domain_by_id_or_slug(session: AsyncSession, identifier: str) -> DomainRegistry | None:
    """Retrieve a domain matching either its slug or primary key ID."""
    clean = identifier.strip()
    if not clean:
        return None
    domain = await get_domain_by_slug(session, clean)
    if domain is not None:
        return domain
    return await session.get(DomainRegistry, clean)


async def create_domain(
    session: AsyncSession,
    slug: str,
    name: str,
    epistemic_type: str = "general",
    authority_priority: int = 3,
    description: str = "",
    id: str | None = None,
    icon: str = "📁",
    color_token: str = "indigo",
    chunking_strategy: str = "standard_markdown",
    classifier_keywords: str | None = None,
    is_system: bool = False,
    is_active: bool = True,
    auto_create_default_subdomain: bool = False,
) -> DomainRegistry:
    """Create or return an existing domain registry entry."""
    existing = await get_domain_by_slug(session, slug)
    if existing is not None:
        return existing

    domain = DomainRegistry(
        id=id or slug,
        slug=slug,
        name=name,
        epistemic_type=epistemic_type,
        authority_priority=authority_priority,
        description=description,
        icon=icon,
        color_token=color_token,
        chunking_strategy=chunking_strategy,
        classifier_keywords=classifier_keywords,
        is_system=is_system,
        is_active=is_active,
    )
    session.add(domain)
    await session.flush()

    if auto_create_default_subdomain:
        sub_slug = f"{slug}.general"
        sub = SubdomainRegistry(
            id=sub_slug,
            domain_id=domain.id,
            slug=sub_slug,
            name="General",
            standard_bodies=None,
        )
        session.add(sub)

    await session.commit()
    await session.refresh(domain)
    return domain


async def update_domain(
    session: AsyncSession,
    qdrant: AsyncQdrantClient | None,
    collection_name: str | None,
    slug: str,
    name: str | None = None,
    epistemic_type: str | None = None,
    authority_priority: int | None = None,
    description: str | None = None,
    icon: str | None = None,
    color_token: str | None = None,
    chunking_strategy: str | None = None,
    classifier_keywords: str | None = None,
    is_active: bool | None = None,
) -> DomainRegistry:
    """Update mutable attributes of a domain in SQLite and propagate epistemic modality changes to chunks and Qdrant."""
    domain = await get_domain_by_slug(session, slug)
    if domain is None:
        raise ValueError(f"Domain '{slug}' not found")

    old_epistemic = domain.epistemic_type
    epistemic_changed = epistemic_type is not None and epistemic_type != old_epistemic

    if name is not None:
        domain.name = name
    if epistemic_type is not None:
        domain.epistemic_type = epistemic_type
    if authority_priority is not None:
        domain.authority_priority = authority_priority
    if description is not None:
        domain.description = description
    if icon is not None:
        domain.icon = icon
    if color_token is not None:
        domain.color_token = color_token
    if chunking_strategy is not None:
        domain.chunking_strategy = chunking_strategy
    if classifier_keywords is not None:
        domain.classifier_keywords = classifier_keywords
    if is_active is not None:
        domain.is_active = is_active

    session.add(domain)

    if epistemic_changed:
        res_chunks = await session.execute(
            select(DocumentChunk).where(DocumentChunk.domain_key == slug)
        )
        for c in res_chunks.scalars().all():
            c.epistemic_modality = epistemic_type
            session.add(c)

    await session.flush()

    if epistemic_changed and qdrant is not None and collection_name:
        await qdrant.set_payload(
            collection_name=collection_name,
            payload={"epistemic_type": epistemic_type},
            points=models.Filter(
                must=[models.FieldCondition(key="domain", match=models.MatchValue(value=slug))]
            ),
            wait=True,
        )

    await session.commit()
    await session.refresh(domain)
    return domain


async def delete_domain_and_reassign(
    session: AsyncSession,
    qdrant: AsyncQdrantClient | None,
    collection_name: str | None,
    slug: str,
    target_domain_slug: str = "general",
    target_subdomain_slug: str | None = None,
) -> dict[str, Any]:
    """Delete a custom domain and atomically reassign documents, entities, chunks, and Qdrant payloads to a target domain."""
    domain = await get_domain_by_slug(session, slug)
    if domain is None:
        raise ValueError(f"Domain '{slug}' not found")
    if domain.is_system:
        raise ValueError(f"Cannot delete system protected domain '{slug}'")
    if target_domain_slug == slug:
        raise ValueError("Target domain must be different from domain being deleted")

    target_domain = await get_domain_by_slug(session, target_domain_slug)
    if target_domain is None:
        raise ValueError(f"Target domain '{target_domain_slug}' not found")

    target_subs = await list_subdomains(session, domain_id=target_domain.id)
    resolved_target_sub_obj: SubdomainRegistry | None = None
    if target_subdomain_slug:
        resolved_target_sub_obj = next((s for s in target_subs if s.slug == target_subdomain_slug), None)
        if resolved_target_sub_obj is None:
            raise ValueError(f"Target subdomain '{target_subdomain_slug}' does not belong to '{target_domain_slug}'")
    else:
        resolved_target_sub_obj = next(
            (s for s in target_subs if s.slug == f"{target_domain.slug}.general"),
            target_subs[0] if target_subs else None,
        )

    resolved_target_sub = (
        resolved_target_sub_obj.slug
        if resolved_target_sub_obj
        else f"{target_domain.slug}.general"
    )
    resolved_target_sub_id = resolved_target_sub_obj.id if resolved_target_sub_obj else None

    in_flight_res = await session.execute(
        select(Document).where(
            Document.primary_domain_id == domain.id,
            Document.status.in_(["queued", "parsing", "classifying", "chunking", "indexing"]),
        )
    )
    if in_flight_res.scalars().first() is not None:
        raise ValueError(f"Cannot delete domain '{slug}': active documents are currently in-flight")

    mappings_res = await session.execute(
        select(EntityDomainMapping).where(EntityDomainMapping.domain_id == domain.id)
    )
    domain_mappings = list(mappings_res.scalars().all())

    target_mappings_res = await session.execute(
        select(EntityDomainMapping).where(EntityDomainMapping.domain_id == target_domain.id)
    )
    target_pairs = {
        (m.entity_id, m.chunk_id) for m in target_mappings_res.scalars().all()
    }

    for m in domain_mappings:
        if (m.entity_id, m.chunk_id) in target_pairs:
            await session.delete(m)
        else:
            m.domain_id = target_domain.id
            session.add(m)

    all_docs_res = await session.execute(select(Document))
    all_docs = list(all_docs_res.scalars().all())
    reassigned_docs_count = 0

    for doc in all_docs:
        doc_modified = False
        if doc.primary_domain_id == domain.id:
            doc.primary_domain_id = target_domain.id
            doc.primary_subdomain_id = resolved_target_sub_id
            reassigned_docs_count += 1
            doc_modified = True

        rewritten = rewrite_classification_json_for_reassignment(
            doc.classification_result,
            old_domain_slug=slug,
            target_domain_slug=target_domain.slug,
            target_subdomain_slug=resolved_target_sub,
            target_epistemic_type=target_domain.epistemic_type,
            target_chunking_strategy=getattr(target_domain, "chunking_strategy", None),
        )
        if rewritten is not None:
            doc.classification_result = rewritten
            doc_modified = True

        if doc_modified:
            doc.updated_at = _now()
            session.add(doc)

    chunks_res = await session.execute(
        select(DocumentChunk).where(DocumentChunk.domain_key == slug)
    )
    reassigned_chunks = list(chunks_res.scalars().all())
    reassigned_chunks_count = len(reassigned_chunks)

    for chunk in reassigned_chunks:
        chunk.domain_key = target_domain.slug
        chunk.subdomain_key = resolved_target_sub
        chunk.epistemic_modality = target_domain.epistemic_type
        session.add(chunk)

    await session.flush()

    if qdrant is not None and collection_name:
        await qdrant.set_payload(
            collection_name=collection_name,
            payload={
                "domain": target_domain.slug,
                "subdomain": resolved_target_sub,
                "epistemic_type": target_domain.epistemic_type,
            },
            points=models.Filter(
                must=[models.FieldCondition(key="domain", match=models.MatchValue(value=slug))]
            ),
            wait=True,
        )

    await session.delete(domain)
    await session.commit()

    return {
        "deleted_slug": slug,
        "target_domain": target_domain.slug,
        "target_subdomain": resolved_target_sub,
        "reassigned_documents": reassigned_docs_count,
        "reassigned_chunks": reassigned_chunks_count,
    }
