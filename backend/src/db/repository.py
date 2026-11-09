"""CRUD helpers, multi-domain taxonomy repository, and atomic 3-tier deletion."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.db.models import (
    CrossDomainEntity,
    Document,
    DocumentChunk,
    DomainLexicon,
    DomainRegistry,
    EntityDomainMapping,
    SubdomainRegistry,
    WorkspaceSettings,
    _now,
    _uuid,
)


class DocumentCancellable(Protocol):
    """Protocol for components that can cancel in-flight document processing."""

    def cancel_document(self, doc_id: str) -> None:
        ...


if TYPE_CHECKING:
    from src.ingestion.worker import IngestionOrchestrator


async def get_document_by_sha(session: AsyncSession, sha256: str) -> Document | None:
    res = await session.execute(select(Document).where(Document.sha256 == sha256))
    return res.scalars().first()


async def delete_document_atomic(
    session: AsyncSession,
    qdrant: AsyncQdrantClient,
    doc_id: str,
    data_dir: Path | None = None,
    collection: str | None = None,
    orchestrator: DocumentCancellable | None = None,
) -> bool:
    """Delete a document across all tiers. Returns False if it does not exist.

    Order: In-flight cancellation signal -> Qdrant purge first (a failure aborts
    before touching SQLite, so the document stays retryable), then SQLite cascade,
    then artifact cleanup.
    """
    if orchestrator is not None:
        orchestrator.cancel_document(doc_id)

    settings = get_settings()
    collection = collection or settings.qdrant_collection
    data_dir = Path(data_dir) if data_dir else settings.data_dir

    doc = await session.get(Document, doc_id)
    if doc is None:
        return False
    sha, ext = doc.sha256, doc.file_type.lstrip(".")

    # Tier 1: vectors
    await qdrant.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            )
        ),
        wait=True,
    )
    # Tier 2: relational (FK ON DELETE CASCADE removes chunks/jobs)
    await session.delete(doc)
    await session.commit()

    # Tier 3: artifacts (sha256-derived names only; no user-controlled paths)
    for sub, suffix in (("raw", f".{ext}"), ("parsed", ".json"), ("markdown", ".md")):
        if len(sha) == 64 and all(c in "0123456789abcdef" for c in sha):
            (data_dir / sub / f"{sha}{suffix}").unlink(missing_ok=True)
    return True


# ============================================================================
# Multi-Domain Taxonomy & Lexicon Helpers
# ============================================================================


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

    # Check for in-flight documents
    in_flight_res = await session.execute(
        select(Document).where(
            Document.primary_domain_id == domain.id,
            Document.status.in_(["queued", "parsing", "classifying", "chunking", "indexing"]),
        )
    )
    if in_flight_res.scalars().first() is not None:
        raise ValueError(f"Cannot delete domain '{slug}': active documents are currently in-flight")

    # Reassign EntityDomainMapping
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

    # Reassign documents & rewrite classification_result JSON
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

        if doc.classification_result:
            try:
                prof = json.loads(doc.classification_result)
                prof_modified = False
                if prof.get("primary_domain") == slug:
                    prof["primary_domain"] = target_domain.slug
                    prof["primary_subdomain"] = resolved_target_sub
                    prof["epistemic_modality"] = target_domain.epistemic_type
                    if getattr(target_domain, "chunking_strategy", None):
                        prof["chunking_strategy"] = target_domain.chunking_strategy
                    prof_modified = True

                for ov in prof.get("section_overrides", []):
                    if ov.get("domain") == slug:
                        ov["domain"] = target_domain.slug
                        ov["subdomain"] = resolved_target_sub
                        ov["epistemic_modality"] = target_domain.epistemic_type
                        prof_modified = True

                if prof_modified:
                    doc.classification_result = json.dumps(prof)
                    doc_modified = True
            except Exception:
                pass

        if doc_modified:
            doc.updated_at = _now()
            session.add(doc)

    # Reassign DocumentChunk rows
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

    # Flush SQLite -> Qdrant set_payload -> Delete DomainRegistry -> Commit
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


async def list_subdomains(session: AsyncSession, domain_id: str | None = None) -> list[SubdomainRegistry]:
    """List subdomains, optionally filtered by domain_id."""
    stmt = select(SubdomainRegistry)
    if domain_id:
        stmt = stmt.where(SubdomainRegistry.domain_id == domain_id)
    stmt = stmt.order_by(SubdomainRegistry.slug)
    res = await session.execute(stmt)
    return list(res.scalars().all())


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

        if doc.classification_result:
            try:
                prof = json.loads(doc.classification_result)
                prof_modified = False
                if prof.get("primary_subdomain") == subdomain_slug:
                    prof["primary_subdomain"] = target_sub.slug
                    prof_modified = True

                for ov in prof.get("section_overrides", []):
                    if ov.get("subdomain") == subdomain_slug:
                        ov["subdomain"] = target_sub.slug
                        prof_modified = True

                if prof_modified:
                    doc.classification_result = json.dumps(prof)
                    doc_modified = True
            except Exception:
                pass

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

    # Map normalized term string to its lexicon entries
    term_to_entries: dict[str, list[DomainLexicon]] = {}
    for entry in all_entries:
        key = entry.term.strip().lower()
        term_to_entries.setdefault(key, []).append(entry)

    matches: list[DomainLexicon] = []
    # Match terms using Unicode-aware word boundary pattern
    for term_lower, entries in term_to_entries.items():
        pattern = r"(?<!\w)" + re.escape(term_lower) + r"(?!\w)"
        if re.search(pattern, text, re.IGNORECASE):
            matches.extend(entries)

    return matches


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
        pattern = r"(?<!\w)" + re.escape(name.lower()) + r"(?!\w)"
        if re.search(pattern, text, re.IGNORECASE):
            matches.append(ent)
    return matches


async def get_workspace_settings(session: AsyncSession) -> WorkspaceSettings:
    """Get the singleton WorkspaceSettings row, initializing it if absent."""
    res = await session.execute(select(WorkspaceSettings).where(WorkspaceSettings.id == "default"))
    ws = res.scalars().first()
    if ws is None:
        ws = WorkspaceSettings(id="default")
        session.add(ws)
        await session.commit()
        await session.refresh(ws)
    return ws


async def update_workspace_settings(
    session: AsyncSession,
    default_upload_mode: str | None = None,
    default_approval_mode: str | None = None,
    high_confidence_threshold: float | None = None,
    default_llm_provider: str | None = None,
    default_llm_model: str | None = None,
    api_keys: dict[str, str | None] | None = None,
    key_validations: dict[str, dict[str, Any] | None] | None = None,
    llm_compat_base_url: str | None = None,
) -> WorkspaceSettings:
    """Update WorkspaceSettings in SQLite and sync runtime configuration."""
    from src.rag.llm import normalize_api_key, sync_runtime_credentials

    ws = await get_workspace_settings(session)
    if default_upload_mode is not None:
        ws.default_upload_mode = default_upload_mode
    if default_approval_mode is not None:
        ws.default_approval_mode = default_approval_mode
    if high_confidence_threshold is not None:
        ws.high_confidence_threshold = high_confidence_threshold
    if default_llm_provider is not None:
        ws.default_llm_provider = default_llm_provider.strip() or None
    if default_llm_model is not None:
        ws.default_llm_model = default_llm_model.strip() or None
    if llm_compat_base_url is not None:
        ws.llm_compat_base_url = llm_compat_base_url.strip() or None

    stored_keys: dict[str, str] = {}
    if ws.llm_api_keys:
        try:
            parsed = json.loads(ws.llm_api_keys)
            if isinstance(parsed, dict):
                stored_keys = {str(k): str(v) for k, v in parsed.items() if v is not None}
        except Exception:
            stored_keys = {}

    stored_validations: dict[str, dict[str, Any]] = {}
    if ws.llm_key_validations:
        try:
            parsed = json.loads(ws.llm_key_validations)
            if isinstance(parsed, dict):
                stored_validations = {str(k): v for k, v in parsed.items() if isinstance(v, dict)}
        except Exception:
            stored_validations = {}

    if api_keys is not None:
        for prov, raw_val in api_keys.items():
            prov_key = prov.strip().lower()
            if raw_val is None or raw_val.strip() == "":
                stored_keys.pop(prov_key, None)
                if not (key_validations and prov_key in key_validations):
                    stored_validations.pop(prov_key, None)
            else:
                norm_key = normalize_api_key(raw_val)
                old_key = stored_keys.get(prov_key)
                stored_keys[prov_key] = norm_key
                if old_key != norm_key and not (key_validations and prov_key in key_validations):
                    stored_validations.pop(prov_key, None)

    if key_validations is not None:
        for prov, v in key_validations.items():
            prov_key = prov.strip().lower()
            if v is None:
                stored_validations.pop(prov_key, None)
            elif isinstance(v, dict):
                stored_validations[prov_key] = v

    ws.llm_api_keys = json.dumps(stored_keys) if stored_keys else None
    ws.llm_key_validations = json.dumps(stored_validations) if stored_validations else None

    ws.updated_at = _now()
    session.add(ws)
    await session.commit()
    await session.refresh(ws)

    # Sync runtime settings (never assign ws.llm_compat_base_url to runtime_settings.llm.base_url)
    sync_runtime_credentials(stored_keys, stored_validations, ws.llm_compat_base_url)

    runtime_settings = get_settings()
    if ws.high_confidence_threshold is not None:
        runtime_settings.ingestion.high_confidence_threshold = ws.high_confidence_threshold
    if default_llm_provider is not None:
        runtime_settings.llm.provider = ws.default_llm_provider or "openai"
    elif ws.default_llm_provider:
        runtime_settings.llm.provider = ws.default_llm_provider
    if default_llm_model is not None:
        runtime_settings.llm.model_name = ws.default_llm_model or ""
    elif ws.default_llm_model:
        runtime_settings.llm.model_name = ws.default_llm_model

    return ws


async def sync_document_metadata_across_stores(
    session: AsyncSession,
    qdrant: AsyncQdrantClient | None,
    collection_name: str | None,
    doc: Document,
) -> None:
    """Synchronize document metadata, chunks, and Qdrant points across stores.

    Order: SQLite updates -> session.flush() -> Qdrant set_payload(wait=True) -> session.commit().
    """
    primary_domain_slug = "general"
    primary_subdomain_slug = "general.general"
    primary_epistemic = "general"

    if doc.primary_domain_id:
        d_obj = await session.get(DomainRegistry, doc.primary_domain_id)
        if d_obj:
            primary_domain_slug = d_obj.slug
            primary_epistemic = d_obj.epistemic_type
    if doc.primary_subdomain_id:
        s_obj = await session.get(SubdomainRegistry, doc.primary_subdomain_id)
        if s_obj:
            primary_subdomain_slug = s_obj.slug

    # Parse and synchronize doc.classification_result
    parsed_profile: dict[str, Any] = {}
    if doc.classification_result:
        try:
            parsed_profile = json.loads(doc.classification_result)
        except Exception:
            parsed_profile = {}

    parsed_profile["primary_domain"] = primary_domain_slug
    parsed_profile["primary_subdomain"] = primary_subdomain_slug
    parsed_profile["epistemic_modality"] = primary_epistemic
    parsed_profile["authority_tier"] = doc.authority_tier
    parsed_profile["jurisdiction"] = doc.jurisdiction
    parsed_profile["effective_year"] = doc.effective_year
    parsed_profile["is_current"] = doc.is_current

    section_overrides = parsed_profile.get("section_overrides") or []
    doc.classification_result = json.dumps(parsed_profile)
    doc.updated_at = _now()
    session.add(doc)

    # Fetch all chunks for this document
    chunks_res = await session.execute(
        select(DocumentChunk).where(DocumentChunk.doc_id == doc.id)
    )
    chunks = list(chunks_res.scalars().all())

    # Map chunks to effective domain, subdomain, epistemic_modality
    grouped_points: dict[tuple[str, str, str], list[str]] = {}

    for chunk in chunks:
        matched_override = None
        if chunk.heading_path and section_overrides:
            h_clean = chunk.heading_path.strip().lower()
            for ov in section_overrides:
                ov_h = (
                    ov.get("heading_path") if isinstance(ov, dict)
                    else getattr(ov, "heading_path", "")
                ).strip().lower()
                if ov_h and (ov_h == h_clean or ov_h in h_clean or h_clean in ov_h):
                    matched_override = ov
                    break

        if matched_override:
            c_domain = matched_override.get("domain") if isinstance(matched_override, dict) else matched_override.domain
            c_subdomain = matched_override.get("subdomain") if isinstance(matched_override, dict) else matched_override.subdomain
            c_epistemic = (
                matched_override.get("epistemic_modality") if isinstance(matched_override, dict)
                else getattr(matched_override, "epistemic_modality", None)
            ) or "general"
        else:
            c_domain = primary_domain_slug
            c_subdomain = primary_subdomain_slug
            c_epistemic = primary_epistemic

        chunk.domain_key = c_domain
        chunk.subdomain_key = c_subdomain
        chunk.epistemic_modality = c_epistemic
        session.add(chunk)

        if chunk.point_id:
            key = (c_domain, c_subdomain, c_epistemic)
            grouped_points.setdefault(key, []).append(chunk.point_id)

    # Invariant: session.flush() -> qdrant.set_payload(..., wait=True) -> session.commit()
    await session.flush()

    if qdrant is not None and collection_name:
        for (g_domain, g_subdomain, g_epistemic), point_ids in grouped_points.items():
            if not point_ids:
                continue
            payload_to_set = {
                "domain": g_domain,
                "subdomain": g_subdomain,
                "epistemic_type": g_epistemic,
                "authority_tier": int(doc.authority_tier),
                "jurisdiction": doc.jurisdiction,
                "effective_year": doc.effective_year,
                "is_current": bool(doc.is_current),
            }
            await qdrant.set_payload(
                collection_name=collection_name,
                payload=payload_to_set,
                points=models.PointIdsList(points=point_ids),
                wait=True,
            )

    await session.commit()
    await session.refresh(doc)

