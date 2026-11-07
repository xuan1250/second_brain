"""Unit tests for Phase 1: Database schema, taxonomy seeding, migrations, and Qdrant partitioning."""
from __future__ import annotations

import json
import uuid
import pytest
from qdrant_client import AsyncQdrantClient, models as qmodels
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import create_async_engine

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
)
from src.db.qdrant import (
    backfill_qdrant_legacy_payloads,
    init_qdrant_collection,
)
from src.db.repository import (
    add_lexicon_term,
    create_domain,
    create_subdomain,
    delete_domain_and_reassign,
    delete_lexicon_term,
    delete_subdomain,
    get_domain_by_slug,
    get_lexicon_matches,
    get_workspace_settings,
    list_domains,
    list_lexicon_terms,
    list_subdomains,
    sync_document_metadata_across_stores,
    update_domain,
    update_workspace_settings,
    upsert_cross_domain_entities,
)
from src.db.session import (
    create_session_factory,
    init_db,
)


@pytest.fixture
def in_memory_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    return engine


@pytest.mark.asyncio
async def test_multidomain_models_creation(in_memory_engine):
    """Test that all SQLModel tables create cleanly and store domain metadata."""
    await init_db(in_memory_engine)
    session_factory = create_session_factory(in_memory_engine)

    async with session_factory() as session:
        # Create a document with multi-domain fields
        doc = Document(
            filename="transformer_spec.pdf",
            file_type=".pdf",
            sha256="a" * 64,
            primary_domain_id="engineering",
            primary_subdomain_id="eng.electrical",
            jurisdiction="IEEE",
            authority_tier=1,
            effective_year=2024,
            is_current=True,
            classification_result='{"domain": "engineering", "confidence": 0.95}',
        )
        session.add(doc)
        await session.commit()
        await session.refresh(doc)

        assert doc.id is not None
        assert doc.primary_domain_id == "engineering"
        assert doc.authority_tier == 1
        assert doc.is_current is True

        # Create a document chunk with domain partition keys
        chunk = DocumentChunk(
            doc_id=doc.id,
            point_id=str(uuid.uuid4()),
            chunk_index=0,
            text="Transformer winding insulation breakdown voltage...",
            domain_key="engineering",
            subdomain_key="eng.electrical",
            epistemic_modality="empirical",
            contextual_prefix="[Domain: Engineering | Subdomain: Electrical | Standard: IEEE C57]",
        )
        session.add(chunk)
        await session.commit()
        await session.refresh(chunk)

        assert chunk.id is not None
        assert chunk.domain_key == "engineering"
        assert chunk.epistemic_modality == "empirical"
        assert "[Domain: Engineering" in chunk.contextual_prefix


@pytest.mark.asyncio
async def test_taxonomy_seeding_integrity(in_memory_engine):
    """Verify default taxonomy (4 domains, 13 subdomains, 60 bilingual lexicon terms) and idempotence."""
    await init_db(in_memory_engine)
    session_factory = create_session_factory(in_memory_engine)

    async with session_factory() as session:
        # 1. Verify 4 default domains
        domains = (await session.execute(select(DomainRegistry))).scalars().all()
        assert len(domains) == 4
        domain_slugs = {d.slug: d for d in domains}
        assert set(domain_slugs.keys()) == {"engineering", "economics", "law", "general"}

        assert domain_slugs["engineering"].epistemic_type == "empirical"
        assert domain_slugs["economics"].epistemic_type == "evaluative"
        assert domain_slugs["law"].epistemic_type == "normative"
        assert domain_slugs["general"].epistemic_type == "general"

        # 2. Verify 13 subdomains
        subdomains = (await session.execute(select(SubdomainRegistry))).scalars().all()
        assert len(subdomains) == 13
        sub_slugs = {sd.slug for sd in subdomains}

        expected_eng = {"eng.electrical", "eng.control", "eng.mechanical", "eng.technology"}
        expected_econ = {"econ.finance", "econ.accounting", "econ.investment", "econ.taxation"}
        expected_law = {"law.statute", "law.regulation", "law.case_law", "law.contract"}
        expected_general = {"general.general"}

        assert expected_eng.issubset(sub_slugs)
        assert expected_econ.issubset(sub_slugs)
        assert expected_law.issubset(sub_slugs)
        assert expected_general.issubset(sub_slugs)

        # 3. Verify bilingual starter lexicon (10 EN + 10 VI terms across 3 domains = 60 entries)
        lexicons = (await session.execute(select(DomainLexicon))).scalars().all()
        assert len(lexicons) == 60

        en_terms = {
            "liability", "depreciation", "current", "control", "standard",
            "equity", "plant", "default", "instrument", "stress"
        }
        vi_terms = {
            "trách nhiệm", "khấu hao", "dòng", "kiểm soát", "tiêu chuẩn",
            "công bằng", "nhà máy", "mặc định", "công cụ", "ứng suất"
        }
        db_terms = {l.term for l in lexicons}
        assert en_terms.issubset(db_terms)
        assert vi_terms.issubset(db_terms)

        for lex in lexicons:
            assert lex.specific_sense != ""
            assert lex.contrastive_warning != ""
            assert lex.domain_id in {"engineering", "economics", "law"}

    # Run init_db second time to assert idempotence
    await init_db(in_memory_engine)
    async with session_factory() as session:
        domains_run2 = (await session.execute(select(DomainRegistry))).scalars().all()
        subdomains_run2 = (await session.execute(select(SubdomainRegistry))).scalars().all()
        lexicons_run2 = (await session.execute(select(DomainLexicon))).scalars().all()
        assert len(domains_run2) == 4
        assert len(subdomains_run2) == 13
        assert len(lexicons_run2) == 60


@pytest.mark.asyncio
async def test_legacy_sqlite_migration_and_backfill(in_memory_engine):
    """Simulate legacy database without new columns, verify column additions and NULL row backfill."""
    async with in_memory_engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE documents (
                id VARCHAR PRIMARY KEY,
                filename VARCHAR NOT NULL,
                file_type VARCHAR NOT NULL,
                sha256 VARCHAR NOT NULL,
                size_bytes INTEGER NOT NULL,
                page_count INTEGER,
                chunk_count INTEGER NOT NULL,
                status VARCHAR NOT NULL,
                error VARCHAR,
                parsing_engine VARCHAR,
                engine_fallback_reason VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            );
        """))
        await conn.execute(text("""
            CREATE TABLE document_chunks (
                id VARCHAR PRIMARY KEY,
                doc_id VARCHAR NOT NULL,
                point_id VARCHAR NOT NULL,
                parent_id VARCHAR,
                chunk_index INTEGER NOT NULL,
                heading_path VARCHAR NOT NULL,
                page_numbers VARCHAR,
                token_count INTEGER NOT NULL,
                text VARCHAR NOT NULL,
                parent_content VARCHAR NOT NULL
            );
        """))
        await conn.execute(text("""
            INSERT INTO documents (id, filename, file_type, sha256, size_bytes, chunk_count, status)
            VALUES ('legacy_doc_1', 'legacy.pdf', '.pdf', 'b' * 64, 500, 1, 'ready');
        """))
        await conn.execute(text("""
            INSERT INTO document_chunks (id, doc_id, point_id, chunk_index, heading_path, token_count, text, parent_content)
            VALUES ('legacy_chunk_1', 'legacy_doc_1', 'legacy_pt_1', 0, '', 15, 'legacy chunk text', '');
        """))

    # Execute init_db which must apply schema alterations and backfill
    await init_db(in_memory_engine)

    session_factory = create_session_factory(in_memory_engine)
    async with session_factory() as session:
        doc = await session.get(Document, "legacy_doc_1")
        assert doc is not None
        assert doc.primary_domain_id == "general"
        assert doc.primary_subdomain_id == "general.general"
        assert doc.authority_tier == 3
        assert doc.is_current is True

        chunk = await session.get(DocumentChunk, "legacy_chunk_1")
        assert chunk is not None
        assert chunk.domain_key == "general"
        assert chunk.subdomain_key == "general.general"
        assert chunk.epistemic_modality == "general"


@pytest.mark.asyncio
async def test_repository_crud(in_memory_engine):
    """Test dynamic CRUD operations for domains, subdomains, lexicon, and cross-domain entities."""
    await init_db(in_memory_engine)
    session_factory = create_session_factory(in_memory_engine)

    async with session_factory() as session:
        # 1. list_domains & get_domain_by_slug
        domains = await list_domains(session)
        assert len(domains) >= 4
        eng = await get_domain_by_slug(session, "engineering")
        assert eng is not None
        assert eng.name == "Engineering & Technology"

        # 2. create_domain
        custom_domain = await create_domain(
            session=session,
            slug="biomedical",
            name="Biomedical Science",
            epistemic_type="empirical",
            authority_priority=2,
            description="Biomedical and health sciences domain.",
        )
        assert custom_domain.id == "biomedical"
        assert custom_domain.slug == "biomedical"

        # Idempotent create_domain
        same_domain = await create_domain(
            session=session,
            slug="biomedical",
            name="Biomedical Duplicate",
        )
        assert same_domain.id == custom_domain.id

        # 3. create_subdomain & list_subdomains
        sub = await create_subdomain(
            session=session,
            slug="bio.genetics",
            name="Genetics & Genomics",
            domain_id=custom_domain.id,
            standard_bodies="NCBI, NIH",
        )
        assert sub.slug == "bio.genetics"
        bio_subs = await list_subdomains(session, domain_id=custom_domain.id)
        assert len(bio_subs) == 1
        assert bio_subs[0].slug == "bio.genetics"

        # 4. add_lexicon_term
        new_term = await add_lexicon_term(
            session=session,
            term="mutation",
            domain_id=custom_domain.id,
            specific_sense="Alteration in nucleotide sequence of genetic material.",
            contrastive_warning="Do not confuse with algorithmic mutation in genetic algorithms.",
        )
        assert new_term.term == "mutation"

        # Update existing term
        updated_term = await add_lexicon_term(
            session=session,
            term="mutation",
            domain_id=custom_domain.id,
            specific_sense="Updated biological mutation sense.",
            contrastive_warning="Updated warning.",
        )
        assert updated_term.specific_sense == "Updated biological mutation sense."

        # 5. upsert_cross_domain_entities
        ents = await upsert_cross_domain_entities(
            session=session,
            entities=[
                {"canonical_name": "T-101", "entity_type": "equipment", "description": "Step-up transformer"},
                CrossDomainEntity(
                    canonical_name="Substation Alpha",
                    entity_type="regulated_facility",
                    description="500kV Substation",
                ),
            ],
        )
        assert len(ents) == 2
        ent_names = {e.canonical_name for e in ents}
        assert "T-101" in ent_names
        assert "Substation Alpha" in ent_names

        # Upsert update description
        updated_ents = await upsert_cross_domain_entities(
            session=session,
            entities=[{"canonical_name": "T-101", "entity_type": "equipment", "description": "Upgraded T-101"}],
        )
        assert len(updated_ents) == 1
        assert updated_ents[0].description == "Upgraded T-101"

        # 6. EntityDomainMapping creation
        doc = Document(filename="doc.pdf", file_type=".pdf", sha256="c" * 64)
        session.add(doc)
        await session.commit()
        await session.refresh(doc)

        chunk = DocumentChunk(doc_id=doc.id, point_id=str(uuid.uuid4()), text="Sample")
        session.add(chunk)
        await session.commit()
        await session.refresh(chunk)

        mapping = EntityDomainMapping(
            entity_id=ents[0].id,
            chunk_id=chunk.id,
            domain_id=custom_domain.id,
            role_in_domain="physical_spec",
        )
        session.add(mapping)
        await session.commit()
        await session.refresh(mapping)
        assert mapping.id is not None
        assert mapping.role_in_domain == "physical_spec"


@pytest.mark.asyncio
async def test_bilingual_lexicon_matching(in_memory_engine):
    """Test get_lexicon_matches with single English tokens and Vietnamese multi-word phrases with diacritics."""
    await init_db(in_memory_engine)
    session_factory = create_session_factory(in_memory_engine)

    async with session_factory() as session:
        # Empty/whitespace test
        assert await get_lexicon_matches(session, "") == []
        assert await get_lexicon_matches(session, "   ") == []

        # 1. English tokens match
        en_text = "Analysis of transformer thermal liability and current operating load under standard limits."
        en_matches = await get_lexicon_matches(session, en_text)
        matched_en_terms = {m.term for m in en_matches}
        assert "liability" in matched_en_terms
        assert "current" in matched_en_terms
        assert "standard" in matched_en_terms
        assert "plant" not in matched_en_terms

        # 2. Vietnamese compound phrases with diacritics match
        vi_text = "Báo cáo phân tích khấu hao tài sản cố định và trách nhiệm pháp lý của nhà máy."
        vi_matches = await get_lexicon_matches(session, vi_text)
        matched_vi_terms = {m.term for m in vi_matches}
        assert "khấu hao" in matched_vi_terms
        assert "trách nhiệm" in matched_vi_terms
        assert "nhà máy" in matched_vi_terms
        assert "ứng suất" not in matched_vi_terms
        assert "công bằng" not in matched_vi_terms

        # 3. Case insensitivity and punctuation resilience
        mixed_text = "Hợp đồng có điều khoản 'MẶC ĐỊNH', 'CÔNG CỤ' tài chính và LIABILITY."
        mixed_matches = await get_lexicon_matches(session, mixed_text)
        mixed_terms = {m.term for m in mixed_matches}
        assert "mặc định" in mixed_terms
        assert "công cụ" in mixed_terms
        assert "liability" in mixed_terms

        # 4. Word boundary isolation (no substring false positives)
        boundary_text = "The plantation was compliant with nonliability rules."
        boundary_matches = await get_lexicon_matches(session, boundary_text)
        boundary_terms = {m.term for m in boundary_matches}
        # "plant" should NOT match "plantation", "liability" should NOT match "nonliability"
        assert "plant" not in boundary_terms
        assert "liability" not in boundary_terms


@pytest.mark.asyncio
async def test_qdrant_partitioning_and_backfill():
    """Test Qdrant payload index registration and zero-re-embedding legacy payload backfill."""
    client = AsyncQdrantClient(":memory:")
    col_name = "test_multidomain_collection"

    # 1. Initialize hybrid collection & payload indexes
    await init_qdrant_collection(client, collection=col_name, dimensions=4)
    assert await client.collection_exists(col_name)

    # 2. Seed 3 points: 2 legacy (missing 'domain'), 1 already partitioned
    id_legacy_1 = str(uuid.uuid4())
    id_legacy_2 = str(uuid.uuid4())
    id_domain_eng = str(uuid.uuid4())

    await client.upsert(
        collection_name=col_name,
        points=[
            qmodels.PointStruct(
                id=id_legacy_1,
                vector={"dense": [0.1, 0.2, 0.3, 0.4]},
                payload={"doc_id": "legacy_doc_1", "text": "Legacy chunk 1"},
            ),
            qmodels.PointStruct(
                id=id_legacy_2,
                vector={"dense": [0.2, 0.3, 0.4, 0.5]},
                payload={"doc_id": "legacy_doc_2", "text": "Legacy chunk 2"},
            ),
            qmodels.PointStruct(
                id=id_domain_eng,
                vector={"dense": [0.5, 0.6, 0.7, 0.8]},
                payload={
                    "doc_id": "eng_doc_1",
                    "domain": "engineering",
                    "subdomain": "eng.electrical",
                    "epistemic_type": "empirical",
                    "authority_tier": 1,
                    "is_current": True,
                },
            ),
        ],
    )

    # 3. Execute legacy backfill
    backfilled_count = await backfill_qdrant_legacy_payloads(client, collection_name=col_name, batch_size=10)
    assert backfilled_count == 2

    # 4. Verify updated payloads
    retrieved = await client.retrieve(
        collection_name=col_name,
        ids=[id_legacy_1, id_legacy_2, id_domain_eng],
        with_payload=True,
    )
    payload_map = {str(p.id): p.payload for p in retrieved}

    # Legacy points backfilled to default 'general' partition
    assert payload_map[id_legacy_1]["domain"] == "general"
    assert payload_map[id_legacy_1]["subdomain"] == "general.general"
    assert payload_map[id_legacy_1]["epistemic_type"] == "general"
    assert payload_map[id_legacy_1]["authority_tier"] == 3
    assert payload_map[id_legacy_1]["is_current"] is True

    assert payload_map[id_legacy_2]["domain"] == "general"
    assert payload_map[id_legacy_2]["subdomain"] == "general.general"

    # Non-legacy point preserved
    assert payload_map[id_domain_eng]["domain"] == "engineering"
    assert payload_map[id_domain_eng]["authority_tier"] == 1

    # 5. Idempotent check (running backfill again should update 0 points)
    second_backfill = await backfill_qdrant_legacy_payloads(client, collection_name=col_name)
    assert second_backfill == 0


@pytest.mark.asyncio
async def test_workspace_settings_singleton_and_runtime_sync(in_memory_engine):
    """Verify WorkspaceSettings singleton seeding, CRUD, and runtime get_settings() synchronization."""
    await init_db(in_memory_engine)
    session_factory = create_session_factory(in_memory_engine)

    async with session_factory() as session:
        ws = await get_workspace_settings(session)
        assert ws.id == "default"
        assert ws.default_upload_mode == "auto"
        assert ws.default_approval_mode == "review_all"
        assert ws.high_confidence_threshold == 0.85
        assert ws.default_llm_provider is None
        assert ws.default_llm_model is None

        # Update settings
        updated_ws = await update_workspace_settings(
            session=session,
            default_upload_mode="fast",
            default_approval_mode="auto_high_confidence",
            high_confidence_threshold=0.92,
            default_llm_provider="anthropic",
            default_llm_model="claude-3-7-sonnet",
        )
        assert updated_ws.default_upload_mode == "fast"
        assert updated_ws.default_approval_mode == "auto_high_confidence"
        assert updated_ws.high_confidence_threshold == 0.92
        assert updated_ws.default_llm_provider == "anthropic"
        assert updated_ws.default_llm_model == "claude-3-7-sonnet"

        # Verify runtime sync
        runtime_settings = get_settings()
        assert runtime_settings.ingestion.high_confidence_threshold == 0.92
        assert runtime_settings.llm.provider == "anthropic"
        assert runtime_settings.llm.model_name == "claude-3-7-sonnet"


@pytest.mark.asyncio
async def test_domain_registry_extended_fields_and_system_protection(in_memory_engine):
    """Verify 6 new columns on DomainRegistry and system domain deletion protection."""
    await init_db(in_memory_engine)
    session_factory = create_session_factory(in_memory_engine)

    async with session_factory() as session:
        domains = await list_domains(session)
        dom_map = {d.slug: d for d in domains}

        for slug, expected_icon, expected_color, expected_strategy in [
            ("engineering", "⚙️", "cyan", "academic_equations"),
            ("economics", "📈", "emerald", "financial_tables"),
            ("law", "⚖️", "amber", "statutory_clauses"),
            ("general", "📄", "slate", "standard_markdown"),
        ]:
            d = dom_map[slug]
            assert d.is_system is True
            assert d.is_active is True
            assert d.icon == expected_icon
            assert d.color_token == expected_color
            assert d.chunking_strategy == expected_strategy

        # System domain deletion rejection
        with pytest.raises(ValueError, match="Cannot delete system protected domain 'engineering'"):
            await delete_domain_and_reassign(
                session=session,
                qdrant=None,
                collection_name=None,
                slug="engineering",
                target_domain_slug="general",
            )


@pytest.mark.asyncio
async def test_custom_domain_lifecycle_and_reassignment(in_memory_engine):
    """Verify custom domain creation with auto-subdomain, update, and deletion reassignment."""
    await init_db(in_memory_engine)
    session_factory = create_session_factory(in_memory_engine)

    qdrant = AsyncQdrantClient(":memory:")
    col_name = "test_custom_domain_collection"
    await init_qdrant_collection(qdrant, collection=col_name, dimensions=4)

    async with session_factory() as session:
        # 1. Create domain with auto_create_default_subdomain=True
        custom = await create_domain(
            session=session,
            slug="biomedical",
            name="Biomedical Science",
            epistemic_type="empirical",
            authority_priority=2,
            description="Biomedical research",
            icon="🧬",
            color_token="rose",
            chunking_strategy="standard_markdown",
            classifier_keywords="genetics, crispr, cell",
            is_system=False,
            is_active=True,
            auto_create_default_subdomain=True,
        )
        assert custom.slug == "biomedical"
        assert custom.is_system is False
        assert custom.icon == "🧬"

        subs = await list_subdomains(session, domain_id=custom.id)
        assert len(subs) == 1
        assert subs[0].slug == "biomedical.general"

        # 2. Add extra subdomain
        sub_genetics = await create_subdomain(
            session=session,
            slug="bio.genetics",
            name="Genetics",
            domain_id=custom.id,
        )
        assert sub_genetics.slug == "bio.genetics"

        # 3. Create document and chunks under biomedical
        doc_id = str(uuid.uuid4())
        chunk_point_id = str(uuid.uuid4())
        doc = Document(
            id=doc_id,
            filename="genetics_paper.pdf",
            file_type="pdf",
            sha256="c" * 64,
            status="ready",
            primary_domain_id=custom.id,
            primary_subdomain_id=sub_genetics.id,
            authority_tier=2,
            classification_result=json.dumps({
                "primary_domain": "biomedical",
                "primary_subdomain": "bio.genetics",
                "epistemic_modality": "empirical",
                "section_overrides": [
                    {
                        "heading_path": "Methodology",
                        "domain": "biomedical",
                        "subdomain": "bio.genetics",
                        "epistemic_modality": "empirical",
                    }
                ],
            }),
        )
        session.add(doc)
        await session.flush()

        chunk = DocumentChunk(
            id=str(uuid.uuid4()),
            doc_id=doc.id,
            point_id=chunk_point_id,
            chunk_index=0,
            text="CRISPR-Cas9 gene editing mechanism...",
            heading_path="Methodology",
            domain_key="biomedical",
            subdomain_key="bio.genetics",
            epistemic_modality="empirical",
        )
        session.add(chunk)
        await session.flush()

        # Seed CrossDomainEntity and EntityDomainMapping
        ent = CrossDomainEntity(canonical_name="CRISPR Tool", entity_type="equipment")
        session.add(ent)
        await session.flush()
        mapping = EntityDomainMapping(
            entity_id=ent.id,
            chunk_id=chunk.id,
            domain_id=custom.id,
            role_in_domain="gene_editing_apparatus",
        )
        session.add(mapping)
        await session.commit()

        # Seed Qdrant point
        await qdrant.upsert(
            collection_name=col_name,
            points=[
                qmodels.PointStruct(
                    id=chunk_point_id,
                    vector={"dense": [0.1, 0.2, 0.3, 0.4]},
                    payload={
                        "domain": "biomedical",
                        "subdomain": "bio.genetics",
                        "epistemic_type": "empirical",
                        "doc_id": doc.id,
                    },
                )
            ],
        )

        # 4. Test update_domain epistemic_type propagation
        updated_domain = await update_domain(
            session=session,
            qdrant=qdrant,
            collection_name=col_name,
            slug="biomedical",
            epistemic_type="normative",
        )
        assert updated_domain.epistemic_type == "normative"

        # Verify chunk epistemic_modality updated
        c_refresh = await session.get(DocumentChunk, chunk.id)
        assert c_refresh.epistemic_modality == "normative"

        # Verify Qdrant payload updated
        q_pts = await qdrant.retrieve(collection_name=col_name, ids=[chunk_point_id], with_payload=True)
        assert q_pts[0].payload["epistemic_type"] == "normative"

        # 5. Test delete_subdomain guard
        # Cannot delete last subdomain: try deleting biomedical.general when bio.genetics also exists (allowed)
        del_res = await delete_subdomain(
            session=session,
            qdrant=qdrant,
            collection_name=col_name,
            domain_slug="biomedical",
            subdomain_slug="bio.genetics",
            target_subdomain_slug="biomedical.general",
        )
        assert del_res == "biomedical.general"

        # Now only 1 subdomain remains in biomedical: deleting it must fail
        with pytest.raises(ValueError, match="Cannot delete the only remaining subdomain"):
            await delete_subdomain(
                session=session,
                qdrant=qdrant,
                collection_name=col_name,
                domain_slug="biomedical",
                subdomain_slug="biomedical.general",
            )

        # 6. Test delete_domain_and_reassign with in-flight doc check
        doc.status = "parsing"
        session.add(doc)
        await session.commit()

        with pytest.raises(ValueError, match="active documents are currently in-flight"):
            await delete_domain_and_reassign(
                session=session,
                qdrant=qdrant,
                collection_name=col_name,
                slug="biomedical",
                target_domain_slug="general",
            )

        # Reset doc status to ready and reassign
        doc.status = "ready"
        session.add(doc)
        await session.commit()

        stats = await delete_domain_and_reassign(
            session=session,
            qdrant=qdrant,
            collection_name=col_name,
            slug="biomedical",
            target_domain_slug="general",
        )
        assert stats["deleted_slug"] == "biomedical"
        assert stats["target_domain"] == "general"
        assert stats["target_subdomain"] == "general.general"
        assert stats["reassigned_documents"] == 1
        assert stats["reassigned_chunks"] == 1

        # Verify domain deleted
        assert await get_domain_by_slug(session, "biomedical") is None

        # Verify Document reassigned and JSON scrubbed
        d_refresh = await session.get(Document, doc.id)
        assert d_refresh.primary_domain_id == "general"
        prof_scrubbed = json.loads(d_refresh.classification_result)
        assert prof_scrubbed["primary_domain"] == "general"
        assert prof_scrubbed["primary_subdomain"] == "general.general"
        assert prof_scrubbed["section_overrides"][0]["domain"] == "general"

        # Verify chunk updated
        c_final = await session.get(DocumentChunk, chunk.id)
        assert c_final.domain_key == "general"
        assert c_final.subdomain_key == "general.general"

        # Verify Qdrant payload updated
        q_final = await qdrant.retrieve(collection_name=col_name, ids=[chunk_point_id], with_payload=True)
        assert q_final[0].payload["domain"] == "general"
        assert q_final[0].payload["subdomain"] == "general.general"

        # Verify EntityDomainMapping preserved under target domain
        m_res = await session.execute(
            select(EntityDomainMapping).where(EntityDomainMapping.chunk_id == chunk.id)
        )
        mapping_final = m_res.scalars().first()
        assert mapping_final is not None
        assert mapping_final.domain_id == "general"


@pytest.mark.asyncio
async def test_lexicon_list_and_delete(in_memory_engine):
    """Verify list_lexicon_terms filtering and delete_lexicon_term helper."""
    await init_db(in_memory_engine)
    session_factory = create_session_factory(in_memory_engine)

    async with session_factory() as session:
        # Add a custom term
        term = await add_lexicon_term(
            session=session,
            term="photovoltaic",
            domain_id="engineering",
            specific_sense="Direct conversion of light into electricity.",
            contrastive_warning="Do not confuse with thermal solar.",
        )
        assert term.id is not None

        # Search term
        matched = await list_lexicon_terms(session, domain_id="engineering", search="photovoltaic")
        assert len(matched) >= 1
        assert matched[0].term == "photovoltaic"

        # Delete term
        deleted = await delete_lexicon_term(session, term.id)
        assert deleted is True

        # Verify gone
        after = await list_lexicon_terms(session, domain_id="engineering", search="photovoltaic")
        assert len(after) == 0


@pytest.mark.asyncio
async def test_sync_document_metadata_across_stores_hybrid(in_memory_engine):
    """Verify sync_document_metadata_across_stores matches heading_path against section_overrides and syncs Qdrant."""
    await init_db(in_memory_engine)
    session_factory = create_session_factory(in_memory_engine)

    qdrant = AsyncQdrantClient(":memory:")
    col_name = "test_sync_across_stores_col"
    await init_qdrant_collection(qdrant, collection=col_name, dimensions=4)

    async with session_factory() as session:
        pt1 = str(uuid.uuid4())
        pt2 = str(uuid.uuid4())

        doc = Document(
            filename="hybrid_doc.pdf",
            file_type="pdf",
            sha256="d" * 64,
            status="ready",
            primary_domain_id="engineering",
            primary_subdomain_id="eng.electrical",
            authority_tier=1,
            jurisdiction="IEEE",
            effective_year=2024,
            is_current=True,
            classification_result=json.dumps({
                "primary_domain": "engineering",
                "primary_subdomain": "eng.electrical",
                "epistemic_modality": "empirical",
                "section_overrides": [
                    {
                        "heading_path": "Financial Projections",
                        "domain": "economics",
                        "subdomain": "econ.finance",
                        "epistemic_modality": "evaluative",
                    }
                ],
            }),
        )
        session.add(doc)
        await session.flush()

        chunk1 = DocumentChunk(
            id=str(uuid.uuid4()),
            doc_id=doc.id,
            point_id=pt1,
            chunk_index=0,
            text="High-voltage wiring specs",
            heading_path="Technical Overview",
            domain_key="engineering",
            subdomain_key="eng.electrical",
            epistemic_modality="empirical",
        )
        chunk2 = DocumentChunk(
            id=str(uuid.uuid4()),
            doc_id=doc.id,
            point_id=pt2,
            chunk_index=1,
            text="Estimated capital expenditure $5M",
            heading_path="Financial Projections > CapEx",
            domain_key="engineering",
            subdomain_key="eng.electrical",
            epistemic_modality="empirical",
        )
        session.add(chunk1)
        session.add(chunk2)
        await session.commit()

        # Seed Qdrant points
        await qdrant.upsert(
            collection_name=col_name,
            points=[
                qmodels.PointStruct(id=pt1, vector={"dense": [0.1, 0.1, 0.1, 0.1]}, payload={}),
                qmodels.PointStruct(id=pt2, vector={"dense": [0.2, 0.2, 0.2, 0.2]}, payload={}),
            ],
        )

        # Call sync helper
        await sync_document_metadata_across_stores(
            session=session,
            qdrant=qdrant,
            collection_name=col_name,
            doc=doc,
        )

        # Verify chunk 1 kept engineering and chunk 2 resolved to economics override
        await session.refresh(chunk1)
        await session.refresh(chunk2)
        assert chunk1.domain_key == "engineering"
        assert chunk1.subdomain_key == "eng.electrical"

        assert chunk2.domain_key == "economics"
        assert chunk2.subdomain_key == "econ.finance"
        assert chunk2.epistemic_modality == "evaluative"

        # Verify Qdrant points updated per PointIdsList
        pts = await qdrant.retrieve(collection_name=col_name, ids=[pt1, pt2], with_payload=True)
        pt_map = {str(p.id): p.payload for p in pts}

        assert pt_map[pt1]["domain"] == "engineering"
        assert pt_map[pt1]["subdomain"] == "eng.electrical"
        assert pt_map[pt1]["epistemic_type"] == "empirical"

        assert pt_map[pt2]["domain"] == "economics"
        assert pt_map[pt2]["subdomain"] == "econ.finance"
        assert pt_map[pt2]["epistemic_type"] == "evaluative"


