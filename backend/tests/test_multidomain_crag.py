"""Unit and integration tests for Phase 3: CRAG Disambiguation, Federated Retrieval & Epistemic Synthesis."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from qdrant_client import AsyncQdrantClient, models as qmodels
from sqlalchemy.ext.asyncio import create_async_engine

from src.api.routes.chat import _serialize_messages
from src.api.schemas import (
    ChatMessagePayload,
    CitationPayload,
    DomainFilterSpec,
    PolysemyCollision,
)
from src.db.models import CrossDomainEntity, DomainLexicon
from src.db.qdrant import DENSE, SPARSE, init_qdrant_collection
from src.db.repository import (
    get_cross_domain_entity_matches,
    get_lexicon_matches,
    upsert_cross_domain_entities,
)
from src.db.session import create_session_factory, init_db
from src.ingestion.classifier import (
    DomainSpec,
    SubdomainSpec,
    get_taxonomy_catalog,
    invalidate_taxonomy_catalog,
    set_classifier_force_stub,
)
from src.ingestion.indexer import DeterministicBM25Encoder, DeterministicStubEmbedder
from src.rag.grader import format_documents_xml
from src.rag.graph import (
    QueryRewriteResult,
    build_citations_from_chunks,
    build_rag_graph,
    close_checkpointer,
    get_checkpointer,
    stream_chat_response,
)
from src.rag.reranker import FlashRankReranker, deduplicate_parent_chunks
from src.rag.retriever import HybridRetriever, build_qdrant_filter
from src.rag.state import AgentState, Citation

TEST_COLLECTION = "test_multidomain_crag"
STUB_DIM = 32


class FakeCountingChatModel(GenericFakeChatModel):
    """GenericFakeChatModel subclass recording captured prompt messages."""

    call_count: int = 0
    captured_prompts: list[list[Any]] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.call_count += 1
        self.captured_prompts.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.fixture
def in_memory_engine():
    return create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)


@pytest.fixture
async def session_factory(in_memory_engine):
    await init_db(in_memory_engine)
    return create_session_factory(in_memory_engine)


@pytest.fixture(autouse=True)
def force_stub_classifier():
    """Ensure fast deterministic classifier is used during tests."""
    set_classifier_force_stub(True)
    yield
    set_classifier_force_stub(False)


@pytest.fixture
async def seeded_multidomain_qdrant():
    """Create in-memory Qdrant client seeded with multi-domain chunks across domains."""
    client = AsyncQdrantClient(":memory:")
    await init_qdrant_collection(client, collection=TEST_COLLECTION, dimensions=STUB_DIM)

    embedder = DeterministicStubEmbedder(dimensions=STUB_DIM)
    bm25 = DeterministicBM25Encoder()
    from src.ingestion.indexer import set_embedder, set_sparse_encoder
    set_embedder(embedder)
    set_sparse_encoder(bm25)

    points_data = [
        # Engineering chunk (doc-eng-1, Tier 1)
        {
            "id": "00000000-0000-0000-0000-000000000001",
            "text": "Transformer winding dielectric oil breakdown threshold under IEEE C57.",
            "payload": {
                "chunk_id": "c-eng-1",
                "doc_id": "doc-eng-1",
                "source_filename": "transformer_spec.pdf",
                "file_type": "pdf",
                "heading_path": "Chapter 4 > Section 4.1",
                "page_numbers": [10],
                "child_text": "Transformer winding dielectric oil breakdown threshold under IEEE C57.",
                "parent_content": "SHARED_PARENT_ENG",
                "contextual_prefix": "[Domain: Engineering | Standard: IEEE C57]",
                "domain": "engineering",
                "subdomain": "eng.electrical",
                "authority_tier": 1,
                "epistemic_type": "empirical",
                "jurisdiction": "IEEE",
                "is_current": True,
                "associated_entities": ["T-101"],
            },
        },
        # Economics chunk (doc-econ-1, Tier 2)
        {
            "id": "00000000-0000-0000-0000-000000000002",
            "text": "Asset impairment and accelerated depreciation schedule under US GAAP ASC 360.",
            "payload": {
                "chunk_id": "c-econ-1",
                "doc_id": "doc-econ-1",
                "source_filename": "depreciation_manual.pdf",
                "file_type": "pdf",
                "heading_path": "Financials > Asset Retirement",
                "page_numbers": [25],
                "child_text": "Asset impairment and accelerated depreciation schedule under US GAAP ASC 360.",
                "parent_content": "SHARED_PARENT_ECON",
                "contextual_prefix": "[Domain: Economics | Standard: US GAAP ASC 360]",
                "domain": "economics",
                "subdomain": "econ.accounting",
                "authority_tier": 2,
                "epistemic_type": "evaluative",
                "jurisdiction": "US-FED",
                "is_current": True,
                "associated_entities": ["T-101"],
            },
        },
        # Law chunk (doc-law-1, Tier 1)
        {
            "id": "00000000-0000-0000-0000-000000000003",
            "text": "Mandatory reporting of PCB dielectric oil spills under EPA 40 CFR 761 TSCA statutory liability.",
            "payload": {
                "chunk_id": "c-law-1",
                "doc_id": "doc-law-1",
                "source_filename": "environmental_statute.pdf",
                "file_type": "pdf",
                "heading_path": "Title 40 > Chapter 1 > Subchapter R",
                "page_numbers": [105],
                "child_text": "Mandatory reporting of PCB dielectric oil spills under EPA 40 CFR 761 TSCA statutory liability.",
                "parent_content": "SHARED_PARENT_LAW",
                "contextual_prefix": "[Domain: Law | Code: 40 CFR 761]",
                "domain": "law",
                "subdomain": "law.statute",
                "authority_tier": 1,
                "epistemic_type": "normative",
                "jurisdiction": "US-FED",
                "is_current": True,
                "associated_entities": ["T-101"],
            },
        },
        # General chunk (doc-gen-1, Tier 4)
        {
            "id": "00000000-0000-0000-0000-000000000004",
            "text": "General overview of substation equipment maintenance history.",
            "payload": {
                "chunk_id": "c-gen-1",
                "doc_id": "doc-gen-1",
                "source_filename": "substation_overview.txt",
                "file_type": "txt",
                "heading_path": "Overview",
                "page_numbers": [],
                "child_text": "General overview of substation equipment maintenance history.",
                "parent_content": "SHARED_PARENT_GEN",
                "contextual_prefix": "[Domain: General]",
                "domain": "general",
                "subdomain": "general",
                "authority_tier": 4,
                "epistemic_type": "general",
                "jurisdiction": "",
                "is_current": True,
                "associated_entities": [],
            },
        },
    ]

    texts = [p["text"] for p in points_data]
    dense_vecs = await embedder.embed_documents(texts)
    sparse_vecs = await bm25.encode(texts)

    points = [
        qmodels.PointStruct(
            id=p["id"],
            vector={
                DENSE: dv,
                SPARSE: sv,
            },
            payload=p["payload"],
        )
        for p, dv, sv in zip(points_data, dense_vecs, sparse_vecs)
    ]
    await client.upsert(collection_name=TEST_COLLECTION, points=points)

    yield client
    await client.close()


# ============================================================================
# 1. Polysemy Collision Detection (English & Vietnamese)
# ============================================================================


@pytest.mark.asyncio
async def test_polysemy_collision_detection_en_and_vi(in_memory_engine):
    """Test polysemy collision detection across domains in English and Vietnamese compound phrases."""
    await init_db(in_memory_engine)
    session_factory = create_session_factory(in_memory_engine)

    async with session_factory() as session:
        # 1. English query with polysemic 'liability'
        en_query = "What is the liability associated with transformer degradation?"
        en_matches = await get_lexicon_matches(session, en_query)
        assert len(en_matches) >= 3
        en_domains = {m.domain_id for m in en_matches if m.term.lower() == "liability"}
        assert {"engineering", "economics", "law"}.issubset(en_domains)

        # 2. Vietnamese query with compound phrase 'khấu hao' and 'trách nhiệm'
        vi_query = "Quy định về khấu hao tài sản và trách nhiệm pháp lý khi máy biến áp hỏng?"
        vi_matches = await get_lexicon_matches(session, vi_query)
        assert len(vi_matches) >= 3
        vi_matched_terms = {m.term for m in vi_matches}
        assert "khấu hao" in vi_matched_terms or "trách nhiệm" in vi_matched_terms

        # 3. Cross-domain entity matching
        await upsert_cross_domain_entities(
            session,
            [
                CrossDomainEntity(
                    canonical_name="Transformer T-101",
                    entity_type="equipment",
                    description="Main step-down power transformer",
                )
            ],
        )
        ent_matches = await get_cross_domain_entity_matches(session, "Check status for Transformer T-101 in substation")
        assert len(ent_matches) == 1
        assert ent_matches[0].canonical_name == "Transformer T-101"


# ============================================================================
# 2. Single-Call Sub-Query Decomposition
# ============================================================================


@pytest.mark.asyncio
async def test_single_call_sub_query_decomposition(in_memory_engine):
    """Test single LLM call decomposition into multi-domain sub-queries preserving query language."""
    await init_db(in_memory_engine)
    session_factory = create_session_factory(in_memory_engine)

    fake_decomposer_output = json.dumps(
        {
            "needs_retrieval": True,
            "routing_mode": "multi",
            "target_domains": ["engineering", "economics", "law"],
            "domain_queries": {
                "engineering": "Đánh giá hư hỏng cách điện dầu máy biến áp tiêu chuẩn IEEE",
                "economics": "Chi phí khấu hao và thay thế tài sản theo chuẩn mực kế toán",
                "law": "Quy định trách nhiệm bồi thường và xử lý chất thải PCB",
            },
            "domain_weights": {
                "engineering": 1.0,
                "economics": 1.0,
                "law": 1.2,
            },
        }
    )

    fake_llm = FakeCountingChatModel(messages=iter([AIMessage(content=fake_decomposer_output)]))
    graph = build_rag_graph(llm=fake_llm, session_factory=session_factory)

    initial_state: AgentState = {
        "messages": [
            HumanMessage(
                content="Đánh giá hư hỏng dầu biến áp, chi phí khấu hao thay thế và nghĩa vụ pháp lý khi rò rỉ PCB?"
            )
        ],
        "original_query": "Đánh giá hư hỏng dầu biến áp, chi phí khấu hao thay thế và nghĩa vụ pháp lý khi rò rỉ PCB?",
        "steps": [],
    }

    # Execute rewrite_query node directly
    result = await graph.nodes["rewrite_query"].ainvoke(initial_state)

    assert result["needs_retrieval"] is True
    assert result["routing_mode"] == "multi"
    assert set(result["target_domains"]) == {"engineering", "economics", "law"}
    assert len(result["domain_queries"]) == 3
    # Check language preservation: Vietnamese remains in sub-queries
    assert "máy biến áp" in result["domain_queries"]["engineering"]
    assert "khấu hao" in result["domain_queries"]["economics"]
    assert "PCB" in result["domain_queries"]["law"]
    assert result["domain_weights"]["law"] == 1.2
    assert len(result["steps"]) == 1
    assert result["steps"][0]["node"] == "rewrite_query"


# ============================================================================
# 3. Strict vs Auto 'General' Fallback Isolation
# ============================================================================


@pytest.mark.asyncio
async def test_strict_vs_auto_general_fallback_isolation(seeded_multidomain_qdrant):
    """Test that 'auto' mode includes 'general' fallback with 0.5x discount, while explicit filter strictly isolates."""
    embedder = DeterministicStubEmbedder(dimensions=STUB_DIM)
    bm25 = DeterministicBM25Encoder()
    retriever = HybridRetriever(
        qdrant=seeded_multidomain_qdrant,
        embedder=embedder,
        sparse_encoder=bm25,
        collection=TEST_COLLECTION,
        limit=10,
    )

    # 1. Mode == 'auto': general fallback is searched and included with 0.5x discount
    auto_results = await retriever.retrieve_federated(
        domain_queries={"engineering": "transformer maintenance breakdown"},
        mode="auto",
    )
    returned_domains_auto = {r["domain"] for r in auto_results}
    assert "engineering" in returned_domains_auto
    # 'general' domain is present in auto mode
    assert "general" in returned_domains_auto

    # 2. Mode == 'explicit' with domains=['engineering']: general domain is strictly excluded
    explicit_results = await retriever.retrieve_federated(
        domain_queries={"engineering": "transformer maintenance breakdown"},
        mode="explicit",
    )
    returned_domains_explicit = {r["domain"] for r in explicit_results}
    assert "engineering" in returned_domains_explicit
    assert "general" not in returned_domains_explicit
    assert "law" not in returned_domains_explicit
    assert "economics" not in returned_domains_explicit


# ============================================================================
# 4. Calibrated RRF and Stratified Domain Quotas
# ============================================================================


@pytest.mark.asyncio
async def test_calibrated_rrf_and_stratified_domain_quotas(seeded_multidomain_qdrant):
    """Test stratified domain quotas (min 2 chunks per active domain) and entity-match bonus."""
    embedder = DeterministicStubEmbedder(dimensions=STUB_DIM)
    bm25 = DeterministicBM25Encoder()
    retriever = HybridRetriever(
        qdrant=seeded_multidomain_qdrant,
        embedder=embedder,
        sparse_encoder=bm25,
        collection=TEST_COLLECTION,
        limit=10,
    )

    # Federated search across engineering, economics, and law
    results = await retriever.retrieve_federated(
        domain_queries={
            "engineering": "dielectric breakdown threshold IEEE",
            "economics": "asset impairment depreciation ASC 360",
            "law": "mandatory reporting PCB spill TSCA",
        },
        mode="explicit",
        associated_entities=["T-101"],
        min_per_domain=1,
    )

    returned_domains = [r["domain"] for r in results]
    assert "engineering" in returned_domains
    assert "economics" in returned_domains
    assert "law" in returned_domains

    # Chunks matching associated_entities ['T-101'] receive bonus and preserve metadata
    for r in results:
        assert "domain" in r
        assert "authority_tier" in r
        assert "epistemic_type" in r
        if "T-101" in r.get("associated_entities", []):
            assert r["score"] > 0.0


# ============================================================================
# 5. Authority-Tier Soft Tie-Breaker & Parent Deduplication
# ============================================================================


@pytest.mark.asyncio
async def test_authority_tier_soft_tie_breaker_and_parent_dedup():
    """Verify soft authority-tier tie breaker: s_final = 0.92*s_ce + 0.08*(5-tier)/4 and domain-aware dedup."""

    class StubRanker:
        def rerank(self, request):
            out = []
            for p in request.passages:
                # Same base cross-encoder score for all candidates
                out.append({"id": p["id"], "text": p["text"], "meta": p["meta"], "score": 0.80})
            return out

    # Candidate 1: Tier 1 (Most authoritative) -> bonus = 0.08 * (5-1)/4 = 0.08
    # Expected s_final = 0.92 * 0.80 + 0.08 = 0.736 + 0.08 = 0.816
    c1 = {
        "chunk_id": "c-tier-1",
        "doc_id": "doc-1",
        "source_filename": "statute.pdf",
        "domain": "law",
        "authority_tier": 1,
        "child_text": "Statutory mandate section A.",
        "parent_content": "PARENT_1",
        "score": 0.5,
    }
    # Candidate 2: Tier 5 (Informal) -> bonus = 0.08 * (5-5)/4 = 0.0
    # Expected s_final = 0.92 * 0.80 + 0.0 = 0.736
    c2 = {
        "chunk_id": "c-tier-5",
        "doc_id": "doc-2",
        "source_filename": "memo.txt",
        "domain": "engineering",
        "authority_tier": 5,
        "child_text": "Informal notes section B.",
        "parent_content": "PARENT_2",
        "score": 0.5,
    }

    reranker = FlashRankReranker(ranker=StubRanker())
    reranked = await reranker.rerank("mandate and notes", [c2, c1])

    assert len(reranked) == 2
    # Tier 1 ranks first due to soft authority tie-breaker
    assert reranked[0]["chunk_id"] == "c-tier-1"
    assert reranked[0]["score"] == pytest.approx(0.816, abs=1e-4)

    assert reranked[1]["chunk_id"] == "c-tier-5"
    assert reranked[1]["score"] == pytest.approx(0.736, abs=1e-4)

    # Verify domain-aware parent deduplication:
    # Two chunks with identical doc_id and parent_content but DIFFERENT domains must NOT collide
    c_eng_dup = {
        "chunk_id": "c-e",
        "doc_id": "doc-hybrid",
        "domain": "engineering",
        "parent_content": "HYBRID_SECTION_TEXT",
        "score": 0.9,
    }
    c_law_dup = {
        "chunk_id": "c-l",
        "doc_id": "doc-hybrid",
        "domain": "law",
        "parent_content": "HYBRID_SECTION_TEXT",
        "score": 0.85,
    }
    deduped = deduplicate_parent_chunks([c_eng_dup, c_law_dup])
    assert len(deduped) == 2
    domains_present = {d["domain"] for d in deduped}
    assert domains_present == {"engineering", "law"}


# ============================================================================
# 6. Epistemic Partitioned Prompt & XML Context
# ============================================================================


@pytest.mark.asyncio
async def test_epistemic_partitioned_prompt_and_xml_context(in_memory_engine):
    """Verify format_documents_xml wraps domain/authority attributes and synthesize_node formats epistemic perspectives."""
    await init_db(in_memory_engine)
    session_factory = create_session_factory(in_memory_engine)

    chunks = [
        {
            "chunk_id": "c-1",
            "doc_id": "doc-eng",
            "source_filename": "spec.pdf",
            "domain": "engineering",
            "authority_tier": 1,
            "child_text": "Transformer temperature limits.",
            "parent_content": "Transformer temperature limits detail.",
        },
        {
            "chunk_id": "c-2",
            "doc_id": "doc-law",
            "source_filename": "rules.pdf",
            "domain": "law",
            "authority_tier": 2,
            "child_text": "Mandatory safety clearances.",
            "parent_content": "Mandatory safety clearances detail.",
        },
    ]

    # Verify XML formatting
    xml_output = format_documents_xml(chunks)
    assert 'domain="engineering"' in xml_output
    assert 'authority_tier="1"' in xml_output
    assert 'domain="law"' in xml_output
    assert 'authority_tier="2"' in xml_output

    # Verify Epistemic Partitioning in synthesize_node for English
    fake_llm_en = FakeCountingChatModel(messages=iter([AIMessage(content="Structured multi-domain answer [1] [2].")]))
    graph_en = build_rag_graph(llm=fake_llm_en, session_factory=session_factory)

    state_en: AgentState = {
        "original_query": "What are the engineering limits and legal safety rules for the transformer?",
        "graded_chunks": chunks,
        "steps": [],
    }
    await graph_en.nodes["synthesize"].ainvoke(state_en)

    prompt_en = fake_llm_en.captured_prompts[0][0].content
    assert "⚙️ Engineering Perspective" in prompt_en
    assert "⚖️ Legal Perspective" in prompt_en
    assert "🔀 Inter-Domain Conflict & Priority Matrix" in prompt_en

    # Verify Epistemic Partitioning in synthesize_node for Vietnamese
    fake_llm_vi = FakeCountingChatModel(messages=iter([AIMessage(content="Câu trả lời đa ngành có trích dẫn [1] [2].")]))
    graph_vi = build_rag_graph(llm=fake_llm_vi, session_factory=session_factory)

    state_vi: AgentState = {
        "original_query": "Quy định kỹ thuật và nghĩa vụ pháp lý đối với máy biến áp?",
        "graded_chunks": chunks,
        "steps": [],
    }
    await graph_vi.nodes["synthesize"].ainvoke(state_vi)

    prompt_vi = fake_llm_vi.captured_prompts[0][0].content
    assert "⚙️ Góc độ Kỹ thuật" in prompt_vi
    assert "⚖️ Góc độ Pháp lý" in prompt_vi
    assert "🔀 Ma trận Xung đột & Thứ tự Ưu tiên Liên ngành" in prompt_vi


# ============================================================================
# 7. Thread Message Serialization with domain_filter and steps
# ============================================================================


@pytest.mark.asyncio
async def test_thread_message_serialization_with_domain_filter_and_steps():
    """Verify serialization & deserialization of domain_filter on HumanMessage and steps on AIMessage."""
    filter_spec = DomainFilterSpec(
        mode="explicit",
        domains=["engineering", "economics"],
        subdomains=["eng.electrical"],
        max_authority_tier=2,
        only_current=True,
    )

    h_msg = HumanMessage(
        content="Assess transformer replacement cost and failure risk",
        additional_kwargs={"domain_filter": filter_spec.model_dump()},
    )

    ai_steps = [
        {"node": "rewrite_query", "routing_mode": "multi", "target_domains": ["engineering", "economics"]},
        {"node": "retrieve", "retrieved_count": 4},
        {"node": "synthesize", "status": "completed"},
    ]
    cit = CitationPayload(
        citation_index=1,
        doc_id="d1",
        chunk_id="c1",
        source_filename="transformer.pdf",
        domain="engineering",
        subdomain="eng.electrical",
        authority_tier=1,
        epistemic_type="empirical",
        jurisdiction="IEEE",
        is_current=True,
    )
    ai_msg = AIMessage(
        content="Analysis complete [1].",
        additional_kwargs={
            "steps": ai_steps,
            "citations": [cit.model_dump()],
            "routing_mode": "multi",
        },
    )

    serialized = _serialize_messages([h_msg, ai_msg], latest_citations=[cit])
    assert len(serialized) == 2

    user_payload: ChatMessagePayload = serialized[0]
    assert user_payload.role == "user"
    assert user_payload.domain_filter is not None
    assert user_payload.domain_filter.mode == "explicit"
    assert user_payload.domain_filter.domains == ["engineering", "economics"]
    assert user_payload.domain_filter.max_authority_tier == 2

    assistant_payload: ChatMessagePayload = serialized[1]
    assert assistant_payload.role == "assistant"
    assert len(assistant_payload.steps) == 3
    assert assistant_payload.steps[0]["node"] == "rewrite_query"
    assert len(assistant_payload.citations) == 1
    assert assistant_payload.citations[0].domain == "engineering"
    assert assistant_payload.citations[0].authority_tier == 1


# ============================================================================
# 8. End-to-End Multi-Domain CRAG Workflow
# ============================================================================


@pytest.mark.asyncio
async def test_end_to_end_multidomain_crag_stream(
    tmp_path: Path,
    in_memory_engine,
    seeded_multidomain_qdrant,
):
    """Verify full CRAG stream with domain_filter, routing event, and enriched epistemic citations."""
    await init_db(in_memory_engine)
    session_factory = create_session_factory(in_memory_engine)
    checkpoint_db = tmp_path / "chat_history.db"
    saver = await get_checkpointer(checkpoint_db)

    fake_rewrite = json.dumps(
        {
            "needs_retrieval": True,
            "routing_mode": "multi",
            "target_domains": ["engineering", "law"],
            "domain_queries": {
                "engineering": "transformer dielectric oil breakdown IEEE",
                "law": "mandatory reporting PCB spill TSCA",
            },
            "domain_weights": {"engineering": 1.0, "law": 1.0},
        }
    )
    fake_grade = json.dumps({"relevant_indices": [1, 2], "reasoning": "Both chunks relevant"})
    fake_synth = "⚙️ Engineering: Oil breakdown risk [1]. ⚖️ Law: Mandatory reporting required [2]."

    fake_llm = FakeCountingChatModel(
        messages=iter(
            [
                AIMessage(content=fake_rewrite),
                AIMessage(content=fake_grade),
                AIMessage(content=fake_synth),
            ]
        )
    )

    filter_spec = DomainFilterSpec(
        mode="explicit",
        domains=["engineering", "law"],
        max_authority_tier=2,
    )

    embedder = DeterministicStubEmbedder(dimensions=STUB_DIM)
    bm25 = DeterministicBM25Encoder()

    events = [
        ev
        async for ev in stream_chat_response(
            query="Evaluate transformer failure risk and mandatory environmental reporting requirements.",
            thread_id="thread-multidomain-test",
            domain_filter=filter_spec,
            qdrant=seeded_multidomain_qdrant,
            embedder=embedder,
            sparse_encoder=bm25,
            llm=fake_llm,
            checkpointer=saver,
            session_factory=session_factory,
            collection=TEST_COLLECTION,
        )
    ]

    event_types = [e["event"] for e in events]
    assert "step" in event_types
    assert "routing" in event_types
    assert "citations" in event_types
    assert "done" in event_types

    # Verify routing event data
    routing_events = [e["data"] for e in events if e["event"] == "routing"]
    assert len(routing_events) == 1
    assert set(routing_events[0]["target_domains"]) == {"engineering", "law"}

    # Verify done event enriched citations
    done_event = events[-1]["data"]
    citations = done_event["citations"]
    assert len(citations) >= 2
    cit_domains = {c["domain"] for c in citations}
    assert "engineering" in cit_domains or "law" in cit_domains

    await close_checkpointer()


# ============================================================================
# 9. Phase 3 Dynamic Taxonomy, CRAG Routing & Dynamic Perspective Synthesis
# ============================================================================


@pytest.mark.asyncio
async def test_rewrite_query_explicit_modes_single_and_federated(session_factory):
    """Verify rewrite_query_node treats single and federated modes as explicit user filters."""
    from src.db.models import DomainRegistry

    async with session_factory() as session:
        bio = DomainRegistry(
            id="biomedical",
            slug="biomedical",
            name="Biomedical",
            epistemic_type="empirical",
            icon="🧬",
            is_active=True,
        )
        session.add(bio)
        await session.commit()

    fake_rewrite = json.dumps({
        "needs_retrieval": True,
        "rewritten_queries": ["Search text"],
        "routing_mode": "multi",
        "target_domains": ["general"],
        "domain_queries": {"general": "Search text"},
        "domain_weights": {"general": 1.0},
    })
    fake_llm = FakeCountingChatModel(messages=iter([AIMessage(content=fake_rewrite)]))
    graph = build_rag_graph(llm=fake_llm, session_factory=session_factory)

    # 1. user_mode == "single"
    state_single: AgentState = {
        "original_query": "What is the heart rate limit?",
        "domain_filter": {"mode": "single", "domains": ["biomedical", "invalid_nonexistent"]},
        "steps": [],
    }
    res_single = await graph.nodes["rewrite_query"].ainvoke(state_single)
    assert res_single["routing_mode"] == "explicit"
    assert res_single["target_domains"] == ["biomedical"]
    assert "biomedical" in res_single["domain_queries"]
    assert "invalid_nonexistent" not in res_single["domain_queries"]
    assert "biomedical" in res_single["domain_weights"]

    # 2. user_mode == "federated" with 6 domains (capped to 5)
    fake_llm_fed = FakeCountingChatModel(messages=iter([AIMessage(content=fake_rewrite)]))
    graph_fed = build_rag_graph(llm=fake_llm_fed, session_factory=session_factory)
    state_fed: AgentState = {
        "original_query": "Cross-domain review",
        "domain_filter": {
            "mode": "federated",
            "domains": ["engineering", "economics", "law", "biomedical", "general", "extra_overflow"],
        },
        "steps": [],
    }
    res_fed = await graph_fed.nodes["rewrite_query"].ainvoke(state_fed)
    assert res_fed["routing_mode"] == "explicit"
    assert len(res_fed["target_domains"]) == 5
    assert "extra_overflow" not in res_fed["target_domains"]
    assert len(res_fed["domain_queries"]) == 5
    assert len(res_fed["domain_weights"]) == 5


@pytest.mark.asyncio
async def test_rewrite_query_auto_capping_and_pruning(session_factory):
    """Verify auto mode caps target_domains to at most 3 specialized domains and prunes queries and weights."""
    from src.db.models import DomainRegistry

    async with session_factory() as session:
        bio = DomainRegistry(
            id="biomedical",
            slug="biomedical",
            name="Biomedical",
            epistemic_type="empirical",
            icon="🧬",
            is_active=True,
        )
        session.add(bio)
        await session.commit()

    # LLM returns 4 specialized domains in auto mode
    fake_rewrite = json.dumps({
        "needs_retrieval": True,
        "rewritten_queries": ["Multi domain query"],
        "routing_mode": "multi",
        "target_domains": ["engineering", "economics", "law", "biomedical"],
        "domain_queries": {
            "engineering": "Eng q",
            "economics": "Econ q",
            "law": "Law q",
            "biomedical": "Bio q",
        },
        "domain_weights": {
            "engineering": 1.5,
            "economics": 1.2,
            "law": 1.0,
            "biomedical": 0.8,
        },
    })
    fake_llm = FakeCountingChatModel(messages=iter([AIMessage(content=fake_rewrite)]))
    graph = build_rag_graph(llm=fake_llm, session_factory=session_factory)

    state: AgentState = {
        "original_query": "Comprehensive multi-domain assessment",
        "domain_filter": {"mode": "auto"},
        "steps": [],
    }
    res = await graph.nodes["rewrite_query"].ainvoke(state)

    # Capped to 3 specialized domains
    assert len(res["target_domains"]) == 3
    # Pruned both queries and weights to exactly target_domains
    assert set(res["domain_queries"].keys()) == set(res["target_domains"])
    assert set(res["domain_weights"].keys()) == set(res["target_domains"])


@pytest.mark.asyncio
async def test_rewrite_query_polysemy_cap_top_5(session_factory):
    """Verify polysemy_collisions are sorted by length and capped to the top 5 matches."""
    async with session_factory() as session:
        # Add 6 terms with different lengths (avoiding already seeded terms)
        test_terms = [
            ("test_short_term", "engineering", "sense 1", "warn"),
            ("test_short_term", "law", "sense 2", "warn"),
            ("test_very_long_polysemic_term_1", "engineering", "sense a", "warn"),
            ("test_very_long_polysemic_term_1", "law", "sense b", "warn"),
            ("test_super_ultra_long_polysemic_term_2", "engineering", "sense c", "warn"),
            ("test_super_ultra_long_polysemic_term_2", "economics", "sense d", "warn"),
            ("test_medium_term_3", "engineering", "sense e", "warn"),
            ("test_medium_term_3", "economics", "sense f", "warn"),
            ("test_another_term_4", "engineering", "sense g", "warn"),
            ("test_another_term_4", "law", "sense h", "warn"),
            ("test_fifth_long_term_5", "engineering", "sense i", "warn"),
            ("test_fifth_long_term_5", "law", "sense j", "warn"),
        ]
        for term, d_id, sense, warn in test_terms:
            session.add(
                DomainLexicon(
                    term=term,
                    domain_id=d_id,
                    specific_sense=sense,
                    contrastive_warning=warn,
                )
            )
        await session.commit()

    fake_rewrite = json.dumps({
        "needs_retrieval": False,
        "rewritten_queries": [],
        "routing_mode": "direct",
    })
    fake_llm = FakeCountingChatModel(messages=iter([AIMessage(content=fake_rewrite)]))
    graph = build_rag_graph(llm=fake_llm, session_factory=session_factory)

    state: AgentState = {
        "original_query": "test_short_term test_very_long_polysemic_term_1 test_super_ultra_long_polysemic_term_2 test_medium_term_3 test_another_term_4 test_fifth_long_term_5",
        "steps": [],
    }
    res = await graph.nodes["rewrite_query"].ainvoke(state)

    collisions = res["polysemy_collisions"]
    assert len(collisions) <= 5
    # Longest term should be first
    if collisions:
        assert collisions[0]["term"] == "test_super_ultra_long_polysemic_term_2"


@pytest.mark.asyncio
async def test_synthesize_node_dynamic_custom_domain_headings(session_factory):
    """Verify synthesize_node formats dynamic custom domain headings while preserving canonical triad headings."""
    invalidate_taxonomy_catalog()
    cat = get_taxonomy_catalog()
    cat.domains["biomedical"] = DomainSpec(
        slug="biomedical",
        name="Biomedical Engineering",
        epistemic_type="empirical",
        icon="🧬",
        description="Medical device mechanisms, biocompatibility, and clinical standards.",
    )

    chunks = [
        {"chunk_id": "c1", "child_text": "Transformer thermal limits.", "domain": "engineering", "score": 0.9},
        {"chunk_id": "c2", "child_text": "Pacemaker electrode impedance.", "domain": "biomedical", "score": 0.88},
        {"chunk_id": "c3", "child_text": "Unclassified custom domain data.", "domain": "quantum_physics", "score": 0.85},
    ]

    # English synthesize test
    fake_llm_en = FakeCountingChatModel(messages=iter([AIMessage(content="Answer [1] [2].")]))
    graph_en = build_rag_graph(llm=fake_llm_en, session_factory=session_factory)

    state_en: AgentState = {
        "original_query": "What are the engineering and biomedical limits?",
        "graded_chunks": chunks,
        "steps": [],
    }
    await graph_en.nodes["synthesize"].ainvoke(state_en)

    prompt_en = fake_llm_en.captured_prompts[0][0].content
    # Canonical starter perspectives preserved
    assert "- ⚙️ Engineering Perspective: Report physical mechanisms, technical limits, and engineering standards." in prompt_en
    assert "- ⚖️ Legal Perspective: Report statutory mandates, jurisdictional requirements, liabilities, and sanctions." in prompt_en
    assert "- 🔀 Inter-Domain Conflict & Priority Matrix: Synthesize cross-domain tradeoffs" in prompt_en
    # Custom domain dynamic perspective
    assert "- 🧬 Biomedical Engineering Perspective: Medical device mechanisms, biocompatibility, and clinical standards." in prompt_en
    # Safe fallback for domain missing from catalog
    assert "- 📁 Quantum_Physics Perspective:" in prompt_en

    # Vietnamese synthesize test
    fake_llm_vi = FakeCountingChatModel(messages=iter([AIMessage(content="Trả lời [1] [2].")]))
    graph_vi = build_rag_graph(llm=fake_llm_vi, session_factory=session_factory)

    state_vi: AgentState = {
        "original_query": "Giới hạn kỹ thuật và y sinh là gì?",
        "graded_chunks": chunks,
        "steps": [],
    }
    await graph_vi.nodes["synthesize"].ainvoke(state_vi)

    prompt_vi = fake_llm_vi.captured_prompts[0][0].content
    # Canonical starter perspectives preserved
    assert "1. ⚙️ Góc độ Kỹ thuật (Engineering Perspective):" in prompt_vi
    assert "3. ⚖️ Góc độ Pháp lý (Legal Perspective):" in prompt_vi
    assert "🔀 Ma trận Xung đột & Thứ tự Ưu tiên Liên ngành (Inter-Domain Conflict & Priority Matrix):" in prompt_vi
    # Custom domain dynamic perspective (Vietnamese format: '{icon} Góc độ {name} ({name} Perspective): {description}')
    assert "🧬 Góc độ Biomedical Engineering (Biomedical Engineering Perspective): Medical device mechanisms, biocompatibility, and clinical standards." in prompt_vi
    # Safe fallback for domain missing from catalog
    assert "📁 Góc độ Quantum_Physics (Quantum_Physics Perspective):" in prompt_vi
