"""End-to-end and component tests for Phase 3 Engineering CAD CRAG Retrieval & Tag-Stem Pinning."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from qdrant_client import AsyncQdrantClient, models as qmodels

from src.config import get_settings
from src.db.models import Document
from src.db.qdrant import DENSE, SPARSE, init_qdrant_collection
from src.db.session import close_engine, create_engine, create_session_factory, init_db
from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.classifier import RecommendedDomainProfile, set_classifier_force_stub
from src.ingestion.entity_extractor import EntityExtractor
from src.ingestion.indexer import (
    DeterministicBM25Encoder,
    DeterministicStubEmbedder,
    HybridIndexer,
    set_embedder,
    set_sparse_encoder,
)
from src.ingestion.parser import DocumentParserService
from src.rag.graph import (
    _detect_engineering_query_intent,
    build_rag_graph,
    close_checkpointer,
    stream_chat_response,
)
from src.rag.llm import set_chat_model
from src.rag.reranker import FlashRankReranker, set_reranker
from src.rag.retriever import HybridRetriever, set_active_qdrant_client
from src.rag.state import AgentState

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "engineering"
TEST_COLLECTION = "test_engineering_crag"
STUB_DIM = 32


class FakeCountingChatModel(GenericFakeChatModel):
    """GenericFakeChatModel subclass recording captured prompt messages."""

    call_count: int = 0
    captured_prompts: list[list[Any]] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.call_count += 1
        self.captured_prompts.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class LowCrossEncoderStubRanker:
    """Simulates MS-MARCO giving low cross-encoder scores to terse tag/legend chunks."""

    def __init__(self, *, high_score_marker: str = "PROSE_DISTRACTOR", low_score: float = 0.05) -> None:
        self.high_score_marker = high_score_marker
        self.low_score = low_score

    def rerank(self, request):
        out = []
        for p in request.passages:
            score = 0.80 if self.high_score_marker in p["text"] else self.low_score
            out.append({"id": p["id"], "text": p["text"], "meta": p["meta"], "score": score})
        return out


@pytest.fixture(autouse=True)
def isolate_test_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate storage directories and reset singletons per test."""
    data_dir = tmp_path / "data"
    db_dir = tmp_path / "data" / "db"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("DB_DIR", str(db_dir))
    get_settings.cache_clear()
    set_classifier_force_stub(True)
    yield
    set_classifier_force_stub(False)
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
async def reset_singletons():
    set_chat_model(None)
    set_embedder(None)
    set_sparse_encoder(None)
    set_reranker(None)
    set_active_qdrant_client(None)
    await close_checkpointer()
    await close_engine()
    yield
    set_chat_model(None)
    set_embedder(None)
    set_sparse_encoder(None)
    set_reranker(None)
    set_active_qdrant_client(None)
    await close_checkpointer()
    await close_engine()


@pytest.fixture
async def seeded_engineering_corpus(tmp_path: Path):
    """Parse, chunk, and index all 4 compact engineering CAD fixtures into in-memory Qdrant + SQLite."""
    client = AsyncQdrantClient(location=":memory:")
    embedder = DeterministicStubEmbedder(dimensions=STUB_DIM)
    sparse = DeterministicBM25Encoder()
    set_embedder(embedder)
    set_sparse_encoder(sparse)
    set_active_qdrant_client(client)

    db_file = tmp_path / "db" / "eng_test.db"
    db_file.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite+aiosqlite:///{db_file.as_posix()}")
    await init_db(engine)
    session_factory = create_session_factory(engine)

    parser = DocumentParserService(data_dir=tmp_path / "parsed_cache")
    chunker = HierarchicalChunker()
    indexer = HybridIndexer(
        qdrant=client,
        embedder=embedder,
        sparse_encoder=sparse,
        collection=TEST_COLLECTION,
    )

    eng_profile = RecommendedDomainProfile(
        primary_domain="engineering",
        primary_subdomain="eng.control",
        epistemic_modality="empirical",
        authority_tier=1,
        confidence=0.95,
    )

    fixture_files = [
        ("doc-wiring", "cabinet_wiring_p2.pdf"),
        ("doc-ce", "process_ce_chart_p3_p4.pdf"),
        ("doc-esds", "esds_logic_p8_p51.pdf"),
        ("doc-lv-swgr", "lv_swgr_logic_p3.pdf"),
    ]

    async with session_factory() as session:
        for doc_id, fname in fixture_files:
            pdf_path = FIXTURE_DIR / fname
            assert pdf_path.is_file(), f"Missing engineering fixture: {pdf_path}"
            parse_res = parser.parse(pdf_path, filename=fname, force_reparse=True)
            assert parse_res.extra_meta.get("engine") == "engineering_cad"

            chunks = chunker.chunk(parse_res, domain_profile=eng_profile)
            assert len(chunks) >= 1

            doc_row = Document(
                id=doc_id,
                filename=fname,
                file_type="pdf",
                sha256=parse_res.sha256,
                status="ready",
                parsing_engine="engineering_cad",
                primary_domain_id="engineering",
                primary_subdomain_id="eng.control",
                authority_tier=1,
                is_current=True,
            )
            session.add(doc_row)
            await session.commit()

            indexed = await indexer.index_chunks(session, doc_id=doc_id, chunks=chunks)
            assert indexed == len(chunks)

    yield {
        "qdrant": client,
        "session_factory": session_factory,
        "embedder": embedder,
        "sparse": sparse,
        "collection": TEST_COLLECTION,
    }

    await engine.dispose()
    await client.close()


# ============================================================================
# 1. Intent Detection & Query Tag-Stem Extraction
# ============================================================================


def test_engineering_query_intent_detection():
    tags = EntityExtractor().extract_query_tags_and_stems("Check tt-10205a")
    assert "TT-10205A" in tags
    assert "LOOP-10205A" in tags

    assert _detect_engineering_query_intent("What causes actuate XV-10101A in C&E matrix?", ["XV-10101A"]) == (
        "matrix_trace",
        12,
    )
    assert _detect_engineering_query_intent("List all terminals for cabinet FCS0201", ["FCS0201"]) == (
        "enumeration",
        12,
    )
    assert _detect_engineering_query_intent("Inspect TT-10205A", ["TT-10205A"]) == ("tag_lookup", 12)
    # "because" must not falsely trigger "cause"
    assert _detect_engineering_query_intent("Why did revenue decline because of higher costs?", []) == ("prose", None)


# ============================================================================
# 2. Wiring Lookup & Exact-Tag Pre-Fetch (cabinet_wiring_p2.pdf)
# ============================================================================


@pytest.mark.asyncio
async def test_wiring_lookup_via_crag_nodes_and_exact_prefetch(seeded_engineering_corpus):
    qdrant = seeded_engineering_corpus["qdrant"]
    session_factory = seeded_engineering_corpus["session_factory"]
    embedder = seeded_engineering_corpus["embedder"]
    sparse = seeded_engineering_corpus["sparse"]

    retriever = HybridRetriever(
        qdrant=qdrant,
        embedder=embedder,
        sparse_encoder=sparse,
        collection=TEST_COLLECTION,
        limit=6,
    )

    fake_rewrite = json.dumps(
        {
            "needs_retrieval": True,
            "routing_mode": "single",
            "target_domains": ["engineering"],
            "domain_queries": {"engineering": "TT-10205A DCS tag terminal block FCS node IO module"},
            "domain_weights": {"engineering": 1.0},
        }
    )
    fake_llm = FakeCountingChatModel(messages=iter([AIMessage(content=fake_rewrite)]))
    reranker = FlashRankReranker(top_n=6, ranker=LowCrossEncoderStubRanker(low_score=0.08))

    graph = build_rag_graph(
        retriever=retriever,
        reranker=reranker,
        llm=fake_llm,
        session_factory=session_factory,
        collection=TEST_COLLECTION,
    )

    query = "What DCS tag, terminal block, FCS node, and IO module correspond to tt-10205a?"
    rw = await graph.nodes["rewrite_query"].ainvoke({"original_query": query, "steps": []})
    assert "TT-10205A" in rw["matched_entities"]
    assert rw["query_intent"] == "enumeration"
    assert rw["engineering_top_n"] == 12

    ret = await graph.nodes["retrieve"].ainvoke(rw)
    candidates = ret["retrieved_chunks"]
    assert any(
        c.get("exact_tag_match") is True
        and "TT-10205A" in c["parent_content"]
        and "TI10205A" in c["parent_content"]
        and "1FTB1" in c["parent_content"]
        and "FCS0201" in c["parent_content"]
        and "AAI143-H" in c["parent_content"]
        for c in candidates
    )

    rr = await graph.nodes["rerank"].ainvoke({**rw, **ret})
    reranked = rr["reranked_chunks"]
    assert len(reranked) >= 1
    assert any("TI10205A" in c["parent_content"] and "1FTB1" in c["parent_content"] for c in reranked)
    assert any(c.get("exact_tag_match") and c["score"] >= 0.85 for c in reranked)


# ============================================================================
# 3. Forward & Reverse C&E Matrix + Legend Context Fetch (process_ce_chart_p3_p4.pdf)
# ============================================================================


@pytest.mark.asyncio
async def test_forward_and_reverse_ce_matrix_with_legend_fetch(seeded_engineering_corpus):
    qdrant = seeded_engineering_corpus["qdrant"]
    session_factory = seeded_engineering_corpus["session_factory"]
    embedder = seeded_engineering_corpus["embedder"]
    sparse = seeded_engineering_corpus["sparse"]

    retriever = HybridRetriever(
        qdrant=qdrant,
        embedder=embedder,
        sparse_encoder=sparse,
        collection=TEST_COLLECTION,
        limit=6,
    )
    reranker = FlashRankReranker(top_n=6, ranker=LowCrossEncoderStubRanker(low_score=0.05))

    # Forward C&E: LAHH-80101A/B/C -> 2 out of 3 + X effects + document_legend_notes
    fwd_rewrite = json.dumps(
        {
            "needs_retrieval": True,
            "routing_mode": "single",
            "target_domains": ["engineering"],
            "domain_queries": {"engineering": "LAHH-80101A/B/C cause effects voting"},
            "domain_weights": {"engineering": 1.0},
        }
    )
    graph_fwd = build_rag_graph(
        retriever=retriever,
        reranker=reranker,
        llm=FakeCountingChatModel(messages=iter([AIMessage(content=fwd_rewrite)])),
        session_factory=session_factory,
        collection=TEST_COLLECTION,
    )
    fwd_query = "What effects and voting apply when LAHH-80101A/B/C trips?"
    rw_fwd = await graph_fwd.nodes["rewrite_query"].ainvoke({"original_query": fwd_query, "steps": []})
    assert rw_fwd["query_intent"] == "matrix_trace"
    assert rw_fwd["engineering_top_n"] == 12

    ret_fwd = await graph_fwd.nodes["retrieve"].ainvoke(rw_fwd)
    fwd_chunks = ret_fwd["retrieved_chunks"]

    assert any(
        c.get("record_type") == "matrix_cause_row"
        and "LAHH-80101A/B/C" in c["parent_content"]
        and "2 out of 3" in c["parent_content"]
        and "XV-10101A" in c["parent_content"]
        and "FCV-20231" in c["parent_content"]
        for c in fwd_chunks
    )
    assert any(
        c.get("record_type") == "document_legend_notes"
        and c.get("is_legend_context") is True
        and "CLOSE / TRIP / ACTIVE SIGNAL" in c["parent_content"]
        for c in fwd_chunks
    )

    rr_fwd = await graph_fwd.nodes["rerank"].ainvoke({**rw_fwd, **ret_fwd})
    assert any(
        c.get("is_legend_context") is True and c["score"] >= 0.72
        for c in rr_fwd["reranked_chunks"]
    )

    # Reverse C&E: XV-10101A -> triggering causes LAHH-80101A/B/C, HZ-10001, UA-FGS-ESD-1
    rev_rewrite = json.dumps(
        {
            "needs_retrieval": True,
            "routing_mode": "single",
            "target_domains": ["engineering"],
            "domain_queries": {"engineering": "XV-10101A triggering causes interlock"},
            "domain_weights": {"engineering": 1.0},
        }
    )
    graph_rev = build_rag_graph(
        retriever=retriever,
        reranker=reranker,
        llm=FakeCountingChatModel(messages=iter([AIMessage(content=rev_rewrite)])),
        session_factory=session_factory,
        collection=TEST_COLLECTION,
    )
    rev_query = "Which causes trigger shutdown valve XV-10101A?"
    rw_rev = await graph_rev.nodes["rewrite_query"].ainvoke({"original_query": rev_query, "steps": []})
    ret_rev = await graph_rev.nodes["retrieve"].ainvoke(rw_rev)
    rev_chunks = ret_rev["retrieved_chunks"]

    assert any(
        c.get("record_type") == "matrix_effect_col"
        and "XV-10101A" in c["parent_content"]
        and "LAHH-80101A/B/C" in c["parent_content"]
        and "HZ-10001" in c["parent_content"]
        and "UA-FGS-ESD-1" in c["parent_content"]
        for c in rev_chunks
    )


# ============================================================================
# 4. Cross-Document Tag-Stem Matching (HZ-10001 <-> HZ10001_R12) + OCR Sheet
# ============================================================================


@pytest.mark.asyncio
async def test_cross_document_tag_stem_and_exploded_ocr_retrieval(seeded_engineering_corpus):
    qdrant = seeded_engineering_corpus["qdrant"]
    embedder = seeded_engineering_corpus["embedder"]
    sparse = seeded_engineering_corpus["sparse"]

    retriever = HybridRetriever(
        qdrant=qdrant,
        embedder=embedder,
        sparse_encoder=sparse,
        collection=TEST_COLLECTION,
        limit=10,
    )

    # Querying hyphenated HZ-10001 must retrieve BOTH C&E Chart AND ESDS Logic (which uses HZ10001_R12)
    hz_stems = EntityExtractor().extract_query_tags_and_stems("Trace HZ-10001 across C&E and ESDS")
    res_hz = await retriever.retrieve_federated(
        domain_queries={"engineering": "HZ-10001 shutdown signal logic"},
        mode="auto",
        associated_entities=hz_stems,
    )
    filenames_hz = {c["source_filename"] for c in res_hz}
    assert "process_ce_chart_p3_p4.pdf" in filenames_hz
    assert "esds_logic_p8_p51.pdf" in filenames_hz

    # Reverse stem query: HZ10001_R12 must also retrieve BOTH ESDS Logic AND C&E Chart
    r12_stems = EntityExtractor().extract_query_tags_and_stems("Where does HZ10001_R12 go?")
    res_r12 = await retriever.retrieve_federated(
        domain_queries={"engineering": "HZ10001_R12"},
        mode="explicit",
        associated_entities=r12_stems,
    )
    filenames_r12 = {c["source_filename"] for c in res_r12}
    assert "process_ce_chart_p3_p4.pdf" in filenames_r12
    assert "esds_logic_p8_p51.pdf" in filenames_r12

    # Exploded CAD OCR sheet (lv_swgr_logic_p3.pdf) via single-collection retrieve()
    ocr_stems = EntityExtractor().extract_query_tags_and_stems("Check CB-401 MCC-401A")
    res_ocr = await retriever.retrieve(
        "CB-401 MCC-401A trip coil",
        associated_entities=ocr_stems,
    )
    assert any(
        c["source_filename"] == "lv_swgr_logic_p3.pdf"
        and c.get("record_type") == "exploded_cad_ocr"
        and c.get("exact_tag_match") is True
        for c in res_ocr
    )


# ============================================================================
# 5. Reranker Score-Floor Protection, Dynamic top_n=12 & Legacy Graceful []
# ============================================================================


@pytest.mark.asyncio
async def test_reranker_score_floors_dynamic_top_n_and_legacy_legends(tmp_path: Path):
    client = AsyncQdrantClient(location=":memory:")
    await init_qdrant_collection(client, collection=TEST_COLLECTION, dimensions=STUB_DIM)
    embedder = DeterministicStubEmbedder(dimensions=STUB_DIM)
    sparse = DeterministicBM25Encoder()

    # Seed 10 high-scoring prose distractors, 1 exact-tag chunk, 1 legend chunk, and 1 legacy point lacking record_type
    seed_rows = [
        {
            "id": f"00000000-0000-0000-0000-0000000000{i:02d}",
            "text": f"PROSE_DISTRACTOR {i} discussing general plant reliability shutdown procedure and valve maintenance.",
            "payload": {
                "chunk_id": f"c-prose-{i}",
                "doc_id": "doc-eng-mix",
                "source_filename": "plant_manual.pdf",
                "file_type": "pdf",
                "heading_path": f"Section {i}",
                "page_numbers": [i],
                "child_text": f"PROSE_DISTRACTOR {i} discussing general plant reliability shutdown procedure and valve maintenance.",
                "parent_content": f"PARENT_PROSE_{i}",
                "domain": "engineering",
                "subdomain": "eng.general",
                "authority_tier": 1,
                "is_current": True,
                "associated_entities": [],
                "record_type": "standard_prose",
            },
        }
        for i in range(1, 11)
    ]
    seed_rows.append(
        {
            "id": "00000000-0000-0000-0000-000000000020",
            "text": "CAUSE TAG: LAHH-80101A/B/C -> XV-10101A Mark X",
            "payload": {
                "chunk_id": "c-tag-lahh",
                "doc_id": "doc-eng-mix",
                "source_filename": "ce_matrix.pdf",
                "file_type": "pdf",
                "heading_path": "Matrix > Cause LAHH-80101A/B/C",
                "page_numbers": [2],
                "child_text": "CAUSE TAG: LAHH-80101A/B/C -> XV-10101A Mark X",
                "parent_content": "PARENT_TAG_LAHH",
                "domain": "engineering",
                "subdomain": "eng.control_systems",
                "authority_tier": 3,
                "is_current": True,
                "associated_entities": ["LAHH-80101A", "LAHH-80101A/B/C", "XV-10101A"],
                "record_type": "matrix_cause_row",
            },
        }
    )
    seed_rows.append(
        {
            "id": "00000000-0000-0000-0000-000000000021",
            "text": "Legend Symbols: X = CLOSE / TRIP / ACTIVE SIGNAL",
            "payload": {
                "chunk_id": "c-legend-1",
                "doc_id": "doc-eng-mix",
                "source_filename": "ce_matrix.pdf",
                "file_type": "pdf",
                "heading_path": "Legend & Notes (Page 1)",
                "page_numbers": [1],
                "child_text": "Legend Symbols: X = CLOSE / TRIP / ACTIVE SIGNAL",
                "parent_content": "PARENT_LEGEND_1",
                "domain": "engineering",
                "subdomain": "eng.control_systems",
                "authority_tier": 3,
                "is_current": True,
                "associated_entities": [],
                "record_type": "document_legend_notes",
            },
        }
    )
    # Legacy point without `record_type`
    seed_rows.append(
        {
            "id": "00000000-0000-0000-0000-000000000099",
            "text": "Legacy pre-upgrade engineering manual without record_type.",
            "payload": {
                "chunk_id": "c-legacy-1",
                "doc_id": "doc-legacy-only",
                "source_filename": "legacy_manual.pdf",
                "file_type": "pdf",
                "heading_path": "Overview",
                "page_numbers": [1],
                "child_text": "Legacy pre-upgrade engineering manual without record_type.",
                "parent_content": "PARENT_LEGACY_1",
                "domain": "general",
                "subdomain": "general",
                "authority_tier": 3,
                "is_current": True,
                "associated_entities": [],
            },
        }
    )

    d_vecs = await embedder.embed_documents([r["text"] for r in seed_rows])
    s_vecs = await sparse.encode([r["text"] for r in seed_rows])
    await client.upsert(
        collection_name=TEST_COLLECTION,
        points=[
            qmodels.PointStruct(id=r["id"], vector={DENSE: dv, SPARSE: sv}, payload=r["payload"])
            for r, dv, sv in zip(seed_rows, d_vecs, s_vecs)
        ],
    )

    retriever = HybridRetriever(
        qdrant=client,
        embedder=embedder,
        sparse_encoder=sparse,
        collection=TEST_COLLECTION,
        limit=15,
    )

    # Legacy document without record_type gracefully returns []
    assert await retriever.fetch_document_legends(["doc-legacy-only"]) == []

    # Unfiltered semantic RRF must still return untagged legend/prose chunks alongside tagged chunks
    fed = await retriever.retrieve_federated(
        domain_queries={"engineering": "plant reliability shutdown valve maintenance"},
        mode="explicit",
        associated_entities=["LAHH-80101A"],
        limit=12,
    )
    fed_ids = {c["chunk_id"] for c in fed}
    assert "c-tag-lahh" in fed_ids
    assert "c-prose-1" in fed_ids
    assert "c-legend-1" in fed_ids

    # Cross-domain fallback when mode='auto' routes to 'economics' while tag lives in 'engineering'
    cross_auto = await retriever.retrieve_federated(
        domain_queries={"economics": "LAHH-80101A"},
        mode="auto",
        associated_entities=["LAHH-80101A"],
        limit=6,
    )
    assert any(c["chunk_id"] == "c-tag-lahh" for c in cross_auto)

    # Explicit domain lock must NOT leak 'engineering' when locked to 'economics'
    cross_explicit = await retriever.retrieve_federated(
        domain_queries={"economics": "LAHH-80101A"},
        mode="explicit",
        associated_entities=["LAHH-80101A"],
        limit=6,
    )
    assert all(c["domain"] == "economics" for c in cross_explicit)

    # Dynamic top_n=12 expansion + score-floor protection in rerank_node
    reranker = FlashRankReranker(top_n=6, ranker=LowCrossEncoderStubRanker(low_score=0.05))
    graph = build_rag_graph(
        retriever=retriever,
        reranker=reranker,
        llm=FakeCountingChatModel(messages=iter([])),
        collection=TEST_COLLECTION,
    )

    # Tag + Legend + 10 distractors -> engineering_top_n=12 keeps all 12 (>6) and puts exact_tag_match #1
    candidates = [dict(c) for c in fed]
    for c in candidates:
        if c["chunk_id"] == "c-legend-1":
            c["is_legend_context"] = True

    eng_reranked = (
        await graph.nodes["rerank"].ainvoke(
            {
                "original_query": "Trace LAHH-80101A causes",
                "engineering_top_n": 12,
                "retrieved_chunks": candidates,
                "steps": [],
            }
        )
    )["reranked_chunks"]

    assert len(eng_reranked) == 12
    # Exact tag match beats Tier-1 prose distractor (0.816) via floor max(s_final, 0.75) + 0.10 >= 0.85
    assert eng_reranked[0]["chunk_id"] == "c-tag-lahh"
    assert eng_reranked[0]["score"] >= 0.85
    legend_hit = next(c for c in eng_reranked if c["chunk_id"] == "c-legend-1")
    assert legend_hit["score"] >= 0.72

    # Prose intent (engineering_top_n=None) stays capped at default top_n=6
    prose_reranked = (
        await graph.nodes["rerank"].ainvoke(
            {
                "original_query": "Summarize plant reliability manual",
                "engineering_top_n": None,
                "retrieved_chunks": candidates,
                "steps": [],
            }
        )
    )["reranked_chunks"]
    assert len(prose_reranked) == 6

    await client.close()


# ============================================================================
# 6. Full stream_chat_response Integration Test Across Multiple CAD Fixtures
# ============================================================================


@pytest.mark.asyncio
async def test_end_to_end_engineering_crag_stream_cross_doc(tmp_path: Path, seeded_engineering_corpus):
    qdrant = seeded_engineering_corpus["qdrant"]
    session_factory = seeded_engineering_corpus["session_factory"]
    embedder = seeded_engineering_corpus["embedder"]
    sparse = seeded_engineering_corpus["sparse"]
    saver = None

    from src.rag.graph import get_checkpointer

    saver = await get_checkpointer(tmp_path / "db" / "chat_eng.db")
    reranker = FlashRankReranker(top_n=6, ranker=LowCrossEncoderStubRanker(low_score=0.05))

    fake_rewrite = json.dumps(
        {
            "needs_retrieval": True,
            "routing_mode": "single",
            "target_domains": ["engineering"],
            "domain_queries": {"engineering": "HZ-10001 C&E matrix cause effects ESDS logic HZ10001_R12"},
            "domain_weights": {"engineering": 1.0},
        }
    )
    fake_grade = json.dumps({"relevant_indices": [1, 2, 3], "reasoning": "Relevant C&E, ESDS logic, and Legend"})
    fake_synth = "HZ-10001 initiates XV-10101A closure (Mark X = CLOSE / TRIP) [1] and connects via HZ10001_R12 in LTRCV_12 [2]."

    fake_llm = FakeCountingChatModel(
        messages=iter(
            [
                AIMessage(content=fake_rewrite),
                AIMessage(content=fake_grade),
                AIMessage(content=fake_synth),
            ]
        )
    )

    events = [
        ev
        async for ev in stream_chat_response(
            query="Trace HZ-10001 causes, effects, and ESDS logic connection",
            thread_id="thread-eng-e2e",
            qdrant=qdrant,
            embedder=embedder,
            sparse_encoder=sparse,
            reranker=reranker,
            llm=fake_llm,
            checkpointer=saver,
            session_factory=session_factory,
            collection=TEST_COLLECTION,
        )
    ]

    assert events[-1]["event"] == "done"
    done_data = events[-1]["data"]
    assert "HZ-10001" in done_data["message"]
    assert len(done_data["citations"]) >= 2
    cited_files = {c["source_filename"] for c in done_data["citations"]}
    assert cited_files & {"process_ce_chart_p3_p4.pdf", "esds_logic_p8_p51.pdf"}

    await close_checkpointer()
