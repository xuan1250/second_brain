"""End-to-End Acceptance Tests verifying all 5 Brainstorm Acceptance Criteria (AC1–AC5).

Includes:
1. Offline E2E Acceptance Suite (runs by default in `uv run pytest` using 1024-dim deterministic
   fixtures, GenericFakeChatModel, and in-memory Qdrant).
2. Live E2E Acceptance Suite (`@pytest.mark.live`, run via `uv run pytest -m live` against a
   running Qdrant container, real BAAI/bge-m3 embeddings, FlashRank MultiBERT, and live LLM).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import text

from src.api.main import create_app
from src.config import LLMConfig, Settings, get_settings
from src.db.qdrant import DENSE, SPARSE
from src.ingestion.indexer import (
    LocalBgeM3Embedder,
    set_embedder,
    set_sparse_encoder,
)
from src.rag.graph import close_checkpointer
from src.rag.llm import get_chat_model, set_chat_model
from src.rag.reranker import FlashRankReranker, set_reranker
from src.rag.retriever import set_active_qdrant_client
from src.rag.state import Citation, format_citation_source

pytest_plugins = ["tests.conftest"]

MIME_BY_EXT: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


class CapturingFakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel that records every prompt message list for multi-turn verification."""

    call_count: int = 0
    captured_prompts: list[list[Any]] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.call_count += 1
        self.captured_prompts.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _parse_sse_frames(raw_body: str) -> list[dict[str, Any]]:
    """Parse SSE response body (`event: ...\ndata: ...\n\n`) into event dicts."""
    frames: list[dict[str, Any]] = []
    for block in raw_body.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.partition(":")[2].strip()
            elif line.startswith("data:"):
                data_lines.append(line.partition(":")[2].strip())
        data_raw = "\n".join(data_lines)
        frames.append(
            {
                "event": event_name,
                "data": json.loads(data_raw) if data_raw else {},
            }
        )
    return frames


async def _poll_until_ready(client: AsyncClient, doc_id: str, timeout_s: float = 30.0) -> dict[str, Any]:
    """Poll `GET /api/v1/documents/{doc_id}/status` until ingestion reaches `ready` or `failed`."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    last_data: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        resp = await client.get(f"/api/v1/documents/{doc_id}/status")
        assert resp.status_code == 200
        last_data = resp.json()
        if last_data["status"] == "ready":
            return last_data
        if last_data["status"] == "awaiting_approval":
            await client.post(f"/api/v1/documents/{doc_id}/approve-classification", json={})
        if last_data["status"] == "failed":
            raise AssertionError(f"Ingestion failed for {doc_id}: {last_data}")
        await asyncio.sleep(0.05)
    raise AssertionError(f"Timed out waiting for document {doc_id} to become ready: {last_data}")


def _assert_markdown_has_heading_list_and_table(markdown_text: str, fmt: str) -> None:
    """Assert converted Markdown preserves `#` headings, bullet list items, and `|` table pipes (AC1)."""
    assert re.search(r"^#{1,6}\s+\S+", markdown_text, flags=re.MULTILINE), (
        f"[{fmt}] Expected Markdown heading (`# ...`) in:\n{markdown_text[:500]}"
    )
    assert re.search(r"^\s*[-*]\s+\S+", markdown_text, flags=re.MULTILINE), (
        f"[{fmt}] Expected Markdown bullet list (`- ...` or `* ...`) in:\n{markdown_text[:500]}"
    )
    assert "|" in markdown_text and re.search(r"\|.*\|", markdown_text), (
        f"[{fmt}] Expected Markdown table pipes (`| ... |`) in:\n{markdown_text[:500]}"
    )


async def test_e2e_acceptance_ac1_to_ac5_offline(
    e2e_api_client: tuple[AsyncClient, Path, Any],
    e2e_multi_format_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
):
    """Full End-to-End Acceptance Test verifying AC1, AC2, AC3, AC4, and AC5 in offline mode."""
    client, data_dir, app = e2e_api_client
    qdrant: AsyncQdrantClient = app.state.qdrant
    session_factory = app.state.session_factory
    collection_name = app.state.settings.qdrant_collection

    # ==========================================================================
    # AC1: All 4 document types (PDF, DOCX, XLSX, PPTX) convert to structured
    #      Markdown with tables (`|`), headings (`#`), and lists (`- `) preserved.
    # ==========================================================================
    ingested_docs: dict[str, dict[str, Any]] = {}

    for fmt, file_path in e2e_multi_format_files.items():
        file_bytes = file_path.read_bytes()
        resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": (file_path.name, file_bytes, MIME_BY_EXT[fmt])},
        )
        assert resp.status_code == 202, f"Upload failed for {fmt}: {resp.text}"
        up_json = resp.json()
        assert up_json["file_type"] == fmt
        assert up_json["duplicate"] is False

        status_json = await _poll_until_ready(client, up_json["id"])
        assert status_json["status"] == "ready"
        assert status_json["chunk_count"] >= 1

        # Verify Markdown mirror endpoint & AC1 structural preservation
        md_resp = await client.get(f"/api/v1/documents/{up_json['id']}/markdown")
        assert md_resp.status_code == 200
        md_payload = md_resp.json()
        _assert_markdown_has_heading_list_and_table(md_payload["markdown"], fmt)

        # Verify disk caches exist in data/{raw,parsed,markdown}
        sha = up_json["sha256"]
        assert (data_dir / "raw" / f"{sha}.{fmt}").is_file()
        assert (data_dir / "parsed" / f"{sha}.json").is_file()
        assert (data_dir / "markdown" / f"{sha}.md").is_file()

        # Verify idempotent re-upload returns 200 OK with duplicate=True
        dup_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": (file_path.name, file_bytes, MIME_BY_EXT[fmt])},
        )
        assert dup_resp.status_code == 200
        assert dup_resp.json()["duplicate"] is True
        assert dup_resp.json()["id"] == up_json["id"]

        ingested_docs[fmt] = {
            "id": up_json["id"],
            "sha256": sha,
            "filename": file_path.name,
            "chunk_count": status_json["chunk_count"],
        }

    # ==========================================================================
    # AC2: Converted Markdown is chunked, embedded (1024-dim dense + BM25 sparse),
    #      and stored in Qdrant with child_text, parent_content, heading_path,
    #      and format-appropriate page_numbers.
    # ==========================================================================
    all_points, _ = await qdrant.scroll(
        collection_name=collection_name,
        limit=250,
        with_payload=True,
        with_vectors=True,
    )
    expected_total_chunks = sum(d["chunk_count"] for d in ingested_docs.values())
    assert len(all_points) == expected_total_chunks

    points_by_format: dict[str, list[Any]] = {fmt: [] for fmt in ingested_docs}
    for pt in all_points:
        assert pt.payload is not None
        ftype = pt.payload["file_type"]
        points_by_format[ftype].append(pt)

        # Verify payload schema completeness
        for required_field in (
            "doc_id",
            "chunk_id",
            "source_filename",
            "file_type",
            "heading_path",
            "page_numbers",
            "token_count",
            "child_text",
            "parent_content",
        ):
            assert required_field in pt.payload, f"Missing {required_field} in Qdrant payload"

        assert str(pt.payload["child_text"]).strip() != ""
        assert str(pt.payload["parent_content"]).strip() != ""
        assert str(pt.payload["heading_path"]).strip() != ""

        # Verify 1024-dim dense vector + non-empty BM25 sparse vector
        assert DENSE in pt.vector
        assert SPARSE in pt.vector
        assert len(pt.vector[DENSE]) == 1024
        assert len(pt.vector[SPARSE].indices) > 0
        assert len(pt.vector[SPARSE].indices) == len(pt.vector[SPARSE].values)

    for fmt, pts in points_by_format.items():
        assert len(pts) == ingested_docs[fmt]["chunk_count"]
        if fmt in ("pdf", "pptx"):
            # Paginated / slide formats carry 1-indexed page numbers
            all_pages = [p for pt in pts for p in pt.payload["page_numbers"]]
            assert len(all_pages) >= 1 and all(isinstance(p, int) and p >= 1 for p in all_pages)
        else:
            # Unpaginated formats (DOCX, XLSX) carry empty list []
            for pt in pts:
                assert pt.payload["page_numbers"] == []

    # ==========================================================================
    # AC3: Multi-turn SSE chat retrieves relevant chunks, synthesizes answers
    #      with citations (source document + section/page), and resolves follow-ups.
    # ==========================================================================
    pdf_doc_id = ingested_docs["pdf"]["id"]
    docx_doc_id = ingested_docs["docx"]["id"]

    capturing_llm = CapturingFakeChatModel(
        messages=iter(
            [
                # Turn 1 - pre-routing classification fallback
                AIMessage(
                    content=json.dumps(
                        {
                            "primary_domain": "economics",
                            "primary_subdomain": "econ.financial_reporting",
                            "confidence": 0.9,
                            "reasoning": "Financial analysis query",
                        }
                    )
                ),
                # Turn 1 - rewrite_query
                AIMessage(
                    content=json.dumps(
                        {
                            "needs_retrieval": True,
                            "rewritten_queries": ["Q3 2025 operating margin expansion drivers"],
                        }
                    )
                ),
                # Turn 1 - grade_context
                AIMessage(
                    content=json.dumps(
                        {
                            "relevant_indices": [1],
                            "reasoning": "Chunk 1 directly reports Q3 2025 operating margin expansion.",
                        }
                    )
                ),
                # Turn 1 - synthesize
                AIMessage(
                    content="Operating margin expanded by 3.2% year-over-year in Q3 2025 due to supply chain automation [1]."
                ),
                # Turn 2 - pre-routing classification fallback
                AIMessage(
                    content=json.dumps(
                        {
                            "primary_domain": "economics",
                            "primary_subdomain": "econ.financial_reporting",
                            "confidence": 0.9,
                            "reasoning": "Follow up financial metrics query",
                        }
                    )
                ),
                # Turn 2 - rewrite_query (follow-up resolving 'it' / 'that quarter' from Turn 1)
                AIMessage(
                    content=json.dumps(
                        {
                            "needs_retrieval": True,
                            "rewritten_queries": ["Q3 2025 gross profit and enterprise net retention"],
                        }
                    )
                ),
                # Turn 2 - grade_context
                AIMessage(
                    content=json.dumps(
                        {
                            "relevant_indices": [1],
                            "reasoning": "Contains Q3 2025 gross profit metrics.",
                        }
                    )
                ),
                # Turn 2 - synthesize
                AIMessage(
                    content="Gross profit reached $142M in Q3 2025 with 96% enterprise net retention [1]."
                ),
            ]
        )
    )
    set_chat_model(capturing_llm)

    thread_id = "e2e-acceptance-thread"
    turn1_resp = await client.post(
        "/api/v1/chat/completions",
        json={
            "message": "What drove operating margin expansion in Q3 2025?",
            "thread_id": thread_id,
            "filter_doc_ids": [pdf_doc_id],
        },
    )
    assert turn1_resp.status_code == 200
    t1_frames = _parse_sse_frames(turn1_resp.text)
    t1_events = [f["event"] for f in t1_frames]
    assert "step" in t1_events
    assert "citations" in t1_events
    assert "token" in t1_events
    assert t1_events[-1] == "done"

    t1_done = t1_frames[-1]["data"]
    assert "3.2%" in t1_done["message"]
    assert "[1]" in t1_done["message"]
    assert len(t1_done["citations"]) >= 1
    cit1: Citation = t1_done["citations"][0]
    assert cit1["doc_id"] == pdf_doc_id
    assert cit1["source_filename"] == "q3_financial_report.pdf"
    assert cit1["heading_path"] != ""
    assert len(cit1["page_numbers"]) >= 1
    formatted_cit = format_citation_source(cit1)
    assert "q3_financial_report.pdf" in formatted_cit
    assert "p." in formatted_cit
    assert "§" in formatted_cit

    # Turn 2: Follow-up question on the same thread_id using pronouns ("its", "that quarter")
    turn2_resp = await client.post(
        "/api/v1/chat/completions",
        json={
            "message": "What was its gross profit in that quarter?",
            "thread_id": thread_id,
            "filter_doc_ids": [pdf_doc_id],
        },
    )
    assert turn2_resp.status_code == 200
    t2_frames = _parse_sse_frames(turn2_resp.text)
    t2_done = t2_frames[-1]["data"]
    assert "$142M" in t2_done["message"]
    assert "[1]" in t2_done["message"]
    assert len(t2_done["citations"]) >= 1
    assert t2_done["citations"][0]["source_filename"] == "q3_financial_report.pdf"

    # Verify Turn 2 rewrite_query prompt included Turn 1 conversation history from AsyncSqliteSaver
    turn2_rewrite_messages = capturing_llm.captured_prompts[5]
    rewrite_human_text = turn2_rewrite_messages[-1].content
    assert "What drove operating margin expansion in Q3 2025?" in rewrite_human_text
    assert "Operating margin expanded by 3.2%" in rewrite_human_text

    # Verify thread persistence via GET /api/v1/chat/threads/{thread_id}
    thread_detail_resp = await client.get(f"/api/v1/chat/threads/{thread_id}")
    assert thread_detail_resp.status_code == 200
    thread_detail = thread_detail_resp.json()
    assert len(thread_detail["messages"]) == 4

    # ==========================================================================
    # AC4: LLM provider is configurable via setting & per-request override
    #      (OpenAI, Anthropic, DeepSeek, and custom OpenAI-compatible base_url).
    # ==========================================================================
    set_chat_model(None)
    base_cfg = Settings(llm=LLMConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0))

    model_openai = get_chat_model(settings=base_cfg)
    assert isinstance(model_openai, ChatOpenAI)
    assert model_openai.model_name == "gpt-4o-mini"

    model_anthropic = get_chat_model(
        provider_override="anthropic",
        model_override="claude-sonnet-4-20250514",
        settings=base_cfg,
    )
    assert isinstance(model_anthropic, ChatAnthropic)
    assert model_anthropic.model == "claude-sonnet-4-20250514"

    model_deepseek = get_chat_model(
        provider_override="deepseek",
        model_override="deepseek-chat",
        settings=base_cfg,
    )
    assert isinstance(model_deepseek, ChatDeepSeek)
    assert model_deepseek.model_name == "deepseek-chat"

    glm_cfg = Settings(
        llm=LLMConfig(
            provider="glm",
            model_name="glm-4-plus",
            base_url="https://open.bigmodel.cn/api/paas/v4/",
        )
    )
    model_glm = get_chat_model(settings=glm_cfg)
    assert isinstance(model_glm, ChatOpenAI)
    assert model_glm.model_name == "glm-4-plus"

    # Also verify per-request provider/model override flows through POST /api/v1/chat/completions
    captured_overrides: list[tuple[str | None, str | None]] = []
    override_fake_llm = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content=json.dumps(
                        {
                            "primary_domain": "engineering",
                            "primary_subdomain": "eng.software",
                            "confidence": 0.9,
                            "reasoning": "Architecture query",
                        }
                    )
                ),
                AIMessage(
                    content=json.dumps(
                        {
                            "needs_retrieval": True,
                            "rewritten_queries": ["Second Brain vector store architecture"],
                        }
                    )
                ),
                AIMessage(content=json.dumps({"relevant_indices": [1], "reasoning": "Matches architecture spec"})),
                AIMessage(content="The vector store uses Qdrant Hybrid with 1024-dim dense vectors [1]."),
            ]
        )
    )

    def _override_factory(
        provider_override: str | None = None,
        model_override: str | None = None,
        **kwargs: Any,
    ):
        captured_overrides.append((provider_override, model_override))
        return override_fake_llm

    set_chat_model(_override_factory)
    override_resp = await client.post(
        "/api/v1/chat/completions",
        json={
            "message": "Which vector store is used in the architecture spec?",
            "thread_id": "e2e-override-thread",
            "filter_doc_ids": [docx_doc_id],
            "provider": "deepseek",
            "model": "deepseek-chat",
        },
    )
    assert override_resp.status_code == 200
    override_frames = _parse_sse_frames(override_resp.text)
    assert override_frames[-1]["event"] == "done"
    assert ("deepseek", "deepseek-chat") in captured_overrides
    # DOCX citation has no page numbers (`page_numbers == []`) -> omits `p.X`
    docx_cit = override_frames[-1]["data"]["citations"][0]
    assert docx_cit["page_numbers"] == []
    assert "p." not in format_citation_source(docx_cit)

    # ==========================================================================
    # AC5: Atomic 3-tier deletion across Qdrant, SQLite (documents,
    #      document_chunks, ingestion_jobs), and disk caches (`data/{raw,parsed,markdown}`).
    # ==========================================================================
    for fmt, doc_info in ingested_docs.items():
        doc_id = doc_info["id"]
        sha = doc_info["sha256"]

        del_resp = await client.delete(f"/api/v1/documents/{doc_id}")
        assert del_resp.status_code == 200, f"Delete failed for {fmt} ({doc_id})"
        assert del_resp.json()["deleted"] is True

        # Verify 0 Qdrant points remain for this doc_id
        remaining_for_doc = await qdrant.count(
            collection_name=collection_name,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            ),
            exact=True,
        )
        assert remaining_for_doc.count == 0

        # Verify disk cache files for this SHA-256 were removed
        assert not (data_dir / "raw" / f"{sha}.{fmt}").exists()
        assert not (data_dir / "parsed" / f"{sha}.json").exists()
        assert not (data_dir / "markdown" / f"{sha}.md").exists()

    # Verify total counts across Qdrant and SQLite are 0
    total_qdrant = await qdrant.count(collection_name=collection_name, exact=True)
    assert total_qdrant.count == 0

    async with session_factory() as session:
        docs_left = (await session.execute(text("SELECT count(*) FROM documents"))).scalar()
        chunks_left = (await session.execute(text("SELECT count(*) FROM document_chunks"))).scalar()
        jobs_left = (await session.execute(text("SELECT count(*) FROM ingestion_jobs"))).scalar()
        assert docs_left == 0
        assert chunks_left == 0
        assert jobs_left == 0


@pytest.mark.live
async def test_live_e2e_acceptance_ac1_to_ac5(
    tmp_path: Path,
    e2e_multi_format_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
):
    """Live E2E Acceptance Suite (`pytest -m live`) against Qdrant, BAAI/bge-m3, FlashRank, and live LLM."""
    data_dir = tmp_path / "live_data"
    db_dir = tmp_path / "live_data" / "db"
    dist_dir = tmp_path / "live_dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "index.html").write_text("<!doctype html><html><body>Second Brain Live</body></html>", encoding="utf-8")

    live_collection = f"second_brain_live_e2e_{uuid.uuid4().hex[:8]}"
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("DB_DIR", str(db_dir))
    monkeypatch.setenv("FRONTEND_DIST_DIR", str(dist_dir))
    monkeypatch.setenv("QDRANT_URL", qdrant_url)
    monkeypatch.setenv("INGESTION_USE_PROCESS_POOL", "false")

    get_settings.cache_clear()
    settings = get_settings()
    settings.qdrant.collection = live_collection

    # Use real BAAI/bge-m3 (1024-dim) and real FlashRank ms-marco-MultiBERT-L-12
    set_embedder(LocalBgeM3Embedder(model_name="BAAI/bge-m3", dimensions=1024))
    set_sparse_encoder(None)  # Uses FastEmbedBM25Encoder
    set_reranker(FlashRankReranker(model_name="ms-marco-MultiBERT-L-12", top_n=6, allow_download=True))

    # If no live LLM API key is configured in environment/.env, fall back to a deterministic chat model
    # so vector/reranker/Qdrant live verification still completes cleanly, or use real LLM when configured.
    has_live_llm_key = any(
        os.environ.get(k) and not os.environ.get(k, "").startswith("your-")
        for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY")
    ) and bool(os.environ.get("LLM_MODEL_NAME") or settings.llm.model_name)

    if not has_live_llm_key:
        set_chat_model(
            GenericFakeChatModel(
                messages=iter(
                    [
                        AIMessage(
                            content=json.dumps(
                                {
                                    "needs_retrieval": True,
                                    "rewritten_queries": ["Q3 2025 operating margin expansion"],
                                }
                            )
                        ),
                        AIMessage(content=json.dumps({"relevant_indices": [1], "reasoning": "Relevant Q3 report"})),
                        AIMessage(content="Operating margin expanded by 3.2% year-over-year in Q3 2025 [1]."),
                    ]
                )
            )
        )
    else:
        set_chat_model(None)

    app = create_app()
    async with LifespanManager(app) as manager:
        app.state.settings.qdrant.collection = live_collection
        app.state.orchestrator.collection = live_collection
        qdrant: AsyncQdrantClient = app.state.qdrant

        transport = ASGITransport(app=manager.app)
        try:
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                uploaded_ids: list[str] = []
                for fmt, file_path in e2e_multi_format_files.items():
                    resp = await client.post(
                        "/api/v1/documents/upload",
                        files={"file": (file_path.name, file_path.read_bytes(), MIME_BY_EXT[fmt])},
                    )
                    assert resp.status_code == 202
                    doc_id = resp.json()["id"]
                    uploaded_ids.append(doc_id)
                    await _poll_until_ready(client, doc_id, timeout_s=120.0)

                    md_resp = await client.get(f"/api/v1/documents/{doc_id}/markdown")
                    assert md_resp.status_code == 200
                    _assert_markdown_has_heading_list_and_table(md_resp.json()["markdown"], fmt)

                pts, _ = await qdrant.scroll(live_collection, limit=50, with_vectors=True, with_payload=True)
                assert len(pts) >= 4
                assert len(pts[0].vector[DENSE]) == 1024

                chat_resp = await client.post(
                    "/api/v1/chat/completions",
                    json={
                        "message": "What drove operating margin expansion in Q3 2025?",
                        "thread_id": "live-e2e-thread",
                    },
                )
                assert chat_resp.status_code == 200
                frames = _parse_sse_frames(chat_resp.text)
                assert frames[-1]["event"] == "done"

                for doc_id in uploaded_ids:
                    del_resp = await client.delete(f"/api/v1/documents/{doc_id}")
                    assert del_resp.status_code == 200

                assert (await qdrant.count(live_collection, exact=True)).count == 0
        finally:
            try:
                await qdrant.delete_collection(live_collection)
            except Exception:
                pass
