"""Batch concurrency and resilience torture integration test.

Validates:
1. 25-file mixed workload (10 digital text PDFs, 5 notes, 5 duplicate files, 3 poison files, 2 in-flight deletions).
2. All 15 valid documents reach `status == "ready"`.
3. 5 duplicate files return `duplicate == True` with zero HTTP 500 IntegrityError crashes.
4. 3 poison files transition to `status == "failed"` with clear error strings without stalling or crashing the process pool.
5. 2 documents deleted mid-pipeline leave ZERO ghost vectors in Qdrant.
6. Zero deadlocks, zero lingering active tasks in orchestrator.
7. Hybrid retrieval queries against Qdrant return valid chunks only from ready documents.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import docx
from httpx import AsyncClient
import pymupdf
import pytest
from qdrant_client import AsyncQdrantClient

from src.ingestion.worker import IngestionOrchestrator
from src.rag.retriever import HybridRetriever


def _create_digital_pdf(target_path: Path, index: int) -> bytes:
    """Deterministic digital text PDF (> 120 chars/page, 0 images) for fast-path routing."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    heading = f"# Chapter {index}: High-Throughput Batch Ingestion Resilience"
    body = (
        f"In section {index}, we explore high-throughput streaming pipelines, "
        f"idempotency keys, content-addressed storage invariants, and cross-tier barrier synchronization. "
        f"Testing concurrency stress with synthetic document workloads guarantees zero orphan vectors.\n"
        f"Continuous operation under stress verifies self-healing worker pool mechanics and auto-recovery.\n"
    ) * 4
    page.insert_textbox(pymupdf.Rect(54, 54, 550, 720), f"{heading}\n\n{body}")
    doc.save(str(target_path))
    doc.close()
    return target_path.read_bytes()


def _create_notes_docx(target_path: Path, index: int) -> bytes:
    """Deterministic Word note document with hierarchical headings and body paragraphs."""
    d = docx.Document()
    d.add_heading(f"Engineering Note {index}: Distributed Consistency Invariants", level=1)
    d.add_paragraph(
        f"Note {index} details pre-upsert relational verification barriers and compensating rollbacks. "
        "Every chunk indexed into vector memory must be validated against relational persistence. "
        "Mid-batch worker recycling purges memory leaks and prevents worker exhaustion."
    )
    d.save(target_path)
    return target_path.read_bytes()


def _create_poison_header_pdf(target_path: Path) -> bytes:
    """Poison file: PDF header with corrupt binary noise."""
    content = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\nCorrupted binary noise that breaks tokenizer and parser\x00\xff\xfe\xaa\xbb%%EOF"
    target_path.write_bytes(content)
    return content


def _create_poison_xref_pdf(target_path: Path) -> bytes:
    """Poison file: Malformed PDF syntax with unresolvable xref offsets."""
    content = (
        b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"xref\n0 2\n0000000000 65535 f\n9999999999 00000 n\n"
        b"trailer\n<< /Size 2 /Root 1 0 R >>\nstartxref\n12\n%%EOF"
    )
    target_path.write_bytes(content)
    return content


def _create_poison_corrupt_docx(target_path: Path) -> bytes:
    """Poison file: Invalid non-zip archive masquerading as .docx."""
    content = b"PK\x03\x04CORRUPTED_ZIP_ARCHIVE_DATA_NOT_A_VALID_DOCX_FILE\x00\x00\xff\xff"
    target_path.write_bytes(content)
    return content


async def _poll_batch_completion(
    client: AsyncClient,
    doc_ids: set[str],
    timeout_s: float = 60.0,
) -> dict[str, dict[str, Any]]:
    """Poll document status endpoints until all doc_ids reach terminal status (ready/failed/deleted)."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    results: dict[str, dict[str, Any]] = {}
    pending = set(doc_ids)

    while pending and asyncio.get_running_loop().time() < deadline:
        for doc_id in list(pending):
            resp = await client.get(f"/api/v1/documents/{doc_id}/status")
            if resp.status_code == 404:
                results[doc_id] = {"status": "deleted"}
                pending.remove(doc_id)
            elif resp.status_code == 200:
                payload = resp.json()
                if payload.get("status") in ("ready", "failed"):
                    results[doc_id] = payload
                    pending.remove(doc_id)
        if pending:
            await asyncio.sleep(0.1)

    if pending:
        raise AssertionError(f"Documents did not complete within {timeout_s}s: {pending}")
    return results


@pytest.mark.asyncio
async def test_batch_concurrency_stress_and_resilience_torture(e2e_api_client, tmp_path: Path):
    """Torture test running 25 mixed synthetic documents concurrently through FastAPI & IngestionOrchestrator.

    Workload breakdown:
    - 10 Digital text PDFs (clean, fast-path qualified)
    - 5 Supported notes (.docx)
    - 5 Duplicate files (concurrently submitted identical hashes)
    - 3 Poison files (corrupt header, broken xref, invalid docx)
    - 2 In-flight deletions (deleted via API while actively in pipeline)
    Total = 25 files
    """
    client, data_dir, app = e2e_api_client
    orchestrator: IngestionOrchestrator = app.state.orchestrator
    qdrant_client: AsyncQdrantClient = app.state.qdrant
    collection_name = app.state.settings.qdrant_collection

    files_dir = tmp_path / "synthetic_batch"
    files_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate 10 digital text PDFs
    digital_files: list[tuple[str, bytes, str]] = []
    for i in range(1, 11):
        filename = f"digital_text_{i:02d}.pdf"
        b = _create_digital_pdf(files_dir / filename, i)
        digital_files.append((filename, b, "application/pdf"))

    # 2. Generate 5 supported DOCX notes
    notes_files: list[tuple[str, bytes, str]] = []
    for i in range(1, 6):
        filename = f"engineering_note_{i:02d}.docx"
        b = _create_notes_docx(files_dir / filename, i)
        notes_files.append((filename, b, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))

    # 3. Generate 3 poison files
    poison_files: list[tuple[str, bytes, str]] = [
        ("poison_header.pdf", _create_poison_header_pdf(files_dir / "poison_header.pdf"), "application/pdf"),
        ("poison_xref.pdf", _create_poison_xref_pdf(files_dir / "poison_xref.pdf"), "application/pdf"),
        (
            "poison_corrupt.docx",
            _create_poison_corrupt_docx(files_dir / "poison_corrupt.docx"),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
    ]

    # 4. Generate 2 delete-target documents
    delete_target_files: list[tuple[str, bytes, str]] = [
        ("delete_me_01.pdf", _create_digital_pdf(files_dir / "delete_me_01.pdf", 901), "application/pdf"),
        ("delete_me_02.pdf", _create_digital_pdf(files_dir / "delete_me_02.pdf", 902), "application/pdf"),
    ]

    # Helper to upload a single file
    async def upload_file(filename: str, content: bytes, mime_type: str) -> dict[str, Any]:
        resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": (filename, content, mime_type)},
        )
        assert resp.status_code in (200, 202), f"Upload failed for {filename}: {resp.status_code} {resp.text}"
        return resp.json()

    # Step A: Register the delete-target files and hook orchestrator._update_state to delete them mid-pipeline
    delete_target_doc_ids: set[str] = set()
    deletion_triggered: set[str] = set()
    original_update_state = orchestrator._update_state

    async def hooked_update_state(doc_id: str, job_id: str, **kwargs):
        res = await original_update_state(doc_id, job_id, **kwargs)
        if doc_id in delete_target_doc_ids and doc_id not in deletion_triggered:
            stage = kwargs.get("stage")
            if stage in ("parsing", "chunking"):
                deletion_triggered.add(doc_id)
                # Issue atomic deletion via HTTP client while active in pipeline
                del_resp = await client.delete(f"/api/v1/documents/{doc_id}")
                assert del_resp.status_code == 200
                assert del_resp.json()["deleted"] is True
        return res

    orchestrator._update_state = hooked_update_state

    for fn, cont, mime in delete_target_files:
        payload = await upload_file(fn, cont, mime)
        delete_target_doc_ids.add(payload["id"])
    assert len(delete_target_doc_ids) == 2

    # Step B: Concurrently upload 15 valid files and 3 poison files
    valid_tasks = [upload_file(fn, cont, mime) for fn, cont, mime in digital_files + notes_files]
    poison_tasks = [upload_file(fn, cont, mime) for fn, cont, mime in poison_files]

    primary_results = await asyncio.gather(*valid_tasks, *poison_tasks, return_exceptions=False)
    valid_results = primary_results[:15]
    poison_results = primary_results[15:]

    valid_doc_ids = {r["id"] for r in valid_results}
    assert len(valid_doc_ids) == 15, f"Expected 15 unique valid doc_ids, got {len(valid_doc_ids)}"

    poison_doc_ids = {r["id"] for r in poison_results}
    assert len(poison_doc_ids) == 3, f"Expected 3 unique poison doc_ids, got {len(poison_doc_ids)}"

    # Step C: Concurrently upload 5 identical duplicates of the first 5 digital files
    dup_tasks = [upload_file(fn, cont, mime) for fn, cont, mime in digital_files[:5]]
    dup_results = await asyncio.gather(*dup_tasks, return_exceptions=False)
    assert len(dup_results) == 5

    for dup_res in dup_results:
        assert dup_res.get("duplicate") is True, f"Expected duplicate=True, got {dup_res}"
        assert dup_res["id"] in valid_doc_ids, f"Duplicate ID {dup_res['id']} not in valid_doc_ids"

    # Step D: Poll until all 20 distinct documents reach terminal state
    all_tracked_ids = valid_doc_ids | poison_doc_ids | delete_target_doc_ids
    assert len(all_tracked_ids) == 20

    terminal_statuses = await _poll_batch_completion(client, all_tracked_ids, timeout_s=60.0)

    # Assert 1: All 15 valid files reach status == "ready"
    for doc_id in valid_doc_ids:
        st = terminal_statuses[doc_id]
        assert st["status"] == "ready", f"Valid doc {doc_id} failed to reach ready: {st}"
        assert st.get("chunk_count", 0) > 0
        if st.get("file_type") == "pdf":
            assert (st.get("page_count") or 0) > 0
        assert st.get("error") is None

    # Assert 2: All 3 poison files reached status == "failed" with clear error message
    for doc_id in poison_doc_ids:
        st = terminal_statuses[doc_id]
        assert st["status"] == "failed", f"Poison doc {doc_id} expected status=failed, got {st}"
        assert st.get("error") is not None
        assert len(st["error"]) > 0

    # Assert 3: 2 deleted documents leave ZERO ghost vectors in Qdrant
    assert len(deletion_triggered) == 2, f"Expected 2 documents to trigger deletion mid-pipeline: {deletion_triggered}"
    for doc_id in delete_target_doc_ids:
        chk_resp = await client.get(f"/api/v1/documents/{doc_id}/status")
        assert chk_resp.status_code == 404

    # Assert 4: Inspect all points in Qdrant vector collection
    points, _ = await qdrant_client.scroll(
        collection_name=collection_name,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    assert len(points) > 0, "Expected non-zero vectors indexed in Qdrant for valid documents"

    indexed_doc_ids = {p.payload.get("doc_id") for p in points}

    # Zero ghost vectors from deleted documents
    for del_id in delete_target_doc_ids:
        assert del_id not in indexed_doc_ids, f"Ghost vector detected for deleted doc_id: {del_id}"

    # Zero vectors from poison documents
    for p_id in poison_doc_ids:
        assert p_id not in indexed_doc_ids, f"Poison doc left vectors in Qdrant: {p_id}"

    # All indexed points belong exclusively to the 15 ready documents
    assert indexed_doc_ids.issubset(valid_doc_ids), f"Unexpected doc_ids in Qdrant: {indexed_doc_ids - valid_doc_ids}"
    assert indexed_doc_ids == valid_doc_ids

    # Drain any remaining background tasks before asserting orchestrator state
    if app.state.background_tasks:
        await asyncio.gather(*list(app.state.background_tasks), return_exceptions=True)

    # Assert 5: Zero deadlocks and zero lingering active tasks in orchestrator
    worker_state = orchestrator.get_worker_state()
    assert worker_state["executing_tasks"] == 0, f"Lingering executing tasks: {worker_state}"
    assert worker_state["queued_tasks"] == 0, f"Lingering queued tasks: {worker_state}"

    # Assert 6: Hybrid retrieval queries return valid chunks only from the 15 ready documents
    retriever = HybridRetriever(
        qdrant=qdrant_client,
        embedder=app.state.embedder,
        sparse_encoder=app.state.sparse_encoder,
        collection=collection_name,
        limit=10,
    )
    search_results = await retriever.retrieve("distributed systems reliability architecture")
    assert len(search_results) > 0
    for passage in search_results:
        p_doc_id = passage.get("doc_id")
        assert p_doc_id in valid_doc_ids, f"Retrieved passage from non-valid doc: {p_doc_id}"
        assert p_doc_id not in delete_target_doc_ids
        assert p_doc_id not in poison_doc_ids
