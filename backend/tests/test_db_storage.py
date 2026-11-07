import hashlib
import os

import pytest
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import load_settings
from src.db.models import ChatThread, Document, DocumentChunk, IngestionJob
from src.db.qdrant import init_qdrant_collection
from src.db.repository import delete_document_atomic
from src.db.session import create_engine, create_session_factory, init_db

COLLECTION = "test_collection"


@pytest.fixture
async def sqlite(tmp_path):
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'db' / 'sb.db').as_posix()}")
    await init_db(engine)
    yield engine, create_session_factory(engine)
    await engine.dispose()


def test_settings_defaults_and_env_override(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("DB_DIR", "/app/data/db")
    s = load_settings()
    assert s.embedding.model_name == "BAAI/bge-m3" and s.embedding.dimensions == 1024
    assert s.reranker.model_name == "ms-marco-MultiBERT-L-12"
    assert s.ingestion.max_workers == 1
    assert s.qdrant_url == "http://qdrant:6333"
    assert s.storage.db_dir == "/app/data/db"


async def test_tables_wal_and_fk(sqlite):
    engine, _ = sqlite
    async with engine.connect() as c:
        tables = {r[0] for r in await c.execute(text("select name from sqlite_master where type='table'"))}
        assert {"documents", "document_chunks", "ingestion_jobs", "chat_threads"} <= tables
        assert (await c.execute(text("PRAGMA journal_mode"))).scalar() == "wal"
        assert (await c.execute(text("PRAGMA foreign_keys"))).scalar() == 1


async def test_cascade_delete(sqlite):
    _, factory = sqlite
    async with factory() as s:
        doc = Document(filename="a.pdf", file_type="pdf", sha256="a" * 64)
        s.add(doc)
        await s.commit()
        s.add(DocumentChunk(doc_id=doc.id, point_id="p1", text="x"))
        s.add(IngestionJob(doc_id=doc.id))
        s.add(ChatThread(title="t"))
        await s.commit()
        await s.execute(text("DELETE FROM documents WHERE id=:i"), {"i": doc.id})
        await s.commit()
        assert (await s.execute(text("select count(*) from document_chunks"))).scalar() == 0
        assert (await s.execute(text("select count(*) from ingestion_jobs"))).scalar() == 0


async def _check_collection(client):
    await init_qdrant_collection(client, COLLECTION, 1024)
    await init_qdrant_collection(client, COLLECTION, 1024)  # idempotent
    info = await client.get_collection(COLLECTION)
    assert info.config.params.vectors["dense"].size == 1024
    assert info.config.params.vectors["dense"].distance == models.Distance.COSINE
    assert info.config.params.sparse_vectors["sparse"].modifier == models.Modifier.IDF


async def test_qdrant_collection_memory():
    client = AsyncQdrantClient(location=":memory:")
    await _check_collection(client)
    await client.close()


async def test_atomic_delete(sqlite, tmp_path):
    _, factory = sqlite
    client = AsyncQdrantClient(location=":memory:")
    await init_qdrant_collection(client, COLLECTION, 4)
    sha = hashlib.sha256(b"hello").hexdigest()
    data = tmp_path / "data"
    for sub, name in (("raw", f"{sha}.pdf"), ("parsed", f"{sha}.json"), ("markdown", f"{sha}.md")):
        (data / sub).mkdir(parents=True)
        (data / sub / name).write_text("x")
    async with factory() as s:
        doc = Document(filename="a.pdf", file_type="pdf", sha256=sha)
        s.add(doc)
        await s.commit()
        s.add(DocumentChunk(doc_id=doc.id, point_id="00000000-0000-0000-0000-000000000001"))
        await s.commit()
        await client.upsert(
            COLLECTION,
            points=[
                models.PointStruct(id="00000000-0000-0000-0000-000000000001", vector={"dense": [1, 0, 0, 0]}, payload={"doc_id": doc.id}),
                models.PointStruct(id="00000000-0000-0000-0000-000000000002", vector={"dense": [0, 1, 0, 0]}, payload={"doc_id": "other"}),
            ],
        )
        assert await delete_document_atomic(s, client, doc.id, data_dir=data, collection=COLLECTION)
        assert (await client.count(COLLECTION)).count == 1
        assert (await s.execute(text("select count(*) from documents"))).scalar() == 0
        assert (await s.execute(text("select count(*) from document_chunks"))).scalar() == 0
        assert not any(data.rglob(f"{sha}*"))
        assert not await delete_document_atomic(s, client, doc.id, data_dir=data, collection=COLLECTION)
    await client.close()


@pytest.mark.live
async def test_qdrant_collection_live():
    client = AsyncQdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    await client.delete_collection(COLLECTION)
    await _check_collection(client)
    await client.delete_collection(COLLECTION)
    await client.close()

