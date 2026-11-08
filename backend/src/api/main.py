"""FastAPI Headless Gateway & Streaming API with lifespan resource management and SPA static serving."""
from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from qdrant_client import AsyncQdrantClient

from src.api.routes import chat, documents, domains, system
from src.config import get_settings, load_settings
from src.db import session as db_session_mod
from src.db.qdrant import backfill_qdrant_legacy_payloads, get_qdrant_client, init_qdrant_collection
from src.db.session import close_engine, create_engine, create_session_factory, init_db
from src.ingestion.indexer import (
    BgeM3SparseEncoder,
    check_collection_sparse_mode_compatibility,
    get_embedder,
    get_sparse_encoder,
)
from src.ingestion.worker import IngestionOrchestrator
from src.rag.graph import close_checkpointer, get_checkpointer
from src.rag.reranker import get_reranker
from src.rag import retriever as retriever_mod
from src.rag.retriever import set_active_qdrant_client

CORS_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
]


def mount_frontend_if_present(app: FastAPI, dist_dir: Path) -> bool:
    """Mount compiled React SPA (`frontend/dist`) at `/` if the directory exists and is not yet mounted."""
    if not dist_dir.is_dir():
        return False
    for route in app.routes:
        if getattr(route, "name", None) == "frontend":
            return True
    app.mount(
        "/",
        StaticFiles(directory=str(dist_dir), html=True),
        name="frontend",
    )
    return True


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Coordinate startup and shutdown of SQLite, Qdrant, ProcessPoolExecutor, and LangGraph checkpointer."""
    get_settings.cache_clear()
    settings = load_settings()

    # 1. Ensure storage and database directories exist
    for sub in ("raw", "parsed", "markdown", "models"):
        (settings.data_dir / sub).mkdir(parents=True, exist_ok=True)
    settings.db_dir.mkdir(parents=True, exist_ok=True)

    # 2. Initialize SQLite engine & session factory (WAL + FK enforcement)
    engine = create_engine(settings.sqlite_url)
    await init_db(engine)
    session_factory = create_session_factory(engine)
    db_session_mod._engine = engine
    db_session_mod._engine_url = settings.sqlite_url
    db_session_mod._factory = session_factory

    # 3. Resolve shared main-process embedder, sparse encoder, and FlashRank reranker
    embedder = get_embedder()
    sparse_encoder = get_sparse_encoder()
    reranker = get_reranker()

    # 4. Initialize Qdrant client and hybrid collection + payload indexes
    preconfigured_qdrant: AsyncQdrantClient | None = (
        getattr(app.state, "qdrant", None) or retriever_mod._qdrant_client_override
    )
    if preconfigured_qdrant is not None:
        qdrant_client = preconfigured_qdrant
        owns_qdrant = False
    elif settings.qdrant_url == ":memory:":
        qdrant_client = AsyncQdrantClient(location=":memory:")
        owns_qdrant = True
    else:
        qdrant_client = get_qdrant_client(settings.qdrant_url)
        owns_qdrant = True

    set_active_qdrant_client(qdrant_client)
    try:
        await init_qdrant_collection(
            qdrant_client,
            collection=settings.qdrant_collection,
            dimensions=embedder.dimensions,
        )
        await backfill_qdrant_legacy_payloads(
            qdrant_client,
            collection_name=settings.qdrant_collection,
        )
        active_sparse = "bge_m3" if isinstance(sparse_encoder, BgeM3SparseEncoder) else "bm25"
        await check_collection_sparse_mode_compatibility(
            qdrant_client,
            settings.qdrant_collection,
            active_sparse_mode=active_sparse,
        )
    except Exception:
        # Allow app startup to proceed even if external Qdrant is still warming up;
        # `/api/v1/system/health` will report degraded status until Qdrant is reachable.
        pass

    # 5. Initialize persistent LangGraph AsyncSqliteSaver checkpointer (`chat_history.db`)
    checkpointer = await get_checkpointer(settings.db_dir / "chat_history.db")

    # 6. Initialize IngestionOrchestrator (spawn pool for parse+chunk; main process for indexing)
    use_pool = os.environ.get("INGESTION_USE_PROCESS_POOL", "true").lower() not in (
        "0",
        "false",
        "no",
    )
    orchestrator = IngestionOrchestrator(
        session_factory=session_factory,
        qdrant=qdrant_client,
        embedder=embedder,
        sparse_encoder=sparse_encoder,
        data_dir=settings.data_dir,
        collection=settings.qdrant_collection,
        max_workers=settings.ingestion.max_workers,
        use_process_pool=use_pool,
    )

    # 7. Populate app.state and mount frontend static bundle if present
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.qdrant = qdrant_client
    app.state.embedder = embedder
    app.state.sparse_encoder = sparse_encoder
    app.state.reranker = reranker
    app.state.checkpointer = checkpointer
    app.state.orchestrator = orchestrator
    app.state.background_tasks: set[asyncio.Task[Any]] = set()

    mount_frontend_if_present(app, settings.frontend_dist_dir)

    try:
        yield
    finally:
        # Drain in-flight background ingestion tasks before closing stores
        bg_tasks: set[asyncio.Task[Any]] = getattr(app.state, "background_tasks", set())
        if bg_tasks:
            await asyncio.gather(*list(bg_tasks), return_exceptions=True)

        orchestrator.shutdown(wait=True)
        await close_checkpointer()

        if owns_qdrant:
            try:
                await qdrant_client.close()
            except Exception:
                pass
            set_active_qdrant_client(None)

        await close_engine()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    application = FastAPI(
        title="Second Brain Knowledge Repository API",
        version="0.1.0",
        description=(
            "Headless FastAPI gateway for multi-format document ingestion (Docling v2 + Qdrant hybrid) "
            "and LangGraph Corrective RAG conversational streaming."
        ),
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(documents.router, prefix="/api/v1")
    application.include_router(domains.router, prefix="/api/v1")
    application.include_router(chat.router, prefix="/api/v1")
    application.include_router(system.router, prefix="/api/v1")

    settings = get_settings()
    mount_frontend_if_present(application, settings.frontend_dist_dir)

    return application


app = create_app()

