"""FastAPI dependency providers for SQLite sessions, Qdrant, IngestionOrchestrator, embedder, and LangGraph."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import Request
from langgraph.checkpoint.base import BaseCheckpointSaver
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import Settings, get_settings
from src.db.session import get_session_factory
from src.ingestion.indexer import (
    BaseEmbedder,
    SparseEncoder,
    get_embedder,
    get_sparse_encoder,
)
from src.ingestion.worker import IngestionOrchestrator
from src.rag.graph import build_rag_graph, get_checkpointer
from src.rag.reranker import FlashRankReranker, get_reranker
from src.rag.retriever import HybridRetriever, get_active_qdrant_client

__all__ = [
    "get_active_checkpointer",
    "get_app_settings",
    "get_db_session",
    "get_db_session_factory",
    "get_orchestrator",
    "get_qdrant",
    "get_rag_app",
    "get_shared_embedder",
    "get_shared_reranker",
    "get_shared_sparse_encoder",
]


def get_app_settings(request: Request) -> Settings:
    """Return the active Settings instance from `app.state` or global settings."""
    state_settings = getattr(request.app.state, "settings", None)
    return state_settings if state_settings is not None else get_settings()


def get_db_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    """Return the active async_sessionmaker from `app.state` or the global session factory."""
    factory = getattr(request.app.state, "session_factory", None)
    return factory if factory is not None else get_session_factory()


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield a scoped AsyncSession for the current request."""
    factory = get_db_session_factory(request)
    async with factory() as session:
        yield session


def get_qdrant(request: Request) -> AsyncQdrantClient:
    """Return the shared AsyncQdrantClient from `app.state` or retriever singleton."""
    client = getattr(request.app.state, "qdrant", None)
    return client if client is not None else get_active_qdrant_client()


def get_shared_embedder(request: Request) -> BaseEmbedder:
    """Return the process-level shared dense embedder (`BAAI/bge-m3` or configured override)."""
    embedder = getattr(request.app.state, "embedder", None)
    return embedder if embedder is not None else get_embedder()


def get_shared_sparse_encoder(request: Request) -> SparseEncoder:
    """Return the process-level shared sparse BM25 encoder."""
    encoder = getattr(request.app.state, "sparse_encoder", None)
    return encoder if encoder is not None else get_sparse_encoder()


def get_shared_reranker(request: Request) -> FlashRankReranker:
    """Return the shared FlashRank reranker instance."""
    reranker = getattr(request.app.state, "reranker", None)
    return reranker if reranker is not None else get_reranker()


def get_orchestrator(request: Request) -> IngestionOrchestrator:
    """Return the shared `IngestionOrchestrator` managed by the FastAPI lifespan."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is not None:
        return orchestrator
    settings = get_app_settings(request)
    orchestrator = IngestionOrchestrator(
        session_factory=get_db_session_factory(request),
        qdrant=get_qdrant(request),
        embedder=get_shared_embedder(request),
        sparse_encoder=get_shared_sparse_encoder(request),
        data_dir=settings.data_dir,
        collection=settings.qdrant_collection,
        max_workers=settings.ingestion.max_workers,
    )
    request.app.state.orchestrator = orchestrator
    return orchestrator


async def get_active_checkpointer(request: Request) -> BaseCheckpointSaver:
    """Return the persistent `AsyncSqliteSaver` checkpointer."""
    saver = getattr(request.app.state, "checkpointer", None)
    if saver is not None:
        return saver
    settings = get_app_settings(request)
    saver = await get_checkpointer(settings.db_dir / "chat_history.db")
    request.app.state.checkpointer = saver
    return saver


async def get_rag_app(request: Request) -> Any:
    """Return a compiled LangGraph CRAG StateGraph wired to the shared checkpointer and retriever."""
    saver = await get_active_checkpointer(request)
    settings = get_app_settings(request)
    retriever = HybridRetriever(
        qdrant=get_qdrant(request),
        embedder=get_shared_embedder(request),
        sparse_encoder=get_shared_sparse_encoder(request),
        collection=settings.qdrant_collection,
    )
    return build_rag_graph(
        retriever=retriever,
        reranker=get_shared_reranker(request),
        checkpointer=saver,
        collection=settings.qdrant_collection,
    )

