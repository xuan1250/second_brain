"""Hybrid Qdrant + SQLite indexer with pre-upsert abort barrier and orphan rollback."""
from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.db.models import Document, DocumentChunk, DomainRegistry
from src.db.qdrant import DENSE, SPARSE, init_qdrant_collection
from src.ingestion.chunker import ChunkPayload
from src.ingestion.entity_extractor import sync_chunk_entities_to_db
from src.ingestion.indexing.embedders import BaseEmbedder, get_embedder
from src.ingestion.indexing.sparse import BgeM3SparseEncoder, SparseEncoder, get_sparse_encoder

logger = logging.getLogger(__name__)


async def check_collection_sparse_mode_compatibility(
    client: AsyncQdrantClient,
    collection: str,
    active_sparse_mode: str,
) -> bool:
    """Check if existing Qdrant points have matching sparse_mode and log a warning if mismatched."""
    try:
        if not await client.collection_exists(collection):
            return True
        res, _ = await client.scroll(
            collection_name=collection,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not res:
            return True
        stored_mode = (res[0].payload or {}).get("sparse_mode")
        if stored_mode and stored_mode != active_sparse_mode:
            logger.warning(
                "Collection '%s' contains points indexed with sparse_mode='%s', but active sparse_mode is '%s'. "
                "Hybrid search recall may be degraded across incompatible sparse vector vocabularies. "
                "Re-indexing with 'uv run python -m src.ingestion.cli <path> --force' is recommended.",
                collection,
                stored_mode,
                active_sparse_mode,
            )
            return False
    except Exception as exc:
        logger.debug("Failed checking sparse_mode compatibility: %s", exc)
    return True


class HybridIndexer:
    """Indexes parent-child chunks into Qdrant (dense + sparse) and SQLite (DocumentChunk)."""

    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        embedder: BaseEmbedder | None = None,
        sparse_encoder: SparseEncoder | None = None,
        collection: str | None = None,
        batch_size: int = 32,
        clear_cuda_cache: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.qdrant = qdrant
        self.embedder = embedder or get_embedder()
        self.sparse_encoder = sparse_encoder or get_sparse_encoder()
        self.collection = collection or settings.qdrant_collection
        self.batch_size = max(1, batch_size)
        self.clear_cuda_cache = (
            clear_cuda_cache
            if clear_cuda_cache is not None
            else settings.ingestion.clear_cuda_cache_after_indexing
        )

    async def _rollback_qdrant_points(
        self,
        doc_id: str,
        upserted_point_ids: list[str],
        warning_msg: str,
    ) -> None:
        """Delete already-upserted Qdrant point IDs when indexing is aborted or fails mid-document."""
        if not upserted_point_ids:
            return
        logger.warning(warning_msg, len(upserted_point_ids), doc_id)
        try:
            await self.qdrant.delete(
                collection_name=self.collection,
                points_selector=models.PointIdsList(points=upserted_point_ids),
                wait=True,
            )
        except Exception as rollback_err:
            logger.error(
                "Failed to rollback Qdrant points for doc %s (%d points orphaned: %s): %s",
                doc_id,
                len(upserted_point_ids),
                upserted_point_ids[:10],
                rollback_err,
            )

    async def index_chunks(
        self,
        session: AsyncSession,
        doc_id: str,
        chunks: list[ChunkPayload],
    ) -> int:
        """Embed and upsert chunks in batches into Qdrant and record DocumentChunk rows in SQLite."""
        if not chunks:
            return 0

        await init_qdrant_collection(
            self.qdrant,
            collection=self.collection,
            dimensions=self.embedder.dimensions,
        )

        active_sparse_mode = "bge_m3" if isinstance(self.sparse_encoder, BgeM3SparseEncoder) else "bm25"
        await check_collection_sparse_mode_compatibility(
            self.qdrant,
            self.collection,
            active_sparse_mode=active_sparse_mode,
        )

        # Purge existing Qdrant vectors for doc_id to ensure retry idempotency.
        # Fail fast on error so ingestion aborts before modifying SQLite chunk records.
        await self.qdrant.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                )
            ),
            wait=True,
        )

        await session.execute(delete(DocumentChunk).where(DocumentChunk.doc_id == doc_id))
        await session.flush()

        db_rows: list[DocumentChunk] = []
        upserted_point_ids: list[str] = []

        try:
            for start in range(0, len(chunks), self.batch_size):
                batch = chunks[start : start + self.batch_size]
                contextualized_texts = [c.contextualized_text for c in batch]

                if isinstance(self.sparse_encoder, BgeM3SparseEncoder) and getattr(
                    self.embedder, "supports_native_sparse", False
                ):
                    dense_vectors, sparse_vectors = await self.embedder.embed_documents_with_sparse(contextualized_texts)
                else:
                    dense_vectors = await self.embedder.embed_documents(contextualized_texts)
                    sparse_vectors = await self.sparse_encoder.encode(contextualized_texts)

                points: list[models.PointStruct] = []
                batch_point_ids: list[str] = []
                for chunk, dense_vec, sparse_vec in zip(batch, dense_vectors, sparse_vectors):
                    point_id = str(uuid.UUID(chunk.chunk_id))
                    batch_point_ids.append(point_id)
                    payload: dict[str, Any] = {
                        "doc_id": doc_id,
                        "chunk_id": chunk.chunk_id,
                        "source_filename": chunk.source_filename,
                        "file_type": chunk.file_type,
                        "heading_path": chunk.heading_path,
                        "page_numbers": list(chunk.page_numbers),
                        "token_count": int(chunk.token_count),
                        "child_text": chunk.child_text,
                        "parent_content": chunk.parent_content,
                        "sparse_mode": active_sparse_mode,
                        "domain": chunk.domain_key or "general",
                        "subdomain": chunk.subdomain_key or "general",
                        "epistemic_type": chunk.epistemic_modality or "general",
                        "authority_tier": int(chunk.authority_tier),
                        "jurisdiction": chunk.jurisdiction,
                        "effective_year": chunk.effective_year,
                        "is_current": True,
                        "associated_entities": list(chunk.associated_entities),
                        "contextual_prefix": chunk.contextual_prefix,
                        "record_type": chunk.record_type,
                    }
                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector={
                                DENSE: dense_vec,
                                SPARSE: sparse_vec,
                            },
                            payload=payload,
                        )
                    )
                    page_str = ",".join(str(p) for p in chunk.page_numbers) if chunk.page_numbers else None
                    db_rows.append(
                        DocumentChunk(
                            id=chunk.chunk_id,
                            doc_id=doc_id,
                            point_id=point_id,
                            parent_id=chunk.parent_id,
                            chunk_index=chunk.chunk_index,
                            heading_path=chunk.heading_path,
                            page_numbers=page_str,
                            token_count=chunk.token_count,
                            text="",
                            parent_content="",
                            domain_key=chunk.domain_key or "general",
                            subdomain_key=chunk.subdomain_key or "general",
                            epistemic_modality=chunk.epistemic_modality or "general",
                            contextual_prefix=chunk.contextual_prefix,
                        )
                    )

                # Pre-upsert SQLite barrier: check document existence and valid status
                doc = await session.get(Document, doc_id, populate_existing=True)
                if doc is None or doc.status in ("failed", "cancelled"):
                    logger.warning(
                        "Document %s was deleted or aborted before Qdrant upsert; cancelling vector indexing",
                        doc_id,
                    )
                    await self._rollback_qdrant_points(
                        doc_id,
                        upserted_point_ids,
                        "Rolling back %d previously upserted points for aborted doc %s",
                    )
                    return 0

                await self.qdrant.upsert(
                    collection_name=self.collection,
                    points=points,
                    wait=True,
                )
                upserted_point_ids.extend(batch_point_ids)

            session.add_all(db_rows)
            try:
                dom_res = await session.execute(select(DomainRegistry.id, DomainRegistry.slug))
                slug_to_id = {row[1]: row[0] for row in dom_res.all()}
                for c in chunks:
                    if c.associated_entities:
                        d_id = slug_to_id.get(c.domain_key or "general")
                        if d_id:
                            await sync_chunk_entities_to_db(
                                session,
                                chunk_id=c.chunk_id,
                                domain_id=d_id,
                                entities=c.associated_entities,
                            )
            except Exception as ent_err:
                logger.debug("Entity database sync skipped: %s", ent_err)
            await session.commit()
            return len(db_rows)
        except Exception:
            await self._rollback_qdrant_points(
                doc_id,
                upserted_point_ids,
                "Indexing failed; rolling back %d orphan Qdrant points for doc %s",
            )
            raise
        finally:
            if self.clear_cuda_cache:
                from src.ingestion import indexer as indexer_mod

                indexer_mod.release_cuda_cache()
