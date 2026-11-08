"""Qdrant client + hybrid collection bootstrap and payload partitioning."""
from __future__ import annotations

import warnings
from qdrant_client import AsyncQdrantClient, models

from src.config import get_settings

DENSE = "dense"
SPARSE = "sparse"


def get_qdrant_client(url: str | None = None) -> AsyncQdrantClient:
    return AsyncQdrantClient(url=url or get_settings().qdrant_url)


async def init_qdrant_collection(
    client: AsyncQdrantClient,
    collection: str | None = None,
    dimensions: int | None = None,
) -> None:
    """Idempotently create the hybrid collection and register multi-domain payload indexes."""
    s = get_settings()
    collection = collection or s.qdrant_collection
    dimensions = dimensions or s.embedding.dimensions
    if not await client.collection_exists(collection):
        await client.create_collection(
            collection_name=collection,
            vectors_config={DENSE: models.VectorParams(size=dimensions, distance=models.Distance.COSINE, on_disk=True)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            hnsw_config=models.HnswConfigDiff(on_disk=True),
        )
    else:
        try:
            await client.update_collection(
                collection_name=collection,
                vectors_config={DENSE: models.VectorParamsDiff(on_disk=True)},
                hnsw_config=models.HnswConfigDiff(on_disk=True),
            )
        except Exception:
            try:
                await client.update_collection(
                    collection_name=collection,
                    vectors_config=models.VectorParamsDiff(on_disk=True),
                    hnsw_config=models.HnswConfigDiff(on_disk=True),
                )
            except Exception as exc:
                warnings.warn(f"Failed or skipped updating on_disk config for collection '{collection}': {exc}", stacklevel=2)

    # Core indices + multi-domain partitioned indices
    payload_indexes: list[tuple[str, models.PayloadSchemaType]] = [
        ("doc_id", models.PayloadSchemaType.KEYWORD),
        ("file_type", models.PayloadSchemaType.KEYWORD),
        ("heading_path", models.PayloadSchemaType.TEXT),
        ("domain", models.PayloadSchemaType.KEYWORD),
        ("subdomain", models.PayloadSchemaType.KEYWORD),
        ("epistemic_type", models.PayloadSchemaType.KEYWORD),
        ("authority_tier", models.PayloadSchemaType.INTEGER),
        ("jurisdiction", models.PayloadSchemaType.KEYWORD),
        ("effective_year", models.PayloadSchemaType.INTEGER),
        ("is_current", models.PayloadSchemaType.BOOL),
        ("associated_entities", models.PayloadSchemaType.KEYWORD),
    ]

    for field, schema in payload_indexes:
        try:
            await client.create_payload_index(collection, field_name=field, field_schema=schema)
        except Exception as exc:
            # Handle cases where index might already exist or warnings on in-memory mock
            warnings.warn(f"Failed or skipped creating payload index '{field}': {exc}", stacklevel=2)


async def backfill_qdrant_legacy_payloads(
    client: AsyncQdrantClient,
    collection_name: str | None = None,
    batch_size: int = 100,
) -> int:
    """Backfill points missing 'domain' with default general partition payload. Zero re-embedding."""
    s = get_settings()
    collection = collection_name or s.qdrant_collection
    if not await client.collection_exists(collection):
        return 0

    total_backfilled = 0
    default_payload = {
        "domain": "general",
        "subdomain": "general.general",
        "epistemic_type": "general",
        "authority_tier": 3,
        "is_current": True,
    }

    while True:
        points, _ = await client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[models.IsEmptyCondition(is_empty=models.PayloadField(key="domain"))]
            ),
            limit=batch_size,
            with_payload=False,
            with_vectors=False,
        )
        if not points:
            break
        point_ids = [p.id for p in points]
        await client.set_payload(
            collection_name=collection,
            payload=default_payload,
            points=point_ids,
        )
        total_backfilled += len(point_ids)

    return total_backfilled
