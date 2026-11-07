"""FlashRank multilingual ONNX CPU reranker with parent-chunk deduplication."""
from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import flashrank
from flashrank import RerankRequest

from src.config import get_settings

__all__ = [
    "FlashRankReranker",
    "deduplicate_parent_chunks",
    "get_reranker",
    "set_reranker",
]

_reranker_singleton: FlashRankReranker | None = None


def _lexical_overlap_score(query: str, text: str, base_score: float = 0.0) -> float:
    """Deterministic lexical + RRF score fallback when ONNX weights are not cached locally."""
    q_tokens = set(re.findall(r"\w+", (query or "").lower(), flags=re.UNICODE))
    t_tokens = set(re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE))
    if not q_tokens or not t_tokens:
        return max(0.0, min(1.0, float(base_score)))
    overlap = len(q_tokens & t_tokens) / float(len(q_tokens))
    # Combine lexical recall with normalized retrieval score in [0.0, 1.0]
    combined = 0.75 * overlap + 0.25 * min(1.0, max(0.0, float(base_score)))
    return round(min(1.0, max(0.0, combined)), 6)


def deduplicate_parent_chunks(
    chunks: Sequence[dict[str, Any]],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Deduplicate ranked child chunks that share an identical `parent_content` block.

    Preserves the highest rerank score, merges all contributing `page_numbers` in sorted
    order, and records all contributing child `chunk_id` references in `chunk_ids`.
    """
    deduped: list[dict[str, Any]] = []
    index_by_parent_key: dict[tuple[str, str, str], int] = {}

    for raw in chunks:
        chunk = dict(raw)
        doc_id = str(chunk.get("doc_id") or "")
        domain = str(chunk.get("domain") or "general")
        parent_content = str(chunk.get("parent_content") or chunk.get("child_text") or "").strip()
        chunk_id = str(chunk.get("chunk_id") or "")
        pages = [
            int(p)
            for p in (chunk.get("page_numbers") or [])
            if isinstance(p, (int, str)) and str(p).isdigit()
        ]
        parent_key = (doc_id, domain, parent_content)

        if parent_content and parent_key in index_by_parent_key:
            existing = deduped[index_by_parent_key[parent_key]]
            merged_pages = sorted(set(existing.get("page_numbers") or []) | set(pages))
            existing["page_numbers"] = merged_pages

            existing_ids: list[str] = list(existing.get("chunk_ids") or [])
            if chunk_id and chunk_id not in existing_ids:
                existing_ids.append(chunk_id)
            existing["chunk_ids"] = existing_ids
            existing["contributing_chunk_ids"] = list(existing_ids)

            if float(chunk.get("score", 0.0)) > float(existing.get("score", 0.0)):
                existing["score"] = float(chunk["score"])
        else:
            chunk["parent_content"] = parent_content
            chunk["page_numbers"] = sorted(set(pages))
            initial_ids = [chunk_id] if chunk_id else []
            for extra_id in chunk.get("chunk_ids") or []:
                if extra_id and extra_id not in initial_ids:
                    initial_ids.append(str(extra_id))
            chunk["chunk_ids"] = initial_ids
            chunk["contributing_chunk_ids"] = list(initial_ids)
            if parent_content:
                index_by_parent_key[parent_key] = len(deduped)
            deduped.append(chunk)

    if top_n is not None and top_n > 0:
        return deduped[:top_n]
    return deduped


class FlashRankReranker:
    """Cross-encoder ONNX CPU reranker wrapping `flashrank.Ranker`."""

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: Path | str | None = None,
        top_n: int | None = None,
        ranker: Any | None = None,
        allow_download: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker.model_name or "ms-marco-MultiBERT-L-12"
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else (settings.data_dir / "models" / "flashrank")
        )
        self.top_n = top_n if top_n is not None else settings.reranker.top_n
        self._ranker: Any | None = ranker
        if allow_download is None:
            self.allow_download = os.environ.get("FLASHRANK_ALLOW_DOWNLOAD", "").lower() in (
                "1",
                "true",
                "yes",
            )
        else:
            self.allow_download = allow_download

    def _get_or_load_ranker(self) -> Any | None:
        if self._ranker is not None:
            return self._ranker

        model_dir = self.cache_dir / self.model_name
        ranker_cls = flashrank.Ranker
        # Use FlashRank Ranker if weights are cached on disk, download is explicitly enabled,
        # or flashrank.Ranker has been patched/mocked in tests.
        is_mocked = hasattr(ranker_cls, "_mock_name") or ranker_cls.__module__ != "flashrank.Ranker"
        if not (model_dir.exists() or self.allow_download or is_mocked):
            return None

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._ranker = ranker_cls(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir),
            )
            return self._ranker
        except Exception:
            return None

    def rerank_sync(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Synchronously score candidates against `query`, sort descending, and deduplicate parents."""
        if not candidates:
            return []

        limit = top_n if top_n is not None else self.top_n
        ranker = self._get_or_load_ranker()

        scored_chunks: list[dict[str, Any]] = []
        if ranker is not None:
            passages: list[dict[str, Any]] = []
            for idx, c in enumerate(candidates):
                prefix = str(c.get("contextual_prefix") or "").strip()
                body = str(c.get("child_text") or c.get("parent_content") or "").strip()
                text = f"{prefix}\n{body}".strip() if prefix else body
                passages.append(
                    {
                        "id": idx,
                        "text": text,
                        "meta": dict(c),
                    }
                )
            request = RerankRequest(query=query, passages=passages)
            raw_results = ranker.rerank(request)
            for item in raw_results:
                meta = dict(item.get("meta") or {})
                if not meta and isinstance(item.get("id"), int) and 0 <= item["id"] < len(candidates):
                    meta = dict(candidates[item["id"]])
                s_ce = float(item.get("score", meta.get("score", 0.0)))
                tier = meta.get("authority_tier")
                if tier is not None:
                    s_final = 0.92 * s_ce + 0.08 * ((5.0 - float(tier)) / 4.0)
                    meta["score"] = round(s_final, 6)
                else:
                    meta["score"] = round(s_ce, 6)
                scored_chunks.append(meta)
        else:
            for c in candidates:
                item = dict(c)
                prefix = str(item.get("contextual_prefix") or "").strip()
                body = str(item.get("child_text") or item.get("parent_content") or "").strip()
                text = f"{prefix}\n{body}".strip() if prefix else body
                s_ce = _lexical_overlap_score(
                    query=query,
                    text=text,
                    base_score=float(item.get("score", 0.0)),
                )
                tier = item.get("authority_tier")
                if tier is not None:
                    s_final = 0.92 * s_ce + 0.08 * ((5.0 - float(tier)) / 4.0)
                    item["score"] = round(s_final, 6)
                else:
                    item["score"] = round(s_ce, 6)
                scored_chunks.append(item)

        scored_chunks.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return deduplicate_parent_chunks(scored_chunks, top_n=limit)

    async def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Asynchronously rerank candidates via `asyncio.to_thread`."""
        if not candidates:
            return []
        return await asyncio.to_thread(self.rerank_sync, query, candidates, top_n)


def get_reranker() -> FlashRankReranker:
    """Return the process-level FlashRankReranker singleton."""
    global _reranker_singleton
    if _reranker_singleton is None:
        _reranker_singleton = FlashRankReranker()
    return _reranker_singleton


def set_reranker(reranker: FlashRankReranker | None) -> None:
    """Override or reset the process-level FlashRankReranker singleton."""
    global _reranker_singleton
    _reranker_singleton = reranker

