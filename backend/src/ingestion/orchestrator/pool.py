"""Worker pool lifecycle, concurrency slots, pause/resume, and cancellation state mixin."""
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)


class WorkerPoolMixin:
    """Mixin providing ProcessPoolExecutor lifecycle, slot tracking, and cancellation primitives."""

    max_workers: int
    worker_idle_timeout_sec: float
    worker_max_tasks: int

    def _init_pool_state(self) -> None:
        self._pool: ProcessPoolExecutor | None = None
        self._pool_completed_docs = 0
        self._pool_reclaim_count = 0
        self._executing_tasks = 0
        self._active_worker_jobs = 0
        self._active_indexing_jobs = 0
        self._last_worker_activity_at = time.time()
        self._idle_timer_handle: asyncio.TimerHandle | None = None
        self._reclaim_on_idle = False

        self._doc_semaphore = asyncio.Semaphore(self.max_workers)
        self._queued_tasks = 0
        self._cancellations: dict[str, asyncio.Event] = {}
        self._parsed_cache: dict[str, Any] = {}
        self._is_paused = False
        self._resume_event = asyncio.Event()
        self._resume_event.set()

    def _get_or_create_cancel_event(self, doc_id: str) -> asyncio.Event:
        """Return an existing set cancellation event or register a fresh unset event."""
        existing = self._cancellations.get(doc_id)
        if existing is not None and existing.is_set():
            return existing
        ev = asyncio.Event()
        self._cancellations[doc_id] = ev
        return ev

    def cancel_document(self, doc_id: str) -> None:
        """Signal in-flight cancellation for a queued or executing document."""
        self._parsed_cache.pop(doc_id, None)
        if doc_id in self._cancellations:
            logger.info("Signalling cancellation for doc_id=%s", doc_id)
            self._cancellations[doc_id].set()
        else:
            ev = asyncio.Event()
            ev.set()
            self._cancellations[doc_id] = ev

    def pause(self) -> None:
        self._is_paused = True
        self._resume_event.clear()
        logger.info("Ingestion orchestrator paused")

    def resume(self) -> None:
        self._is_paused = False
        self._resume_event.set()
        logger.info("Ingestion orchestrator resumed")

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    def get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            ctx = multiprocessing.get_context("spawn")
            self._pool = ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx)
        return self._pool

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer_handle is not None:
            self._idle_timer_handle.cancel()
            self._idle_timer_handle = None

    def _acquire_worker_slot(self) -> None:
        self._executing_tasks += 1
        self._active_worker_jobs = self._executing_tasks
        self._cancel_idle_timer()

    def _release_worker_slot(self) -> None:
        self._executing_tasks = max(0, self._executing_tasks - 1)
        self._active_worker_jobs = self._executing_tasks
        self._last_worker_activity_at = time.time()
        if self._executing_tasks == 0 and self._pool is not None:
            if self._reclaim_on_idle or (
                self.worker_max_tasks > 0 and self._pool_completed_docs >= self.worker_max_tasks
            ):
                self._reclaim_on_idle = False
                self.reclaim_worker_pool(wait=False, force=False)
            elif self.worker_idle_timeout_sec > 0:
                try:
                    loop = asyncio.get_running_loop()
                    self._idle_timer_handle = loop.call_later(
                        self.worker_idle_timeout_sec, self._on_idle_timeout
                    )
                except RuntimeError:
                    pass

    def _on_idle_timeout(self) -> None:
        logger.info(
            "Worker pool idle timeout (%ss) reached; reclaiming worker pool",
            self.worker_idle_timeout_sec,
        )
        self.reclaim_worker_pool(wait=False, force=False)

    def reclaim_worker_pool(self, *, wait: bool = False, force: bool = False) -> bool:
        self._cancel_idle_timer()
        if (self._executing_tasks > 0 or self._active_worker_jobs > 0) and not force:
            if self._pool is not None:
                self._reclaim_on_idle = True
            return False
        if self._pool is not None:
            self._reclaim_on_idle = False
            self._pool_completed_docs = 0
            self._pool.shutdown(wait=wait, cancel_futures=False)
            self._pool = None
            self._pool_reclaim_count += 1
            logger.info("Worker pool successfully reclaimed (reclaim_count=%d)", self._pool_reclaim_count)
            return True
        return False

    def get_worker_state(self) -> dict[str, Any]:
        return {
            "worker_pool_active": self._pool is not None,
            "ingestion_paused": self._is_paused,
            "queued_tasks": self._queued_tasks,
            "executing_tasks": self._executing_tasks,
            "active_worker_jobs": self._active_worker_jobs,
            "active_indexing_jobs": self._active_indexing_jobs,
            "worker_reclaim_count": self._pool_reclaim_count,
            "reclaim_deferred": self._reclaim_on_idle,
            "worker_idle_timeout_sec": self.worker_idle_timeout_sec,
            "worker_max_tasks": self.worker_max_tasks,
        }

    def shutdown(self, wait: bool = True) -> None:
        self._cancel_idle_timer()
        self._reclaim_on_idle = False
        self.reclaim_worker_pool(wait=wait, force=True)
