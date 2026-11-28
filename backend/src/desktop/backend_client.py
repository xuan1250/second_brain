"""Loopback HTTP client helpers for desktop-to-sidecar reindex and memory reclaim."""
from __future__ import annotations

import json
import logging
import threading
import urllib.error
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.desktop.app import DesktopApp


def post_loopback_json(url: str, *, timeout: float = 15.0) -> dict[str, Any]:
    """Send an empty-body HTTP POST to a loopback backend endpoint and parse JSON."""
    from src.desktop import app as app_mod

    req = app_mod.urllib.request.Request(url, data=b"", method="POST")
    with app_mod.urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return payload


def trigger_corpus_reindex(
    api_url: str,
    *,
    log_fn: Callable[..., None] | None = None,
) -> tuple[int, str | None]:
    """Request background re-indexing on the restarted backend after sparse-mode change."""
    url = f"{api_url.rstrip('/')}/api/v1/system/reindex"
    try:
        payload = post_loopback_json(url, timeout=15.0)
        return int(payload.get("queued_count", 0)), None
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        if log_fn is not None:
            log_fn(logging.WARNING, "Corpus re-index dispatch failed: %s", exc)
        return 0, str(exc)


def reclaim_memory_async(app: DesktopApp) -> None:
    """Trigger `/api/v1/system/memory/reclaim` in a background thread and notify via tray."""

    def _worker() -> None:
        if not app.supervisor.is_backend_running:
            app.tray.notify("Backend service is not running yet.", title="Second Brain")
            return

        url = f"{app.supervisor.api_url}/api/v1/system/memory/reclaim"
        try:
            payload = post_loopback_json(url, timeout=15.0)
            mem = payload.get("memory") or {}
            rss_mb = mem.get("process_rss_mb")
            rss_text = f"{float(rss_mb):.0f} MB RSS" if rss_mb is not None else "RSS updated"
            if payload.get("deferred"):
                active_jobs = payload.get("active_worker_jobs", 0)
                msg = f"Ingestion active ({active_jobs} job(s)); worker reclaim deferred ({rss_text})."
            else:
                parts: list[str] = []
                if payload.get("worker_pool_reclaimed"):
                    parts.append("worker pool reclaimed")
                else:
                    parts.append("worker pool idle")
                if payload.get("cuda_cache_cleared"):
                    parts.append("VRAM cache cleared")
                parts.append(rss_text)
                if mem.get("cuda_available") and mem.get("cuda_allocated_mb") is not None:
                    parts.append(f"CUDA {float(mem['cuda_allocated_mb']):.0f} MB")
                msg = " | ".join(parts)
            app.tray.notify(msg, title="Second Brain — Memory Reclaim")
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            app._log(logging.WARNING, "Tray memory reclaim failed: %s", exc)
            app.tray.notify(f"Memory reclaim failed: {exc}", title="Second Brain")

    threading.Thread(target=_worker, name="desktop-memory-reclaim", daemon=True).start()
