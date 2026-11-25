"""Thread-safe reference-counted rotating log handler registry for desktop components."""
from __future__ import annotations

import logging
import os
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

_REGISTRY_LOCK = threading.Lock()
_HANDLER_REGISTRY: dict[str, tuple[RotatingFileHandler, int]] = {}


def _normalize_log_key(log_path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(str(log_path)))


def acquire_log_handler(
    log_path: Path | str,
    *,
    max_bytes: int = LOG_MAX_BYTES,
    backup_count: int = LOG_BACKUP_COUNT,
) -> RotatingFileHandler:
    """Acquire or reuse a refcounted `RotatingFileHandler` for `log_path`.

    Sharing one handler per normalized path prevents concurrent rollover
    conflicts (WinError 32) between the desktop app and sidecar supervisor.
    """
    target = Path(log_path)
    key = _normalize_log_key(target)

    with _REGISTRY_LOCK:
        existing = _HANDLER_REGISTRY.get(key)
        if existing is not None:
            handler, refcount = existing
            _HANDLER_REGISTRY[key] = (handler, refcount + 1)
            return handler

        target.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            target,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        _HANDLER_REGISTRY[key] = (handler, 1)
        return handler


def release_log_handler(log_path: Path | str) -> None:
    """Decrement refcount for `log_path` and close the underlying handle when it reaches zero."""
    key = _normalize_log_key(log_path)
    handler_to_close: RotatingFileHandler | None = None

    with _REGISTRY_LOCK:
        existing = _HANDLER_REGISTRY.get(key)
        if existing is None:
            return
        handler, refcount = existing
        if refcount > 1:
            _HANDLER_REGISTRY[key] = (handler, refcount - 1)
            return
        _HANDLER_REGISTRY.pop(key, None)
        handler_to_close = handler

    if handler_to_close is not None:
        try:
            handler_to_close.flush()
            handler_to_close.close()
        except Exception:
            pass
