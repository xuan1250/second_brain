"""Atomic file I/O helpers for desktop configuration, control files, and manifests."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path | str, data: Any, *, fsync: bool = True) -> Path:
    """Atomically write JSON data to `path` using a same-directory temp file and `os.replace`."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(data, "model_dump_json"):
        payload = data.model_dump_json(indent=2) + "\n"
    else:
        payload = json.dumps(data, indent=2) + "\n"

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            prefix=f".{target.stem}_",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(payload)
            tmp_file.flush()
            if fsync:
                os.fsync(tmp_file.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass
        raise

    return target
