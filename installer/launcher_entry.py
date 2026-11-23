"""Consoleless `pythonw.exe` launcher entrypoint for Second Brain Desktop."""
from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path


def _guard_windowless_stdio() -> None:
    """Redirect missing `stdout`/`stderr` streams under `pythonw.exe` to `os.devnull`."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _configure_runtime_paths() -> Path:
    """Resolve packaged (`{app}\\app\\backend`) or repo (`<repo>\\backend`) layout and wire `sys.path`."""
    entry_path = Path(__file__).resolve()
    app_or_repo_root = entry_path.parents[1]
    backend_dir = app_or_repo_root / "backend"

    if len(entry_path.parents) > 2:
        install_root = entry_path.parents[2]
        packaged_app_dir = install_root / "app"
        if app_or_repo_root == packaged_app_dir and (packaged_app_dir / "backend").is_dir():
            backend_dir = packaged_app_dir / "backend"
            if not (os.environ.get("SECOND_BRAIN_APP_DIR") or "").strip():
                os.environ["SECOND_BRAIN_APP_DIR"] = str(install_root)

    backend_str = str(backend_dir)
    normalized_backend = os.path.normcase(os.path.abspath(backend_str))
    existing_entries = {
        os.path.normcase(os.path.abspath(item))
        for item in sys.path
        if isinstance(item, str) and item
    }
    if normalized_backend not in existing_entries:
        sys.path.insert(0, backend_str)

    return backend_dir


def main() -> int:
    """Initialize frozen/spawn safety, stdio guards, and invoke `src.desktop.app.main()`."""
    multiprocessing.freeze_support()
    _guard_windowless_stdio()
    _configure_runtime_paths()

    from src.desktop.app import main as desktop_main

    return int(desktop_main())


if __name__ == "__main__":
    raise SystemExit(main())
