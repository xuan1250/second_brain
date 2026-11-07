"""CLI entry point for batch file/folder ingestion (`uv run python -m src.ingestion.cli <path>`)."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.db.qdrant import get_qdrant_client, init_qdrant_collection
from src.db.session import get_engine, get_session_factory, init_db
from src.ingestion.indexer import DeterministicBM25Encoder, DeterministicStubEmbedder
from src.ingestion.parser import SUPPORTED_EXTENSIONS
from src.ingestion.worker import IngestionOrchestrator

console = Console()


def collect_target_files(target: Path) -> list[Path]:
    """Collect supported document files from a single path or recursively from a directory."""
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {target}")
    if target.is_file():
        if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {target.suffix}")
        return [target]
    files = [
        p
        for p in sorted(target.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


async def run_batch_ingestion(
    target_path: Path,
    *,
    force_reparse: bool = False,
    stub_embedder: bool = False,
    in_memory_qdrant: bool = False,
    use_process_pool: bool = True,
    approval_mode: str = "auto_all",
) -> int:
    """Ingest all supported files under target_path with Rich progress reporting."""
    files = collect_target_files(target_path)
    if not files:
        console.print(f"[yellow]No supported documents found in {target_path}[/yellow]")
        return 0

    engine = get_engine()
    await init_db(engine)
    session_factory = get_session_factory()

    if in_memory_qdrant:
        from qdrant_client import AsyncQdrantClient

        qdrant = AsyncQdrantClient(location=":memory:")
    else:
        qdrant = get_qdrant_client()

    embedder = DeterministicStubEmbedder(dimensions=1024) if stub_embedder else None
    sparse = DeterministicBM25Encoder() if stub_embedder else None

    results: list[tuple[str, str, int, str]] = []
    failures = 0

    try:
        await init_qdrant_collection(qdrant)
        async with IngestionOrchestrator(
            session_factory=session_factory,
            qdrant=qdrant,
            embedder=embedder,
            sparse_encoder=sparse,
            use_process_pool=use_process_pool,
        ) as orchestrator:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task("Ingesting documents...", total=len(files))
                for fpath in files:
                    progress.update(task_id, description=f"Processing [cyan]{fpath.name}[/cyan]")
                    try:
                        doc = await orchestrator.ingest_file(
                            fpath,
                            force_reparse=force_reparse,
                            approval_mode=approval_mode,
                        )
                        results.append((doc.filename, doc.file_type, doc.chunk_count, doc.status))
                    except Exception as exc:
                        failures += 1
                        results.append((fpath.name, fpath.suffix.lstrip("."), 0, f"failed: {exc}"))
                    progress.advance(task_id)
    finally:
        await qdrant.close()

    table = Table(title="Batch Ingestion Summary")
    table.add_column("Filename", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Chunks", justify="right", style="green")
    table.add_column("Status", style="bold")

    for fname, ftype, chunks, status in results:
        table.add_row(fname, ftype, str(chunks), status)

    console.print(table)
    return 1 if failures > 0 else 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ingestion.cli",
        description="Batch ingest a document file or folder into the Second Brain knowledge repository.",
    )
    parser.add_argument("path", type=Path, help="Path to a document file or directory of documents")
    parser.add_argument("--force", action="store_true", help="Force re-parsing even if SHA-256 is cached/ready")
    parser.add_argument("--stub-embedder", action="store_true", help="Use deterministic stub embedder (offline)")
    parser.add_argument("--in-memory-qdrant", action="store_true", help="Use in-memory Qdrant instance")
    parser.add_argument("--no-pool", action="store_true", help="Run parse/chunk in thread instead of spawn pool")
    parser.add_argument(
        "--approval-mode",
        choices=["review_all", "auto_high_confidence", "auto_all"],
        default="auto_all",
        help="Approval mode for classification gate (default: auto_all)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(
            run_batch_ingestion(
                args.path,
                force_reparse=args.force,
                stub_embedder=args.stub_embedder,
                in_memory_qdrant=args.in_memory_qdrant,
                use_process_pool=not args.no_pool,
                approval_mode=args.approval_mode,
            )
        )
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
