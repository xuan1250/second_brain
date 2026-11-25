"""Stage runner facade for ingestion orchestrator."""
from __future__ import annotations

from src.ingestion.orchestrator.stage_classification import run_process_to_classification
from src.ingestion.orchestrator.stage_indexing import (
    run_complete_approved_ingestion,
    run_process_document,
)

__all__ = [
    "run_complete_approved_ingestion",
    "run_process_document",
    "run_process_to_classification",
]
