"""Document listing, inspection, approval, metadata update, and deletion schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentResponse(BaseModel):
    """Serialized representation of a Document and its latest IngestionJob state."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: str
    sha256: str
    size_bytes: int = 0
    page_count: int | None = None
    chunk_count: int = 0
    status: str = "queued"
    error: str | None = None
    error_message: str | None = None
    job_id: str | None = None
    stage: str | None = None
    progress: float = 0.0
    duration_ms: int | None = None
    stage_timings: dict[str, int] = Field(default_factory=dict)
    duplicate: bool = False
    parsing_engine: str | None = None
    engine_fallback_reason: str | None = None
    primary_domain_id: str | None = None
    primary_subdomain_id: str | None = None
    jurisdiction: str | None = None
    authority_tier: int = 3
    effective_year: int | None = None
    is_current: bool = True
    classification_result: str | None = None
    created_at: datetime
    updated_at: datetime
    documents: list[DocumentResponse] | None = None


class DocumentStatusResponse(DocumentResponse):
    """Detailed status polling payload for `/api/v1/documents/{id}/status`."""


class DocumentListResponse(BaseModel):
    """Paginated document list response supporting both `items` and `documents` keys."""

    items: list[DocumentResponse] = Field(default_factory=list)
    documents: list[DocumentResponse] = Field(default_factory=list)
    total: int = 0
    limit: int = 100
    offset: int = 0


class HeadingOutlineItem(BaseModel):
    """Single Markdown heading entry for the Document Inspector outline."""

    level: int = Field(..., ge=1, le=6)
    text: str
    heading_path: str
    line: int = Field(..., ge=1)


class DocumentMarkdownResponse(BaseModel):
    """Converted Markdown content and extracted heading outline for a document."""

    id: str
    doc_id: str
    filename: str
    file_type: str
    page_count: int | None = None
    markdown: str
    headings: list[HeadingOutlineItem] = Field(default_factory=list)


class DocumentDeleteResponse(BaseModel):
    """Atomic 3-tier document deletion response."""

    id: str
    deleted: bool = True
    status: str = "deleted"


class ApprovedClassificationRequest(BaseModel):
    """User-approved classification parameters and optional overrides."""

    domain: str | None = None
    subdomain: str | None = None
    authority_tier: int | None = None
    jurisdiction: str | None = None
    effective_year: int | None = None
    is_current: bool | None = None
    section_overrides: list[dict[str, Any]] | None = None


class ApproveBatchRequest(BaseModel):
    """Batch approval request payload."""

    doc_ids: list[str] = Field(default_factory=list)


class ApproveBatchResponse(BaseModel):
    """Batch approval summary response."""

    approved_count: int = 0
    documents: list[DocumentResponse] = Field(default_factory=list)


class SectionOverrideSchema(BaseModel):
    """Section-level domain override schema."""

    heading_path: str
    domain: str
    subdomain: str
    epistemic_modality: str = "general"
    confidence: float = 0.0
    authority_tier: int = 3
    jurisdiction: str | None = None


class UpdateDocumentMetadataRequest(BaseModel):
    """Request payload for updating document metadata."""

    primary_domain_id: str | None = None
    primary_subdomain_id: str | None = None
    domain: str | None = None
    subdomain: str | None = None
    authority_tier: int | None = None
    jurisdiction: str | None = None
    effective_year: int | None = None
    is_current: bool | None = None
    section_overrides: list[SectionOverrideSchema] | None = None
