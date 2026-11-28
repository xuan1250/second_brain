"""Conversational CRAG request, citation, domain filter, polysemy, and thread schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CitationPayload(BaseModel):
    """Structured citation metadata linking an inline `[N]` marker to a source chunk."""

    citation_index: int = Field(..., ge=1)
    doc_id: str
    chunk_id: str
    source_filename: str
    page_numbers: list[int] = Field(default_factory=list)
    heading_path: str = ""
    excerpt: str = ""
    score: float = 0.0
    domain: str | None = None
    subdomain: str | None = None
    authority_tier: int | None = None
    epistemic_type: str | None = None
    jurisdiction: str | None = None
    is_current: bool = True


class DomainFilterSpec(BaseModel):
    """Specification for domain-targeted retrieval and namespace filtering."""

    mode: str = "auto"
    domains: list[str] = Field(default_factory=list)
    subdomains: list[str] = Field(default_factory=list)
    max_authority_tier: int | None = None
    only_current: bool = True
    include_general_fallback: bool = True


class PolysemyCollision(BaseModel):
    """Detected cross-domain polysemic or homonym term collision."""

    term: str
    domains: list[str] = Field(default_factory=list)
    senses: dict[str, str] = Field(default_factory=dict)
    contrastive_warning: str = ""


class ChatRequest(BaseModel):
    """Request body for `POST /api/v1/chat/completions`."""

    message: str = ""
    query: str | None = None
    thread_id: str | None = None
    filter_doc_ids: list[str] | None = None
    domain_filter: DomainFilterSpec | None = None
    provider: str | None = None
    model: str | None = None

    @model_validator(mode="after")
    def _normalize_message(self) -> ChatRequest:
        if not self.message and self.query is not None:
            self.message = self.query
        return self


class ChatMessagePayload(BaseModel):
    """Serialized conversation turn returned by thread history endpoints."""

    role: str
    content: str
    citations: list[CitationPayload] = Field(default_factory=list)
    domain_filter: DomainFilterSpec | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)


class ThreadSummary(BaseModel):
    """Summary metadata for a saved conversation thread in `chat_threads`."""

    model_config = ConfigDict(from_attributes=True)

    thread_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ThreadListResponse(BaseModel):
    """List of saved conversation threads (newest first)."""

    threads: list[ThreadSummary] = Field(default_factory=list)
    items: list[ThreadSummary] = Field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0


class ThreadDetailResponse(BaseModel):
    """Full conversation message and citation history for a single thread."""

    thread_id: str
    title: str = "New chat"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    messages: list[ChatMessagePayload] = Field(default_factory=list)
    citations: list[CitationPayload] = Field(default_factory=list)


class ThreadDeleteResponse(BaseModel):
    """Thread deletion confirmation payload."""

    thread_id: str
    deleted: bool = True
    status: str = "deleted"
