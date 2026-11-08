"""Pydantic v2 request and response schemas for the FastAPI gateway (snake_case JSON)."""
from __future__ import annotations

import ipaddress
import re
import urllib.parse
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.rag.llm import is_valid_api_key


def _sanitize_string(v: str) -> str:
    cleaned = re.sub(r"[\r\n\x00-\x1f\x7f-\x9f]", " ", v)
    return " ".join(cleaned.split())


ALLOWED_COLOR_TOKENS: set[str] = {
    "cyan",
    "emerald",
    "amber",
    "rose",
    "violet",
    "blue",
    "teal",
    "orange",
    "fuchsia",
    "slate",
    "indigo",
}

ALLOWED_CHUNKING_STRATEGIES: set[str] = {
    "standard_markdown",
    "academic_equations",
    "statutory_clauses",
    "financial_tables",
}

ALLOWED_EPISTEMIC_TYPES: set[str] = {
    "empirical",
    "evaluative",
    "normative",
    "descriptive",
    "general",
}

ALLOWED_LLM_PROVIDERS: set[str] = {
    "openai",
    "anthropic",
    "deepseek",
    "openai_compatible",
    "",
}


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


class SystemHealthResponse(BaseModel):
    """Health check status for SQLite and Qdrant."""

    status: str = "ok"
    database: str = "ok"
    qdrant: str = "ok"
    collection: str


class ProviderOption(BaseModel):
    """Read-only description of an LLM provider option for the UI selector (never exposes raw keys)."""

    provider: str
    label: str
    configured: bool
    default_model: str = ""
    key_source: Literal["ui", "env", "none"] = "none"
    masked_key: str | None = None
    validation_status: Literal["valid", "invalid", "untested", "missing"] = "missing"
    validation_error: str | None = None
    base_url: str | None = None


class MemoryStatsResponse(BaseModel):
    """Host process RSS, PyTorch CUDA memory telemetry, and orchestrator worker pool metrics."""

    process_rss_mb: float = 0.0
    cuda_available: bool = False
    cuda_allocated_mb: float = 0.0
    cuda_reserved_mb: float = 0.0
    cuda_driver_free_mb: float | None = None
    cuda_driver_total_mb: float | None = None
    worker_pool_active: bool = False
    active_worker_jobs: int = 0
    active_indexing_jobs: int = 0
    worker_reclaim_count: int = 0
    reclaim_deferred: bool = False
    worker_idle_timeout_sec: float = 60.0
    worker_max_tasks: int = 0
    classifier_resident: bool = False
    classifier_model_name: str | None = None


class MemoryReclaimResponse(BaseModel):
    """Outcome of an on-demand memory / VRAM reclamation request."""

    status: str = "ok"
    worker_pool_reclaimed: bool = False
    deferred: bool = False
    cuda_cache_cleared: bool = False
    active_worker_jobs: int = 0
    memory: MemoryStatsResponse


class IngestionPauseResumeResponse(BaseModel):
    """Payload returned by ingestion pause/resume control endpoints."""

    status: str = "ok"
    ingestion_paused: bool


class SystemStatsResponse(BaseModel):
    """System-wide counts, Qdrant vector statistics, and active read-only model config."""

    total_documents: int = 0
    ready_documents: int = 0
    total_chunks: int = 0
    total_threads: int = 0
    qdrant_vectors_count: int = 0
    vector_count: int = 0
    ingestion_paused: bool = False
    documents_by_status: dict[str, int] = Field(default_factory=dict)
    llm: dict[str, Any] = Field(default_factory=dict)
    embedding: dict[str, Any] = Field(default_factory=dict)
    reranker: dict[str, Any] = Field(default_factory=dict)
    ingestion: dict[str, Any] = Field(default_factory=dict)
    classifier: dict[str, Any] = Field(default_factory=dict)
    memory: MemoryStatsResponse | None = None
    configured_providers: list[ProviderOption] = Field(default_factory=list)
    available_providers: list[ProviderOption] = Field(default_factory=list)



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


# ============================================================================
# Multi-Domain Taxonomy, Lexicon & Workspace Settings Schemas
# ============================================================================


class SubdomainItemResponse(BaseModel):
    """Serialized representation of a SubdomainRegistry row."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    domain_id: str
    slug: str
    name: str
    standard_bodies: str | None = None


class SubdomainCreateRequest(BaseModel):
    """Payload to create a new subdomain under a domain."""

    slug: str = Field(..., min_length=2, max_length=64)
    name: str = Field(..., min_length=2, max_length=100)
    standard_bodies: str | None = Field(default=None, max_length=200)

    @field_validator("name", "standard_bodies", mode="before")
    @classmethod
    def sanitize_text(cls, v: Any) -> Any:
        if isinstance(v, str):
            return _sanitize_string(v)
        return v

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, v: str) -> str:
        s = _sanitize_string(v).lower()
        if not re.match(r"^[a-z0-9_.-]+$", s):
            raise ValueError("Subdomain slug can only contain lowercase alphanumeric, underscores, hyphens, and periods")
        return s


class SubdomainUpdateRequest(BaseModel):
    """Payload to update an existing subdomain."""

    name: str | None = Field(default=None, min_length=2, max_length=100)
    standard_bodies: str | None = Field(default=None, max_length=200)

    @field_validator("name", "standard_bodies", mode="before")
    @classmethod
    def sanitize_text(cls, v: Any) -> Any:
        if isinstance(v, str):
            return _sanitize_string(v)
        return v


class DomainItemResponse(BaseModel):
    """Serialized representation of a DomainRegistry row including nested subdomains."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    slug: str
    name: str
    epistemic_type: str
    authority_priority: int = 3
    description: str = ""
    icon: str = "📁"
    color_token: str = "indigo"
    chunking_strategy: str = "standard_markdown"
    classifier_keywords: str | None = None
    is_system: bool = False
    is_active: bool = True
    document_count: int = 0
    subdomains: list[SubdomainItemResponse] = Field(default_factory=list)


class DomainCreateRequest(BaseModel):
    """Payload to create a new domain."""

    slug: str = Field(..., min_length=2, max_length=32, pattern=r"^[a-z][a-z0-9_-]{1,31}$")
    name: str = Field(..., min_length=2, max_length=80)
    description: str = Field(..., min_length=10, max_length=500)
    classifier_keywords: str | None = Field(default=None, max_length=300)
    color_token: str = Field(default="indigo")
    chunking_strategy: str = Field(default="standard_markdown")
    epistemic_type: str = Field(default="general")
    authority_priority: int = Field(default=3, ge=1, le=5)
    icon: str = Field(default="📁", max_length=10)
    is_active: bool = Field(default=True)

    @field_validator("name", "description", "icon", mode="before")
    @classmethod
    def sanitize_text_fields(cls, v: Any) -> Any:
        if isinstance(v, str):
            return _sanitize_string(v)
        return v

    @field_validator("color_token")
    @classmethod
    def validate_color_token(cls, v: str) -> str:
        token = v.strip().lower()
        if token not in ALLOWED_COLOR_TOKENS:
            raise ValueError(f"color_token must be one of {sorted(ALLOWED_COLOR_TOKENS)}")
        return token

    @field_validator("chunking_strategy")
    @classmethod
    def validate_chunking_strategy(cls, v: str) -> str:
        strategy = v.strip()
        if strategy not in ALLOWED_CHUNKING_STRATEGIES:
            raise ValueError(f"chunking_strategy must be one of {sorted(ALLOWED_CHUNKING_STRATEGIES)}")
        return strategy

    @field_validator("epistemic_type")
    @classmethod
    def validate_epistemic_type(cls, v: str) -> str:
        epistemic = v.strip().lower()
        if epistemic not in ALLOWED_EPISTEMIC_TYPES:
            raise ValueError(f"epistemic_type must be one of {sorted(ALLOWED_EPISTEMIC_TYPES)}")
        return epistemic

    @field_validator("classifier_keywords", mode="before")
    @classmethod
    def validate_classifier_keywords(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            v = _sanitize_string(v)
            if not v:
                return None
            tokens = [t.strip() for t in v.split(",") if t.strip()]
            if len(tokens) > 25:
                raise ValueError("classifier_keywords cannot exceed 25 comma-separated tokens")
            for t in tokens:
                if len(t) > 40:
                    raise ValueError(f"Classifier keyword token '{t}' cannot exceed 40 characters")
            return ", ".join(tokens)
        return v


class DomainUpdateRequest(BaseModel):
    """Payload to update an existing domain."""

    name: str | None = Field(default=None, min_length=2, max_length=80)
    epistemic_type: str | None = None
    authority_priority: int | None = Field(default=None, ge=1, le=5)
    description: str | None = Field(default=None, min_length=10, max_length=500)
    icon: str | None = Field(default=None, max_length=10)
    color_token: str | None = None
    chunking_strategy: str | None = None
    classifier_keywords: str | None = Field(default=None, max_length=300)
    is_active: bool | None = None
    re_index: bool = False

    @field_validator("name", "description", "icon", mode="before")
    @classmethod
    def sanitize_text_fields(cls, v: Any) -> Any:
        if isinstance(v, str):
            return _sanitize_string(v)
        return v

    @field_validator("color_token")
    @classmethod
    def validate_color_token(cls, v: str | None) -> str | None:
        if v is None:
            return None
        token = v.strip().lower()
        if token not in ALLOWED_COLOR_TOKENS:
            raise ValueError(f"color_token must be one of {sorted(ALLOWED_COLOR_TOKENS)}")
        return token

    @field_validator("chunking_strategy")
    @classmethod
    def validate_chunking_strategy(cls, v: str | None) -> str | None:
        if v is None:
            return None
        strategy = v.strip()
        if strategy not in ALLOWED_CHUNKING_STRATEGIES:
            raise ValueError(f"chunking_strategy must be one of {sorted(ALLOWED_CHUNKING_STRATEGIES)}")
        return strategy

    @field_validator("epistemic_type")
    @classmethod
    def validate_epistemic_type(cls, v: str | None) -> str | None:
        if v is None:
            return None
        epistemic = v.strip().lower()
        if epistemic not in ALLOWED_EPISTEMIC_TYPES:
            raise ValueError(f"epistemic_type must be one of {sorted(ALLOWED_EPISTEMIC_TYPES)}")
        return epistemic

    @field_validator("classifier_keywords", mode="before")
    @classmethod
    def validate_classifier_keywords(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            v = _sanitize_string(v)
            if not v:
                return None
            tokens = [t.strip() for t in v.split(",") if t.strip()]
            if len(tokens) > 25:
                raise ValueError("classifier_keywords cannot exceed 25 comma-separated tokens")
            for t in tokens:
                if len(t) > 40:
                    raise ValueError(f"Classifier keyword token '{t}' cannot exceed 40 characters")
            return ", ".join(tokens)
        return v


class DomainDeleteResponse(BaseModel):
    """Response payload for domain deletion and target reassignment."""

    deleted_slug: str
    target_domain: str
    target_subdomain: str
    reassigned_documents: int
    reassigned_chunks: int
    re_index_queued: int = 0


class LexiconItemResponse(BaseModel):
    """Serialized representation of a DomainLexicon row."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    term: str
    domain_id: str
    specific_sense: str
    contrastive_warning: str
    created_at: datetime | None = None


class LexiconCreateRequest(BaseModel):
    """Payload to add a polysemic term definition to the domain lexicon."""

    term: str = Field(..., min_length=2, max_length=80)
    domain_id: str
    specific_sense: str = Field(..., min_length=5, max_length=300)
    contrastive_warning: str = Field(..., min_length=5, max_length=300)

    @field_validator("term", "specific_sense", "contrastive_warning", mode="before")
    @classmethod
    def sanitize_text(cls, v: Any) -> Any:
        if isinstance(v, str):
            return _sanitize_string(v)
        return v


class LexiconUpdateRequest(BaseModel):
    """Payload to update an existing domain lexicon entry."""

    specific_sense: str | None = Field(default=None, min_length=5, max_length=300)
    contrastive_warning: str | None = Field(default=None, min_length=5, max_length=300)

    @field_validator("specific_sense", "contrastive_warning", mode="before")
    @classmethod
    def sanitize_text(cls, v: Any) -> Any:
        if isinstance(v, str):
            return _sanitize_string(v)
        return v


def _validate_safe_base_url(url: str | None) -> str | None:
    """Validate candidate Base URL against SSRF, internal services, and self-loopback."""
    if not url:
        return None
    cleaned = url.strip()
    if not cleaned:
        return None
    parsed = urllib.parse.urlparse(cleaned)
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("Base URL must start with http:// or https://")
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise ValueError("Base URL must include a valid hostname")

    # Cloud metadata domains and internal container services
    blocked_hosts = {"metadata.google.internal", "qdrant"}
    if hostname in blocked_hosts:
        raise ValueError(f"Base URL host '{hostname}' is not permitted")

    # Check port for self-loopback
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)

    # Check IP addresses
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_link_local or ip.is_unspecified or ip.is_multicast:
            raise ValueError("Base URL cannot target link-local, unspecified, or multicast IP addresses")
        if str(ip) in ("169.254.169.254", "fd00:ec2::254"):
            raise ValueError("Base URL cannot target cloud metadata services")
        if ip.is_loopback and port == 8000:
            raise ValueError("Base URL cannot target the internal server port (8000)")
    except ValueError as ve:
        if "Base URL" in str(ve):
            raise
        if hostname == "localhost" and port == 8000:
            raise ValueError("Base URL cannot target the internal server port (8000)")
        if hostname.startswith("169.254."):
            raise ValueError("Base URL cannot target link-local address range")

    return cleaned


class WorkspaceSettingsResponse(BaseModel):
    """Workspace-wide upload, approval, and AI defaults."""

    model_config = ConfigDict(from_attributes=True)

    id: str = "default"
    default_upload_mode: str = "fast"
    default_approval_mode: str = "review_all"
    high_confidence_threshold: float = 0.85
    default_llm_provider: str | None = None
    default_llm_model: str | None = None
    llm_compat_base_url: str | None = None
    available_providers: list[ProviderOption] = Field(default_factory=list)
    updated_at: datetime | None = None


class WorkspaceSettingsUpdateRequest(BaseModel):
    """Payload to update workspace-wide defaults."""

    default_upload_mode: Literal["auto", "gpu", "fast"] | None = None
    default_approval_mode: Literal["review_all", "auto_high_confidence", "auto_all"] | None = None
    high_confidence_threshold: float | None = Field(default=None, ge=0.50, le=0.99)
    default_llm_provider: str | None = None
    default_llm_model: str | None = None
    api_keys: dict[str, str | None] | None = None
    llm_compat_base_url: str | None = None

    @field_validator("default_llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str | None) -> str | None:
        if v is None:
            return None
        prov = v.strip().lower()
        if prov not in ALLOWED_LLM_PROVIDERS:
            raise ValueError(f"default_llm_provider must be one of {sorted(ALLOWED_LLM_PROVIDERS)}")
        return prov

    @field_validator("api_keys")
    @classmethod
    def validate_api_keys(cls, v: dict[str, str | None] | None) -> dict[str, str | None] | None:
        if v is None:
            return None
        allowed_providers = {"openai", "anthropic", "deepseek", "openai_compatible"}
        cleaned: dict[str, str | None] = {}
        for prov, key in v.items():
            norm_prov = prov.strip().lower()
            if norm_prov not in allowed_providers:
                raise ValueError(f"Unknown API key provider '{prov}'. Must be one of {sorted(allowed_providers)}")
            if key is not None and key.strip() != "":
                if not is_valid_api_key(key):
                    raise ValueError(
                        f"Invalid API key format for '{prov}' (placeholders and whitespace/control characters are not allowed)"
                    )
                cleaned[norm_prov] = key.strip()
            else:
                cleaned[norm_prov] = "" if key is not None else None
        return cleaned

    @field_validator("llm_compat_base_url")
    @classmethod
    def validate_llm_compat_base_url(cls, v: str | None) -> str | None:
        return _validate_safe_base_url(v)


class VerifyKeyRequest(BaseModel):
    provider: Literal["openai", "anthropic", "deepseek", "openai_compatible"]
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    save_if_valid: bool = False


class VerifyKeyResponse(BaseModel):
    provider: str
    valid: bool
    validation_status: Literal["valid", "invalid", "untested", "missing"]
    validation_error: str | None = None
    key_source: Literal["ui", "env", "none"] = "none"
    masked_key: str | None = None
    tested_model: str = ""
    provider_option: ProviderOption


