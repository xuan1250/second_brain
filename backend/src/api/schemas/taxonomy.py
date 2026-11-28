"""Multi-domain taxonomy, subdomain, and polysemy lexicon request/response schemas."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.api.schemas.common import _DomainValidationMixin, _sanitize_string


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


class DomainCreateRequest(_DomainValidationMixin, BaseModel):
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


class DomainUpdateRequest(_DomainValidationMixin, BaseModel):
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
