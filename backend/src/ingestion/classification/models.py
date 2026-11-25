"""Classification domain profile schemas, taxonomy specs, and abstract classifier interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.ingestion.constants import DEFAULT_DOMAIN, DEFAULT_EPISTEMIC, DEFAULT_SUBDOMAIN


class SectionOverride(BaseModel):
    """Domain categorization override for a specific H1/H2 section within a hybrid document."""

    heading_path: str
    domain: str
    subdomain: str
    epistemic_modality: str
    confidence: float = 0.0
    authority_tier: int = 3
    jurisdiction: Optional[str] = None
    chunking_strategy: str = "standard_markdown"


class RecommendedDomainProfile(BaseModel):
    """Structured domain categorization, authority tier, and section-level overrides."""

    primary_domain: str = DEFAULT_DOMAIN
    primary_subdomain: str = DEFAULT_SUBDOMAIN
    epistemic_modality: str = DEFAULT_EPISTEMIC
    confidence: float = 0.0
    confidence_subdomain: float = 0.0
    authority_tier: int = 3
    jurisdiction: Optional[str] = None
    effective_year: Optional[int] = None
    is_current: bool = True
    is_hybrid: bool = False
    section_overrides: list[SectionOverride] = Field(default_factory=list)
    reasoning: Optional[str] = None
    chunking_strategy: str = "standard_markdown"


class DomainClassificationResult(BaseModel):
    """Wrapper holding classification outcome, fallback diagnostics, and timings."""

    recommended: RecommendedDomainProfile
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    execution_ms: int = 0
    raw_answers: Optional[dict[str, Any]] = None


@dataclass
class SubdomainSpec:
    """Declarative metadata for a domain subdomain."""

    slug: str
    name: str
    domain_slug: str
    standard_bodies: Optional[str] = None
    description: Optional[str] = None


@dataclass
class DomainSpec:
    """Declarative metadata and keyword configuration for a knowledge domain."""

    slug: str
    name: str
    epistemic_type: str = DEFAULT_EPISTEMIC
    description: str = ""
    icon: str = "📁"
    color_token: str = "slate"
    chunking_strategy: str = "standard_markdown"
    classifier_keywords: Optional[str] = None
    authority_priority: int = 3
    is_system: bool = False
    is_active: bool = True
    subdomains: dict[str, SubdomainSpec] = field(default_factory=dict)


class BaseDomainClassifier(ABC):
    """Abstract interface for domain classifiers."""

    @abstractmethod
    async def classify_document(
        self,
        text: str,
        *,
        sections: Optional[list[dict[str, str]]] = None,
        filename: Optional[str] = None,
    ) -> DomainClassificationResult:
        """Classify document domain, subdomain, epistemic modality, authority, and section overrides."""
