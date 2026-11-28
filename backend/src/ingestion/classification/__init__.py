"""Document domain classification subpackage."""
from src.ingestion.classification.heuristics import (
    StubLayaClassifier,
    _extract_authority_and_jurisdiction,
    _score_domain_and_subdomain,
)
from src.ingestion.classification.laya import (
    LayaDomainClassifier,
    _extract_laya_answer,
    get_classifier,
    get_classifier_model_name,
    is_classifier_resident,
    set_classifier_force_stub,
    unload_classifier,
)
from src.ingestion.classification.llm_fallback import classify_document_with_llm
from src.ingestion.classification.models import (
    BaseDomainClassifier,
    DomainClassificationResult,
    DomainSpec,
    RecommendedDomainProfile,
    SectionOverride,
    SubdomainSpec,
)
from src.ingestion.classification.taxonomy import (
    DEFAULT_SUBDOMAIN_FOR_DOMAIN,
    DOMAIN_EPISTEMIC_MAP,
    SUBDOMAIN_DOMAIN_MAP,
    TaxonomyCatalog,
    _make_keyword_pattern,
    build_dynamic_taxonomy_questions,
    get_taxonomy_catalog,
    invalidate_taxonomy_catalog,
    refresh_taxonomy_catalog,
)

__all__ = [
    "BaseDomainClassifier",
    "DEFAULT_SUBDOMAIN_FOR_DOMAIN",
    "DOMAIN_EPISTEMIC_MAP",
    "DomainClassificationResult",
    "DomainSpec",
    "LayaDomainClassifier",
    "RecommendedDomainProfile",
    "SUBDOMAIN_DOMAIN_MAP",
    "SectionOverride",
    "StubLayaClassifier",
    "SubdomainSpec",
    "TaxonomyCatalog",
    "_extract_authority_and_jurisdiction",
    "_extract_laya_answer",
    "_make_keyword_pattern",
    "_score_domain_and_subdomain",
    "build_dynamic_taxonomy_questions",
    "classify_document_with_llm",
    "get_classifier",
    "get_classifier_model_name",
    "get_taxonomy_catalog",
    "invalidate_taxonomy_catalog",
    "is_classifier_resident",
    "refresh_taxonomy_catalog",
    "set_classifier_force_stub",
    "unload_classifier",
]
