"""Deterministic citation authority/jurisdiction heuristics and StubLayaClassifier."""
from __future__ import annotations

import re
import time
from typing import Callable, Optional

from src.ingestion.classification.models import (
    BaseDomainClassifier,
    DomainClassificationResult,
    RecommendedDomainProfile,
    SectionOverride,
)
from src.ingestion.classification.taxonomy import TaxonomyCatalog, get_taxonomy_catalog
from src.ingestion.constants import DEFAULT_DOMAIN, DEFAULT_SUBDOMAIN, normalize_chunking_strategy
from src.ingestion.tag_patterns import (
    ASME_PREFIX_RE,
    COLON_YEAR_RE,
    FORM_10K_RE,
    FORM_10K_YEAR_RE,
    IEC_PREFIX_RE,
    IEEE_PREFIX_RE,
    IEEE_YEAR_RE,
    IFRS_IAS_PREFIX_RE,
    QCVN_PREFIX_RE,
    TCVN_PREFIX_RE,
    US_CFR_PREFIX_RE,
    US_CODE_PREFIX_RE,
    US_GAAP_ASC_PREFIX_RE,
    VAS_PREFIX_RE,
    VN_CIRCULAR_CITATION_RE,
    VN_CIRCULAR_YEAR_RE,
    VN_DECREE_CITATION_RE,
    VN_DECREE_YEAR_RE,
    VN_LAW_CITATION_RE,
    VN_LAW_YEAR_RE,
    YEAR_4DIGIT_RE,
)


def _extract_vn_law_year(sample: str) -> Optional[int]:
    m = VN_LAW_YEAR_RE.search(sample)
    if not m:
        return None
    yr_short = int(m.group(1))
    return 2000 + yr_short if yr_short < 50 else 1900 + yr_short


def _extract_group1_int(pattern: re.Pattern[str], sample: str) -> Optional[int]:
    m = pattern.search(sample)
    return int(m.group(1)) if (m and m.group(1)) else None


_AUTHORITY_CITATION_RULES: tuple[
    tuple[re.Pattern[str], int, str, Optional[Callable[[str], Optional[int]]]], ...
] = (
    (VN_LAW_CITATION_RE, 1, "VN-STATUTE", _extract_vn_law_year),
    (VN_DECREE_CITATION_RE, 2, "VN-GOV", lambda s: _extract_group1_int(VN_DECREE_YEAR_RE, s)),
    (VN_CIRCULAR_CITATION_RE, 2, "VN-MINISTRY", lambda s: _extract_group1_int(VN_CIRCULAR_YEAR_RE, s)),
    (TCVN_PREFIX_RE, 1, "VN-STANDARD", lambda s: _extract_group1_int(COLON_YEAR_RE, s)),
    (QCVN_PREFIX_RE, 1, "VN-STANDARD", lambda s: _extract_group1_int(COLON_YEAR_RE, s)),
    (VAS_PREFIX_RE, 1, "VN-VAS", None),
    (US_CODE_PREFIX_RE, 1, "US-FED", None),
    (US_CFR_PREFIX_RE, 2, "US-FED", None),
    (IEEE_PREFIX_RE, 1, "IEEE", lambda s: _extract_group1_int(IEEE_YEAR_RE, s)),
    (IEC_PREFIX_RE, 1, "IEC", None),
    (ASME_PREFIX_RE, 1, "ASME", None),
    (IFRS_IAS_PREFIX_RE, 1, "IFRS", None),
    (US_GAAP_ASC_PREFIX_RE, 1, "US-GAAP", None),
    (FORM_10K_RE, 2, "SEC", lambda s: _extract_group1_int(FORM_10K_YEAR_RE, s)),
)


def _extract_authority_and_jurisdiction(text: str) -> tuple[int, Optional[str], Optional[int]]:
    """Heuristically extract authority tier, jurisdiction, and effective year from standard/statute citations."""
    sample = text[:3000]
    for match_re, tier, jurisdiction, year_fn in _AUTHORITY_CITATION_RULES:
        if match_re.search(sample):
            return tier, jurisdiction, year_fn(sample) if year_fn else None

    return 3, None, _extract_group1_int(YEAR_4DIGIT_RE, sample)


_STARTER_SUBDOMAIN_CUES: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "engineering": (
        ("eng.electrical", ("transformer", "dielectric", "voltage", "kv", "ieee", "máy biến áp", "điện")),
        ("eng.control", ("scada", "pid", "plc", "feedback", "control", "điều khiển")),
        ("eng.mechanical", ("stress", "thermal", "fatigue", "heat", "cơ học", "nhiệt")),
    ),
    "economics": (
        ("econ.accounting", ("balance sheet", "depreciation", "kế toán", "khấu hao", "ifrs", "gaap", "vas")),
        ("econ.taxation", ("tax", "thuế", "irc")),
        ("econ.investment", ("dcf", "npv", "irr", "portfolio", "đầu tư")),
    ),
    "law": (
        ("law.regulation", ("cfr", "regulation", "nghị định", "quy chuẩn", "directive", "agency")),
        ("law.contract", ("contract", "agreement", "covenant", "warranty", "hợp đồng")),
        ("law.case_law", ("case", "court", "precedent", "appellate", "tòa án", "bản án")),
    ),
}

_STARTER_FALLBACK_SUBDOMAINS: dict[str, str] = {
    "engineering": "eng.technology",
    "economics": "econ.finance",
    "law": "law.statute",
}


def _score_domain_and_subdomain(
    text: str,
    catalog: TaxonomyCatalog,
) -> tuple[dict[str, float], dict[str, str]]:
    """Score all non-general domains and select subdomains using catalog keywords and cached regexes."""
    text_lower = text.lower()
    domain_scores: dict[str, float] = {}
    best_subdomains: dict[str, str] = {}

    for slug, d in catalog.domains.items():
        if slug == DEFAULT_DOMAIN or not d.is_active:
            continue

        pattern = catalog.get_domain_keyword_pattern(slug)
        score = 0.0
        if pattern is not None:
            matches = pattern.findall(text_lower)
            weight = 1.0 if d.is_system else 3.0
            score = float(len(matches)) * weight
        domain_scores[slug] = score

        assigned_sub = catalog.get_default_subdomain(slug)
        if slug in _STARTER_SUBDOMAIN_CUES:
            matched_cue = False
            for sub_slug, cues in _STARTER_SUBDOMAIN_CUES[slug]:
                if any(k in text_lower for k in cues):
                    assigned_sub = sub_slug
                    matched_cue = True
                    break
            if not matched_cue:
                fallback_sub = _STARTER_FALLBACK_SUBDOMAINS.get(slug)
                if fallback_sub and fallback_sub in d.subdomains:
                    assigned_sub = fallback_sub
        else:
            best_sub_hits = -1
            best_sub = assigned_sub
            for s_slug in d.subdomains:
                s_pattern = catalog.get_subdomain_keyword_pattern(slug, s_slug)
                if s_pattern is not None:
                    s_hits = len(s_pattern.findall(text_lower))
                    if s_hits > best_sub_hits:
                        best_sub_hits = s_hits
                        best_sub = s_slug
            if best_sub_hits > 0:
                assigned_sub = best_sub

        best_subdomains[slug] = assigned_sub

    return domain_scores, best_subdomains


class StubLayaClassifier(BaseDomainClassifier):
    """Deterministic, zero-network domain classifier for testing and offline environments."""

    def __init__(
        self,
        default_domain: Optional[str] = None,
        default_confidence: float = 0.92,
        inject_overrides: Optional[list[SectionOverride]] = None,
    ) -> None:
        self.default_domain = default_domain
        self.default_confidence = default_confidence
        self.inject_overrides = inject_overrides or []

    async def classify_document(
        self,
        text: str,
        *,
        sections: Optional[list[dict[str, str]]] = None,
        filename: Optional[str] = None,
    ) -> DomainClassificationResult:
        t0 = time.perf_counter()
        sample = f"{filename or ''}\n{text[:4000]}".lower()
        catalog = get_taxonomy_catalog()

        primary_domain = self.default_domain
        subdomain = None
        confidence = self.default_confidence

        scores, best_subs = _score_domain_and_subdomain(sample, catalog)

        if primary_domain is None:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if sorted_scores and sorted_scores[0][1] > 0:
                primary_domain = sorted_scores[0][0]
                total = max(1.0, sum(scores.values()))
                confidence = round(sorted_scores[0][1] / total, 2)
                if confidence > 0.5:
                    confidence = min(0.98, round(confidence + 0.3, 2))
                subdomain = best_subs.get(primary_domain, catalog.get_default_subdomain(primary_domain))
            else:
                primary_domain = DEFAULT_DOMAIN
                subdomain = DEFAULT_SUBDOMAIN
                confidence = 0.50
        else:
            subdomain = best_subs.get(primary_domain, catalog.get_default_subdomain(primary_domain))

        epistemic = catalog.get_epistemic_modality(primary_domain)
        chunking_strategy = catalog.get_chunking_strategy(primary_domain)
        tier, jurisdiction, eff_year = _extract_authority_and_jurisdiction(text)

        overrides: list[SectionOverride] = list(self.inject_overrides)
        if sections and not overrides:
            for sec in sections:
                h_path = sec.get("heading_path", "")
                h_text = sec.get("text", "")
                combined = f"{h_path}\n{h_text[:500]}"

                sec_scores, sec_subs = _score_domain_and_subdomain(combined, catalog)
                sec_sorted = sorted(sec_scores.items(), key=lambda x: x[1], reverse=True)
                sec_domain = None
                if sec_sorted and sec_sorted[0][1] > 0:
                    sec_domain = sec_sorted[0][0]
                else:
                    combined_lower = combined.lower()
                    if any(w in combined_lower for w in ("legal", "statute", "compliance", "law", "pháp luật", "quy định", "hợp đồng")):
                        sec_domain = "law"
                    elif any(w in combined_lower for w in ("tax", "accounting", "financial", "economics", "kế toán", "thuế", "tài chính")):
                        sec_domain = "economics"
                    elif any(w in combined_lower for w in ("technical", "engineering", "specifications", "kỹ thuật")):
                        sec_domain = "engineering"

                if sec_domain and sec_domain != primary_domain:
                    sec_sub = sec_subs.get(sec_domain, catalog.get_default_subdomain(sec_domain))
                    sec_strategy = catalog.get_chunking_strategy(sec_domain)
                    overrides.append(
                        SectionOverride(
                            heading_path=h_path or "Section",
                            domain=sec_domain,
                            subdomain=sec_sub,
                            epistemic_modality=catalog.get_epistemic_modality(sec_domain),
                            confidence=0.85,
                            authority_tier=tier,
                            jurisdiction=jurisdiction,
                            chunking_strategy=sec_strategy,
                        )
                    )

        for ov in overrides:
            ov.chunking_strategy = normalize_chunking_strategy(
                getattr(ov, "chunking_strategy", None),
                catalog.get_chunking_strategy(ov.domain),
            )

        profile = RecommendedDomainProfile(
            primary_domain=primary_domain,
            primary_subdomain=subdomain,
            epistemic_modality=epistemic,
            confidence=confidence,
            confidence_subdomain=confidence,
            authority_tier=tier,
            jurisdiction=jurisdiction,
            effective_year=eff_year,
            is_current=True,
            is_hybrid=len(overrides) > 0,
            section_overrides=overrides,
            reasoning=f"Stub classification based on keyword matching (domain={primary_domain})",
            chunking_strategy=chunking_strategy,
        )

        elapsed_ms = max(1, int((time.perf_counter() - t0) * 1000))
        return DomainClassificationResult(
            recommended=profile,
            fallback_used=False,
            execution_ms=elapsed_ms,
        )
