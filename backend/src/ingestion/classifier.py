"""Laya multilingual domain decision classifier with H1/H2 section overrides and opt-in LLM fallback."""
from __future__ import annotations

import gc
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings

logger = logging.getLogger(__name__)

# Primary domain epistemic mapping (canonical fallbacks)
DOMAIN_EPISTEMIC_MAP: dict[str, str] = {
    "engineering": "empirical",
    "economics": "evaluative",
    "law": "normative",
    "general": "general",
}

# Subdomain taxonomy mappings (canonical fallbacks)
SUBDOMAIN_DOMAIN_MAP: dict[str, str] = {
    "eng.electrical": "engineering",
    "eng.control": "engineering",
    "eng.mechanical": "engineering",
    "eng.technology": "engineering",
    "econ.finance": "economics",
    "econ.accounting": "economics",
    "econ.investment": "economics",
    "econ.taxation": "economics",
    "law.statute": "law",
    "law.regulation": "law",
    "law.case_law": "law",
    "law.contract": "law",
}

DEFAULT_SUBDOMAIN_FOR_DOMAIN: dict[str, str] = {
    "engineering": "eng.electrical",
    "economics": "econ.finance",
    "law": "law.statute",
    "general": "general.general",
}


class SectionOverride(BaseModel):
    """Domain categorization override for a specific H1/H2 section within a hybrid document."""

    heading_path: str
    domain: str  # engineering | economics | law | general | or any custom domain slug
    subdomain: str
    epistemic_modality: str  # empirical | evaluative | normative | general
    confidence: float = 0.0
    authority_tier: int = 3
    jurisdiction: Optional[str] = None
    chunking_strategy: str = "standard_markdown"


class RecommendedDomainProfile(BaseModel):
    """Structured domain categorization, authority tier, and section-level overrides."""

    primary_domain: str = "general"
    primary_subdomain: str = "general.general"
    epistemic_modality: str = "general"
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
    slug: str
    name: str
    domain_slug: str
    standard_bodies: Optional[str] = None
    description: Optional[str] = None


@dataclass
class DomainSpec:
    slug: str
    name: str
    epistemic_type: str = "general"
    description: str = ""
    icon: str = "📁"
    color_token: str = "slate"
    chunking_strategy: str = "standard_markdown"
    classifier_keywords: Optional[str] = None
    authority_priority: int = 3
    is_system: bool = False
    is_active: bool = True
    subdomains: dict[str, SubdomainSpec] = field(default_factory=dict)


@dataclass
class TaxonomyCatalog:
    domains: dict[str, DomainSpec] = field(default_factory=dict)

    def get_domain(self, slug: str) -> Optional[DomainSpec]:
        return self.domains.get(slug)

    def get_epistemic_modality(self, domain_slug: str) -> str:
        d = self.domains.get(domain_slug)
        if d:
            return d.epistemic_type
        return DOMAIN_EPISTEMIC_MAP.get(domain_slug, "general")

    def get_chunking_strategy(self, domain_slug: str) -> str:
        d = self.domains.get(domain_slug)
        if d:
            return d.chunking_strategy
        if domain_slug == "engineering":
            return "academic_equations"
        if domain_slug == "economics":
            return "financial_tables"
        if domain_slug == "law":
            return "statutory_clauses"
        return "standard_markdown"

    def get_default_subdomain(self, domain_slug: str) -> str:
        d = self.domains.get(domain_slug)
        if d and d.subdomains:
            if f"{domain_slug}.general" in d.subdomains:
                return f"{domain_slug}.general"
            return next(iter(d.subdomains.keys()))
        return DEFAULT_SUBDOMAIN_FOR_DOMAIN.get(
            domain_slug,
            f"{domain_slug}.general" if domain_slug != "general" else "general.general",
        )

    def domain_for_subdomain(self, sub_slug: str) -> Optional[str]:
        for d_slug, d in self.domains.items():
            if sub_slug in d.subdomains:
                return d_slug
        return SUBDOMAIN_DOMAIN_MAP.get(sub_slug)


def _build_starter_catalog() -> TaxonomyCatalog:
    cat = TaxonomyCatalog()
    cat.domains["engineering"] = DomainSpec(
        slug="engineering",
        name="Engineering & Technology",
        epistemic_type="empirical",
        description=(
            "Physical systems, electrical power, transformers, dielectric analysis, "
            "circuits, mechanics, control systems, software systems, datasheets, IEEE, IEC, TCVN."
        ),
        icon="⚙️",
        color_token="indigo",
        chunking_strategy="academic_equations",
        classifier_keywords=(
            "transformer,voltage,dielectric,kilovolt,kv,ieee,iec,tcvn,máy biến áp,"
            "điện áp,dòng điện,scada,pid,mechanical,thermal,fatigue"
        ),
        authority_priority=1,
        is_system=True,
        is_active=True,
        subdomains={
            "eng.electrical": SubdomainSpec(
                slug="eng.electrical",
                name="Electrical & Power Systems",
                domain_slug="engineering",
                standard_bodies="IEEE, IEC, NFPA",
            ),
            "eng.control": SubdomainSpec(
                slug="eng.control",
                name="Control Systems & Automation",
                domain_slug="engineering",
                standard_bodies="ISA, IEC 61131",
            ),
            "eng.mechanical": SubdomainSpec(
                slug="eng.mechanical",
                name="Mechanical & Thermal Systems",
                domain_slug="engineering",
                standard_bodies="ASME, ISO",
            ),
            "eng.technology": SubdomainSpec(
                slug="eng.technology",
                name="Software, Computing & IT",
                domain_slug="engineering",
                standard_bodies="ISO/IEC, RFC",
            ),
        },
    )
    cat.domains["economics"] = DomainSpec(
        slug="economics",
        name="Economics & Finance",
        epistemic_type="evaluative",
        description=(
            "Corporate finance, accounting standards, balance sheets, auditing, "
            "asset depreciation schedules, taxation, DCF valuation, IFRS, US GAAP, VAS, Thông tư tài chính."
        ),
        icon="📈",
        color_token="emerald",
        chunking_strategy="financial_tables",
        classifier_keywords=(
            "balance sheet,depreciation,amortization,ebitda,revenue,ifrs,gaap,vas,"
            "thông tư,kế toán,khấu hao,tài chính,chi phí,lợi nhuận,dcf,npv,10-k,wacc"
        ),
        authority_priority=1,
        is_system=True,
        is_active=True,
        subdomains={
            "econ.accounting": SubdomainSpec(
                slug="econ.accounting",
                name="Financial Accounting & Auditing",
                domain_slug="economics",
                standard_bodies="IFRS, US GAAP, VAS",
            ),
            "econ.finance": SubdomainSpec(
                slug="econ.finance",
                name="Corporate Finance & Capital",
                domain_slug="economics",
                standard_bodies="SEC, Basel III",
            ),
            "econ.investment": SubdomainSpec(
                slug="econ.investment",
                name="Asset Valuation & Investment",
                domain_slug="economics",
                standard_bodies="CFA, DCF",
            ),
            "econ.taxation": SubdomainSpec(
                slug="econ.taxation",
                name="Corporate Taxation & Compliance",
                domain_slug="economics",
                standard_bodies="OECD, IRC, BTC",
            ),
        },
    )
    cat.domains["law"] = DomainSpec(
        slug="law",
        name="Law & Regulation",
        epistemic_type="normative",
        description=(
            "Statutory codes, legislation, administrative regulations, statutory compliance, "
            "court decisions, commercial contracts, liabilities, penalties, 40 CFR, Luật, Nghị định."
        ),
        icon="⚖️",
        color_token="amber",
        chunking_strategy="statutory_clauses",
        classifier_keywords=(
            "statute,regulation,pursuant to,cfr,u.s.c.,indemnity,covenant,notwithstanding,"
            "luật,nghị định,điều khoản,khoản,điểm,hợp đồng,bồi thường,vi phạm"
        ),
        authority_priority=1,
        is_system=True,
        is_active=True,
        subdomains={
            "law.statute": SubdomainSpec(
                slug="law.statute",
                name="Statutory Legislation & Codes",
                domain_slug="law",
                standard_bodies="U.S.C., National Statutes",
            ),
            "law.regulation": SubdomainSpec(
                slug="law.regulation",
                name="Administrative Regulations & Rules",
                domain_slug="law",
                standard_bodies="CFR, Decrees",
            ),
            "law.case_law": SubdomainSpec(
                slug="law.case_law",
                name="Judicial Precedents & Rulings",
                domain_slug="law",
                standard_bodies="Appellate Courts",
            ),
            "law.contract": SubdomainSpec(
                slug="law.contract",
                name="Commercial Contracts & Agreements",
                domain_slug="law",
                standard_bodies="UCC, FIDIC",
            ),
        },
    )
    cat.domains["general"] = DomainSpec(
        slug="general",
        name="General & Miscellaneous",
        epistemic_type="general",
        description=(
            "Meeting minutes, company announcements, general prose, "
            "unclassified informational documents without specialized domain focus."
        ),
        icon="📁",
        color_token="slate",
        chunking_strategy="standard_markdown",
        classifier_keywords="",
        authority_priority=5,
        is_system=True,
        is_active=True,
        subdomains={
            "general.general": SubdomainSpec(
                slug="general.general",
                name="General Informational",
                domain_slug="general",
                standard_bodies=None,
            )
        },
    )
    return cat


_TAXONOMY_CATALOG: Optional[TaxonomyCatalog] = None


def get_taxonomy_catalog() -> TaxonomyCatalog:
    """Return the cached taxonomy catalog, initializing from starter defaults if not populated."""
    global _TAXONOMY_CATALOG
    if _TAXONOMY_CATALOG is None:
        _TAXONOMY_CATALOG = _build_starter_catalog()
    return _TAXONOMY_CATALOG


def invalidate_taxonomy_catalog() -> None:
    """Clear the cached taxonomy catalog so next access reloads from DB or defaults."""
    global _TAXONOMY_CATALOG
    _TAXONOMY_CATALOG = None


async def refresh_taxonomy_catalog(session: AsyncSession) -> TaxonomyCatalog:
    """Reload taxonomy catalog from SQLite DB (domains and subdomains) and update cache."""
    global _TAXONOMY_CATALOG
    try:
        from sqlalchemy.orm import selectinload
        from src.db.models import DomainRegistry

        stmt = (
            select(DomainRegistry)
            .where(DomainRegistry.is_active == True)  # noqa: E712
            .options(selectinload(DomainRegistry.subdomains))
            .order_by(DomainRegistry.authority_priority, DomainRegistry.slug)
        )
        res = await session.execute(stmt)
        domains_db = list(res.scalars().all())
        if not domains_db:
            _TAXONOMY_CATALOG = _build_starter_catalog()
            return _TAXONOMY_CATALOG

        starter_cat = _build_starter_catalog()
        cat = TaxonomyCatalog()
        for d in domains_db:
            sub_dict: dict[str, SubdomainSpec] = {}
            for s in (d.subdomains or []):
                sub_dict[s.slug] = SubdomainSpec(
                    slug=s.slug,
                    name=s.name,
                    domain_slug=d.slug,
                    standard_bodies=s.standard_bodies,
                )
            if not sub_dict:
                default_sub_slug = f"{d.slug}.general" if d.slug != "general" else "general.general"
                sub_dict[default_sub_slug] = SubdomainSpec(
                    slug=default_sub_slug,
                    name=f"{d.name} General",
                    domain_slug=d.slug,
                )

            # Preserve rich descriptions/keywords for system domains if DB field is blank
            starter = starter_cat.get_domain(d.slug)
            desc = d.description or (starter.description if starter else "")
            keywords = d.classifier_keywords or (starter.classifier_keywords if starter else None)
            chunking_strat = d.chunking_strategy or (starter.chunking_strategy if starter else "standard_markdown")

            cat.domains[d.slug] = DomainSpec(
                slug=d.slug,
                name=d.name,
                epistemic_type=d.epistemic_type,
                description=desc,
                icon=d.icon or "📁",
                color_token=d.color_token or "slate",
                chunking_strategy=chunking_strat,
                classifier_keywords=keywords,
                authority_priority=d.authority_priority,
                is_system=d.is_system,
                is_active=d.is_active,
                subdomains=sub_dict,
            )
        _TAXONOMY_CATALOG = cat
        return _TAXONOMY_CATALOG
    except Exception as exc:
        logger.warning("Failed refreshing taxonomy catalog: %s; falling back to starter catalog", exc)
        if _TAXONOMY_CATALOG is None:
            _TAXONOMY_CATALOG = _build_starter_catalog()
        return _TAXONOMY_CATALOG


def build_dynamic_taxonomy_questions(catalog: TaxonomyCatalog) -> dict[str, dict[str, Any]]:
    """Build dynamic taxonomy questions and criteria from the catalog."""
    criteria: dict[str, str] = {}
    for slug, d in catalog.domains.items():
        if d.description:
            criteria[slug] = f"{d.name}: {d.description}"
        else:
            criteria[slug] = f"{d.name} domain knowledge and documentation."

    questions: dict[str, dict[str, Any]] = {
        "domain": {
            "type": "choice",
            "instructions": (
                "Classify the primary professional domain of this document (English/Vietnamese) / "
                "Xác định lĩnh vực chuyên môn chính của văn bản (tiếng Anh hoặc tiếng Việt):"
            ),
            "criteria": criteria,
        }
    }

    # Add subdomain questions for domains with >= 2 subdomains
    for slug, d in catalog.domains.items():
        if len(d.subdomains) >= 2:
            sub_q_key = f"{slug}_subdomain"
            sub_criteria: dict[str, str] = {}
            for s_slug, s in d.subdomains.items():
                body = f" ({s.standard_bodies})" if s.standard_bodies else ""
                sub_criteria[s_slug] = f"{s.name}{body}"
            questions[sub_q_key] = {
                "type": "choice",
                "instructions": (
                    f"Determine the specific {d.name} subdomain / "
                    f"Xác định chuyên ngành {d.name}:"
                ),
                "criteria": sub_criteria,
            }

    return questions


# Bilingual / language-agnostic taxonomy question definitions for Laya (starter defaults)
TAXONOMY_QUESTIONS: dict[str, dict[str, Any]] = build_dynamic_taxonomy_questions(_build_starter_catalog())


def _extract_authority_and_jurisdiction(text: str) -> tuple[int, Optional[str], Optional[int]]:
    """Heuristically extract authority tier, jurisdiction, and effective year from standard/statute citations."""
    sample = text[:3000]

    # Vietnamese statutory citations
    if re.search(r"Luật\s+số\s+[\d/]+QH[\d]+", sample, re.IGNORECASE):
        yr_m = re.search(r"QH(\d{2})", sample)
        year = None
        if yr_m:
            yr_short = int(yr_m.group(1))
            year = 2000 + yr_short if yr_short < 50 else 1900 + yr_short
        return 1, "VN-STATUTE", year

    if re.search(r"Nghị\s+định\s+[\d/]+NĐ-CP", sample, re.IGNORECASE):
        yr_m = re.search(r"(\d{4})/NĐ-CP", sample)
        year = int(yr_m.group(1)) if yr_m else None
        return 2, "VN-GOV", year

    if re.search(r"Thông\s+tư\s+[\d/]+TT-BTC", sample, re.IGNORECASE):
        yr_m = re.search(r"(\d{4})/TT-BTC", sample)
        year = int(yr_m.group(1)) if yr_m else None
        return 2, "VN-MINISTRY", year

    # Vietnamese standards
    if re.search(r"TCVN\s+[\d\-:]+", sample, re.IGNORECASE):
        yr_m = re.search(r":(\d{4})", sample)
        year = int(yr_m.group(1)) if yr_m else None
        return 1, "VN-STANDARD", year

    if re.search(r"QCVN\s+[\d\-:]+", sample, re.IGNORECASE):
        yr_m = re.search(r":(\d{4})", sample)
        year = int(yr_m.group(1)) if yr_m else None
        return 1, "VN-STANDARD", year

    if re.search(r"\bVAS\s+\d+", sample, re.IGNORECASE):
        return 1, "VN-VAS", None

    # US / International Statutory & Regulatory
    if re.search(r"\b\d+\s+U\.S\.C\.", sample, re.IGNORECASE):
        return 1, "US-FED", None

    if re.search(r"\b\d+\s+CFR\s+\d+", sample, re.IGNORECASE):
        return 2, "US-FED", None

    # International Engineering Standards
    if re.search(r"\bIEEE\s+[A-Z0-9.]+", sample, re.IGNORECASE):
        yr_m = re.search(r"IEEE\s+[A-Z0-9.]+(?:-(\d{4}))?", sample, re.IGNORECASE)
        year = int(yr_m.group(1)) if (yr_m and yr_m.group(1)) else None
        return 1, "IEEE", year

    if re.search(r"\bIEC\s+\d+", sample, re.IGNORECASE):
        return 1, "IEC", None

    if re.search(r"\bASME\s+[A-Z0-9.]+", sample, re.IGNORECASE):
        return 1, "ASME", None

    # International Accounting Standards
    if re.search(r"\b(?:IFRS|IAS)\s+\d+", sample, re.IGNORECASE):
        return 1, "IFRS", None

    if re.search(r"\b(?:US\s+GAAP|ASC\s+\d+)", sample, re.IGNORECASE):
        return 1, "US-GAAP", None

    # SEC filings
    if re.search(r"\bForm\s+10-K\b", sample, re.IGNORECASE):
        yr_m = re.search(r"Ended\s+[A-Za-z]+\s+\d{1,2},\s+(\d{4})", sample, re.IGNORECASE)
        year = int(yr_m.group(1)) if yr_m else None
        return 2, "SEC", year

    # Default general tier
    yr_m = re.search(r"\b(20\d{2}|19\d{2})\b", sample)
    year = int(yr_m.group(1)) if yr_m else None
    return 3, None, year


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


def _make_keyword_pattern(kw: str) -> str:
    """Safely escape keyword and apply word boundaries or whitespace boundaries for punctuation."""
    escaped = re.escape(kw)
    prefix = r"\b" if re.match(r"^\w", kw) else r"(?:(?<=\W)|^)"
    suffix = r"\b" if re.search(r"\w$", kw) else r"(?:(?=\W)|$)"
    return f"{prefix}{escaped}{suffix}"


def _score_domain_and_subdomain(
    text: str,
    catalog: TaxonomyCatalog,
) -> tuple[dict[str, float], dict[str, str]]:
    """Score all non-general domains and select subdomains using catalog keywords and safe regexes."""
    text_lower = text.lower()
    domain_scores: dict[str, float] = {}
    best_subdomains: dict[str, str] = {}

    for slug, d in catalog.domains.items():
        if slug == "general" or not d.is_active:
            continue

        raw_kws: list[str] = []
        if d.classifier_keywords:
            raw_kws = [k.strip() for k in d.classifier_keywords.split(",") if k.strip()]
        else:
            # Fallback: tokenized words from domain name and subdomains
            words = re.findall(r"\w+", d.name.lower())
            for s in d.subdomains.values():
                words.extend(re.findall(r"\w+", s.name.lower()))
                if s.standard_bodies:
                    words.extend([b.strip().lower() for b in s.standard_bodies.split(",") if b.strip()])
            raw_kws = [w for w in set(words) if len(w) > 2]

        score = 0.0
        if raw_kws:
            # ReDoS protection: escape every keyword and compile safe boundary regex
            patterns = [_make_keyword_pattern(kw) for kw in raw_kws]
            pattern = re.compile(rf"(?:{'|'.join(patterns)})", re.IGNORECASE)
            matches = pattern.findall(text_lower)
            # Custom domains get weight 3.0 per hit if custom, else 1.0 (starter counts match raw hits)
            weight = 1.0 if d.is_system else 3.0
            score = float(len(matches)) * weight

        domain_scores[slug] = score

        # Subdomain resolution
        assigned_sub = catalog.get_default_subdomain(slug)
        # Check specific cues if starter domain
        if slug == "engineering":
            if any(k in text_lower for k in ("transformer", "dielectric", "voltage", "kv", "ieee", "máy biến áp", "điện")):
                assigned_sub = "eng.electrical"
            elif any(k in text_lower for k in ("scada", "pid", "plc", "feedback", "control", "điều khiển")):
                assigned_sub = "eng.control"
            elif any(k in text_lower for k in ("stress", "thermal", "fatigue", "heat", "cơ học", "nhiệt")):
                assigned_sub = "eng.mechanical"
            elif "eng.technology" in d.subdomains:
                assigned_sub = "eng.technology"
        elif slug == "economics":
            if any(k in text_lower for k in ("balance sheet", "depreciation", "kế toán", "khấu hao", "ifrs", "gaap", "vas")):
                assigned_sub = "econ.accounting"
            elif any(k in text_lower for k in ("tax", "thuế", "irc")):
                assigned_sub = "econ.taxation"
            elif any(k in text_lower for k in ("dcf", "npv", "irr", "portfolio", "đầu tư")):
                assigned_sub = "econ.investment"
            elif "econ.finance" in d.subdomains:
                assigned_sub = "econ.finance"
        elif slug == "law":
            if any(k in text_lower for k in ("cfr", "regulation", "nghị định", "quy chuẩn", "directive", "agency")):
                assigned_sub = "law.regulation"
            elif any(k in text_lower for k in ("contract", "agreement", "covenant", "warranty", "hợp đồng")):
                assigned_sub = "law.contract"
            elif any(k in text_lower for k in ("case", "court", "precedent", "appellate", "tòa án", "bản án")):
                assigned_sub = "law.case_law"
            elif "law.statute" in d.subdomains:
                assigned_sub = "law.statute"
        else:
            # Custom domain: match subdomain names and standard bodies
            best_sub_hits = -1
            best_sub = assigned_sub
            for s_slug, s in d.subdomains.items():
                s_words = [w for w in re.findall(r"\w+", s.name.lower()) if len(w) > 2]
                if s.standard_bodies:
                    s_words.extend([b.strip().lower() for b in s.standard_bodies.split(",") if b.strip()])
                if s_words:
                    s_patterns = [_make_keyword_pattern(w) for w in s_words]
                    s_pattern = re.compile(rf"(?:{'|'.join(s_patterns)})", re.IGNORECASE)
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
                # Boost confidence if dominant
                if confidence > 0.5:
                    confidence = min(0.98, round(confidence + 0.3, 2))
                subdomain = best_subs.get(primary_domain, catalog.get_default_subdomain(primary_domain))
            else:
                primary_domain = "general"
                subdomain = "general.general"
                confidence = 0.50
        else:
            subdomain = best_subs.get(primary_domain, catalog.get_default_subdomain(primary_domain))

        epistemic = catalog.get_epistemic_modality(primary_domain)
        chunking_strategy = catalog.get_chunking_strategy(primary_domain)
        tier, jurisdiction, eff_year = _extract_authority_and_jurisdiction(text)

        overrides: list[SectionOverride] = list(self.inject_overrides)
        # Scan sections for overrides if sections provided
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
            if not getattr(ov, "chunking_strategy", None) or ov.chunking_strategy == "standard_markdown":
                ov.chunking_strategy = catalog.get_chunking_strategy(ov.domain)

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


def _extract_laya_answer(pred: Any, q_key: str) -> tuple[Optional[str], float]:
    """Safely extract choice/answer and confidence from Laya agent prediction payload.

    Supports both Laya v1/v2 schema (nested under 'answers' with 'choice' and 'answer_confidence')
    and legacy/flat dictionaries.
    """
    if not isinstance(pred, dict):
        return None, 0.0

    # Look under "answers" dict first (laya-rl-agent standard), then root level
    answers_dict = pred.get("answers")
    item = None
    if isinstance(answers_dict, dict) and q_key in answers_dict:
        item = answers_dict.get(q_key)
    elif q_key in pred:
        item = pred.get(q_key)

    if not isinstance(item, dict):
        return None, 0.0

    # Choice may be in "choice" (standard for choice questions) or "answer" or "value"
    choice = item.get("choice")
    if choice is None:
        choice = item.get("answer")
    if choice is None:
        choice = item.get("value")
    if choice is not None:
        choice = str(choice).strip()

    # Calibrated confidence: prefer "answer_confidence" then "confidence"
    raw_conf = item.get("answer_confidence")
    if raw_conf is None:
        raw_conf = item.get("confidence", 0.0)
    try:
        conf = float(raw_conf)
    except (TypeError, ValueError):
        conf = 0.0

    return choice, conf


class LayaDomainClassifier(BaseDomainClassifier):
    """Lazy singleton loading convaiinnovations/laya-multilingual with H1/H2 override scanning and LLM fallback."""

    def __init__(
        self,
        model_name: str = "convaiinnovations/laya-multilingual",
        device: Optional[str] = None,
        enable_llm_fallback: bool = True,
        high_confidence_threshold: float = 0.85,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.enable_llm_fallback = enable_llm_fallback
        self.high_confidence_threshold = high_confidence_threshold
        self._agent: Any = None
        self._load_failed: bool = False
        self._load_error: Optional[str] = None

    def _ensure_model_loaded(self) -> Any:
        """Lazily initialize the Laya agent."""
        if self._agent is not None:
            return self._agent
        if self._load_failed:
            raise RuntimeError(f"Laya model load previously failed: {self._load_error}")

        t0 = time.perf_counter()
        try:
            import laya

            # Support laya.Laya or laya.load
            if hasattr(laya, "Laya") and callable(getattr(laya, "Laya")):
                self._agent = laya.Laya(self.model_name)
            elif hasattr(laya, "load") and callable(getattr(laya, "load")):
                self._agent = laya.load(self.model_name, device=self.device)
            elif hasattr(laya, "Agent") and callable(getattr(laya, "Agent")):
                self._agent = laya.Agent(self.model_name, device=self.device)
            else:
                raise AttributeError("laya module does not expose Laya, load, or Agent entry points")

            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logger.info("Loaded Laya classifier model '%s' in %dms", self.model_name, elapsed_ms)
            return self._agent
        except Exception as exc:
            self._load_failed = True
            self._load_error = str(exc)
            logger.warning("Failed loading Laya classifier model '%s': %s", self.model_name, exc)
            raise

    def unload(self) -> bool:
        """Unload resident model weights and release memory."""
        unloaded = False
        if self._agent is not None:
            self._agent = None
            unloaded = True
        self._load_failed = False
        self._load_error = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return unloaded

    @property
    def is_resident(self) -> bool:
        return self._agent is not None

    async def _classify_with_llm(self, text: str, filename: Optional[str]) -> Optional[RecommendedDomainProfile]:
        """Opt-in LLM fallback when Laya confidence is low (< 0.60)."""
        settings = get_settings()
        sample = text[:3500]
        catalog = get_taxonomy_catalog()

        domain_lines = [
            f"- '{slug}': {d.name} ({d.description})"
            for slug, d in catalog.domains.items()
        ]
        subdomain_lines = [
            f"- {slug}: {', '.join(d.subdomains.keys())}"
            for slug, d in catalog.domains.items()
            if d.subdomains
        ]
        domain_options = " | ".join(f"'{s}'" for s in catalog.domains.keys())

        system_msg = SystemMessage(
            content=(
                "You are a professional multi-domain classification expert for an enterprise RAG system.\n"
                f"Classify the document into exactly one primary domain ({domain_options}).\n\n"
                "Allowed domains:\n"
                + "\n".join(domain_lines) + "\n\n"
                "Allowed subdomains:\n"
                + "\n".join(subdomain_lines) + "\n\n"
                "Assign epistemic modality: empirical, evaluative, normative, or general.\n"
                "Assign authority tier: 1 (highest statute/standard) to 5 (informal).\n\n"
                "Respond strictly in valid JSON matching this schema:\n"
                "{\n"
                '  "primary_domain": "string",\n'
                '  "primary_subdomain": "string",\n'
                '  "epistemic_modality": "string",\n'
                '  "confidence": 0.0 to 1.0,\n'
                '  "authority_tier": 1 to 5,\n'
                '  "jurisdiction": "string or null",\n'
                '  "effective_year": integer or null,\n'
                '  "reasoning": "brief explanation"\n'
                "}"
            )
        )
        human_msg = HumanMessage(
            content=(
                f"<document_sample filename=\"{filename or 'Unknown'}\">\n"
                f"{sample}\n"
                "</document_sample>"
            )
        )

        try:
            from src.rag.llm import get_chat_model, resolve_provider_and_model

            prov, _ = resolve_provider_and_model(settings=settings)
            llm = get_chat_model(settings=settings, temperature=0.0)

            response = await llm.ainvoke([system_msg, human_msg])
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                content = "".join(str(c) for c in content)

            # Strip markdown json fences
            clean_json = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.MULTILINE)
            clean_json = re.sub(r"\s*```$", "", clean_json.strip(), flags=re.MULTILINE)
            data = json.loads(clean_json)

            domain = data.get("primary_domain", "general").lower()
            if domain not in catalog.domains:
                domain = "general"
            subdomain = data.get("primary_subdomain")
            dom_spec = catalog.get_domain(domain)
            if not subdomain or (dom_spec and subdomain not in dom_spec.subdomains):
                subdomain = catalog.get_default_subdomain(domain)
            epistemic = catalog.get_epistemic_modality(domain)
            chunking_strategy = catalog.get_chunking_strategy(domain)
            confidence = float(data.get("confidence", 0.85))
            tier = int(data.get("authority_tier", 3))
            jurisdiction = data.get("jurisdiction")
            eff_year = data.get("effective_year")

            return RecommendedDomainProfile(
                primary_domain=domain,
                primary_subdomain=subdomain,
                epistemic_modality=epistemic,
                confidence=confidence,
                confidence_subdomain=confidence,
                authority_tier=tier,
                jurisdiction=jurisdiction,
                effective_year=eff_year,
                is_current=True,
                is_hybrid=False,
                chunking_strategy=chunking_strategy,
                reasoning=f"LLM fallback classification ({prov}): {data.get('reasoning', '')}",
            )
        except Exception as exc:
            logger.debug("Opt-in LLM classification fallback failed: %s", exc)
            return None

    async def classify_document(
        self,
        text: str,
        *,
        sections: Optional[list[dict[str, str]]] = None,
        filename: Optional[str] = None,
    ) -> DomainClassificationResult:
        t0 = time.perf_counter()

        # Try to load and invoke Laya
        try:
            agent = self._ensure_model_loaded()
            raw_answers: dict[str, Any] = {}
            catalog = get_taxonomy_catalog()
            questions = build_dynamic_taxonomy_questions(catalog)

            # Prepare state context (title/filename + first 1000 tokens of text)
            state_text = f"Title / Filename: {filename or 'Unknown'}\n\n{text[:3500]}"

            # Stage 1: Domain classification
            domain_q = {"domain": questions["domain"]}
            domain_pred = agent.predict(state_text, domain_q)
            raw_answers["domain_prediction"] = domain_pred

            extracted_domain, extracted_conf = _extract_laya_answer(domain_pred, "domain")
            domain_ans = extracted_domain or "general"
            domain_conf = extracted_conf

            if domain_ans not in catalog.domains:
                domain_ans = "general"

            # Stage 2: Subdomain classification conditioned on domain
            sub_q_key = f"{domain_ans}_subdomain"
            subdomain_ans = catalog.get_default_subdomain(domain_ans)
            sub_conf = domain_conf

            if sub_q_key in questions:
                sub_q = {sub_q_key: questions[sub_q_key]}
                sub_pred = agent.predict(state_text, sub_q)
                raw_answers["subdomain_prediction"] = sub_pred
                cand_sub, cand_conf = _extract_laya_answer(sub_pred, sub_q_key)
                dom_obj = catalog.get_domain(domain_ans)
                if cand_sub and dom_obj and cand_sub in dom_obj.subdomains:
                    subdomain_ans = cand_sub
                    sub_conf = cand_conf

            # Authority and jurisdiction heuristics
            tier, jurisdiction, eff_year = _extract_authority_and_jurisdiction(text)

            # Stage 3: H1/H2 Section-override scanning
            section_overrides: list[SectionOverride] = []
            if sections:
                for sec in sections[:20]:
                    h_path = sec.get("heading_path", "")
                    sec_content = sec.get("text", "").strip()
                    if len(sec_content) < 80:
                        continue

                    sec_state = f"Heading: {h_path}\n\n{sec_content[:2000]}"
                    sec_pred = agent.predict(sec_state, domain_q)
                    sec_domain, sec_conf = _extract_laya_answer(sec_pred, "domain")

                    # Section override threshold: confidence >= 0.75 and different domain
                    if sec_domain and sec_domain in catalog.domains and sec_domain != domain_ans and sec_conf >= 0.75:
                        sec_sub = catalog.get_default_subdomain(sec_domain)
                        sec_sub_key = f"{sec_domain}_subdomain"
                        if sec_sub_key in questions:
                            sec_sub_pred = agent.predict(sec_state, {sec_sub_key: questions[sec_sub_key]})
                            cand_sec_sub, _ = _extract_laya_answer(sec_sub_pred, sec_sub_key)
                            sec_dom_obj = catalog.get_domain(sec_domain)
                            if cand_sec_sub and sec_dom_obj and cand_sec_sub in sec_dom_obj.subdomains:
                                sec_sub = cand_sec_sub

                        section_overrides.append(
                            SectionOverride(
                                heading_path=h_path or "Section",
                                domain=sec_domain,
                                subdomain=sec_sub,
                                epistemic_modality=catalog.get_epistemic_modality(sec_domain),
                                confidence=sec_conf,
                                authority_tier=tier,
                                jurisdiction=jurisdiction,
                                chunking_strategy=catalog.get_chunking_strategy(sec_domain),
                            )
                        )

            # Check low-confidence LLM fallback trigger (< 0.60)
            if domain_conf < 0.60 and self.enable_llm_fallback:
                logger.info(
                    "Laya domain confidence %.2f < 0.60; attempting opt-in LLM classification fallback",
                    domain_conf,
                )
                llm_profile = await self._classify_with_llm(text, filename)
                if llm_profile is not None:
                    llm_profile.section_overrides = section_overrides
                    llm_profile.is_hybrid = len(section_overrides) > 0
                    elapsed_ms = max(1, int((time.perf_counter() - t0) * 1000))
                    return DomainClassificationResult(
                        recommended=llm_profile,
                        fallback_used=True,
                        fallback_reason=f"Laya low confidence ({domain_conf:.2f} < 0.60)",
                        execution_ms=elapsed_ms,
                        raw_answers=raw_answers,
                    )

            profile = RecommendedDomainProfile(
                primary_domain=domain_ans,
                primary_subdomain=subdomain_ans,
                epistemic_modality=catalog.get_epistemic_modality(domain_ans),
                confidence=domain_conf,
                confidence_subdomain=sub_conf,
                authority_tier=tier,
                jurisdiction=jurisdiction,
                effective_year=eff_year,
                is_current=True,
                is_hybrid=len(section_overrides) > 0,
                section_overrides=section_overrides,
                reasoning=f"Laya decision: domain={domain_ans} ({domain_conf:.2f}), subdomain={subdomain_ans}",
                chunking_strategy=catalog.get_chunking_strategy(domain_ans),
            )
            elapsed_ms = max(1, int((time.perf_counter() - t0) * 1000))
            return DomainClassificationResult(
                recommended=profile,
                fallback_used=False,
                execution_ms=elapsed_ms,
                raw_answers=raw_answers,
            )

        except Exception as exc:
            # Fall back to StubLayaClassifier or LLM when Laya fails to load or run
            logger.warning("Laya classifier execution failed (%s); engaging fallback classifier", exc)
            if self.enable_llm_fallback:
                llm_profile = await self._classify_with_llm(text, filename)
                if llm_profile is not None:
                    elapsed_ms = max(1, int((time.perf_counter() - t0) * 1000))
                    return DomainClassificationResult(
                        recommended=llm_profile,
                        fallback_used=True,
                        fallback_reason=f"Laya error ({exc}); resolved via LLM",
                        execution_ms=elapsed_ms,
                    )

            stub = StubLayaClassifier()
            stub_res = await stub.classify_document(text, sections=sections, filename=filename)
            stub_res.fallback_used = True
            stub_res.fallback_reason = f"Laya error ({exc}); resolved via heuristic stub"
            stub_res.execution_ms = max(1, int((time.perf_counter() - t0) * 1000))
            return stub_res


# Module-level singleton state
_CLASSIFIER_INSTANCE: Optional[BaseDomainClassifier] = None
_FORCE_STUB: bool = False


def set_classifier_force_stub(force: bool) -> None:
    """Explicitly force use of StubLayaClassifier (e.g. for deterministic unit test suites)."""
    global _FORCE_STUB, _CLASSIFIER_INSTANCE
    _FORCE_STUB = force
    if force and isinstance(_CLASSIFIER_INSTANCE, LayaDomainClassifier):
        _CLASSIFIER_INSTANCE.unload()
        _CLASSIFIER_INSTANCE = None


def get_classifier(model_name: Optional[str] = None) -> BaseDomainClassifier:
    """Return the lazy singleton classifier instance."""
    global _CLASSIFIER_INSTANCE
    if _CLASSIFIER_INSTANCE is not None:
        return _CLASSIFIER_INSTANCE

    if _FORCE_STUB:
        _CLASSIFIER_INSTANCE = StubLayaClassifier()
        return _CLASSIFIER_INSTANCE

    settings = get_settings()
    m_name = model_name or settings.ingestion.laya_model_name
    _CLASSIFIER_INSTANCE = LayaDomainClassifier(
        model_name=m_name,
        device=settings.ingestion.device if settings.ingestion.device != "auto" else None,
        enable_llm_fallback=settings.ingestion.enable_llm_classification_fallback,
        high_confidence_threshold=settings.ingestion.high_confidence_threshold,
    )
    return _CLASSIFIER_INSTANCE


def unload_classifier() -> bool:
    """Unload resident classifier weights and free VRAM/RAM."""
    global _CLASSIFIER_INSTANCE
    if _CLASSIFIER_INSTANCE is not None:
        if isinstance(_CLASSIFIER_INSTANCE, LayaDomainClassifier):
            _CLASSIFIER_INSTANCE.unload()
        _CLASSIFIER_INSTANCE = None
        gc.collect()
        return True
    return False


def is_classifier_resident() -> bool:
    """Check if Laya classifier model is currently resident in memory."""
    if _CLASSIFIER_INSTANCE is not None and isinstance(_CLASSIFIER_INSTANCE, LayaDomainClassifier):
        return _CLASSIFIER_INSTANCE.is_resident
    return False


def get_classifier_model_name() -> Optional[str]:
    """Get active classifier model name if resident or configured."""
    if _CLASSIFIER_INSTANCE is not None and isinstance(_CLASSIFIER_INSTANCE, LayaDomainClassifier):
        return _CLASSIFIER_INSTANCE.model_name
    return get_settings().ingestion.laya_model_name
