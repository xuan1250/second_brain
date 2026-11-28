"""Starter taxonomy catalog, SQLite refresh lifecycle, and compiled keyword regex caching."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.ingestion.classification.models import DomainSpec, SubdomainSpec
from src.ingestion.constants import DEFAULT_DOMAIN, DEFAULT_EPISTEMIC, DEFAULT_SUBDOMAIN

logger = logging.getLogger(__name__)

# Primary domain epistemic mapping (canonical fallbacks)
DOMAIN_EPISTEMIC_MAP: dict[str, str] = {
    "engineering": "empirical",
    "economics": "evaluative",
    "law": "normative",
    "general": DEFAULT_EPISTEMIC,
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
    DEFAULT_DOMAIN: DEFAULT_SUBDOMAIN,
}


def _make_keyword_pattern(kw: str) -> str:
    """Safely escape keyword and apply word boundaries or whitespace boundaries for punctuation."""
    escaped = re.escape(kw)
    prefix = r"\b" if re.match(r"^\w", kw) else r"(?:(?<=\W)|^)"
    suffix = r"\b" if re.search(r"\w$", kw) else r"(?:(?=\W)|$)"
    return f"{prefix}{escaped}{suffix}"


def _compile_keywords_regex(keywords: list[str]) -> Optional[re.Pattern[str]]:
    if not keywords:
        return None
    patterns = [_make_keyword_pattern(kw) for kw in keywords]
    return re.compile(rf"(?:{'|'.join(patterns)})", re.IGNORECASE)


@dataclass
class TaxonomyCatalog:
    """In-memory domain/subdomain registry with cached compiled keyword patterns."""

    domains: dict[str, DomainSpec] = field(default_factory=dict)
    _domain_regex_cache: dict[str, Optional[re.Pattern[str]]] = field(default_factory=dict, repr=False)
    _subdomain_regex_cache: dict[tuple[str, str], Optional[re.Pattern[str]]] = field(
        default_factory=dict, repr=False
    )

    def get_domain(self, slug: str) -> Optional[DomainSpec]:
        return self.domains.get(slug)

    def get_epistemic_modality(self, domain_slug: str) -> str:
        d = self.domains.get(domain_slug)
        if d:
            return d.epistemic_type
        return DOMAIN_EPISTEMIC_MAP.get(domain_slug, DEFAULT_EPISTEMIC)

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
            f"{domain_slug}.general" if domain_slug != DEFAULT_DOMAIN else DEFAULT_SUBDOMAIN,
        )

    def domain_for_subdomain(self, sub_slug: str) -> Optional[str]:
        for d_slug, d in self.domains.items():
            if sub_slug in d.subdomains:
                return d_slug
        return SUBDOMAIN_DOMAIN_MAP.get(sub_slug)

    def get_domain_keyword_pattern(self, domain_slug: str) -> Optional[re.Pattern[str]]:
        """Return cached compiled keyword regex for a domain slug."""
        if domain_slug in self._domain_regex_cache:
            return self._domain_regex_cache[domain_slug]
        d = self.domains.get(domain_slug)
        if d is None:
            self._domain_regex_cache[domain_slug] = None
            return None
        if d.classifier_keywords:
            raw_kws = [k.strip() for k in d.classifier_keywords.split(",") if k.strip()]
        else:
            words = re.findall(r"\w+", d.name.lower())
            for s in d.subdomains.values():
                words.extend(re.findall(r"\w+", s.name.lower()))
                if s.standard_bodies:
                    words.extend([b.strip().lower() for b in s.standard_bodies.split(",") if b.strip()])
            raw_kws = [w for w in set(words) if len(w) > 2]
        compiled = _compile_keywords_regex(raw_kws)
        self._domain_regex_cache[domain_slug] = compiled
        return compiled

    def get_subdomain_keyword_pattern(
        self, domain_slug: str, sub_slug: str
    ) -> Optional[re.Pattern[str]]:
        """Return cached compiled keyword regex for a custom domain's subdomain."""
        cache_key = (domain_slug, sub_slug)
        if cache_key in self._subdomain_regex_cache:
            return self._subdomain_regex_cache[cache_key]
        d = self.domains.get(domain_slug)
        s = d.subdomains.get(sub_slug) if d else None
        if s is None:
            self._subdomain_regex_cache[cache_key] = None
            return None
        s_words = [w for w in re.findall(r"\w+", s.name.lower()) if len(w) > 2]
        if s.standard_bodies:
            s_words.extend([b.strip().lower() for b in s.standard_bodies.split(",") if b.strip()])
        compiled = _compile_keywords_regex(s_words)
        self._subdomain_regex_cache[cache_key] = compiled
        return compiled


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
                default_sub_slug = f"{d.slug}.general" if d.slug != DEFAULT_DOMAIN else DEFAULT_SUBDOMAIN
                sub_dict[default_sub_slug] = SubdomainSpec(
                    slug=default_sub_slug,
                    name=f"{d.name} General",
                    domain_slug=d.slug,
                )

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
