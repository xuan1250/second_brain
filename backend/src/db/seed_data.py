"""Static taxonomy and polysemy lexicon seed data for SQLite initialization."""
from __future__ import annotations

from typing import Any

from src.db.seed_lexicon import STARTER_LEXICON

DEFAULT_DOMAINS: list[dict[str, Any]] = [
    {
        "id": "engineering",
        "slug": "engineering",
        "name": "Engineering & Technology",
        "epistemic_type": "empirical",
        "authority_priority": 1,
        "description": "Physical systems, electrical & mechanical engineering, control systems, and technical specifications.",
        "icon": "⚙️",
        "color_token": "cyan",
        "chunking_strategy": "academic_equations",
        "classifier_keywords": None,
        "is_system": True,
        "is_active": True,
    },
    {
        "id": "economics",
        "slug": "economics",
        "name": "Economics & Finance",
        "epistemic_type": "evaluative",
        "authority_priority": 2,
        "description": "Corporate finance, financial accounting, valuation, investments, and taxation policies.",
        "icon": "📈",
        "color_token": "emerald",
        "chunking_strategy": "financial_tables",
        "classifier_keywords": None,
        "is_system": True,
        "is_active": True,
    },
    {
        "id": "law",
        "slug": "law",
        "name": "Law & Regulation",
        "epistemic_type": "normative",
        "authority_priority": 1,
        "description": "Statutes, administrative regulations, case law precedents, and commercial contracts.",
        "icon": "⚖️",
        "color_token": "amber",
        "chunking_strategy": "statutory_clauses",
        "classifier_keywords": None,
        "is_system": True,
        "is_active": True,
    },
    {
        "id": "general",
        "slug": "general",
        "name": "General Knowledge",
        "epistemic_type": "general",
        "authority_priority": 5,
        "description": "General domain knowledge and unclassified documentation.",
        "icon": "📄",
        "color_token": "slate",
        "chunking_strategy": "standard_markdown",
        "classifier_keywords": None,
        "is_system": True,
        "is_active": True,
    },
]

DEFAULT_SUBDOMAINS: list[dict[str, Any]] = [
    # Engineering (4)
    {
        "id": "eng.electrical",
        "domain_id": "engineering",
        "slug": "eng.electrical",
        "name": "Electrical & Power Systems",
        "standard_bodies": "IEEE, IEC, NFPA 70 (NEC), NEMA",
    },
    {
        "id": "eng.control",
        "domain_id": "engineering",
        "slug": "eng.control",
        "name": "Control Systems & Automation",
        "standard_bodies": "ISA, IEC 61131, ISO 13849",
    },
    {
        "id": "eng.mechanical",
        "domain_id": "engineering",
        "slug": "eng.mechanical",
        "name": "Mechanical & Structural Engineering",
        "standard_bodies": "ASME Boiler & Pressure Vessel Code, ISO, ASTM",
    },
    {
        "id": "eng.technology",
        "domain_id": "engineering",
        "slug": "eng.technology",
        "name": "Information Technology & Distributed Systems",
        "standard_bodies": "RFCs, W3C, ISO/IEC 27001",
    },
    # Economics (4)
    {
        "id": "econ.finance",
        "domain_id": "economics",
        "slug": "econ.finance",
        "name": "Corporate Finance & Valuation",
        "standard_bodies": "Corporate Treasury, CFA Standards",
    },
    {
        "id": "econ.accounting",
        "domain_id": "economics",
        "slug": "econ.accounting",
        "name": "Financial Accounting & Reporting",
        "standard_bodies": "US GAAP (FASB), IFRS (IASB)",
    },
    {
        "id": "econ.investment",
        "domain_id": "economics",
        "slug": "econ.investment",
        "name": "Investment & Asset Management",
        "standard_bodies": "SEC Filings (10-K, 10-Q), Investment Prospectuses",
    },
    {
        "id": "econ.taxation",
        "domain_id": "economics",
        "slug": "econ.taxation",
        "name": "Taxation & Fiscal Policy",
        "standard_bodies": "Internal Revenue Code (IRC), OECD BEPS",
    },
    # Law (4)
    {
        "id": "law.statute",
        "domain_id": "law",
        "slug": "law.statute",
        "name": "Statutory Law & Legislation",
        "standard_bodies": "Federal & National Legislative Gazettes",
    },
    {
        "id": "law.regulation",
        "domain_id": "law",
        "slug": "law.regulation",
        "name": "Administrative & Regulatory Law",
        "standard_bodies": "Code of Federal Regulations (CFR), EU Directives",
    },
    {
        "id": "law.case_law",
        "domain_id": "law",
        "slug": "law.case_law",
        "name": "Case Law & Judicial Precedent",
        "standard_bodies": "Supreme Court, Appellate Reporters",
    },
    {
        "id": "law.contract",
        "domain_id": "law",
        "slug": "law.contract",
        "name": "Contract Law & Commercial Agreements",
        "standard_bodies": "EPC Contracts, PPAs, Bilateral Instruments",
    },
    # General (1)
    {
        "id": "general.general",
        "domain_id": "general",
        "slug": "general.general",
        "name": "General Knowledge & Catch-All",
        "standard_bodies": None,
    },
]

__all__ = [
    "DEFAULT_DOMAINS",
    "DEFAULT_SUBDOMAINS",
    "STARTER_LEXICON",
]
