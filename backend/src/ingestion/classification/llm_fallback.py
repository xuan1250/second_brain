"""Opt-in LLM classification fallback when Laya confidence is low or unavailable."""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.ingestion.classification.models import RecommendedDomainProfile
from src.ingestion.classification.taxonomy import get_taxonomy_catalog
from src.ingestion.constants import DEFAULT_DOMAIN

logger = logging.getLogger(__name__)


async def classify_document_with_llm(
    text: str,
    filename: Optional[str],
) -> Optional[RecommendedDomainProfile]:
    """Classify document domain via configured chat LLM using XML-isolated sample context."""
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

        clean_json = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.MULTILINE)
        clean_json = re.sub(r"\s*```$", "", clean_json.strip(), flags=re.MULTILINE)
        data = json.loads(clean_json)

        domain = data.get("primary_domain", DEFAULT_DOMAIN).lower()
        if domain not in catalog.domains:
            domain = DEFAULT_DOMAIN
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
