"""Shared Pydantic schema constants, string/URL sanitizers, and domain validation mixin."""
from __future__ import annotations

import ipaddress
import re
import urllib.parse
from typing import Any

from pydantic import field_validator

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


def _sanitize_string(v: str) -> str:
    cleaned = re.sub(r"[\r\n\x00-\x1f\x7f-\x9f]", " ", v)
    return " ".join(cleaned.split())


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


class _DomainValidationMixin:
    """Shared field validators for DomainCreateRequest and DomainUpdateRequest."""

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
