"""Shared repository helpers for JSON settings parsing, regex matching, SHA checks, and JSON reassignment."""
from __future__ import annotations

import json
import re

from src.db.settings_bootstrap import parse_settings_json_dict

__all__ = [
    "compile_word_boundary_regex",
    "is_valid_sha256_hex",
    "parse_settings_json_dict",
    "rewrite_classification_json_for_reassignment",
]


def compile_word_boundary_regex(term: str) -> re.Pattern[str]:
    """Compile a Unicode-aware, case-insensitive word-boundary regex for `term`."""
    return re.compile(r"(?<!\w)" + re.escape(term.strip().lower()) + r"(?!\w)", re.IGNORECASE)


def is_valid_sha256_hex(sha: str | None) -> bool:
    """Return True if `sha` is a 64-character lowercase hexadecimal SHA-256 digest."""
    return bool(sha and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha))


def rewrite_classification_json_for_reassignment(
    raw_json: str | None,
    *,
    target_subdomain_slug: str,
    old_domain_slug: str | None = None,
    target_domain_slug: str | None = None,
    target_epistemic_type: str | None = None,
    target_chunking_strategy: str | None = None,
    old_subdomain_slug: str | None = None,
) -> str | None:
    """Rewrite a document's serialized classification JSON during domain or subdomain deletion.

    Returns the updated JSON string when modified, or None if no changes were needed or input is invalid.
    """
    if not raw_json:
        return None
    try:
        prof = json.loads(raw_json)
    except Exception:
        return None
    if not isinstance(prof, dict):
        return None

    prof_modified = False
    if old_domain_slug is not None and target_domain_slug is not None:
        if prof.get("primary_domain") == old_domain_slug:
            prof["primary_domain"] = target_domain_slug
            prof["primary_subdomain"] = target_subdomain_slug
            if target_epistemic_type is not None:
                prof["epistemic_modality"] = target_epistemic_type
            if target_chunking_strategy:
                prof["chunking_strategy"] = target_chunking_strategy
            prof_modified = True

        for ov in prof.get("section_overrides", []):
            if isinstance(ov, dict) and ov.get("domain") == old_domain_slug:
                ov["domain"] = target_domain_slug
                ov["subdomain"] = target_subdomain_slug
                if target_epistemic_type is not None:
                    ov["epistemic_modality"] = target_epistemic_type
                prof_modified = True
    elif old_subdomain_slug is not None:
        if prof.get("primary_subdomain") == old_subdomain_slug:
            prof["primary_subdomain"] = target_subdomain_slug
            prof_modified = True

        for ov in prof.get("section_overrides", []):
            if isinstance(ov, dict) and ov.get("subdomain") == old_subdomain_slug:
                ov["subdomain"] = target_subdomain_slug
                prof_modified = True

    return json.dumps(prof) if prof_modified else None
