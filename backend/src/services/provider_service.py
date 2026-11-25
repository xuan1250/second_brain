"""LLM provider definitions, credential availability checks, and UI option builders."""
from __future__ import annotations

import os
from typing import Any

from src.api.schemas import ProviderOption
from src.config import Settings
from src.core.credentials import (
    _load_dotenv_if_present,
    get_provider_validation_state,
    mask_api_key,
    resolve_provider_credential,
)

PROVIDER_DEFINITIONS: tuple[tuple[str, str], ...] = (
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic"),
    ("deepseek", "DeepSeek"),
    ("openai_compatible", "OpenAI-Compatible (Custom / GLM / vLLM)"),
)

FALLBACK_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5-20250929",
    "deepseek": "deepseek-chat",
    "openai_compatible": "glm-4-plus",
}

_UNSET: Any = object()


def provider_has_credential(
    provider: str,
    settings: Settings | None = None,
    *,
    resolved_key: str | None | Any = _UNSET,
    resolved_base_url: str | None | Any = _UNSET,
) -> bool:
    """Return True when the provider has the minimum required credential material."""
    prov = provider.strip().lower()
    if resolved_key is _UNSET or resolved_base_url is _UNSET:
        key, _, base_url = resolve_provider_credential(prov, settings)
        if resolved_key is _UNSET:
            resolved_key = key
        if resolved_base_url is _UNSET:
            resolved_base_url = base_url

    if prov == "openai_compatible":
        return bool(resolved_key and resolved_base_url)
    return bool(resolved_key)


def build_provider_options(settings: Settings) -> tuple[list[ProviderOption], list[ProviderOption]]:
    """Inspect environment and settings (without exposing secrets) to report provider availability."""
    _load_dotenv_if_present()
    active_provider = (os.environ.get("LLM_PROVIDER") or settings.llm.provider or "openai").strip().lower()
    active_model = (os.environ.get("LLM_MODEL_NAME") or settings.llm.model_name or "").strip()

    all_options: list[ProviderOption] = []
    for prov_id, label in PROVIDER_DEFINITIONS:
        resolved_key, key_source, resolved_base_url = resolve_provider_credential(prov_id, settings)
        is_configured = provider_has_credential(
            prov_id,
            resolved_key=resolved_key,
            resolved_base_url=resolved_base_url,
        )
        masked = mask_api_key(resolved_key)
        val_status, val_err = get_provider_validation_state(
            prov_id,
            resolved_key,
            is_configured,
            resolved_base_url,
        )

        prov_entry = settings.llm.providers.get(prov_id)
        entry_model = (prov_entry.model_name if prov_entry else "").strip()
        default_model = entry_model or (active_model if active_provider == prov_id else "")

        all_options.append(
            ProviderOption(
                provider=prov_id,
                label=label,
                configured=is_configured,
                default_model=default_model,
                key_source=key_source,
                masked_key=masked,
                validation_status=val_status,
                validation_error=val_err,
                base_url=resolved_base_url,
            )
        )

    configured_only = [opt for opt in all_options if opt.configured]
    return configured_only, all_options
