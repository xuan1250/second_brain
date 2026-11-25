"""Workspace settings, LLM provider options, and credential verification schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.api.schemas.common import ALLOWED_LLM_PROVIDERS, _validate_safe_base_url
from src.core.credentials import is_valid_api_key


class ProviderOption(BaseModel):
    """Read-only description of an LLM provider option for the UI selector (never exposes raw keys)."""

    provider: str
    label: str
    configured: bool
    default_model: str = ""
    key_source: Literal["ui", "env", "none"] = "none"
    masked_key: str | None = None
    validation_status: Literal["valid", "invalid", "untested", "missing"] = "missing"
    validation_error: str | None = None
    base_url: str | None = None


class WorkspaceSettingsResponse(BaseModel):
    """Workspace-wide upload, approval, and AI defaults."""

    model_config = ConfigDict(from_attributes=True)

    id: str = "default"
    default_upload_mode: str = "fast"
    default_approval_mode: str = "review_all"
    high_confidence_threshold: float = 0.85
    default_llm_provider: str | None = None
    default_llm_model: str | None = None
    llm_compat_base_url: str | None = None
    available_providers: list[ProviderOption] = Field(default_factory=list)
    updated_at: datetime | None = None


class WorkspaceSettingsUpdateRequest(BaseModel):
    """Payload to update workspace-wide defaults."""

    default_upload_mode: Literal["auto", "gpu", "fast"] | None = None
    default_approval_mode: Literal["review_all", "auto_high_confidence", "auto_all"] | None = None
    high_confidence_threshold: float | None = Field(default=None, ge=0.50, le=0.99)
    default_llm_provider: str | None = None
    default_llm_model: str | None = None
    api_keys: dict[str, str | None] | None = None
    llm_compat_base_url: str | None = None

    @field_validator("default_llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str | None) -> str | None:
        if v is None:
            return None
        prov = v.strip().lower()
        if prov not in ALLOWED_LLM_PROVIDERS:
            raise ValueError(f"default_llm_provider must be one of {sorted(ALLOWED_LLM_PROVIDERS)}")
        return prov

    @field_validator("api_keys")
    @classmethod
    def validate_api_keys(cls, v: dict[str, str | None] | None) -> dict[str, str | None] | None:
        if v is None:
            return None
        allowed_providers = {"openai", "anthropic", "deepseek", "openai_compatible"}
        cleaned: dict[str, str | None] = {}
        for prov, key in v.items():
            norm_prov = prov.strip().lower()
            if norm_prov not in allowed_providers:
                raise ValueError(f"Unknown API key provider '{prov}'. Must be one of {sorted(allowed_providers)}")
            if key is not None and key.strip() != "":
                if not is_valid_api_key(key):
                    raise ValueError(
                        f"Invalid API key format for '{prov}' (placeholders and whitespace/control characters are not allowed)"
                    )
                cleaned[norm_prov] = key.strip()
            else:
                cleaned[norm_prov] = "" if key is not None else None
        return cleaned

    @field_validator("llm_compat_base_url")
    @classmethod
    def validate_llm_compat_base_url(cls, v: str | None) -> str | None:
        return _validate_safe_base_url(v)


class VerifyKeyRequest(BaseModel):
    provider: Literal["openai", "anthropic", "deepseek", "openai_compatible"]
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    save_if_valid: bool = False


class VerifyKeyResponse(BaseModel):
    provider: str
    valid: bool
    validation_status: Literal["valid", "invalid", "untested", "missing"]
    validation_error: str | None = None
    key_source: Literal["ui", "env", "none"] = "none"
    masked_key: str | None = None
    tested_model: str = ""
    provider_option: ProviderOption
