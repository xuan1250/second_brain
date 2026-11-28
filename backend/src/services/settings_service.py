"""Workspace settings synchronization and live LLM credential verification service."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    ProviderOption,
    VerifyKeyRequest,
    VerifyKeyResponse,
    WorkspaceSettingsResponse,
    WorkspaceSettingsUpdateRequest,
    _validate_safe_base_url,
)
from src.config import Settings
from src.core.credentials import (
    _key_validations,
    compute_key_hash,
    is_valid_api_key,
    mask_api_key,
    normalize_api_key,
    resolve_provider_credential,
    sanitize_llm_error,
)
from src.db.models import WorkspaceSettings
from src.db.repository import get_workspace_settings, update_workspace_settings
from src.rag.llm import get_chat_model
from src.services.provider_service import (
    FALLBACK_MODELS,
    build_provider_options,
    provider_has_credential,
)


def apply_workspace_settings_to_app_state(
    ws: WorkspaceSettings,
    settings: Settings,
    *,
    default_llm_provider_set: bool = False,
    default_llm_model_set: bool = False,
) -> None:
    """Sync persisted WorkspaceSettings fields onto the active app_state Settings object.

    Preserves base_url isolation: `ws.llm_compat_base_url` is never assigned to
    `settings.llm.base_url`, only to `settings.llm.providers['openai_compatible'].base_url`.
    """
    if ws.high_confidence_threshold is not None:
        settings.ingestion.high_confidence_threshold = ws.high_confidence_threshold

    if default_llm_provider_set:
        settings.llm.provider = ws.default_llm_provider or "openai"
    elif ws.default_llm_provider:
        settings.llm.provider = ws.default_llm_provider

    if default_llm_model_set:
        settings.llm.model_name = ws.default_llm_model or ""
    elif ws.default_llm_model:
        settings.llm.model_name = ws.default_llm_model

    compat_entry = settings.llm.providers.get("openai_compatible")
    if compat_entry is not None and ws.llm_compat_base_url:
        compat_entry.base_url = ws.llm_compat_base_url


def build_workspace_settings_response(
    ws: WorkspaceSettings,
    settings: Settings,
) -> WorkspaceSettingsResponse:
    """Assemble WorkspaceSettingsResponse enriched with masked provider options."""
    _, available_providers = build_provider_options(settings)
    return WorkspaceSettingsResponse(
        id=ws.id,
        default_upload_mode=ws.default_upload_mode,
        default_approval_mode=ws.default_approval_mode,
        high_confidence_threshold=ws.high_confidence_threshold,
        default_llm_provider=ws.default_llm_provider,
        default_llm_model=ws.default_llm_model,
        llm_compat_base_url=ws.llm_compat_base_url,
        available_providers=available_providers,
        updated_at=ws.updated_at,
    )


async def get_workspace_settings_view(
    session: AsyncSession,
    settings: Settings,
) -> WorkspaceSettingsResponse:
    """Fetch workspace settings and merge them with live provider options."""
    ws = await get_workspace_settings(session)
    return build_workspace_settings_response(ws, settings)


async def update_workspace_settings_view(
    session: AsyncSession,
    settings: Settings,
    payload: WorkspaceSettingsUpdateRequest,
) -> WorkspaceSettingsResponse:
    """Persist workspace settings updates and synchronize runtime configuration."""
    ws = await update_workspace_settings(
        session=session,
        default_upload_mode=payload.default_upload_mode,
        default_approval_mode=payload.default_approval_mode,
        high_confidence_threshold=payload.high_confidence_threshold,
        default_llm_provider=payload.default_llm_provider,
        default_llm_model=payload.default_llm_model,
        api_keys=payload.api_keys,
        llm_compat_base_url=payload.llm_compat_base_url,
    )
    apply_workspace_settings_to_app_state(
        ws,
        settings,
        default_llm_provider_set=payload.default_llm_provider is not None,
        default_llm_model_set=payload.default_llm_model is not None,
    )
    return build_workspace_settings_response(ws, settings)


def _make_verify_error_response(
    settings: Settings,
    *,
    provider: str,
    validation_status: Literal["invalid", "missing"],
    validation_error: str,
    key_source: Literal["ui", "env", "none"],
    masked_key: str | None,
) -> VerifyKeyResponse:
    """Build a uniform VerifyKeyResponse for pre-flight validation and security guard failures."""
    _, all_opts = build_provider_options(settings)
    opt = next((o for o in all_opts if o.provider == provider), None)
    return VerifyKeyResponse(
        provider=provider,
        valid=False,
        validation_status=validation_status,
        validation_error=validation_error,
        key_source=key_source,
        masked_key=masked_key,
        tested_model="",
        provider_option=opt
        or ProviderOption(
            provider=provider,
            label=provider,
            configured=False,
            validation_status=validation_status,
            validation_error=validation_error,
        ),
    )


def _resolve_test_model(
    provider: str,
    requested_model: str | None,
    settings: Settings,
) -> str:
    """Select the model identifier to ping during key verification."""
    if requested_model and requested_model.strip():
        return requested_model.strip()
    prov_entry = settings.llm.providers.get(provider)
    if prov_entry and prov_entry.model_name and prov_entry.model_name.strip():
        return prov_entry.model_name.strip()
    return FALLBACK_MODELS.get(provider, "gpt-4o")


async def verify_provider_key(
    payload: VerifyKeyRequest,
    session: AsyncSession,
    settings: Settings,
) -> VerifyKeyResponse:
    """Verify an API key via a 5s bounded LLM ping and optionally persist it to WorkspaceSettings."""
    raw_base_url = payload.base_url if payload.provider == "openai_compatible" else None
    resolved_key, resolved_source, resolved_base_url = resolve_provider_credential(
        payload.provider, settings
    )

    candidate_base_url: str | None = None
    if payload.provider == "openai_compatible":
        candidate_base_url = (
            raw_base_url.strip() if raw_base_url and raw_base_url.strip() else resolved_base_url
        )
        if candidate_base_url:
            try:
                candidate_base_url = _validate_safe_base_url(candidate_base_url)
            except ValueError as ve:
                return _make_verify_error_response(
                    settings,
                    provider=payload.provider,
                    validation_status="invalid",
                    validation_error=str(ve),
                    key_source=resolved_source,
                    masked_key=mask_api_key(resolved_key),
                )

        # Anti-exfiltration guard: disallow probing an unsaved Base URL with a stored key.
        if candidate_base_url != resolved_base_url and not (payload.api_key and payload.api_key.strip()):
            return _make_verify_error_response(
                settings,
                provider=payload.provider,
                validation_status="invalid",
                validation_error="Please enter the API key when verifying a new Base URL.",
                key_source=resolved_source,
                masked_key=mask_api_key(resolved_key),
            )

    has_explicit_key = bool(payload.api_key and payload.api_key.strip())
    candidate_key = normalize_api_key(payload.api_key) if has_explicit_key else resolved_key
    candidate_source: Literal["ui", "env", "none"] = "ui" if has_explicit_key else resolved_source

    if not candidate_key:
        return _make_verify_error_response(
            settings,
            provider=payload.provider,
            validation_status="missing",
            validation_error="No API key configured in Workspace Settings or backend/.env.",
            key_source="none",
            masked_key=None,
        )

    if not is_valid_api_key(candidate_key):
        return _make_verify_error_response(
            settings,
            provider=payload.provider,
            validation_status="invalid",
            validation_error=(
                "Invalid API key format (placeholders and whitespace/control characters are not allowed)."
            ),
            key_source=candidate_source,
            masked_key=mask_api_key(candidate_key),
        )

    test_model = _resolve_test_model(payload.provider, payload.model, settings)
    k_hash = compute_key_hash(
        candidate_key,
        candidate_base_url if payload.provider == "openai_compatible" else None,
    )

    llm_kwargs: dict[str, Any] = {
        "provider_override": payload.provider,
        "model_override": test_model,
        "settings": settings,
        "api_key": candidate_key,
        "max_tokens": 1,
        "timeout": 5.0,
        "max_retries": 0,
    }
    if payload.provider == "openai_compatible" and candidate_base_url:
        llm_kwargs["base_url"] = candidate_base_url

    valid = False
    validation_status: Literal["valid", "invalid"] = "invalid"
    validation_error: str | None = None

    try:
        llm = get_chat_model(**llm_kwargs)
        await asyncio.wait_for(llm.ainvoke([HumanMessage(content="ping")]), timeout=5.0)
        valid = True
        validation_status = "valid"
        validation_error = None
    except Exception as exc:
        clean_err, _ = sanitize_llm_error(
            exc, provider=payload.provider, candidate_key=candidate_key
        )
        valid = False
        validation_status = "invalid"
        validation_error = clean_err

    val_entry = {
        "status": validation_status,
        "error": validation_error,
        "key_hash": k_hash,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

    if payload.save_if_valid:
        api_keys_update = {payload.provider: candidate_key} if has_explicit_key else None
        compat_url_update = (
            candidate_base_url
            if (payload.provider == "openai_compatible" and raw_base_url is not None)
            else None
        )
        default_llm_provider_update: str | None = None
        default_llm_model_update: str | None = None

        if valid:
            ws = await get_workspace_settings(session)
            current_default = (ws.default_llm_provider or "").strip().lower()
            should_promote = not current_default or not provider_has_credential(
                current_default, settings
            )
            if should_promote:
                default_llm_provider_update = payload.provider
                default_llm_model_update = test_model

        ws_updated = await update_workspace_settings(
            session=session,
            api_keys=api_keys_update,
            key_validations={payload.provider: val_entry},
            llm_compat_base_url=compat_url_update,
            default_llm_provider=default_llm_provider_update,
            default_llm_model=default_llm_model_update,
        )
        apply_workspace_settings_to_app_state(
            ws_updated,
            settings,
            default_llm_provider_set=default_llm_provider_update is not None,
            default_llm_model_set=default_llm_model_update is not None,
        )
    else:
        _key_validations[payload.provider] = val_entry

    _, all_opts = build_provider_options(settings)
    matching_option = next(
        (opt for opt in all_opts if opt.provider == payload.provider),
        ProviderOption(
            provider=payload.provider,
            label=payload.provider,
            configured=bool(candidate_key),
            default_model=test_model,
            key_source=candidate_source,
            masked_key=mask_api_key(candidate_key),
            validation_status=validation_status,
            validation_error=validation_error,
            base_url=candidate_base_url if payload.provider == "openai_compatible" else None,
        ),
    )

    if not payload.save_if_valid:
        matching_option = matching_option.model_copy(
            update={
                "validation_status": validation_status,
                "validation_error": validation_error,
                "masked_key": mask_api_key(candidate_key) or matching_option.masked_key,
                "key_source": candidate_source if candidate_key else matching_option.key_source,
                "configured": provider_has_credential(
                    payload.provider,
                    resolved_key=candidate_key,
                    resolved_base_url=candidate_base_url,
                ),
                "base_url": (
                    candidate_base_url
                    if payload.provider == "openai_compatible"
                    else matching_option.base_url
                ),
            }
        )

    return VerifyKeyResponse(
        provider=payload.provider,
        valid=valid,
        validation_status=validation_status,
        validation_error=validation_error,
        key_source=matching_option.key_source,
        masked_key=matching_option.masked_key,
        tested_model=test_model,
        provider_option=matching_option,
    )
