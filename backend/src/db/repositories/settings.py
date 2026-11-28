"""WorkspaceSettings singleton repository with runtime credential & config synchronization."""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.core.credentials import normalize_api_key, sync_runtime_credentials
from src.db.models import WorkspaceSettings, _now
from src.db.repositories.common import parse_settings_json_dict


async def get_workspace_settings(session: AsyncSession) -> WorkspaceSettings:
    """Get the singleton WorkspaceSettings row, initializing it if absent."""
    res = await session.execute(select(WorkspaceSettings).where(WorkspaceSettings.id == "default"))
    ws = res.scalars().first()
    if ws is None:
        ws = WorkspaceSettings(id="default")
        session.add(ws)
        await session.commit()
        await session.refresh(ws)
    return ws


async def update_workspace_settings(
    session: AsyncSession,
    default_upload_mode: str | None = None,
    default_approval_mode: str | None = None,
    high_confidence_threshold: float | None = None,
    default_llm_provider: str | None = None,
    default_llm_model: str | None = None,
    api_keys: dict[str, str | None] | None = None,
    key_validations: dict[str, dict[str, Any] | None] | None = None,
    llm_compat_base_url: str | None = None,
) -> WorkspaceSettings:
    """Update WorkspaceSettings in SQLite and sync runtime configuration."""
    ws = await get_workspace_settings(session)
    if default_upload_mode is not None:
        ws.default_upload_mode = default_upload_mode
    if default_approval_mode is not None:
        ws.default_approval_mode = default_approval_mode
    if high_confidence_threshold is not None:
        ws.high_confidence_threshold = high_confidence_threshold
    if default_llm_provider is not None:
        ws.default_llm_provider = default_llm_provider.strip() or None
    if default_llm_model is not None:
        ws.default_llm_model = default_llm_model.strip() or None
    if llm_compat_base_url is not None:
        ws.llm_compat_base_url = llm_compat_base_url.strip() or None

    stored_keys: dict[str, str] = {
        str(k): str(v)
        for k, v in parse_settings_json_dict(ws.llm_api_keys).items()
        if v is not None
    }
    stored_validations: dict[str, dict[str, Any]] = {
        str(k): v
        for k, v in parse_settings_json_dict(ws.llm_key_validations).items()
        if isinstance(v, dict)
    }

    if api_keys is not None:
        for prov, raw_val in api_keys.items():
            prov_key = prov.strip().lower()
            if raw_val is None or raw_val.strip() == "":
                stored_keys.pop(prov_key, None)
                if not (key_validations and prov_key in key_validations):
                    stored_validations.pop(prov_key, None)
            else:
                norm_key = normalize_api_key(raw_val)
                old_key = stored_keys.get(prov_key)
                stored_keys[prov_key] = norm_key
                if old_key != norm_key and not (key_validations and prov_key in key_validations):
                    stored_validations.pop(prov_key, None)

    if key_validations is not None:
        for prov, v in key_validations.items():
            prov_key = prov.strip().lower()
            if v is None:
                stored_validations.pop(prov_key, None)
            elif isinstance(v, dict):
                stored_validations[prov_key] = v

    ws.llm_api_keys = json.dumps(stored_keys) if stored_keys else None
    ws.llm_key_validations = json.dumps(stored_validations) if stored_validations else None

    ws.updated_at = _now()
    session.add(ws)
    await session.commit()
    await session.refresh(ws)

    # Sync runtime settings (never assign ws.llm_compat_base_url to runtime_settings.llm.base_url)
    sync_runtime_credentials(stored_keys, stored_validations, ws.llm_compat_base_url)

    runtime_settings = get_settings()
    if ws.high_confidence_threshold is not None:
        runtime_settings.ingestion.high_confidence_threshold = ws.high_confidence_threshold
    if default_llm_provider is not None:
        runtime_settings.llm.provider = ws.default_llm_provider or "openai"
    elif ws.default_llm_provider:
        runtime_settings.llm.provider = ws.default_llm_provider
    if default_llm_model is not None:
        runtime_settings.llm.model_name = ws.default_llm_model or ""
    elif ws.default_llm_model:
        runtime_settings.llm.model_name = ws.default_llm_model

    return ws
