"""WorkspaceSettings startup restoration and JSON parsing helpers."""
from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.core.credentials import resolve_provider_credential, sync_runtime_credentials
from src.db.models import WorkspaceSettings

# Emit warnings on the `src.db.session` logger so startup log capture filters catch malformed JSON.
_session_logger = logging.getLogger("src.db.session")


def parse_settings_json_dict(raw: str | None) -> dict[str, Any]:
    """Parse a persisted WorkspaceSettings JSON object, returning {} on malformed input."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as exc:
        _session_logger.warning("Failed to parse persisted WorkspaceSettings JSON: %s", exc)
        return {}


async def restore_workspace_settings_and_credentials(session: AsyncSession) -> WorkspaceSettings:
    """Ensure singleton WorkspaceSettings exists and hydrate runtime settings & LLM credentials."""
    ws_res = await session.execute(select(WorkspaceSettings).where(WorkspaceSettings.id == "default"))
    ws = ws_res.scalars().first()
    if ws is None:
        ws = WorkspaceSettings(id="default")
        session.add(ws)
        await session.flush()

    runtime_settings = get_settings()
    if ws.high_confidence_threshold is not None:
        runtime_settings.ingestion.high_confidence_threshold = ws.high_confidence_threshold
    if ws.default_llm_provider:
        runtime_settings.llm.provider = ws.default_llm_provider
        if not ws.default_llm_model:
            prov_cfg = runtime_settings.llm.providers.get(ws.default_llm_provider.strip().lower())
            if prov_cfg and prov_cfg.model_name:
                runtime_settings.llm.model_name = prov_cfg.model_name
    if ws.default_llm_model:
        runtime_settings.llm.model_name = ws.default_llm_model

    api_keys = {
        str(k): str(v)
        for k, v in parse_settings_json_dict(ws.llm_api_keys).items()
        if v is not None
    }
    validations = {
        str(k): v
        for k, v in parse_settings_json_dict(ws.llm_key_validations).items()
        if isinstance(v, dict)
    }

    sync_runtime_credentials(api_keys, validations, ws.llm_compat_base_url)

    # When no workspace default provider is pinned, promote a configured provider over an unconfigured default.
    if not ws.default_llm_provider:
        def _provider_has_credential(prov: str) -> bool:
            key, _, base_url = resolve_provider_credential(prov, runtime_settings)
            return bool(key and base_url) if prov == "openai_compatible" else bool(key)

        active_provider = (runtime_settings.llm.provider or "openai").strip().lower()
        if not _provider_has_credential(active_provider):
            for cand in ("openai", "deepseek", "anthropic", "openai_compatible"):
                if _provider_has_credential(cand):
                    runtime_settings.llm.provider = cand
                    cand_entry = runtime_settings.llm.providers.get(cand)
                    if cand_entry and cand_entry.model_name:
                        runtime_settings.llm.model_name = cand_entry.model_name
                    break

    return ws
