"""FastAPI endpoints for Workspace Settings and LLM credential verification."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_app_settings, get_db_session
from src.api.schemas import (
    VerifyKeyRequest,
    VerifyKeyResponse,
    WorkspaceSettingsResponse,
    WorkspaceSettingsUpdateRequest,
)
from src.config import Settings
from src.services.settings_service import (
    get_workspace_settings_view,
    update_workspace_settings_view,
    verify_provider_key,
)

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=WorkspaceSettingsResponse)
async def get_settings_endpoint(
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
) -> WorkspaceSettingsResponse:
    """Get the current workspace settings merged with available LLM provider options."""
    return await get_workspace_settings_view(session, settings)


@router.patch("", response_model=WorkspaceSettingsResponse)
async def update_settings_endpoint(
    payload: WorkspaceSettingsUpdateRequest,
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
) -> WorkspaceSettingsResponse:
    """Update workspace settings and sync runtime configuration."""
    return await update_workspace_settings_view(session, settings, payload)


@router.post("/verify-key", response_model=VerifyKeyResponse)
async def verify_key_endpoint(
    payload: VerifyKeyRequest,
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
) -> VerifyKeyResponse:
    """Verify an API key live via bounded LLM ping and optionally persist to WorkspaceSettings."""
    return await verify_provider_key(payload, session, settings)
