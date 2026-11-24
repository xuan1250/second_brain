"""Regression tests for LLM credential persistence, default provider recovery, and dotenv fallback."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from sqlalchemy import select

from src.api.routes.domains import verify_key_endpoint
from src.api.schemas import VerifyKeyRequest
from src.config import get_settings, load_settings
from src.db.models import WorkspaceSettings
from src.db.repository import get_workspace_settings, update_workspace_settings
from src.db.session import create_engine, create_session_factory, init_db
from src.rag.llm import (
    _load_dotenv_if_present,
    clear_ui_credentials,
    compute_key_hash,
    get_provider_validation_state,
    resolve_provider_credential,
    set_chat_model,
)


@pytest.fixture(autouse=True)
def isolated_credential_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate storage, settings cache, in-memory credentials, and LLM env vars per test."""
    data_dir = tmp_path / "data"
    db_dir = tmp_path / "data" / "db"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("DB_DIR", str(db_dir))

    for env_var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "LLM_API_KEY",
        "LLM_PROVIDER",
        "LLM_MODEL_NAME",
        "LLM_BASE_URL",
        "SECOND_BRAIN_DESKTOP_HOME",
    ):
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setattr("src.rag.llm._dotenv_managed_keys", set())
    clear_ui_credentials()
    set_chat_model(None)
    get_settings.cache_clear()
    yield
    clear_ui_credentials()
    set_chat_model(None)
    get_settings.cache_clear()


async def test_init_db_restores_persisted_credentials_validations_and_default_provider(tmp_path: Path):
    """Startup init_db restores SQLite UI keys, validation status, and default LLM provider/model."""
    db_path = tmp_path / "data" / "db" / "second_brain.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    deepseek_key = "sk-deepseek-persist-1234567890"

    try:
        await init_db(engine)
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            await update_workspace_settings(
                session=session,
                api_keys={"deepseek": deepseek_key},
                key_validations={
                    "deepseek": {
                        "status": "valid",
                        "error": None,
                        "key_hash": compute_key_hash(deepseek_key),
                        "checked_at": "2026-01-01T00:00:00+00:00",
                    }
                },
                default_llm_provider="deepseek",
                default_llm_model="deepseek-chat",
            )

        # Simulate a fresh process restart: clear in-memory credentials and settings cache
        clear_ui_credentials()
        get_settings.cache_clear()
        fresh_settings = load_settings()
        assert fresh_settings.llm.provider == "openai"
        assert get_settings().llm.provider == "openai"
        assert resolve_provider_credential("deepseek")[0] is None

        # Re-run startup hydration against the persisted SQLite database
        await init_db(engine)

        restored_key, key_source, _ = resolve_provider_credential("deepseek")
        assert restored_key == deepseek_key
        assert key_source == "ui"

        val_status, val_error = get_provider_validation_state("deepseek", restored_key, True)
        assert val_status == "valid"
        assert val_error is None
        assert get_settings().llm.provider == "deepseek"
        assert get_settings().llm.model_name == "deepseek-chat"
    finally:
        await engine.dispose()


async def test_init_db_promotes_configured_provider_when_default_unset(tmp_path: Path):
    """When default_llm_provider is unset and OpenAI has no key, init_db selects a configured provider."""
    db_path = tmp_path / "data" / "db" / "second_brain.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    deepseek_key = "sk-deepseek-unset-default-987654"

    try:
        await init_db(engine)
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            await update_workspace_settings(
                session=session,
                api_keys={"deepseek": deepseek_key},
            )
            ws = await get_workspace_settings(session)
            assert ws.default_llm_provider is None

        clear_ui_credentials()
        get_settings.cache_clear()
        assert get_settings().llm.provider == "openai"
        assert resolve_provider_credential("openai")[0] is None

        await init_db(engine)

        restored_key, key_source, _ = resolve_provider_credential("deepseek")
        assert restored_key == deepseek_key
        assert key_source == "ui"
        assert get_settings().llm.provider == "deepseek"
        expected_model = get_settings().llm.providers["deepseek"].model_name
        assert get_settings().llm.model_name == expected_model
    finally:
        await engine.dispose()


async def test_load_dotenv_if_present_supports_desktop_home_config_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """_load_dotenv_if_present falls back to SECOND_BRAIN_DESKTOP_HOME/config/.env cleanly."""
    fake_backend_dir = tmp_path / "backend"
    fake_backend_dir.mkdir(parents=True)
    desktop_home = tmp_path / "desktop_home"
    (desktop_home / "config").mkdir(parents=True)

    repo_dotenv = fake_backend_dir / ".env"
    desktop_dotenv = desktop_home / "config" / ".env"

    # Repo .env only has a template placeholder and lacks DEEPSEEK_API_KEY
    repo_dotenv.write_text("OPENAI_API_KEY=your-openai-key\n", encoding="utf-8")
    desktop_dotenv.write_text('DEEPSEEK_API_KEY="sk-deepseek-desktop-12345678"\n', encoding="utf-8")

    monkeypatch.setattr("src.rag.llm.BACKEND_DIR", fake_backend_dir)
    monkeypatch.setenv("SECOND_BRAIN_DESKTOP_HOME", str(desktop_home))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    _load_dotenv_if_present()

    assert "OPENAI_API_KEY" not in os.environ
    assert os.environ.get("DEEPSEEK_API_KEY") == "sk-deepseek-desktop-12345678"
    key, source, _ = resolve_provider_credential("deepseek")
    assert key == "sk-deepseek-desktop-12345678"
    assert source == "env"

    # Valid keys in repo .env take precedence, while desktop config/.env supplies missing keys
    repo_dotenv.write_text("OPENAI_API_KEY=sk-openai-repo-12345678\n", encoding="utf-8")
    desktop_dotenv.write_text(
        "OPENAI_API_KEY=sk-openai-desktop-87654321\n"
        "DEEPSEEK_API_KEY=sk-deepseek-desktop-12345678\n",
        encoding="utf-8",
    )

    _load_dotenv_if_present()

    assert os.environ.get("OPENAI_API_KEY") == "sk-openai-repo-12345678"
    assert os.environ.get("DEEPSEEK_API_KEY") == "sk-deepseek-desktop-12345678"

    # Clearing a previously loaded key from desktop config/.env unloads it on refresh
    desktop_dotenv.write_text("OPENAI_API_KEY=sk-openai-desktop-87654321\nDEEPSEEK_API_KEY=\n", encoding="utf-8")
    _load_dotenv_if_present()
    assert "DEEPSEEK_API_KEY" not in os.environ

    # Explicit dotenv_path is honored even when PYTEST_CURRENT_TEST is set
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    explicit_dotenv = tmp_path / "explicit.env"
    explicit_dotenv.write_text("ANTHROPIC_API_KEY=sk-ant-explicit-12345678\n", encoding="utf-8")
    _load_dotenv_if_present(dotenv_path=explicit_dotenv)
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-explicit-12345678"


async def test_verify_key_promotes_default_provider_when_unset_or_unconfigured(tmp_path: Path):
    """Verify & Save promotes default_llm_provider/model only when unset or unconfigured."""
    db_path = tmp_path / "data" / "db" / "second_brain.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    deepseek_key = "sk-deepseek-verify-12345678"
    anthropic_key = "sk-ant-verify-12345678"
    openai_key = "sk-proj-verify-12345678"

    try:
        await init_db(engine)
        session_factory = create_session_factory(engine)
        set_chat_model(GenericFakeChatModel(messages=iter([AIMessage(content="pong") for _ in range(3)])))

        async with session_factory() as session:
            # 1. With no default provider set, verifying DeepSeek promotes it as workspace default
            resp1 = await verify_key_endpoint(
                payload=VerifyKeyRequest(
                    provider="deepseek",
                    api_key=deepseek_key,
                    model="deepseek-chat",
                    save_if_valid=True,
                ),
                session=session,
                settings=get_settings(),
            )
            assert resp1.valid is True
            ws = await get_workspace_settings(session)
            assert ws.default_llm_provider == "deepseek"
            assert ws.default_llm_model == "deepseek-chat"
            assert get_settings().llm.provider == "deepseek"
            assert get_settings().llm.model_name == "deepseek-chat"

            # 2. Verifying another provider while the active default still has a valid key keeps the default
            resp2 = await verify_key_endpoint(
                payload=VerifyKeyRequest(
                    provider="anthropic",
                    api_key=anthropic_key,
                    model="claude-sonnet-4-5-20250929",
                    save_if_valid=True,
                ),
                session=session,
                settings=get_settings(),
            )
            assert resp2.valid is True
            ws = await get_workspace_settings(session)
            assert ws.default_llm_provider == "deepseek"
            assert ws.default_llm_model == "deepseek-chat"

            # 3. Clearing the default provider's key allows a newly verified provider to take over
            await update_workspace_settings(session=session, api_keys={"deepseek": ""})
            resp3 = await verify_key_endpoint(
                payload=VerifyKeyRequest(
                    provider="openai",
                    api_key=openai_key,
                    model="gpt-4o",
                    save_if_valid=True,
                ),
                session=session,
                settings=get_settings(),
            )
            assert resp3.valid is True
            ws = await get_workspace_settings(session)
            assert ws.default_llm_provider == "openai"
            assert ws.default_llm_model == "gpt-4o"
            assert get_settings().llm.provider == "openai"
    finally:
        await engine.dispose()


async def test_init_db_logs_warning_on_malformed_workspace_settings_json(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    """Malformed persisted JSON in WorkspaceSettings logs a warning instead of crashing startup."""
    db_path = tmp_path / "data" / "db" / "second_brain.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path.as_posix()}")

    try:
        await init_db(engine)
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            row = (
                await session.execute(select(WorkspaceSettings).where(WorkspaceSettings.id == "default"))
            ).scalars().first()
            assert row is not None
            row.llm_api_keys = "{malformed-json"
            row.llm_key_validations = "{also-malformed"
            session.add(row)
            await session.commit()

        with caplog.at_level(logging.WARNING, logger="src.db.session"):
            await init_db(engine)

        assert "Failed to parse persisted WorkspaceSettings JSON" in caplog.text
    finally:
        await engine.dispose()
