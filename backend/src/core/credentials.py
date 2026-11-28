"""Zero-dependency runtime credential registry, validation state, .env loader, and error scrubbing."""
from __future__ import annotations

import hashlib
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.config import BACKEND_DIR as _DEFAULT_BACKEND_DIR, Settings, get_settings

BACKEND_DIR = _DEFAULT_BACKEND_DIR

_OPENAI_COMPATIBLE_PROVIDERS = {"openai_compatible", "custom", "glm", "zhipu", "vllm", "ollama_openai"}
_SUPPORTED_INIT_PROVIDERS = {"openai", "anthropic", "deepseek"}
_PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}
_PLACEHOLDER_PREFIXES = ("your-", "replace-", "changeme", "placeholder", "<your")
_MANAGED_API_ENV_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "LLM_API_KEY")

# In-memory UI credential registry & validation state (never rebind these dicts; mutate in-place)
_ui_api_keys: dict[str, str] = {}
_key_validations: dict[str, dict[str, Any]] = {}
_ui_compat_base_url: str | None = None
_dotenv_managed_keys: set[str] = set()


def _get_backend_dir() -> Path:
    """Resolve BACKEND_DIR, honoring monkeypatches on either src.rag.llm or this module."""
    llm_mod = sys.modules.get("src.rag.llm")
    if llm_mod is not None:
        llm_dir = getattr(llm_mod, "BACKEND_DIR", None)
        if isinstance(llm_dir, Path) and llm_dir != _DEFAULT_BACKEND_DIR:
            return llm_dir
    return BACKEND_DIR


def _get_dotenv_managed_keys() -> set[str]:
    """Resolve _dotenv_managed_keys set, honoring monkeypatches on src.rag.llm."""
    llm_mod = sys.modules.get("src.rag.llm")
    if llm_mod is not None:
        val = getattr(llm_mod, "_dotenv_managed_keys", None)
        if isinstance(val, set):
            return val
    return _dotenv_managed_keys


def get_ui_compat_base_url() -> str | None:
    """Return the active in-memory UI base URL for openai_compatible providers."""
    return _ui_compat_base_url


def normalize_api_key(value: str | None) -> str:
    """Strip leading/trailing whitespace and surrounding single/double quotes."""
    if not value:
        return ""
    return value.strip().strip("'").strip('"').strip()


def is_valid_api_key(value: str | None) -> bool:
    """Return True if `value` is a non-empty API key and not a template placeholder."""
    cleaned = normalize_api_key(value)
    if not cleaned or len(cleaned) < 8:
        return False
    if bool(re.search(r"[\s\x00-\x1f\x7f]", cleaned)):
        return False
    lower = cleaned.lower()
    if any(lower.startswith(prefix) for prefix in _PLACEHOLDER_PREFIXES):
        return False
    if lower in {"none", "null", "todo", "xxx"}:
        return False
    return True


def mask_api_key(key: str | None) -> str | None:
    """Mask key for safe display. Returns None if invalid."""
    if not is_valid_api_key(key):
        return None
    cleaned = normalize_api_key(key)
    if len(cleaned) < 16:
        return "••••" + cleaned[-2:]
    return f"{cleaned[:4]}••••{cleaned[-4:]}"


def compute_key_hash(key: str | None, base_url: str | None = None) -> str | None:
    """Compute 16-char SHA-256 digest of key + base_url for validation binding."""
    if not is_valid_api_key(key):
        return None
    norm_key = normalize_api_key(key)
    norm_url = (base_url or "").strip().rstrip("/")
    raw = f"{norm_key}|{norm_url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def sync_runtime_credentials(
    api_keys: dict[str, str] | None,
    validations: dict[str, dict[str, Any]] | None,
    compat_base_url: str | None,
) -> None:
    """Populate in-memory UI credentials, validations, and openai_compatible base_url."""
    global _ui_compat_base_url
    _ui_api_keys.clear()
    if api_keys:
        _ui_api_keys.update(
            {
                prov.strip().lower(): normalize_api_key(k)
                for prov, k in api_keys.items()
                if k and is_valid_api_key(k)
            }
        )

    _key_validations.clear()
    if validations:
        _key_validations.update(
            {
                prov.strip().lower(): dict(v)
                for prov, v in validations.items()
                if isinstance(v, dict)
            }
        )

    _ui_compat_base_url = compat_base_url.strip() if (compat_base_url and compat_base_url.strip()) else None

    try:
        s = get_settings()
        if "openai_compatible" in s.llm.providers:
            if _ui_compat_base_url:
                s.llm.providers["openai_compatible"].base_url = _ui_compat_base_url
    except Exception:
        pass


def clear_ui_credentials() -> None:
    """Clear all in-memory UI credentials, validations, and compat base URL."""
    global _ui_compat_base_url
    _ui_api_keys.clear()
    _key_validations.clear()
    _ui_compat_base_url = None


def resolve_provider_credential(
    provider: str,
    settings: Settings | None = None,
) -> tuple[str | None, Literal["ui", "env", "none"], str | None]:
    """Resolve (api_key, key_source, base_url) with UI Key > .env Key precedence."""
    _load_dotenv_if_present()
    s = settings or get_settings()
    cfg = s.llm
    prov = provider.strip().lower()
    provider_entry = cfg.providers.get(prov)

    if prov in _OPENAI_COMPATIBLE_PROVIDERS:
        base_url = (
            _ui_compat_base_url
            or os.environ.get("LLM_BASE_URL")
            or (provider_entry.base_url if provider_entry else None)
            or cfg.base_url
        )
        ui_key = _ui_api_keys.get(prov) or _ui_api_keys.get("openai_compatible")
        if ui_key and is_valid_api_key(ui_key):
            return normalize_api_key(ui_key), "ui", base_url

        for env_var in ("LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
            val = os.environ.get(env_var)
            if val and is_valid_api_key(val):
                return normalize_api_key(val), "env", base_url

        return None, "none", base_url

    # Standard providers (openai, anthropic, deepseek)
    base_url = (
        os.environ.get("LLM_BASE_URL")
        or (provider_entry.base_url if provider_entry else None)
        or cfg.base_url
    )
    ui_key = _ui_api_keys.get(prov)
    if ui_key and is_valid_api_key(ui_key):
        return normalize_api_key(ui_key), "ui", base_url

    env_var = _PROVIDER_ENV_KEYS.get(prov)
    if env_var:
        val = os.environ.get(env_var)
        if val and is_valid_api_key(val):
            return normalize_api_key(val), "env", base_url

    return None, "none", base_url


def record_key_validation(
    provider: str,
    status: Literal["valid", "invalid"],
    error: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Record validation status and key_hash for provider."""
    prov = provider.strip().lower()
    prov_key = "openai_compatible" if prov in _OPENAI_COMPATIBLE_PROVIDERS else prov
    resolved_key, _, resolved_base_url = resolve_provider_credential(prov_key, settings)
    k_hash = compute_key_hash(
        resolved_key,
        resolved_base_url if prov_key == "openai_compatible" else None,
    )
    record = {
        "status": status,
        "error": error,
        "key_hash": k_hash,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    _key_validations[prov_key] = record
    return record


def get_provider_validation_state(
    provider: str,
    resolved_key: str | None,
    is_configured: bool,
    base_url: str | None = None,
) -> tuple[Literal["valid", "invalid", "untested", "missing"], str | None]:
    """Return (validation_status, error_message) bound to key+url hash."""
    prov = provider.strip().lower()
    prov_key = "openai_compatible" if prov in _OPENAI_COMPATIBLE_PROVIDERS else prov
    entry = _key_validations.get(prov_key) or _key_validations.get(prov)
    current_hash = compute_key_hash(
        resolved_key,
        base_url if prov_key == "openai_compatible" else None,
    )
    if (
        current_hash
        and entry
        and entry.get("key_hash") == current_hash
        and entry.get("status") in ("valid", "invalid")
    ):
        return entry["status"], entry.get("error")
    if not is_configured:
        return "missing", entry.get("error") if (entry and entry.get("status") == "invalid") else None
    return "untested", None


def sanitize_llm_error(
    exc: Exception | str,
    provider: str | None = None,
    candidate_key: str | None = None,
) -> tuple[str, bool]:
    """Scrub keys from error message and determine if it represents an auth error."""
    msg = str(exc)
    lower_msg = msg.lower()
    auth_markers = (
        "401",
        "invalid_api_key",
        "invalid api key",
        "authentication_error",
        "authenticationerror",
        "authentication failed",
        "invalid x-api-key",
        "missing api key",
        "incorrect api key",
        "unauthorized",
    )
    is_auth_error = any(marker in lower_msg for marker in auth_markers)

    secrets_to_scrub: set[str] = set()
    if candidate_key and is_valid_api_key(candidate_key):
        secrets_to_scrub.add(normalize_api_key(candidate_key))
    for k in _ui_api_keys.values():
        if is_valid_api_key(k):
            secrets_to_scrub.add(normalize_api_key(k))
    for env_var in _MANAGED_API_ENV_KEYS:
        val = os.environ.get(env_var)
        if val and is_valid_api_key(val):
            secrets_to_scrub.add(normalize_api_key(val))

    for secret in sorted(secrets_to_scrub, key=len, reverse=True):
        if secret:
            msg = msg.replace(secret, "[REDACTED]")

    msg = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED]", msg)

    if len(msg) > 240:
        msg = msg[:237] + "..."

    return msg, is_auth_error


def _load_dotenv_file(
    path: Path,
    *,
    loaded_api_keys: set[str] | None = None,
    invalid_api_keys: set[str] | None = None,
) -> None:
    """Parse a single .env file and merge valid keys into os.environ."""
    if not path.is_file():
        return
    standalone = loaded_api_keys is None and invalid_api_keys is None
    loaded = loaded_api_keys if loaded_api_keys is not None else set()
    invalid = invalid_api_keys if invalid_api_keys is not None else set()
    managed_keys = _get_dotenv_managed_keys()

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        if key in _MANAGED_API_ENV_KEYS:
            if key in loaded:
                continue
            if is_valid_api_key(value):
                # Allow live-reloading from .env if not set or previously loaded from .env
                if key not in os.environ or key in managed_keys or not is_valid_api_key(os.environ.get(key)):
                    os.environ[key] = value
                    managed_keys.add(key)
                loaded.add(key)
            else:
                invalid.add(key)
        elif key not in os.environ and value:
            os.environ[key] = value
            managed_keys.add(key)

    if standalone:
        for key in invalid - loaded:
            if key in managed_keys:
                os.environ.pop(key, None)
                managed_keys.discard(key)


def _load_dotenv_if_present(dotenv_path: Path | None = None) -> None:
    """Load/refresh valid keys from .env files and strip template placeholders from os.environ."""
    # First strip any placeholder values injected by docker-compose env_file
    for env_key in _MANAGED_API_ENV_KEYS:
        curr = os.environ.get(env_key)
        if curr is not None and not is_valid_api_key(curr):
            os.environ.pop(env_key, None)

    if dotenv_path is None and "PYTEST_CURRENT_TEST" in os.environ:
        return

    if dotenv_path is not None:
        candidates = [dotenv_path]
    else:
        candidates = [_get_backend_dir() / ".env"]
        desktop_home = (os.environ.get("SECOND_BRAIN_DESKTOP_HOME") or "").strip()
        if desktop_home:
            candidates.append(Path(desktop_home).expanduser() / "config" / ".env")

    loaded_api_keys: set[str] = set()
    invalid_api_keys: set[str] = set()
    for path in candidates:
        _load_dotenv_file(path, loaded_api_keys=loaded_api_keys, invalid_api_keys=invalid_api_keys)

    managed_keys = _get_dotenv_managed_keys()
    for key in invalid_api_keys - loaded_api_keys:
        if key in managed_keys:
            os.environ.pop(key, None)
            managed_keys.discard(key)
