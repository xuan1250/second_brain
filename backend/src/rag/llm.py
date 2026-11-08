"""Multi-provider LLM factory driven by config.yaml and .env."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.config import BACKEND_DIR, Settings, get_settings

T = TypeVar("T", bound=BaseModel)

_OPENAI_COMPATIBLE_PROVIDERS = {"openai_compatible", "custom", "glm", "zhipu", "vllm", "ollama_openai"}
_SUPPORTED_INIT_PROVIDERS = {"openai", "anthropic", "deepseek"}
_PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}
_PLACEHOLDER_PREFIXES = ("your-", "replace-", "changeme", "placeholder", "<your")

# In-memory UI credential registry & validation state
_ui_api_keys: dict[str, str] = {}
_key_validations: dict[str, dict[str, Any]] = {}
_ui_compat_base_url: str | None = None

# Optional test override for offline deterministic testing
_chat_model_override: BaseChatModel | Callable[..., BaseChatModel] | None = None
_dotenv_managed_keys: set[str] = set()


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
        for prov, k in api_keys.items():
            if k and is_valid_api_key(k):
                _ui_api_keys[prov.strip().lower()] = normalize_api_key(k)

    _key_validations.clear()
    if validations:
        for prov, v in validations.items():
            if isinstance(v, dict):
                _key_validations[prov.strip().lower()] = dict(v)

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
    for env_var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "LLM_API_KEY"):
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


def _load_dotenv_if_present(dotenv_path: Path | None = None) -> None:
    """Load/refresh valid keys from backend/.env and strip template placeholders from os.environ."""
    # First strip any placeholder values injected by docker-compose env_file
    for env_key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "LLM_API_KEY"):
        curr = os.environ.get(env_key)
        if curr is not None and not is_valid_api_key(curr):
            os.environ.pop(env_key, None)

    if dotenv_path is None and "PYTEST_CURRENT_TEST" in os.environ:
        return

    path = dotenv_path or (BACKEND_DIR / ".env")
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        if key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "LLM_API_KEY"):
            if is_valid_api_key(value):
                # Allow live-reloading from .env if not set or previously loaded from .env
                if key not in os.environ or key in _dotenv_managed_keys or not is_valid_api_key(os.environ.get(key)):
                    os.environ[key] = value
                    _dotenv_managed_keys.add(key)
            elif key in _dotenv_managed_keys:
                os.environ.pop(key, None)
                _dotenv_managed_keys.discard(key)
        elif key not in os.environ and value:
            os.environ[key] = value
            _dotenv_managed_keys.add(key)


def set_chat_model(model: BaseChatModel | Callable[..., BaseChatModel] | None) -> None:
    """Register or clear an injected chat model (or factory) for offline unit testing."""
    global _chat_model_override
    _chat_model_override = model


def get_chat_model_override() -> BaseChatModel | Callable[..., BaseChatModel] | None:
    """Return the active chat model override, if any."""
    return _chat_model_override


def resolve_provider_and_model(
    provider_override: str | None = None,
    model_override: str | None = None,
    settings: Settings | None = None,
) -> tuple[str, str]:
    """Resolve active (provider, model_name) from overrides, env, and config.yaml."""
    cfg = (settings or get_settings()).llm

    raw_provider = (
        provider_override
        if provider_override is not None and provider_override.strip()
        else (os.environ.get("LLM_PROVIDER") or cfg.provider or "openai")
    ).strip().lower()

    provider_entry = cfg.providers.get(raw_provider)
    provider_default_model = provider_entry.model_name if provider_entry else ""

    if model_override is not None and model_override.strip():
        raw_model = model_override.strip()
    elif os.environ.get("LLM_MODEL_NAME"):
        raw_model = os.environ["LLM_MODEL_NAME"].strip()
    elif provider_override and provider_override.strip() and provider_override.strip().lower() != cfg.provider.lower():
        raw_model = (provider_default_model or cfg.model_name or "").strip()
    else:
        raw_model = (cfg.model_name or provider_default_model or "").strip()

    # Support "provider:model_id" shorthand when provider_override wasn't passed separately
    if ":" in raw_model and not (provider_override and provider_override.strip()):
        prefix, _, rest = raw_model.partition(":")
        if prefix.strip().lower() in (_SUPPORTED_INIT_PROVIDERS | _OPENAI_COMPATIBLE_PROVIDERS):
            raw_provider = prefix.strip().lower()
            raw_model = rest.strip()

    if not raw_model:
        raise ValueError(
            "No LLM model_name configured. Set llm.model_name in config.yaml, "
            "LLM_MODEL_NAME in environment, or pass model_override."
        )

    return raw_provider, raw_model


def get_chat_model(
    provider_override: str | None = None,
    model_override: str | None = None,
    *,
    settings: Settings | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Instantiate a LangChain BaseChatModel configured via config.yaml and .env.

    - Supports `openai`, `anthropic`, and `deepseek` (via `langchain-deepseek`) through
      LangChain's `init_chat_model`.
    - Supports any OpenAI-compatible endpoint (e.g. Zhipu GLM, local vLLM, custom proxies)
      via `ChatOpenAI` whenever `base_url` is set or provider is an OpenAI-compatible alias.
    - Honors an injected test model registered via `set_chat_model(...)`.
    """
    if _chat_model_override is not None:
        if callable(_chat_model_override) and not isinstance(_chat_model_override, BaseChatModel):
            return _chat_model_override(
                provider_override=provider_override,
                model_override=model_override,
                settings=settings,
                **kwargs,
            )
        return _chat_model_override

    _load_dotenv_if_present()
    s = settings or get_settings()
    cfg = s.llm

    provider, model_name = resolve_provider_and_model(
        provider_override=provider_override,
        model_override=model_override,
        settings=s,
    )

    resolved_key, key_source, resolved_base_url = resolve_provider_credential(provider, s)
    explicit_key = kwargs.pop("api_key", None)
    api_key = explicit_key if is_valid_api_key(explicit_key) else resolved_key

    temperature = float(kwargs.pop("temperature", cfg.temperature))
    base_url = kwargs.pop("base_url", None) or resolved_base_url

    # Custom base_url or OpenAI-compatible provider -> ChatOpenAI
    if base_url or provider in _OPENAI_COMPATIBLE_PROVIDERS:
        if not api_key:
            raise ValueError(
                f"Missing API key for provider '{provider}'. "
                "Please set OPENAI_API_KEY or LLM_API_KEY in backend/.env (no container restart required)."
            )
        openai_kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": api_key,
            **kwargs,
        }
        # Reasoning models (o1, o3) reject non-default temperature
        if not any(model_name.lower().startswith(p) for p in ("o1", "o3")):
            openai_kwargs.setdefault("temperature", temperature)
        if base_url:
            openai_kwargs["base_url"] = base_url
        return ChatOpenAI(**openai_kwargs)

    if provider not in _SUPPORTED_INIT_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. Supported providers: "
            f"{sorted(_SUPPORTED_INIT_PROVIDERS | _OPENAI_COMPATIBLE_PROVIDERS)} or set llm.base_url."
        )

    expected_env = _PROVIDER_ENV_KEYS.get(provider)
    if not api_key:
        raise ValueError(
            f"Missing API key for provider '{provider}' (model '{model_name}'). "
            f"Please set a valid {expected_env} in backend/.env (changes take effect immediately)."
        )

    init_kwargs: dict[str, Any] = {
        "api_key": api_key,
        **kwargs,
    }
    # Anthropic reasoning / v5+ models (e.g. claude-sonnet-5-5, fable) reject non-default temperature
    omit_temp = provider == "anthropic" and any(
        tag in model_name.lower() for tag in ("5-5", "fable", "thinking")
    )
    if not omit_temp and "temperature" not in init_kwargs:
        init_kwargs["temperature"] = temperature

    return init_chat_model(
        model=model_name,
        model_provider=provider,
        **init_kwargs,
    )


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """Extract a JSON object from raw model output (handles fenced ```json blocks)."""
    text = (raw_text or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    else:
        brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if brace_match:
            text = brace_match.group(0).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


async def invoke_structured_output(
    llm: BaseChatModel,
    schema: type[T],
    messages: Sequence[BaseMessage],
    fallback_parser: Callable[[str], T] | None = None,
) -> T:
    """Invoke `llm` with structured Pydantic output `schema`.

    Tries `llm.with_structured_output(schema)` first; falls back to JSON extraction
    from `await llm.ainvoke(messages)` for fake/test models (`GenericFakeChatModel`)
    or providers that do not implement native tool/schema binding.
    """
    try:
        structured_llm = llm.with_structured_output(schema)
        result = await structured_llm.ainvoke(list(messages))
        if isinstance(result, schema):
            return result
        if isinstance(result, dict):
            return schema.model_validate(result)
        if isinstance(result, BaseMessage):
            content = result.content if isinstance(result.content, str) else json.dumps(result.content)
            parsed_dict = _extract_json_object(content)
            if parsed_dict is not None:
                return schema.model_validate(parsed_dict)
    except NotImplementedError:
        pass
    except Exception:
        # If structured output binding fails at runtime, fall through to raw invocation
        pass

    ai_msg = await llm.ainvoke(list(messages))
    content = ai_msg.content if isinstance(ai_msg.content, str) else json.dumps(ai_msg.content)
    parsed_dict = _extract_json_object(content)
    if parsed_dict is not None:
        return schema.model_validate(parsed_dict)
    if fallback_parser is not None:
        return fallback_parser(content)
    raise ValueError(f"Could not parse structured {schema.__name__} from model output: {content!r}")

