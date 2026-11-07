"""Multi-provider LLM factory driven by config.yaml and .env."""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

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

# Optional test override for offline deterministic testing
_chat_model_override: BaseChatModel | Callable[..., BaseChatModel] | None = None
_dotenv_managed_keys: set[str] = set()


def is_valid_api_key(value: str | None) -> bool:
    """Return True if `value` is a non-empty API key and not a template placeholder."""
    if not value:
        return False
    cleaned = value.strip().strip("'").strip('"')
    if not cleaned:
        return False
    lower = cleaned.lower()
    if any(lower.startswith(prefix) for prefix in _PLACEHOLDER_PREFIXES):
        return False
    if lower in {"none", "null", "todo", "xxx"}:
        return False
    return True


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

    temperature = float(kwargs.pop("temperature", cfg.temperature))
    provider_entry = cfg.providers.get(provider)
    provider_base_url = provider_entry.base_url if provider_entry else None
    base_url = (
        kwargs.pop("base_url", None)
        or os.environ.get("LLM_BASE_URL")
        or provider_base_url
        or cfg.base_url
    )

    # Custom base_url or OpenAI-compatible provider -> ChatOpenAI
    if base_url or provider in _OPENAI_COMPATIBLE_PROVIDERS:
        raw_candidates = [
            kwargs.pop("api_key", None),
            os.environ.get("LLM_API_KEY"),
            os.environ.get("OPENAI_API_KEY"),
            os.environ.get("DEEPSEEK_API_KEY"),
        ]
        api_key = next((k for k in raw_candidates if is_valid_api_key(k)), None)
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
    explicit_key = kwargs.get("api_key")
    if expected_env and not is_valid_api_key(explicit_key or os.environ.get(expected_env)):
        raise ValueError(
            f"Missing API key for provider '{provider}' (model '{model_name}'). "
            f"Please set a valid {expected_env} in backend/.env (changes take effect immediately)."
        )

    init_kwargs: dict[str, Any] = {
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

