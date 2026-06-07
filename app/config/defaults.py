"""Shared config defaults used by both bootstrap and WebUI fallbacks."""

from __future__ import annotations

from app.config.llm_model_presets import DEFAULT_ALT_BASE_URL, DEFAULT_DASHSCOPE_BASE_URL

DEFAULT_OPENAI_COMPATIBLE_BASE_URL = DEFAULT_DASHSCOPE_BASE_URL
DEFAULT_OPENAI_COMPATIBLE_PROVIDER = "openai"

DEFAULT_VISION_LLM_PROVIDER = DEFAULT_OPENAI_COMPATIBLE_PROVIDER
DEFAULT_VISION_OPENAI_MODEL_NAME = "gemini-3.1-flash-lite"

DEFAULT_TEXT_LLM_PROVIDER = DEFAULT_OPENAI_COMPATIBLE_PROVIDER
DEFAULT_TEXT_OPENAI_MODEL_NAME = "deepseek-v4-flash"

DEFAULT_LLM_APP_CONFIG = {
    "vision_llm_provider": DEFAULT_VISION_LLM_PROVIDER,
    "vision_openai_model_name": DEFAULT_VISION_OPENAI_MODEL_NAME,
    "vision_openai_api_key": "",
    "vision_openai_base_url": DEFAULT_ALT_BASE_URL,
    "text_llm_provider": DEFAULT_TEXT_LLM_PROVIDER,
    "text_openai_model_name": DEFAULT_TEXT_OPENAI_MODEL_NAME,
    "text_openai_api_key": "",
    "text_openai_base_url": DEFAULT_ALT_BASE_URL,
    "llm_dashscope_api_key": "",
    "llm_dashscope_base_url": DEFAULT_DASHSCOPE_BASE_URL,
    "llm_alt_api_key": "",
    "llm_alt_base_url": DEFAULT_ALT_BASE_URL,
    "llm_qwen_use_alt_gateway": False,
}


def build_default_app_config(app_config: dict | None = None) -> dict:
    """Force the shared LLM defaults into a fresh app config."""
    merged = dict(app_config or {})
    merged.update(DEFAULT_LLM_APP_CONFIG)
    return merged


def merge_missing_app_defaults(app_config: dict | None = None) -> dict:
    """Backfill missing keys without overriding saved user values."""
    merged = dict(app_config or {})
    for key, value in DEFAULT_LLM_APP_CONFIG.items():
        merged.setdefault(key, value)
    return merged


def normalize_openai_compatible_model_name(
    model_name: str,
    provider: str = DEFAULT_OPENAI_COMPATIBLE_PROVIDER,
) -> str:
    """Strip only the internal OpenAI-compatible provider prefix if present."""
    normalized = (model_name or "").strip()
    provider_prefix = f"{provider}/"
    if normalized.lower().startswith(provider_prefix):
        return normalized[len(provider_prefix):]
    return normalized


def get_openai_compatible_ui_values(
    full_model_name: str,
    default_model: str,
    provider: str = DEFAULT_OPENAI_COMPATIBLE_PROVIDER,
) -> tuple[str, str]:
    """Keep the UI provider fixed while preserving the full model identifier."""
    current_model = normalize_openai_compatible_model_name(
        full_model_name or default_model,
        provider=provider,
    )
    return provider, current_model or default_model
