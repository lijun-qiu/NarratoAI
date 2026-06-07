"""按模型名称自动选择 LLM 网关与 API Key（百炼 Qwen / 4022 其他）。"""

from __future__ import annotations

from typing import Any, Literal

from app.config.llm_model_presets import DEFAULT_ALT_BASE_URL, DEFAULT_DASHSCOPE_BASE_URL

LLMRole = Literal["vision", "text"]


def _normalize_model_name(model_name: str) -> str:
    text = (model_name or "").strip().lower()
    if "/" in text:
        text = text.split("/", 1)[-1].strip()
    return text


def qwen_use_alt_gateway(app_config: dict[str, Any] | None = None) -> bool:
    """百炼欠费/不可用时，可让 Qwen 也走备用网关（4022）。"""
    from app.config import config

    cfg = app_config if app_config is not None else config.app
    return bool(cfg.get("llm_qwen_use_alt_gateway"))


def uses_dashscope_gateway(
    model_name: str,
    app_config: dict[str, Any] | None = None,
) -> bool:
    """Qwen 系列默认走阿里百炼；可配置改走备用网关。"""
    if qwen_use_alt_gateway(app_config):
        return False
    model = _normalize_model_name(model_name)
    return model.startswith("qwen")


def resolve_llm_credentials(
    model_name: str,
    *,
    role: LLMRole = "text",
    app_config: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """
    根据模型 ID 返回 (api_key, base_url)。

    - qwen* → llm_dashscope_*（百炼）
    - 其他 → llm_alt_* 或 vision_openai_* / text_openai_*（4022 等）
    """
    from app.config import config

    cfg = app_config if app_config is not None else config.app
    role_prefix = f"{role}_openai"

    dashscope_key = (
        str(cfg.get("llm_dashscope_api_key") or cfg.get("dashscope_api_key") or "")
        .strip()
    )
    dashscope_url = (
        str(
            cfg.get("llm_dashscope_base_url")
            or cfg.get("dashscope_base_url")
            or DEFAULT_DASHSCOPE_BASE_URL
        )
        .strip()
        .rstrip("/")
    )

    alt_key = str(cfg.get("llm_alt_api_key") or "").strip()
    alt_url = str(cfg.get("llm_alt_base_url") or DEFAULT_ALT_BASE_URL).strip().rstrip("/")
    role_key = str(cfg.get(f"{role_prefix}_api_key") or "").strip()
    role_url = str(cfg.get(f"{role_prefix}_base_url") or "").strip().rstrip("/")

    if not alt_key:
        alt_key = role_key
    if not alt_url:
        alt_url = role_url or DEFAULT_ALT_BASE_URL

    if uses_dashscope_gateway(model_name, cfg):
        api_key = dashscope_key
        base_url = dashscope_url
        if not api_key and role_url and "dashscope" in role_url.lower():
            api_key = role_key
            base_url = role_url or dashscope_url
        return api_key, base_url

    return alt_key, alt_url


def get_role_model_name(
    role: LLMRole,
    app_config: dict[str, Any] | None = None,
) -> str:
    from app.config import config

    cfg = app_config if app_config is not None else config.app
    provider = str(cfg.get(f"{role}_llm_provider") or "openai").lower()
    return str(cfg.get(f"{role}_{provider}_model_name") or "").strip()


def resolve_role_credentials(
    role: LLMRole,
    *,
    model_name: str | None = None,
    app_config: dict[str, Any] | None = None,
) -> tuple[str, str, str]:
    """返回 (model_name, api_key, base_url)。"""
    from app.config import config

    cfg = app_config if app_config is not None else config.app
    resolved_model = (model_name or get_role_model_name(role, cfg)).strip()
    api_key, base_url = resolve_llm_credentials(resolved_model, role=role, app_config=cfg)
    return resolved_model, api_key, base_url


def format_llm_connection_error(message: str) -> str:
    """将常见 API 错误转为可读中文提示。"""
    text = (message or "").strip()
    lower = text.lower()
    if "arrearage" in lower or "overdue-payment" in lower or "good standing" in lower:
        return (
            "阿里百炼账户欠费或停用（Arrearage）。请登录 "
            "https://bailian.console.aliyun.com/ 充值后再试；"
            "或在「双网关」勾选「Qwen 改走备用网关」临时使用 4022。"
        )
    if "authentication" in lower or "api_key" in lower or "invalid api key" in lower:
        return "API Key 认证失败，请检查双网关中的 Key 是否正确。"
    if "not found" in lower or "404" in lower or "model_not_found" in lower:
        return "模型不存在或当前网关不支持该模型，请换模型或换网关。"
    if "rate limit" in lower or "429" in lower:
        return "请求过于频繁或额度用尽，请稍后重试或切换备用模型。"
    return text


def describe_llm_route(model_name: str, *, role: LLMRole = "text") -> str:
    """供日志 / UI 展示路由结果。"""
    api_key, base_url = resolve_llm_credentials(model_name, role=role)
    gateway = (
        "备用网关（Qwen 已改道）"
        if qwen_use_alt_gateway() and _normalize_model_name(model_name).startswith("qwen")
        else ("百炼 DashScope" if uses_dashscope_gateway(model_name) else "备用网关")
    )
    key_hint = f"{api_key[:8]}…" if len(api_key) > 8 else "(未配置)"
    return f"{gateway} · {base_url or '(无)'} · key {key_hint}"
