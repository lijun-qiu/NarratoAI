"""抽帧视觉模型轮换：主模型额度用尽时自动切换备用模型继续分析。"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable

from loguru import logger

from app.config.llm_gateway_router import resolve_llm_credentials
from app.services.llm.migration_adapter import create_vision_analyzer

DEFAULT_VISION_FALLBACK_MODEL_NAMES = "qwen-vl-plus,qwen2.5-vl-72b-instruct"

_QUOTA_ERROR_PATTERNS = (
    "quota",
    "rate limit",
    "ratelimit",
    "rate_limit",
    "429",
    "insufficient",
    "exceeded",
    "limit exceeded",
    "allocationquota",
    "throttling",
    "too many requests",
    "余额",
    "额度",
    "用尽",
    "不足",
    "exhausted",
    "billing",
    "credit",
    "tokens limit",
)

_MODEL_UNAVAILABLE_PATTERNS = (
    "model_not_found",
    "does not exist",
    "model does not exist",
    "invalid model",
    "unknown model",
    "not found",
    "no access",
    "do not have access",
    "模型不存在",
    "无权访问",
    "无权限",
    "error code: 404",
    "404",
)


def is_quota_or_rate_limit_error(message: str) -> bool:
    """判断是否为额度/限流类错误，可尝试切换备用模型。"""
    text = (message or "").strip().lower()
    if not text:
        return False
    return any(pattern in text for pattern in _QUOTA_ERROR_PATTERNS)


def is_switchable_vision_model_error(message: str) -> bool:
    """额度/限流或模型不可用（404 等）时，可切换备用视觉模型重试。"""
    text = (message or "").strip().lower()
    if not text:
        return False
    if is_quota_or_rate_limit_error(text):
        return True
    return any(pattern in text for pattern in _MODEL_UNAVAILABLE_PATTERNS)


def resolve_vision_model_chain(
    primary_model: str,
    fallback_raw: str | list[str] | None = None,
) -> list[str]:
    """主模型 + 备用模型去重列表。"""
    models: list[str] = []
    primary = (primary_model or "").strip()
    if primary:
        models.append(primary)

    extras: list[str] = []
    if isinstance(fallback_raw, list):
        extras = [str(item).strip() for item in fallback_raw if str(item).strip()]
    elif isinstance(fallback_raw, str) and fallback_raw.strip():
        extras = [
            part.strip()
            for part in re.split(r"[,，\n;；]+", fallback_raw)
            if part.strip()
        ]
    elif fallback_raw is None:
        extras = [
            part.strip()
            for part in re.split(r"[,，\n;；]+", DEFAULT_VISION_FALLBACK_MODEL_NAMES)
            if part.strip()
        ]

    for model in extras:
        if model not in models:
            models.append(model)
    return models


class VisionModelRotation:
    """按顺序尝试视觉模型；某模型额度用尽后自动切换下一个。"""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        base_url: str,
        model_names: list[str],
        extract_response: Callable[[Any], tuple[str, str]],
    ) -> None:
        self.provider = (provider or "openai").lower()
        self.api_key = api_key
        self.base_url = base_url or ""
        self.model_names = [name for name in model_names if name]
        if not self.model_names:
            raise ValueError("未配置任何视觉模型")
        self._extract_response = extract_response
        self._exhausted: set[str] = set()
        self._lock = asyncio.Lock()
        self._analyzers: dict[str, Any] = {}
        self.models_used: set[str] = set()
        self.pending_switch_message: str = ""

    def _create_analyzer(self, model_name: str) -> Any:
        cache_key = f"{model_name}::{self.provider}"
        if cache_key not in self._analyzers:
            api_key, base_url = resolve_llm_credentials(model_name, role="vision")
            if not api_key:
                api_key = self.api_key
            if not base_url:
                base_url = self.base_url
            self._analyzers[cache_key] = create_vision_analyzer(
                provider=self.provider,
                api_key=api_key,
                model=model_name,
                base_url=base_url,
            )
        return self._analyzers[cache_key]

    def _available_models(self) -> list[str]:
        available = [name for name in self.model_names if name not in self._exhausted]
        return available or list(self.model_names)

    async def _mark_exhausted(self, model_name: str, reason: str) -> None:
        async with self._lock:
            if model_name in self._exhausted:
                return
            self._exhausted.add(model_name)
            logger.warning(f"视觉模型 {model_name} 不可用（{reason[:120]}），尝试切换备用模型")
            next_models = self._available_models()
            if next_models and next_models[0] != model_name:
                self.pending_switch_message = (
                    f"视觉模型 {model_name} 额度/限流不可用，已切换至 {next_models[0]} 继续分析…"
                )

    async def analyze_images(
        self,
        *,
        images: list[str],
        prompt: str,
        batch_size: int,
        max_concurrency: int = 1,
    ) -> tuple[Any | None, str, str]:
        """
        Returns:
            (raw_results, model_used, error_message)
        """
        last_error = ""
        for model_name in self._available_models():
            try:
                analyzer = self._create_analyzer(model_name)
                raw_results = await analyzer.analyze_images(
                    images=images,
                    prompt=prompt,
                    batch_size=max(1, batch_size),
                    max_concurrency=max(1, max_concurrency),
                )
                raw_response, error_message = self._extract_response(raw_results)
                if error_message and is_switchable_vision_model_error(error_message):
                    await self._mark_exhausted(model_name, error_message)
                    last_error = error_message
                    continue
                self.models_used.add(model_name)
                return raw_results, model_name, ""
            except Exception as exc:
                message = str(exc)
                if is_switchable_vision_model_error(message):
                    await self._mark_exhausted(model_name, message)
                    last_error = message
                    continue
                raise

        return None, "", last_error or "所有配置的视觉模型均不可用，请检查额度或备用模型列表"
